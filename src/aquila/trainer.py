"""
Trainer module for SNP neural networks.
Handles training loop, validation, early stopping, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import time

from .metrics import MultiTaskLoss, MetricsCalculator


class EarlyStopping:
    """Early stopping handler to stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better) or 'max' for metrics (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class VarTrainer:
    """Trainer for variant neural networks with multi-task learning."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_regression_tasks: int = 0,
        num_classification_tasks: int = 0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_type: str = 'mse',
        uncertainty_weighting: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = './checkpoints',
        early_stopping_patience: int = 20,
        gradient_clip_norm: Optional[float] = 1.0,
        scheduler_type: str = 'reduce_on_plateau',
        scheduler_params: Optional[Dict] = None,
        num_epochs: Optional[int] = None,
        val_score_r_weight: float = 0.8,
        val_score_r2_weight: float = 0.2,
        is_distributed: bool = False,
        rank: int = 0,
        use_mixed_precision: bool = False,
        huber_delta: float = 1.0,
    ):
        """
        Initialize trainer.

        Args:
            model: SNPNeuralNetwork model
            train_loader: Training data loader
            val_loader: Validation data loader
            num_regression_tasks: Number of regression tasks
            num_classification_tasks: Number of classification tasks
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            loss_type: Loss type for regression ('mse' or 'mae')
            uncertainty_weighting: Use uncertainty weighting for multi-task loss
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            gradient_clip_norm: Gradient clipping norm (None to disable)
            scheduler_type: Type of learning rate scheduler ('reduce_on_plateau' or 'one_cycle')
            scheduler_params: Additional parameters for the scheduler
            num_epochs: Total number of epochs (required for OneCycleLR)
            val_score_r_weight: Weight for Pearson R in validation score (default 0.8)
            val_score_r2_weight: Weight for R² in validation score (default 0.2)
            is_distributed: Whether running in distributed mode
            rank: Process rank for distributed training
            use_mixed_precision: Enable mixed precision training (FP16) for faster training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.is_distributed = is_distributed
        self.rank = rank
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_clip_norm = gradient_clip_norm
        self.use_mixed_precision = use_mixed_precision

        # Initialize GradScaler for mixed precision training
        if self.use_mixed_precision:
            if not torch.cuda.is_available():
                raise ValueError(
                    "Mixed precision training requires CUDA. Set device='cuda' or use --device cuda")
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Validation score weights
        self.val_score_r_weight = val_score_r_weight
        self.val_score_r2_weight = val_score_r2_weight

        # Loss function
        self.criterion = MultiTaskLoss(
            num_regression_tasks=num_regression_tasks,
            num_classification_tasks=num_classification_tasks,
            loss_type=loss_type,
            uncertainty_weighting=uncertainty_weighting,
            huber_delta=huber_delta,
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.criterion.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler_type = scheduler_type.lower()
        if scheduler_params is None:
            scheduler_params = {}

        if self.scheduler_type == 'reduce_on_plateau':
            # ReduceLROnPlateau scheduler (default)
            default_params = {
                'mode': 'min',
                'factor': 0.5,
                'patience': 5,
                'verbose': True
            }
            default_params.update(scheduler_params)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                **default_params
            )
        elif self.scheduler_type == 'one_cycle':
            # OneCycleLR scheduler
            if num_epochs is None:
                raise ValueError(
                    "num_epochs must be provided for OneCycleLR scheduler")

            steps_per_epoch = len(train_loader)
            total_steps = num_epochs * steps_per_epoch

            default_params = {
                'max_lr': learning_rate,
                'total_steps': total_steps,
                'pct_start': 0.3,
                'anneal_strategy': 'cos',
                'div_factor': 25.0,
                'final_div_factor': 1e4,
            }
            default_params.update(scheduler_params)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                **default_params
            )
        elif self.scheduler_type == 'cosine_warmup':
            # Cosine Annealing with Warmup
            if num_epochs is None:
                raise ValueError(
                    "num_epochs must be provided for cosine_warmup scheduler")

            steps_per_epoch = len(train_loader)
            total_steps = num_epochs * steps_per_epoch
            warmup_epochs = scheduler_params.get('warmup_epochs', 5)
            warmup_steps = warmup_epochs * steps_per_epoch

            # Cosine annealing steps (after warmup)
            cosine_steps = total_steps - warmup_steps

            # Create warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=scheduler_params.get('warmup_start_factor', 0.01),
                end_factor=1.0,
                total_iters=warmup_steps
            )

            # Create cosine annealing scheduler
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_steps,
                eta_min=scheduler_params.get('min_lr', 1e-6)
            )

            # Combine them using SequentialLR
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            self.warmup_steps = warmup_steps
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                             f"Supported types: 'reduce_on_plateau', 'one_cycle', 'cosine_warmup'")

        # Early stopping (based on weighted validation score, higher is better)
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max'  # For validation score, higher is better
        )

        # Metrics
        self.metrics_calculator = MetricsCalculator()

        # Training history - store all metrics
        self.history = {
            'train_loss': [],
            'train_r2': [],
            'train_r': [],
            'val_loss': [],
            'val_r2': [],
            'val_r': [],
            'val_score': [],  # Weighted validation score
            'learning_rate': [],
        }

        # Track best model by weighted validation score
        self.best_val_score = float('-inf')  # Higher is better
        self.best_epoch = 0
        self.best_metrics = {}

        # Track starting epoch for resume functionality
        self.start_epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        all_predictions = []
        all_targets = []
        all_masks = []

        for batch in self.train_loader:
            # Move data to device
            snp_data = batch['snp'].to(self.device)

            targets = {}
            masks = {}

            if 'regression_targets' in batch:
                targets['regression'] = batch['regression_targets'].to(
                    self.device)
                masks['regression'] = batch['regression_mask'].to(self.device)
                all_targets.append(targets['regression'].cpu().numpy())
                all_masks.append(masks['regression'].cpu().numpy())

            if 'classification_targets' in batch:
                targets['classification'] = batch['classification_targets'].to(
                    self.device)
                masks['classification'] = batch['classification_mask'].to(
                    self.device)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                # Use autocast for forward pass
                with torch.amp.autocast(device_type='cuda'):
                    predictions = self.model(snp_data)
                    # Compute loss
                    loss, loss_dict = self.criterion(
                        predictions, targets, masks)

                # Backward pass with scaler
                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first)
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_norm
                    )

                # Record scale before optimizer step
                # If scale decreases after scaler.update(), it means overflow occurred
                # and optimizer.step() was skipped
                scale_before = self.scaler.get_scale()

                # Optimizer step with scaler (may be skipped if overflow detected)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Check if optimizer actually stepped (scale didn't decrease)
                stepped = self.scaler.get_scale() >= scale_before

                # Only step scheduler if optimizer actually updated parameters
                if stepped and self.scheduler_type in ['one_cycle', 'cosine_warmup']:
                    self.scheduler.step()
            else:
                # Standard precision training
                predictions = self.model(snp_data)

                # Compute loss
                loss, loss_dict = self.criterion(predictions, targets, masks)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip_norm
                    )

                self.optimizer.step()

                # Step per-batch schedulers after optimizer step
                if self.scheduler_type in ['one_cycle', 'cosine_warmup']:
                    self.scheduler.step()

            # Store predictions for metrics (outside autocast context for mixed precision)
            if 'regression' in predictions:
                all_predictions.append(
                    predictions['regression'].detach().cpu().numpy())

            epoch_losses.append(loss.item())

        # Compute training metrics
        metrics = {'train_loss': np.mean(epoch_losses)}

        # Compute training R² and Pearson using MetricsCalculator
        if all_predictions and all_targets:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            all_masks = np.concatenate(all_masks, axis=0)

            train_reg_metrics = self.metrics_calculator.compute_regression_metrics(
                all_predictions, all_targets, all_masks
            )

            if 'avg_r2' in train_reg_metrics:
                metrics['train_r2'] = train_reg_metrics['avg_r2']
            if 'avg_pearson' in train_reg_metrics:
                metrics['train_pearson'] = train_reg_metrics['avg_pearson']

        return metrics

    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}, {}

        self.model.eval()
        epoch_losses = []

        # Collect all predictions and targets
        all_predictions = {'regression': [], 'classification': []}
        all_targets = {'regression': [], 'classification': []}
        all_masks = {'regression': [], 'classification': []}

        for batch in self.val_loader:
            snp_data = batch['snp'].to(self.device)

            targets = {}
            masks = {}

            if 'regression_targets' in batch:
                targets['regression'] = batch['regression_targets'].to(
                    self.device)
                masks['regression'] = batch['regression_mask'].to(self.device)
                all_targets['regression'].append(
                    targets['regression'].cpu().numpy())
                all_masks['regression'].append(
                    masks['regression'].cpu().numpy())

            if 'classification_targets' in batch:
                targets['classification'] = batch['classification_targets'].to(
                    self.device)
                masks['classification'] = batch['classification_mask'].to(
                    self.device)
                all_targets['classification'].append(
                    targets['classification'].cpu().numpy())
                all_masks['classification'].append(
                    masks['classification'].cpu().numpy())

            # Forward pass (use autocast for mixed precision if enabled)
            if self.use_mixed_precision:
                with torch.amp.autocast(device_type='cuda'):
                    predictions = self.model(snp_data)
                    # Compute loss
                    loss, _ = self.criterion(predictions, targets, masks)
            else:
                predictions = self.model(snp_data)
                # Compute loss
                loss, _ = self.criterion(predictions, targets, masks)

            # Store predictions
            if 'regression' in predictions:
                all_predictions['regression'].append(
                    predictions['regression'].cpu().numpy())
            if 'classification' in predictions:
                all_predictions['classification'].append(
                    predictions['classification'].cpu().numpy())
            epoch_losses.append(loss.item())

        # Concatenate all batches (only for tasks that have data)
        for key in ['regression', 'classification']:
            if all_predictions[key] and all_targets[key]:
                all_predictions[key] = np.concatenate(
                    all_predictions[key], axis=0)
                all_targets[key] = np.concatenate(all_targets[key], axis=0)
                all_masks[key] = np.concatenate(all_masks[key], axis=0)
            else:
                # Clear empty lists to avoid issues downstream
                all_predictions[key] = None
                all_targets[key] = None
                all_masks[key] = None

        # Compute metrics
        metrics = {'val_loss': np.mean(epoch_losses)}

        if all_predictions['regression'] is not None:
            # Compute regression metrics using MetricsCalculator
            reg_metrics = self.metrics_calculator.compute_regression_metrics(
                all_predictions['regression'],
                all_targets['regression'],
                all_masks['regression']
            )

            # Extract key metrics for display
            if 'avg_r2' in reg_metrics:
                metrics['val_r2'] = reg_metrics['avg_r2']
            if 'avg_pearson' in reg_metrics:
                metrics['val_pearson'] = reg_metrics['avg_pearson']

            # Store all detailed metrics
            metrics.update({f'val_reg_{k}': v for k, v in reg_metrics.items()})

        if all_predictions['classification'] is not None:
            cls_metrics = self.metrics_calculator.compute_classification_metrics(
                all_predictions['classification'],
                all_targets['classification'],
                all_masks['classification']
            )
            metrics.update({f'val_cls_{k}': v for k, v in cls_metrics.items()})

        return metrics, all_predictions

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint (only on rank 0 for distributed training)."""
        # Only save on rank 0
        if self.is_distributed and self.rank != 0:
            # Non-rank-0 processes wait here for rank 0 to finish saving
            dist.barrier()
            return

        # Get model state dict (handle DDP wrapper)
        if self.is_distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'best_val_score': self.best_val_score,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'early_stopping_counter': self.early_stopping.counter,
            'early_stopping_best_score': self.early_stopping.best_score,
        }

        # Save scaler state if using mixed precision
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            # print(f"Saved best model at epoch {epoch}")

        # Synchronize all ranks after checkpoint saving (rank 0 signals completion)
        if self.is_distributed:
            dist.barrier()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint (handles DDP wrapper)."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state dict (handle DDP wrapper)
        state_dict = checkpoint['model_state_dict']
        if self.is_distributed:
            # Model is wrapped with DDP, load into module
            self.model.module.load_state_dict(state_dict)
        else:
            # Regular model
            self.model.load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']

        # Restore scaler state if using mixed precision
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore best model tracking
        if 'best_val_score' in checkpoint:
            self.best_val_score = checkpoint['best_val_score']
        if 'best_epoch' in checkpoint:
            self.best_epoch = checkpoint['best_epoch']
        if 'best_metrics' in checkpoint:
            self.best_metrics = checkpoint['best_metrics']

        # Restore early stopping state
        if 'early_stopping_counter' in checkpoint:
            self.early_stopping.counter = checkpoint['early_stopping_counter']
        if 'early_stopping_best_score' in checkpoint:
            self.early_stopping.best_score = checkpoint['early_stopping_best_score']

        return checkpoint['epoch'], checkpoint['metrics']

    def train(self, num_epochs: int, verbose: bool = True) -> Dict[str, list]:
        """
        Train the model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        if self.rank == 0:
            if self.is_distributed:
                # Get parameter count from module for DDP
                num_params = sum(p.numel()
                                 for p in self.model.module.parameters())
            else:
                num_params = sum(p.numel() for p in self.model.parameters())
            print(f"Training on device: {self.device}")
            print(f"Number of parameters: {num_params:,}")
            print()  # Empty line before training starts

        start_time = time.time()

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            # Set epoch for DistributedSampler to ensure different data shuffling per epoch
            if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Store learning rate at the beginning of each epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)

            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_r2'].append(train_metrics.get('train_r2', 0.0))
            self.history['train_r'].append(
                train_metrics.get('train_pearson', 0.0))

            # Validate
            if self.val_loader is not None:
                val_metrics, _ = self.validate()

                # Synchronize validation metrics across all ranks for consistency
                # This ensures all ranks have the same validation results
                if self.is_distributed:
                    # Average validation loss across all ranks
                    val_loss_tensor = torch.tensor([val_metrics['val_loss']],
                                                   device=self.device, dtype=torch.float32)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_metrics['val_loss'] = (
                        val_loss_tensor.item() / dist.get_world_size())

                    # Average other metrics if they exist
                    if 'val_r2' in val_metrics:
                        val_r2_tensor = torch.tensor([val_metrics['val_r2']],
                                                     device=self.device, dtype=torch.float32)
                        dist.all_reduce(val_r2_tensor, op=dist.ReduceOp.SUM)
                        val_metrics['val_r2'] = (
                            val_r2_tensor.item() / dist.get_world_size())

                    if 'val_pearson' in val_metrics:
                        val_r_tensor = torch.tensor([val_metrics['val_pearson']],
                                                    device=self.device, dtype=torch.float32)
                        dist.all_reduce(val_r_tensor, op=dist.ReduceOp.SUM)
                        val_metrics['val_pearson'] = (
                            val_r_tensor.item() / dist.get_world_size())

                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_r2'].append(val_metrics.get('val_r2', 0.0))
                self.history['val_r'].append(
                    val_metrics.get('val_pearson', 0.0))

                # Calculate weighted validation score
                val_r = val_metrics.get('val_pearson', 0.0)
                val_r2 = val_metrics.get('val_r2', 0.0)
                val_score = self.val_score_r_weight * val_r + self.val_score_r2_weight * val_r2
                self.history['val_score'].append(val_score)

                # Learning rate scheduling (only for ReduceLROnPlateau)
                if self.scheduler_type == 'reduce_on_plateau':
                    self.scheduler.step(val_metrics['val_loss'])

                # Check for improvement based on weighted validation score
                is_best = val_score > self.best_val_score
                if is_best:
                    self.best_val_score = val_score
                    self.best_epoch = epoch + 1
                    # Store all metrics for the best model
                    self.best_metrics = {
                        'epoch': epoch + 1,
                        'train_loss': train_metrics['train_loss'],
                        'train_r2': train_metrics.get('train_r2', 0.0),
                        'train_r': train_metrics.get('train_pearson', 0.0),
                        'val_loss': val_metrics['val_loss'],
                        'val_r2': val_metrics.get('val_r2', 0.0),
                        'val_r': val_metrics.get('val_pearson', 0.0),
                        'val_score': val_score,
                    }
                    # Add all detailed metrics
                    for key, value in val_metrics.items():
                        if key.startswith('val_reg_'):
                            self.best_metrics[key] = value

                # Save checkpoint
                self.save_checkpoint(epoch + 1, val_metrics, is_best)

                # Print single-line progress (only on rank 0)
                if verbose and self.rank == 0:
                    epoch_time = time.time() - epoch_start
                    train_r2 = train_metrics.get('train_r2', 0.0)
                    train_pearson = train_metrics.get('train_pearson', 0.0)
                    val_r2 = val_metrics.get('val_r2', 0.0)
                    val_pearson = val_metrics.get('val_pearson', 0.0)

                    output = f"Epoch {epoch + 1} - {epoch_time:.0f}s  "
                    output += f"train_loss: {train_metrics['train_loss']:.4f} - "
                    output += f"train_r: {train_pearson:.4f} - "
                    output += f"train_r2: {train_r2:.4f} - "
                    output += f"valid_loss: {val_metrics['val_loss']:.4f} - "
                    output += f"valid_r: {val_pearson:.4f} - "
                    output += f"valid_r2: {val_r2:.4f} - "
                    output += f"valid_score: {val_score:.4f}"

                    if is_best:
                        output += " - best!"

                    print(output)

                # Early stopping (based on weighted validation score)
                # All ranks must check early stopping to stay synchronized
                should_stop = self.early_stopping(val_score)

                # Synchronize early stopping decision across all ranks
                if self.is_distributed:
                    # Use allreduce to ensure all ranks agree on stopping
                    stop_tensor = torch.tensor([1 if should_stop else 0],
                                               device=self.device, dtype=torch.int32)
                    dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                    should_stop = stop_tensor.item() > 0

                if should_stop:
                    if self.rank == 0:
                        print(
                            f"\nEarly stopping triggered at epoch {epoch + 1}")
                        print(
                            f"Best epoch: {self.best_epoch} with val_score: {self.best_val_score:.4f}")
                        print(
                            f"  (val_r: {self.best_metrics.get('val_r', 0.0):.4f}, val_r2: {self.best_metrics.get('val_r2', 0.0):.4f})")
                    break
            else:
                # No validation set, just save checkpoint
                self.save_checkpoint(epoch + 1, train_metrics)

                if verbose and self.rank == 0:
                    epoch_time = time.time() - epoch_start
                    train_r2 = train_metrics.get('train_r2', 0.0)
                    output = f"Epoch {epoch + 1} - {epoch_time:.0f}s - "
                    output += f"train_loss: {train_metrics['train_loss']:.4f} - "
                    output += f"train_r2: {train_r2:.4f}"
                    print(output)

        if self.rank == 0:
            elapsed_time = time.time() - start_time
            print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")
            print(f"Best model at epoch {self.best_epoch}:")
            print(f"  Validation Score: {self.best_val_score:.4f}")
            if self.best_metrics:
                print(
                    f"  Validation Pearson R: {self.best_metrics.get('val_r', 0.0):.4f}")
                print(
                    f"  Validation R²: {self.best_metrics.get('val_r2', 0.0):.4f}")
                print(
                    f"  Validation Loss: {self.best_metrics.get('val_loss', 0.0):.4f}")

        # Save training history as JSON (only on rank 0)
        if self.rank == 0:
            history_path_json = self.checkpoint_dir / 'training_history.json'
            with open(history_path_json, 'w') as f:
                json.dump(self.history, f, indent=2)

            # Save training history as TSV for easier viewing (in parent directory)
            history_path_tsv = self.checkpoint_dir.parent / 'training_history.tsv'
            import pandas as pd

            # Create DataFrame from history - handle different lengths
            num_epochs = len(self.history['train_loss'])
            history_data = {
                'epoch': range(1, num_epochs + 1),
                'train_loss': self.history['train_loss'],
                'train_r2': self.history['train_r2'],
                'train_r': self.history['train_r'],
                'learning_rate': self.history['learning_rate'],
            }

            # Add validation metrics if available
            if self.history['val_loss']:
                history_data['val_loss'] = self.history['val_loss']
                history_data['val_r2'] = self.history['val_r2']
                history_data['val_r'] = self.history['val_r']
                history_data['val_score'] = self.history['val_score']

            history_df = pd.DataFrame(history_data)
            history_df.to_csv(history_path_tsv, sep='\t',
                              index=False, float_format='%.6f')
            history_path = history_path_tsv

            # Save best model metrics with all details
            best_metrics_path = self.checkpoint_dir / 'best_metrics.json'
            with open(best_metrics_path, 'w') as f:
                json.dump(self.best_metrics, f, indent=2)

            print(f"\nMetrics saved:")
            print(f"  Training history (TSV): {history_path_tsv}")
            print(f"  Training history (JSON): {history_path_json}")
            print(f"  Best model metrics: {best_metrics_path}")

        # Synchronize all ranks after file saving (if distributed)
        if self.is_distributed:
            dist.barrier()

        return self.history
