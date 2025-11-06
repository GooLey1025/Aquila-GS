#!/usr/bin/env python3
"""
Aquila VCF Training Script

Train a multi-task deep learning model for genomic prediction using VCF input.
Supports multi-branch architectures for SNP/INDEL/SV variants.

Usage:
    python aquila_train.py --config params.yaml --vcf data.vcf --pheno data.pheno.tsv
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import time
from aquila.varnn import create_model_from_config
from aquila.trainer import VarTrainer
from aquila.data_utils import create_data_loaders
from aquila.utils import set_seed, save_config, load_config, print_model_summary
import pandas as pd
from aquila.metrics import MetricsCalculator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Aquila model with VCF input')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--vcf',
        type=str,
        default=None,
        help='Path to VCF file (overrides config)'
    )
    
    parser.add_argument(
        '--pheno',
        type=str,
        default=None,
        help='Path to phenotype file (overrides config)'
    )
    
    parser.add_argument(
        '--encoding-type',
        type=str,
        default='snp_vcf',
        choices=['snp_vcf', 'snp_indel_vcf', 'snp_indel_sv_vcf'],
        help='VCF encoding type: snp_vcf (SNP only), snp_indel_vcf (SNP+INDEL), snp_indel_sv_vcf (SNP+INDEL+SV)'
    )
    
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='./outputs',
        help='Output directory for checkpoints and results'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (cuda or cpu)'
    )
    
    parser.add_argument(
        '-dr',
        '--data-restart',
        action='store_true',
        help='Ignore data cache and re-process data from scratch'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from latest checkpoint'
    )
    
    parser.add_argument(
        '-st',
        '--skew-threshold',
        type=float,
        default=2.0,
        help='Skewness threshold for auto log transformation (default: 2.0, set to 0 to disable)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print("=" * 80)
    print("AQUILA: VCF Training")
    print("=" * 80)
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.vcf:
        config['data']['geno_file'] = args.vcf
    if args.pheno:
        config['data']['pheno_file'] = args.pheno
    
    # Set encoding type
    encoding_type = args.encoding_type
    config['data']['encoding_type'] = encoding_type
    
    # Set random seed
    set_seed(args.seed)
    
    print(f"Random seed: {args.seed}")
    print(f"Encoding type: {encoding_type}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, output_dir / 'params.yaml')
    
    # Extract configuration
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    train_config = config.get('train', {})
    
    # Get data paths
    vcf_file = data_config['geno_file']
    pheno_file = data_config['pheno_file']
    
    # Get classification task configuration
    classification_tasks = data_config.get('classification_tasks', None)
    
    print(f"\nData Configuration:")
    print(f"  VCF file: {vcf_file}")
    print(f"  Phenotype file: {pheno_file}")
    print(f"  Encoding type: {encoding_type}")
    
    # Explain task assignment logic
    if classification_tasks is not None:
        print(f"  Classification tasks specified: {classification_tasks}")
    else:
        print(f"  All traits are treated as regression")
    
    # Create data loaders
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)
    
    time1 = time.time()
    train_loader, val_loader, test_loader, normalization_stats = create_data_loaders(
        geno_path=vcf_file,
        pheno_path=pheno_file,
        classification_tasks=classification_tasks,
        batch_size=train_config.get('batch_size', 32),
        val_split=train_config.get('val_split', 0.2),
        test_split=train_config.get('test_split', 0.0),
        num_workers=train_config.get('num_workers', 4),
        normalize_regression=train_config.get('normalize_regression', True),
        encoding_type=encoding_type,
        cache_dir=output_dir,
        data_restart=args.data_restart,
        skew_threshold=args.skew_threshold,
    )
    time2 = time.time()
    print(f"Time taken to load/process data: {time2 - time1:.2f} seconds")
    
    # Print log transformation info
    if normalization_stats and 'log_transformed_tasks' in normalization_stats:
        log_transformed = normalization_stats['log_transformed_tasks']
        if log_transformed:
            print(f"\nLog-transformed phenotypes ({len(log_transformed)}):")
            for task in log_transformed:
                print(f"  - {task}")
        else:
            print(f"\nNo phenotypes required log transformation (skew threshold: {args.skew_threshold})")
    
    # Save normalization statistics
    if normalization_stats:
        import pickle
        norm_path = output_dir / 'normalization_stats.pkl'
        with open(norm_path, 'wb') as f:
            pickle.dump(normalization_stats, f)
        print(f"\nNormalization statistics saved to: {norm_path}")
    
    # Get actual task lists from first batch
    sample_batch = next(iter(train_loader))
    num_regression_tasks = 0
    num_classification_tasks = 0
    
    # Determine sequence length(s) for multi-branch
    is_multi_branch = encoding_type in ['snp_indel_vcf', 'snp_indel_sv_vcf']
    
    if is_multi_branch:
        # Multi-branch: get seq_length for each variant type
        seq_length = {}
        for key in sample_batch.keys():
            if key not in ['sample_id', 'regression_targets', 'regression_mask', 
                          'classification_targets', 'classification_mask']:
                seq_length[key] = sample_batch[key].shape[1]
        print(f"\nMulti-branch sequence lengths: {seq_length}")
    else:
        # Single-branch
        seq_length = sample_batch['snp'].shape[1]
        print(f"\nSequence length (variants): {seq_length}")
    
    if 'regression_targets' in sample_batch:
        num_regression_tasks = sample_batch['regression_targets'].shape[1]
    if 'classification_targets' in sample_batch:
        num_classification_tasks = sample_batch['classification_targets'].shape[1]
    
    print(f"Number of regression tasks: {num_regression_tasks}")
    print(f"Number of classification tasks: {num_classification_tasks}")
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)
    
    # Generate task names if not provided
    regression_task_names = None
    classification_task_names = None
    
    if num_regression_tasks > 0:
        # Get task names from normalization stats if available
        if normalization_stats and 'regression_tasks' in normalization_stats:
            regression_task_names = normalization_stats['regression_tasks']
        else:
            regression_task_names = [f"regression_task_{i}" for i in range(num_regression_tasks)]
    
    if num_classification_tasks > 0:
        if classification_tasks:
            classification_task_names = classification_tasks
        else:
            classification_task_names = [f"classification_task_{i}" for i in range(num_classification_tasks)]
    
    print(f"\nTask Configuration:")
    if regression_task_names:
        print(f"  Regression tasks: {regression_task_names}")
    if classification_task_names:
        print(f"  Classification tasks: {classification_task_names}")
    
    # Create model from config
    model = create_model_from_config(
        config=config,
        seq_length=seq_length,
        regression_tasks=regression_task_names,
        classification_tasks=classification_task_names
    )
    
    # Print model summary
    batch_size = train_config.get('batch_size', 32)
    
    if is_multi_branch:
        # Multi-branch: create dict input
        model_input_size = {}
        for vtype, vlen in seq_length.items():
            model_input_size[vtype] = (batch_size, vlen, 8)
        print(f"\nMulti-branch model input sizes: {model_input_size}")
    else:
        # Single-branch
        model_input_size = (batch_size, seq_length, 8)
    
    print_model_summary(
        model=model,
        input_size=model_input_size,
        device=args.device,
        verbose=config.get('model', {}).get('verbose', False),
        encoding_type=encoding_type
    )
    
    # Create trainer
    print("\n" + "=" * 80)
    print("Training Setup")
    print("=" * 80)
    
    trainer = VarTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_regression_tasks=num_regression_tasks,
        num_classification_tasks=num_classification_tasks,
        learning_rate=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 1e-5),
        loss_type=train_config.get('loss_type', 'mse'),
        uncertainty_weighting=train_config.get('uncertainty_weighting', True),
        device=args.device,
        checkpoint_dir=str(output_dir / 'checkpoints'),
        early_stopping_patience=train_config.get('early_stopping_patience', 20),
        gradient_clip_norm=train_config.get('gradient_clip_norm', 1.0),
        scheduler_type=train_config.get('scheduler_type', 'reduce_on_plateau'),
        scheduler_params=train_config.get('scheduler_params', None),
        num_epochs=train_config.get('num_epochs', 100),
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Learning rate: {train_config.get('learning_rate', 1e-4)}")
    print(f"  Batch size: {train_config.get('batch_size', 32)}")
    print(f"  Max epochs: {train_config.get('num_epochs', 100)}")
    print(f"  Early stopping patience: {train_config.get('early_stopping_patience', 20)}")
    print(f"  Loss type: {train_config.get('loss_type', 'mse')}")
    print(f"  Uncertainty weighting: {train_config.get('uncertainty_weighting', True)}")
    print(f"  Scheduler type: {train_config.get('scheduler_type', 'reduce_on_plateau')}")
    
    # Resume from checkpoint if requested
    if args.resume:
        checkpoint_path = output_dir / 'checkpoints' / 'latest_checkpoint.pt'
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            epoch, metrics = trainer.load_checkpoint(str(checkpoint_path))
            trainer.start_epoch = epoch
            print(f"Resuming from epoch {epoch + 1}")
            print(f"Best validation score so far: {trainer.best_val_score:.4f} at epoch {trainer.best_epoch}")
            print(f"Early stopping counter: {trainer.early_stopping.counter}/{train_config.get('early_stopping_patience', 20)}")
        else:
            print("\nWarning: --resume specified but no checkpoint found. Starting fresh training.")
    
    # Train model
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")
    
    history = trainer.train(
        num_epochs=train_config.get('num_epochs', 100),
        verbose=True
    )
    
    # Evaluation on test set if available
    if test_loader:
        print("\n" + "=" * 80)
        print("Evaluating on Test Set")
        print("=" * 80)
        
        # Load best model
        best_checkpoint = output_dir / 'checkpoints' / 'best_checkpoint.pt'
        if best_checkpoint.exists():
            trainer.load_checkpoint(str(best_checkpoint))
            print("Loaded best checkpoint")
        
        # Collect predictions and targets from test set
        trainer.model.eval()
        all_sample_ids = []
        all_predictions_reg = []
        all_targets_reg = []
        all_masks_reg = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle multi-branch or single-branch input
                if is_multi_branch:
                    # Multi-branch: create dict input
                    variant_inputs = {}
                    for key in batch.keys():
                        if key not in ['sample_id', 'regression_targets', 'regression_mask',
                                      'classification_targets', 'classification_mask']:
                            variant_inputs[key] = batch[key].to(trainer.device)
                    snp_data = variant_inputs
                else:
                    # Single-branch
                    snp_data = batch['snp'].to(trainer.device)
                
                all_sample_ids.extend(batch['sample_id'])
                
                # Get predictions
                predictions = trainer.model(snp_data)
                
                if 'regression_targets' in batch and 'regression' in predictions:
                    all_predictions_reg.append(predictions['regression'].cpu().numpy())
                    all_targets_reg.append(batch['regression_targets'].numpy())
                    all_masks_reg.append(batch['regression_mask'].numpy())
        
        # Concatenate all batches
        if all_predictions_reg:
            predictions_normalized = np.concatenate(all_predictions_reg, axis=0)
            targets_normalized = np.concatenate(all_targets_reg, axis=0)
            masks = np.concatenate(all_masks_reg, axis=0)
            
            # Calculate metrics on normalized data
            metrics_calc = MetricsCalculator()
            test_metrics_normalized = metrics_calc.compute_regression_metrics(
                predictions_normalized, targets_normalized, masks
            )
            
            # Load normalization statistics for denormalization
            norm_stats_path = output_dir / 'normalization_stats.pkl'
            if norm_stats_path.exists():
                import pickle
                with open(norm_stats_path, 'rb') as f:
                    norm_stats = pickle.load(f)
                
                regression_means = norm_stats['regression_means']
                regression_stds = norm_stats['regression_stds']
                regression_task_names = norm_stats['regression_tasks']
                log_transformed_tasks = norm_stats.get('log_transformed_tasks', [])
                
                # Denormalize predictions and targets (reverse Z-score)
                predictions_log_scale = predictions_normalized.copy()
                targets_log_scale = targets_normalized.copy()
                
                for i in range(len(regression_means)):
                    # Apply inverse Z-score: log_scale = (normalized * std) + mean
                    predictions_log_scale[:, i] = (predictions_normalized[:, i] * regression_stds[i]) + regression_means[i]
                    targets_log_scale[:, i] = (targets_normalized[:, i] * regression_stds[i]) + regression_means[i]
                
                # Calculate metrics on log-transformed scale (after reverse Z-score)
                test_metrics_log_scale = metrics_calc.compute_regression_metrics(
                    predictions_log_scale, targets_log_scale, masks
                )
                
                # Reverse log transformation to get original scale
                predictions_original = predictions_log_scale.copy()
                targets_original = targets_log_scale.copy()
                
                for i, task_name in enumerate(regression_task_names):
                    if task_name in log_transformed_tasks:
                        # Reverse log(x+1) transformation: original = exp(log_value) - 1
                        predictions_original[:, i] = np.expm1(predictions_log_scale[:, i])
                        targets_original[:, i] = np.expm1(targets_log_scale[:, i])
                
                # Calculate metrics on original scale
                test_metrics_original = metrics_calc.compute_regression_metrics(
                    predictions_original, targets_original, masks
                )
                
                # Print both sets of metrics
                if log_transformed_tasks:
                    print("\n" + "=" * 80)
                    print("Test Set Results (Log-Transformed Scale):")
                    print("=" * 80)
                    for key, value in test_metrics_log_scale.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                
                print("\n" + "=" * 80)
                print("Test Set Results (Original Scale):")
                print("=" * 80)
                for key, value in test_metrics_original.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                
                # Save predictions on both scales
                import json
                
                # Original scale predictions (primary)
                pred_data_original = {'Sample_ID': all_sample_ids}
                for i, task_name in enumerate(regression_task_names):
                    target_col = []
                    pred_col = []
                    for j in range(len(all_sample_ids)):
                        if masks[j, i] == 0:
                            target_col.append("Missing")
                        else:
                            target_col.append(targets_original[j, i])
                        pred_col.append(predictions_original[j, i])
                    
                    pred_data_original[f'{task_name}_Target'] = target_col
                    pred_data_original[f'{task_name}_Pred'] = pred_col
                
                df_original = pd.DataFrame(pred_data_original)
                pred_path_original = output_dir / 'test_predictions_original_scale.tsv'
                df_original.to_csv(pred_path_original, sep='\t', index=False, float_format='%.6f')
                print(f"\nPredictions (original scale) saved to: {pred_path_original}")
                
                # Log scale predictions (if any log transformations were applied)
                if log_transformed_tasks:
                    pred_data_log = {'Sample_ID': all_sample_ids}
                    for i, task_name in enumerate(regression_task_names):
                        target_col = []
                        pred_col = []
                        for j in range(len(all_sample_ids)):
                            if masks[j, i] == 0:
                                target_col.append("Missing")
                            else:
                                target_col.append(targets_log_scale[j, i])
                            pred_col.append(predictions_log_scale[j, i])
                        
                        pred_data_log[f'{task_name}_Target'] = target_col
                        pred_data_log[f'{task_name}_Pred'] = pred_col
                    
                    df_log = pd.DataFrame(pred_data_log)
                    pred_path_log = output_dir / 'test_predictions_log_scale.tsv'
                    df_log.to_csv(pred_path_log, sep='\t', index=False, float_format='%.6f')
                    print(f"Predictions (log scale) saved to: {pred_path_log}")
                
                # Save metrics as JSON
                metrics_path_original = output_dir / 'test_metrics_original_scale.json'
                with open(metrics_path_original, 'w') as f:
                    json.dump(test_metrics_original, f, indent=2)
                print(f"Metrics (original scale) saved to: {metrics_path_original}")
                
                if log_transformed_tasks:
                    metrics_path_log = output_dir / 'test_metrics_log_scale.json'
                    with open(metrics_path_log, 'w') as f:
                        json.dump(test_metrics_log_scale, f, indent=2)
                    print(f"Metrics (log scale) saved to: {metrics_path_log}")
            else:
                print("\nWarning: Normalization statistics not found. Predictions saved in normalized scale.")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Checkpoints: {output_dir / 'checkpoints'}")
    print(f"  - Config: {output_dir / 'params.yaml'}")
    print(f"  - Training history (TSV): {output_dir / 'training_history.tsv'}")
    print(f"  - Training history (JSON): {output_dir / 'checkpoints' / 'training_history.json'}")


if __name__ == '__main__':
    main()

