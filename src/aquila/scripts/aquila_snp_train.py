#!/usr/bin/env python3
"""
Aquila SNP Training Script

Train a multi-task deep learning model for genomic prediction.

Usage:
    python aquila_snp_train.py --config params.yaml
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import time
from aquila.snpnn import SNPNeuralNetwork, create_model_from_config
from aquila.trainer import SNPTrainer
from aquila.data_utils import create_data_loaders
from aquila.utils import set_seed, save_config, load_config, print_model_summary
import pandas as pd
from aquila.metrics import MetricsCalculator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Aquila SNP model')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--geno',
        type=str,
        default=None,
        help='Path to genotype file (overrides config)'
    )
    
    parser.add_argument(
        '--pheno',
        type=str,
        default=None,
        help='Path to phenotype file (overrides config)'
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
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print("=" * 80)
    print("AQUILA: SNP Neural Network Training")
    print("=" * 80)
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.geno:
        config['data']['geno_file'] = args.geno
    if args.pheno:
        config['data']['pheno_file'] = args.pheno
    
    # Set random seed
    set_seed(args.seed)
    
    print(f"Random seed: {args.seed}")
    
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
    geno_file = data_config.get('geno_file', 'data/data.geno')
    pheno_file = data_config.get('pheno_file', 'data/data.pheno')
    
    # Get classification task configuration
    classification_tasks = data_config.get('classification_tasks', None)
    
    # Get encoding type
    encoding_type = data_config.get('encoding_type', 'token')
    
    print(f"\nData Configuration:")
    print(f"  Genotype file: {geno_file}")
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
        geno_path=geno_file,
        pheno_path=pheno_file,
        classification_tasks=classification_tasks,
        batch_size=train_config.get('batch_size', 32),
        val_split=train_config.get('val_split', 0.2),
        test_split=train_config.get('test_split', 0.0),
        num_workers=train_config.get('num_workers', 4),
        normalize_regression=train_config.get('normalize_regression', True),
        encoding_type=encoding_type,
    )
    time2 = time.time()
    print(f"Time taken to load data: {time2 - time1} seconds")
    
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
    seq_length = sample_batch['snp'].shape[1]
    
    if 'regression_targets' in sample_batch:
        num_regression_tasks = sample_batch['regression_targets'].shape[1]
    if 'classification_targets' in sample_batch:
        num_classification_tasks = sample_batch['classification_targets'].shape[1]
    
    print(f"\nDataset Information:")
    print(f"  Sequence length (SNPs): {seq_length}")
    print(f"  Number of regression tasks: {num_regression_tasks}")
    print(f"  Number of classification tasks: {num_classification_tasks}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"  Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"  Test samples: {len(test_loader.dataset)}")
    
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
    
    # Print model summary with torchinfo
    # Prepare input size based on encoding type
    batch_size = train_config.get('batch_size', 32)
    if encoding_type == 'diploid_onehot':
        # For diploid one-hot: (batch_size, seq_length, 8)
        model_input_size = (batch_size, seq_length, 8)
    else:
        # For token encoding: (batch_size, seq_length)
        model_input_size = (batch_size, seq_length)
    
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
    
    trainer = SNPTrainer(
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
                
                # Denormalize predictions and targets
                predictions_denormalized = predictions_normalized.copy()
                targets_denormalized = targets_normalized.copy()
                
                for i in range(len(regression_means)):
                    # Apply inverse Z-score: original = (normalized * std) + mean
                    predictions_denormalized[:, i] = (predictions_normalized[:, i] * regression_stds[i]) + regression_means[i]
                    targets_denormalized[:, i] = (targets_normalized[:, i] * regression_stds[i]) + regression_means[i]
                
                # Calculate metrics on denormalized data
                test_metrics_denormalized = metrics_calc.compute_regression_metrics(
                    predictions_denormalized, targets_denormalized, masks
                )
                
                print("\n" + "=" * 80)
                print("Test Set Results (Denormalized/Original Scale):")
                print("=" * 80)
                for key, value in test_metrics_denormalized.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                
                # Save denormalized predictions as TSV
                pred_data_denormalized = {'Sample_ID': all_sample_ids}
                for i, task_name in enumerate(regression_task_names):
                    # Create target column with "Missing" for invalid values
                    target_col = []
                    pred_col = []
                    for j in range(len(all_sample_ids)):
                        if masks[j, i] == 0:  # Invalid/missing value
                            target_col.append("Missing")
                        else:
                            target_col.append(targets_denormalized[j, i])
                        pred_col.append(predictions_denormalized[j, i])
                    
                    pred_data_denormalized[f'{task_name}_Target'] = target_col
                    pred_data_denormalized[f'{task_name}_Pred'] = pred_col
                
                df_denormalized = pd.DataFrame(pred_data_denormalized)
                pred_path_denormalized = output_dir / 'test_predictions_denormalized.tsv'
                df_denormalized.to_csv(pred_path_denormalized, sep='\t', index=False, float_format='%.6f')
                print(f"\nPredictions saved to: {pred_path_denormalized}")
                
                # Save metrics as JSON
                import json
                metrics_path_denormalized = output_dir / 'test_metrics_denormalized.json'
                with open(metrics_path_denormalized, 'w') as f:
                    json.dump(test_metrics_denormalized, f, indent=2)
                print(f"Metrics saved to: {metrics_path_denormalized}")
            else:
                print("\nWarning: Normalization statistics not found. Predictions will be saved in normalized scale.")
                
                # Save predictions as TSV (without task names, in normalized scale)
                pred_data = {'Sample_ID': all_sample_ids}
                for i in range(predictions_normalized.shape[1]):
                    # Create target column with "Missing" for invalid values
                    target_col = []
                    pred_col = []
                    for j in range(len(all_sample_ids)):
                        if masks[j, i] == 0:  # Invalid/missing value
                            target_col.append("Missing")
                        else:
                            target_col.append(targets_normalized[j, i])
                        pred_col.append(predictions_normalized[j, i])
                    
                    pred_data[f'Task_{i}_Target'] = target_col
                    pred_data[f'Task_{i}_Pred'] = pred_col
                
                df_pred = pd.DataFrame(pred_data)
                pred_path = output_dir / 'test_predictions.tsv'
                df_pred.to_csv(pred_path, sep='\t', index=False, float_format='%.6f')
                print(f"\nPredictions saved to: {pred_path}")
    
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

