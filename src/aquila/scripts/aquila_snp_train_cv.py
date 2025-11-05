#!/usr/bin/env python3
"""
Aquila SNP Training Script with K-Fold Cross-Validation

Train multiple models using k-fold cross-validation for robust evaluation.

Usage:
    python aquila_snp_train_cv.py --config params.yaml --n-folds 10
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import time
import json
from aquila.snpnn import SNPNeuralNetwork, create_model_from_config
from aquila.trainer import SNPTrainer
from aquila.data_utils import create_kfold_data_loaders
from aquila.utils import set_seed, save_config, load_config, print_model_summary
import pandas as pd
from aquila.metrics import MetricsCalculator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Aquila SNP model with k-fold cross-validation')
    
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
        default='./outputs_cv',
        help='Output directory for checkpoints and results'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=10,
        help='Number of folds for cross-validation'
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


def evaluate_fold_on_val(
    trainer, 
    val_loader, 
    normalization_stats,
    fold_output_dir
):
    """
    Evaluate a trained model on the validation set and save results.
    
    Returns:
        metrics_denormalized, predictions_denormalized, targets_denormalized, masks, sample_ids, task_names
    """
    trainer.model.eval()
    all_sample_ids = []
    all_predictions_reg = []
    all_targets_reg = []
    all_masks_reg = []
    
    with torch.no_grad():
        for batch in val_loader:
            snp_data = batch['snp'].to(trainer.device)
            all_sample_ids.extend(batch['sample_id'])
            
            # Get predictions
            predictions = trainer.model(snp_data)
            
            if 'regression_targets' in batch and 'regression' in predictions:
                all_predictions_reg.append(predictions['regression'].cpu().numpy())
                all_targets_reg.append(batch['regression_targets'].numpy())
                all_masks_reg.append(batch['regression_mask'].numpy())
    
    # Concatenate all batches
    if not all_predictions_reg:
        return None, None, None, None, None, None
    
    predictions_normalized = np.concatenate(all_predictions_reg, axis=0)
    targets_normalized = np.concatenate(all_targets_reg, axis=0)
    masks = np.concatenate(all_masks_reg, axis=0)
    
    # Denormalize predictions and targets
    regression_means = normalization_stats['regression_means']
    regression_stds = normalization_stats['regression_stds']
    regression_task_names = normalization_stats['regression_tasks']
    
    predictions_denormalized = predictions_normalized.copy()
    targets_denormalized = targets_normalized.copy()
    
    for i in range(len(regression_means)):
        # Apply inverse Z-score: original = (normalized * std) + mean
        predictions_denormalized[:, i] = (predictions_normalized[:, i] * regression_stds[i]) + regression_means[i]
        targets_denormalized[:, i] = (targets_normalized[:, i] * regression_stds[i]) + regression_means[i]
    
    # Calculate metrics on denormalized data
    metrics_calc = MetricsCalculator()
    metrics_denormalized = metrics_calc.compute_regression_metrics(
        predictions_denormalized, targets_denormalized, masks
    )
    
    return metrics_denormalized, predictions_denormalized, targets_denormalized, masks, all_sample_ids, regression_task_names


def save_fold_predictions(
    predictions_denormalized,
    targets_denormalized,
    masks,
    sample_ids,
    task_names,
    output_path
):
    """Save predictions to TSV file."""
    pred_data = {'Sample_ID': sample_ids}
    
    for i, task_name in enumerate(task_names):
        target_col = []
        pred_col = []
        for j in range(len(sample_ids)):
            if masks[j, i] == 0:  # Invalid/missing value
                target_col.append("Missing")
            else:
                target_col.append(targets_denormalized[j, i])
            pred_col.append(predictions_denormalized[j, i])
        
        pred_data[f'{task_name}_Target'] = target_col
        pred_data[f'{task_name}_Pred'] = pred_col
    
    df = pd.DataFrame(pred_data)
    df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')


def aggregate_cv_results(fold_results, output_dir):
    """
    Aggregate results from all folds and save summary.
    
    Args:
        fold_results: List of dictionaries containing fold metrics
        output_dir: Path to output directory
    """
    print("\n" + "=" * 80)
    print("Cross-Validation Summary")
    print("=" * 80)
    
    # Collect metrics across folds
    n_folds = len(fold_results)
    
    # Get all metric keys from first fold
    if not fold_results or 'val_metrics' not in fold_results[0]:
        print("Warning: No validation metrics found in fold results")
        return
    
    metric_keys = [k for k in fold_results[0]['val_metrics'].keys() if isinstance(fold_results[0]['val_metrics'][k], (int, float))]
    
    # Compute mean and std for each metric
    aggregated_metrics = {}
    per_fold_metrics = {key: [] for key in metric_keys}
    
    for fold_result in fold_results:
        for key in metric_keys:
            if key in fold_result['val_metrics']:
                per_fold_metrics[key].append(fold_result['val_metrics'][key])
    
    for key in metric_keys:
        values = per_fold_metrics[key]
        if values:
            aggregated_metrics[f'{key}_mean'] = float(np.mean(values))
            aggregated_metrics[f'{key}_std'] = float(np.std(values))
            aggregated_metrics[f'{key}_min'] = float(np.min(values))
            aggregated_metrics[f'{key}_max'] = float(np.max(values))
    
    # Create summary dictionary
    summary = {
        'n_folds': n_folds,
        'aggregated_metrics': aggregated_metrics,
        'per_fold_metrics': {
            f'fold_{i}': {k: v for k, v in fold_results[i]['val_metrics'].items() if isinstance(v, (int, float))}
            for i in range(n_folds)
        }
    }
    
    # Save JSON summary
    summary_json_path = output_dir / 'cv_summary.json'
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved CV summary JSON: {summary_json_path}")
    
    # Create human-readable text summary
    summary_text_path = output_dir / 'cv_summary.txt'
    with open(summary_text_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{n_folds}-Fold Cross-Validation Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Aggregated Metrics (Mean ± Std):\n")
        f.write("-" * 80 + "\n")
        for key in sorted([k for k in aggregated_metrics.keys() if k.endswith('_mean')]):
            base_key = key[:-5]  # Remove '_mean'
            mean_val = aggregated_metrics[key]
            std_val = aggregated_metrics.get(f'{base_key}_std', 0.0)
            min_val = aggregated_metrics.get(f'{base_key}_min', 0.0)
            max_val = aggregated_metrics.get(f'{base_key}_max', 0.0)
            f.write(f"  {base_key:30s}: {mean_val:8.4f} ± {std_val:6.4f} (min: {min_val:8.4f}, max: {max_val:8.4f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-Fold Metrics:\n")
        f.write("=" * 80 + "\n")
        for i in range(n_folds):
            f.write(f"\nFold {i}:\n")
            f.write("-" * 40 + "\n")
            fold_metrics = summary['per_fold_metrics'][f'fold_{i}']
            for key in sorted(fold_metrics.keys()):
                f.write(f"  {key:30s}: {fold_metrics[key]:8.4f}\n")
    
    print(f"Saved CV summary text: {summary_text_path}")
    
    # Print summary to console
    print("\nAggregated Metrics (Mean ± Std):")
    print("-" * 80)
    for key in sorted([k for k in aggregated_metrics.keys() if k.endswith('_mean')]):
        base_key = key[:-5]
        mean_val = aggregated_metrics[key]
        std_val = aggregated_metrics.get(f'{base_key}_std', 0.0)
        print(f"  {base_key:30s}: {mean_val:8.4f} ± {std_val:6.4f}")


def main():
    """Main cross-validation training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print("=" * 80)
    print("AQUILA: SNP Neural Network Training with K-Fold Cross-Validation")
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
    print(f"Number of folds: {args.n_folds}")
    
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
    
    # Store results from all folds
    fold_results = []
    
    # Create k-fold data loaders
    print("\n" + "=" * 80)
    print("Starting K-Fold Cross-Validation")
    print("=" * 80)
    
    time_start_cv = time.time()
    
    kfold_generator = create_kfold_data_loaders(
        geno_path=geno_file,
        pheno_path=pheno_file,
        classification_tasks=classification_tasks,
        n_folds=args.n_folds,
        batch_size=train_config.get('batch_size', 32),
        num_workers=train_config.get('num_workers', 4),
        normalize_regression=train_config.get('normalize_regression', True),
        encoding_type=encoding_type,
        random_seed=args.seed,
    )
    
    # Iterate through folds
    for fold_idx, train_loader, val_loader, normalization_stats, seq_length in kfold_generator:
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print("=" * 80)
        
        fold_start_time = time.time()
        
        # Create fold output directory
        fold_output_dir = output_dir / f'fold_{fold_idx}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save fold-specific configuration
        save_config(config, fold_output_dir / 'params.yaml')
        
        # Save normalization statistics
        if normalization_stats:
            import pickle
            norm_path = fold_output_dir / 'normalization_stats.pkl'
            with open(norm_path, 'wb') as f:
                pickle.dump(normalization_stats, f)
        
        # Get actual task lists from first batch
        sample_batch = next(iter(train_loader))
        num_regression_tasks = 0
        num_classification_tasks = 0
        
        if 'regression_targets' in sample_batch:
            num_regression_tasks = sample_batch['regression_targets'].shape[1]
        if 'classification_targets' in sample_batch:
            num_classification_tasks = sample_batch['classification_targets'].shape[1]
        
        print(f"\nDataset Information:")
        print(f"  Sequence length (SNPs): {seq_length}")
        print(f"  Number of regression tasks: {num_regression_tasks}")
        print(f"  Number of classification tasks: {num_classification_tasks}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        
        # Generate task names
        regression_task_names = None
        classification_task_names = None
        
        if num_regression_tasks > 0:
            if normalization_stats and 'regression_tasks' in normalization_stats:
                regression_task_names = normalization_stats['regression_tasks']
            else:
                regression_task_names = [f"regression_task_{i}" for i in range(num_regression_tasks)]
        
        if num_classification_tasks > 0:
            if classification_tasks:
                classification_task_names = classification_tasks
            else:
                classification_task_names = [f"classification_task_{i}" for i in range(num_classification_tasks)]
        
        # Create model from config (fresh model for each fold)
        print(f"\nCreating Model for Fold {fold_idx}")
        print("-" * 80)
        
        # Set fold-specific seed for model initialization
        set_seed(args.seed + fold_idx + 1000)
        
        model = create_model_from_config(
            config=config,
            seq_length=seq_length,
            regression_tasks=regression_task_names,
            classification_tasks=classification_task_names
        )
        
        # Initialize lazy modules with a dummy forward pass
        model.to(args.device)
        model.eval()
        with torch.no_grad():
            dummy_batch = sample_batch['snp'][:1].to(args.device)  # Use first sample from batch
            _ = model(dummy_batch)
        
        # Print model summary only for first fold
        if fold_idx == 0:
            batch_size = train_config.get('batch_size', 32)
            if encoding_type == 'diploid_onehot':
                model_input_size = (batch_size, seq_length, 8)
            else:
                model_input_size = (batch_size, seq_length)
            
            print_model_summary(
                model=model,
                input_size=model_input_size,
                device=args.device,
                verbose=config.get('model', {}).get('verbose', False),
                encoding_type=encoding_type
            )
        
        # Create trainer
        print(f"\nTraining Setup for Fold {fold_idx}")
        print("-" * 80)
        
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
            checkpoint_dir=str(fold_output_dir / 'checkpoints'),
            early_stopping_patience=train_config.get('early_stopping_patience', 20),
            gradient_clip_norm=train_config.get('gradient_clip_norm', 1.0),
            scheduler_type=train_config.get('scheduler_type', 'reduce_on_plateau'),
            scheduler_params=train_config.get('scheduler_params', None),
            num_epochs=train_config.get('num_epochs', 100),
        )
        
        print(f"  Device: {args.device}")
        print(f"  Learning rate: {train_config.get('learning_rate', 1e-4)}")
        print(f"  Batch size: {train_config.get('batch_size', 32)}")
        print(f"  Max epochs: {train_config.get('num_epochs', 100)}")
        
        # Train model
        print(f"\nTraining Fold {fold_idx}")
        print("-" * 80)
        
        history = trainer.train(
            num_epochs=train_config.get('num_epochs', 100),
            verbose=True
        )
        
        # Evaluate on validation set
        print(f"\nEvaluating Fold {fold_idx} on Validation Set")
        print("-" * 80)
        
        # Load best model
        best_checkpoint = fold_output_dir / 'checkpoints' / 'best_checkpoint.pt'
        if best_checkpoint.exists():
            trainer.load_checkpoint(str(best_checkpoint))
            print("Loaded best checkpoint")
        
        # Evaluate and save results
        val_results = evaluate_fold_on_val(
            trainer, val_loader, normalization_stats, fold_output_dir
        )
        
        if val_results[0] is not None:
            metrics_denorm, preds_denorm, targets_denorm, masks, sample_ids, task_names = val_results
            
            # Print metrics
            print("\nValidation Set Results (Denormalized/Original Scale):")
            for key, value in metrics_denorm.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
            
            # Save predictions
            pred_path = fold_output_dir / 'val_predictions_denormalized.tsv'
            save_fold_predictions(
                preds_denorm, targets_denorm, masks, sample_ids, task_names, pred_path
            )
            print(f"\nPredictions saved to: {pred_path}")
            
            # Save metrics
            metrics_path = fold_output_dir / 'val_metrics_denormalized.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics_denorm, f, indent=2)
            print(f"Metrics saved to: {metrics_path}")
            
            # Store fold results
            fold_results.append({
                'fold_idx': fold_idx,
                'val_metrics': metrics_denorm,
                'training_time': time.time() - fold_start_time
            })
        
        fold_end_time = time.time()
        print(f"\nFold {fold_idx} completed in {fold_end_time - fold_start_time:.2f} seconds")
        
        # Clear GPU cache between folds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    time_end_cv = time.time()
    print(f"\n{'='*80}")
    print(f"All {args.n_folds} folds completed in {time_end_cv - time_start_cv:.2f} seconds")
    print(f"{'='*80}")
    
    # Aggregate results across folds
    aggregate_cv_results(fold_results, output_dir)
    
    print("\n" + "=" * 80)
    print("Cross-Validation Training Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Fold results: {output_dir}/fold_*/")
    print(f"  - CV summary: {output_dir}/cv_summary.json")
    print(f"  - CV summary (text): {output_dir}/cv_summary.txt")


if __name__ == '__main__':
    main()

