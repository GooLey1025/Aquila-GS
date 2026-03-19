#!/usr/bin/env python3
"""
Aquila VCF Prediction Script

Make predictions using a trained model with VCF input.
Supports both single-branch and multi-branch architectures.

Usage:
    python aquila_predict.py --model-dir path/to/model_dir \
                            --vcf path/to/test.vcf \
                            --output predictions.tsv

    # Or with direct paths:
    python aquila_predict.py --checkpoint path/to/best_checkpoint.pt \
                            --config path/to/params.yaml \
                            --vcf path/to/test.vcf \
                            --output predictions.tsv
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Dict, Optional, List, Tuple

from aquila.varnn import create_model_from_config
from aquila.encoding import parse_genotype_file
from aquila.utils import load_config
from aquila.metrics import MetricsCalculator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Make predictions with Aquila VCF model'
    )

    # Option 1: Model directory (auto-finds checkpoint, config, and normalization stats)
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Model directory containing checkpoint, params.yaml, and normalization_stats.pkl'
    )

    # Option 2: Direct paths to checkpoint and config
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (best_checkpoint.pt)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file (params.yaml)'
    )

    # Option 3: Output directory (deprecated, kept for backwards compatibility)
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='[Deprecated] Use --model-dir instead'
    )

    # Input data
    parser.add_argument(
        '--vcf',
        type=str,
        default=None,
        help='Path to input VCF file for prediction (default: read from config file)'
    )

    parser.add_argument(
        '--pheno',
        type=str,
        default=None,
        help='Path to phenotype file (optional, for computing metrics if available)'
    )

    # Output options
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='predictions.tsv',
        help='Output file for predictions (default: predictions.tsv)'
    )

    # Model options
    parser.add_argument(
        '--encoding-type',
        type=str,
        default=None,
        choices=['token', 'diploid_onehot', 'snp_vcf', 'indel_vcf', 'sv_vcf', 'snp_indel_vcf', 'snp_indel_sv_vcf'],
        help='Encoding type for VCF parsing (default: read from config file)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for inference (default: 64)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu, default: cuda if available)'
    )

    # Normalization stats (optional, will look in output-dir if not provided)
    parser.add_argument(
        '--normalization-stats',
        type=str,
        default=None,
        help='Path to normalization_stats.pkl file (optional, will look in output-dir if not provided)'
    )

    return parser.parse_args()


def find_file_in_dir(directory: Path, filename_pattern: str) -> Optional[Path]:
    """Find a file in directory matching the pattern."""
    if not directory.exists():
        return None
    
    # Exact match first
    exact_match = directory / filename_pattern
    if exact_match.exists():
        return exact_match
    
    # Try glob pattern
    matches = list(directory.glob(filename_pattern))
    return matches[0] if matches else None


def load_model_and_config(args) -> Tuple[dict, Path, Path]:
    """
    Load model checkpoint and config based on args.

    Returns:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint
        norm_stats_path: Path to normalization stats
        output_dir: Base directory (model_dir or output_dir)
    """
    model_dir = None
    output_dir = None
    checkpoint_path = None
    config_path = None
    norm_stats_path = None

    # Model directory takes priority
    if args.model_dir:
        model_dir = Path(args.model_dir)

        # If model_dir doesn't exist, try finding it relative to current working directory
        if not model_dir.exists():
            cwd_model_dir = Path.cwd() / model_dir
            if cwd_model_dir.exists():
                model_dir = cwd_model_dir
                print(f"\nNote: Resolved model_dir relative to cwd: {model_dir}")

        # Find checkpoint in model_dir root (for when model_dir is a trial directory)
        checkpoint_path = find_file_in_dir(model_dir / 'checkpoints', 'best_checkpoint.pt')
        if not checkpoint_path:
            checkpoint_path = find_file_in_dir(model_dir / 'checkpoints', '*.pt')

        # Find config in model_dir root
        config_path = find_file_in_dir(model_dir, 'params.yaml')

        # Find normalization stats in model_dir root
        norm_stats_path = find_file_in_dir(model_dir, 'normalization_stats.pkl')

        # Also check parent directories (for when model_dir is a trial directory)
        # Check if parent has checkpoints (e.g., model_dir=705rice_conv_mha.../trial_0)
        if not checkpoint_path:
            parent_checkpoint = find_file_in_dir(model_dir.parent / 'checkpoints', 'best_checkpoint.pt')
            if not parent_checkpoint:
                parent_checkpoint = find_file_in_dir(model_dir.parent / 'checkpoints', '*.pt')
            if parent_checkpoint:
                checkpoint_path = parent_checkpoint

        # Also check trial subdirectories (e.g., model_dir=705rice_conv_mha...)
        if not checkpoint_path or not config_path:
            for trial_dir in sorted(model_dir.glob('trial_*')):
                if trial_dir.is_dir():
                    if not checkpoint_path:
                        cp = find_file_in_dir(trial_dir / 'checkpoints', 'best_checkpoint.pt')
                        if cp:
                            checkpoint_path = cp
                    if not config_path:
                        cfg = find_file_in_dir(trial_dir, 'params.yaml')
                        if cfg:
                            config_path = cfg
                    if not norm_stats_path:
                        ns = find_file_in_dir(trial_dir, 'normalization_stats.pkl')
                        if ns:
                            norm_stats_path = ns
                    if checkpoint_path and config_path:
                        break

        output_dir = model_dir

    # Fallback to output-dir (backwards compatibility)
    elif args.output_dir:
        output_dir = Path(args.output_dir)

        # Find checkpoint (try various patterns)
        checkpoint_path = find_file_in_dir(output_dir / 'checkpoints', 'best_checkpoint.pt')
        if not checkpoint_path:
            checkpoint_path = find_file_in_dir(output_dir / 'checkpoints', '*.pt')

        # Find config
        config_path = find_file_in_dir(output_dir, 'params.yaml')

        # Find normalization stats
        norm_stats_path = find_file_in_dir(output_dir, 'normalization_stats.pkl')

    # Override with explicit args if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    if args.config:
        config_path = Path(args.config)
    if args.normalization_stats:
        norm_stats_path = Path(args.normalization_stats)

    # Validation
    if not checkpoint_path:
        raise ValueError("Checkpoint not found. Provide --model-dir (with checkpoint in checkpoints/ or trial_*/checkpoints/), "
                        "--output-dir with checkpoint, or --checkpoint.")
    if not config_path:
        raise ValueError("Config not found. Provide --model-dir (with params.yaml in root or trial_*/), "
                        "--output-dir with params.yaml, or --config.")

    # Load config
    config = load_config(str(config_path))

    return config, checkpoint_path, norm_stats_path, output_dir


def load_vcf_data(vcf_path: str, encoding_type: str, config: dict) -> Tuple[dict, List[str], Dict]:
    """
    Load and encode VCF data.
    
    Returns:
        variant_data: Dict of encoded variant tensors (for multi-branch) or single tensor
        sample_ids: List of sample IDs
        seq_length: int or dict of sequence lengths
    """
    # Determine variant type from config
    variant_type = config.get('data', {}).get('variant_type')
    
    # Check if multi-branch
    is_multi_branch = encoding_type in ['snp_indel_vcf', 'snp_indel_sv_vcf'] or variant_type in ['snp_indel', 'snp_indel_sv']
    
    print(f"\nLoading VCF data: {vcf_path}")
    print(f"  Encoding type: {encoding_type}")
    print(f"  Variant type: {variant_type}")
    print(f"  Multi-branch: {is_multi_branch}")
    
    # Parse genotype with encoding
    if is_multi_branch:
        # Multi-branch: returns dict
        variant_data = parse_genotype_file(vcf_path, encoding_type, variant_type)
        
        # Extract sample IDs from first variant type
        first_variant_type = list(variant_data.keys())[0]
        sample_ids = variant_data[first_variant_type][1]
        
        # seq_length as dict
        seq_length = {vtype: data[0].shape[1] for vtype, data in variant_data.items()}
        
        # Convert to tensor dict
        variant_tensors = {vtype: torch.from_numpy(data[0]).long() 
                         for vtype, data in variant_data.items()}
    else:
        # Single-branch
        result = parse_genotype_file(vcf_path, encoding_type, variant_type)
        snp_matrix, sample_ids, _ = result
        
        # Convert to tensor
        variant_tensors = torch.from_numpy(snp_matrix).long()
        seq_length = snp_matrix.shape[1]
    
    print(f"  Loaded {len(sample_ids)} samples")
    if is_multi_branch:
        print(f"  Sequence lengths: {seq_length}")
    else:
        print(f"  Sequence length: {seq_length}")
    
    return variant_tensors, sample_ids, seq_length


def load_normalization_stats(norm_stats_path: Optional[Path]) -> Optional[Dict]:
    """Load normalization statistics for denormalization."""
    if norm_stats_path is None or not norm_stats_path.exists():
        print("\nWarning: Normalization stats not found. Predictions will be in normalized scale.")
        return None
    
    print(f"\nLoading normalization stats: {norm_stats_path}")
    with open(norm_stats_path, 'rb') as f:
        norm_stats = pickle.load(f)
    
    # Print normalization info
    regression_tasks = norm_stats.get('regression_tasks', [])
    log_transformed = norm_stats.get('log_transformed_tasks', [])
    
    print(f"  Regression tasks: {regression_tasks}")
    if log_transformed:
        print(f"  Log-transformed tasks: {log_transformed}")
    
    return norm_stats


def create_model(config: dict, seq_length, device: str, norm_stats: Optional[Dict] = None):
    """Create model from config."""
    # Get task names from config or norm_stats
    model_config = config.get('model', {})
    train_config = config.get('train', {})
    data_config = config.get('data', {})
    
    # Try to get task names from various sources (in order of priority)
    regression_tasks = model_config.get('regression_tasks', None)
    classification_tasks = model_config.get('classification_tasks', None)
    
    # If not in model config, check train config
    if regression_tasks is None:
        regression_tasks = train_config.get('regression_tasks', None)
    if classification_tasks is None:
        classification_tasks = train_config.get('classification_tasks', None)
    
    # Check data config
    if classification_tasks is None:
        classification_tasks = data_config.get('classification_tasks', None)
    
    # If still None, try to get from norm_stats
    if regression_tasks is None and norm_stats is not None:
        regression_tasks = norm_stats.get('regression_tasks', None)
    if classification_tasks is None and norm_stats is not None:
        classification_tasks = norm_stats.get('classification_tasks', None)
    
    print("\nCreating model...")
    print(f"  Regression tasks: {regression_tasks}")
    print(f"  Classification tasks: {classification_tasks}")
    
    model = create_model_from_config(
        config=config,
        seq_length=seq_length,
        regression_tasks=regression_tasks,
        classification_tasks=classification_tasks
    )
    
    model = model.to(device)
    model.eval()
    
    return model


def load_checkpoint(model, checkpoint_path: Path, device: str, is_distributed: bool = False):
    """Load model checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle DDP wrapper
    state_dict = checkpoint['model_state_dict']
    if is_distributed:
        model.module.load_state_dict(state_dict)
    else:
        # Use strict=False to allow loading checkpoints with slightly different architecture
        # (e.g., extra branch_projections layers that may exist in training but not in inference)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  Warning: Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"  Warning: Unexpected keys in state_dict (ignored): {unexpected_keys}")
    
    # Print epoch info
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    
    print("  Model loaded successfully")
    return model


def predict(model, variant_data, sample_ids, device: str, batch_size: int, is_multi_branch: bool):
    """Run prediction on VCF data."""
    print("\nRunning inference...")
    
    all_predictions = {
        'regression': [],
        'classification': [],
    }
    
    num_samples = len(sample_ids)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_idx = i // batch_size
            if batch_idx % 50 == 0:
                print(f"  Processing batch {batch_idx + 1}/{num_batches}")
            
            # Get batch
            if is_multi_branch:
                batch = {k: v[i:i+batch_size].to(device) for k, v in variant_data.items()}
            else:
                batch = variant_data[i:i+batch_size].to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Collect predictions
            if 'regression' in outputs:
                all_predictions['regression'].append(outputs['regression'].cpu().numpy())
            if 'classification' in outputs:
                # Convert logits to probabilities
                probs = torch.sigmoid(outputs['classification'])
                all_predictions['classification'].append(probs.cpu().numpy())
    
    # Concatenate all batches
    predictions = {}
    
    if all_predictions['regression']:
        predictions['regression'] = np.concatenate(all_predictions['regression'], axis=0)
    
    if all_predictions['classification']:
        predictions['classification'] = np.concatenate(all_predictions['classification'], axis=0)
    
    print(f"  Inference complete. Shape: {predictions.get('regression', predictions.get('classification')).shape}")
    
    return predictions


def denormalize_predictions(predictions: np.ndarray, norm_stats: Dict, 
                             regression_tasks: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Denormalize predictions using normalization statistics.
    
    Returns:
        predictions_log_scale: Predictions in log scale (after reverse Z-score)
        predictions_original: Predictions in original scale (after reverse log transform)
    """
    regression_means = norm_stats['regression_means']
    regression_stds = norm_stats['regression_stds']
    log_transformed_tasks = norm_stats.get('log_transformed_tasks', [])
    
    # Step 1: Reverse Z-score normalization (normalized -> log scale)
    # log_scale = (normalized * std) + mean
    predictions_log_scale = predictions.copy()
    for i in range(len(regression_means)):
        predictions_log_scale[:, i] = (
            predictions[:, i] * regression_stds[i]) + regression_means[i]
    
    # Step 2: Reverse log(x+1) transformation if applied
    predictions_original = predictions_log_scale.copy()
    for i, task_name in enumerate(regression_tasks):
        if task_name in log_transformed_tasks:
            # Reverse log(x+1): original = exp(log_value) - 1
            predictions_original[:, i] = np.expm1(predictions_log_scale[:, i])
    
    return predictions_log_scale, predictions_original


def save_predictions(sample_ids: List[str], predictions_dict: Dict, predictions_original: np.ndarray, predictions_log_scale: np.ndarray,
                   norm_stats: Optional[Dict], output_path: Path, is_classification: bool = False,
                   classification_tasks: List[str] = None):
    """Save predictions to file."""
    print(f"\nSaving predictions to: {output_path}")
    
    # Get task names
    if norm_stats:
        regression_tasks = norm_stats.get('regression_tasks', [])
    else:
        regression_tasks = [f"regression_task_{i}" for i in range(predictions_original.shape[1])]
    
    # Build results DataFrame
    results = {'Sample_ID': sample_ids}
    
    # Add regression predictions
    if 'regression' in predictions_dict:
        preds = predictions_original if predictions_original is not None else predictions_dict['regression']
        
        for i, task_name in enumerate(regression_tasks):
            results[f'{task_name}_Pred'] = preds[:, i]
    
    # Add classification predictions
    if is_classification and classification_tasks:
        cls_preds = predictions_dict['classification']
        for i, task_name in enumerate(classification_tasks):
            results[f'{task_name}_Prob'] = cls_preds[:, i]
            results[f'{task_name}_Pred'] = (cls_preds[:, i] >= 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if str(output_path).endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, sep='\t', index=False, float_format='%.6f')
    
    print(f"  Saved {len(df)} predictions")
    print("\nPreview:")
    print(df.head(10))
    
    return df


def main():
    """Main prediction function."""
    args = parse_args()
    
    print("=" * 80)
    print("AQUILA: VCF Prediction")
    print("=" * 80)
    
    # Load model, config, and normalization stats
    config, checkpoint_path, norm_stats_path, output_dir = load_model_and_config(args)
    
    # Load normalization stats
    norm_stats = load_normalization_stats(norm_stats_path)
    
    # Determine VCF file path (command line args override config)
    vcf_path = args.vcf
    if vcf_path is None:
        # Read from config file
        vcf_path = config.get('data', {}).get('geno_file')
        if vcf_path:
            print(f"\nUsing VCF from config: {vcf_path}")
        else:
            raise ValueError("No VCF file specified. Provide --vcf or ensure geno_file is in config.")
    
    # Determine encoding type (command line args override config)
    encoding_type = args.encoding_type
    if encoding_type is None:
        # Read from config file
        encoding_type = config.get('data', {}).get('encoding_type', 'diploid_onehot')
    
    # Check if multi-branch
    variant_type = config.get('data', {}).get('variant_type')
    is_multi_branch = encoding_type in ['snp_indel_vcf', 'snp_indel_sv_vcf'] or variant_type in ['snp_indel', 'snp_indel_sv']
    
    # Load VCF data
    variant_data, sample_ids, seq_length = load_vcf_data(
        vcf_path, encoding_type, config
    )
    
    # Create model
    device = args.device
    model = create_model(config, seq_length, device, norm_stats)
    
    # Initialize model with a dummy forward pass to build dynamic layers (e.g., branch_projections)
    # This ensures the model structure matches the checkpoint exactly
    print("\nInitializing model with dummy forward pass...")
    model.eval()
    with torch.no_grad():
        if is_multi_branch:
            # Multi-branch: create dummy inputs for each branch type
            dummy_input = {}
            for vtype, length in seq_length.items():
                dummy_input[vtype] = torch.randn(1, length, 8, device=device)
        else:
            # Single branch: seq_length is an integer
            length = seq_length if isinstance(seq_length, int) else seq_length.get('snp', seq_length)
            dummy_input = torch.randn(1, length, 8, device=device)
        
        # Run dummy forward pass to initialize dynamic layers
        _ = model(dummy_input)
    print("  Model initialized.")
    
    # Load checkpoint
    model = load_checkpoint(model, checkpoint_path, device)
    
    # Run prediction
    predictions = predict(model, variant_data, sample_ids, device, args.batch_size, is_multi_branch)
    
    # Get task info
    is_classification = 'classification' in predictions
    classification_tasks = config.get('model', {}).get('classification_tasks', 
                      config.get('data', {}).get('classification_tasks', None))
    
    # Denormalize if normalization stats available
    predictions_log_scale = None
    predictions_original = None
    
    if norm_stats and 'regression' in predictions:
        predictions_log_scale, predictions_original = denormalize_predictions(
            predictions['regression'], 
            norm_stats,
            norm_stats.get('regression_tasks', [])
        )
        print("\nPredictions denormalized to original scale")
    else:
        predictions_original = predictions.get('regression', predictions.get('classification'))
    
    # Save predictions
    output_path = Path(args.output)
    save_predictions(
        sample_ids=sample_ids,
        predictions_dict=predictions,
        predictions_original=predictions_original,
        predictions_log_scale=predictions_log_scale,
        norm_stats=norm_stats,
        output_path=output_path,
        is_classification=is_classification,
        classification_tasks=classification_tasks
    )
    
    # Compute metrics if phenotype file provided
    if args.pheno and norm_stats:
        print("\n" + "=" * 80)
        print("Computing Metrics")
        print("=" * 80)
        
        # Load true labels
        from aquila.data_utils import parse_phenotype_file
        pheno_df, regression_cols, classification_cols = parse_phenotype_file(
            args.pheno, 
            classification_tasks=config.get('data', {}).get('classification_tasks')
        )
        
        # Match samples
        pheno_df = pheno_df.set_index('Sample_ID')
        common_samples = [s for s in sample_ids if s in pheno_df.index]
        
        if len(common_samples) < len(sample_ids):
            print(f"Warning: Only {len(common_samples)}/{len(sample_ids)} samples have labels")
        
        # Get indices for common samples
        sample_idx = [sample_ids.index(s) for s in common_samples]
        
        # Get predictions for common samples
        pred_vals = predictions_original[sample_idx] if predictions_original is not None else predictions['regression'][sample_idx]
        
        # Get true values (need to apply same normalization)
        true_vals = pheno_df.loc[common_samples, regression_cols].values
        
        # Apply Z-score normalization (same as training)
        true_normalized = true_vals.copy()
        for i, col in enumerate(regression_cols):
            mean = norm_stats['regression_means'][i]
            std = norm_stats['regression_stds'][i]
            true_normalized[:, i] = (true_vals[:, i] - mean) / std
        
        # Compute metrics on normalized scale
        metrics_calc = MetricsCalculator()
        
        # Create masks (assume all valid)
        masks = np.ones_like(pred_vals)
        
        metrics_norm = metrics_calc.compute_regression_metrics(pred_vals, true_normalized, masks)
        
        # Compute metrics on original scale
        metrics_orig = metrics_calc.compute_regression_metrics(pred_vals, true_vals, masks)
        
        print("\nMetrics (Normalized Scale):")
        for key, value in metrics_norm.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        print("\nMetrics (Original Scale):")
        for key, value in metrics_orig.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        # Save metrics
        metrics_path = output_path.parent / 'prediction_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'normalized': metrics_norm,
                'original': metrics_orig
            }, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
    
    print("\n" + "=" * 80)
    print("Prediction Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
