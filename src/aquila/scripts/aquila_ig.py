#!/usr/bin/env python3
"""
Aquila Integrated Gradients Script

Compute integrated gradients for genomic variant attribution using trained models.
Supports multi-branch architectures for SNP/INDEL/SV variants.

Usage:
    python aquila_ig.py --config params.yaml --checkpoint checkpoints/best_checkpoint.pt --vcf input.vcf.gz --output output_dir

"""

import argparse
import torch
import numpy as np
from pathlib import Path
import os
import json
import pandas as pd
import h5py
from typing import Dict, List, Optional, Tuple

from aquila.varnn import create_model_from_config
from aquila.utils import load_config
from aquila.encoding import parse_genotype_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute integrated gradients for Aquila model'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file (same as used for training)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--vcf',
        type=str,
        default=None,
        help='Path to VCF file with genotype data (overrides config)'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='ig_output',
        help='Output directory for integrated gradients results'
    )

    parser.add_argument(
        '--encoding-type',
        type=str,
        default='diploid_onehot',
        choices=['token', 'diploid_onehot', 'snp_vcf', 'indel_vcf', 'sv_vcf', 'snp_indel_vcf', 'snp_indel_sv_vcf'],
        help='Encoding type (must match training)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for computation'
    )

    parser.add_argument(
        '--num-steps',
        type=int,
        default=50,
        help='Number of interpolation steps for integrated gradients (default: 50)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1, recommended for IG)'
    )

    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help='Task name to compute IG for (regression task). If not specified, compute for all tasks.'
    )

    parser.add_argument(
        '--samples-set',
        type=str,
        default=None,
        help='Path to txt file with sample IDs (one per line). If not specified, compute for all samples.'
    )

    return parser.parse_args()


def load_model_and_data(args):
    """
    Load model from checkpoint and prepare data.

    Returns:
        model: Loaded model in eval mode
        data_dict: Dictionary with genotype data for each variant type
        sample_ids: List of sample IDs (filtered if samples_set is provided)
        snp_ids: Dictionary with variant IDs for each type
        regression_task_names: List of regression task names
    """
    print("=" * 80)
    print("AQUILA: Integrated Gradients")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Override encoding type
    encoding_type = args.encoding_type
    config['data']['encoding_type'] = encoding_type

    print(f"Encoding type: {encoding_type}")

    # Get variant type from config
    variant_type = config.get('data', {}).get('variant_type')
    print(f"Variant type: {variant_type}")

    # Determine if multi-branch
    is_multi_branch = encoding_type in ['snp_indel_vcf', 'snp_indel_sv_vcf'] or variant_type in ['snp_indel', 'snp_indel_sv']

    # Load genotype data
    print(f"\nLoading genotype data from: {args.vcf}")

    # For multi-branch, we need to load SNP, INDEL, SV separately
    # The VCF file should contain all variant types
    # We'll use the encoding function which handles the splitting

    # First, load full genotype data to understand structure
    from aquila.data_utils import create_data_loaders

    # Create a dummy data loader to get the processed data
    # We only need to load the data, not create full loaders
    # So we'll directly use the encoding function

    # Load genotype data based on encoding
    if encoding_type == 'diploid_onehot':
        # For diploid_onehot, we need to parse differently based on variant_type
        if is_multi_branch:
            # Multi-branch: load separate files or single file with variant type annotation
            data_config = config.get('data', {})

            # Use command line VCF if provided, otherwise read from config
            if args.vcf:
                geno_file = args.vcf
            else:
                geno_file = data_config.get('geno_file')
                if geno_file:
                    print(f"Using VCF from config: {geno_file}")
                else:
                    raise ValueError("No VCF file specified. Please provide --vcf or set geno_file in config.")

            # Load multi-branch data using parse_genotype_file (supports .vcf.gz)
            print(f"Loading multi-branch genotype data from: {geno_file}")
            result = parse_genotype_file(geno_file, encoding_type='diploid_onehot', variant_type=variant_type)

            # result is a dict: {'snp': (matrix, samples, ids), 'indel': ..., 'sv': ...}
            data_dict = {}
            sample_ids = []
            snp_ids = {}

            if 'snp' in result:
                snp_matrix, sample_ids, snp_id_list = result['snp']
                data_dict['snp'] = snp_matrix
                snp_ids['snp'] = snp_id_list
                print(f"  Loaded {len(snp_id_list)} SNPs for {len(sample_ids)} samples")

            if 'indel' in result:
                indel_matrix, _, indel_id_list = result['indel']
                data_dict['indel'] = indel_matrix
                snp_ids['indel'] = indel_id_list
                print(f"  Loaded {len(indel_id_list)} INDELs")

            if 'sv' in result:
                sv_matrix, _, sv_id_list = result['sv']
                data_dict['sv'] = sv_matrix
                snp_ids['sv'] = sv_id_list
                print(f"  Loaded {len(sv_id_list)} SVs")
        else:
            # Single branch
            data_config = config.get('data', {})

            # Use command line VCF if provided, otherwise read from config
            if args.vcf:
                vcf_file = args.vcf
            else:
                vcf_file = data_config.get('geno_file')
                if vcf_file:
                    print(f"Using VCF from config: {vcf_file}")
                else:
                    raise ValueError("No VCF file specified. Please provide --vcf or set geno_file in config.")

            print("Loading genotype data...")
            snp_matrix, sample_ids, snp_id_list = parse_genotype_file(vcf_file, encoding_type='diploid_onehot')
            data_dict = {'snp': snp_matrix}
            snp_ids = {'snp': snp_id_list}
            print(f"Loaded {len(snp_id_list)} variants for {len(sample_ids)} samples")
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")

    # Filter samples if samples_set is provided
    if args.samples_set:
        print(f"\nFiltering samples using: {args.samples_set}")
        with open(args.samples_set, 'r') as f:
            target_samples = set(line.strip() for line in f if line.strip())
        
        # Get indices of target samples
        sample_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}
        target_indices = [sample_to_idx[sid] for sid in target_samples if sid in sample_to_idx]
        
        if not target_indices:
            raise ValueError("No matching samples found in VCF!")
        
        print(f"  Found {len(target_indices)} matching samples out of {len(target_samples)} requested")
        
        # Filter data
        filtered_sample_ids = [sample_ids[i] for i in target_indices]
        filtered_data_dict = {}
        for vtype, data in data_dict.items():
            filtered_data_dict[vtype] = data[target_indices]
        
        data_dict = filtered_data_dict
        sample_ids = filtered_sample_ids
        print(f"  Filtered to {len(sample_ids)} samples")

    print(f"\nLoaded data shapes:")
    for vtype, data in data_dict.items():
        print(f"  {vtype}: {data.shape}")

    # Get sequence lengths
    seq_length = {}
    for vtype, data in data_dict.items():
        seq_length[vtype] = data.shape[1]

    print(f"\nSequence lengths: {seq_length}")

    # Load normalization stats if available (for task names)
    output_dir = Path(config.get('output_dir', './outputs'))
    norm_stats_path = output_dir / 'normalization_stats.pkl'

    regression_task_names = []
    if norm_stats_path.exists():
        import pickle
        with open(norm_stats_path, 'rb') as f:
            norm_stats = pickle.load(f)
        regression_task_names = norm_stats.get('regression_tasks', [])
        print(f"\nRegression tasks: {regression_task_names}")

    # Load checkpoint first to get model dimensions
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Extract num_targets from checkpoint if available
    checkpoint_num_targets = None
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Try to find regression head weights to determine num_targets
    for key in state_dict.keys():
        if 'head_blocks.regression' in key and '.network.4.weight' in key:
            checkpoint_num_targets = state_dict[key].shape[0]
            print(f"Detected {checkpoint_num_targets} regression targets from checkpoint")
            break

    # Update config if checkpoint has different number of targets
    if checkpoint_num_targets and checkpoint_num_targets != len(regression_task_names):
        print(f"Updating regression tasks from {len(regression_task_names)} to {checkpoint_num_targets}")
        # Create dummy task names if not available
        if not regression_task_names:
            regression_task_names = [f"task_{i}" for i in range(checkpoint_num_targets)]
        # Update config for model creation
        model_config = config.get('model', {})
        if 'heads' in model_config and 'regression' in model_config['heads']:
            for block in model_config['heads']['regression']:
                if isinstance(block, dict) and block.get('name') == 'regression_head':
                    block['num_targets'] = checkpoint_num_targets

    # Determine which task(s) to compute IG for
    if args.task:
        task_indices = [regression_task_names.index(args.task)] if args.task in regression_task_names else [0]
    elif regression_task_names:
        task_indices = list(range(len(regression_task_names)))
    else:
        task_indices = [0]
        regression_task_names = ['regression_task_0']

    num_tasks = len(task_indices)
    print(f"\nComputing IG for {num_tasks} task(s): {[regression_task_names[i] for i in task_indices]}")

    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    model = create_model_from_config(
        config=config,
        seq_length=seq_length,
        regression_tasks=regression_task_names,
        classification_tasks=None
    )

    # Handle both DDP and non-DDP checkpoints
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DDP)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model = model.to(args.device)
    model.eval()

    print("Model loaded successfully")

    return model, data_dict, sample_ids, snp_ids, task_indices, regression_task_names, is_multi_branch


class IntegratedGradients:
    """
    Compute integrated gradients for model attribution.

    Integrated Gradients = (input - baseline) * integral(gradient at interpolated points)

    For diploid_onehot encoding, baseline is all zeros (missing genotype).
    """

    def __init__(self, model, device, num_steps=50, is_multi_branch=False):
        """
        Initialize IntegratedGradients.

        Args:
            model: PyTorch model
            device: Device to use
            num_steps: Number of interpolation steps
            is_multi_branch: Whether model is multi-branch architecture
        """
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.is_multi_branch = is_multi_branch

    def compute_ig_single_sample_single_task(self, input_dict: Dict[str, torch.Tensor],
                                              task_idx: int = 0,
                                              target_type: str = 'regression') -> Dict[str, np.ndarray]:
        """
        Compute integrated gradients for a single sample and single task.

        Args:
            input_dict: Dictionary with input tensors for each variant type
                        Each tensor shape: (1, seq_len, 8)
            task_idx: Task index to compute gradient for
            target_type: 'regression' or 'classification'

        Returns:
            Dictionary with integrated gradients for each variant type
            Each value shape: (1, seq_len, 8)
        """
        self.model.eval()

        # Create baseline (all zeros - missing genotype)
        baseline_dict = {}
        for vtype, input_tensor in input_dict.items():
            baseline_dict[vtype] = torch.zeros_like(input_tensor)

        # Compute integrated gradients using Riemann sum approximation
        # alpha from 0 to 1
        alphas = torch.linspace(0, 1, self.num_steps, device=self.device)

        # Store gradients at each step
        accumulated_grads = {}
        for vtype in input_dict.keys():
            accumulated_grads[vtype] = torch.zeros_like(input_dict[vtype], device=self.device)

        # Compute gradients at each interpolated point
        for alpha in alphas:
            # Create interpolated input
            interpolated_dict = {}
            for vtype, input_tensor in input_dict.items():
                interpolated_input = baseline_dict[vtype] + alpha * (input_tensor - baseline_dict[vtype])
                interpolated_input.requires_grad = True
                interpolated_dict[vtype] = interpolated_input

            # Forward pass
            # For multi-branch models, pass dict; for single-branch, pass first tensor
            if self.is_multi_branch:
                outputs = self.model(interpolated_dict)
            else:
                # Single-branch model: get the first (and only) variant type
                first_vtype = list(interpolated_dict.keys())[0]
                outputs = self.model(interpolated_dict[first_vtype])

            # Get target value
            if target_type == 'regression':
                target = outputs['regression'][0, task_idx]
            else:
                target = outputs['classification'][0, task_idx]

            # Backward pass
            self.model.zero_grad()
            target.backward()

            # Accumulate gradients (weighted by alpha step)
            for vtype in input_dict.keys():
                if interpolated_dict[vtype].grad is not None:
                    # Use trapezoidal rule for better approximation
                    if alpha == alphas[0] or alpha == alphas[-1]:
                        weight = 0.5
                    else:
                        weight = 1.0

                    # Keep on device for accumulation, convert to CPU at the end
                    accumulated_grads[vtype] += weight * interpolated_dict[vtype].grad

        # Multiply by (input - baseline)
        final_ig = {}
        for vtype in input_dict.keys():
            # Integrated gradients = sum(gradients) * (input - baseline) / num_steps
            # Actually for trapezoidal: sum(gradients) * step_size where step_size = 1/num_steps
            step_size = 1.0 / self.num_steps
            final_ig[vtype] = accumulated_grads[vtype].cpu().numpy() * step_size

            # Also multiply by (input - baseline) for full IG formula
            # But since baseline=0, this is just the input
            # Actually the formula is: IG = (input - baseline) * integral
            # Since baseline=0: IG = input * integral
            final_ig[vtype] = final_ig[vtype] * input_dict[vtype].cpu().numpy()

        return final_ig


def compute_ig_for_all_samples_all_tasks(model, data_dict, sample_ids, snp_ids,
                                         task_indices, regression_task_names,
                                         is_multi_branch, args):
    """
    Compute integrated gradients for all samples and all tasks.

    Returns:
        Dictionary with IG results for each sample and task
    """
    print("\n" + "=" * 80)
    print("Computing Integrated Gradients")
    print("=" * 80)

    ig_computer = IntegratedGradients(model, args.device, args.num_steps, is_multi_branch)

    num_samples = len(sample_ids)
    num_tasks = len(task_indices)
    
    print(f"Processing {num_samples} samples x {num_tasks} tasks = {num_samples * num_tasks} computations...")

    # Store results: results[sample_idx][task_idx] = {vtype: ig_scores}
    # ig_scores shape: (1, seq_len, 8)
    all_results = {}

    for sample_idx in range(num_samples):
        if sample_idx % 10 == 0:
            print(f"  Processing sample {sample_idx + 1}/{num_samples}...")

        sample_id = sample_ids[sample_idx]

        # Prepare input for this sample
        input_dict = {}
        for vtype, data in data_dict.items():
            # Get sample data: shape (seq_len, 8) -> add batch dim -> (1, seq_len, 8)
            sample_data = data[sample_idx]
            input_tensor = torch.from_numpy(sample_data).unsqueeze(0).to(args.device)
            input_dict[vtype] = input_tensor

        # Compute IG for each task
        sample_results = {}
        for task_idx in task_indices:
            ig_scores = ig_computer.compute_ig_single_sample_single_task(
                input_dict,
                task_idx=task_idx,
                target_type='regression'
            )
            task_name = regression_task_names[task_idx]
            sample_results[task_name] = ig_scores

        all_results[sample_id] = sample_results

    return all_results


def save_results_h5(results, sample_ids, snp_ids, regression_task_names, args, output_dir):
    """
    Save integrated gradients results to HDF5 file.
    
    Saves 8-dimensional IG scores (not aggregated).
    """
    print("\n" + "=" * 80)
    print("Saving Results to HDF5")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = output_dir / 'ig_results.h5'

    # Determine total number of variants per vtype
    # Assuming all tasks have the same variant structure
    first_sample = list(results.keys())[0]
    first_task = list(results[first_sample].keys())[0]
    
    with h5py.File(h5_path, 'w') as f:
        # Create groups for metadata
        f.attrs['num_samples'] = len(sample_ids)
        f.attrs['num_tasks'] = len(regression_task_names)
        f.attrs['num_steps'] = args.num_steps
        
        # Save sample IDs
        f.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
        
        # Save task names
        task_names_bytes = [t.encode('utf-8') for t in regression_task_names]
        f.create_dataset('task_names', data=np.array(task_names_bytes, dtype='S'))
        
        # Save variant IDs for each vtype
        for vtype, variant_ids in snp_ids.items():
            variant_ids_bytes = [v.encode('utf-8') for v in variant_ids]
            f.create_dataset(f'variant_ids/{vtype}', data=np.array(variant_ids_bytes, dtype='S'))
        
        # Save IG scores for each sample and task
        print("\nSaving IG scores...")
        for sample_idx, sample_id in enumerate(sample_ids):
            if sample_idx % 50 == 0:
                print(f"  Saving sample {sample_idx + 1}/{len(sample_ids)}...")
            
            sample_results = results[sample_id]
            
            for task_name in regression_task_names:
                ig_scores = sample_results[task_name]
                
                # Create group path: /sample_{idx}/task_{name}
                sample_group = f'sample_{sample_idx}'
                task_group = f'{sample_group}/{task_name}'
                
                # Save each variant type
                for vtype, scores in ig_scores.items():
                    # scores shape: (1, seq_len, 8) -> (seq_len, 8)
                    scores_2d = scores[0]
                    f.create_dataset(f'{task_group}/{vtype}', data=scores_2d)

    print(f"  Saved: {h5_path}")

    # Also save a summary JSON
    summary = {
        'num_samples': len(sample_ids),
        'sample_ids': sample_ids,
        'num_tasks': len(regression_task_names),
        'task_names': regression_task_names,
        'variant_types': list(snp_ids.keys()),
        'num_variants': {vtype: len(ids) for vtype, ids in snp_ids.items()},
        'num_steps': args.num_steps,
    }

    summary_output = output_dir / 'ig_summary.json'
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_output}")

    print(f"\nAll results saved to: {output_dir}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Load model and data
    model, data_dict, sample_ids, snp_ids, task_indices, regression_task_names, is_multi_branch = load_model_and_data(args)

    # Compute IG for all samples and all tasks
    results = compute_ig_for_all_samples_all_tasks(
        model, data_dict, sample_ids, snp_ids,
        task_indices, regression_task_names,
        is_multi_branch, args
    )

    # Save results to HDF5
    save_results_h5(results, sample_ids, snp_ids, regression_task_names, args, args.output)

    print("\n" + "=" * 80)
    print("Integrated Gradients Computation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
