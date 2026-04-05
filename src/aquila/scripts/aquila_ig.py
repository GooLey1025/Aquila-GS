#!/usr/bin/env python3
"""
Aquila Integrated Gradients Script

Compute integrated gradients for genomic variant attribution using trained models.
Supports multi-branch architectures for SNP/INDEL/SV variants.

Usage:
    # Method 1: Provide config and checkpoint explicitly
    python aquila_ig.py --config params.yaml --checkpoint checkpoints/best_checkpoint.pt --vcf input.vcf.gz --output output_dir
    
    # Method 2: Provide model directory (auto-detect config and checkpoint)
    python aquila_ig.py --model-dir /path/to/model_dir --vcf input.vcf.gz --output output_dir

"""

import argparse
import torch
import numpy as np
from pathlib import Path
import os
import json
import pandas as pd
import h5py
import pickle
from typing import Dict, List, Optional, Tuple, Union
import shutil

from aquila.varnn import create_model_from_config
from aquila.utils import load_config
from aquila.encoding import parse_genotype_file


import re

def sanitize_hdf5_key(key: str) -> str:
    """Sanitize a string to be a valid HDF5 group/dataset name.

    HDF5 names cannot contain: / \\ : * ? " < > #
    Also cannot start with . or be empty.
    Replace all invalid chars with underscore.
    """
    if not key:
        return "_"
    # Replace any invalid HDF5 characters with underscore
    key = re.sub(r'[/\\:*?"<>#,]', '_', key)
    # Replace leading/trailing spaces, dots
    key = key.strip().lstrip('.')
    if not key:
        return "_"
    return key


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute integrated gradients for Aquila model'
    )

    # Add mutually exclusive group for --config/--checkpoint vs --model-dir
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file (same as used for training)'
    )
    config_group.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    config_group.add_argument(
        '--model-dir',
        type=str,
        help='Path to model directory (auto-detect params.yaml and checkpoints)'
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
        choices=['token', 'diploid_onehot', 'onehot', 'snp_vcf', 'indel_vcf', 'sv_vcf', 'snp_indel_vcf', 'snp_indel_sv_vcf'],
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
        help='Task name to compute IG for. Can be regression or classification task. If not specified, compute for all tasks.'
    )

    parser.add_argument(
        '--task-type',
        type=str,
        default=None,
        choices=['regression', 'classification', None],
        help='Type of task to compute IG for. If not specified, auto-detect from checkpoint.'
    )

    parser.add_argument(
        '--samples-set',
        type=str,
        default=None,
        help='Path to txt file with sample IDs (one per line). If not specified, compute for all samples.'
    )

    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Enable streaming mode: compute and save results incrementally to reduce memory usage'
    )

    return parser.parse_args()


def resolve_model_dir(args):
    """
    Resolve config and checkpoint paths from model directory.
    
    Returns:
        Tuple of (config_path, checkpoint_path)
    """
    model_dir = Path(args.model_dir)
    
    # Find params.yaml
    config_path = model_dir / 'params.yaml'
    if not config_path.exists():
        # Try other possible names
        for name in ['config.yaml', 'training_config.yaml']:
            alt_path = model_dir / name
            if alt_path.exists():
                config_path = alt_path
                break
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file in {model_dir}")
    
    # Find checkpoint
    checkpoint_dir = model_dir / 'checkpoints'
    checkpoint_path = None
    
    # Prefer best_checkpoint.pt
    if checkpoint_dir.exists():
        best_ckpt = checkpoint_dir / 'best_checkpoint.pt'
        if best_ckpt.exists():
            checkpoint_path = best_ckpt
        else:
            # Try latest_checkpoint.pt
            latest_ckpt = checkpoint_dir / 'latest_checkpoint.pt'
            if latest_ckpt.exists():
                checkpoint_path = latest_ckpt
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"Could not find checkpoint in {model_dir}/checkpoints")
    
    return str(config_path), str(checkpoint_path)


def load_task_names_from_model_dir(model_dir: Path) -> Tuple[List[str], List[str], Dict]:
    """
    Load task names and config from model directory.
    
    Returns:
        Tuple of (regression_tasks, classification_tasks, extra_info)
    """
    regression_tasks = []
    classification_tasks = []
    extra_info = {}
    
    # Try normalization_stats.pkl
    norm_stats_path = model_dir / 'normalization_stats.pkl'
    if norm_stats_path.exists():
        with open(norm_stats_path, 'rb') as f:
            norm_stats = pickle.load(f)
        regression_tasks = norm_stats.get('regression_tasks', [])
        extra_info['regression_means'] = norm_stats.get('regression_means', {})
        extra_info['regression_stds'] = norm_stats.get('regression_stds', {})
    
    # Try task_mapping.tsv
    task_mapping_path = model_dir / 'task_mapping.tsv'
    if task_mapping_path.exists():
        df = pd.read_csv(task_mapping_path, sep='\t')
        if 'task_name' in df.columns and 'task_type' in df.columns:
            for _, row in df.iterrows():
                if row['task_type'] == 'regression' and row['task_name'] not in regression_tasks:
                    regression_tasks.append(row['task_name'])
                elif row['task_type'] == 'classification' and row['task_name'] not in classification_tasks:
                    classification_tasks.append(row['task_name'])
    
    # Try data_config.json
    data_config_path = model_dir / 'data_cache' / 'data_config.json'
    if data_config_path.exists():
        with open(data_config_path, 'r') as f:
            data_config = json.load(f)
        if 'regression_tasks' in data_config and not regression_tasks:
            regression_tasks = data_config['regression_tasks']
        if 'classification_tasks' in data_config and not classification_tasks:
            classification_tasks = data_config['classification_tasks']
    
    return regression_tasks, classification_tasks, extra_info


def detect_num_targets_from_checkpoint(state_dict: Dict, task_type: str = 'regression') -> Optional[int]:
    """
    Detect number of targets from checkpoint state dict.
    
    Args:
        state_dict: Model state dictionary
        task_type: 'regression' or 'classification'
    
    Returns:
        Number of targets, or None if not detected
    """
    # Try different patterns for regression head weights
    patterns = [
        f'head_blocks.{task_type}.0.network.4.weight',  # Standard pattern
        f'head_blocks.{task_type}.0.network.3.weight',    # Alternative with fewer layers
    ]
    
    for pattern in patterns:
        if pattern in state_dict:
            return state_dict[pattern].shape[0]
    
    # Fallback: search for any weight matching the pattern
    for key in state_dict.keys():
        if f'head_blocks.{task_type}' in key and key.endswith('.weight'):
            # Get the first dimension
            num_targets = state_dict[key].shape[0]
            # Verify it's reasonable (not too large)
            if num_targets < 1000:
                return num_targets
    
    return None


def load_model_and_data(args):
    """
    Load model from checkpoint and prepare data.

    Returns:
        model: Loaded model in eval mode
        data_dict: Dictionary with genotype data for each variant type
        sample_ids: List of sample IDs (filtered if samples_set is provided)
        snp_ids: Dictionary with variant IDs for each type
        task_info: Dictionary with task information
        is_multi_branch: Boolean indicating if model is multi-branch
    """
    print("=" * 80)
    print("AQUILA: Integrated Gradients")
    print("=" * 80)

    # Handle --model-dir option
    if args.model_dir:
        print(f"\nResolving paths from model directory: {args.model_dir}")
        config_path, checkpoint_path = resolve_model_dir(args)
        args.config = config_path
        args.checkpoint = checkpoint_path
        model_dir = Path(args.model_dir)
        
        # Load task names from model directory
        regression_tasks, classification_tasks, extra_info = load_task_names_from_model_dir(model_dir)
        print(f"Loaded {len(regression_tasks)} regression tasks from model directory")
        print(f"Loaded {len(classification_tasks)} classification tasks from model directory")
    else:
        regression_tasks = []
        classification_tasks = []
        extra_info = {}
        model_dir = None

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

    # Load genotype data based on encoding
    if encoding_type in ('diploid_onehot', 'onehot'):
        if is_multi_branch:
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

            print(f"Loading multi-branch genotype data from: {geno_file}")
            result = parse_genotype_file(geno_file, encoding_type=encoding_type, variant_type=variant_type)

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
            snp_matrix, sample_ids, snp_id_list = parse_genotype_file(
                vcf_file, encoding_type=encoding_type, variant_type=variant_type
            )
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
        
        # Filter data - ensure correct indexing
        filtered_sample_ids = [sample_ids[i] for i in target_indices]
        filtered_data_dict = {}
        for vtype, data in data_dict.items():
            # Data shape should be (n_samples, seq_len, C)
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

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Detect task type
    if args.task_type:
        task_type = args.task_type
    else:
        # Auto-detect from checkpoint
        regression_detected = detect_num_targets_from_checkpoint(state_dict, 'regression')
        classification_detected = detect_num_targets_from_checkpoint(state_dict, 'classification')
        
        if classification_detected and classification_detected > 0:
            task_type = 'classification'
        else:
            task_type = 'regression'
    
    print(f"Task type: {task_type}")

    # Detect number of targets
    num_targets = detect_num_targets_from_checkpoint(state_dict, task_type)
    print(f"Detected {num_targets} {task_type} targets from checkpoint")

    # Update task lists if needed
    if task_type == 'regression':
        if not regression_tasks and num_targets:
            regression_tasks = [f"task_{i}" for i in range(num_targets)]
        # Update config
        model_config = config.get('model', {})
        if 'heads' in model_config and 'regression' in model_config['heads']:
            for block in model_config['heads']['regression']:
                if isinstance(block, dict) and block.get('name') == 'regression_head':
                    if num_targets:
                        block['num_targets'] = num_targets
        task_names = regression_tasks
    else:  # classification
        if not classification_tasks and num_targets:
            classification_tasks = [f"task_{i}" for i in range(num_targets)]
        task_names = classification_tasks

    # Determine which task(s) to compute IG for
    if args.task:
        if args.task in task_names:
            task_indices = [task_names.index(args.task)]
        else:
            raise ValueError(f"Task '{args.task}' not found in available tasks: {task_names}")
    else:
        task_indices = list(range(len(task_names)))

    if not task_names:
        task_names = [f'{task_type}_task_0']
        task_indices = [0]

    num_tasks = len(task_indices)
    print(f"\nComputing IG for {num_tasks} task(s): {[task_names[i] for i in task_indices]}")

    # Create model
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    model = create_model_from_config(
        config=config,
        seq_length=seq_length,
        regression_tasks=regression_tasks,
        classification_tasks=classification_tasks
    )

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

    # Package task info
    task_info = {
        'task_type': task_type,
        'task_names': task_names,
        'task_indices': task_indices,
        'num_targets': num_targets,
        'regression_tasks': regression_tasks,
        'classification_tasks': classification_tasks,
    }

    return model, data_dict, sample_ids, snp_ids, task_info, is_multi_branch


class IntegratedGradients:
    """
    Compute integrated gradients for model attribution.

    Integrated Gradients = (input - baseline) * integral(gradient at interpolated points)

    For diploid_onehot / onehot encodings, baseline is all zeros (missing genotype).
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
                        Each tensor shape: (1, seq_len, C) with C=8 or 3
            task_idx: Task index to compute gradient for
            target_type: 'regression' or 'classification'

        Returns:
            Dictionary with integrated gradients for each variant type
            Each value shape: (1, seq_len, C)
        """
        self.model.eval()

        # Create baseline (all zeros - missing genotype)
        baseline_dict = {}
        for vtype, input_tensor in input_dict.items():
            baseline_dict[vtype] = torch.zeros_like(input_tensor)

        # Compute integrated gradients using trapezoidal rule
        alphas = torch.linspace(0, 1, self.num_steps, device=self.device)

        # Store gradients at each step
        accumulated_grads = {}
        for vtype in input_dict.keys():
            accumulated_grads[vtype] = torch.zeros_like(input_dict[vtype], device=self.device)

        # Compute gradients at each interpolated point
        for step_idx, alpha in enumerate(alphas):
            # Create interpolated input
            interpolated_dict = {}
            for vtype, input_tensor in input_dict.items():
                interpolated_input = baseline_dict[vtype] + alpha * (input_tensor - baseline_dict[vtype])
                interpolated_input.requires_grad = True
                interpolated_dict[vtype] = interpolated_input

            # Forward pass
            if self.is_multi_branch:
                outputs = self.model(interpolated_dict)
            else:
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

            # Accumulate gradients with trapezoidal weights
            for vtype in input_dict.keys():
                if interpolated_dict[vtype].grad is not None:
                    # Trapezoidal rule: endpoints have weight 0.5, interior points have weight 1.0
                    if step_idx == 0 or step_idx == len(alphas) - 1:
                        weight = 0.5
                    else:
                        weight = 1.0
                    accumulated_grads[vtype] += weight * interpolated_dict[vtype].grad

        # Final IG computation
        # IG = sum(gradients) * step_size * (input - baseline)
        # For trapezoidal: step_size = 1 / (num_steps - 1)
        step_size = 1.0 / (self.num_steps - 1)
        
        final_ig = {}
        for vtype in input_dict.keys():
            # Apply step size
            ig = accumulated_grads[vtype].cpu().numpy() * step_size
            
            # Multiply by (input - baseline) for full IG formula
            # Since baseline=0, this is just input
            input_np = input_dict[vtype].cpu().numpy()
            final_ig[vtype] = ig * input_np

        return final_ig


def compute_ig_streaming(model, data_dict, sample_ids, snp_ids,
                         task_info, is_multi_branch, args):
    """
    Compute and save integrated gradients in streaming mode (one sample at a time).
    """
    print("\n" + "=" * 80)
    print("Computing Integrated Gradients (Streaming Mode)")
    print("=" * 80)

    ig_computer = IntegratedGradients(model, args.device, args.num_steps, is_multi_branch)

    task_type = task_info['task_type']
    task_names = task_info['task_names']
    task_indices = task_info['task_indices']

    num_samples = len(sample_ids)
    num_tasks = len(task_indices)
    
    print(f"Processing {num_samples} samples x {num_tasks} tasks = {num_samples * num_tasks} computations...")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / 'ig_results.h5'

    with h5py.File(h5_path, 'w') as f:
        # Save metadata
        f.attrs['num_samples'] = num_samples
        f.attrs['num_tasks'] = num_tasks
        f.attrs['num_steps'] = args.num_steps
        f.attrs['task_type'] = task_type
        
        # Save sample IDs
        f.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
        
        # Save task names
        task_names_bytes = [t.encode('utf-8') for t in task_names]
        f.create_dataset('task_names', data=np.array(task_names_bytes, dtype='S'))
        
        # Save variant IDs
        for vtype, variant_ids in snp_ids.items():
            variant_ids_bytes = [v.encode('utf-8') for v in variant_ids]
            f.create_dataset(f'variant_ids/{vtype}', data=np.array(variant_ids_bytes, dtype='S'))

        # Process each sample
        for sample_idx in range(num_samples):
            if sample_idx % 10 == 0:
                print(f"  Processing sample {sample_idx + 1}/{num_samples}...")

            sample_id = sample_ids[sample_idx]

            # Prepare input for this sample
            input_dict = {}
            for vtype, data in data_dict.items():
                sample_data = data[sample_idx]
                input_tensor = torch.from_numpy(sample_data).unsqueeze(0).to(args.device)
                input_dict[vtype] = input_tensor

            # Compute IG for each task
            for task_idx in task_indices:
                ig_scores = ig_computer.compute_ig_single_sample_single_task(
                    input_dict,
                    task_idx=task_idx,
                    target_type=task_type
                )
                task_name = task_names[task_idx]
                
                # Save immediately to HDF5
                safe_id = sanitize_hdf5_key(sample_id)
                sample_group = safe_id
                task_group = f'{sample_group}/{task_name}'
                
                for vtype, scores in ig_scores.items():
                    scores_2d = scores[0]  # (1, seq_len, C) -> (seq_len, C)
                    f.create_dataset(f'{task_group}/{vtype}', data=scores_2d)

    print(f"  Saved: {h5_path}")
    return h5_path


def compute_ig_in_memory(model, data_dict, sample_ids, snp_ids,
                        task_info, is_multi_branch, args):
    """
    Compute integrated gradients and keep all results in memory.
    """
    print("\n" + "=" * 80)
    print("Computing Integrated Gradients")
    print("=" * 80)

    ig_computer = IntegratedGradients(model, args.device, args.num_steps, is_multi_branch)

    task_type = task_info['task_type']
    task_names = task_info['task_names']
    task_indices = task_info['task_indices']

    num_samples = len(sample_ids)
    num_tasks = len(task_indices)
    
    print(f"Processing {num_samples} samples x {num_tasks} tasks = {num_samples * num_tasks} computations...")

    all_results = {}

    for sample_idx in range(num_samples):
        if sample_idx % 10 == 0:
            print(f"  Processing sample {sample_idx + 1}/{num_samples}...")

        sample_id = sample_ids[sample_idx]

        # Prepare input for this sample
        input_dict = {}
        for vtype, data in data_dict.items():
            sample_data = data[sample_idx]
            input_tensor = torch.from_numpy(sample_data).unsqueeze(0).to(args.device)
            input_dict[vtype] = input_tensor

        # Compute IG for each task
        sample_results = {}
        for task_idx in task_indices:
            ig_scores = ig_computer.compute_ig_single_sample_single_task(
                input_dict,
                task_idx=task_idx,
                target_type=task_type
            )
            task_name = task_names[task_idx]
            sample_results[task_name] = ig_scores

        all_results[sample_id] = sample_results

    return all_results


def save_results_h5(results, sample_ids, snp_ids, task_info, args, output_dir):
    """
    Save integrated gradients results to HDF5 file.
    """
    print("\n" + "=" * 80)
    print("Saving Results to HDF5")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = output_dir / 'ig_results.h5'

    task_type = task_info['task_type']
    task_names = task_info['task_names']
    task_indices = task_info['task_indices']
    # Only save the tasks that were actually computed (may be a subset of all task_names)
    computed_task_names = [task_names[idx] for idx in task_indices]

    with h5py.File(h5_path, 'w') as f:
        # Save metadata
        f.attrs['num_samples'] = len(sample_ids)
        f.attrs['num_tasks'] = len(computed_task_names)
        f.attrs['num_steps'] = args.num_steps
        f.attrs['task_type'] = task_type
        
        # Save sample IDs
        f.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
        
        # Save task names (only computed tasks)
        task_names_bytes = [t.encode('utf-8') for t in computed_task_names]
        f.create_dataset('task_names', data=np.array(task_names_bytes, dtype='S'))
        
        # Save variant IDs
        for vtype, variant_ids in snp_ids.items():
            variant_ids_bytes = [v.encode('utf-8') for v in variant_ids]
            f.create_dataset(f'variant_ids/{vtype}', data=np.array(variant_ids_bytes, dtype='S'))
        
        # Save IG scores
        print("\nSaving IG scores...")
        for sample_idx, sample_id in enumerate(sample_ids):
            if sample_idx % 50 == 0:
                print(f"  Saving sample {sample_idx + 1}/{len(sample_ids)}...")

            sample_results = results[sample_id]

            for task_name in computed_task_names:
                ig_scores = sample_results[task_name]
                safe_id = sanitize_hdf5_key(sample_id)
                sample_group = safe_id
                task_group = f'{sample_group}/{task_name}'
                
                for vtype, scores in ig_scores.items():
                    scores_2d = scores[0]
                    f.create_dataset(f'{task_group}/{vtype}', data=scores_2d)

    print(f"  Saved: {h5_path}")

    # Save summary JSON
    summary = {
        'num_samples': len(sample_ids),
        'sample_ids': sample_ids,
        'num_tasks': len(computed_task_names),
        'task_names': computed_task_names,
        'task_type': task_type,
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
    model, data_dict, sample_ids, snp_ids, task_info, is_multi_branch = load_model_and_data(args)

    # Choose compute method based on streaming flag
    if args.streaming:
        h5_path = compute_ig_streaming(
            model, data_dict, sample_ids, snp_ids,
            task_info, is_multi_branch, args
        )
        
        # Save summary for streaming mode
        output_dir = Path(args.output)
        summary = {
            'num_samples': len(sample_ids),
            'sample_ids': sample_ids,
            'num_tasks': len(task_info['task_names']),
            'task_names': task_info['task_names'],
            'task_type': task_info['task_type'],
            'variant_types': list(snp_ids.keys()),
            'num_variants': {vtype: len(ids) for vtype, ids in snp_ids.items()},
            'num_steps': args.num_steps,
        }
        summary_output = output_dir / 'ig_summary.json'
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {summary_output}")
        print(f"\nAll results saved to: {output_dir}")
    else:
        # Compute in memory
        results = compute_ig_in_memory(
            model, data_dict, sample_ids, snp_ids,
            task_info, is_multi_branch, args
        )

        # Save results to HDF5
        save_results_h5(results, sample_ids, snp_ids, task_info, args, args.output)

    print("\n" + "=" * 80)
    print("Integrated Gradients Computation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
