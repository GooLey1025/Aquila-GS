#!/usr/bin/env python3
"""
Aquila SHAP Interpretation Script

Compute SHAP (SHapley Additive exPlanations) values for genomic variant attribution
using trained Aquila models. Uses shap.explainers._gradient.GradientExplainer with
_PyTorchGradient backend (equivalent to DeepSHAP algorithm).

The output HDF5 format is IDENTICAL to aquila_ig.py, so aquila_ig_interpretation.py
can process SHAP results directly without any modification.

Output shape: (num_variants, channels=8) per sample per task - same as IG!

Usage:
    # Method 1: Provide config and checkpoint explicitly
    python aquila_shap.py --config params.yaml --checkpoint checkpoints/best_checkpoint.pt \\
        --vcf input.vcf.gz --output shap_output

    # Method 2: Provide model directory (auto-detect config and checkpoint)
    python aquila_shap.py --model-dir /path/to/model_dir --vcf input.vcf.gz --output shap_output

    # With background sampling (recommended for better SHAP estimates)
    python aquila_shap.py --model-dir /path/to/model_dir --vcf input.vcf.gz \\
        --output shap_output --background-samples 100 --batch-size 8
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
import h5py
import pickle
from typing import Dict, List, Optional, Tuple
import shap
from shap.explainers._gradient import GradientExplainer as PyTorchGradientExplainer
import warnings

from aquila.varnn import create_model_from_config
from aquila.utils import load_config
from aquila.encoding import parse_genotype_file

# Suppress shap verbose output
shap.utils.__dict__['warning'] = lambda *args, **kwargs: None


import re

def sanitize_hdf5_key(key: str) -> str:
    """Sanitize a string to be a valid HDF5 group/dataset name."""
    if not key:
        return "_"
    key = re.sub(r'[/\\:*?"<>#,]', '_', key)
    key = key.strip().lstrip('.')
    return key if key else "_"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute SHAP values for Aquila model'
    )

    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to configuration YAML file'
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
        help='Path to VCF file with genotype data'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='shap_output',
        help='Output directory for SHAP results'
    )

    parser.add_argument(
        '--encoding-type',
        type=str,
        default='diploid_onehot',
        choices=['token', 'diploid_onehot', 'onehot', 'snp_vcf', 'indel_vcf',
                 'sv_vcf', 'snp_indel_vcf', 'snp_indel_sv_vcf'],
        help='Encoding type (must match training)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for computation'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for SHAP computation (default: 8)'
    )

    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help='Task name to compute SHAP for. If not specified, compute for all tasks.'
    )

    parser.add_argument(
        '--task-type',
        type=str,
        default=None,
        choices=['regression', 'classification', None],
        help='Type of task to compute SHAP for. If not specified, auto-detect.'
    )

    parser.add_argument(
        '--background-samples',
        type=int,
        default=50,
        help='Number of background samples for SHAP (default: 50). '
             'Set to 0 to use mean baseline.'
    )

    parser.add_argument(
        '--samples-set',
        type=str,
        default=None,
        help='Path to txt file with sample IDs to compute SHAP for '
             '(one per line). If not specified, compute for all samples.'
    )

    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Enable streaming mode: compute and save results incrementally'
    )

    return parser.parse_args()


def resolve_model_dir(args):
    """Resolve config and checkpoint paths from model directory."""
    model_dir = Path(args.model_dir)

    config_path = model_dir / 'params.yaml'
    if not config_path.exists():
        for name in ['config.yaml', 'training_config.yaml']:
            alt_path = model_dir / name
            if alt_path.exists():
                config_path = alt_path
                break
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file in {model_dir}")

    checkpoint_dir = model_dir / 'checkpoints'
    checkpoint_path = None
    if checkpoint_dir.exists():
        best_ckpt = checkpoint_dir / 'best_checkpoint.pt'
        if best_ckpt.exists():
            checkpoint_path = best_ckpt
        else:
            latest_ckpt = checkpoint_dir / 'latest_checkpoint.pt'
            if latest_ckpt.exists():
                checkpoint_path = latest_ckpt

    if checkpoint_path is None:
        raise FileNotFoundError(f"Could not find checkpoint in {model_dir}/checkpoints")

    return str(config_path), str(checkpoint_path)


def load_task_names_from_model_dir(model_dir: Path) -> Tuple[List[str], List[str], Dict]:
    """Load task names and config from model directory."""
    regression_tasks, classification_tasks = [], []
    extra_info = {}

    norm_stats_path = model_dir / 'normalization_stats.pkl'
    if norm_stats_path.exists():
        with open(norm_stats_path, 'rb') as f:
            norm_stats = pickle.load(f)
        regression_tasks = norm_stats.get('regression_tasks', [])
        extra_info['regression_means'] = norm_stats.get('regression_means', {})
        extra_info['regression_stds'] = norm_stats.get('regression_stds', {})

    task_mapping_path = model_dir / 'task_mapping.tsv'
    if task_mapping_path.exists():
        df = pd.read_csv(task_mapping_path, sep='\t')
        if 'task_name' in df.columns and 'task_type' in df.columns:
            for _, row in df.iterrows():
                if row['task_type'] == 'regression' and row['task_name'] not in regression_tasks:
                    regression_tasks.append(row['task_name'])
                elif row['task_type'] == 'classification' and row['task_name'] not in classification_tasks:
                    classification_tasks.append(row['task_name'])

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
    """Detect number of targets from checkpoint state dict."""
    patterns = [
        f'head_blocks.{task_type}.0.network.4.weight',
        f'head_blocks.{task_type}.0.network.3.weight',
    ]
    for pattern in patterns:
        if pattern in state_dict:
            return state_dict[pattern].shape[0]
    for key in state_dict.keys():
        if f'head_blocks.{task_type}' in key and key.endswith('.weight'):
            num_targets = state_dict[key].shape[0]
            if num_targets < 1000:
                return num_targets
    return None


def load_model_and_data(args):
    """
    Load model from checkpoint and prepare data.
    """
    print("=" * 80)
    print("AQUILA: SHAP Interpretation")
    print("=" * 80)

    if args.model_dir:
        print(f"\nResolving paths from model directory: {args.model_dir}")
        config_path, checkpoint_path = resolve_model_dir(args)
        args.config = config_path
        args.checkpoint = checkpoint_path
        model_dir = Path(args.model_dir)
        regression_tasks, classification_tasks, extra_info = load_task_names_from_model_dir(model_dir)
        print(f"Loaded {len(regression_tasks)} regression tasks")
        print(f"Loaded {len(classification_tasks)} classification tasks")
    else:
        regression_tasks, classification_tasks, extra_info = [], [], {}
        model_dir = None

    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    encoding_type = args.encoding_type
    config['data']['encoding_type'] = encoding_type
    print(f"Encoding type: {encoding_type}")

    variant_type = config.get('data', {}).get('variant_type')
    print(f"Variant type: {variant_type}")

    is_multi_branch = encoding_type in ['snp_indel_vcf', 'snp_indel_sv_vcf'] or variant_type in ['snp_indel', 'snp_indel_sv']

    print(f"\nLoading genotype data from: {args.vcf}")

    data_config = config.get('data', {})

    if encoding_type in ('diploid_onehot', 'onehot'):
        if is_multi_branch:
            if args.vcf:
                geno_file = args.vcf
            else:
                geno_file = data_config.get('geno_file')
                if not geno_file:
                    raise ValueError("No VCF file specified.")
            print(f"Loading multi-branch genotype data from: {geno_file}")
            result = parse_genotype_file(geno_file, encoding_type=encoding_type, variant_type=variant_type)

            data_dict = {}
            sample_ids = []
            snp_ids = {}

            if 'snp' in result:
                data_dict['snp'] = result['snp']['matrix']
                sample_ids = result['snp']['sample_ids']
                snp_ids['snp'] = result['snp']['variant_ids']
                print(f"  Loaded {len(snp_ids['snp'])} SNPs for {len(sample_ids)} samples")
            if 'indel' in result:
                data_dict['indel'] = result['indel']['matrix']
                snp_ids['indel'] = result['indel']['variant_ids']
                print(f"  Loaded {len(snp_ids['indel'])} INDELs")
            if 'sv' in result:
                data_dict['sv'] = result['sv']['matrix']
                snp_ids['sv'] = result['sv']['variant_ids']
                print(f"  Loaded {len(snp_ids['sv'])} SVs")
        else:
            if args.vcf:
                vcf_file = args.vcf
            else:
                vcf_file = data_config.get('geno_file')
                if not vcf_file:
                    raise ValueError("No VCF file specified.")
            print("Loading genotype data...")
            result = parse_genotype_file(
                vcf_file, encoding_type=encoding_type, variant_type=variant_type
            )
            if isinstance(result, dict):
                snp_matrix = result['matrix']
                sample_ids = result['sample_ids']
                snp_id_list = result['variant_ids']
            else:
                snp_matrix, sample_ids, snp_id_list = result
            data_dict = {'snp': snp_matrix}
            snp_ids = {'snp': snp_id_list}
            print(f"Loaded {len(snp_id_list)} variants for {len(sample_ids)} samples")
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")

    # Filter target samples
    if args.samples_set:
        print(f"\nFiltering target samples using: {args.samples_set}")
        with open(args.samples_set, 'r') as f:
            target_samples = set(line.strip() for line in f if line.strip())
        sample_to_idx = {sid: idx for idx, sid in enumerate(sample_ids)}
        target_indices = [sample_to_idx[sid] for sid in target_samples if sid in sample_to_idx]
        if not target_indices:
            raise ValueError("No matching samples found in VCF!")
        print(f"  Found {len(target_indices)} matching samples")
        filtered_sample_ids = [sample_ids[i] for i in target_indices]
        filtered_data_dict = {vtype: data[target_indices] for vtype, data in data_dict.items()}
        data_dict = filtered_data_dict
        sample_ids = filtered_sample_ids

    print(f"\nLoaded data shapes:")
    for vtype, data in data_dict.items():
        print(f"  {vtype}: {data.shape}")

    seq_length = {vtype: data.shape[1] for vtype, data in data_dict.items()}
    print(f"Sequence lengths: {seq_length}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if args.task_type:
        task_type = args.task_type
    else:
        regression_detected = detect_num_targets_from_checkpoint(state_dict, 'regression')
        classification_detected = detect_num_targets_from_checkpoint(state_dict, 'classification')
        if classification_detected and classification_detected > 0:
            task_type = 'classification'
        else:
            task_type = 'regression'

    print(f"Task type: {task_type}")

    num_targets = detect_num_targets_from_checkpoint(state_dict, task_type)
    print(f"Detected {num_targets} {task_type} targets from checkpoint")

    if task_type == 'regression':
        if not regression_tasks and num_targets:
            regression_tasks = [f"task_{i}" for i in range(num_targets)]
        model_config = config.get('model', {})
        if 'heads' in model_config and 'regression' in model_config['heads']:
            for block in model_config['heads']['regression']:
                if isinstance(block, dict) and block.get('name') == 'regression_head':
                    if num_targets:
                        block['num_targets'] = num_targets
        task_names = regression_tasks
    else:
        if not classification_tasks and num_targets:
            classification_tasks = [f"task_{i}" for i in range(num_targets)]
        task_names = classification_tasks

    if args.task:
        if args.task in task_names:
            task_indices = [task_names.index(args.task)]
        else:
            raise ValueError(f"Task '{args.task}' not found: {task_names}")
    else:
        task_indices = list(range(len(task_names)))

    if not task_names:
        task_names = [f'{task_type}_task_0']
        task_indices = [0]

    num_tasks = len(task_indices)
    print(f"\nComputing SHAP for {num_tasks} task(s): {[task_names[i] for i in task_indices]}")

    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    model = create_model_from_config(
        config=config,
        seq_length=seq_length,
        regression_tasks=regression_tasks,
        classification_tasks=classification_tasks
    )

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

    task_info = {
        'task_type': task_type,
        'task_names': task_names,
        'task_indices': task_indices,
        'num_targets': num_targets,
        'regression_tasks': regression_tasks,
        'classification_tasks': classification_tasks,
    }

    return model, data_dict, sample_ids, snp_ids, task_info, is_multi_branch


def build_shap_explainer(model, data_dict, sample_ids, args, is_multi_branch, task_type):
    """
    Build SHAP GradientExplainer with background data.

    Uses shap.explainers._gradient.GradientExplainer with _PyTorchGradient backend.
    The model wrapper MUST be an nn.Module (not a function) for PyTorch backend detection.
    Computes: shap_value = (input - baseline) * expected_gradient
    Returns per-channel attribution: (batch, seq_len, channels, num_tasks)
    """
    print("\n" + "=" * 80)
    print("Building SHAP Explainer (DeepSHAP via GradientExplainer + PyTorch)")
    print("=" * 80)

    print(f"Background samples: {args.background_samples}")

    # Build background dataset
    n_total = len(sample_ids)
    if args.background_samples > 0 and args.background_samples < n_total:
        n_bg = min(args.background_samples, n_total)
        bg_indices = np.random.choice(n_total, size=n_bg, replace=False)
        bg_dict = {}
        for vtype, data in data_dict.items():
            bg_dict[vtype] = torch.from_numpy(data[bg_indices]).float().to(args.device)
        print(f"  Using {n_bg} random background samples")
    else:
        bg_dict = {}
        for vtype, data in data_dict.items():
            bg_dict[vtype] = torch.from_numpy(data).float().mean(axis=0, keepdims=True).to(args.device)
        print(f"  Using mean baseline across {n_total} samples")

    class SHAPModelWrapper(torch.nn.Module):
        """nn.Module wrapper so GradientExplainer uses _PyTorchGradient backend.

        For multi-branch models: concatenates all variant-type tensors along
        channel dimension before passing to the backbone. For single-branch:
        directly passes the tensor through.
        """
        def __init__(self, aquila_model, data_dict_keys, is_multi_branch, task_type):
            super().__init__()
            self.aquila_model = aquila_model
            self.data_dict_keys = list(data_dict_keys)
            self.is_multi_branch = is_multi_branch
            self.task_type = task_type

        def forward(self, x_input):
            # x_input is a single concatenated tensor: (batch, seq_len, total_channels)
            if not self.is_multi_branch:
                # Single branch: VarNN expects a single tensor (batch, seq, 8), NOT a dict
                x = x_input  # (batch, seq_len, 8)
            else:
                # Multi-branch: split concatenated tensor back into per-type dict
                x_dict = {}
                offset = 0
                for vtype in self.data_dict_keys:
                    if vtype == 'snp':
                        ch = 8
                    elif vtype == 'indel':
                        ch = 8
                    elif vtype == 'sv':
                        ch = 8
                    else:
                        ch = 8
                    x_dict[vtype] = x_input[..., offset:offset + ch]
                    offset += ch
                x = x_dict
            outputs = self.aquila_model(x)
            if isinstance(outputs, dict):
                if self.task_type == 'regression' and 'regression' in outputs:
                    return outputs['regression']
                elif 'classification' in outputs:
                    return outputs['classification']
                else:
                    return list(outputs.values())[0]
            return outputs

    # Prepare single tensor for background (required by PyTorchGradient).
    # For multi-branch: pad shorter tensors to max seq_len so concatenation works.
    if is_multi_branch and len(data_dict) > 1:
        max_seq_len = max(data[k].shape[1] for k in data_dict)
        print(f"  Multi-branch: padding to max seq_len={max_seq_len}")
        padded_tensors = []
        for k in sorted(data_dict.keys()):
            t = bg_dict[k]
            if t.shape[1] < max_seq_len:
                pad = torch.zeros(*t.shape[:2], max_seq_len - t.shape[1], t.shape[2], device=t.device)
                t = torch.cat([t, pad], dim=1)
            padded_tensors.append(t)
        bg_concat = torch.cat(padded_tensors, dim=-1)
    else:
        bg_tensors = [bg_dict[k] for k in sorted(bg_dict.keys())]
        bg_concat = torch.cat(bg_tensors, dim=-1)
    print(f"  Background tensor: {bg_concat.shape}")

    # Wrap model
    data_dict_keys = list(data_dict.keys())
    wrapped_model = SHAPModelWrapper(model, data_dict_keys, is_multi_branch, task_type)
    wrapped_model.to(args.device)
    wrapped_model.eval()

    # Also prepare single input tensor for explainer
    # (will be converted from input_dict inside shap computation)
    input_keys = sorted(data_dict_keys)

    # Use PyTorchGradient via shap.explainers._gradient.GradientExplainer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = PyTorchGradientExplainer(
            model=wrapped_model,
            data=bg_concat,
            batch_size=args.batch_size
        )

    print("  Explainer built successfully (PyTorch backend)")
    return explainer, input_keys


def compute_shap_streaming(model, data_dict, sample_ids, snp_ids,
                            task_info, is_multi_branch, args):
    """
    Compute SHAP values in streaming mode (one sample at a time).

    SHAP GradientExplainer.shap_values() returns ALL tasks at once:
    - Shape: (batch, seq_len, num_tasks) or (batch, seq_len, 1)
    - One scalar value per variant per task (NOT per-channel like IG)
    """
    print("\n" + "=" * 80)
    print("Computing SHAP Values (Streaming Mode)")
    print("=" * 80)

    explainer, input_keys = build_shap_explainer(model, data_dict, sample_ids, args, is_multi_branch, task_info['task_type'])

    task_type = task_info['task_type']
    task_names = task_info['task_names']
    task_indices = task_info['task_indices']
    num_targets = task_info['num_targets'] or len(task_names)

    num_samples = len(sample_ids)
    num_tasks = len(task_indices)

    print(f"\nProcessing {num_samples} samples x {num_tasks} tasks")
    print(f"  SHAP output: (batch=1, seq_len, channels=8, num_tasks) -> per-channel attribution")
    print(f"  HDF5 format: identical to IG output, compatible with aquila_ig_interpretation.py")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / 'shap_results.h5'

    with h5py.File(h5_path, 'w') as f:
        f.attrs['num_samples'] = num_samples
        f.attrs['num_tasks'] = num_tasks
        f.attrs['background_samples'] = args.background_samples
        f.attrs['batch_size'] = args.batch_size
        f.attrs['task_type'] = task_type
        f.attrs['num_targets'] = num_targets
        f.attrs['method'] = 'GradientExplainer_DeepSHAP'

        f.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))

        task_names_bytes = [t.encode('utf-8') for t in task_names]
        f.create_dataset('task_names', data=np.array(task_names_bytes, dtype='S'))

        for vtype, variant_ids in snp_ids.items():
            variant_ids_bytes = [v.encode('utf-8') for v in variant_ids]
            f.create_dataset(f'variant_ids/{vtype}', data=np.array(variant_ids_bytes, dtype='S'))

        for sample_idx in range(num_samples):
            if sample_idx % 10 == 0:
                print(f"  Processing sample {sample_idx + 1}/{num_samples}...")

            sample_id = sample_ids[sample_idx]
            safe_id = sanitize_hdf5_key(sample_id)

            # Prepare input: concatenate all variant types into single tensor (with padding)
            input_tensors = []
            for vtype in sorted(data_dict.keys()):
                sample_data = data_dict[vtype][sample_idx]
                input_tensors.append(torch.from_numpy(sample_data).unsqueeze(0).float().to(args.device))
            # Pad to max seq_len for multi-branch
            if is_multi_branch and len(input_tensors) > 1:
                max_seq_len = max(t.shape[1] for t in input_tensors)
                padded = []
                for t in input_tensors:
                    if t.shape[1] < max_seq_len:
                        pad = torch.zeros(1, max_seq_len - t.shape[1], t.shape[2], device=t.device)
                        t = torch.cat([t, pad], dim=1)
                    padded.append(t)
                input_concat = torch.cat(padded, dim=-1)
            else:
                input_concat = torch.cat(input_tensors, dim=-1)

            # Compute SHAP: shap_values returns (batch, seq_len, total_channels, num_tasks)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_out = explainer.shap_values(input_concat)

            # shap_out: (batch, seq_len, total_channels, num_tasks)
            # We need to split it back per variant type
            if isinstance(shap_out, torch.Tensor):
                shap_out = shap_out.cpu().numpy()
            sv = shap_out  # (1, seq_len, total_channels, num_tasks)

            if sv is None:
                continue

            # sv shape: (batch=1, seq_len, total_channels, num_tasks)
            # Save each requested task, per variant type
            sorted_vtypes = sorted(data_dict.keys())
            ch_per_vtype = {}
            for vtype in sorted_vtypes:
                if vtype == 'snp':
                    ch_per_vtype[vtype] = 8
                elif vtype == 'indel':
                    ch_per_vtype[vtype] = 8
                elif vtype == 'sv':
                    ch_per_vtype[vtype] = 8
                else:
                    ch_per_vtype[vtype] = 8

            for t_idx, task_idx in enumerate(task_indices):
                task_name = task_names[task_idx]
                sample_group = safe_id
                task_group = f'{sample_group}/{task_name}'

                for vtype in sorted_vtypes:
                    ch = ch_per_vtype[vtype]
                    if len(sorted_vtypes) == 1:
                        vtype_sv = sv  # (1, seq_len, 8, num_tasks)
                    else:
                        vtype_offset = sum(ch_per_vtype[vt] for vt in sorted_vtypes if vt != vtype)
                        vtype_sv = sv[..., vtype_offset:vtype_offset + ch]

                    # Extract this task: squeeze batch dim -> (seq_len, channels)
                    task_sv = vtype_sv[0, :, :, t_idx] if t_idx < vtype_sv.shape[3] else vtype_sv[0, :, :, 0]
                    f.create_dataset(f'{task_group}/{vtype}', data=task_sv)

    print(f"  Saved: {h5_path}")
    return h5_path


def compute_shap_in_memory(model, data_dict, sample_ids, snp_ids,
                            task_info, is_multi_branch, args):
    """
    Compute SHAP values and keep all results in memory.

    SHAP GradientExplainer.shap_values() returns ALL tasks at once:
    - Shape: (batch, seq_len, num_tasks) or (batch, seq_len, 1)
    """
    print("\n" + "=" * 80)
    print("Computing SHAP Values")
    print("=" * 80)

    explainer, input_keys = build_shap_explainer(model, data_dict, sample_ids, args, is_multi_branch, task_info['task_type'])

    task_type = task_info['task_type']
    task_names = task_info['task_names']
    task_indices = task_info['task_indices']

    num_samples = len(sample_ids)
    num_tasks = len(task_indices)

    print(f"\nProcessing {num_samples} samples x {num_tasks} tasks")
    print(f"  SHAP output: (batch=1, seq_len, channels=8, num_tasks) -> per-channel attribution")

    all_results = {}

    for sample_idx in range(num_samples):
        if sample_idx % 10 == 0:
            print(f"  Processing sample {sample_idx + 1}/{num_samples}...")

        sample_id = sample_ids[sample_idx]

        # Concatenate all variant types into single tensor (with padding for multi-branch)
        input_tensors = []
        for vtype in sorted(data_dict.keys()):
            sample_data = data_dict[vtype][sample_idx]
            input_tensors.append(torch.from_numpy(sample_data).unsqueeze(0).float().to(args.device))
        # Pad to max seq_len for multi-branch
        if is_multi_branch and len(input_tensors) > 1:
            max_seq_len = max(t.shape[1] for t in input_tensors)
            padded = []
            for t in input_tensors:
                if t.shape[1] < max_seq_len:
                    pad = torch.zeros(1, max_seq_len - t.shape[1], t.shape[2], device=t.device)
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            input_concat = torch.cat(padded, dim=-1)
        else:
            input_concat = torch.cat(input_tensors, dim=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_out = explainer.shap_values(input_concat)

        if isinstance(shap_out, torch.Tensor):
            shap_out = shap_out.cpu().numpy()

        sv = shap_out  # (1, seq_len, total_channels, num_tasks)
        if sv is None:
            sv = np.zeros((1, input_concat.shape[1], input_concat.shape[2], 1))

        sorted_vtypes = sorted(data_dict.keys())
        ch_per_vtype = {}
        for vtype in sorted_vtypes:
            ch_per_vtype[vtype] = 8  # diploid_onehot

        sample_results = {}
        for t_idx, task_idx in enumerate(task_indices):
            task_name = task_names[task_idx]
            scores_dict = {}
            for vtype in sorted_vtypes:
                ch = ch_per_vtype[vtype]
                if len(sorted_vtypes) == 1:
                    vtype_sv = sv
                else:
                    vtype_offset = sum(ch_per_vtype[vt] for vt in sorted_vtypes if vt != vtype)
                    vtype_sv = sv[..., vtype_offset:vtype_offset + ch]
                scores_dict[vtype] = vtype_sv[0, :, :, t_idx] if t_idx < vtype_sv.shape[3] else vtype_sv[0, :, :, 0]
            sample_results[task_name] = scores_dict

        all_results[sample_id] = sample_results

    return all_results


def save_results_h5(results, sample_ids, snp_ids, task_info, args, output_dir):
    """Save SHAP results to HDF5 file."""
    print("\n" + "=" * 80)
    print("Saving Results to HDF5")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = output_dir / 'shap_results.h5'

    task_type = task_info['task_type']
    task_names = task_info['task_names']

    num_targets = task_info['num_targets'] or len(task_names)

    with h5py.File(h5_path, 'w') as f:
        f.attrs['num_samples'] = len(sample_ids)
        f.attrs['num_tasks'] = len(task_names)
        f.attrs['num_targets'] = num_targets
        f.attrs['background_samples'] = args.background_samples
        f.attrs['batch_size'] = args.batch_size
        f.attrs['task_type'] = task_type
        f.attrs['method'] = 'GradientExplainer_DeepSHAP'

        f.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
        task_names_bytes = [t.encode('utf-8') for t in task_names]
        f.create_dataset('task_names', data=np.array(task_names_bytes, dtype='S'))

        for vtype, variant_ids in snp_ids.items():
            variant_ids_bytes = [v.encode('utf-8') for v in variant_ids]
            f.create_dataset(f'variant_ids/{vtype}', data=np.array(variant_ids_bytes, dtype='S'))

        print("\nSaving SHAP scores...")
        for sample_idx, sample_id in enumerate(sample_ids):
            if sample_idx % 50 == 0:
                print(f"  Saving sample {sample_idx + 1}/{len(sample_ids)}...")

            sample_results = results[sample_id]
            safe_id = sanitize_hdf5_key(sample_id)

            for task_name in task_names:
                shap_scores = sample_results[task_name]
                sample_group = safe_id
                task_group = f'{sample_group}/{task_name}'

                for vtype, scores in shap_scores.items():
                    # SHAP returns (seq_len, channels) - same as IG format
                    f.create_dataset(f'{task_group}/{vtype}', data=scores)

    print(f"  Saved: {h5_path}")

    summary = {
        'num_samples': len(sample_ids),
        'sample_ids': sample_ids,
        'num_tasks': len(task_names),
        'task_names': task_names,
        'task_type': task_type,
        'variant_types': list(snp_ids.keys()),
        'num_variants': {vtype: len(ids) for vtype, ids in snp_ids.items()},
        'background_samples': args.background_samples,
        'batch_size': args.batch_size,
        'method': 'GradientExplainer_DeepSHAP',
    }

    summary_output = output_dir / 'shap_summary.json'
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_output}")

    print(f"\nAll results saved to: {output_dir}")


def main():
    """Main function."""
    args = parse_args()

    model, data_dict, sample_ids, snp_ids, task_info, is_multi_branch = load_model_and_data(args)

    if args.streaming:
        h5_path = compute_shap_streaming(
            model, data_dict, sample_ids, snp_ids,
            task_info, is_multi_branch, args
        )

        output_dir = Path(args.output)
        summary = {
            'num_samples': len(sample_ids),
            'sample_ids': sample_ids,
            'num_tasks': len(task_info['task_names']),
            'task_names': task_info['task_names'],
            'task_type': task_info['task_type'],
            'variant_types': list(snp_ids.keys()),
            'num_variants': {vtype: len(ids) for vtype, ids in snp_ids.items()},
            'background_samples': args.background_samples,
            'batch_size': args.batch_size,
            'method': 'GradientExplainer_DeepSHAP',
        }
        summary_output = output_dir / 'shap_summary.json'
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {summary_output}")
        print(f"\nAll results saved to: {output_dir}")
    else:
        results = compute_shap_in_memory(
            model, data_dict, sample_ids, snp_ids,
            task_info, is_multi_branch, args
        )
        save_results_h5(results, sample_ids, snp_ids, task_info, args, args.output)

    print("\n" + "=" * 80)
    print("SHAP Computation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
