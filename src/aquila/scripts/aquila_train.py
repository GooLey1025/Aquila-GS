#!/usr/bin/env python3
"""
Aquila VCF Training Script

Train a multi-task deep learning model for genomic prediction using VCF input.
Supports multi-branch architectures for SNP/INDEL/SV variants.

Usage:
    Single GPU:
        python aquila_train.py --config params.yaml
    
    Multi-GPU (using torchrun):
        torchrun --nproc_per_node=2 aquila_train.py --config params.yaml
        
        Note: num_gpu in config is not used. Specify GPU count in torchrun command.

"""

import argparse
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
import time
import os
from typing import Dict, Optional
from aquila.varnn import create_model_from_config
from aquila.trainer import VarTrainer
from aquila.data_utils import create_data_loaders
from aquila.utils import set_seed, save_config, load_config, print_model_summary, merge_wandb_config
from aquila.hpo import load_hpo_config, run_optuna_search, train_single_run
import pandas as pd
from aquila.metrics import MetricsCalculator


def save_postprocess_data(
    train_loader,
    val_loader,
    test_loader,
    vcf_file: str,
    pheno_file: str,
    normalization_stats: Dict,
    output_dir: Path,
    print_func=None
):
    """
    Save postprocessed genotype (VCF) and normalized phenotype data for benchmark models.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        test_loader: Test data loader (optional)
        vcf_file: Path to original VCF file
        pheno_file: Path to original phenotype file
        normalization_stats: Normalization statistics dict
        output_dir: Output directory
        print_func: Print function (for distributed training)
    """
    if print_func is None:
        print_func = print

    postprocess_dir = output_dir / 'data_postprocess'
    postprocess_dir.mkdir(parents=True, exist_ok=True)

    print_func("\n" + "=" * 80)
    print_func("Saving Postprocessed Data for Benchmark Models")
    print_func("=" * 80)

    # Collect normalized phenotypes from datasets (not loaders, to avoid DistributedSampler issues)
    def collect_phenotypes_from_dataset(dataset):
        """
        Collect phenotypes directly from dataset to avoid DistributedSampler sampling.
        """
        sample_ids = []
        regression_targets = []
        regression_masks = []
        classification_targets = []
        classification_masks = []

        # Collect data from all samples in the dataset
        # Subset.__getitem__ handles index mapping automatically
        for idx in range(len(dataset)):
            item = dataset[idx]

            if 'sample_id' in item:
                sample_ids.append(item['sample_id'])

            if 'regression_targets' in item:
                reg_target = item['regression_targets']
                reg_mask = item['regression_mask']
                if isinstance(reg_target, torch.Tensor):
                    regression_targets.append(reg_target.cpu().numpy())
                    regression_masks.append(reg_mask.cpu().numpy())
                else:
                    regression_targets.append(np.array(reg_target))
                    regression_masks.append(np.array(reg_mask))

            if 'classification_targets' in item:
                cls_target = item['classification_targets']
                cls_mask = item['classification_mask']
                if isinstance(cls_target, torch.Tensor):
                    classification_targets.append(cls_target.cpu().numpy())
                    classification_masks.append(cls_mask.cpu().numpy())
                else:
                    classification_targets.append(np.array(cls_target))
                    classification_masks.append(np.array(cls_mask))

        result = {
            'sample_ids': sample_ids,
            'regression_targets': np.array(regression_targets) if regression_targets else None,
            'regression_masks': np.array(regression_masks) if regression_masks else None,
            'classification_targets': np.array(classification_targets) if classification_targets else None,
            'classification_masks': np.array(classification_masks) if classification_masks else None,
        }
        return result

    # Collect phenotypes for each split (access datasets directly to avoid DistributedSampler)
    train_pheno = collect_phenotypes_from_dataset(train_loader.dataset)
    val_pheno = collect_phenotypes_from_dataset(
        val_loader.dataset) if val_loader else None
    test_pheno = collect_phenotypes_from_dataset(
        test_loader.dataset) if test_loader else None

    # Get task names
    regression_task_names = normalization_stats.get(
        'regression_tasks', []) if normalization_stats else []

    # Get classification task names from dataset
    classification_task_names = None
    if train_loader.dataset:
        # Handle Subset wrapper
        if isinstance(train_loader.dataset, torch.utils.data.Subset):
            actual_dataset = train_loader.dataset.dataset
        else:
            actual_dataset = train_loader.dataset

        if hasattr(actual_dataset, 'classification_cols') and actual_dataset.classification_cols:
            classification_task_names = actual_dataset.classification_cols

    # Save normalized phenotype files
    def save_phenotype_file(pheno_data, split_name):
        pheno_data_list = []

        for i, sample_id in enumerate(pheno_data['sample_ids']):
            row = {'Sample_ID': sample_id}

            # Add regression tasks
            if pheno_data['regression_targets'] is not None:
                for j, task_name in enumerate(regression_task_names):
                    if pheno_data['regression_masks'][i, j]:
                        row[task_name] = pheno_data['regression_targets'][i, j]
                    else:
                        row[task_name] = np.nan

            # Add classification tasks
            if pheno_data['classification_targets'] is not None and classification_task_names:
                for j, task_name in enumerate(classification_task_names):
                    if pheno_data['classification_masks'][i, j]:
                        row[task_name] = pheno_data['classification_targets'][i, j]
                    else:
                        row[task_name] = np.nan

            pheno_data_list.append(row)

        df = pd.DataFrame(pheno_data_list)
        pheno_path = postprocess_dir / f'pheno_{split_name}_normalized.tsv'
        df.to_csv(pheno_path, sep='\t', index=False, float_format='%.6f')
        print_func(
            f"  Saved normalized phenotypes: {pheno_path} ({len(df)} samples)")
        return pheno_path

    train_pheno_path = save_phenotype_file(train_pheno, 'train')
    if val_pheno:
        val_pheno_path = save_phenotype_file(val_pheno, 'valid')
    if test_pheno:
        test_pheno_path = save_phenotype_file(test_pheno, 'test')

    # Filter and save VCF files for each split
    def filter_vcf_by_samples(vcf_file, sample_ids, output_vcf_path):
        """
        Filter VCF file to include only specified samples.
        """
        sample_set = set(sample_ids)
        keep_indices = None

        with open(vcf_file, 'r') as f_in, open(output_vcf_path, 'w') as f_out:
            # Copy header lines
            for line in f_in:
                if line.startswith('#'):
                    if line.startswith('#CHROM'):
                        # Parse header to get sample column indices
                        header_fields = line.strip().split('\t')
                        # Samples start after FORMAT column
                        vcf_sample_ids = header_fields[9:]

                        # Find indices of samples to keep
                        keep_indices = []
                        for i, vcf_sample_id in enumerate(vcf_sample_ids):
                            if vcf_sample_id in sample_set:
                                # +9 because first 9 columns are metadata
                                keep_indices.append(i + 9)

                        if not keep_indices:
                            print_func(
                                f"  Warning: No matching samples found in VCF for split")
                            return False

                        # Write header with filtered samples
                        filtered_header = header_fields[:9] + \
                            [vcf_sample_ids[i - 9] for i in keep_indices]
                        f_out.write('\t'.join(filtered_header) + '\n')
                    else:
                        f_out.write(line)
                else:
                    # Process variant lines
                    if keep_indices is None:
                        print_func(f"  Error: VCF header not found")
                        return False

                    fields = line.strip().split('\t')
                    if len(fields) < 10:
                        continue

                    # Keep metadata columns (first 9) and filtered sample columns
                    filtered_fields = fields[:9] + \
                        [fields[i] for i in keep_indices]
                    f_out.write('\t'.join(filtered_fields) + '\n')

        return True

    # Save filtered VCF files
    def save_vcf_for_split(pheno_data, split_name):
        sample_ids = pheno_data['sample_ids']
        vcf_path = postprocess_dir / f'geno_{split_name}.vcf'

        if filter_vcf_by_samples(vcf_file, sample_ids, vcf_path):
            print_func(
                f"  Saved filtered VCF: {vcf_path} ({len(sample_ids)} samples)")
            return vcf_path
        else:
            return None

    train_vcf_path = save_vcf_for_split(train_pheno, 'train')
    if val_pheno:
        val_vcf_path = save_vcf_for_split(val_pheno, 'valid')
    if test_pheno:
        test_vcf_path = save_vcf_for_split(test_pheno, 'test')

    print_func(f"\nPostprocessed data saved to: {postprocess_dir}")
    print_func(
        "  - Normalized phenotype files: pheno_{train/valid/test}_normalized.tsv")
    print_func("  - Filtered VCF files: geno_{train/valid/test}.vcf")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Aquila model with VCF input')

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
        default=None,
        help='Output directory for checkpoints and results (overrides config, default: ./outputs)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility. Default is %(default)s.'
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

    parser.add_argument(
        '--local-rank',
        type=int,
        default=-1,
        help='Local rank for distributed training (automatically set by torchrun)'
    )

    parser.add_argument(
        '-dss',
        '--data-split-seed',
        type=int,
        default=42,
        help='Random seed for dataset splitting (default: 42). Keep this fixed to ensure consistent data splits across runs.'
    )

    parser.add_argument(
        '--data-split-file',
        type=str,
        default=None,
        help='Path to TSV file with columns Sample_ID and Split (train/valid/test). If provided, will use this file to determine data splits instead of random splitting.'
    )

    parser.add_argument(
        '-mp',
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training (FP16) for faster training and reduced memory usage'
    )

    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases (wandb) logging for experiment tracking'
    )

    parser.add_argument(
        '--wandb-project',
        type=str,
        default='aquila-gs',
        help='W&B project name (default: aquila-gs)'
    )

    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None,
        help='W&B run name (default: auto-generated from config and timestamp)'
    )

    parser.add_argument(
        '--wandb-tags',
        type=str,
        nargs='+',
        default=None,
        help='W&B tags for organizing runs'
    )

    parser.add_argument(
        '-hpo',
        '--hyperparameter-optimization',
        action='store_true',
        help='Enable hyperparameter optimization using Optuna (reads hpo section from config)'
    )

    parser.add_argument(
        '--save-postprocess-data',
        action='store_true',
        help='Save postprocessed genotype and normalized phenotype data for benchmark models (saved to output_dir/data_postprocess/)'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Detect distributed training (launched by torchrun)
    is_distributed = False
    rank = 0
    world_size = 1
    local_rank = 0

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        is_distributed = True
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # Initialize process group with device_id to avoid barrier() warnings
        dist.init_process_group(backend='nccl', device_id=local_rank)
        torch.cuda.set_device(local_rank)

    # Only print on rank 0
    def print_rank0(*args_print, **kwargs):
        if rank == 0:
            print(*args_print, **kwargs)

    # Load configuration
    print_rank0("=" * 80)
    if is_distributed:
        print_rank0("AQUILA: VCF Training (Distributed)")
        print_rank0("=" * 80)
        print_rank0(f"\nDistributed Training Mode")
        print_rank0(f"  World size: {world_size} GPUs")
        print_rank0(f"  Backend: NCCL")
    else:
        print_rank0("AQUILA: VCF Training")
        print_rank0("=" * 80)

    print_rank0(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)

    # Check if running hyperparameter optimization
    # Enable HPO if: 1) command line flag is set, OR 2) config has hpo.enabled: true
    hpo_enabled = args.hyperparameter_optimization
    hpo_config = config.get('hpo')

    if not hpo_enabled and hpo_config and hpo_config.get('enabled', False):
        hpo_enabled = True
        if rank == 0:
            print_rank0("\n🔍 HPO enabled via config file (hpo.enabled: true)")

    if hpo_enabled:
        # HPO is incompatible with distributed training
        if is_distributed:
            print_rank0(
                "\n❌ Error: Hyperparameter optimization cannot be used with distributed training (torchrun).")
            print_rank0(
                "   Please run HPO without torchrun.")
            print_rank0(
                "   Example: python aquila_train.py --config params.yaml -hpo")
            if is_distributed:
                dist.destroy_process_group()
            return

        if not hpo_config:
            print_rank0(
                "\n❌ Error: Hyperparameter optimization enabled but 'hpo' section not found in config file.")
            print_rank0(
                "   Please add an 'hpo' section to your config file with hyperparameter search space.")
            return

        try:
            hpo_config_normalized = load_hpo_config(hpo_config)
        except ValueError as e:
            print_rank0(f"\n❌ Error loading HPO configuration: {e}")
            return

        # Remove hpo section from base config (will be merged per trial)
        base_config = {k: v for k, v in config.items() if k != 'hpo'}

        # Create Optuna output directory
        optuna_output_dir = Path(args.output) / 'optuna_search'
        optuna_output_dir.mkdir(parents=True, exist_ok=True)

        print_rank0("\n" + "=" * 80)
        print_rank0("Optuna Hyperparameter Optimization")
        print_rank0("=" * 80)

        # Prepare arguments for train_single_run
        train_kwargs = {
            'args': args,
            'is_distributed': False,
            'rank': 0,
            'world_size': 1,
            'local_rank': 0,
            'print_rank0_func': None  # Will use default print function in train_single_run
        }

        # Run Optuna optimization
        study = run_optuna_search(
            base_config=base_config,
            hpo_config=hpo_config_normalized,
            output_dir=optuna_output_dir,
            train_single_run_func=train_single_run,
            train_single_run_kwargs=train_kwargs,
            n_trials=hpo_config_normalized.get('n_trials', 100),
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name
        )

        # Cleanup distributed training
        if is_distributed:
            dist.destroy_process_group()

        return

    # Check if running in WandB sweep mode
    is_sweep = False
    wandb_config = None
    if args.use_wandb and rank == 0:
        try:
            import wandb
            # Check if wandb is initialized (sweep mode)
            if wandb.run is not None:
                is_sweep = True
                wandb_config = dict(wandb.config)
                print_rank0("\n🔍 WandB Sweep mode detected!")
                print_rank0(f"   Sweep ID: {wandb.run.sweep_id}")
                print_rank0(f"   Run ID: {wandb.run.id}")

                # Merge wandb config into YAML config
                config = merge_wandb_config(config, wandb_config)
                print_rank0(
                    "   Merged WandB sweep hyperparameters into config")
        except ImportError:
            pass
        except Exception as e:
            print_rank0(f"   Warning: Could not detect sweep mode: {e}")

    # Override with command line arguments
    if args.vcf:
        config['data']['geno_file'] = args.vcf
    if args.pheno:
        config['data']['pheno_file'] = args.pheno

    # Set encoding type
    encoding_type = args.encoding_type
    config['data']['encoding_type'] = encoding_type

    # Set random seed (different seed per rank for distributed training)
    # if is_distributed:
    #     set_seed(args.seed + rank)
    # else:
    set_seed(args.seed)

    print_rank0(f"Random seed: {args.seed}")
    print_rank0(f"Encoding type: {encoding_type}")

    # Set device for distributed or single GPU training
    if is_distributed:
        device = f'cuda:{local_rank}'
    else:
        device = args.device

    # Determine output directory: command line > config > default
    if args.output:
        output_dir = Path(args.output)
    elif 'output_dir' in config:
        output_dir = Path(config['output_dir'])
    else:
        output_dir = Path('./outputs')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration (only on rank 0)
    if rank == 0:
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

    print_rank0(f"\nData Configuration:")
    print_rank0(f"  VCF file: {vcf_file}")
    print_rank0(f"  Phenotype file: {pheno_file}")
    print_rank0(f"  Encoding type: {encoding_type}")

    # Explain task assignment logic
    if classification_tasks is not None:
        print_rank0(
            f"  Classification tasks specified: {classification_tasks}")
    else:
        print_rank0(f"  All traits are treated as regression")

    # Create data loaders
    print_rank0("\n" + "=" * 80)
    print_rank0("Loading Data")
    print_rank0("=" * 80)
    print_rank0(
        f"\nData split seed: {args.data_split_seed} (keeps data split consistent across runs)")

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
        rank=rank,
        world_size=world_size,
        use_distributed_sampler=is_distributed,
        data_split_seed=args.data_split_seed,
        augmentation_config=config.get('augmentation', None),
        data_split_file=args.data_split_file,
    )
    time2 = time.time()
    print_rank0(
        f"Time taken to load/process data: {time2 - time1:.2f} seconds")

    # Print log transformation info
    if normalization_stats and 'log_transformed_tasks' in normalization_stats:
        log_transformed = normalization_stats['log_transformed_tasks']
        if log_transformed:
            print_rank0(
                f"\nLog-transformed phenotypes ({len(log_transformed)}):")
            for task in log_transformed:
                print_rank0(f"  - {task}")
        else:
            print_rank0(
                f"\nNo phenotypes required log transformation (skew threshold: {args.skew_threshold})")

    # Save normalization statistics (only on rank 0)
    if normalization_stats and rank == 0:
        import pickle
        norm_path = output_dir / 'normalization_stats.pkl'
        with open(norm_path, 'wb') as f:
            pickle.dump(normalization_stats, f)
        print_rank0(f"\nNormalization statistics saved to: {norm_path}")

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
        print_rank0(f"\nMulti-branch sequence lengths: {seq_length}")
    else:
        # Single-branch
        seq_length = sample_batch['snp'].shape[1]
        print_rank0(f"\nSequence length (variants): {seq_length}")

    if 'regression_targets' in sample_batch:
        num_regression_tasks = sample_batch['regression_targets'].shape[1]
    if 'classification_targets' in sample_batch:
        num_classification_tasks = sample_batch['classification_targets'].shape[1]

    print_rank0(f"Number of regression tasks: {num_regression_tasks}")
    print_rank0(f"Number of classification tasks: {num_classification_tasks}")
    print_rank0(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print_rank0(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print_rank0(f"Test samples: {len(test_loader.dataset)}")

    # Save data split information (sample IDs with train/valid/test flags)
    if rank == 0:
        split_data = []

        # Collect train sample IDs
        for batch in train_loader:
            if 'sample_id' in batch:
                for sample_id in batch['sample_id']:
                    split_data.append(
                        {'Sample_ID': sample_id, 'Split': 'train'})

        # Collect validation sample IDs
        if val_loader:
            for batch in val_loader:
                if 'sample_id' in batch:
                    for sample_id in batch['sample_id']:
                        split_data.append(
                            {'Sample_ID': sample_id, 'Split': 'valid'})

        # Collect test sample IDs
        if test_loader:
            for batch in test_loader:
                if 'sample_id' in batch:
                    for sample_id in batch['sample_id']:
                        split_data.append(
                            {'Sample_ID': sample_id, 'Split': 'test'})

        # Save to TSV file
        if split_data:
            split_df = pd.DataFrame(split_data)
            split_path = output_dir / 'data_split.tsv'
            split_df.to_csv(split_path, sep='\t', index=False)
            print_rank0(f"\nData split information saved to: {split_path}")
            print_rank0(f"  Total samples: {len(split_df)}")
            print_rank0(
                f"  Train: {len(split_df[split_df['Split'] == 'train'])}")
            if val_loader:
                print_rank0(
                    f"  Valid: {len(split_df[split_df['Split'] == 'valid'])}")
            if test_loader:
                print_rank0(
                    f"  Test: {len(split_df[split_df['Split'] == 'test'])}")

    # Save postprocessed data for benchmark models if requested
    if args.save_postprocess_data and rank == 0:
        save_postprocess_data(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            vcf_file=vcf_file,
            pheno_file=pheno_file,
            normalization_stats=normalization_stats,
            output_dir=output_dir,
            print_func=print_rank0
        )

    # Create model
    print_rank0("\n" + "=" * 80)
    print_rank0("Creating Model")
    print_rank0("=" * 80)

    # Generate task names if not provided
    regression_task_names = None
    classification_task_names = None

    if num_regression_tasks > 0:
        # Get task names from normalization stats if available
        if normalization_stats and 'regression_tasks' in normalization_stats:
            regression_task_names = normalization_stats['regression_tasks']
        else:
            regression_task_names = [
                f"regression_task_{i}" for i in range(num_regression_tasks)]

    if num_classification_tasks > 0:
        if classification_tasks:
            classification_task_names = classification_tasks
        else:
            classification_task_names = [
                f"classification_task_{i}" for i in range(num_classification_tasks)]

    print_rank0(f"\nTask Configuration:")
    if regression_task_names:
        print_rank0(f"  Regression tasks: {regression_task_names}")
    if classification_task_names:
        print_rank0(f"  Classification tasks: {classification_task_names}")

    # Create model from config
    model = create_model_from_config(
        config=config,
        seq_length=seq_length,
        regression_tasks=regression_task_names,
        classification_tasks=classification_task_names
    )

    # Move model to device
    model = model.to(device)

    # Initialize lazy parameters with a dummy forward pass before DDP wrapping
    if is_distributed:
        model.eval()
        with torch.no_grad():
            # Create dummy input for initialization
            if is_multi_branch:
                dummy_input = {}
                for vtype, vlen in seq_length.items():
                    dummy_input[vtype] = torch.zeros(1, vlen, 8, device=device)
            else:
                dummy_input = torch.zeros(1, seq_length, 8, device=device)

            # Run dummy forward pass to initialize all parameters
            _ = model(dummy_input)
        model.train()

        # Now wrap with DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # All parameters are used in this configuration
        )
        print_rank0(f"Model wrapped with DistributedDataParallel")

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

    # Initialize wandb (only on rank 0)
    wandb_run = None
    if args.use_wandb and rank == 0:
        try:
            import wandb
            from datetime import datetime

            # Check if already initialized (sweep mode)
            if wandb.run is None:
                # Normal run mode - initialize wandb
                # Generate run name if not provided
                wandb_name = args.wandb_name
                if wandb_name is None:
                    config_name = Path(args.config).stem
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wandb_name = f"{config_name}_{timestamp}"

                # Initialize wandb
                wandb.init(
                    project=args.wandb_project,
                    name=wandb_name,
                    tags=args.wandb_tags,
                    config={
                        'config_file': str(args.config),
                        'seed': args.seed,
                        'data_split_seed': args.data_split_seed,
                        'device': device,
                        'encoding_type': encoding_type,
                        'is_distributed': is_distributed,
                        'world_size': world_size,
                        'mixed_precision': args.mixed_precision,
                        **config  # Include all config parameters
                    },
                    dir=str(output_dir),  # Save wandb logs to output directory
                )
                wandb_run = wandb.run
                print_rank0(f"\n📊 W&B initialized: {wandb_run.url}")
            else:
                # Sweep mode - wandb already initialized
                wandb_run = wandb.run
                print_rank0(f"\n📊 W&B Sweep Run: {wandb_run.url}")
                # Update config with merged values
                wandb.config.update(config, allow_val_change=True)
        except ImportError:
            print_rank0(
                "\n⚠️  Warning: wandb not installed. Install with: pip install wandb")
            args.use_wandb = False
        except Exception as e:
            print_rank0(f"\n⚠️  Warning: Failed to initialize wandb: {e}")
            args.use_wandb = False

    # Create trainer
    print_rank0("\n" + "=" * 80)
    print_rank0("Training Setup")
    print_rank0("=" * 80)

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
        device=device,
        checkpoint_dir=str(output_dir / 'checkpoints'),
        early_stopping_patience=train_config.get(
            'early_stopping_patience', 20),
        gradient_clip_norm=train_config.get('gradient_clip_norm', 1.0),
        scheduler_type=train_config.get('scheduler_type', 'reduce_on_plateau'),
        scheduler_params=train_config.get('scheduler_params', None),
        num_epochs=train_config.get('num_epochs', 100),
        is_distributed=is_distributed,
        rank=rank,
        use_mixed_precision=args.mixed_precision,
        huber_delta=train_config.get('huber_delta', 1.0),
        wandb_run=wandb_run,
    )

    loss_type = train_config.get('loss_type', 'mse')
    print_rank0(f"\nTraining Configuration:")
    print_rank0(f"  Device: {device}")
    print_rank0(f"  Learning rate: {train_config.get('learning_rate', 1e-4)}")
    print_rank0(f"  Batch size: {train_config.get('batch_size', 32)}")
    print_rank0(f"  Max epochs: {train_config.get('num_epochs', 100)}")
    print_rank0(
        f"  Early stopping patience: {train_config.get('early_stopping_patience', 20)}")
    print_rank0(f"  Loss type: {loss_type}")
    if loss_type == 'huber':
        print_rank0(f"  Huber delta: {train_config.get('huber_delta', 1.0)}")
    print_rank0(
        f"  Uncertainty weighting: {train_config.get('uncertainty_weighting', True)}")
    print_rank0(
        f"  Scheduler type: {train_config.get('scheduler_type', 'reduce_on_plateau')}")
    print_rank0(f"  Mixed precision: {args.mixed_precision}")

    # Resume from checkpoint if requested
    if args.resume:
        checkpoint_path = output_dir / 'checkpoints' / 'latest_checkpoint.pt'
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            epoch, metrics = trainer.load_checkpoint(str(checkpoint_path))
            trainer.start_epoch = epoch
            print(f"Resuming from epoch {epoch + 1}")
            print(
                f"Best validation Pearson R so far: {trainer.best_val_r:.4f} at epoch {trainer.best_epoch}")
            print(
                f"Early stopping counter: {trainer.early_stopping.counter}/{train_config.get('early_stopping_patience', 20)}")
        else:
            print(
                "\nWarning: --resume specified but no checkpoint found. Starting fresh training.")

    # Train model
    print_rank0("\n" + "=" * 80)
    print_rank0("Starting Training")
    print_rank0("=" * 80 + "\n")

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
                    all_predictions_reg.append(
                        predictions['regression'].cpu().numpy())
                    all_targets_reg.append(batch['regression_targets'].numpy())
                    all_masks_reg.append(batch['regression_mask'].numpy())

        # Concatenate all batches
        if all_predictions_reg:
            predictions_normalized = np.concatenate(
                all_predictions_reg, axis=0)
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
                log_transformed_tasks = norm_stats.get(
                    'log_transformed_tasks', [])

                # Denormalize predictions and targets (reverse Z-score)
                predictions_log_scale = predictions_normalized.copy()
                targets_log_scale = targets_normalized.copy()

                for i in range(len(regression_means)):
                    # Apply inverse Z-score: log_scale = (normalized * std) + mean
                    predictions_log_scale[:, i] = (
                        predictions_normalized[:, i] * regression_stds[i]) + regression_means[i]
                    targets_log_scale[:, i] = (
                        targets_normalized[:, i] * regression_stds[i]) + regression_means[i]

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
                        predictions_original[:, i] = np.expm1(
                            predictions_log_scale[:, i])
                        targets_original[:, i] = np.expm1(
                            targets_log_scale[:, i])

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
                df_original.to_csv(pred_path_original, sep='\t',
                                   index=False, float_format='%.6f')
                print(
                    f"\nPredictions (original scale) saved to: {pred_path_original}")

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
                    df_log.to_csv(pred_path_log, sep='\t',
                                  index=False, float_format='%.6f')
                    print(f"Predictions (log scale) saved to: {pred_path_log}")

                # Save metrics as JSON
                metrics_path_original = output_dir / 'test_metrics_original_scale.json'
                with open(metrics_path_original, 'w') as f:
                    json.dump(test_metrics_original, f, indent=2)
                print(
                    f"Metrics (original scale) saved to: {metrics_path_original}")

                if log_transformed_tasks:
                    metrics_path_log = output_dir / 'test_metrics_log_scale.json'
                    with open(metrics_path_log, 'w') as f:
                        json.dump(test_metrics_log_scale, f, indent=2)
                    print(f"Metrics (log scale) saved to: {metrics_path_log}")
            else:
                print(
                    "\nWarning: Normalization statistics not found. Predictions saved in normalized scale.")

    print_rank0("\n" + "=" * 80)
    print_rank0("Training Complete!")
    print_rank0("=" * 80)
    print_rank0(f"\nResults saved to: {output_dir}")
    print_rank0(f"  - Checkpoints: {output_dir / 'checkpoints'}")
    print_rank0(f"  - Config: {output_dir / 'params.yaml'}")
    print_rank0(
        f"  - Training history (TSV): {output_dir / 'training_history.tsv'}")
    print_rank0(
        f"  - Training history (JSON): {output_dir / 'checkpoints' / 'training_history.json'}")

    # Cleanup distributed training
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
