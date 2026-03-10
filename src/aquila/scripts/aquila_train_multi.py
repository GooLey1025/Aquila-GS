#!/usr/bin/env python3
"""
Aquila Multi-Seed Training Script

Supports two modes:
1. Cross-Validation mode: k-fold cross-validation with multiple seeds per fold
2. Fixed Split mode: Use a pre-defined data split file with multiple seeds

Usage:
    # Cross-validation mode
    python aquila_train_multi.py --config params.yaml --n-folds 5 --n-seeds 3
    
    # Fixed split mode
    python aquila_train_multi.py --config params.yaml --data-split-file data_split.tsv --n-seeds 3

This will run training jobs in parallel across available GPUs with dynamic GPU allocation.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
import subprocess
# NOTE: Do NOT import torch at module level - import it inside functions after setting CUDA_VISIBLE_DEVICES

from aquila.utils import load_config, save_config, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Aquila multi-seed training (CV or fixed split)'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--n-folds',
        type=int,
        default=None,
        help='Number of folds for cross-validation (required for CV mode, ignored in fixed split mode)'
    )

    parser.add_argument(
        '--data-split-file',
        type=str,
        default=None,
        help='Path to pre-defined data split file (TSV format with Sample_ID and Split columns). '
             'If provided, uses fixed split mode instead of CV mode.'
    )

    parser.add_argument(
        '--n-seeds',
        type=int,
        default=3,
        help='Number of random seeds to run (default: 3)'
    )

    parser.add_argument(
        '--cv-seed',
        type=int,
        default=42,
        help='Random seed for cross-validation data splitting (only used in CV mode, default: 42)'
    )

    parser.add_argument(
        '--seed-base',
        type=int,
        default=42,
        help='Base seed for model training (seeds will be seed_base, seed_base+1, ..., seed_base+n_seeds-1) (default: 42)'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='./outputs_multi',
        help='Output directory for results (default: ./outputs_multi)'
    )

    parser.add_argument(
        '--n-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use for parallel execution (default: auto-detect)'
    )

    parser.add_argument(
        '--n-workers',
        type=int,
        default=1,
        help='Number of worker processes per GPU (default: 1, meaning one worker per GPU). '
             'Set to 2 or higher to run multiple training tasks per GPU when GPU utilization is low.'
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
        help='VCF encoding type'
    )

    parser.add_argument(
        '-mp',
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training'
    )

    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )

    parser.add_argument(
        '--wandb-project',
        type=str,
        default='aquila-gs-multi',
        help='W&B project name'
    )

    return parser.parse_args()


def detect_available_gpus() -> List[int]:
    """Detect available GPUs."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []

        num_gpus = torch.cuda.device_count()
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

        if cuda_visible is not None:
            visible_gpu_ids = [int(x.strip())
                               for x in cuda_visible.split(',') if x.strip()]
            return visible_gpu_ids

        return list(range(num_gpus))
    except ImportError:
        return []


def generate_fold_data_split_file(
    vcf_file: str,
    pheno_file: str,
    encoding_type: str,
    classification_tasks: Optional[List[str]],
    fold_idx: int,
    n_folds: int,
    cv_seed: int,
    output_file: Path
) -> None:
    """
    Generate data_split.tsv file for a specific fold using lightweight approach.
    This function does NOT import torch or create DataLoaders - it only uses sklearn KFold.
    This file will be used by aquila_train.py to load the correct train/val split.
    """
    from sklearn.model_selection import KFold
    from aquila.data_utils import parse_phenotype_file

    # Lightweight: Only read sample IDs, don't load full data
    # 1. Get sample IDs from VCF file (read header only)
    print(f"Reading sample IDs from VCF file...")
    vcf_sample_ids = None
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#CHROM'):
                fields = line.strip().split('\t')
                vcf_sample_ids = fields[9:]  # All columns after FORMAT
                print(f"  Found {len(vcf_sample_ids)} samples in VCF")
                break

    if vcf_sample_ids is None:
        raise ValueError(
            "No sample IDs found in VCF file. Make sure file has #CHROM header line.")

    # 2. Get sample IDs from phenotype file and determine valid samples
    print(f"Reading phenotype file...")
    pheno_df, regression_cols, classification_cols = parse_phenotype_file(
        pheno_path=pheno_file,
        classification_tasks=classification_tasks
    )

    # Determine valid samples (samples with at least one non-missing trait)
    # This matches the logic in VariantsDataset
    pheno_sample_ids = pheno_df['sample_id'].tolist()

    # Find samples that exist in both VCF and phenotype
    vcf_sample_set = set(vcf_sample_ids)
    pheno_sample_set = set(pheno_sample_ids)
    common_samples = sorted(list(vcf_sample_set & pheno_sample_set))

    # Filter to valid samples (have at least one non-missing trait)
    valid_samples = []
    for sample_id in common_samples:
        sample_row = pheno_df[pheno_df['sample_id'] == sample_id].iloc[0]
        # Check if sample has at least one non-missing regression or classification trait
        has_valid_trait = False
        if regression_cols:
            has_valid_trait = sample_row[regression_cols].notna().any()
        if not has_valid_trait and classification_cols:
            has_valid_trait = sample_row[classification_cols].notna().any()
        if has_valid_trait:
            valid_samples.append(sample_id)

    print(
        f"  Found {len(valid_samples)} valid samples (in both VCF and phenotype with valid traits)")

    # 3. Use sklearn KFold to generate splits (lightweight, no torch)
    n_samples = len(valid_samples)
    indices = np.arange(n_samples)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)

    # Get the specific fold
    fold_splits = list(kfold.split(indices))
    if fold_idx >= len(fold_splits):
        raise ValueError(
            f"Fold {fold_idx} not found. Total folds: {len(fold_splits)}")

    train_indices, val_indices = fold_splits[fold_idx]

    # 4. Map indices back to sample IDs
    train_sample_ids = [valid_samples[i] for i in train_indices]
    val_sample_ids = [valid_samples[i] for i in val_indices]

    # 5. Create split data
    split_data = []
    for sample_id in train_sample_ids:
        split_data.append({'Sample_ID': sample_id, 'Split': 'train'})
    for sample_id in val_sample_ids:
        split_data.append({'Sample_ID': sample_id, 'Split': 'valid'})

    # Debug: Print fold information
    train_sample_ids_set = set(train_sample_ids)
    val_sample_ids_set = set(val_sample_ids)
    print(f"  Generated data split for fold {fold_idx}:")
    print(f"    Train samples: {len(train_sample_ids_set)}")
    print(f"    Val samples: {len(val_sample_ids_set)}")
    print(
        f"    Train/Val overlap: {len(train_sample_ids_set & val_sample_ids_set)}")

    # Save to TSV file
    split_df = pd.DataFrame(split_data)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_file, sep='\t', index=False)

    # Verify: Check that train and val sets don't overlap
    train_samples = set(split_df[split_df['Split'] == 'train']['Sample_ID'])
    val_samples = set(split_df[split_df['Split'] == 'valid']['Sample_ID'])
    overlap = train_samples & val_samples
    if overlap:
        raise ValueError(
            f"Error: Train and validation sets overlap for fold {fold_idx}! "
            f"Overlapping samples: {list(overlap)[:10]}..."
        )


def worker_process(
    worker_id: int,
    gpu_queue,  # Manager().Queue() with GPU IDs for dynamic allocation
    task_queue,  # Manager().Queue() with tasks
    result_queue,  # Manager().Queue() for results
    config_file: str,
    output_dir: str,  # Convert to string for serialization
    # Pre-generated split files by task_id (as strings)
    # For CV mode: task_id = fold_idx
    # For fixed split mode: task_id = 0 (single split)
    split_files: Dict[int, str],
    vcf_file: str,
    pheno_file: str,
    encoding_type: str,
    use_mixed_precision: bool,
    use_wandb: bool,
    wandb_project: str,
    data_split_seed: int = 42,
    skew_threshold: float = 2.0,
    data_restart: bool = False,
    is_cv_mode: bool = True  # True for CV mode, False for fixed split mode
) -> None:
    """
    Worker process that runs continuously, pulling tasks from queue and executing them.
    Dynamically acquires GPU from queue for each task, ensuring balanced load distribution.
    """
    print(f"Worker {worker_id} started (PID: {os.getpid()})")

    # Convert output_dir back to Path
    output_dir = Path(output_dir)

    # Continuously process tasks from queue
    while True:
        gpu_id = None
        try:
            # Get task from queue (blocks until task is available)
            # No timeout - worker will wait until task arrives or None sentinel
            task = task_queue.get()

            # Check for sentinel value to signal worker shutdown
            if task is None:
                print(f"Worker {worker_id} shutting down")
                break

            # Dynamically acquire a GPU from the queue
            # This ensures idle GPUs are prioritized
            gpu_id = gpu_queue.get()

            # Set CUDA_VISIBLE_DEVICES BEFORE any torch imports
            # NOTE: In spawn mode, this module is re-imported, so we need to set env before torch import
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            # Extract task parameters
            # fold_idx for CV mode, 0 for fixed split mode
            task_id = task['task_id']
            seed = task['seed']
            wandb_name = task.get('wandb_name', f'task_{task_id}_seed_{seed}')
            data_split_file = Path(split_files[task_id])

            # Run the training task
            result = run_single_task_seed(
                config_file=config_file,
                task_id=task_id,
                seed=seed,
                gpu_id=gpu_id,
                output_dir=output_dir,
                data_split_file=data_split_file,
                vcf_file=vcf_file,
                pheno_file=pheno_file,
                encoding_type=encoding_type,
                use_mixed_precision=use_mixed_precision,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_name=wandb_name,
                data_split_seed=data_split_seed,
                skew_threshold=skew_threshold,
                data_restart=data_restart,
                is_cv_mode=is_cv_mode
            )

            # Put result in result queue
            result_queue.put(result)

            # Mark task as done
            task_queue.task_done()

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(
                f"Worker {worker_id} (GPU {gpu_id if gpu_id is not None else 'N/A'}) error processing task: {e}")
            print(f"  Traceback: {error_msg}")

            # Put error result in queue
            if 'task' in locals():
                result_queue.put({
                    'task_id': task.get('task_id', -1),
                    'seed': task.get('seed', -1),
                    'gpu_id': gpu_id if gpu_id is not None else -1,
                    'status': 'error',
                    'error': str(e)
                })
                task_queue.task_done()
        finally:
            # Always return GPU to queue if it was acquired
            if gpu_id is not None:
                try:
                    gpu_queue.put(gpu_id)
                except Exception:
                    print(
                        f"Warning: Worker {worker_id} could not return GPU {gpu_id} to queue")


def run_single_task_seed(
    config_file: str,
    task_id: int,  # fold_idx for CV mode, 0 for fixed split mode
    seed: int,
    gpu_id: int,
    output_dir: Path,
    data_split_file: Path,  # Pre-generated split file path
    vcf_file: str,
    pheno_file: str,
    encoding_type: str,
    use_mixed_precision: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_name: Optional[str] = None,
    data_split_seed: int = 42,
    skew_threshold: float = 2.0,
    data_restart: bool = False,
    is_cv_mode: bool = True  # True for CV mode, False for fixed split mode
) -> Dict:
    """
    Run training for a single task (fold in CV mode, or single split in fixed mode) and seed combination.
    """
    # Initialize variables that might be used in exception handling
    if is_cv_mode:
        task_output_dir = output_dir / f'fold_{task_id}' / f'seed_{seed}'
    else:
        task_output_dir = output_dir / f'seed_{seed}'

    start_time = time.time()

    # Set CUDA_VISIBLE_DEVICES BEFORE any torch imports
    # NOTE: In spawn mode, this module is re-imported, so we need to set env before torch import
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Print worker start info to console (this is from the worker process)
    if is_cv_mode:
        print(
            f"▶ Starting: Fold {task_id}, Seed {seed}, GPU {gpu_id} (PID: {os.getpid()})")
    else:
        print(
            f"▶ Starting: Seed {seed}, GPU {gpu_id} (PID: {os.getpid()})")
    print(f"  CUDA_VISIBLE_DEVICES={gpu_id}")
    print(f"  Output: {task_output_dir}")

    try:
        # Use pre-generated split file directly
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Build command to call aquila_train.py
        script_dir = Path(__file__).parent
        aquila_train_script = script_dir / 'aquila_train.py'

        # Find project root (directory containing 'src')
        project_root = script_dir.parent.parent
        while project_root.parent != project_root:
            if (project_root / 'src').exists():
                break
            project_root = project_root.parent

        # Ensure all paths are absolute
        config_file_abs = str(Path(config_file).resolve())
        vcf_file_abs = str(Path(vcf_file).resolve()
                           ) if vcf_file else vcf_file
        pheno_file_abs = str(Path(pheno_file).resolve()
                             ) if pheno_file else pheno_file
        # Use the pre-generated split file directly (no copy needed)
        data_split_file_abs = str(Path(data_split_file).resolve())

        cmd = [
            sys.executable,
            str(aquila_train_script.resolve()),
            '--config', config_file_abs,
            '--vcf', vcf_file_abs,
            '--pheno', pheno_file_abs,
            '--encoding-type', encoding_type,
            '--output', str(task_output_dir.resolve()),
            '--seed', str(seed),
            '--device', 'cuda:0',
            '--data-split-file', data_split_file_abs,
            '--data-split-seed', str(data_split_seed),
            '--skew-threshold', str(skew_threshold),
            '--data-restart',
        ]

        if use_mixed_precision:
            cmd.append('--mixed-precision')

        if use_wandb:
            cmd.append('--use-wandb')
            cmd.extend(['--wandb-project', wandb_project])
            if wandb_name:
                cmd.extend(['--wandb-name', wandb_name])

        if data_restart:
            cmd.append('--data-restart')

        # Create log file for this task/seed combination
        log_file = task_output_dir / 'training.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Run aquila_train.py from project root, redirecting output to log file
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=f,  # Redirect stdout to log file
                stderr=subprocess.STDOUT,  # Redirect stderr to same file
                text=True,
                env=os.environ.copy()  # Inherit environment (including CUDA_VISIBLE_DEVICES)
            )

        # Print summary message to console
        if result.returncode == 0:
            if is_cv_mode:
                print(
                    f"✓ Completed: Fold {task_id}, Seed {seed}, GPU {gpu_id}")
            else:
                print(f"✓ Completed: Seed {seed}, GPU {gpu_id}")
            print(f"  Log: {log_file}")
        else:
            if is_cv_mode:
                print(f"✗ Failed: Fold {task_id}, Seed {seed}, GPU {gpu_id}")
            else:
                print(f"✗ Failed: Seed {seed}, GPU {gpu_id}")
            print(f"  Log: {log_file}")
            # Print last few lines of log for quick error inspection
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"  Last 10 lines of log:")
                        for line in lines[-10:]:
                            print(f"    {line.rstrip()}")
            except Exception:
                pass

        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            # Error details are already printed above
            raise RuntimeError(
                f"aquila_train.py failed with return code {result.returncode}. See log: {log_file}")

        # Extract metrics from the output directory
        metrics = extract_metrics_from_output("", task_output_dir)

        # Summary already printed above, just add timing info
        print(f"  Time: {elapsed_time:.1f}s")
        if metrics:
            val_r = metrics.get('val_r', metrics.get('val_pearson', None))
            val_r2 = metrics.get('val_r2', None)
            print(
                f"  Best validation Pearson R: {val_r if val_r is not None else 'N/A'}")
            if val_r2 is not None:
                print(f"  Best validation R²: {val_r2:.4f}")
        else:
            print(
                f"  Warning: Could not extract metrics from {task_output_dir}")

        return {
            'task_id': task_id,
            'seed': seed,
            'gpu_id': gpu_id,
            'output_dir': str(task_output_dir),
            'metrics': metrics,
            'elapsed_time': elapsed_time,
            'status': 'success'
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        import traceback
        error_msg = traceback.format_exc()
        if is_cv_mode:
            print(f"Failed: Fold {task_id}, Seed {seed}, GPU {gpu_id}")
        else:
            print(f"Failed: Seed {seed}, GPU {gpu_id}")
        print(f"  Error: {error_msg}")

        return {
            'task_id': task_id,
            'seed': seed,
            'gpu_id': gpu_id,
            'output_dir': str(task_output_dir),
            'metrics': None,
            'elapsed_time': elapsed_time,
            'status': 'failed',
            'error': str(e)
        }


def extract_metrics_from_output(config_path: str, output_dir: Path) -> Optional[Dict]:
    """
    Extract metrics from a training output directory.
    Reads from best_metrics.json in the output_dir root.
    """
    # best_metrics.json is saved in the output_dir root
    metrics_file = output_dir / 'best_metrics.json'
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            print(f"Warning: Could not read metrics from {metrics_file}: {e}")
            return None

    # Fallback: try checkpoints subdirectory (for backward compatibility)
    metrics_file_alt = output_dir / 'checkpoints' / 'best_metrics.json'
    if metrics_file_alt.exists():
        try:
            with open(metrics_file_alt, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            print(
                f"Warning: Could not read metrics from {metrics_file_alt}: {e}")
            return None

    return None


def collect_all_results(results: List[Dict], output_dir: Path, is_cv_mode: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect all results and compute averages."""
    results_data = []

    for result in results:
        if result['status'] == 'success':
            metrics = result.get('metrics', {})
            if metrics:
                # Handle both val_r and val_pearson (val_r is preferred, val_pearson is fallback)
                val_r = metrics.get('val_r', None)
                if val_r is None:
                    val_r = metrics.get('val_pearson', None)

                result_row = {
                    'seed': result['seed'],
                    'gpu_id': result['gpu_id'],
                    'val_r': val_r,
                    'val_r2': metrics.get('val_r2', None),
                    'val_loss': metrics.get('val_loss', None),
                    'elapsed_time': result.get('elapsed_time', None),
                    'output_dir': result.get('output_dir', '')
                }

                # Add fold column only in CV mode
                if is_cv_mode:
                    result_row['fold'] = result['task_id']

                results_data.append(result_row)
            else:
                task_desc = f"Fold {result['task_id']}, Seed {result['seed']}" if is_cv_mode else f"Seed {result['seed']}"
                print(
                    f"Warning: {task_desc} completed but no metrics found")
                print(f"  Output dir: {result.get('output_dir', 'N/A')}")
        else:
            task_desc = f"Fold {result.get('task_id', '?')}, Seed {result.get('seed', '?')}" if is_cv_mode else f"Seed {result.get('seed', '?')}"
            print(
                f"Warning: {task_desc} failed: {result.get('error', 'Unknown error')}")

    if not results_data:
        print("Warning: No successful results to collect")
        print(f"Total results received: {len(results)}")
        for i, result in enumerate(results):
            print(
                f"  Result {i+1}: status={result.get('status')}, metrics={'present' if result.get('metrics') else 'missing'}")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(results_data)

    # Save detailed results
    results_file = output_dir / 'results_detailed.tsv'
    df.to_csv(results_file, sep='\t', index=False, float_format='%.6f')
    print(f"\nDetailed results saved to: {results_file}")

    # Compute averages
    summary_data = []

    if is_cv_mode:
        # Per-fold averages
        for fold_idx in sorted(df['fold'].unique()):
            fold_df = df[df['fold'] == fold_idx]
            summary_data.append({
                'type': f'fold_{fold_idx}',
                'n_runs': len(fold_df),
                'val_r_mean': fold_df['val_r'].mean() if 'val_r' in fold_df.columns and fold_df['val_r'].notna().any() else None,
                'val_r_std': fold_df['val_r'].std() if 'val_r' in fold_df.columns and fold_df['val_r'].notna().any() else None,
                'val_r2_mean': fold_df['val_r2'].mean() if 'val_r2' in fold_df.columns and fold_df['val_r2'].notna().any() else None,
                'val_r2_std': fold_df['val_r2'].std() if 'val_r2' in fold_df.columns and fold_df['val_r2'].notna().any() else None,
            })

    # Overall average
    summary_data.append({
        'type': 'overall',
        'n_runs': len(df),
        'val_r_mean': df['val_r'].mean() if 'val_r' in df.columns and df['val_r'].notna().any() else None,
        'val_r_std': df['val_r'].std() if 'val_r' in df.columns and df['val_r'].notna().any() else None,
        'val_r2_mean': df['val_r2'].mean() if 'val_r2' in df.columns and df['val_r2'].notna().any() else None,
        'val_r2_std': df['val_r2'].std() if 'val_r2' in df.columns and df['val_r2'].notna().any() else None,
    })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_file = output_dir / 'results_summary.tsv'
    summary_df.to_csv(summary_file, sep='\t', index=False, float_format='%.6f')
    print(f"Summary results saved to: {summary_file}")

    return df, summary_df


def main():
    """Main function supporting both CV and fixed split modes."""
    # CRITICAL: Set multiprocessing start method BEFORE creating any Manager or ProcessPoolExecutor
    # This must be done early, before any multiprocessing objects are created
    # 'spawn' is required for CUDA compatibility (avoids fork issues with CUDA contexts)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError as e:
        # If start method was already set, check if it's 'spawn'
        current_method = mp.get_start_method()
        if current_method != 'spawn':
            print(
                f"Warning: Could not set start method to 'spawn'. Current method: {current_method}")
            print(f"  Error: {e}")
            print(
                "  This may cause issues with CUDA in multiprocessing. Continuing anyway...")
        else:
            # Already set to 'spawn', which is fine
            pass

    args = parse_args()

    # Determine mode: CV mode or fixed split mode
    is_cv_mode = args.data_split_file is None

    if is_cv_mode:
        if args.n_folds is None:
            print(
                "Error: --n-folds is required when --data-split-file is not provided (CV mode)")
            return
        print("=" * 80)
        print("AQUILA: Cross-Validation Training (Multi-Seed)")
        print("=" * 80)
    else:
        if args.n_folds is not None:
            print(
                "Warning: --n-folds is ignored when --data-split-file is provided (fixed split mode)")
        print("=" * 80)
        print("AQUILA: Fixed Split Training (Multi-Seed)")
        print("=" * 80)

    # Convert config file path to absolute path
    config_file_path = Path(args.config)
    if not config_file_path.is_absolute():
        # If relative path, resolve relative to current working directory
        config_file_path = config_file_path.resolve()
    config_file_abs = str(config_file_path)

    # Load configuration
    config = load_config(config_file_abs)

    # Override config with command line arguments
    if args.vcf:
        config['data']['geno_file'] = args.vcf
    if args.pheno:
        config['data']['pheno_file'] = args.pheno
    config['data']['encoding_type'] = args.encoding_type

    # Get data paths and convert to absolute paths
    vcf_file = args.vcf or config['data']['geno_file']
    pheno_file = args.pheno or config['data']['pheno_file']

    # Convert data file paths to absolute paths if they are relative
    if vcf_file and not Path(vcf_file).is_absolute():
        vcf_file = str(Path(vcf_file).resolve())
    if pheno_file and not Path(pheno_file).is_absolute():
        pheno_file = str(Path(pheno_file).resolve())

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, output_dir / 'config.yaml')

    # Detect available GPUs
    available_gpus = detect_available_gpus()
    if not available_gpus:
        print("Error: No GPUs detected")
        return

    n_gpus = args.n_gpus if args.n_gpus else len(available_gpus)
    n_gpus = min(n_gpus, len(available_gpus))

    # Calculate total number of workers
    # n_workers is the number of workers per GPU
    n_workers_per_gpu = args.n_workers
    total_workers = n_workers_per_gpu * n_gpus

    print(f"\nConfiguration:")
    print(f"  Config file: {config_file_abs}")
    if is_cv_mode:
        print(f"  Mode: Cross-Validation")
        print(f"  Number of folds: {args.n_folds}")
        print(f"  Number of seeds per fold: {args.n_seeds}")
        print(f"  Total training jobs: {args.n_folds * args.n_seeds}")
    else:
        print(f"  Mode: Fixed Split")
        print(f"  Data split file: {args.data_split_file}")
        print(f"  Number of seeds: {args.n_seeds}")
        print(f"  Total training jobs: {args.n_seeds}")
    print(f"  Available GPUs: {available_gpus}")
    print(f"  Using GPUs: {n_gpus}")
    print(f"  Workers per GPU: {n_workers_per_gpu}")
    print(f"  Total worker processes: {total_workers}")
    print(f"  Output directory: {output_dir}")

    # Prepare split files
    split_files = {}

    if is_cv_mode:
        # CV mode: Generate split files for all folds
        print(f"\n{'=' * 80}")
        print("Generating data split files for all folds (main process)")
        print(f"{'=' * 80}")

        classification_tasks = config['data'].get('classification_tasks', None)

        # Generate split files for all folds
        for fold_idx in range(args.n_folds):
            fold_output_dir = output_dir / f'fold_{fold_idx}'
            fold_output_dir.mkdir(parents=True, exist_ok=True)

            # Create a temporary seed_0 directory for the split file (will be reused by all seeds)
            temp_seed_dir = fold_output_dir / 'seed_0'
            temp_seed_dir.mkdir(parents=True, exist_ok=True)
            data_split_file = temp_seed_dir / 'data_split.tsv'

            print(f"Generating split file for fold {fold_idx}...")
            generate_fold_data_split_file(
                vcf_file=vcf_file,
                pheno_file=pheno_file,
                encoding_type=args.encoding_type,
                classification_tasks=classification_tasks,
                fold_idx=fold_idx,
                n_folds=args.n_folds,
                cv_seed=args.cv_seed,
                output_file=data_split_file
            )
            split_files[fold_idx] = data_split_file
            print(f"  ✓ Saved to: {data_split_file}")

        print(f"\nAll {args.n_folds} split files generated successfully!")
    else:
        # Fixed split mode: Use provided data split file
        print(f"\n{'=' * 80}")
        print("Using provided data split file")
        print(f"{'=' * 80}")

        data_split_file_path = Path(args.data_split_file)
        if not data_split_file_path.is_absolute():
            data_split_file_path = data_split_file_path.resolve()

        if not data_split_file_path.exists():
            print(f"Error: Data split file not found: {data_split_file_path}")
            return

        # Copy split file to output directory for reference
        output_split_file = output_dir / 'data_split.tsv'
        import shutil
        shutil.copy2(data_split_file_path, output_split_file)
        print(f"  ✓ Using data split file: {data_split_file_path}")
        print(f"  ✓ Copied to output directory: {output_split_file}")

        # For fixed split mode, task_id is always 0
        split_files[0] = str(data_split_file_path)

    # Prepare all training tasks
    tasks = []
    if is_cv_mode:
        for fold_idx in range(args.n_folds):
            for seed_idx in range(args.n_seeds):
                seed = args.seed_base + seed_idx
                wandb_name = f'fold_{fold_idx}_seed_{seed}'
                tasks.append({
                    'task_id': fold_idx,
                    'seed': seed,
                    'wandb_name': wandb_name
                })
    else:
        for seed_idx in range(args.n_seeds):
            seed = args.seed_base + seed_idx
            wandb_name = f'seed_{seed}'
            tasks.append({
                'task_id': 0,  # Always 0 for fixed split mode
                'seed': seed,
                'wandb_name': wandb_name
            })

    print(f"\n{'=' * 80}")
    if is_cv_mode:
        print("Starting Cross-Validation Training")
    else:
        print("Starting Fixed Split Training")
    print(f"{'=' * 80}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Workers per GPU: {n_workers_per_gpu}")
    print(f"Total worker processes: {total_workers}")
    print(f"Dynamic GPU allocation: Workers will acquire GPUs dynamically from queue")
    print(f"  - Idle GPUs (0 tasks) are prioritized")
    print(f"  - After all GPUs have at least 1 task, additional tasks use --n-workers limit")

    # Create queues for task distribution, result collection, and GPU management
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    gpu_queue = manager.Queue()

    # Initialize GPU queue: Put each GPU ID into queue n_workers_per_gpu times
    # This allows up to n_workers_per_gpu concurrent tasks per GPU
    # Example: 3 GPUs, n_workers_per_gpu=2 -> queue: [GPU0, GPU0, GPU1, GPU1, GPU2, GPU2]
    gpu_list = available_gpus[:n_gpus]
    for gpu_id in gpu_list:
        for _ in range(n_workers_per_gpu):
            gpu_queue.put(gpu_id)

    print(
        f"  GPU queue initialized: {n_gpus} GPUs × {n_workers_per_gpu} slots = {gpu_queue.qsize()} total slots")

    # Put all tasks into the task queue
    for task in tasks:
        task_queue.put(task)

    # Convert split_files Path objects to strings for serialization
    split_files_str = {task_id: str(path) if isinstance(path, Path) else path
                       for task_id, path in split_files.items()}

    # Start worker processes (they will dynamically acquire GPUs from queue)
    workers = []
    for worker_id in range(total_workers):
        worker = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                gpu_queue,  # Pass GPU queue for dynamic allocation
                task_queue,
                result_queue,
                config_file_abs,
                str(output_dir),  # Convert to string for serialization
                split_files_str,  # Use string version
                vcf_file,
                pheno_file,
                args.encoding_type,
                args.mixed_precision,
                args.use_wandb,
                args.wandb_project,
                args.cv_seed,
                config.get('train', {}).get('skew_threshold', 2.0),
                False,  # data_restart
                is_cv_mode  # Pass mode flag
            )
        )
        worker.start()
        workers.append(worker)
        print(f"Started worker {worker_id} (PID: {worker.pid})")

    # Run jobs and collect results
    start_time = time.time()
    results = []
    completed_tasks = 0
    total_tasks = len(tasks)

    # Collect results as workers complete tasks
    # Block until all tasks are completed (no timeout - wait indefinitely)
    while completed_tasks < total_tasks:
        # Block indefinitely until a result is available
        # This is fine since workers will eventually complete their training tasks
        result = result_queue.get()
        results.append(result)
        completed_tasks += 1
        elapsed = time.time() - start_time
        print(
            f"Progress: {completed_tasks}/{total_tasks} tasks completed (elapsed: {elapsed/60:.1f} min)")

    # Signal workers to shutdown by putting None sentinels
    for _ in range(total_workers):
        task_queue.put(None)

    # Wait for all workers to finish
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            print(f"Warning: Worker {worker.pid} did not terminate gracefully")
        else:
            print(f"Worker {worker.pid} terminated successfully")

    total_time = time.time() - start_time

    # Collect and summarize results
    print(f"\n{'=' * 80}")
    if is_cv_mode:
        print("Cross-Validation Results")
    else:
        print("Fixed Split Training Results")
    print(f"{'=' * 80}")

    df, summary_df = collect_all_results(
        results, output_dir, is_cv_mode=is_cv_mode)

    if not df.empty:
        # Print summary
        print(f"\nSummary:")
        print(summary_df.to_string(index=False))

    print(
        f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {output_dir}")

    print(f"\n{'=' * 80}")
    if is_cv_mode:
        print("Cross-Validation Complete!")
    else:
        print("Fixed Split Training Complete!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
