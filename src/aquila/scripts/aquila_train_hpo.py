#!/usr/bin/env python3
"""
Aquila Hyperparameter Optimization Script

Runs Optuna-based hyperparameter optimization with parallel GPU support.
Each trial runs as a separate subprocess to ensure proper GPU isolation.

Usage:
    python aquila_train_hpo.py --config params.yaml -hpo
    
    # Specify GPU configuration
    python aquila_train_hpo.py --config params.yaml -hpo --n-gpus 4 --n-workers 2

This will run HPO trials in parallel across available GPUs with dynamic GPU allocation.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time
import yaml
from typing import List, Dict, Optional
import multiprocessing as mp
import subprocess
import pandas as pd
# NOTE: Do NOT import torch at module level - import it inside functions after setting CUDA_VISIBLE_DEVICES

from aquila.utils import load_config, save_config
from aquila.hpo import load_hpo_config, create_optuna_study, suggest_hyperparameters, save_optuna_results, merge_optuna_config
from aquila.data_utils import parse_phenotype_file
import optuna


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Aquila hyperparameter optimization with parallel GPU support'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file with HPO section'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='./outputs_hpo',
        help='Output directory for HPO results (default: ./outputs_hpo)'
    )

    parser.add_argument(
        '--n-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use for parallel trials (default: auto-detect all available GPUs)'
    )

    parser.add_argument(
        '--n-workers',
        type=int,
        default=1,
        help='Number of worker processes per GPU (default: 1, meaning one trial per GPU at a time). '
             'Set to 2 or higher to run multiple trials per GPU when GPU utilization is low.'
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
        default='diploid_onehot',
        choices=['token', 'diploid_onehot', 'snp_vcf', 'indel_vcf', 'sv_vcf', 'snp_indel_vcf', 'snp_indel_sv_vcf'],
        help='Encoding type'
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
        default='aquila-gs-hpo',
        help='W&B project name'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '-dss',
        '--data-split-seed',
        type=int,
        default=42,
        help='Random seed for dataset splitting'
    )

    parser.add_argument(
        '-dsf',
        '--data-split-file',
        type=str,
        default=None,
        help='Path to TSV file with columns Sample_ID and Split (train/valid/test). '
             'If provided, will use this file to determine data splits instead of random splitting.'
    )

    parser.add_argument(
        '-st',
        '--skew-threshold',
        type=float,
        default=2.0,
        help='Skewness threshold for auto log transformation'
    )

    parser.add_argument(
        '-dr',
        '--data-restart',
        action='store_true',
        help='Ignore data cache and re-process data from scratch'
    )

    parser.add_argument(
        '-rst',
        '--restart',
        action='store_true',
        help='Delete existing Optuna study and start fresh. This will remove all previous trials.'
    )

    return parser.parse_args()


def _generate_per_phenotype_metrics(study, output_dir: Path, pheno_file: str, base_config: Dict):
    """
    Generate metrics_per_phenotype.tsv from the best trial's best_metrics.json.

    Args:
        study: Optuna study object
        output_dir: Output directory for HPO results
        pheno_file: Path to phenotype file
        base_config: Base configuration dict
    """
    if not study.trials:
        print("No trials found, skipping per-phenotype metrics generation")
        return

    # Find best trial
    completed_trials = [t for t in study.trials if t.state ==
                        optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("No completed trials found, skipping per-phenotype metrics generation")
        return

    best_trial = study.best_trial
    best_trial_number = best_trial.number

    # Load best_metrics.json from best trial
    best_trial_dir = output_dir / f"trial_{best_trial_number}"
    metrics_file = best_trial_dir / 'best_metrics.json'
    if not metrics_file.exists():
        # Try checkpoints directory
        metrics_file = best_trial_dir / 'checkpoints' / 'best_metrics.json'

    if not metrics_file.exists():
        print(
            f"Warning: best_metrics.json not found for best trial {best_trial_number}")
        return

    with open(metrics_file, 'r') as f:
        best_metrics = json.load(f)

    # Try multiple methods to get phenotype names
    phenotype_names = []

    # Method 1: Try to load from normalization_stats.pkl (most reliable)
    try:
        norm_stats_file = best_trial_dir / 'normalization_stats.pkl'
        if not norm_stats_file.exists():
            norm_stats_file = best_trial_dir / 'checkpoints' / 'normalization_stats.pkl'

        if norm_stats_file.exists():
            import pickle
            with open(norm_stats_file, 'rb') as f:
                norm_stats = pickle.load(f)
            if 'regression_tasks' in norm_stats:
                phenotype_names = norm_stats['regression_tasks'].copy()
                # Also check for classification tasks if they exist
                if 'classification_tasks' in norm_stats and norm_stats['classification_tasks']:
                    phenotype_names.extend(norm_stats['classification_tasks'])
                if phenotype_names:
                    print(
                        f"✓ Loaded {len(phenotype_names)} phenotype names from normalization_stats.pkl")
                    print(
                        f"  Phenotypes: {phenotype_names[:5]}{'...' if len(phenotype_names) > 5 else ''}")
    except Exception as e:
        import traceback
        print(
            f"Warning: Could not load phenotype names from normalization_stats.pkl: {e}")
        print(f"  Traceback: {traceback.format_exc()}")

    # Method 2: Try to parse from phenotype file
    if not phenotype_names:
        try:
            # Ensure pheno_file is absolute path
            pheno_path = Path(pheno_file)
            if not pheno_path.is_absolute():
                # Try relative to output_dir or current working directory
                possible_paths = [
                    output_dir.parent / pheno_file,
                    Path.cwd() / pheno_file,
                    Path(pheno_file).resolve()
                ]
                for pp in possible_paths:
                    if pp.exists():
                        pheno_file = str(pp)
                        break

            if Path(pheno_file).exists():
                _, regression_cols, classification_cols = parse_phenotype_file(
                    pheno_file,
                    classification_tasks=base_config.get(
                        'data', {}).get('classification_tasks', None)
                )
                # Combine regression and classification tasks
                phenotype_names = regression_cols + classification_cols
                print(
                    f"Loaded {len(phenotype_names)} phenotype names from phenotype file: {pheno_file}")
            else:
                print(f"Warning: Phenotype file not found: {pheno_file}")
        except Exception as e:
            import traceback
            print(
                f"Warning: Could not parse phenotype file to get phenotype names: {e}")
            print(f"  Traceback: {traceback.format_exc()}")

    # Method 3: Fallback: infer from metrics keys (least reliable)
    if not phenotype_names:
        print("Warning: Using fallback method to infer phenotype names from metrics keys")
        task_indices = set()
        for key in best_metrics.keys():
            if key.startswith('val_reg_task_'):
                # Extract task index from key like "val_reg_task_0_pearson" or "val_reg_task_0_r2"
                parts = key.split('_')
                if len(parts) >= 4 and parts[3].isdigit():
                    task_indices.add(int(parts[3]))
        # Sort by task index and create phenotype names
        phenotype_names = [f"task_{idx}" for idx in sorted(task_indices)]
        print(
            f"Inferred {len(phenotype_names)} phenotype names from metrics keys")

    if not phenotype_names:
        print(
            "Warning: No phenotype names found, skipping per-phenotype metrics generation")
        return

    # Extract metrics for each phenotype
    results = []
    for i, pheno_name in enumerate(phenotype_names):
        result = {'phenotype': pheno_name}

        # Try to extract metrics for this task
        # For regression tasks
        task_key_prefix = f"val_reg_task_{i}"
        train_task_key_prefix = f"train_reg_task_{i}"

        # Validation metrics
        result['valid_r'] = best_metrics.get(f'{task_key_prefix}_pearson',
                                             best_metrics.get('val_r', None))
        result['valid_r2'] = best_metrics.get(f'{task_key_prefix}_r2',
                                              best_metrics.get('val_r2', None))
        result['valid_mse'] = best_metrics.get(f'{task_key_prefix}_mse', None)
        result['valid_rmse'] = best_metrics.get(
            f'{task_key_prefix}_rmse', None)
        result['valid_mae'] = best_metrics.get(f'{task_key_prefix}_mae', None)

        # Training metrics
        result['train_r'] = best_metrics.get(f'{train_task_key_prefix}_pearson',
                                             best_metrics.get('train_r', None))
        result['train_r2'] = best_metrics.get(f'{train_task_key_prefix}_r2',
                                              best_metrics.get('train_r2', None))
        result['train_mse'] = best_metrics.get(
            f'{train_task_key_prefix}_mse', None)
        result['train_rmse'] = best_metrics.get(
            f'{train_task_key_prefix}_rmse', None)
        result['train_mae'] = best_metrics.get(
            f'{train_task_key_prefix}_mae', None)

        # Add best hyperparameters
        for key, value in study.best_params.items():
            result[key] = value

        result['trial_number'] = best_trial_number
        result['status'] = 'success'

        results.append(result)

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    metrics_per_phenotype_file = output_dir / 'metrics_per_phenotype.tsv'
    results_df.to_csv(metrics_per_phenotype_file, sep='\t', index=False)

    print(f"\nPer-phenotype metrics saved to: {metrics_per_phenotype_file}")

    # Print summary
    successful = results_df[results_df['status'] == 'success']
    if len(successful) > 0:
        print(
            f"\nPer-phenotype summary (from best trial {best_trial_number}):")
        print(f"  Total phenotypes: {len(results_df)}")
        if 'valid_r' in successful.columns:
            valid_r_mean = successful['valid_r'].mean()
            valid_r_median = successful['valid_r'].median()
            print(f"  Average Valid R: {valid_r_mean:.4f}")
            print(f"  Median Valid R: {valid_r_median:.4f}")


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


def worker_process(
    worker_id: int,
    gpu_queue: mp.Queue,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    study_name: str,
    storage_url: str,
    hpo_config: Dict,
    base_config: Dict,
    base_output_dir: Path,
    config_file: str,
    vcf_file: str,
    pheno_file: str,
    encoding_type: str,
    use_mixed_precision: bool,
    use_wandb: bool,
    wandb_project: str,
    seed: int,
    data_split_seed: int,
    data_split_file: Optional[str],
    skew_threshold: float,
    data_restart: bool
) -> None:
    """
    Worker process that runs HPO trials in parallel.
    Each trial runs as a subprocess to ensure proper GPU isolation.
    """
    print(f"Worker {worker_id} started (PID: {os.getpid()})")

    # Create a study instance for this worker (connects to shared storage)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )

    # Find project root and aquila_train.py script
    script_dir = Path(__file__).parent
    aquila_train_script = script_dir / 'aquila_train.py'
    project_root = script_dir.parent.parent
    while project_root.parent != project_root:
        if (project_root / 'src').exists():
            break
        project_root = project_root.parent

    # Continuously process trials from queue
    while True:
        gpu_id = None
        try:
            # Get task from queue (blocks until task is available)
            task = task_queue.get()
            if task is None:
                # Sentinel value - shutdown worker
                break

            # Dynamically acquire a GPU from the queue
            gpu_id = gpu_queue.get()

            # Set CUDA_VISIBLE_DEVICES BEFORE any torch imports
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            # Ask for a trial from the study
            try:
                trial = study.ask()
            except Exception as e:
                # No more trials available
                print(f"Worker {worker_id}: No more trials available: {e}")
                # Return GPU and break
                gpu_queue.put(gpu_id)
                break

            trial_number = trial.number
            print(
                f"Worker {worker_id} (GPU {gpu_id}): Starting trial {trial_number}")

            # Suggest hyperparameters
            trial_params = suggest_hyperparameters(trial, hpo_config)

            print(
                f"Worker {worker_id} (GPU {gpu_id}): Trial {trial_number} hyperparameters:")
            for key, value in trial_params.items():
                print(f"  {key}: {value}")

            # Merge trial parameters into base config
            trial_config = merge_optuna_config(base_config, trial_params)

            # Create trial-specific output directory
            trial_output_dir = base_output_dir / f"trial_{trial_number}"
            trial_output_dir.mkdir(parents=True, exist_ok=True)

            # Save trial config to a temporary file
            trial_config_file = trial_output_dir / 'trial_config.yaml'
            save_config(trial_config, trial_config_file)

            # Build command to call aquila_train.py
            cmd = [
                sys.executable,
                str(aquila_train_script.resolve()),
                '--config', str(trial_config_file.resolve()),
                '--vcf', vcf_file,
                '--pheno', pheno_file,
                '--encoding-type', encoding_type,
                '--output', str(trial_output_dir.resolve()),
                '--seed', str(seed),
                '--device', 'cuda:0',  # Use cuda:0 since CUDA_VISIBLE_DEVICES is set
                '--data-split-seed', str(data_split_seed),
                '--skew-threshold', str(skew_threshold),
            ]

            # Add data split file if provided
            if data_split_file:
                cmd.extend(['--data-split-file', data_split_file])

            if use_mixed_precision:
                cmd.append('--mixed-precision')

            if use_wandb:
                cmd.append('--use-wandb')
                cmd.extend(['--wandb-project', wandb_project])
                cmd.extend(['--wandb-name', f'hpo_trial_{trial_number}'])

            if data_restart:
                cmd.append('--data-restart')

            # Create log file for this trial
            log_file = trial_output_dir / 'training.log'
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Run aquila_train.py as subprocess
            start_time = time.time()
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=str(project_root),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=os.environ.copy()  # Inherit environment (including CUDA_VISIBLE_DEVICES)
                )

            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                # Extract metrics from the output directory
                metrics_file = trial_output_dir / 'best_metrics.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    best_valid_r = metrics.get(
                        'val_r', metrics.get('val_pearson', 0.0))
                else:
                    # Fallback: try checkpoints directory
                    metrics_file_alt = trial_output_dir / 'checkpoints' / 'best_metrics.json'
                    if metrics_file_alt.exists():
                        with open(metrics_file_alt, 'r') as f:
                            metrics = json.load(f)
                        best_valid_r = metrics.get(
                            'val_r', metrics.get('val_pearson', 0.0))
                    else:
                        print(
                            f"Worker {worker_id} (GPU {gpu_id}): Warning: No metrics file found for trial {trial_number}")
                        best_valid_r = 0.0

                print(
                    f"Worker {worker_id} (GPU {gpu_id}): Trial {trial_number} completed: val_r={best_valid_r:.6f} (time: {elapsed_time/60:.1f} min)")

                # Report result to study
                study.tell(trial, best_valid_r)

                # Put success result in queue
                result_queue.put({
                    'trial_number': trial_number,
                    'status': 'success',
                    'value': best_valid_r,
                    'elapsed_time': elapsed_time
                })
            else:
                print(
                    f"Worker {worker_id} (GPU {gpu_id}): Trial {trial_number} failed (return code: {result.returncode})")
                print(f"  Log: {log_file}")

                # Report failure to study
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

                # Put failure result in queue
                result_queue.put({
                    'trial_number': trial_number,
                    'status': 'failed',
                    'elapsed_time': elapsed_time
                })

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(
                f"Worker {worker_id} (GPU {gpu_id if gpu_id is not None else 'N/A'}) error: {e}")
            print(f"  Traceback: {error_msg}")
            if 'trial' in locals():
                try:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    result_queue.put({
                        'trial_number': trial.number,
                        'status': 'failed',
                        'error': str(e)
                    })
                except:
                    pass
        finally:
            # Always return GPU to queue if it was acquired
            if gpu_id is not None:
                try:
                    gpu_queue.put(gpu_id)
                except Exception:
                    print(
                        f"Warning: Worker {worker_id} could not return GPU {gpu_id} to queue")

    print(f"Worker {worker_id} shutting down")


def main():
    """Main function for HPO with parallel GPU support."""
    # CRITICAL: Set multiprocessing start method BEFORE creating any Manager
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        current_method = mp.get_start_method()
        if current_method != 'spawn':
            print(
                f"Warning: Could not set start method to 'spawn'. Current method: {current_method}")
            print("  This may cause issues with CUDA in multiprocessing.")

    args = parse_args()

    print("=" * 80)
    print("AQUILA: Hyperparameter Optimization (Parallel)")
    print("=" * 80)

    # Load configuration
    config_file_path = Path(args.config)
    if not config_file_path.is_absolute():
        config_file_path = config_file_path.resolve()
    config_file_abs = str(config_file_path)

    print(f"\nLoading configuration from: {config_file_abs}")
    config = load_config(config_file_abs)

    # Check HPO configuration
    hpo_config = config.get('hpo')
    if not hpo_config:
        print("\n❌ Error: 'hpo' section not found in config file.")
        print("   Please add an 'hpo' section to your config file with hyperparameter search space.")
        return

    try:
        hpo_config_normalized = load_hpo_config(hpo_config)
    except ValueError as e:
        print(f"\n❌ Error loading HPO configuration: {e}")
        return

    # Override config with command line arguments
    if args.vcf:
        config['data']['geno_file'] = args.vcf
    if args.pheno:
        config['data']['pheno_file'] = args.pheno
    config['data']['encoding_type'] = args.encoding_type

    # Get data paths
    vcf_file = args.vcf or config['data']['geno_file']
    pheno_file = args.pheno or config['data']['pheno_file']

    # Convert data file paths to absolute paths if they are relative
    if vcf_file and not Path(vcf_file).is_absolute():
        vcf_file = str(Path(vcf_file).resolve())
    if pheno_file and not Path(pheno_file).is_absolute():
        pheno_file = str(Path(pheno_file).resolve())

    # Convert data split file to absolute path if provided
    data_split_file = None
    if args.data_split_file:
        data_split_file_path = Path(args.data_split_file)
        if not data_split_file_path.is_absolute():
            data_split_file_path = data_split_file_path.resolve()
        data_split_file = str(data_split_file_path)

    # Remove hpo section from base config (will be merged per trial)
    base_config = {k: v for k, v in config.items() if k != 'hpo'}

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save base configuration
    save_config(config, output_dir / 'base_config.yaml')

    # Detect available GPUs
    available_gpus = detect_available_gpus()
    if not available_gpus:
        print("Error: No GPUs detected")
        return

    n_gpus = args.n_gpus if args.n_gpus else len(available_gpus)
    n_gpus = min(n_gpus, len(available_gpus))

    # Calculate total number of workers
    n_workers_per_gpu = args.n_workers
    total_workers = n_workers_per_gpu * n_gpus

    n_trials = hpo_config_normalized.get('n_trials', 100)
    study_name = f"aquila_hpo_{output_dir.name}"

    # Setup storage (required for parallel execution)
    storage_file = output_dir / 'optuna_study.db'
    storage_url = f'sqlite:///{storage_file}'

    print(f"\nHPO Configuration:")
    print(f"  Config file: {config_file_abs}")
    print(f"  Number of trials: {n_trials}")
    print(
        f"  Optimization direction: {hpo_config_normalized.get('direction', 'maximize')}")
    print(f"  Metric: {hpo_config_normalized.get('metric', 'best/val_r')}")
    print(
        f"  Parameters to optimize: {len(hpo_config_normalized['parameters'])}")
    print(f"  Available GPUs: {available_gpus}")
    print(f"  Using GPUs: {available_gpus[:n_gpus]}")
    print(f"  Workers per GPU: {n_workers_per_gpu}")
    print(f"  Total worker processes: {total_workers}")
    print(f"  Output directory: {output_dir}")
    print(f"  Study name: {study_name}")
    print(f"  Storage: {storage_url}")
    if data_split_file:
        print(f"  Data split file: {data_split_file} (fixed split)")
    else:
        print(f"  Data split seed: {args.data_split_seed} (random split)")

    # Handle restart: delete existing study if requested
    if args.restart:
        print(f"\n⚠️  Restart requested: Deleting existing Optuna study '{study_name}'")
        if storage_file.exists():
            storage_file.unlink()
            print(f"  Deleted: {storage_file}")
        else:
            print(f"  Storage file not found: {storage_file}")

    # Create study
    study = create_optuna_study(
        hpo_config=hpo_config_normalized,
        study_name=study_name,
        storage=storage_url
    )

    # Create queues for task distribution, result collection, and GPU management
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    gpu_queue = manager.Queue()

    # Initialize GPU queue: Put each GPU ID into queue n_workers_per_gpu times
    gpu_list = available_gpus[:n_gpus]
    for gpu_id in gpu_list:
        for _ in range(n_workers_per_gpu):
            gpu_queue.put(gpu_id)

    print(
        f"\n  GPU queue initialized: {n_gpus} GPUs × {n_workers_per_gpu} slots = {gpu_queue.qsize()} total slots")

    # Prepare all trial tasks (just trial numbers for tracking)
    for trial_idx in range(n_trials):
        task_queue.put(trial_idx)

    # Put sentinel values to signal workers to shutdown
    for _ in range(total_workers):
        task_queue.put(None)

    # Start worker processes
    workers = []
    for worker_id in range(total_workers):
        worker = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                gpu_queue,
                task_queue,
                result_queue,
                study_name,
                storage_url,
                hpo_config_normalized,
                base_config,
                output_dir,
                config_file_abs,
                vcf_file,
                pheno_file,
                args.encoding_type,
                args.mixed_precision,
                args.use_wandb,
                args.wandb_project,
                args.seed,
                args.data_split_seed,
                data_split_file,  # Use converted absolute path
                args.skew_threshold,
                args.data_restart
            )
        )
        worker.start()
        workers.append(worker)
        print(f"Started worker {worker_id} (PID: {worker.pid})")

    # Collect results and monitor progress
    start_time = time.time()
    completed_trials = 0

    print(f"\n{'=' * 80}")
    print("Starting HPO Trials")
    print(f"{'=' * 80}\n")

    while completed_trials < n_trials:
        try:
            result = result_queue.get(timeout=600)  # 10 minute timeout
            completed_trials += 1
            elapsed = time.time() - start_time

            if result['status'] == 'success':
                print(
                    f"Progress: {completed_trials}/{n_trials} trials completed (elapsed: {elapsed/60:.1f} min)")
                print(
                    f"  Trial {result['trial_number']} val_r: {result['value']:.6f} (time: {result['elapsed_time']/60:.1f} min)")
            elif result['status'] == 'failed':
                print(
                    f"Progress: {completed_trials}/{n_trials} trials completed (elapsed: {elapsed/60:.1f} min)")
                print(
                    f"  Trial {result['trial_number']} failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"Warning: Error monitoring progress: {e}")
            # Check if workers are still alive
            alive_workers = sum(1 for w in workers if w.is_alive())
            if alive_workers == 0 and completed_trials < n_trials:
                print(
                    f"Error: All workers died but only {completed_trials}/{n_trials} trials completed")
                break

    # Wait for all workers to finish
    for worker in workers:
        worker.join(timeout=30)
        if worker.is_alive():
            print(f"Warning: Worker {worker.pid} did not terminate gracefully")

    total_time = time.time() - start_time

    # Reload study to get final results
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # Save results
    save_optuna_results(study, output_dir)

    # Generate metrics_per_phenotype.tsv from best trial
    try:
        _generate_per_phenotype_metrics(
            study, output_dir, pheno_file, base_config)
    except Exception as e:
        print(f"Warning: Failed to generate metrics_per_phenotype.tsv: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    completed_trials_list = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials_list:
        print(f"\n{'=' * 80}")
        print("HPO Optimization Complete")
        print(f"{'=' * 80}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.6f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(
            f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Results saved to: {output_dir}")
        print(f"{'=' * 80}\n")
    else:
        print(f"\n{'=' * 80}")
        print("HPO Optimization Complete")
        print(f"{'=' * 80}")
        print(f"⚠️  Warning: No completed trials found!")
        print(f"   Total trials: {len(study.trials)}")
        print(f"   Completed: {len(completed_trials_list)}")
        print(
            f"   Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
