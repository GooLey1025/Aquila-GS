"""
Optuna-based Hyperparameter Optimization for Aquila

This module provides functions for running Optuna hyperparameter optimization
using search spaces defined in the config file's 'hpo' section.
"""

try:
    import optuna
except ImportError:
    raise ImportError(
        "Optuna is required for hyperparameter optimization. "
        "Install it with: pip install optuna"
    )

# CRITICAL: Delay ALL imports that might use CUDA to allow CUDA_VISIBLE_DEVICES
# to be set in worker processes before any CUDA initialization.
# This includes torch and aquila modules that import torch.
# These will be imported inside functions that need them.

from typing import Dict, Optional, Any, Callable
from pathlib import Path
import json
import copy
import time
import argparse
import os
import multiprocessing as mp
import sys

# Delay torch and aquila imports to allow CUDA_VISIBLE_DEVICES to be set if needed
# torch, dist, and aquila modules will be imported inside functions that need them


class OptunaStopError(Exception):
    """Custom exception to stop Optuna optimization immediately."""
    pass


def merge_optuna_config(config: Dict, trial_params: Dict) -> Dict:
    """
    Merge Optuna trial parameters into YAML config.

    Supports dot notation for nested dictionaries and list indices.
    Also supports special 'branches_*' parameters that apply to all branches (snp, indel, sv).
    
    Examples:
        - 'train.learning_rate' -> config['train']['learning_rate']
        - 'model.embedder.0.kernel_size' -> config['model']['embedder'][0]['kernel_size']
        - 'branches_embedder_kernel_size' -> applies to all branches:
            config['train']['branches']['snp']['embedder'][0]['kernel_size']
            config['train']['branches']['indel']['embedder'][0]['kernel_size']
            config['train']['branches']['sv']['embedder'][0]['kernel_size']

    Args:
        config: Original YAML configuration dictionary
        trial_params: Optuna trial parameters dictionary with dot notation keys

    Returns:
        Merged configuration dictionary
    """
    merged_config = copy.deepcopy(config)
    
    # Define branch names
    branch_names = ['snp', 'indel', 'sv']
    
    # Map from simplified parameter name to full path for branches
    # Format: 'branches_X' -> ('train', 'branches', ..., final_key)
    # The function will insert branch_name after 'branches'
    branches_param_map = {
        'branches_embedder_kernel_size': ('train', 'branches', 'embedder', 0, 'kernel_size'),
        'branches_downconv_kernel_size': ('train', 'branches', 'trunk', 0, 'kernel_size'),
        'branches_downconv_dropout': ('train', 'branches', 'trunk', 0, 'dropout'),
        'branches_transformer_dropout': ('train', 'branches', 'trunk', 1, 'dropout'),
        'branches_transformer_num_heads': ('train', 'branches', 'trunk', 1, 'num_heads'),
        'branches_transformer_d_ff': ('train', 'branches', 'trunk', 1, 'd_ff'),
        'branches_pool_dropout': ('train', 'branches', 'trunk', 2, 'dropout'),
        'branches_pool_num_heads': ('train', 'branches', 'trunk', 2, 'num_heads'),
        'fusion_dropout': ('train', 'fusion', 0, 'dropout'),
        'mlp_hidden_features': ('train', 'shared_trunk', 0, 'hidden_features'),
        'mlp_dropout': ('train', 'shared_trunk', 0, 'dropout'),
        'heads_hidden_features': ('train', 'heads', 'regression', 0, 'hidden_features'),
        'heads_dropout': ('train', 'heads', 'regression', 0, 'dropout'),
    }
    
    # Separate branches_* parameters from regular parameters
    regular_params = {}
    branches_params = {}
    
    for key, value in trial_params.items():
        if key.startswith('branches_') or key in ['fusion_dropout', 'mlp_hidden_features', 
                                                   'mlp_dropout', 'heads_hidden_features', 
                                                   'heads_dropout']:
            if key in branches_param_map:
                branches_params[key] = value
        else:
            regular_params[key] = value
    
    # Handle regular parameters (dot notation)
    for key, value in regular_params.items():
        if '.' in key:
            parts = key.split('.')
            current = merged_config
            
            # Navigate/create nested structure
            skip_param = False
            for i, part in enumerate(parts[:-1]):
                # Check if part is a list index (numeric string)
                if part.isdigit():
                    idx = int(part)
                    # Current should be a list
                    if not isinstance(current, list):
                        skip_param = True
                        break
                    # Skip if index is out of range (don't modify non-existent list elements)
                    if idx >= len(current):
                        skip_param = True
                        break
                    current = current[idx]
                else:
                    # Part is a dictionary key
                    if part not in current:
                        # Check if next part is a digit
                        if i + 1 < len(parts) - 1 and parts[i + 1].isdigit():
                            # Next part is an index, so current[part] should be a list
                            current[part] = []
                        else:
                            # Next part is a key, so current[part] should be a dict
                            current[part] = {}
                    elif isinstance(current[part], list):
                        # Current part is a list, check if next part is an index
                        if i + 1 < len(parts) - 1 and parts[i + 1].isdigit():
                            # Next part is an index, keep as list
                            pass
                        else:
                            # Next part is a key, but current is list - this is an error
                            raise ValueError(
                                f"Cannot access key '{parts[i+1]}' on list at path '{'.'.join(parts[:i+1])}'"
                            )
                    elif not isinstance(current[part], dict):
                        # If existing value is not a dict and not a list, convert it
                        # Check if next part is a digit
                        if i + 1 < len(parts) - 1 and parts[i + 1].isdigit():
                            current[part] = []
                        else:
                            current[part] = {}
                    current = current[part]

            # Set the final value only if not skipped
            if not skip_param and current is not None and isinstance(current, dict):
                final_key = parts[-1]
                current[final_key] = value
        else:
            # Top-level key
            merged_config[key] = value
    
    # Handle branches_* and other train-level parameters
    for param_name, value in branches_params.items():
        if param_name not in branches_param_map:
            continue

        path_parts = branches_param_map[param_name]

        # Determine if this applies to all branches or just one level
        if 'branches' in path_parts:
            # Check if this is a multi-branch architecture before applying
            # Skip if train.branches doesn't exist (single-branch architecture)
            if 'branches' not in merged_config.get('train', {}):
                # Skip multi-branch parameters for single-branch configs
                continue
            # Apply to all branches (snp, indel, sv)
            target_branches = branch_names
        else:
            # Apply to single target (like fusion, mlp, heads)
            target_branches = [None]  # Single target, no branch name

        for branch in target_branches:
            try:
                # Build path
                path = list(path_parts)

                # Insert branch_name after 'branches' if applicable
                if branch is not None:
                    branches_idx = path.index('branches')
                    path.insert(branches_idx + 1, branch)

                # Navigate to parent of final key
                current = merged_config
                for p in path[:-1]:
                    if isinstance(p, int):
                        current = current[p]
                    else:
                        current = current[p]

                # Set the value
                final_key = path[-1]
                if isinstance(current, dict):
                    current[final_key] = value
            except (KeyError, IndexError, TypeError):
                continue

    return merged_config


def load_hpo_config(hpo_config: Dict) -> Dict:
    """
    Load and validate HPO configuration from config file's 'hpo' section.

    Args:
        hpo_config: HPO configuration dictionary from config file

    Returns:
        Validated and normalized HPO configuration

    Raises:
        ValueError: If required fields are missing or invalid
    """
    if not hpo_config:
        raise ValueError(
            "HPO configuration is empty. Please add 'hpo' section to your config file.")

    # Set defaults
    hpo_config_normalized = {
        'n_trials': hpo_config.get('n_trials', 100),
        'storage': hpo_config.get('storage', None),
        'metric': hpo_config.get('metric', 'best/val_r'),
        'direction': hpo_config.get('direction', 'maximize'),
        # Read seed from config, default to 42
        'seed': hpo_config.get('seed', 42),
        'parameters': hpo_config.get('parameters', {})
    }

    if not hpo_config_normalized['parameters']:
        raise ValueError(
            "HPO 'parameters' section is empty. Please define hyperparameters to search.")

    # Validate parameter definitions
    for param_name, param_def in hpo_config_normalized['parameters'].items():
        if not isinstance(param_def, dict):
            raise ValueError(
                f"Parameter '{param_name}' definition must be a dictionary")

        param_type = param_def.get('type')
        if param_type not in ['log_uniform', 'uniform', 'int_uniform', 'categorical']:
            raise ValueError(
                f"Parameter '{param_name}' has invalid type '{param_type}'. "
                f"Must be one of: log_uniform, uniform, int_uniform, categorical"
            )

        # Validate type-specific requirements
        if param_type == 'categorical':
            if 'choices' not in param_def:
                raise ValueError(
                    f"Parameter '{param_name}' (categorical) must have 'choices' field")
        elif param_type in ['log_uniform', 'uniform']:
            if 'low' not in param_def or 'high' not in param_def:
                raise ValueError(
                    f"Parameter '{param_name}' ({param_type}) must have 'low' and 'high' fields"
                )
        elif param_type == 'int_uniform':
            if 'low' not in param_def or 'high' not in param_def:
                raise ValueError(
                    f"Parameter '{param_name}' (int_uniform) must have 'low' and 'high' fields"
                )

    return hpo_config_normalized


def create_optuna_study(
    hpo_config: Dict,
    study_name: Optional[str] = None,
    storage: Optional[str] = None
) -> optuna.Study:
    """
    Create Optuna study with TPE sampler.

    Args:
        hpo_config: Normalized HPO configuration
        study_name: Name for the study (default: auto-generated)
        storage: Storage URL for study persistence (from hpo_config if None)

    Returns:
        Created Optuna study
    """
    storage_url = storage or hpo_config.get('storage')
    direction = hpo_config.get('direction', 'maximize')
    seed = hpo_config.get('seed', 42)  # Read seed from config, default to 42

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True
    )

    return study


def suggest_hyperparameters(trial: optuna.Trial, hpo_config: Dict) -> Dict[str, Any]:
    """
    Suggest hyperparameters from Optuna trial based on HPO configuration.

    Args:
        trial: Optuna trial object
        hpo_config: Normalized HPO configuration

    Returns:
        Dictionary of suggested hyperparameters
    """
    params = {}
    parameters_def = hpo_config['parameters']

    for param_name, param_def in parameters_def.items():
        param_type = param_def['type']

        if param_type == 'log_uniform':
            params[param_name] = trial.suggest_float(
                param_name,
                param_def['low'],
                param_def['high'],
                log=True
            )
        elif param_type == 'uniform':
            params[param_name] = trial.suggest_float(
                param_name,
                param_def['low'],
                param_def['high']
            )
        elif param_type == 'int_uniform':
            params[param_name] = trial.suggest_int(
                param_name,
                param_def['low'],
                param_def['high']
            )
        elif param_type == 'categorical':
            params[param_name] = trial.suggest_categorical(
                param_name,
                param_def['choices']
            )

    return params


def save_optuna_results(study: optuna.Study, output_dir: Path):
    """
    Save Optuna study results to files.

    Args:
        study: Optuna study object
        output_dir: Output directory for saving results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if there are any completed trials
    completed_trials = [t for t in study.trials if t.state ==
                        optuna.trial.TrialState.COMPLETE]

    if not completed_trials:
        # No completed trials - save summary with warning
        summary_path = output_dir / 'optuna_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Optuna Study Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"⚠️  WARNING: No completed trials found!\n\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(f"Completed: {len(completed_trials)}\n")
            f.write(
                f"Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")
            f.write(
                f"Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
            f.write(
                f"\nAll trials failed or were pruned. Please check the error messages above.\n")

        print(f"\n⚠️  Warning: No completed trials found!")
        print(f"   Total trials: {len(study.trials)}")
        print(f"   Completed: {len(completed_trials)}")
        print(
            f"   Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        print(
            f"   Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"\n   Summary saved to: {summary_path}")
        return

    # Save best parameters as JSON
    best_params_path = output_dir / 'optuna_best_params.json'
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)

    # Save study summary
    summary_path = output_dir / 'optuna_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Optuna Study Summary\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Best trial number: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.6f}\n")
        f.write(f"Best parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nTotal trials: {len(study.trials)}\n")
        f.write(f"Completed: {len(completed_trials)}\n")
        f.write(
            f"Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")
        f.write(
            f"Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")

    print(f"\nOptuna results saved to:")
    print(f"  Best parameters: {best_params_path}")
    print(f"  Summary: {summary_path}")


def run_optuna_search(
    base_config: Dict,
    hpo_config: Dict,
    output_dir: Path,
    train_single_run_func: Callable,
    train_single_run_kwargs: Dict,
    n_trials: Optional[int] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    n_gpus: int = 0,
    n_workers_per_gpu: int = 1
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization.

    Args:
        base_config: Base configuration dictionary (without hpo section)
        hpo_config: Normalized HPO configuration
        output_dir: Output directory for results
        train_single_run_func: Function to run a single training trial
        train_single_run_kwargs: Additional keyword arguments for train_single_run_func
        n_trials: Number of trials (overrides hpo_config if provided)
        use_wandb: Whether to log Optuna study to WandB
        wandb_project: WandB project name (if use_wandb is True)
        wandb_name: WandB run name for the study (if use_wandb is True)
        n_gpus: Number of GPUs to use for parallel trials (0 for sequential)
        n_workers_per_gpu: Number of worker processes per GPU

    Returns:
        Optuna study object with optimization results
    """
    n_trials = n_trials or hpo_config.get('n_trials', 100)
    study_name = f"aquila_hpo_{output_dir.name}"

    # Setup storage (required for parallel execution, optional for sequential)
    storage_url = hpo_config.get('storage')
    if n_gpus > 0 and storage_url is None:
        # Use SQLite storage for parallel execution
        storage_file = output_dir / 'optuna_study.db'
        storage_url = f'sqlite:///{storage_file}'
        print(f"Using SQLite storage for parallel HPO: {storage_url}")

    # Create study
    study = create_optuna_study(
        hpo_config=hpo_config,
        study_name=study_name,
        storage=storage_url
    )

    # Initialize WandB callback for Optuna study logging
    wandb_callback = None
    if use_wandb:
        try:
            from optuna.integration import WandbCallback
            import wandb

            # Initialize WandB run for the study
            if wandb.run is None:
                study_wandb_name = wandb_name or f"optuna_study_{study_name}"
                wandb.init(
                    project=wandb_project or 'aquila-gs-optuna',
                    name=study_wandb_name,
                    config={
                        'study_name': study_name,
                        'n_trials': n_trials,
                        'direction': hpo_config.get('direction', 'maximize'),
                        'metric': hpo_config.get('metric', 'best/val_r'),
                        'parameters': list(hpo_config['parameters'].keys()),
                    },
                    dir=str(output_dir),
                )

            # Create WandB callback for Optuna
            wandb_callback = WandbCallback(
                metric_name=hpo_config.get('metric', 'best/val_r'),
                as_multirun=False,  # Log all trials in one run
            )
            print(f"\n📊 WandB integration enabled for Optuna study")
            print(f"   Project: {wandb_project or 'aquila-gs-optuna'}")
            print(
                f"   Run: {wandb.run.name if wandb.run else study_wandb_name}")
        except ImportError:
            print(
                f"\n⚠️  Warning: optuna[wandb] not installed. Install with: pip install 'optuna[wandb]'")
            print(f"   Continuing without WandB integration for Optuna study.")
            use_wandb = False
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to initialize WandB callback: {e}")
            print(f"   Continuing without WandB integration for Optuna study.")
            use_wandb = False

    print(f"\n{'='*80}")
    print(f"Starting Optuna Hyperparameter Optimization")
    print(f"{'='*80}")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Optimization direction: {hpo_config.get('direction', 'maximize')}")
    print(f"Metric: {hpo_config.get('metric', 'best/val_r')}")
    print(f"Parameters to optimize: {len(hpo_config['parameters'])}")
    for param_name in hpo_config['parameters'].keys():
        print(f"  - {param_name}")
    if n_gpus > 0:
        print(
            f"Parallel execution: {n_gpus} GPUs × {n_workers_per_gpu} workers = {n_gpus * n_workers_per_gpu} parallel trials")
    else:
        print(f"Sequential execution (no GPUs)")
    print(f"{'='*80}\n")

    # Run optimization (parallel or sequential)
    if n_gpus > 0:
        # Parallel execution using multiprocessing
        study = _run_optuna_search_parallel(
            study_name=study_name,
            storage_url=storage_url,
            hpo_config=hpo_config,
            base_config=base_config,
            output_dir=output_dir,
            train_single_run_func=train_single_run_func,
            train_single_run_kwargs=train_single_run_kwargs,
            n_trials=n_trials,
            n_gpus=n_gpus,
            n_workers_per_gpu=n_workers_per_gpu,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            wandb_callback=wandb_callback
        )
    else:
        # Sequential execution (original implementation)
        study = _run_optuna_search_sequential(
            study_name=study_name,
            storage_url=storage_url,
            hpo_config=hpo_config,
            base_config=base_config,
            output_dir=output_dir,
            train_single_run_func=train_single_run_func,
            train_single_run_kwargs=train_single_run_kwargs,
            n_trials=n_trials,
            use_wandb=use_wandb,
            wandb_callback=wandb_callback
        )

    # Save results
    save_optuna_results(study, output_dir)

    # Print summary
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        print(f"\n{'='*80}")
        print(f"Optuna Optimization Complete")
        print(f"{'='*80}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.6f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Optuna Optimization Complete")
        print(f"{'='*80}")
        print(f"⚠️  Warning: No completed trials found!")
        print(f"   Total trials: {len(study.trials)}")
        print(f"   Completed: {len(completed_trials)}")
        print(
            f"   Failed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        print(
            f"   Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"{'='*80}\n")

    return study


def _run_optuna_search_sequential(
    study_name: str,
    storage_url: Optional[str],
    hpo_config: Dict,
    base_config: Dict,
    output_dir: Path,
    train_single_run_func: Callable,
    train_single_run_kwargs: Dict,
    n_trials: int,
    use_wandb: bool,
    wandb_callback: Optional[Any]
) -> optuna.Study:
    """Run Optuna optimization sequentially (single process)."""
    # Create study
    study = create_optuna_study(
        hpo_config=hpo_config,
        study_name=study_name,
        storage=storage_url
    )

    # Flag to track if we should stop due to error
    stop_flag = {'should_stop': False, 'error': None}

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        print(f"\nTrial {trial.number + 1}/{n_trials}")
        print(f"{'-'*80}")

        # Suggest hyperparameters
        trial_params = suggest_hyperparameters(trial, hpo_config)

        print("Suggested hyperparameters:")
        for key, value in trial_params.items():
            print(f"  {key}: {value}")

        # Merge trial parameters into base config
        trial_config = merge_optuna_config(base_config, trial_params)

        # Create trial-specific output directory
        trial_output_dir = output_dir / f"trial_{trial.number}"

        # Run training
        try:
            best_valid_r = train_single_run_func(
                config=trial_config,
                trial_output_dir=trial_output_dir,
                **train_single_run_kwargs
            )

            print(f"\nTrial {trial.number + 1} completed:")
            print(f"  Best validation Pearson R: {best_valid_r:.6f}")

            return best_valid_r
        except Exception as e:
            print(f"\n❌ Trial {trial.number + 1} failed with error: {e}")
            import traceback
            traceback.print_exc()
            print(
                f"\n⚠️  Stopping optimization due to error in trial {trial.number + 1}")
            # Set stop flag
            stop_flag['should_stop'] = True
            stop_flag['error'] = f"Trial {trial.number + 1} failed: {e}"
            # Re-raise to mark trial as failed
            raise

    def check_stop_callback(study, trial):
        """Callback to check if we should stop optimization."""
        if stop_flag['should_stop']:
            study.stop()

    try:
        callbacks = [check_stop_callback]
        if wandb_callback is not None:
            callbacks.append(wandb_callback)

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=callbacks if callbacks else None
        )

        # Check if we should stop due to error
        if stop_flag['should_stop']:
            print(
                f"\n\n❌ Optimization stopped due to error: {stop_flag['error']}")
            print(f"Completed {len(study.trials)} trials before stopping.")
            raise OptunaStopError(stop_flag['error'])
    except OptunaStopError as e:
        print(f"\n\n❌ Optimization stopped due to error: {e}")
        print(f"Completed {len(study.trials)} trials before stopping.")
        raise
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print(f"Completed {len(study.trials)} trials before interruption.")

    return study


def _run_optuna_search_parallel(
    study_name: str,
    storage_url: str,
    hpo_config: Dict,
    base_config: Dict,
    output_dir: Path,
    train_single_run_func: Callable,
    train_single_run_kwargs: Dict,
    n_trials: int,
    n_gpus: int,
    n_workers_per_gpu: int,
    use_wandb: bool,
    wandb_project: Optional[str],
    wandb_name: Optional[str],
    wandb_callback: Optional[Any]
) -> optuna.Study:
    """Run Optuna optimization in parallel using multiple GPUs."""
    # CRITICAL: Set multiprocessing start method BEFORE creating any Manager
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, check if it's 'spawn'
        if mp.get_start_method() != 'spawn':
            print(
                "Warning: Multiprocessing start method is not 'spawn'. This may cause CUDA issues.")

    # Detect available GPUs
    def detect_available_gpus() -> list:
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

    available_gpus = detect_available_gpus()
    if len(available_gpus) < n_gpus:
        print(
            f"Warning: Requested {n_gpus} GPUs but only {len(available_gpus)} available. Using {len(available_gpus)} GPUs.")
        n_gpus = len(available_gpus)

    gpu_list = available_gpus[:n_gpus]
    total_workers = n_gpus * n_workers_per_gpu

    print(f"Parallel HPO Configuration:")
    print(f"  Available GPUs: {available_gpus}")
    print(f"  Using GPUs: {gpu_list}")
    print(f"  Workers per GPU: {n_workers_per_gpu}")
    print(f"  Total worker processes: {total_workers}")

    # Create study with storage (required for parallel execution)
    study = create_optuna_study(
        hpo_config=hpo_config,
        study_name=study_name,
        storage=storage_url
    )

    # Create queues for progress updates
    manager = mp.Manager()
    progress_queue = manager.Queue()  # For progress updates

    # Assign GPU to each worker: round-robin assignment
    # Worker 0 -> GPU 0, Worker 1 -> GPU 1, Worker 2 -> GPU 2, Worker 3 -> GPU 0, etc.
    worker_gpu_assignments = []
    for worker_id in range(total_workers):
        gpu_idx = worker_id % n_gpus
        assigned_gpu = gpu_list[gpu_idx]
        worker_gpu_assignments.append(assigned_gpu)

    print(f"GPU assignments:")
    for worker_id, gpu_id in enumerate(worker_gpu_assignments):
        print(f"  Worker {worker_id} -> GPU {gpu_id}")

    # Start worker processes
    workers = []
    for worker_id in range(total_workers):
        assigned_gpu = worker_gpu_assignments[worker_id]

        # Create environment dict with CUDA_VISIBLE_DEVICES set
        worker_env = os.environ.copy()
        worker_env['CUDA_VISIBLE_DEVICES'] = str(assigned_gpu)

        worker = mp.Process(
            target=_hpo_worker_process,
            args=(
                worker_id,
                assigned_gpu,  # Pass assigned GPU directly
                progress_queue,
                study_name,
                storage_url,
                hpo_config,
                base_config,
                output_dir,
                train_single_run_func,
                train_single_run_kwargs,
                n_trials
            ),
            env=worker_env  # Pass environment with CUDA_VISIBLE_DEVICES set
        )
        worker.start()
        workers.append(worker)

    # Monitor progress
    completed_trials = 0
    start_time = time.time()

    # Wait for all trials to complete
    while completed_trials < n_trials:
        try:
            progress = progress_queue.get(timeout=300)  # 5 minute timeout
            if progress['type'] == 'completed':
                completed_trials += 1
                elapsed = time.time() - start_time
                print(
                    f"Progress: {completed_trials}/{n_trials} trials completed (elapsed: {elapsed/60:.1f} min)")
                if progress.get('value') is not None:
                    print(
                        f"  Trial {progress.get('trial_number', '?')} val_r: {progress['value']:.6f}")
            elif progress['type'] == 'failed':
                completed_trials += 1
                print(
                    f"Trial {progress.get('trial_number', '?')} failed: {progress.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Warning: Error monitoring progress: {e}")
            # Check if workers are still alive
            alive_workers = sum(1 for w in workers if w.is_alive())
            if alive_workers == 0 and completed_trials < n_trials:
                print(
                    f"Error: All workers died but only {completed_trials}/{n_trials} trials completed")
                break

    # Workers will exit when they can't get more trials from study

    # Wait for all workers to finish
    for worker in workers:
        worker.join(timeout=30)
        if worker.is_alive():
            print(f"Warning: Worker {worker.pid} did not terminate gracefully")

    # Reload study to get final results
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    return study


def _hpo_worker_process(
    worker_id: int,
    assigned_gpu_id: int,
    progress_queue: mp.Queue,
    study_name: str,
    storage_url: str,
    hpo_config: Dict,
    base_config: Dict,
    output_dir: Path,
    train_single_run_func: Callable,
    train_single_run_kwargs: Dict,
    n_trials: int
) -> None:
    """Worker process that runs trials in parallel."""
    # CUDA_VISIBLE_DEVICES is already set via Process(env=...) parameter
    # Verify it's set correctly
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    print(
        f"Worker {worker_id} started (PID: {os.getpid()}, GPU: {assigned_gpu_id})")
    print(
        f"  CUDA_VISIBLE_DEVICES={cuda_visible} (expected: {assigned_gpu_id})")

    # Double-check: ensure CUDA_VISIBLE_DEVICES matches assigned GPU
    if cuda_visible != str(assigned_gpu_id):
        print(
            f"  WARNING: CUDA_VISIBLE_DEVICES mismatch! Setting to {assigned_gpu_id}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(assigned_gpu_id)

    # Create a study instance for this worker (connects to shared storage)
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_url
    )

    trials_completed = 0

    while trials_completed < n_trials:
        trial = None
        try:

            # Ask for a trial from the study (this will suggest hyperparameters)
            # This is thread-safe when using database storage
            try:
                trial = study.ask()
            except Exception as e:
                # No more trials available (all n_trials completed)
                print(f"Worker {worker_id}: No more trials available: {e}")
                break

            # Extract suggested hyperparameters from trial
            # Note: study.ask() already suggested parameters, we just need to get them
            # But we need to call suggest_* methods to get the values
            # Actually, we should use the suggest_hyperparameters function
            trial_params = suggest_hyperparameters(trial, hpo_config)

            print(
                f"Worker {worker_id} (GPU {assigned_gpu_id}): Starting trial {trial.number}")
            print(f"  Hyperparameters:")
            for key, value in trial_params.items():
                print(f"    {key}: {value}")

            # Merge trial parameters into base config
            trial_config = merge_optuna_config(base_config, trial_params)

            # Create trial-specific output directory
            trial_output_dir = output_dir / f"trial_{trial.number}"

            # Override device in train_single_run_kwargs to force cuda:0
            # After setting CUDA_VISIBLE_DEVICES, the assigned GPU becomes GPU 0 in this process
            modified_kwargs = train_single_run_kwargs.copy()
            if modified_kwargs.get('args') is not None:
                # Modify args.device directly to force cuda:0
                modified_kwargs['args'].device = 'cuda:0'

            # Run training
            try:
                best_valid_r = train_single_run_func(
                    config=trial_config,
                    trial_output_dir=trial_output_dir,
                    **modified_kwargs
                )

                print(
                    f"Worker {worker_id} (GPU {assigned_gpu_id}): Trial {trial.number} completed: val_r={best_valid_r:.6f}")

                # Report result to study
                study.tell(trial, best_valid_r)

                # Report progress
                progress_queue.put({
                    'type': 'completed',
                    'trial_number': trial.number,
                    'value': best_valid_r
                })

                trials_completed += 1

            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                print(
                    f"Worker {worker_id} (GPU {assigned_gpu_id}): Trial {trial.number} failed: {e}")
                print(f"  Traceback: {error_msg}")

                # Report failure to study
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

                # Report progress
                progress_queue.put({
                    'type': 'failed',
                    'trial_number': trial.number,
                    'error': str(e)
                })

                trials_completed += 1

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Worker {worker_id} (GPU {assigned_gpu_id}) error: {e}")
            print(f"  Traceback: {error_msg}")
            if trial is not None:
                try:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    progress_queue.put({
                        'type': 'failed',
                        'trial_number': trial.number,
                        'error': str(e)
                    })
                    trials_completed += 1
                except:
                    pass

    print(
        f"Worker {worker_id} shutting down (completed {trials_completed} trials)")


def train_single_run(
    config: Dict,
    trial_output_dir: Optional[Path] = None,
    args: Optional[Any] = None,
    is_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    print_rank0_func=None
) -> float:
    """
    Run a single training run with given configuration.

    This function is used by Optuna to run individual trials during
    hyperparameter optimization.

    Args:
        config: Configuration dictionary (may include hpo trial parameters)
        trial_output_dir: Output directory for this trial (if None, uses args.output)
        args: Command line arguments (required if trial_output_dir is None)
        is_distributed: Whether running in distributed mode
        rank: Process rank for distributed training
        world_size: World size for distributed training
        local_rank: Local rank for distributed training
        print_rank0_func: Function to print only on rank 0

    Returns:
        Best validation Pearson R (best_valid_r)
    """
    if print_rank0_func is None:
        def print_rank0(*args_print, **kwargs):
            if rank == 0:
                print(*args_print, **kwargs)
    else:
        print_rank0 = print_rank0_func

    # Determine output directory
    if trial_output_dir is None:
        if args is None:
            raise ValueError(
                "Either trial_output_dir or args must be provided")
        output_dir = Path(args.output)
    else:
        output_dir = trial_output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Override config with command line arguments if provided
    if args:
        if hasattr(args, 'vcf') and args.vcf:
            config['data']['geno_file'] = args.vcf
        if hasattr(args, 'pheno') and args.pheno:
            config['data']['pheno_file'] = args.pheno

    # Set encoding type
    if args and hasattr(args, 'encoding_type'):
        encoding_type = args.encoding_type
    else:
        encoding_type = config['data'].get('encoding_type', 'snp_vcf')
    config['data']['encoding_type'] = encoding_type

    # Import aquila modules here (after CUDA_VISIBLE_DEVICES may have been set in worker processes)
    from aquila.utils import set_seed, save_config

    # Set random seed
    seed = args.seed if args and hasattr(args, 'seed') else 42
    set_seed(seed)

    # Import torch here (after CUDA_VISIBLE_DEVICES may have been set in worker processes)
    import torch
    import torch.distributed as dist

    # Set device
    # For parallel GPU search, CUDA_VISIBLE_DEVICES is set in worker process
    # so we should use 'cuda:0' (which maps to the visible GPU) or just 'cuda'
    if is_distributed:
        device = f'cuda:{local_rank}'
    else:
        # In worker processes with CUDA_VISIBLE_DEVICES set, use 'cuda' or 'cuda:0'
        # Both will use the single visible GPU (indexed as 0 in the worker process)
        if args and hasattr(args, 'device') and args.device:
            device = args.device
            # If device is 'cuda' or 'cuda:0', ensure it uses the visible GPU
            if device.startswith('cuda'):
                # In worker process, CUDA_VISIBLE_DEVICES is set, so 'cuda:0' is correct
                # But if args.device is 'cuda', keep it as 'cuda' (will use GPU 0)
                pass
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Ensure we're using the correct GPU in worker processes
        if torch.cuda.is_available() and device.startswith('cuda'):
            # Set the device to ensure we use the visible GPU
            if ':' in device:
                gpu_idx = int(device.split(':')[1])
            else:
                gpu_idx = 0

            # CRITICAL: Clear CUDA cache and set device before any operations
            torch.cuda.empty_cache()
            torch.cuda.set_device(gpu_idx)

            # Verify we're using the correct GPU
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible:
                print(
                    f"train_single_run: CUDA_VISIBLE_DEVICES={cuda_visible}, Using device={device}, GPU index={gpu_idx}")
                print(
                    f"train_single_run: PyTorch sees {torch.cuda.device_count()} GPU(s)")
                print(
                    f"train_single_run: Current GPU: {torch.cuda.get_device_name(gpu_idx)}")
                print(
                    f"train_single_run: GPU Memory - Allocated: {torch.cuda.memory_allocated(gpu_idx)/(1024**3):.2f} GB, Reserved: {torch.cuda.memory_reserved(gpu_idx)/(1024**3):.2f} GB")

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

    # Create data loaders
    data_split_seed = args.data_split_seed if args and hasattr(
        args, 'data_split_seed') else 42
    data_restart = args.data_restart if args and hasattr(
        args, 'data_restart') else False
    skew_threshold = args.skew_threshold if args and hasattr(
        args, 'skew_threshold') else 2.0

    # Import aquila modules here (after CUDA_VISIBLE_DEVICES may have been set in worker processes)
    from aquila.data_utils import create_data_loaders

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
        data_restart=data_restart,
        skew_threshold=skew_threshold,
        rank=rank,
        world_size=world_size,
        use_distributed_sampler=is_distributed,
        data_split_seed=data_split_seed,
        augmentation_config=config.get('augmentation', None),
    )

    # Save normalization statistics (only on rank 0)
    if normalization_stats and rank == 0:
        import pickle
        norm_path = output_dir / 'normalization_stats.pkl'
        with open(norm_path, 'wb') as f:
            pickle.dump(normalization_stats, f)

    # Get actual task lists from first batch
    sample_batch = next(iter(train_loader))
    num_regression_tasks = 0
    num_classification_tasks = 0

    # Determine sequence length(s) for multi-branch
    # Check both old format (encoding_type) and new format (variant_type)
    variant_type = config.get('data', {}).get('variant_type')
    is_multi_branch = encoding_type in ['snp_indel_vcf', 'snp_indel_sv_vcf'] or variant_type in ['snp_indel', 'snp_indel_sv']

    if is_multi_branch:
        seq_length = {}
        for key in sample_batch.keys():
            if key not in ['sample_id', 'regression_targets', 'regression_mask',
                           'classification_targets', 'classification_mask']:
                seq_length[key] = sample_batch[key].shape[1]
    else:
        seq_length = sample_batch['snp'].shape[1]

    if 'regression_targets' in sample_batch:
        num_regression_tasks = sample_batch['regression_targets'].shape[1]
    if 'classification_targets' in sample_batch:
        num_classification_tasks = sample_batch['classification_targets'].shape[1]

    # Generate task names if not provided
    regression_task_names = None
    classification_task_names = None

    if num_regression_tasks > 0:
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

    # Create model from config
    # Import aquila modules here (after CUDA_VISIBLE_DEVICES may have been set in worker processes)
    from aquila.varnn import create_model_from_config

    model = create_model_from_config(
        config=config,
        seq_length=seq_length,
        regression_tasks=regression_task_names,
        classification_tasks=classification_task_names
    )

    # Move model to device
    model = model.to(device)

    # Initialize lazy parameters with a dummy forward pass
    # This is required for LazyModule to initialize all parameters
    model.eval()
    with torch.no_grad():
        if is_multi_branch:
            dummy_input = {}
            for vtype, vlen in seq_length.items():
                dummy_input[vtype] = torch.zeros(1, vlen, 8, device=device)
        else:
            dummy_input = torch.zeros(1, seq_length, 8, device=device)
        _ = model(dummy_input)
    model.train()

    # Wrap with DDP if distributed
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

    # Initialize wandb (only on rank 0)
    wandb_run = None
    if args and hasattr(args, 'use_wandb') and args.use_wandb and rank == 0:
        try:
            import wandb
            from datetime import datetime

            if wandb.run is None:
                wandb_name = args.wandb_name if hasattr(
                    args, 'wandb_name') else None
                if wandb_name is None:
                    config_name = Path(
                        args.config).stem if hasattr(args, 'config') and args.config else "trial"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wandb_name = f"{config_name}_{timestamp}"

                wandb.init(
                    project=args.wandb_project if hasattr(
                        args, 'wandb_project') else 'aquila-gs',
                    name=wandb_name,
                    tags=args.wandb_tags if hasattr(
                        args, 'wandb_tags') else None,
                    config={
                        'config_file': str(args.config) if hasattr(args, 'config') and args.config else 'trial',
                        'seed': seed,
                        'data_split_seed': data_split_seed,
                        'device': device,
                        'encoding_type': encoding_type,
                        'is_distributed': is_distributed,
                        'world_size': world_size,
                        'mixed_precision': args.mixed_precision if hasattr(args, 'mixed_precision') else False,
                        **config
                    },
                    dir=str(output_dir),
                )
                wandb_run = wandb.run
            else:
                wandb_run = wandb.run
                wandb.config.update(config, allow_val_change=True)
        except ImportError:
            pass
        except Exception as e:
            pass

    # Create trainer
    use_mixed_precision = args.mixed_precision if args and hasattr(
        args, 'mixed_precision') else False

    # Import aquila modules here (after CUDA_VISIBLE_DEVICES may have been set in worker processes)
    from aquila.trainer import VarTrainer

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
        use_mixed_precision=use_mixed_precision,
        huber_delta=train_config.get('huber_delta', 1.0),
        wandb_run=wandb_run,
    )

    # Train model
    trainer.train(
        num_epochs=train_config.get('num_epochs', 100),
        verbose=False  # Reduce verbosity for Optuna trials
    )

    # Return best validation Pearson R
    return trainer.best_metrics.get('val_r', 0.0)
