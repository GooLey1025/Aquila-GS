"""
Utility functions for Aquila SNP neural networks.
"""

import torch
import numpy as np
import random
import yaml
from pathlib import Path
from typing import Dict


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config: Dict, path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Path to save file
    """
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to: {path}")


def merge_wandb_config(config: Dict, wandb_config: Dict) -> Dict:
    """
    Merge WandB sweep config into YAML config.

    WandB config uses dot notation (e.g., 'train.learning_rate'),
    which needs to be merged into nested dict structure.

    Args:
        config: Original YAML configuration dictionary
        wandb_config: WandB config dictionary with dot notation keys

    Returns:
        Merged configuration dictionary
    """
    merged_config = config.copy()

    for key, value in wandb_config.items():
        # Skip wandb internal keys
        if key.startswith('_') or key in ['wandb_version', 'wandb_job_type']:
            continue

        # Handle dot notation (e.g., 'train.learning_rate')
        if '.' in key:
            parts = key.split('.')
            current = merged_config

            # Navigate/create nested structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    # If existing value is not a dict, convert it
                    current[part] = {}
                current = current[part]

            # Set the final value
            current[parts[-1]] = value
        else:
            # Top-level key
            merged_config[key] = value

    return merged_config


def load_config(path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_model_summary(model: torch.nn.Module, input_size=None, device='cpu', verbose=True, encoding_type='token'):
    """
    Print model architecture and parameter counts using torchinfo.

    Args:
        model: PyTorch model
        input_size: Input tensor size for detailed summary
                    - For 'token': (batch_size, seq_length)
                    - For 'diploid_onehot': (batch_size, seq_length, 8)
        device: Device to use for summary
        verbose: If True, print detailed layer-by-layer summary
        encoding_type: Type of encoding ('token' or 'diploid_onehot')
    """
    from torchinfo import summary

    print("\nModel Architecture Summary:")
    print("=" * 80)

    if input_size is not None:
        # Determine dtype based on encoding type
        if encoding_type == 'token':
            # Token encoding: (batch_size, seq_length) with integer values {0,1,2,3}
            dtype = torch.long
        elif encoding_type == 'diploid_onehot':
            # Diploid one-hot encoding: (batch_size, seq_length, 8) with float values
            dtype = torch.float
        else:
            # Default to long for backward compatibility
            dtype = torch.long

        # Detailed summary with input size
        summary(
            model,
            input_size=input_size,
            dtypes=[dtype],
            device=device,
            depth=2 if verbose else 1,
            col_names=["input_size", "output_size", "num_params"],
            row_settings=["var_names"] if verbose else [],
            verbose=1
        )
    else:
        # Summary without input size
        summary(
            model,
            device=device,
            depth=3,
            col_names=["num_params", "trainable"],
            verbose=1
        )


def count_snps_missing(snp_matrix: np.ndarray) -> Dict[str, float]:
    """
    Count missing SNPs in the data.

    Args:
        snp_matrix: (n_samples, n_snps) array

    Returns:
        Dictionary with missing statistics
    """
    total_values = snp_matrix.size
    missing_values = (snp_matrix == 3).sum()
    missing_rate = missing_values / total_values

    # Per-sample missing rate
    missing_per_sample = (snp_matrix == 3).sum(axis=1) / snp_matrix.shape[1]

    # Per-SNP missing rate
    missing_per_snp = (snp_matrix == 3).sum(axis=0) / snp_matrix.shape[0]

    return {
        'total_values': int(total_values),
        'missing_values': int(missing_values),
        'missing_rate': float(missing_rate),
        'avg_missing_per_sample': float(missing_per_sample.mean()),
        'avg_missing_per_snp': float(missing_per_snp.mean()),
        'max_missing_per_sample': float(missing_per_sample.max()),
        'max_missing_per_snp': float(missing_per_snp.max()),
    }


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_training_header():
    """Print training header."""
    print("\n" + "=" * 80)
    print(" " * 25 + "AQUILA SNP TRAINING")
    print("=" * 80)


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
