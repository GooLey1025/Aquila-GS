#!/usr/bin/env python3
"""
Select the best model from multiple parameter trials based on validation R.

Usage:
    python parse_best_model.py <results_summary.tsv> [--output-dir DIR] [--print-only-path]

Output:
    Prints the path to the best model directory.
"""

import argparse
import json
import sys
from pathlib import Path


def parse_results_summary(summary_path: Path) -> dict:
    """
    Parse results_summary.tsv to find the best model.
    The file has columns: type, n_runs, val_r_mean, val_r_std, ...
    Types include: fold_0, fold_1, ..., overall
    """
    import pandas as pd

    df = pd.read_csv(summary_path, sep='\t')

    # Filter to individual fold results (not overall)
    fold_df = df[df['type'].str.startswith('fold_')].copy()

    if fold_df.empty:
        raise ValueError(f"No fold results found in {summary_path}")

    # Find the best fold (highest val_r_mean)
    best_fold_row = fold_df.loc[fold_df['val_r_mean'].idxmax()]

    return {
        'best_fold': best_fold_row['type'],
        'best_fold_idx': int(best_fold_row['type'].split('_')[1]),
        'val_r_mean': float(best_fold_row['val_r_mean']),
        'val_r_std': float(best_fold_row['val_r_std']) if 'val_r_std' in best_fold_row else None,
    }


def find_model_dir(output_dir: Path, fold_idx: int) -> Path:
    """
    Find the model directory for a specific fold.
    """
    fold_dir = output_dir / f"fold_{fold_idx}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    # Look for the model directory (seed_0 for the first seed)
    seed_dir = fold_dir / "seed_0"
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    return seed_dir


def main():
    parser = argparse.ArgumentParser(description='Select best model from parameter trials')
    parser.add_argument('summary_file', type=str, help='Path to results_summary.tsv')
    parser.add_argument('-o', '--output-dir', type=str, default='.',
                        help='Output directory containing fold directories')
    parser.add_argument('--print-only-path', action='store_true',
                        help='Only print the best model path')

    args = parser.parse_args()

    summary_path = Path(args.summary_file)
    output_dir = Path(args.output_dir)

    if not summary_path.exists():
        print(f"Error: {summary_path} not found", file=sys.stderr)
        sys.exit(1)

    result = parse_results_summary(summary_path)

    best_fold_idx = result['best_fold_idx']
    best_model_dir = find_model_dir(output_dir, best_fold_idx)

    if args.print_only_path:
        print(best_model_dir)
    else:
        print(f"Best fold: {result['best_fold']}")
        print(f"Val R mean: {result['val_r_mean']:.6f}")
        if result['val_r_std'] is not None:
            print(f"Val R std: {result['val_r_std']:.6f}")
        print(f"Best model dir: {best_model_dir}")

    return best_model_dir


if __name__ == '__main__':
    main()
