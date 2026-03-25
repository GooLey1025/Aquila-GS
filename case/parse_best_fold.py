#!/usr/bin/env python3
"""
Parse results_summary.tsv to find the best fold (highest val_r_mean),
then return the path to data_split.tsv for that fold with a specified seed.
"""

import argparse
import os
import pandas as pd


def parse_best_fold(results_summary_path, output_dir=None, default_seed=42, verbose=True):
    """
    Parse results_summary.tsv and find the best fold.
    
    Args:
        results_summary_path: Path to results_summary.tsv
        output_dir: If provided, write the best fold info to this directory
        default_seed: Seed to use for data_split.tsv (default: 42)
        verbose: If True, print detailed info (default: True)
    
    Returns:
        Tuple of (best_fold_name, data_split_path, val_r_mean)
    """
    # Read results summary
    df = pd.read_csv(results_summary_path, sep='\t')
    
    # Filter only fold rows (exclude 'overall')
    fold_df = df[df['type'].str.startswith('fold_')].copy()
    
    # Find the fold with highest val_r_mean
    best_row = fold_df.loc[fold_df['val_r_mean'].idxmax()]
    best_fold = best_row['type']
    best_val_r_mean = best_row['val_r_mean']
    
    if verbose:
        print(f"Best fold: {best_fold}")
        print(f"val_r_mean: {best_val_r_mean:.6f}")
        print(f"val_r_std: {best_row['val_r_std']:.6f}")
        print(f"val_r2_mean: {best_row['val_r2_mean']:.6f}")
    
    # Get the directory containing results_summary.tsv
    results_dir = os.path.dirname(os.path.abspath(results_summary_path))
    
    # Construct path to data_split.tsv with specified seed
    data_split_path = os.path.join(results_dir, best_fold, f"seed_{default_seed}", "data_split.tsv")
    
    # Check if the file exists
    if not os.path.exists(data_split_path):
        # Try to find any available seed for this fold
        fold_dir = os.path.join(results_dir, best_fold)
        if os.path.exists(fold_dir):
            # Find seed directories
            seed_dirs = [d for d in os.listdir(fold_dir) if d.startswith('seed_')]
            if seed_dirs:
                # Sort and pick the first one
                seed_dirs.sort()
                fallback_seed = seed_dirs[0].replace('seed_', '')
                data_split_path = os.path.join(results_dir, best_fold, seed_dirs[0], "data_split.tsv")
                if verbose:
                    print(f"Warning: seed_{default_seed} not found, using {fallback_seed}")
    
    if verbose:
        print(f"\nBest data_split.tsv path:")
        print(data_split_path)
    
    # Optionally write to output file
    if output_dir:
        output_file = os.path.join(output_dir, "best_fold_info.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(f"best_fold={best_fold}\n")
            f.write(f"data_split={data_split_path}\n")
            f.write(f"val_r_mean={best_val_r_mean:.6f}\n")
        print(f"\nInfo written to: {output_file}")
    
    return best_fold, data_split_path, best_val_r_mean


def main():
    parser = argparse.ArgumentParser(description='Parse best fold from results_summary.tsv')
    parser.add_argument('results_summary', help='Path to results_summary.tsv')
    parser.add_argument('-o', '--output-dir', help='Output directory for best_fold_info.txt', default=None)
    parser.add_argument('-s', '--seed', type=int, default=42, help='Seed to use (default: 42)')
    parser.add_argument('--print-only-path', action='store_true', help='Print ONLY the data_split.tsv path, nothing else (for shell scripts)')

    args = parser.parse_args()
    
    verbose = not args.print_only_path
    best_fold, data_split_path, val_r_mean = parse_best_fold(args.results_summary, args.output_dir, args.seed, verbose=verbose)
    
    if args.print_only_path:
        print(data_split_path)


if __name__ == "__main__":
    main()
