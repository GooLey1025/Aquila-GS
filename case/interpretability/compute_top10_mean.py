#!/usr/bin/env python3
"""
Compute mean and variance of position_importance across top 10 trials.
Usage: python compute_top10_mean.py <top10_dir>
"""

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compute mean and variance across top trials")
    parser.add_argument("top10_dir", help="Directory containing trial_X/position_importance/ subdirectories")
    parser.add_argument("--trait", required=True, help="Trait name (e.g., HD_BLUP)")
    parser.add_argument("-o", "--output", help="Output file path")
    return parser.parse_args()


def extract_trials_from_dir(base_dir):
    """Find all trial directories with position_importance."""
    trials = []
    pattern = re.compile(r"trial_(\d+)")
    for d in sorted(glob.glob(os.path.join(base_dir, "trial_*"))):
        match = pattern.search(d)
        if match:
            trials.append(int(match.group(1)))
    return sorted(trials)


def main():
    args = parse_args()
    base_dir = os.path.abspath(args.top10_dir)
    trait = args.trait
    
    # Find all trial directories
    trials = extract_trials_from_dir(base_dir)
    print(f"Found {len(trials)} trials: {trials}")
    
    # Load all importance files
    imp_files = []
    for trial in trials:
        imp_file = os.path.join(base_dir, f"trial_{trial}", f"importance_ranking_{trait}.tsv")
        if os.path.exists(imp_file):
            imp_files.append((trial, imp_file))
        else:
            print(f"  [WARN] Missing: {imp_file}")
    
    if not imp_files:
        print("Error: No importance files found")
        return
    
    print(f"Loading {len(imp_files)} importance files...")
    
    # Parse locus_id to extract chr and pos
    dfs = []
    for trial, fpath in imp_files:
        df = pd.read_csv(fpath, sep='\t')
        df['snp_parts'] = df['locus_id'].str.split('-')
        df['chr'] = df['snp_parts'].str[1].astype(int)
        df['pos'] = df['snp_parts'].str[2].astype(int)
        df = df[['locus_id', 'chr', 'pos', 'importance']].copy()
        df = df.rename(columns={'importance': f'imp_{trial}'})
        dfs.append(df)
    
    # Start with locus_id, chr, pos from first file
    merged = dfs[0][['locus_id', 'chr', 'pos']].copy()
    
    # Merge all by locus_id
    for df in dfs:
        merged = merged.merge(df, on=['locus_id', 'chr', 'pos'], how='outer')
    
    # Fill NaN with 0 for missing values
    imp_cols = [f'imp_{trial}' for trial, _ in imp_files]
    merged[imp_cols] = merged[imp_cols].fillna(0)

    # Per-trial min-max normalization before averaging across trials
    for col in imp_cols:
        col_min = merged[col].min()
        col_max = merged[col].max()
        if col_max > col_min:
            merged[col] = (merged[col] - col_min) / (col_max - col_min)
        else:
            merged[col] = 0.0

    # Compute mean and variance (now across normalized values)
    merged['importance_mean'] = merged[imp_cols].mean(axis=1)
    merged['importance_std'] = merged[imp_cols].std(axis=1)
    merged['importance_var'] = merged[imp_cols].var(axis=1)
    merged['importance_count'] = (merged[imp_cols] > 0).sum(axis=1)
    
    # Sort by mean importance (descending) and assign rank
    merged = merged.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    merged['rank'] = range(1, len(merged) + 1)
    
    # Sort by chr and pos for Manhattan plot
    merged_for_plot = merged.sort_values(['chr', 'pos'])
    
    # Output files
    if args.output:
        out_dir = os.path.dirname(args.output) or '.'
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = base_dir
    
    # Save mean+var file (for plotting)
    mean_var_file = os.path.join(out_dir, f"importance_ranking_{trait}_mean.tsv")
    merged[['rank', 'locus_id', 'importance_mean', 'importance_std', 'importance_var', 'importance_count']].to_csv(
        mean_var_file, sep='\t', index=False
    )
    print(f"Saved: {mean_var_file}")
    
    # Save simple file for gwas_ig_plot.py compatibility
    simple_file = os.path.join(out_dir, f"importance_ranking_{trait}.tsv")
    merged_for_plot[['rank', 'locus_id', 'importance_mean']].rename(
        columns={'importance_mean': 'importance'}
    ).to_csv(simple_file, sep='\t', index=False)
    print(f"Saved: {simple_file}")
    
    # Summary
    print(f"\n--- Summary ---")
    print(f"Total loci: {len(merged)}")
    print(f"Mean importance: {merged['importance_mean'].mean():.6f}")
    print(f"Std (across trials): {merged['importance_std'].mean():.6f}")
    print(f"Min: {merged['importance_mean'].min():.6f}")
    print(f"Max: {merged['importance_mean'].max():.6f}")
    
    # Top 10
    print(f"\n--- Top 10 by Mean Importance ---")
    print(merged[['rank', 'locus_id', 'importance_mean', 'importance_std', 'importance_count']].head(10).to_string(index=False))
    
    return simple_file


if __name__ == "__main__":
    main()
