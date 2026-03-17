#!/usr/bin/env python3
"""
Aquila IG Interpretation Script

Analyze integrated gradients results to identify important genomic loci.
Outputs ranked locus importance scores for each task.

Usage:
    python aquila_ig_interpretation.py -i ig_results.h5 -o output_dir
    python aquila_ig_interpretation.py -i ig_results.h5 -o output_dir --tasks GYP_BLUP,GW_BLUP --top-k 50
"""

import argparse
import h5py
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze integrated gradients results for locus importance'
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input HDF5 file (from aquila_ig.py)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Output directory for analysis results'
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        default='all',
        help='Comma-separated list of tasks to analyze (default: all)'
    )
    
    parser.add_argument(
        '--variant-type',
        type=str,
        default='snp',
        help='Variant type to analyze (default: snp)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of top important loci to output (default: 100)'
    )
    
    parser.add_argument(
        '--aggregation',
        type=str,
        default='abs_sum',
        choices=['mean', 'sum', 'abs_sum', 'abs_mean', 'max'],
        help='Method to aggregate 8D IG scores to 1D (default: abs_sum)'
    )
    
    parser.add_argument(
        '--sample-mode',
        type=str,
        default='mean',
        choices=['mean', 'max', 'sum'],
        help='Method to aggregate across samples (default: mean)'
    )
    
    return parser.parse_args()


def aggregate_ig_scores(ig_scores: np.ndarray, method: str = 'abs_sum') -> np.ndarray:
    """
    Aggregate 8-dimensional IG scores to 1-dimensional.
    
    Args:
        ig_scores: Array of shape (num_variants, 8)
        method: Aggregation method
    
    Returns:
        Array of shape (num_variants,)
    """
    if method == 'abs_sum':
        return np.abs(ig_scores).sum(axis=1)
    elif method == 'abs_mean':
        return np.abs(ig_scores).mean(axis=1)
    elif method == 'sum':
        return ig_scores.sum(axis=1)
    elif method == 'mean':
        return ig_scores.mean(axis=1)
    elif method == 'max':
        return np.abs(ig_scores).max(axis=1)
    else:
        return np.abs(ig_scores).sum(axis=1)


def analyze_locus_importance(
    h5_path: str,
    tasks: List[str],
    variant_type: str,
    agg_method: str,
    sample_mode: str
) -> Dict[str, Dict]:
    """
    Analyze locus importance from IG results.
    
    Args:
        h5_path: Path to HDF5 file
        tasks: List of task names to analyze
        variant_type: Variant type (e.g., 'snp')
        agg_method: Method to aggregate 8D to 1D
        sample_mode: Method to aggregate across samples
    
    Returns:
        Dictionary mapping task_name to importance data
    """
    results = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Get metadata
        num_samples = f.attrs['num_samples']
        sample_ids = [s.decode('utf-8') for s in f['sample_ids'][:]]
        
        # Get variant IDs
        variant_ids = None
        if f'variant_ids/{variant_type}' in f:
            variant_ids = [v.decode('utf-8') for v in f[f'variant_ids/{variant_type}'][:]]
        
        # Process each task
        for task_name in tasks:
            print(f"  Processing task: {task_name}")
            
            # Collect IG scores for this task across all samples
            task_scores = []
            for sample_idx in range(num_samples):
                sample_key = f'sample_{sample_idx}'
                data_key = f'{sample_key}/{task_name}/{variant_type}'
                
                if data_key in f:
                    ig_scores = f[data_key][:]  # shape: (num_variants, 8)
                    
                    # Aggregate 8D -> 1D
                    agg_scores = aggregate_ig_scores(ig_scores, agg_method)
                    task_scores.append(agg_scores)
            
            if not task_scores:
                print(f"    Warning: No data found for task {task_name}")
                continue
            
            # Stack and aggregate across samples
            all_scores = np.stack(task_scores, axis=0)  # shape: (num_samples, num_variants)
            
            if sample_mode == 'mean':
                importance = all_scores.mean(axis=0)
            elif sample_mode == 'max':
                importance = all_scores.max(axis=0)
            elif sample_mode == 'sum':
                importance = all_scores.sum(axis=0)
            else:
                importance = all_scores.mean(axis=0)
            
            # Get ranking
            ranking_indices = np.argsort(importance)[::-1]  # Descending order
            
            results[task_name] = {
                'importance': importance,
                'ranking_indices': ranking_indices,
                'variant_ids': variant_ids,
                'num_variants': len(importance),
                'num_samples': num_samples,
                'sample_ids': sample_ids,
            }
    
    return results


def save_results(results: Dict, output_dir: Path, top_k: int):
    """
    Save analysis results to files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save top K loci for each task
    for task_name, data in results.items():
        importance = data['importance']
        ranking_indices = data['ranking_indices']
        variant_ids = data['variant_ids']
        
        # Get top K
        top_indices = ranking_indices[:top_k]
        top_importance = importance[top_indices]
        
        # Create DataFrame
        rows = []
        for rank, (idx, imp) in enumerate(zip(top_indices, top_importance), 1):
            if variant_ids:
                locus_id = variant_ids[idx]
            else:
                locus_id = f"locus_{idx}"
            
            rows.append({
                'rank': rank,
                'locus_id': locus_id,
                'importance': imp,
                'locus_index': idx
            })
        
        df = pd.DataFrame(rows)
        
        # Save to TSV
        output_path = output_dir / f'top_{top_k}_loci_{task_name}.tsv'
        df.to_csv(output_path, sep='\t', index=False)
        print(f"  Saved: {output_path}")
    
    # 2. Save all loci ranking for each task
    for task_name, data in results.items():
        importance = data['importance']
        ranking_indices = data['ranking_indices']
        variant_ids = data['variant_ids']
        
        rows = []
        for rank, idx in enumerate(ranking_indices, 1):
            if variant_ids:
                locus_id = variant_ids[idx]
            else:
                locus_id = f"locus_{idx}"
            
            rows.append({
                'rank': rank,
                'locus_id': locus_id,
                'importance': importance[idx],
                'locus_index': idx
            })
        
        df = pd.DataFrame(rows)
        
        # Save to TSV
        output_path = output_dir / f'importance_ranking_{task_name}.tsv'
        df.to_csv(output_path, sep='\t', index=False)
        print(f"  Saved: {output_path}")
    
    # 3. Save task summary JSON
    summary = {}
    for task_name, data in results.items():
        importance = data['importance']
        
        summary[task_name] = {
            'num_variants': int(data['num_variants']),
            'num_samples': int(data['num_samples']),
            'sample_ids': data['sample_ids'],
            'top_10_loci': [
                {
                    'rank': i + 1,
                    'locus_id': data['variant_ids'][data['ranking_indices'][i]] if data['variant_ids'] else f"locus_{data['ranking_indices'][i]}",
                    'importance': float(importance[data['ranking_indices'][i]])
                }
                for i in range(min(10, len(data['ranking_indices'])))
            ],
            'importance_stats': {
                'mean': float(importance.mean()),
                'std': float(importance.std()),
                'min': float(importance.min()),
                'max': float(importance.max()),
                'median': float(np.median(importance))
            }
        }
    
    summary_path = output_dir / 'task_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    
    # 4. Save combined ranking (average across tasks)
    if len(results) > 1:
        combined_importance = None
        for task_name, data in results.items():
            if combined_importance is None:
                combined_importance = data['importance'].copy()
            else:
                combined_importance += data['importance']
        
        combined_importance /= len(results)
        
        ranking_indices = np.argsort(combined_importance)[::-1]
        variant_ids = list(results.values())[0]['variant_ids']
        
        rows = []
        for rank, idx in enumerate(ranking_indices, 1):
            if variant_ids:
                locus_id = variant_ids[idx]
            else:
                locus_id = f"locus_{idx}"
            
            rows.append({
                'rank': rank,
                'locus_id': locus_id,
                'combined_importance': combined_importance[idx],
                'locus_index': idx
            })
        
        df = pd.DataFrame(rows)
        output_path = output_dir / 'combined_importance_ranking.tsv'
        df.to_csv(output_path, sep='\t', index=False)
        print(f"  Saved: {output_path}")
        
        # Top K combined
        top_combined = df.head(top_k)
        output_path = output_dir / f'combined_top_{top_k}_loci.tsv'
        top_combined.to_csv(output_path, sep='\t', index=False)
        print(f"  Saved: {output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 80)
    print("AQUILA IG Interpretation")
    print("=" * 80)
    
    # Validate input
    h5_path = Path(args.input)
    if not h5_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output_dir}")
    
    # Get available tasks from HDF5
    with h5py.File(args.input, 'r') as f:
        available_tasks = [t.decode('utf-8') for t in f['task_names'][:]]
    
    print(f"\nAvailable tasks: {len(available_tasks)}")
    
    # Determine tasks to analyze
    if args.tasks == 'all':
        tasks_to_analyze = available_tasks
    else:
        tasks_to_analyze = [t.strip() for t in args.tasks.split(',')]
        # Validate
        for task in tasks_to_analyze:
            if task not in available_tasks:
                raise ValueError(f"Task '{task}' not found in HDF5. Available: {available_tasks}")
    
    print(f"Tasks to analyze: {tasks_to_analyze}")
    print(f"Aggregation method: {args.aggregation}")
    print(f"Sample mode: {args.sample_mode}")
    print(f"Top K: {args.top_k}")
    
    # Analyze
    print("\n" + "=" * 80)
    print("Analyzing locus importance")
    print("=" * 80)
    
    results = analyze_locus_importance(
        args.input,
        tasks_to_analyze,
        args.variant_type,
        args.aggregation,
        args.sample_mode
    )
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    save_results(results, output_dir, args.top_k)
    
    print(f"\nAll results saved to: {args.output_dir}")
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
