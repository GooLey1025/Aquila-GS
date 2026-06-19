#!/usr/bin/env python3
"""
Aquila Directed Evolution Script

Single-seed CLI.  All reusable logic lives in src/aquila/evolve.py.
This script contains only: argument parsing, log writers, and the run_evolution orchestrator.

Usage:
    python aquila_evolve.py --model-dir path/to/model --vcf input.vcf.gz --pheno GYP
    python aquila_evolve.py --model-dir path/to/model --vcf input.vcf.gz --direction-file trait_direction.tsv
    python aquila_evolve.py --model-dir path/to/model --vcf input.vcf.gz \\
        --direction-file trait_direction.tsv --sites-to-evolve snp_list.txt \\
        --output-dir output --strategy combinatorial --iterations 300
"""

import argparse
import gzip
import json
import time
from math import isnan
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Reusable core logic lives here
from aquila.evolve import (
    DirectionModes,
    RoundResult,
    clone_genotype_tensors,
    compute_selection_index_normalized,
    create_model,
    denormalize_predictions,
    get_encoding_dim,
    load_checkpoint,
    load_model_and_config,
    load_vcf_data,
    predict_all_phenos,
    strategy_combinatorial,
    strategy_screening,
)

# Encoding-level utilities still needed here
from aquila.encoding import parse_genotype_file, write_evolved_vcf
from aquila.utils import load_config


###############################################################################
# Command-line argument parsing
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description='Aquila-GS Directed Evolution: optimize phenotype via SNP mutation'
    )
    parser.add_argument(
        '--model-dir', type=str, required=True,
        help='Trained model directory containing checkpoint, params.yaml, and normalization_stats.pkl'
    )
    parser.add_argument(
        '--vcf', type=str, required=True,
        help='Input VCF file (single sample recommended)'
    )
    parser.add_argument(
        '--pheno', type=str, default=None,
        help='Target phenotype name to optimize. '
             'If not provided: optimize all phenotypes jointly using a selection index '
             '(per-trait direction defined in --direction-file; traits not in the file default to neutral).'
    )
    parser.add_argument(
        '--output-vcf', type=str, default=None,
        help='Output VCF path. Default: input name + "_evolve.vcf.gz"'
    )
    parser.add_argument(
        '--output-pred', type=str, default=None,
        help='Output prediction TSV path. Default: input name + "_evolve_predictions.tsv"'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for all results. Default: same directory as input VCF'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device: cuda or cpu (default: cuda if available)'
    )
    parser.add_argument(
        '--mode', type=str, default='maximize',
        choices=['maximize', 'minimize'],
        help='Optimization direction when --pheno is provided (default: maximize)'
    )
    parser.add_argument(
        '--direction-file', type=str, default=None,
        help='Trait direction file (no header, whitespace-separated). '
             'Supports two formats: '
             '(1) String keywords: maximize / minimize / neutral / maintain. '
             '(2) Numeric values: a float representing the target percentage change '
             'relative to baseline (e.g. -2.954240 means "target = baseline * (1 - 0.02954240)"). '
             'Auto-detected: lines with parseable floats are treated as numeric targets. '
             'Used only in multi-pheno mode (no --pheno). '
             'All tasks without an entry default to neutral. '
             'Format: <phenotype> <task_type> <direction>'
    )
    parser.add_argument(
        '--min-improve', type=float, default=0,
        help='Minimum phenotype improvement to accept a mutation (default: %(default)s)'
    )
    parser.add_argument(
        '--top-k', type=int, default=None,
        help='Top-K cap for screening mode. '
             'If not specified: apply ALL SNPs with positive gain. '
             'If specified: cap at this many SNPs. '
             'Combinatorial mode always uses this value (default: 8 when omitted).'
    )
    parser.add_argument(
        '--mc-samples', type=int, default=1,
        help='Number of random genetic backgrounds for interaction-aware SNP gain estimation. '
             'default=1 falls back to original marginal screening. '
             '>=2 activates Monte-Carlo combinatorial marginal contribution. '
             'Recommended: 16-64 for interaction-aware screening.'
    )
    parser.add_argument(
        '--strategy', type=str, default='screening',
        choices=['screening', 'combinatorial'],
        help='Evolution strategy: screening (MC interaction-aware SNP screening), '
             'combinatorial (random+combo)'
    )
    parser.add_argument(
        '--iterations', type=int, default=None,
        help='Fixed number of evolution iterations. '
             'If not specified: patience mode (stop when no improvement for N rounds). '
             'If specified: run exactly this many iterations.'
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Early stop patience: stop if no improvement for this many consecutive rounds '
             '(default: 10, only used in patience mode)'
    )
    parser.add_argument(
        '--max-iterations', type=int, default=100,
        help='Safety cap on maximum iterations in patience mode (default: %(default)s)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Batch size for model inference (default: %(default)s)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility in screening or combinatorial strategy (default: 42)'
    )
    parser.add_argument(
        '--phased', action='store_true', default=True,
        help='Write phased GT in output VCF (0|1 format, default: True)'
    )
    parser.add_argument(
        '--unphased', dest='phased', action='store_false',
        help='Write unphased GT in output VCF (0/1 format)'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help='Print detailed per-SNP screening results'
    )
    parser.add_argument(
        '--sites-to-evolve', type=str, default=None,
        help='Path to a list file (one SNP ID per line, matching VCF ID column). '
             'Only these sites will be considered for directed evolution. '
             'All other sites are locked. Default: all sites'
    )
    parser.add_argument(
        '--ref-vcf', type=str, default=None,
        help='Population VCF panel to constrain mutation candidates. '
             'Only genotypes observed in this panel will be used as mutation candidates. '
             'Only SNP sites are considered (8-dim diploid one-hot encoding).'
    )
    parser.add_argument(
        '--target-vcf', type=str, default=None,
        help='Target genome VCF for priority-based evolution. '
             'In screening mode, SNPs where input VCF differs from target VCF '
             'are evolved first, ordered by descending SI gain potential. '
             'Each round mutates one SNP.'
    )
    parser.add_argument(
        '--save-site-gain', action='store_true',
        help='Write per-SNP gain contribution log for each round in combinatorial mode. '
             'Outputs combinatorial_si_per_round.tsv (one row per round), '
             'combinatorial_site_max_gains.tsv (one row per SNP), and '
             'combinatorial_site_mc_details.tsv (MC interaction-aware gain stats, '
             'written when --mc-samples >= 2).'
    )
    parser.add_argument(
        '--save-all-rounds', action='store_true',
        help='Save VCF and intermediate files for every round. '
             'Useful for tracing the evolution path but slower and disk-intensive '
             '(one VCF per round in screening mode).'
    )
    parser.add_argument(
        '--homozygous', action='store_true', default=False,
        help='When enabled, only generate homozygous (REF/REF or ALT/ALT) genotypes '
             'as mutation candidates. No heterozygous (REF/ALT) genotypes will appear '
             'as candidates in SNP screening. Existing heterozygous genotypes remain '
             'unchanged unless mutated, in which case they become homozygous.'
    )
    return parser.parse_args()


###############################################################################
# Log writers (CLI-specific, not reusable)
###############################################################################

def write_screening_log(out_dir: str, round_num: int, log_dict: dict,
                        regression_tasks: List[str], verbose: bool = False):
    """Write per-SNP screening results for a round."""
    screen_dir = Path(out_dir) / "round_screening"
    screen_dir.mkdir(parents=True, exist_ok=True)
    log_path = screen_dir / f"round{round_num:03d}_screening.tsv"
    per_snp_gains = log_dict.get('per_snp_gains')
    snp_screening = log_dict.get('snp_screening')
    selected_indices = log_dict.get('selected_indices', [])
    n_tasks = len(regression_tasks)
    rows = []
    for snp_idx in selected_indices:
        row = {'snp_idx': snp_idx, 'gain': float(per_snp_gains[snp_idx]) if per_snp_gains is not None else 0.0}
        if snp_screening is not None and snp_idx < snp_screening.shape[0]:
            for t_idx in range(n_tasks):
                if t_idx < snp_screening.shape[1]:
                    row[regression_tasks[t_idx]] = float(snp_screening[snp_idx, t_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(log_path, sep='\t', index=False)
    if verbose:
        print(f"  Screening log saved to: {log_path}")


def write_combo_log(out_dir: str, round_num: int, log_dict: dict,
                    regression_tasks: List[str], verbose: bool = False):
    """Write combinatorial search log for a round — all phenotype predictions for every combo."""
    screen_dir = Path(out_dir) / "round_screening"
    screen_dir.mkdir(parents=True, exist_ok=True)
    log_path = screen_dir / f"round{round_num:03d}_combos.tsv"
    target_preds = log_dict.get('target_preds', np.array([]))
    all_preds = log_dict.get('all_preds', np.zeros((0, len(regression_tasks))))
    gains = log_dict.get('gains', np.array([]))
    combo_masks = log_dict.get('combo_masks', np.zeros((0, 0)))
    snp_sets = log_dict.get('snp_sets', [])
    n_combos = len(target_preds)
    k = combo_masks.shape[1] if combo_masks.size > 0 else 0
    rows = []
    for i in range(n_combos):
        mask = combo_masks[i] if i < len(combo_masks) and combo_masks.size > 0 else []
        selected_snps = [snp_sets[j][0] for j in range(len(mask)) if mask[j] == 1] if len(mask) > 0 else []
        row = {
            'round': round_num, 'combo_idx': i,
            'gain': float(gains[i]) if i < len(gains) else 0.0,
            'target_pred': float(target_preds[i]) if i < len(target_preds) else 0.0,
            'selected_snp_count': int(sum(mask)) if len(mask) > 0 else 0,
            'selected_snps': ','.join(map(str, selected_snps)),
        }
        for t_idx, task in enumerate(regression_tasks):
            if t_idx < all_preds.shape[1]:
                row[f'{task}_pred'] = float(all_preds[i, t_idx])
        for j in range(k):
            row[f'snp_{j}_idx']   = snp_sets[j][0] if j < len(snp_sets) else -1
            row[f'snp_{j}_gain']  = snp_sets[j][1] if j < len(snp_sets) else 0.0
            row[f'snp_{j}_sel']   = int(mask[j])   if j < len(mask) else 0
        rows.append(row)
    pd.DataFrame(rows).to_csv(log_path, sep='\t', index=False)
    if verbose:
        print(f"  Combo log saved to: {log_path}")


def append_cumulative_combo_log(out_dir: str, round_num: int, log_dict: dict,
                                regression_tasks: List[str]):
    """Append cumulative combo results across ALL rounds to a single TSV."""
    log_path = Path(out_dir) / "round_screening" / "all_rounds_combos.tsv"
    target_preds = log_dict.get('target_preds', np.array([]))
    all_preds    = log_dict.get('all_preds', np.zeros((0, len(regression_tasks))))
    gains        = log_dict.get('gains', np.array([]))
    combo_masks  = log_dict.get('combo_masks', np.zeros((0, 0)))
    snp_sets     = log_dict.get('snp_sets', [])
    n_combos = len(target_preds)
    k = combo_masks.shape[1] if combo_masks.size > 0 else 0
    rows = []
    for i in range(n_combos):
        mask = combo_masks[i] if i < len(combo_masks) and combo_masks.size > 0 else []
        selected_snps = [snp_sets[j][0] for j in range(len(mask)) if mask[j] == 1] if len(mask) > 0 else []
        row = {
            'round': round_num, 'combo_idx': i,
            'gain': float(gains[i]) if i < len(gains) else 0.0,
            'target_pred': float(target_preds[i]) if i < len(target_preds) else 0.0,
            'selected_snp_count': int(sum(mask)) if len(mask) > 0 else 0,
            'selected_snps': ','.join(map(str, selected_snps)),
        }
        for t_idx, task in enumerate(regression_tasks):
            if t_idx < all_preds.shape[1]:
                row[f'{task}_pred'] = float(all_preds[i, t_idx])
        for j in range(k):
            row[f'snp_{j}_idx']   = snp_sets[j][0] if j < len(snp_sets) else -1
            row[f'snp_{j}_gain']  = snp_sets[j][1] if j < len(snp_sets) else 0.0
            row[f'snp_{j}_sel']   = int(mask[j])   if j < len(mask) else 0
        rows.append(row)
    df_new = pd.DataFrame(rows)
    if log_path.exists():
        df_combined = pd.concat([pd.read_csv(log_path, sep='\t'), df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(log_path, sep='\t', index=False)


def build_round_detail(log_dict: dict, regression_tasks: List[str],
                       target_pheno_idx: int, current_score: float, mode: str) -> dict:
    """Build a detailed round dict from the log_dict returned by strategies."""
    combo_all_preds = log_dict.get('all_preds')
    combo_gains     = log_dict.get('gains')
    combo_masks     = log_dict.get('combo_masks')
    snp_sets        = log_dict.get('snp_sets', [])
    per_snp_gains   = log_dict.get('per_snp_gains')

    detail: dict = {
        'strategy':           log_dict.get('strategy', 'unknown'),
        'current_score':      float(current_score),
        'target_pheno_idx':  target_pheno_idx,
        'n_tasks':           len(regression_tasks),
    }
    if snp_sets:
        detail['top_k_snps'] = [{'snp_idx': int(s[0]), 'single_gain': float(s[1])} for s in snp_sets]
    if per_snp_gains is not None:
        detail['snp_gains'] = {int(i): float(g) for i, g in enumerate(per_snp_gains) if g != 0}

    if combo_all_preds is not None and len(combo_all_preds) > 0:
        n_combos = len(combo_all_preds)
        combos = []
        for i in range(n_combos):
            row = {task: float(combo_all_preds[i, t_idx]) for t_idx, task in enumerate(regression_tasks)}
            row['target_value'] = float(combo_all_preds[i, target_pheno_idx])
            row['target_gain']  = float(combo_gains[i]) if (combo_gains is not None and i < len(combo_gains)) else 0.0
            if combo_masks is not None and i < len(combo_masks):
                row['n_selected']            = int(np.sum(combo_masks[i]))
                row['selected_snp_indices']  = [int(snp_sets[j][0]) for j in range(len(combo_masks[i]))
                                                if combo_masks[i][j] == 1 and j < len(snp_sets)]
            combos.append(row)
        reverse = (mode == 'maximize')
        combos_sorted = sorted(combos, key=lambda x: x['target_gain'], reverse=reverse)
        detail['combos']          = combos
        detail['combos_ranked']   = combos_sorted
        detail['best_combo_idx']  = 0
        detail['best_combo']      = combos_sorted[0]
    else:
        detail['combos'] = detail['combos_ranked'] = []
        detail['best_combo_idx'] = -1
        detail['best_combo'] = None
    return detail


def write_phenotype_summary(out_dir: str, round_num: int, rr: RoundResult,
                             baseline_preds: np.ndarray, regression_tasks: List[str], pheno_name: str):
    """Append all-phenotype predictions per round to a summary TSV."""
    screen_dir = Path(out_dir) / "round_screening"
    screen_dir.mkdir(parents=True, exist_ok=True)
    log_path = screen_dir / "phenotype_summary.tsv"
    rows = []
    for t_idx, task in enumerate(regression_tasks):
        base_v = rr.all_phenotypes_baseline.get(task, float('nan'))
        evo_v  = rr.all_phenotypes_evolved.get(task, float('nan'))
        gain_v = evo_v - base_v if not (isnan(evo_v) or isnan(base_v)) else 0.0
        ggain  = evo_v - baseline_preds[t_idx] if not isnan(evo_v) else 0.0
        rows.append({
            'round': round_num, 'phenotype': task,
            'baseline': base_v, 'evolved': evo_v,
            'gain': gain_v, 'global_gain': ggain,
            'is_target': int(task == pheno_name),
        })
    df = pd.DataFrame(rows)
    if log_path.exists():
        df.to_csv(log_path, sep='\t', index=False, mode='a', header=False)
    else:
        df.to_csv(log_path, sep='\t', index=False, header=True)


def write_round_detail_json(out_dir: str, round_num: int, detail: dict):
    """Write per-round detailed JSON."""
    log_path = Path(out_dir) / "round_screening" / f"round{round_num:03d}_detail.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(detail, f, indent=2, default=str)


def write_cumulative_round_details(out_dir: str, round_num: int, detail: dict):
    """Append round detail as one JSON object per line (jsonl)."""
    log_path = Path(out_dir) / "round_details.jsonl"
    with open(log_path, 'a') as f:
        f.write(json.dumps(detail, default=str) + '\n')


def write_round_predictions(
    out_dir: str, round_num: int, sample_id: str,
    regression_tasks: List[str], baseline_preds: np.ndarray,
    preds_norm: np.ndarray, baseline_norm: np.ndarray,
    norm_stats, log_tasks: List[str],
):
    """Append round predictions to a cumulative TSV (normalized + denormalized)."""
    pred_path = Path(out_dir) / "round_predictions.tsv"
    preds_norm_flat = preds_norm.ravel()
    baseline_norm_flat = np.asarray(baseline_norm).ravel()

    has_norm = isinstance(norm_stats, dict) and 'regression_means' in norm_stats
    if has_norm:
        preds_norm_2d = preds_norm_flat.reshape(1, -1)
        baseline_norm_2d = baseline_norm_flat.reshape(1, -1)
        preds_denorm = denormalize_predictions(preds_norm_2d, norm_stats, regression_tasks, log_tasks)[0]
        baseline_denorm = denormalize_predictions(baseline_norm_2d, norm_stats, regression_tasks, log_tasks)[0]
    else:
        preds_denorm = preds_norm_flat
        baseline_denorm = baseline_norm_flat

    row_data = {'Round': round_num, 'Sample_ID': sample_id}

    for t_idx, task in enumerate(regression_tasks):
        if t_idx < len(preds_denorm):
            row_data[f'{task}_pred']       = preds_denorm[t_idx]
            row_data[f'{task}_pred_norm']  = preds_norm_flat[t_idx]
            row_data[f'{task}_baseline']   = baseline_denorm[t_idx]
            row_data[f'{task}_baseline_norm'] = baseline_norm_flat[t_idx]
            row_data[f'{task}_gain']       = preds_denorm[t_idx] - baseline_denorm[t_idx]
            row_data[f'{task}_gain_norm']  = preds_norm_flat[t_idx] - baseline_norm_flat[t_idx]

    write_header = not pred_path.exists() or pred_path.stat().st_size == 0
    with open(pred_path, 'a', newline='') as f:
        df_new = pd.DataFrame([row_data])
        df_new.to_csv(f, sep='\t', index=False, header=write_header)


def write_intermediate_round_progress(
    out_dir: str, round_num: int, round_result: RoundResult,
    regression_tasks: List[str], baseline_preds: np.ndarray,
):
    """Write detailed per-round progress file (baseline + evolved state rows)."""
    progress_path = Path(out_dir) / "round_progress.tsv"
    rr = round_result
    base_row = {
        'round': round_num, 'stage': 'baseline',
        'accepted': 'accepted' if rr.accepted else 'rejected',
        'gain': 0.0, 'n_mutations': 0, 'mutated_snps': '',
    }
    for t in regression_tasks:
        bv = rr.all_phenotypes_baseline.get(t, float('nan'))
        base_row[f'{t}_val'] = bv
        base_row[f'{t}_gain_from_global'] = bv - baseline_preds[regression_tasks.index(t)]

    evo_row = {
        'round': round_num, 'stage': 'evolved',
        'accepted': 'accepted' if rr.accepted else 'rejected',
        'gain': rr.gain, 'n_mutations': len(rr.mutations),
        'mutated_snps': ','.join(str(s) for s, _ in rr.mutations),
    }
    for t in regression_tasks:
        ev = rr.all_phenotypes_evolved.get(t, float('nan'))
        bv = rr.all_phenotypes_baseline.get(t, float('nan'))
        evo_row[f'{t}_val'] = ev
        evo_row[f'{t}_gain_from_baseline'] = ev - bv if not (isnan(ev) or isnan(bv)) else float('nan')
        evo_row[f'{t}_gain_from_global']   = ev - baseline_preds[regression_tasks.index(t)] if not isnan(ev) else float('nan')

    write_header = not progress_path.exists() or progress_path.stat().st_size == 0
    with open(progress_path, 'a', newline='') as f:
        pd.DataFrame([base_row, evo_row]).to_csv(f, sep='\t', index=False, header=write_header)


def write_intermediate_combo_progress(
    out_dir: str, round_num: int, combo_idx: int, combo_preds: np.ndarray,
    combo_mask: np.ndarray, snp_sets: List[Tuple[int, float]],
    current_score: float, baseline_preds: np.ndarray,
    regression_tasks: List[str], mode: str, target_pheno_idx: int,
):
    """Append per-combo intermediate results to a cumulative TSV."""
    combo_progress_path = Path(out_dir) / "round_combos_progress.tsv"
    n_combos = len(combo_preds)
    rows = []
    for i in range(n_combos):
        selected_snps = [snp_sets[j][0] for j in range(len(combo_mask[i])) if combo_mask[i][j] == 1]
        target_pred = combo_preds[i, target_pheno_idx]
        gain = (target_pred - current_score) if mode == 'maximize' else (current_score - target_pred)
        row = {
            'round': round_num, 'combo_idx': i,
            'n_selected': int(np.sum(combo_mask[i])),
            'gain': gain, 'target_pred': target_pred,
            'selected_snps': ','.join(map(str, selected_snps)),
        }
        for t_idx, task in enumerate(regression_tasks):
            row[f'{task}_pred'] = combo_preds[i, t_idx]
        rows.append(row)
    write_header = not combo_progress_path.exists() or combo_progress_path.stat().st_size == 0
    with open(combo_progress_path, 'a', newline='') as f:
        pd.DataFrame(rows).to_csv(f, sep='\t', index=False, header=write_header)


def write_summary_log(log_path: str, round_results: List[RoundResult],
                      regression_tasks: List[str], pheno_name: str):
    """Write comprehensive multi-section summary TSV."""
    out_dir = Path(log_path).parent
    lines = ["# ==" * 20 + " SECTION 1: Per-round SI " + "=="] * 1
    lines.append("# Round\tSI_baseline\tSI_evolved\tSI_gain\tAccepted")
    for rr in round_results:
        lines.append(f"{rr.round_num}\t{rr.si_baseline:.6f}\t{rr.si_evolved:.6f}\t"
                     f"{rr.si_gain:.6f}\t{int(rr.accepted)}")

    lines.append("\n# == SECTION 2: Top SNPs per round ==")
    lines.append("# Round\tRank\tSNP_idx\tSingle_gain")
    for rr in round_results:
        gains = rr.snp_gains
        if not isinstance(gains, dict):
            gains = {}
        for rank, (snp_idx, gain) in enumerate(sorted(gains.items(), key=lambda x: -x[1])):
            lines.append(f"{rr.round_num}\t{rank+1}\t{snp_idx}\t{gain:.6f}")

    lines.append("\n# == SECTION 3: All phenotype values per round ==")
    lines.append('#\t'.join([''] + regression_tasks))
    for rr in round_results:
        vals = [rr.all_phenotypes_evolved.get(t, float('nan'))
                if isinstance(rr.all_phenotypes_evolved, dict) else float('nan')
                for t in regression_tasks]
        lines.append(f"{rr.round_num}\t" + '\t'.join(f"{v:.4f}" for v in vals))

    lines.append("\n# == SECTION 4: Per-SNP gains ==")
    lines.append("# SNP_idx\tMax_gain\tMean_gain\tN_rounds")
    snp_gain_sums: Dict[int, float] = {}
    snp_gain_counts: Dict[int, int] = {}
    for rr in round_results:
        gains = rr.snp_gains
        if not isinstance(gains, dict):
            gains = {}
        for snp_idx, gain in gains.items():
            snp_gain_sums[snp_idx]   = snp_gain_sums.get(snp_idx, 0.0) + gain
            snp_gain_counts[snp_idx]  = snp_gain_counts.get(snp_idx, 0) + 1
    for snp_idx in sorted(snp_gain_sums):
        cnt = snp_gain_counts[snp_idx]
        lines.append(f"{snp_idx}\t{snp_gain_sums[snp_idx]:.6f}\t"
                     f"{snp_gain_sums[snp_idx]/cnt:.6f}\t{cnt}")

    lines.append("\n# == SECTION 5: Combo results summary ==")
    lines.append("# Round\tN_combos\tBest_combo_idx\tBest_gain")
    for rr in round_results:
        n_combos = len(rr.combo_results)
        if n_combos > 0:
            best_idx = max(rr.combo_results, key=lambda k: rr.combo_results[k].get('_gain', 0))
            best_gain = rr.combo_results[best_idx].get('_gain', 0)
            lines.append(f"{rr.round_num}\t{n_combos}\t{best_idx}\t{best_gain:.6f}")

    lines.append("\n# == SECTION 6: Mutations applied ==")
    lines.append("# Round\tMutation_idx\tSNP_idx\tEncoding")
    for rr in round_results:
        for mut_idx, (snp_idx, enc, gain) in enumerate(rr.mutations):
            enc_str = ' '.join(f"{x:.3f}" for x in enc.cpu().numpy().flatten()[:4]) if hasattr(enc, 'cpu') else str(enc)
            lines.append(f"{rr.round_num}\t{mut_idx}\t{snp_idx}\t{enc_str}")

    lines.append("\n# == SECTION 7: Evolution trajectories ==")
    lines.append("# Round\tSI")
    for rr in round_results:
        lines.append(f"{rr.round_num}\t{rr.si_evolved:.6f}")

    with open(log_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_screening_si_log(out_dir: str, round_results: List[RoundResult],
                           regression_tasks: List[str]):
    """Write screening mode SI logs."""
    out_path = Path(out_dir)
    si_path = out_path / "screening_si_per_round.tsv"
    max_path = out_path / "screening_snp_max_gains.tsv"

    # Header matches user's expected format
    si_header = ["#Round", "Evaluated_SNP", "Evaluated_SNP_ID",
                 "SI_baseline", "SI_evolved", "SI_gain",
                 "SI_original_contribution",
                 "score_baseline", "score_evolved", "accepted"]
    si_lines = ["\t".join(si_header)]

    snp_max_sums: Dict[int, float] = {}
    snp_max_counts: Dict[int, int] = {}
    for rr in round_results:
        # Convert SNP idx to chr_pos format (e.g. chr1_174554)
        snp_display = rr.evaluated_snp_id.replace("SNP-", "").replace("-", "_") \
            if rr.evaluated_snp_id else str(rr.evaluated_snp_idx)
        si_lines.append(
            f"{rr.round_num}\t{snp_display}\t{rr.evaluated_snp_id}\t"
            f"{rr.si_baseline:.3f}\t{rr.si_evolved:.3f}\t{rr.si_gain:+.3f}\t"
            f"{rr.si_original_contribution:+.3f}\t"
            f"{rr.score_baseline:.2f}\t{rr.score_evolved:.2f}\t"
            f"{rr.accepted}"
        )
        if rr.evaluated_snp_idx >= 0:
            gain = rr.si_original_contribution
            if rr.evaluated_snp_idx not in snp_max_sums:
                snp_max_sums[rr.evaluated_snp_idx] = gain
                snp_max_counts[rr.evaluated_snp_idx] = 1
            else:
                if gain > snp_max_sums[rr.evaluated_snp_idx]:
                    snp_max_sums[rr.evaluated_snp_idx] = gain
                snp_max_counts[rr.evaluated_snp_idx] += 1
    with open(si_path, 'w') as f:
        f.write('\n'.join(si_lines) + '\n')
    max_lines = ["# snp_idx\tmax_original_contribution\tn_rounds"]
    for snp_idx in sorted(snp_max_sums):
        max_lines.append(f"{snp_idx}\t{snp_max_sums[snp_idx]:.6f}\t{snp_max_counts[snp_idx]}")
    with open(max_path, 'w') as f:
        f.write('\n'.join(max_lines) + '\n')


def _per_snp_contribution_in_round(rr: RoundResult) -> Dict[int, float]:
    """Decompose round SI gain into per-SNP marginal contributions."""
    from aquila.evolve import per_snp_contribution_in_round as _impl
    return _impl(rr)


def write_combinatorial_site_contributions(out_dir: str, round_results: List[RoundResult],
                                            variant_info: dict):
    """Write combinatorial mode per-SNP SI contribution logs."""
    out_path = Path(out_dir)
    snp_ids = variant_info.get('snp', {}).get('ids', [])
    contrib_sum: Dict[int, float] = {}
    contrib_rounds: Dict[int, List[int]] = {}
    mc_contrib_sum: Dict[int, float] = {}
    mc_contrib_rounds: Dict[int, List[int]] = {}

    for rr in round_results:
        if not rr.accepted:
            continue
        per_round = _per_snp_contribution_in_round(rr)
        for snp_idx, contrib in per_round.items():
            contrib_sum[snp_idx]   = contrib_sum.get(snp_idx, 0.0) + contrib
            contrib_rounds.setdefault(snp_idx, []).append(rr.round_num)
        if rr.snp_mc_details:
            for snp_idx, mc in rr.snp_mc_details.items():
                if 'mean' in mc:
                    mc_val = float(mc['mean'])
                    mc_contrib_sum[snp_idx]   = mc_contrib_sum.get(snp_idx, 0.0) + mc_val
                    mc_contrib_rounds.setdefault(snp_idx, []).append(rr.round_num)

    contrib_path = out_path / "combinatorial_site_contributions.tsv"
    lines = ["# snp_idx\tsnp_id\tcontribution\tavg_contribution\trounds_seen\tfirst_round\tlast_round"]
    for snp_idx in sorted(contrib_sum.keys()):
        total = contrib_sum[snp_idx]
        rounds = sorted(contrib_rounds[snp_idx])
        avg = total / len(rounds)
        snp_id = snp_ids[snp_idx] if 0 <= snp_idx < len(snp_ids) else ""
        lines.append(f"{snp_idx}\t{snp_id}\t{total:+.6f}\t{avg:+.6f}\t{len(rounds)}\t{rounds[0]}\t{rounds[-1]}")
    with open(contrib_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Per-SNP contribution log saved to: {contrib_path}")

    if mc_contrib_sum:
        mc_path = out_path / "combinatorial_site_contributions_mc.tsv"
        mc_lines = ["# snp_idx\tsnp_id\tmc_contribution\tavg_mc_contribution\trounds_seen\tfirst_round\tlast_round"]
        for snp_idx in sorted(mc_contrib_sum.keys()):
            total = mc_contrib_sum[snp_idx]
            rounds = sorted(mc_contrib_rounds[snp_idx])
            avg = total / len(rounds)
            snp_id = snp_ids[snp_idx] if 0 <= snp_idx < len(snp_ids) else ""
            mc_lines.append(f"{snp_idx}\t{snp_id}\t{total:+.6f}\t{avg:+.6f}\t{len(rounds)}\t{rounds[0]}\t{rounds[-1]}")
        with open(mc_path, 'w') as f:
            f.write('\n'.join(mc_lines) + '\n')


def write_combinatorial_snp_gains(out_dir: str, round_results: List[RoundResult],
                                   regression_tasks: List[str], variant_info: dict):
    """Write combinatorial mode per-round SI and per-SNP gain logs."""
    out_path = Path(out_dir)
    si_path = out_path / "combinatorial_si_per_round.tsv"
    max_path = out_path / "combinatorial_site_max_gains.tsv"
    mc_path  = out_path / "combinatorial_site_mc_details.tsv"

    snp_ids = variant_info.get('snp', {}).get('ids', [])

    # Section 1: SI per round
    si_lines = ["# round\tsi_baseline\tsi_evolved\tsi_gain\taccepted\tn_mutations"]
    for rr in round_results:
        si_lines.append(f"{rr.round_num}\t{rr.si_baseline:.6f}\t{rr.si_evolved:.6f}\t"
                        f"{rr.si_gain:.6f}\t{int(rr.accepted)}\t{len(rr.mutations)}")
    with open(si_path, 'w') as f:
        f.write('\n'.join(si_lines) + '\n')

    # Section 2: per-SNP max gains
    snp_sums: Dict[int, float] = {}
    snp_counts: Dict[int, int] = {}
    for rr in round_results:
        for snp_idx, gain in rr.snp_gains.items():
            snp_sums[snp_idx]   = snp_sums.get(snp_idx, 0.0) + gain
            snp_counts[snp_idx] = snp_counts.get(snp_idx, 0) + 1

    max_lines = ["# snp_idx\tsnp_id\ttotal_gain\tmean_gain\tn_rounds\tfirst_round\tlast_round"]
    first_round: Dict[int, int] = {}
    last_round: Dict[int, int] = {}
    for rr in round_results:
        for snp_idx in rr.snp_gains:
            if snp_idx not in first_round:
                first_round[snp_idx] = rr.round_num
            last_round[snp_idx] = rr.round_num

    for snp_idx in sorted(snp_sums.keys()):
        total = snp_sums[snp_idx]
        cnt   = snp_counts[snp_idx]
        snp_id = snp_ids[snp_idx] if 0 <= snp_idx < len(snp_ids) else ""
        max_lines.append(f"{snp_idx}\t{snp_id}\t{total:.6f}\t{total/cnt:.6f}\t{cnt}\t"
                         f"{first_round.get(snp_idx,0)}\t{last_round.get(snp_idx,0)}")
    with open(max_path, 'w') as f:
        f.write('\n'.join(max_lines) + '\n')

    # Section 3: MC interaction details
    mc_lines = ["# snp_idx\tsnp_id\tround\tmc_mean\tmc_std\tmc_median\tmc_max\tmc_min\tn_samples"]
    for rr in round_results:
        if not rr.snp_mc_details:
            continue
        for snp_idx, mc in rr.snp_mc_details.items():
            snp_id = snp_ids[snp_idx] if 0 <= snp_idx < len(snp_ids) else ""
            mc_lines.append(f"{snp_idx}\t{snp_id}\t{rr.round_num}\t"
                            f"{mc.get('mean',0):.6f}\t{mc.get('std',0):.6f}\t"
                            f"{mc.get('median',0):.6f}\t{mc.get('max',0):.6f}\t{mc.get('min',0):.6f}\t"
                            f"{mc.get('n_samples',0)}")
    if len(mc_lines) > 1:
        with open(mc_path, 'w') as f:
            f.write('\n'.join(mc_lines) + '\n')


###############################################################################
# Main evolution orchestrator
###############################################################################

def run_evolution(args):
    """Run the directed evolution loop."""
    from aquila.evolve import (
        apply_multi_snp_mutations,
        build_mutation_candidates_for_snp,
        compute_snp_priority_order,
        load_direction_file,
        load_panel_genotypes,
    )

    print("=" * 80)
    print("AQUILA-GS: Directed Evolution")
    print("=" * 80)
    print(f"  Strategy:    {args.strategy}")
    print(f"  Mode:       {'multi-phenotype (selection index, per-trait direction)' if args.pheno is None else args.mode}")
    print(f"  Min improve: {args.min_improve}")
    print(f"  Top-K:      {args.top_k}")
    print(f"  Homozygous: {args.homozygous}")
    if args.iterations is not None:
        print(f"  Iterations: {args.iterations} (fixed-round mode)")
    else:
        print(f"  Patience:   {args.patience} (patience mode, max {args.max_iterations})")
    print(f"  Device:     {args.device}")
    print(f"  Input VCF:  {args.vcf}")

    input_path = Path(args.vcf)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous cumulative files
    for fname in ("round_predictions.tsv", "round_details.jsonl"):
        fpath = out_dir / fname
        if fpath.exists():
            fpath.unlink()

    summary_filename = "evolve_summary.tsv"
    print(f"  Output dir:  {out_dir}")

    # --- Load model ---
    config, checkpoint_path, norm_stats = load_model_and_config(args.model_dir)
    print(f"\n[Model] Config: {checkpoint_path.parent}")

    encoding_type = config.get('data', {}).get('encoding_type', 'diploid_onehot')
    if encoding_type in ['snp_vcf', 'snp_indel_vcf', 'snp_indel_sv_vcf']:
        encoding_type = 'diploid_onehot'

    # --- Load VCF data ---
    variant_tensors, variant_info, sample_ids, seq_length, is_multi_branch = load_vcf_data(
        args.vcf, encoding_type, config
    )

    if len(sample_ids) > 1:
        print(f"\nWarning: Input VCF has {len(sample_ids)} samples. "
              f"Only the first sample will be evolved.")

    # --- Evolvable indices ---
    evolvable_indices: Optional[List[int]] = None
    if args.sites_to_evolve is not None:
        allowed_ids = set()
        with open(args.sites_to_evolve) as f:
            for line in f:
                sid = line.strip()
                if sid and not sid.startswith('#'):
                    allowed_ids.add(sid)
        all_ids = variant_info.get('snp', {}).get('ids', [])
        evolvable_indices = [i for i, sid in enumerate(all_ids) if sid in allowed_ids]
        if len(evolvable_indices) == 0:
            raise ValueError(
                f"No SNP IDs from '{args.sites_to_evolve}' matched the VCF. "
                f"Check that the list uses the same ID format as the VCF."
            )
        print(f"\n[Sites] {len(evolvable_indices)} evolvable sites from '{args.sites_to_evolve}' "
              f"(model still sees all {len(all_ids)} sites)")
    else:
        all_ids = variant_info.get('snp', {}).get('ids', [])
        print(f"\n[Sites] All {len(all_ids)} sites are evolvable")

    # --- Create model ---
    device = args.device
    model = create_model(config, seq_length, device, norm_stats)
    print("\n[Model] Initializing with dummy forward pass...")
    model.eval()
    with torch.no_grad():
        if is_multi_branch:
            dummy = {k: torch.randn(1, v.shape[1], v.shape[-1], device=device)
                     for k, v in variant_tensors.items()}
        else:
            t = variant_tensors
            if t.ndim == 3:
                dummy = torch.randn(1, t.shape[1], t.shape[2], device=device)
            else:
                dummy = torch.zeros(1, t.shape[1], dtype=torch.long, device=device)
        _ = model(dummy)

    model, _ = load_checkpoint(model, checkpoint_path, device)

    # --- Regression tasks ---
    regression_tasks = (
        norm_stats.get('regression_tasks') if isinstance(norm_stats, dict) else [])
    if not regression_tasks:
        regression_tasks = config.get('train', {}).get('regression_tasks', [])
    if not regression_tasks:
        regression_tasks = config.get('model', {}).get('regression_tasks', [])
    print(f"\n[Model] Regression tasks: {regression_tasks}")

    # --- Multi-pheno vs single-pheno ---
    use_selection_index = False
    pheno_idx = 0
    if args.pheno is None:
        use_selection_index = True
        print(f"\n[Target] Multi-phenotype optimization (selection index, equal weights)")
    else:
        if args.pheno not in regression_tasks:
            for i, task in enumerate(regression_tasks):
                if args.pheno.lower() in task.lower():
                    pheno_idx = i
                    print(f"  Closest match: {task} (index {i})")
                    break
            else:
                pheno_idx = 0
        else:
            pheno_idx = regression_tasks.index(args.pheno)
        print(f"\n[Target] Phenotype: {args.pheno} (index {pheno_idx})")

    # --- Initial prediction (needed before direction-mode computation) ---
    current_tensors = clone_genotype_tensors(variant_tensors, is_multi_branch)
    if is_multi_branch:
        init_batch = {k: v[0:1].to(device) for k, v in current_tensors.items()}
    else:
        init_batch = current_tensors[0:1].to(device)

    with torch.no_grad():
        outputs = model(init_batch)
    init_preds_norm = (
        outputs['regression'].cpu().numpy()[0]
        if 'regression' in outputs
        else torch.sigmoid(outputs['classification']).cpu().numpy()[0]
    )

    log_tasks = norm_stats.get('log_transformed_tasks', []) if isinstance(norm_stats, dict) else []
    if isinstance(norm_stats, dict) and 'regression_means' in norm_stats:
        init_preds = denormalize_predictions(
            init_preds_norm.reshape(1, -1), norm_stats, regression_tasks, log_tasks
        )[0]
    else:
        init_preds = init_preds_norm

    baseline_preds = init_preds.copy()
    baseline_preds_norm = init_preds_norm.copy()

    # --- Direction modes ---
    direction: Optional[DirectionModes] = None
    if args.direction_file is not None:
        if args.pheno is not None:
            raise ValueError("--direction-file is only valid in multi-pheno mode (omit --pheno)")
        dir_traits, dir_weights, dir_modes, dir_numeric = load_direction_file(args.direction_file)
        trait_to_weight = dict(zip(dir_traits, dir_weights))
        trait_to_mode   = dict(zip(dir_traits, dir_modes))
        trait_to_numeric = dict(zip(dir_traits, dir_numeric))
        aligned_weights  = np.array([trait_to_weight.get(t, 1.0) for t in regression_tasks])
        aligned_modes    = np.array([trait_to_mode.get(t, 'neutral')   for t in regression_tasks])
        aligned_numeric  = np.array([trait_to_numeric.get(t, 0.0) for t in regression_tasks])

        # For each numeric_target trait, compute:
        #   target_denorm = baseline_denorm * (1 + direction_pct / 100)
        #   target_z      = (target_denorm - mean) / std   <- stored in DirectionModes
        # The Z-score target is needed by compute_selection_index_normalized (which works
        # in Z-score space).  target_denorm is stored separately for human-readable logging.
        numeric_targets_norm = np.zeros(len(regression_tasks))
        numeric_targets_denorm_for_log = np.zeros(len(regression_tasks))
        is_numeric = False
        for t_idx, mode in enumerate(aligned_modes):
            if mode == 'numeric_target':
                pct = aligned_numeric[t_idx]
                baseline_actual = init_preds[t_idx]
                target_denorm = baseline_actual * (1.0 + pct / 100.0)
                numeric_targets_denorm_for_log[t_idx] = target_denorm
                if isinstance(norm_stats, dict) and 'regression_means' in norm_stats:
                    reg_means = norm_stats.get('regression_means', {})
                    reg_stds  = norm_stats.get('regression_stds', {})
                    t_name = regression_tasks[t_idx]
                    mean = reg_means.get(t_name, 0.0) if isinstance(reg_means, dict) else 0.0
                    std  = reg_stds.get(t_name, 1.0)  if isinstance(reg_stds, dict) else 1.0
                    if std == 0:
                        std = 1.0
                    numeric_targets_norm[t_idx] = (target_denorm - mean) / std
                else:
                    # No normalization stats: assume baseline is already in Z-score space.
                    numeric_targets_norm[t_idx] = baseline_actual
                is_numeric = True

        direction = DirectionModes(
            modes=aligned_modes,
            weights=aligned_weights,
            numeric_targets_norm=numeric_targets_norm,
            _is_numeric=is_numeric,
        )
        print(f"\n[Target] Direction modes from {args.direction_file}:")
        for t_idx, (t, m, w) in enumerate(zip(regression_tasks, aligned_modes, aligned_weights)):
            if m == 'numeric_target':
                pct = aligned_numeric[t_idx]
                print(f"  {t}: numeric_target ({pct:+.4f}%), target_denorm={numeric_targets_denorm_for_log[t_idx]:+.4f}")
            else:
                print(f"  {t}: {m} (weight={w})")

    if direction is None:
        direction = DirectionModes(
            modes=np.array(['maximize'] * len(regression_tasks)),
            weights=np.ones(len(regression_tasks)),
        )
        print(f"\n[Target] Direction: all traits maximize (default)")

    encoding_dim = get_encoding_dim(variant_tensors)
    print(f"[Encoding] Dimension: {encoding_dim}")

    if use_selection_index:
        current_score = float(compute_selection_index_normalized(
            init_preds_norm.reshape(1, -1), direction, init_preds_norm.reshape(1, -1)
        )[0])
    else:
        current_score = float(init_preds_norm[pheno_idx])

    print(f"\n[Baseline] Predictions (selection index, normalized space):")
    for t_idx, task in enumerate(regression_tasks):
        m = direction.modes[t_idx] if direction else 'maximize'
        print(f"  {task} ({m}): denorm={init_preds[t_idx]:.4f}, norm_z={init_preds_norm[t_idx]:+.4f}")
    print(f"  Selection index (z-space SI): {current_score:+.4f}")

    # --- Panel genotypes ---
    panel_genotypes = None
    if args.ref_vcf:
        n_snps = variant_tensors.shape[1] if not is_multi_branch else variant_tensors['snp'].shape[1]
        print(f"\n[Ref panel] Loading from {args.ref_vcf}...")
        panel_genotypes = load_panel_genotypes(args.ref_vcf, n_snps)
        print(f"  Loaded panel genotypes for {n_snps} SNPs")

    # --- SNP priority order (target-VCF mode) ---
    snp_priority_order: Optional[List[int]] = None
    if args.target_vcf is not None and args.strategy == 'screening':
        n_snps = variant_tensors.shape[1] if not is_multi_branch else variant_tensors['snp'].shape[1]
        priority, non_priority = compute_snp_priority_order(args.vcf, args.target_vcf, n_snps)
        snp_priority_order = priority + non_priority
        patience_mode = False
        print(f"  [Screening] Priority order: {len(snp_priority_order)} SNPs (target VCF mode)")

    # --- SNP priority order ---
    # Pure screening (--sites-to-evolve, no --target-vcf): one SNP per round
    is_pure_screening = (
        args.strategy == 'screening' and snp_priority_order is None
    )
    if is_pure_screening:
        patience_mode = False
        # One round per evolvable SNP
        n_evolvable = len(evolvable_indices) if evolvable_indices is not None \
            else (variant_tensors.shape[1] if not is_multi_branch else variant_tensors['snp'].shape[1])
        max_rounds = n_evolvable
        print(f"  [Screening] One round per SNP: {n_evolvable} rounds")
    else:
        patience_mode = args.iterations is None
        max_rounds = args.iterations if args.iterations else args.max_iterations
    print(f"\n[Evolution] Starting {'patience' if patience_mode else 'fixed-round'} mode...")
    print(f"  Max rounds: {max_rounds}")

    is_priority_screening = (
        args.strategy == 'screening' and snp_priority_order is not None
        and len(snp_priority_order) == max_rounds
    )

    pbar = tqdm(range(1, max_rounds + 1), desc="Evolution rounds", unit="round")
    round_results: List[RoundResult] = []
    best_tensors = current_tensors
    best_score = current_score
    snp_cursor = 0
    patience_count = 0
    # Track cumulative predictions for cumulative SI computation
    current_preds_norm = init_preds_norm.copy()

    # Shuffle evolvable pool once for reproducibility (seed applies to screening only)
    if args.strategy == 'screening' and evolvable_indices is not None and not is_priority_screening:
        shuffled_pool = evolvable_indices.copy()
        rng_seed = np.random.RandomState(args.seed)
        rng_seed.shuffle(shuffled_pool)
        evolvable_indices = shuffled_pool
    elif args.strategy == 'screening' and evolvable_indices is None and not is_priority_screening:
        # No --sites-to-evolve: all genome-wide SNPs are evolvable, still shuffle for seed reproducibility
        n_snps = variant_tensors.shape[1] if not is_multi_branch else variant_tensors['snp'].shape[1]
        full_pool = list(range(n_snps))
        rng_seed = np.random.RandomState(args.seed)
        rng_seed.shuffle(full_pool)
        evolvable_indices = full_pool

    for round_num in pbar:
        round_start = time.time()

        if args.verbose and not is_priority_screening:
            print(f"\n{'='*60}")
            print(f"Round {round_num}/{max_rounds} {'(patience)' if patience_mode else '(fixed)'} | score={current_score:.4f}")

        # Determine which SNP was evaluated in this round (for screening log)
        evaluated_snp_idx = -1
        evaluated_snp_id = ""
        si_contrib: float = 0.0
        snp_list_for_round = None
        if args.strategy == 'screening':
            if is_pure_screening:
                # Each round: evaluate exactly one SNP from evolvable_indices
                snp_list_for_round = evolvable_indices if evolvable_indices is not None else \
                    list(range(variant_tensors.shape[1] if not is_multi_branch
                               else variant_tensors['snp'].shape[1]))
                evaluated_snp_idx = snp_list_for_round[snp_cursor] if snp_cursor < len(snp_list_for_round) else -1
                snp_to_eval = [snp_list_for_round[snp_cursor]]
                snp_cursor += 1
            else:
                snp_to_eval = [snp_priority_order[snp_cursor]] if snp_priority_order else None
                if snp_priority_order:
                    evaluated_snp_idx = snp_priority_order[snp_cursor]
            if evaluated_snp_idx >= 0:
                all_snp_ids = variant_info.get('snp', {}).get('ids', [])
                evaluated_snp_id = all_snp_ids[evaluated_snp_idx] if evaluated_snp_idx < len(all_snp_ids) else ""
            evolved_tensors, evolved_score, mutations, log_dict = strategy_screening(
                model, current_tensors, variant_info, current_score,
                encoding_dim, device, is_multi_branch, pheno_idx,
                args.mode, args.top_k or 8, args.min_improve, args.verbose,
                regression_tasks, norm_stats,
                current_preds=init_preds, current_preds_norm=init_preds_norm,
                use_selection_index=use_selection_index,
                direction=direction,
                panel_genotypes=panel_genotypes,
                snp_priority_order=snp_to_eval,
                evolvable_indices=evolvable_indices,
                mc_samples=args.mc_samples,
                original_preds_norm=init_preds_norm,
                apply_all_positive_gain=args.top_k is None,
                homozygous_only=args.homozygous,
                seed=args.seed,
            )
            if log_dict:
                eval_results = log_dict.get('eval_results', {})
                if evaluated_snp_idx in eval_results:
                    si_contrib = float(eval_results[evaluated_snp_idx].get('si_original_contribution', 0.0))
                elif eval_results:
                    si_contrib = float(max(r.get('si_original_contribution', 0.0) for r in eval_results.values()))
            if not is_pure_screening:
                snp_cursor += 1
        else:
            evolved_tensors, evolved_score, mutations, log_dict = strategy_combinatorial(
                model, current_tensors, variant_info, current_score,
                encoding_dim, device, is_multi_branch, pheno_idx,
                args.mode, args.top_k or 8, args.min_improve, args.seed + round_num - 1,
                args.verbose, regression_tasks, norm_stats,
                current_preds=init_preds, current_preds_norm=init_preds_norm,
                use_selection_index=use_selection_index,
                direction=direction,
                panel_genotypes=panel_genotypes,
                evolvable_indices=evolvable_indices,
                mc_samples=args.mc_samples,
                homozygous_only=args.homozygous,
            )

        combo_msg = log_dict.get('combo_msg', '') if log_dict else ''
        round_gain = evolved_score - current_score if evolved_score is not None else 0.0

        # For pure screening: accept if strategy_screening applied any mutations
        # (it already applied only SNPs with positive individual gain).
        # For other modes: use cumulative SI gain threshold.
        if is_pure_screening:
            mutations = log_dict.get('mutations', []) if log_dict else []
            accepted = len(mutations) > 0
        else:
            accepted = abs(round_gain) >= args.min_improve

        if args.verbose:
            print(f"  Evolved score: {evolved_score:.4f} (gain: {round_gain:+.4f}) {combo_msg}")

        if args.save_all_rounds and evolved_tensors is not None:
            stem = input_path.stem.replace('.vcf', '')
            suffix = '.vcf.gz' if input_path.suffix == '.gz' else '.vcf'
            round_vcf_dir = out_dir / "round_vcf"
            round_vcf_dir.mkdir(exist_ok=True)
            round_vcf = round_vcf_dir / f"round{round_num:03d}{suffix}"
            evolved_genotypes = {}
            if is_multi_branch:
                for vtype, tensor in evolved_tensors.items():
                    evolved_genotypes[vtype] = tensor[0].cpu().numpy()
            else:
                evolved_genotypes['snp'] = evolved_tensors[0].cpu().numpy()
            write_evolved_vcf(
                original_vcf_path=args.vcf,
                output_vcf_path=str(round_vcf),
                evolved_genotypes=evolved_genotypes,
                evolved_sample_name=f"{sample_ids[0]}_round{round_num}",
                phased=args.phased,
            )

        # Record combo results
        per_snp_gains = log_dict.get('per_snp_gains') if log_dict else None
        # Compute cumulative SI (with maintain penalty) for score_baseline/score_evolved
        if use_selection_index and direction is not None:
            score_baseline = float(
                compute_selection_index_normalized(
                    init_preds_norm.reshape(1, -1), direction, init_preds_norm.reshape(1, -1)
                )[0]
            )
            if evolved_tensors is not None and evolved_score is not None:
                if is_multi_branch:
                    pred_batch = {k: v[0:1].to(device) for k, v in evolved_tensors.items()}
                else:
                    pred_batch = evolved_tensors[0:1].to(device)
                evolved_preds_norm = predict_all_phenos(model, pred_batch, device, is_multi_branch)
                score_evolved = float(
                    compute_selection_index_normalized(
                        evolved_preds_norm, direction, init_preds_norm.reshape(1, -1)
                    )[0]
                )
            else:
                score_evolved = score_baseline
        else:
            score_baseline = float(current_score)
            score_evolved = float(evolved_score) if evolved_score is not None else float(current_score)

        rr = RoundResult(
            round_num=round_num,
            si_baseline=current_score,
            si_evolved=evolved_score if evolved_score is not None else current_score,
            si_gain=round_gain,
            accepted=accepted,
            mutations=[(s, e, g) for s, e, g in (mutations or [])],
            snp_gains=({int(i): float(g) for i, g in enumerate(per_snp_gains) if g != 0}
                       if per_snp_gains is not None else {}),
            evaluated_snp_idx=evaluated_snp_idx,
            evaluated_snp_id=evaluated_snp_id,
            score_baseline=score_baseline,
            score_evolved=score_evolved,
            si_original_contribution=si_contrib,
        )

        if log_dict:
            mc_details = log_dict.get('snp_mc_details')
            if mc_details:
                rr.snp_mc_details = mc_details
            combo_all_preds = log_dict.get('all_preds')
            combo_gains = log_dict.get('gains')
            if combo_all_preds is not None and len(combo_all_preds) > 0:
                for mask_idx in range(len(combo_all_preds)):
                    rr.combo_results[mask_idx] = {
                        t: float(combo_all_preds[mask_idx, i])
                        for i, t in enumerate(regression_tasks)
                    }
                    rr.combo_results[mask_idx]['_gain'] = (
                        float(combo_gains[mask_idx])
                        if combo_gains is not None and mask_idx < len(combo_gains)
                        else 0.0
                    )

        # Per-phenotype predictions for this round
        if evolved_tensors is not None and evolved_score is not None:
            if is_multi_branch:
                pred_batch = {k: v[0:1].to(device) for k, v in evolved_tensors.items()}
            else:
                pred_batch = evolved_tensors[0:1].to(device)
            preds_norm = predict_all_phenos(model, pred_batch, device, is_multi_branch)
            if isinstance(norm_stats, dict) and 'regression_means' in norm_stats:
                preds = denormalize_predictions(preds_norm, norm_stats, regression_tasks, log_tasks)[0]
            else:
                preds = preds_norm[0]
            rr.all_phenotypes_baseline = dict(zip(regression_tasks, init_preds.tolist()))
            rr.all_phenotypes_evolved  = dict(zip(regression_tasks, preds.tolist()))

            # Write round predictions TSV (only when saving all rounds)
            if args.save_all_rounds:
                write_round_predictions(
                    out_dir, round_num, sample_ids[0],
                    regression_tasks, baseline_preds,
                    preds_norm, init_preds_norm,
                    norm_stats, log_tasks,
                )

        round_results.append(rr)

        if accepted and evolved_tensors is not None:
            current_tensors = evolved_tensors
            current_score   = evolved_score if evolved_score is not None else current_score
            best_tensors    = evolved_tensors
            best_score      = current_score
            patience_count  = 0
            # Update current_preds_norm for cumulative SI tracking in next round
            if use_selection_index and direction is not None:
                if is_multi_branch:
                    pred_batch = {k: v[0:1].to(device) for k, v in evolved_tensors.items()}
                else:
                    pred_batch = evolved_tensors[0:1].to(device)
                current_preds_norm = predict_all_phenos(model, pred_batch, device, is_multi_branch)[0]
            if args.strategy == 'combinatorial' and args.save_site_gain:
                write_combo_log(out_dir, round_num, log_dict, regression_tasks, verbose=args.verbose)
                append_cumulative_combo_log(out_dir, round_num, log_dict, regression_tasks)
            round_detail = build_round_detail(
                log_dict, regression_tasks, pheno_idx, current_score, args.mode
            )
            write_round_detail_json(out_dir, round_num, round_detail)
            write_cumulative_round_details(out_dir, round_num, round_detail)
            combo_preds = log_dict.get('all_preds', np.zeros((0, len(regression_tasks))))
            combo_masks = log_dict.get('combo_masks', np.zeros((0, 0)))
            snp_sets    = log_dict.get('snp_sets', [])
            if combo_preds.size > 0:
                write_intermediate_combo_progress(
                    out_dir, round_num, 0, combo_preds, combo_masks, snp_sets,
                    current_score, baseline_preds, regression_tasks, args.mode, pheno_idx,
                )
        else:
            patience_count += 1
            if args.verbose:
                print(f"  Round rejected: gain={round_gain:.6f} < min_improve={args.min_improve}")

        if patience_mode and patience_count >= args.patience:
            print(f"\n[Evolution] Early stopping: no improvement for {args.patience} rounds.")
            break

        pbar.set_postfix({'score': f'{current_score:.4f}', 'gain': f'{round_gain:+.4f}'})

    # --- Print all output file paths ---
    print(f"\n{'='*80}")
    print("OUTPUT FILES")
    print(f"{'='*80}")
    print(f"  Round predictions (each round):   {out_dir}/round_predictions.tsv")
    if is_priority_screening:
        print(f"  Per-round SI log:                {out_dir}/screening_si_per_round.tsv")
        print(f"  Per-SNP max gain log:           {out_dir}/screening_snp_max_gains.tsv")
    if is_pure_screening:
        write_screening_si_log(str(out_dir), round_results, regression_tasks)
        print(f"  Per-round SI log:                {out_dir}/screening_si_per_round.tsv")
        print(f"  Per-SNP max gain log:           {out_dir}/screening_snp_max_gains.tsv")
    if args.strategy == 'combinatorial' and args.save_site_gain:
        print(f"  Per-round SI log:                {out_dir}/combinatorial_si_per_round.tsv")
        print(f"  Per-SNP max gain log:           {out_dir}/combinatorial_site_max_gains.tsv")
        print(f"  Per-SNP MC interaction gain:      {out_dir}/combinatorial_site_mc_details.tsv")
        print(f"  Per-SNP contribution (combo):   {out_dir}/combinatorial_site_contributions.tsv")
        print(f"  Per-SNP MC contribution (combo): {out_dir}/combinatorial_site_contributions_mc.tsv")
    if args.save_all_rounds:
        print(f"  Per-round VCFs:                  {out_dir}/round_vcf/round001.vcf.gz ... round{max_rounds:03d}.vcf.gz")

    # --- Write summary log ---
    summary_path = str(out_dir / summary_filename)
    write_summary_log(summary_path, round_results, regression_tasks,
                       args.pheno if args.pheno else "selection_index")

    # --- Write evolved VCF ---
    output_vcf = args.output_vcf
    if output_vcf is None:
        stem = input_path.stem.replace('.vcf', '')
        suffix = '.vcf.gz' if input_path.suffix == '.gz' else '.vcf'
        output_vcf = str(out_dir / f"{stem}_evolve{suffix}")

    evolved_genotypes = {}
    if is_multi_branch:
        for vtype, tensor in best_tensors.items():
            evolved_genotypes[vtype] = tensor[0].cpu().numpy()
    else:
        evolved_genotypes['snp'] = best_tensors[0].cpu().numpy()

    write_evolved_vcf(
        original_vcf_path=args.vcf,
        output_vcf_path=output_vcf,
        evolved_genotypes=evolved_genotypes,
        evolved_sample_name=f"{sample_ids[0]}_evolve",
        phased=args.phased,
        verbose=True,
    )

    # --- Write prediction TSV ---
    output_pred = args.output_pred
    if output_pred is None:
        output_pred = str(out_dir / f"{Path(output_vcf).stem}_predictions.tsv")

    if is_multi_branch:
        final_batch = {k: v[0:1].to(device) for k, v in best_tensors.items()}
    else:
        final_batch = best_tensors[0:1].to(device)

    with torch.no_grad():
        outputs = model(final_batch)
    final_preds_norm = (
        outputs['regression'].cpu().numpy()[0]
        if 'regression' in outputs
        else torch.sigmoid(outputs['classification']).cpu().numpy()[0]
    )
    if isinstance(norm_stats, dict) and 'regression_means' in norm_stats:
        final_preds = denormalize_predictions(
            final_preds_norm.reshape(1, -1), norm_stats, regression_tasks, log_tasks
        )[0]
    else:
        final_preds = final_preds_norm

    pred_dict = {'Sample_ID': [f"{sample_ids[0]}_evolve"]}
    for t_idx, task in enumerate(regression_tasks):
        pred_dict[f'{task}_Pred'] = [final_preds[t_idx]]
    pd.DataFrame(pred_dict).to_csv(output_pred, sep='\t', index=False)
    print(f"\n  All phenotype predictions saved to: {output_pred}")

    print(f"\n  Final predictions:")
    for t_idx, task in enumerate(regression_tasks):
        m = direction.modes[t_idx] if direction else 'maximize'
        gain = final_preds[t_idx] - init_preds[t_idx]
        print(f"    {task} ({m}): {final_preds[t_idx]:.4f}  "
              f"(baseline: {init_preds[t_idx]:.4f}, gain: {gain:+.4f})")

    # --- Combinatorial extra logs ---
    if args.strategy == 'combinatorial':
        if args.save_site_gain:
            write_combinatorial_snp_gains(str(out_dir), round_results, regression_tasks, variant_info)
            write_combinatorial_site_contributions(str(out_dir), round_results, variant_info)
        else:
            print(f"\n  (Use --save-site-gain to write per-SNP contribution logs)")


def main():
    run_evolution(parse_args())


if __name__ == '__main__':
    main()
