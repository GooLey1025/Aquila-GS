"""
Core reusable components for Aquila-GS directed evolution.

This module is imported by:
  - src/aquila/scripts/aquila_evolve.py      (single-seed CLI)
  - src/aquila/scripts/aquila_evolve_multi.py  (multi-seed parallel runner)

Module organization:
  Dataclasses       DirectionModes, RoundResult
  GPU utils         get_available_gpus
  Model I/O         load_model_and_config, create_model, load_checkpoint
  VCF I/O           load_vcf_data, write_final_evolved_vcf
  Prediction        predict_all_phenos, denormalize_predictions,
                    write_final_predictions
  VCF analysis     parse_vcf_gt_column, compute_qtn_allele_freq_changes,
                    merge_evolved_vcfs
  Direction utils   load_direction_file, load_panel_genotypes,
                    compute_snp_priority_order
  SI computation    compute_selection_index_normalized,
                    best_combo_by_selection_index, _si_from_preds
  Genotype ops      get_encoding_dim, clone_genotype_tensors,
                    apply_single_snp_mutation, apply_multi_snp_mutations,
                    build_mutation_candidates_for_snp
  MC gains          compute_combinatorial_gains
  Strategies        _screening_evaluate_one_snp,
                    strategy_screening, strategy_combinatorial
"""

import gzip
import json
import pickle
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from aquila.encoding import parse_genotype_file, write_evolved_vcf
from aquila.utils import load_config
from aquila.varnn import create_model_from_config


###############################################################################
# Dataclasses
###############################################################################

@dataclass
class DirectionModes:
    """
    Holds direction-mode metadata for each task alongside the SI weight array.

    mode:  'maximize' | 'minimize' | 'neutral' | 'maintain' | 'numeric_target'
    weight: +1.0 (maximize), -1.0 (minimize), 0.0 (neutral), 0.0 (maintain)
            For numeric_target, weight is always +1.0 (positive contribution
            means moving toward the target; negative means moving away).
            Maintains have weight=0 in the weighted sum so they don't pull
            the SI toward/away from baseline; their penalty is applied
            separately in compute_selection_index_normalized.

    baseline_preds_norm: (n_tasks,) — Z-score baseline for maintain traits.
                         None when no maintain traits are defined.

    numeric_targets_norm: (n_tasks,) — Z-score target for each numeric_target trait.
                         Computed as:
                           target_denorm = baseline_denorm * (1 + direction%/100)
                           target_z      = (target_denorm - mean) / std
                         Compatible with preds_norm and baseline_preds_norm (both Z-score).
                         None when no numeric_target traits are defined.

    _is_numeric: bool — True when at least one trait uses numeric_target mode.
                 Convenience flag for downstream code.
    """
    modes: np.ndarray            # (n_tasks,) str
    weights: np.ndarray          # (n_tasks,) float — same sign convention as before
    baseline_preds_norm: Optional[np.ndarray] = None   # (n_tasks,)
    numeric_targets_norm: Optional[np.ndarray] = None  # (n_tasks,)
    _is_numeric: bool = False

    def has_maintain(self) -> bool:
        return bool(np.any(self.modes == 'maintain'))

    def get_maintain_indices(self) -> np.ndarray:
        return np.where(self.modes == 'maintain')[0]

    def has_numeric_target(self) -> bool:
        return self._is_numeric


@dataclass
class RoundResult:
    """Record of a single evolution round."""
    round_num: int
    si_baseline: float
    si_evolved: float
    si_gain: float
    accepted: bool
    # SI gain for this round (alias of si_gain, used by log writers)
    gain: float = 0.0
    # Per-phenotype values for this round
    all_phenotypes_baseline: Dict[str, float] = None
    all_phenotypes_evolved:  Dict[str, float] = None
    # Strategy-specific data
    combo_results:   Dict[int, Dict[str, float]] = None
    snp_gains:      Dict[int, float] = None
    snp_mc_details:  Dict[int, dict] = None
    mutations: List[Tuple[int, object, float]] = field(default_factory=list)
    # Screening-specific: evaluated SNP for this round
    evaluated_snp_idx: int = -1
    evaluated_snp_id: str = ""
    # Cumulative SI (with maintain penalty, reflects all accumulated mutations)
    score_baseline: float = 0.0
    score_evolved: float = 0.0
    # SI contribution of this SNP relative to the ORIGINAL (VCF) genotype.
    # = SI(with only this SNP) - SI(original genotype).
    # Distinct from si_gain (which is relative to the cumulative genotype at start of this round).
    si_original_contribution: float = 0.0

    def __post_init__(self):
        if self.all_phenotypes_baseline is None:
            self.all_phenotypes_baseline = {}
        if self.all_phenotypes_evolved is None:
            self.all_phenotypes_evolved = {}
        if self.combo_results is None:
            self.combo_results = {}
        if self.snp_gains is None:
            self.snp_gains = {}
        if self.snp_mc_details is None:
            self.snp_mc_details = {}
        if self.gain == 0.0 and self.si_gain != 0.0:
            self.gain = self.si_gain


###############################################################################
# GPU utilities
###############################################################################

def get_available_gpus() -> List[int]:
    """Return list of available GPU device IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


###############################################################################
# Model loading utilities
###############################################################################

def find_file_in_dir(directory: Path, filename_pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    exact = directory / filename_pattern
    if exact.exists() and exact.is_file():
        return exact
    matches = list(directory.glob(filename_pattern))
    return matches[0] if matches else None


def load_model_and_config(model_dir: str):
    """
    Load model checkpoint, config, and normalization stats from a model directory.

    Searches in model_dir/checkpoints/, model_dir/, and model_dir.parent/
    for maximum robustness.
    """
    model_dir = Path(model_dir)

    for subdir in [model_dir / 'checkpoints', model_dir.parent / 'checkpoints', model_dir]:
        checkpoint_path = find_file_in_dir(subdir, 'best_checkpoint.pt')
        if not checkpoint_path:
            pt_matches = list(subdir.glob('*.pt'))
            checkpoint_path = pt_matches[0] if pt_matches else None
        if checkpoint_path:
            break

    config_path = find_file_in_dir(model_dir, 'params.yaml')
    if not config_path:
        config_path = find_file_in_dir(model_dir.parent, 'params.yaml')

    norm_stats_path = find_file_in_dir(model_dir, 'normalization_stats.pkl')
    if not norm_stats_path:
        norm_stats_path = find_file_in_dir(model_dir.parent, 'normalization_stats.pkl')

    if not checkpoint_path:
        raise FileNotFoundError(f"Checkpoint not found in {model_dir}")
    if not config_path:
        raise FileNotFoundError(f"params.yaml not found in {model_dir}")

    config = load_config(config_path)
    norm_stats = None
    if norm_stats_path:
        with open(norm_stats_path, 'rb') as f:
            norm_stats = pickle.load(f)

    return config, checkpoint_path, norm_stats


def create_model(config: dict, seq_length, device: str,
                 norm_stats: Optional[dict] = None):
    """Instantiate a varnn model from config, move to device, set to eval."""
    model_config = config.get('model', {})
    train_config = config.get('train', {})
    data_config  = config.get('data', {})

    regression_tasks = (
        model_config.get('regression_tasks') or
        train_config.get('regression_tasks') or
        data_config.get('regression_tasks') or
        (isinstance(norm_stats, dict) and norm_stats.get('regression_tasks')) or None
    )
    classification_tasks = (
        model_config.get('classification_tasks') or
        train_config.get('classification_tasks') or
        data_config.get('classification_tasks')
    )

    model = create_model_from_config(
        config=config,
        seq_length=seq_length,
        regression_tasks=regression_tasks,
        classification_tasks=classification_tasks,
    )
    model = model.to(device)
    model.eval()
    return model


def load_checkpoint(model, checkpoint_path: Path, device: str):
    """Load model weights from a checkpoint file. Returns (model, epoch_or_None)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    epoch = checkpoint.get('epoch')
    return model, epoch


###############################################################################
# VCF data loading
###############################################################################

def load_vcf_data(vcf_path: str, encoding_type: str, config: dict):
    """
    Load and encode a VCF file into genotype tensors and metadata.

    Returns:
        variant_tensors: torch.Tensor or dict of such (multi-branch)
        variant_info:    dict of {vtype: {refs, alts, chroms, positions, ids}}
        sample_ids:      list of sample names
        seq_length:      int or dict (per-branch)
        is_multi_branch: bool
    """
    variant_type = config.get('data', {}).get('variant_type')
    is_multi_branch = (
        encoding_type in ['snp_indel_vcf', 'snp_indel_sv_vcf'] or
        variant_type in ['snp_indel', 'snp_indel_sv']
    )

    if is_multi_branch:
        variant_data = parse_genotype_file(vcf_path, encoding_type, variant_type)
        first_vtype = list(variant_data.keys())[0]
        sample_ids = variant_data[first_vtype]['sample_ids']
        seq_length = {vtype: data['matrix'].shape[1] for vtype, data in variant_data.items()}

        variant_tensors = {}
        variant_info = {}
        for vtype, data in variant_data.items():
            key = vtype.lower()
            arr = data['matrix']
            variant_tensors[key] = torch.from_numpy(arr).float()
            variant_info[key] = {
                'refs':     data.get('refs', []),
                'alts':     data.get('alts', []),
                'chroms':   data.get('chroms', []),
                'positions': data.get('positions', []),
                'ids':      data.get('variant_ids', []),
            }
    else:
        result = parse_genotype_file(vcf_path, encoding_type, variant_type)
        if isinstance(result, dict):
            snp_matrix = result['matrix']
            sample_ids = result['sample_ids']
            variant_info = {
                'snp': {
                    'refs':     result.get('refs', []),
                    'alts':     result.get('alts', []),
                    'chroms':   result.get('chroms', []),
                    'positions': result.get('positions', []),
                    'ids':      result.get('variant_ids', []),
                }
            }
        else:
            snp_matrix, sample_ids, _ = result
            variant_info = {'snp': {'refs': [], 'alts': [], 'chroms': [],
                                    'positions': [], 'ids': []}}

        variant_tensors = torch.from_numpy(snp_matrix).float()
        seq_length = snp_matrix.shape[1]

    return variant_tensors, variant_info, sample_ids, seq_length, is_multi_branch


###############################################################################
# Prediction utilities
###############################################################################

def predict_all_phenos(
    model, genotype_batch, device: str, is_multi_branch: bool,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Run prediction on all phenotypes in small batches.

    Args:
        model:            The neural network model.
        genotype_batch:    Tensor (N, seq, dim) or dict of such tensors.
        device:           Compute device.
        is_multi_branch:  Whether model uses multi-branch input.
        batch_size:       Number of samples per sub-batch.

    Returns:
        Array of predicted values for all phenotypes, shape (N, n_tasks).
    """
    if is_multi_branch:
        n = next(iter(genotype_batch.values())).shape[0]
    else:
        n = genotype_batch.shape[0]

    all_preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            if is_multi_branch:
                sub_batch = {k: v[start:end].to(device) for k, v in genotype_batch.items()}
            else:
                sub_batch = genotype_batch[start:end].to(device)
            outputs = model(sub_batch)
            if 'regression' in outputs:
                preds = outputs['regression'].cpu().numpy()
            else:
                preds = torch.sigmoid(outputs['classification']).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)


def denormalize_predictions(
    preds_norm: np.ndarray, norm_stats,
    tasks: List[str], log_tasks: List[str],
) -> np.ndarray:
    """
    Denormalize Z-score predictions back to original scale.

    Args:
        preds_norm:  (N, n_tasks) predictions in Z-score space.
        norm_stats:  dict with regression_means/stds and log_transformed_tasks.
                     May arrive as a numpy object (e.g. from pickle corruption);
                     in that case this function returns preds_norm unchanged.
        tasks:       List of task names.
        log_tasks:   List of tasks that were log-transformed during training.

    Returns:
        (N, n_tasks) denormalized predictions.
    """
    preds = preds_norm.copy()
    # Guard: norm_stats must be a proper dict
    if not isinstance(norm_stats, dict) or 'regression_means' not in norm_stats:
        return preds
    reg_means = norm_stats.get('regression_means', {})
    reg_stds  = norm_stats.get('regression_stds',  {})
    for i, task in enumerate(tasks):
        # Handle both dict and array forms: dict by task name, array by position
        if isinstance(reg_means, dict):
            mean = reg_means.get(task, 0.0)
        else:
            mean = float(reg_means[i]) if hasattr(reg_means, '__len__') and i < len(reg_means) else 0.0
        if isinstance(reg_stds, dict):
            std = reg_stds.get(task, 1.0)
        else:
            std = float(reg_stds[i]) if hasattr(reg_stds, '__len__') and i < len(reg_stds) else 1.0
        if std == 0:
            std = 1.0
        if task in log_tasks:
            preds[:, i] = np.exp(preds[:, i] * std + mean)
        else:
            preds[:, i] = preds[:, i] * std + mean
    return preds


###############################################################################
# Evolved VCF / prediction writing
###############################################################################

def write_final_evolved_vcf(
    original_vcf_path: str,
    output_vcf_path: str,
    evolved_tensors,
    variant_info: dict,
    sample_name: str,
    phased: bool = True,
) -> None:
    """
    Write the final evolved VCF from evolved tensors.

    Args:
        original_vcf_path: Path to the original input VCF file.
        output_vcf_path:   Path to write the evolved VCF.
        evolved_tensors:   Evolved genotype tensor(s).
        variant_info:      Variant metadata dict.
        sample_name:       New sample name.
        phased:            Use '|' (phased) or '/' (unphased) separator.
    """
    evolved_genotypes = {}
    if isinstance(evolved_tensors, dict):
        for vtype, tensor in evolved_tensors.items():
            arr = tensor[0].cpu().numpy()
            evolved_genotypes[vtype] = arr
    else:
        arr = evolved_tensors[0].cpu().numpy()
        evolved_genotypes['snp'] = arr

    write_evolved_vcf(
        original_vcf_path=original_vcf_path,
        output_vcf_path=output_vcf_path,
        evolved_genotypes=evolved_genotypes,
        evolved_sample_name=sample_name,
        phased=phased,
        verbose=False,
    )


def write_final_predictions(
    output_pred_path: str,
    evolved_tensors,
    model,
    device: str,
    is_multi_branch: bool,
    regression_tasks: List[str],
    norm_stats: Optional[dict],
    log_tasks: List[str],
    sample_name: str,
) -> np.ndarray:
    """
    Predict all phenotypes for the evolved genotype and save as TSV.

    Returns the denormalized prediction array (n_tasks,).
    """
    if is_multi_branch:
        final_batch = {k: v[0:1].to(device) for k, v in evolved_tensors.items()}
    else:
        final_batch = evolved_tensors[0:1].to(device)

    with torch.no_grad():
        outputs = model(final_batch)

    if 'regression' in outputs:
        final_preds_norm = outputs['regression'].cpu().numpy()[0]
    else:
        final_preds_norm = torch.sigmoid(outputs['classification']).cpu().numpy()[0]

    if isinstance(norm_stats, dict) and 'regression_means' in norm_stats:
        final_preds = denormalize_predictions(
            final_preds_norm.reshape(1, -1), norm_stats, regression_tasks, log_tasks
        )[0]
    else:
        final_preds = final_preds_norm

    pred_dict = {'Sample_ID': [sample_name]}
    for t_idx, task in enumerate(regression_tasks):
        pred_dict[f'{task}_Pred'] = [final_preds[t_idx]]
    pd.DataFrame(pred_dict).to_csv(output_pred_path, sep='\t', index=False)
    return final_preds


###############################################################################
# VCF analysis utilities (multi-run)
###############################################################################

def parse_vcf_gt_column(vcf_path: str, sample_name: str = '') -> Dict[str, float]:
    """
    Parse genotype dosages from a VCF for a specific sample.

    Returns dict: vid -> allele frequency (AF = count_alt / (2 * N_non_missing)).
    For single-sample VCFs, AF = dosage / 2.
    """
    opener = gzip.open if vcf_path.endswith('.gz') else open
    mode   = 'rt' if vcf_path.endswith('.gz') else 'r'

    af_dict = {}
    with opener(vcf_path, mode) as f:
        for line in f:
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            line = line.rstrip()
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) < 10:
                continue
            vid = fields[2] if fields[2] else f"{fields[0]}:{fields[1]}"
            geno_field = fields[9].split(':')[0]
            alleles = geno_field.replace('|', '/').split('/')
            try:
                dosage   = sum(int(a) for a in alleles if a != '.')
                n_alleles = sum(1 for a in alleles if a != '.')
                af = dosage / (2 * n_alleles) if n_alleles > 0 else 0.0
            except (ValueError, ZeroDivisionError):
                af = 0.0
            af_dict[vid] = af
    return af_dict


def compute_qtn_allele_freq_changes(
    baseline_vcf_path: str,
    evolved_vcf_paths: List[str],
    qtn_id_list: List[str],
) -> pd.DataFrame:
    """
    Compute allele frequency changes for QTN sites across multiple evolved VCFs.

    Args:
        baseline_vcf_path:  Original input VCF (baseline allele frequencies).
        evolved_vcf_paths:  List of evolved VCF paths (one per seed).
        qtn_id_list:        List of QTN variant IDs to analyze.

    Returns:
        DataFrame with columns: qtn_id, baseline_af, mean_af, max_af, min_af,
        std_af, mean_af_change, max_abs_af_change, n_evolved.
    """
    baseline_af = parse_vcf_gt_column(baseline_vcf_path)
    evolved_af_lists: Dict[str, List[float]] = {q: [] for q in qtn_id_list}

    for evcf_path in evolved_vcf_paths:
        if not Path(evcf_path).exists():
            continue
        ev_af = parse_vcf_gt_column(evcf_path)
        for q in qtn_id_list:
            evolved_af_lists[q].append(ev_af.get(q, baseline_af.get(q, 0.0)))

    rows = []
    for q in qtn_id_list:
        vals = evolved_af_lists[q]
        bl   = baseline_af.get(q, 0.0)
        if vals:
            changes = [v - bl for v in vals]
            rows.append({
                'qtn_id':            q,
                'baseline_af':       bl,
                'mean_af':          np.mean(vals),
                'max_af':           np.max(vals),
                'min_af':           np.min(vals),
                'std_af':           np.std(vals),
                'mean_af_change':   np.mean(changes),
                'max_abs_af_change': np.max(np.abs(changes)),
                'n_evolved':        len(vals),
            })
        else:
            rows.append({
                'qtn_id': q, 'baseline_af': bl,
                'mean_af': 0.0, 'max_af': 0.0, 'min_af': 0.0, 'std_af': 0.0,
                'mean_af_change': 0.0, 'max_abs_af_change': 0.0, 'n_evolved': 0,
            })

    return pd.DataFrame(rows)


def merge_evolved_vcfs(
    baseline_vcf_path: str,
    evolved_vcf_paths: List[str],
    output_vcf_path: str,
    sample_prefix: str = "seed",
) -> None:
    """
    Merge multiple evolved VCFs into a single VCF with all samples.

    Adds three INFO fields computed across evolved samples:
      AF_CHANGE = mean(evolved_AF) - baseline_AF
      AF_STD    = std(evolved_AF)
      N_EVOLVED = number of evolved samples with data for this variant

    Args:
        baseline_vcf_path:  Original input VCF.
        evolved_vcf_paths:  List of evolved VCFs to merge.
        output_vcf_path:    Output merged VCF path.
        sample_prefix:      Prefix for sample names (e.g. "seed" -> "seed_0001").
    """
    opener = gzip.open if baseline_vcf_path.endswith('.gz') else open
    mode   = 'rt' if baseline_vcf_path.endswith('.gz') else 'r'

    # Parse baseline AF
    vid_to_baseline = parse_vcf_gt_column(baseline_vcf_path)

    # Collect evolved AF dicts
    all_evolved_af: List[Dict[str, float]] = []
    for evcf_path in evolved_vcf_paths:
        if Path(evcf_path).exists():
            all_evolved_af.append(parse_vcf_gt_column(evcf_path))

    if not all_evolved_af:
        raise ValueError("No valid evolved VCFs found")

    # Collect sample names
    sample_names = [f"{sample_prefix}_{i+1:04d}" for i in range(len(evolved_vcf_paths))]

    # Read all evolved VCF data into memory (variant -> {sample_idx -> gt})
    # For speed, build a lookup: vid -> list of GT strings
    ev_gt_lookup: Dict[str, List[str]] = {}
    ev_vcf_list = [Path(p) for p in evolved_vcf_paths]

    for evcf_path in evolved_vcf_paths:
        if not Path(evcf_path).exists():
            continue
        ev_opener = gzip.open if evcf_path.endswith('.gz') else open
        ev_mode   = 'rt' if evcf_path.endswith('.gz') else 'r'
        with ev_opener(evcf_path, ev_mode) as ef:
            for eline in ef:
                if isinstance(eline, bytes):
                    eline = eline.decode('utf-8')
                if eline.startswith('#'):
                    continue
                efields = eline.rstrip().split('\t')
                if len(efields) < 10:
                    continue
                vid = efields[2] if efields[2] else f"{efields[0]}:{efields[1]}"
                if vid not in ev_gt_lookup:
                    ev_gt_lookup[vid] = [''] * len(evolved_vcf_paths)
                idx = evolved_vcf_paths.index(evcf_path)
                ev_gt_lookup[vid][idx] = efields[9]

    # Build GT lookup from baseline for missing entries
    bl_opener = gzip.open if baseline_vcf_path.endswith('.gz') else open
    bl_mode   = 'rt' if baseline_vcf_path.endswith('.gz') else 'r'
    bl_gt_lookup: Dict[str, str] = {}
    with bl_opener(baseline_vcf_path, bl_mode) as bf:
        for bline in bf:
            if isinstance(bline, bytes):
                bline = bline.decode('utf-8')
            if bline.startswith('#'):
                continue
            bfields = bline.rstrip().split('\t')
            if len(bfields) < 10:
                continue
            vid = bfields[2] if bfields[2] else f"{bfields[0]}:{bfields[1]}"
            bl_gt_lookup[vid] = bfields[9]

    output_tmp = output_vcf_path + '.tmp'
    out_opener = gzip.open if output_vcf_path.endswith('.gz') else open
    out_mode   = 'wt' if output_vcf_path.endswith('.gz') else 'w'

    with opener(baseline_vcf_path, mode) as fin, out_opener(output_tmp, out_mode) as fout:
        for line in fin:
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            line = line.rstrip()
            if line.startswith('##'):
                if 'AF_CHANGE' not in line:
                    fout.write(line + '\n')
            elif line.startswith('#CHROM'):
                # Add new INFO headers
                for info_tag, info_desc in [
                    ('AF_CHANGE', 'Mean allele frequency change across evolved samples'),
                    ('AF_STD',    'Std of allele frequency across evolved samples'),
                    ('N_EVOLVED', 'Number of evolved samples'),
                ]:
                    fout.write(f'##INFO=<ID={info_tag},Number=1,Type=Float,Description="{info_desc}">\n')
                # Write new CHROM line
                fields = line.split('\t')
                chrom_part = fields[:9]
                fout.write('\t'.join(chrom_part + sample_names) + '\n')
            else:
                fields = line.split('\t')
                if len(fields) < 10:
                    continue
                vid       = fields[2] if fields[2] else f"{fields[0]}:{fields[1]}"
                baseline_af = vid_to_baseline.get(vid, 0.0)

                # AF stats across evolved samples
                ev_af_vals = [ev_af.get(vid, baseline_af) for ev_af in all_evolved_af]
                mean_af  = np.mean(ev_af_vals)
                std_af   = np.std(ev_af_vals)  if len(ev_af_vals) > 1 else 0.0
                af_change = mean_af - baseline_af

                # Update INFO
                info_parts = fields[7].split(';') if fields[7] != '.' else []
                info_dict  = {}
                for part in info_parts:
                    if '=' in part:
                        k, v = part.split('=', 1)
                        info_dict[k] = v
                info_dict['AF_CHANGE']  = f'{af_change:+.6f}'
                info_dict['AF_STD']      = f'{std_af:.6f}'
                info_dict['N_EVOLVED']   = str(sum(1 for v in ev_af_vals
                                                   if not np.isclose(v, baseline_af)))
                new_info = ';'.join(f'{k}={v}' for k, v in info_dict.items())

                # Sample GTs
                gt_list = ev_gt_lookup.get(vid, [''] * len(evolved_vcf_paths))
                for i, gt in enumerate(gt_list):
                    if not gt:
                        gt_list[i] = bl_gt_lookup.get(vid, './.:.')

                fout.write('\t'.join(fields[:7]) + '\t' +
                           new_info + '\t' + fields[8] + '\t' +
                           '\t'.join(gt_list) + '\n')

    Path(output_tmp).replace(output_vcf_path)


###############################################################################
# Direction / panel utilities
###############################################################################

def load_direction_file(path: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load direction file and return (trait_names, weights, modes, numeric_values).

    direction: maximize=+1.0, minimize=-1.0, neutral=0.0, maintain=0.0
    maintain traits have weight=0 in the weighted sum; their penalty is applied
    separately in the SI computation (SI -= |delta| in Z-score space).

    Auto-detects numeric direction values (floats like -2.954240) vs string
    keywords (maximize/minimize/neutral/maintain).  Numeric values represent
    the target percentage change relative to baseline:
        target = baseline * (1 + direction/100)
    The numeric_values array stores the raw float for each trait (0.0 for
    keyword-based directions).

    Returns (trait_names, weights, modes, numeric_values).
    """
    KEYWORD_MODES = {'maximize', 'minimize', 'neutral', 'maintain'}
    trait_names, weights, modes, numeric_values = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            trait     = parts[0].strip()
            direction = parts[2].strip()

            # Auto-detect: try to parse as float for numeric target mode
            try:
                num_val = float(direction)
                # 0.0% change effectively means "maintain current value"
                if num_val == 0.0:
                    trait_names.append(trait)
                    weights.append(0.0)
                    modes.append('maintain')
                    numeric_values.append(0.0)
                else:
                    trait_names.append(trait)
                    weights.append(1.0)       # positive weight = moving toward target
                    modes.append('numeric_target')
                    numeric_values.append(num_val)  # raw percentage change
            except ValueError:
                # String keyword
                trait_names.append(trait)
                numeric_values.append(0.0)
                direction_lower = direction.lower()
                if   direction_lower == 'maximize':  weights.append(1.0);  modes.append('maximize')
                elif direction_lower == 'minimize':  weights.append(-1.0); modes.append('minimize')
                elif direction_lower == 'neutral':  weights.append(0.0);  modes.append('neutral')
                elif direction_lower == 'maintain': weights.append(0.0);  modes.append('maintain')
                else:                             weights.append(1.0);  modes.append('maximize')

    return trait_names, np.array(weights), np.array(modes), np.array(numeric_values)


def load_panel_genotypes(ref_vcf_path: str, n_snps: int) -> List[set]:
    """
    Load population panel VCF and extract available phased GT classes per SNP.

    Uses pysam to read the panel VCF and records all unique phased genotype
    classes observed at each SNP site across all samples.

    Args:
        ref_vcf_path: Path to population panel VCF.
        n_snps:       Total number of SNP sites (must match input VCF ordering).

    Returns:
        List of length n_snps, each entry is a set of observed phased genotype
        classes: 0=0|0, 1=0|1, 2=1|0, 3=1|1. Empty set means no data.
    """
    import pysam

    panel = pysam.VariantFile(ref_vcf_path)
    panel_gts: List[set] = [set() for _ in range(n_snps)]
    snp_idx = 0
    for rec in panel:
        if len(rec.alleles) != 2 or rec.id is None:
            continue
        if snp_idx >= n_snps:
            break
        for sample_name in rec.samples:
            gt = rec.samples[sample_name].get('GT')
            if gt is None:
                continue
            allele1, allele2 = gt[0], gt[1]
            gt_class = allele1 * 2 + allele2
            panel_gts[snp_idx].add(gt_class)
        snp_idx += 1
    panel.close()
    return panel_gts


def compute_snp_priority_order(
    input_vcf_path: str,
    target_vcf_path: str,
    n_snps: int,
) -> Tuple[List[int], List[int]]:
    """
    Compare input and target VCFs to determine SNP priority for evolution.

    Identifies SNPs where the input sample's GT differs from the target genome's
    GT, and returns them ordered by priority for directed evolution.

    Returns:
        (priority_snps, non_priority_snps):
            - priority_snps:     SNP indices where input GT differs from target GT
            - non_priority_snps: SNP indices where input GT matches target GT
    """
    import pysam

    input_vcf = pysam.VariantFile(input_vcf_path)
    input_gt_by_id: dict = {}
    for rec in input_vcf:
        if len(rec.alleles) != 2 or rec.id is None:
            continue
        for sample_name in rec.samples:
            gt = rec.samples[sample_name].get('GT')
            if gt is not None:
                input_gt_by_id[rec.id] = gt
                break
    input_vcf.close()

    if not input_gt_by_id:
        raise ValueError(f"No SNPs found in input VCF: {input_vcf_path}")

    target_vcf   = pysam.VariantFile(target_vcf_path)
    priority_snps, non_priority_snps = [], []

    snp_idx = 0
    for rec in target_vcf:
        if len(rec.alleles) != 2 or rec.id is None:
            continue
        if snp_idx >= n_snps:
            break
        target_gt = None
        for sample_name in rec.samples:
            gt = rec.samples[sample_name].get('GT')
            if gt is not None:
                target_gt = gt
                break
        input_gt = input_gt_by_id.get(rec.id)
        if target_gt is not None and input_gt is not None:
            if target_gt != input_gt:
                priority_snps.append(snp_idx)
            else:
                non_priority_snps.append(snp_idx)
        snp_idx += 1

    target_vcf.close()
    return priority_snps, non_priority_snps


###############################################################################
# Selection index computation
###############################################################################

def compute_selection_index_normalized(
    preds_norm: np.ndarray,
    direction: DirectionModes,
    baseline_preds_norm: np.ndarray,
    trait_stds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute SI in Z-score space with optional maintain-trait penalty.

    SI (keyword modes) = Σ(weight_i * (pred_i - baseline_i) / std_i)
                       - Σ_maintain |pred_i - baseline_i| / std_i

    SI (numeric_target mode):
        For each numeric_target trait with direction pct (e.g. -2.5):
            target_actual = baseline_actual * (1 + pct/100)
            target_zscore = (target_actual - mean) / std
            progress = (pred_z - baseline_z) / (target_z - baseline_z)
        progress = 1.0 means evolved reached the target exactly;
        >1.0 means overshooting, <1.0 means falling short.
        A negative progress means moving away from the target.

    Args:
        preds_norm:            (N, n_tasks) predictions in Z-score space.
        direction:            DirectionModes with weights and baseline.
        baseline_preds_norm:  (n_tasks,) Z-score baseline for comparison.
        trait_stds:           (n_tasks,) standard deviations for each trait.
                              If None, assumed to be 1.0 (already in Z-score space).

    Returns:
        SI values of shape (N,).
    """
    n_tasks = preds_norm.shape[1]
    si = np.zeros(preds_norm.shape[0])

    # Ensure 1D shape for consistent indexing (use flatten to avoid mutating caller's array)
    if baseline_preds_norm.ndim == 2:
        baseline_preds_norm = baseline_preds_norm.flatten()

    for t_idx in range(n_tasks):
        mode = direction.modes[t_idx]

        if mode == 'numeric_target':
            # progress = (pred_z - baseline_z) / (target_z - baseline_z)
            # = 1.0 when evolved reaches the target, >1.0 when overshooting, <1.0 when short
            delta_norm = preds_norm[:, t_idx] - baseline_preds_norm[t_idx]
            target_z = direction.numeric_targets_norm[t_idx]
            denom = target_z - baseline_preds_norm[t_idx]
            if abs(denom) < 1e-12:
                pass  # target == baseline: no contribution
            else:
                si = si + delta_norm / denom
        elif mode == 'maximize':
            delta = preds_norm[:, t_idx] - baseline_preds_norm[t_idx]
            if trait_stds is not None:
                delta = delta / trait_stds[t_idx]
            si = si + delta
        elif mode == 'minimize':
            delta = preds_norm[:, t_idx] - baseline_preds_norm[t_idx]
            if trait_stds is not None:
                delta = delta / trait_stds[t_idx]
            si = si - delta
        elif mode == 'neutral':
            pass
        elif mode == 'maintain':
            pass  # penalty handled below

    if direction.has_maintain():
        maint_idx = direction.get_maintain_indices()
        if maint_idx.size > 0:
            for t_idx in maint_idx:
                delta = preds_norm[:, t_idx] - baseline_preds_norm[t_idx]
                if trait_stds is not None:
                    delta = delta / trait_stds[t_idx]
                si = si - np.abs(delta)
    return si


def best_combo_by_selection_index(
    preds_norm: np.ndarray,
    direction: DirectionModes,
    baseline_preds_norm: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """
    Find the best combination by selection index.

    Returns:
        (best_combo_idx, si_values for all combos)
    """
    si = compute_selection_index_normalized(preds_norm, direction, baseline_preds_norm)
    best_idx = int(np.argmax(si))
    return best_idx, si


def _si_from_preds(
    preds_norm: np.ndarray,
    direction: Optional[DirectionModes],
    baseline_norm: np.ndarray,
) -> np.ndarray:
    """
    Compute SI from predictions given a baseline.
    Used internally by compute_combinatorial_gains.

    Supports numeric_target mode where contribution is based on
    the proportion of progress toward the numeric target.
    """
    if direction is not None and (direction.has_maintain() or direction.has_numeric_target()):
        return compute_selection_index_normalized(preds_norm, direction, baseline_norm)
    w = direction.weights if direction is not None else np.ones(preds_norm.shape[1])
    return ((preds_norm - baseline_norm) * w).sum(axis=1)


###############################################################################
# Genotype manipulation utilities
###############################################################################

def get_encoding_dim(variant_tensors) -> int:
    """Get the encoding dimension (last axis size) for SNP tensors."""
    if isinstance(variant_tensors, dict):
        t = variant_tensors.get('snp') or list(variant_tensors.values())[0]
    else:
        t = variant_tensors
    return t.shape[-1] if t.ndim == 3 else 1


def clone_genotype_tensors(variant_tensors, is_multi_branch: bool):
    """Create a deep copy of variant tensors."""
    if is_multi_branch:
        return {k: v.clone() for k, v in variant_tensors.items()}
    return variant_tensors.clone()


def apply_single_snp_mutation(
    variant_tensors, snp_idx: int, new_encoding: torch.Tensor, is_multi_branch: bool,
):
    """Apply a single-SNP mutation by replacing the encoding at snp_idx."""
    if is_multi_branch:
        result = {k: v.clone() for k, v in variant_tensors.items()}
        result['snp'][0, snp_idx] = new_encoding
    else:
        result = variant_tensors.clone()
        result[0, snp_idx] = new_encoding
    return result


def apply_multi_snp_mutations(
    variant_tensors, mutations: List[Tuple[int, torch.Tensor]], is_multi_branch: bool,
):
    """Apply multiple SNP mutations at once."""
    if is_multi_branch:
        result = {k: v.clone() for k, v in variant_tensors.items()}
        for snp_idx, new_encoding in mutations:
            result['snp'][0, snp_idx] = new_encoding
    else:
        result = variant_tensors.clone()
        for snp_idx, new_encoding in mutations:
            result[0, snp_idx] = new_encoding
    return result


def build_mutation_candidates_for_snp(
    original_encoding, encoding_dim: int,
    ref: str, alt: str, available_gts=None,
    homozygous_only: bool = False,
) -> List[torch.Tensor]:
    """
    Build alternative genotype encoding vectors for a single SNP.

    Args:
        original_encoding: Current encoding tensor for this SNP.
        encoding_dim:     Encoding dimension (8 for diploid_onehot, 4 for class, etc.).
        ref:              Reference allele character.
        alt:              Alternate allele character.
        available_gts:     Optional set of GT classes observed in population panel.

    Returns:
        List of alternative encoding tensors (torch.Tensor), one per candidate.
    """
    if encoding_dim == 8:
        # 4-dim diploid SNP genotype class one-hot:
        #   class 0 = REF/REF, class 1 = REF/ALT, class 2 = ALT/ALT
        # (Homozygous REF, Heterozygous, Homozygous ALT)
        # Encoding vector is 8-dim: [A, C, G, T] for hap-A + [A, C, G, T] for hap-B
        ref_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        alt_map = {'A': 4, 'C': 5, 'G': 6, 'T': 7}
        vec_to_allele = [0, 1, 2, 3, 0, 1, 2, 3]
        vec = original_encoding.cpu().numpy()

        ref_nuc_idx = ref_map.get(ref, -1)
        alt_nuc_idx = ref_map.get(alt, -1)
        ref_pos = ref_map.get(ref, -1)   # one-hot position for REF: 0-3
        alt_pos = alt_map.get(alt, -1)   # one-hot position for ALT: 4-7
        if ref_nuc_idx < 0 or alt_nuc_idx < 0 or ref_pos < 0 or alt_pos < 0:
            return []

        # Determine current genotype class from the 8-dim encoding
        # For a biallelic SNP, only 3 classes are possible: REF/REF, REF/ALT, ALT/ALT
        hap_a_allele = -1
        hap_b_allele = -1
        for base_idx in range(4):
            if vec[base_idx] > 0.5:
                hap_a_allele = vec_to_allele[base_idx]
                break
        for base_idx in range(4, 8):
            if vec[base_idx] > 0.5:
                hap_b_allele = vec_to_allele[base_idx]
                break

        allele_a = hap_a_allele if hap_a_allele >= 0 else ref_nuc_idx
        allele_b = hap_b_allele if hap_b_allele >= 0 else ref_nuc_idx

        is_ref_a = allele_a == ref_nuc_idx
        is_ref_b = allele_b == ref_nuc_idx
        is_alt_a = allele_a == alt_nuc_idx
        is_alt_b = allele_b == alt_nuc_idx

        orig_class = 0  # REF/REF
        if (is_ref_a and is_alt_b) or (is_alt_a and is_ref_b):
            orig_class = 1  # REF/ALT (heterozygous)
        elif is_alt_a and is_alt_b:
            orig_class = 2  # ALT/ALT (homozygous ALT)

        all_gt_classes = set(range(3))
        if available_gts is not None:
            all_gt_classes = available_gts

        if homozygous_only:
            all_gt_classes = {c for c in all_gt_classes if c not in (1,)}

        candidates = []
        for cls in all_gt_classes:
            if cls == orig_class:
                continue
            new_vec = np.zeros(8)
            if cls == 0:  # REF/REF
                new_vec[ref_pos] = 1.0
                new_vec[4 + ref_pos] = 1.0
            elif cls == 1:  # REF/ALT (heterozygous)
                new_vec[ref_pos] = 1.0
                new_vec[alt_pos] = 1.0
            elif cls == 2:  # ALT/ALT
                new_vec[alt_nuc_idx] = 1.0  # hap-A = ALT (nucleotide index for positions 0-3)
                new_vec[alt_pos] = 1.0       # hap-B = ALT (alt_map position for positions 4-7)
            candidates.append(torch.from_numpy(new_vec).float())
        return candidates

    elif encoding_dim == 4:
        # 4-dim genotype class one-hot: [0/0, 0/1, 1/0, 1/1]
        vec = original_encoding.cpu().numpy()
        orig_class = int(np.argmax(vec)) if vec.sum() > 0 else 0

        all_gts = set(range(4))
        if available_gts is not None:
            all_gts = available_gts

        if homozygous_only:
            # Exclude heterozygous (cls=1 and cls=2) — only 0/0 and 1/1 allowed
            all_gts = {c for c in all_gts if c not in (1, 2)}

        candidates = []
        for cls in all_gts:
            if cls == orig_class:
                continue
            new_vec = np.zeros(4)
            new_vec[cls] = 1.0
            candidates.append(torch.from_numpy(new_vec).float())
        return candidates

    elif encoding_dim == 3:
        # 3-dim classic encoding: [0/0=0, 0/1=1, 1/1=2]
        vec = original_encoding.cpu().numpy()
        orig_class = int(np.sum(np.arange(3) * vec)) if vec.sum() > 0 else 0

        all_gts = set(range(3))
        if available_gts is not None:
            all_gts = available_gts

        if homozygous_only:
            # Exclude heterozygous (cls=1) — only 0/0 and 1/1 allowed
            all_gts = {c for c in all_gts if c != 1}

        candidates = []
        for cls in all_gts:
            if cls == orig_class:
                continue
            new_vec = np.zeros(3)
            new_vec[cls] = 1.0
            candidates.append(torch.from_numpy(new_vec).float())
        return candidates

    elif encoding_dim == 1:
        # 1-dim token: additive {0, 1, 2}
        vec = original_encoding.cpu().numpy()
        orig_val = int(round(float(vec[0]))) if vec.ndim == 1 else int(round(float(vec[0])))

        candidates = []
        for val in range(3):
            if val == orig_val:
                continue
            if homozygous_only and val == 1:
                continue
            candidates.append(torch.tensor([[float(val)]], dtype=torch.float32))
        return candidates

    return []


###############################################################################
# Monte-Carlo combinatorial gain estimation
###############################################################################

def compute_combinatorial_gains(
    model,
    variant_tensors,
    variant_info: dict,
    current_preds_norm: np.ndarray,
    snp_indices: List[int],
    evolvable_pool: List[int],
    direction: Optional[DirectionModes],
    regression_tasks: List[str],
    encoding_dim: int,
    panel_genotypes: Optional[List[set]],
    n_mc: int,
    top_k: int,
    rng,
    device: str,
    is_multi_branch: bool,
    norm_stats: Optional[dict],
    homozygous_only: bool = False,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Estimate combinatorial marginal contribution for each SNP via Monte-Carlo sampling.

    For each SNP i in snp_indices:
        For each background sample m = 1..n_mc:
            Sample a random background S_m (size ~top_k, from evolvable_pool excluding i)
            SI_bg[m] = SI(S_m)
            For each of the N_cand alternative genotypes gt of SNP i:
                delta[m, gt] = SI(S_m with gt) - SI_bg[m]
            delta_best[m] = max_gt(delta[m, gt])
        gain_i = mean_m(delta_best[m])

    Returns:
        combo_gains: array of shape (len(snp_indices),) — average combinatorial gain per SNP
        snp_details: list of dicts with per-SNP stats (mean/std/median/max/min/n_samples)
    """
    n_snps = len(snp_indices)
    pool_list = list(evolvable_pool)

    # Step 1: sample background SNP indices for each of the M backgrounds
    bg_idx_matrix = np.full((n_mc, top_k), -1, dtype=int)
    for m in range(n_mc):
        size_m = rng.integers(1, top_k + 1)
        pool_excl = [j for j in pool_list if j not in snp_indices]
        if not pool_excl:
            pool_excl = pool_list
        chosen = rng.choice(pool_excl, size=min(size_m, len(pool_excl)), replace=False)
        bg_idx_matrix[m, :len(chosen)] = chosen

    # Step 2: build background tensors and all (background + alt) tensors
    alt_tensors_per_snp: List[List] = [[] for _ in range(n_snps)]
    bg_tensors_list = []

    for m in range(n_mc):
        bg_indices = [idx for idx in bg_idx_matrix[m] if idx != -1]
        if bg_indices:
            bg_mutations = [(idx, None) for idx in bg_indices]
            bg_mut_tensors = apply_multi_snp_mutations(variant_tensors, bg_mutations, is_multi_branch)
        else:
            bg_mut_tensors = (variant_tensors.clone() if not is_multi_branch
                              else {k: v.clone() for k, v in variant_tensors.items()})

        if is_multi_branch:
            bg_tensors_list.append({k: vv[0] for k, vv in bg_mut_tensors.items()})
        else:
            bg_tensors_list.append(bg_mut_tensors[0])

        # For each SNP i, build S_m + i[gt] for all alternative gts
        for si_idx, snp_idx in enumerate(snp_indices):
            ref = variant_info['snp']['refs'][snp_idx] if snp_idx < len(variant_info['snp']['refs']) else 'A'
            alt = variant_info['snp']['alts'][snp_idx] if snp_idx < len(variant_info['snp']['alts']) else 'T'
            avail_gts = panel_genotypes[snp_idx] if panel_genotypes else None

            if is_multi_branch:
                orig = variant_tensors['snp'][0, snp_idx]
            else:
                orig = variant_tensors[0, snp_idx]

            candidates = build_mutation_candidates_for_snp(orig, encoding_dim, ref, alt, avail_gts, homozygous_only)
            if not candidates:
                candidates = [orig]

            for cand in candidates:
                if bg_indices:
                    base_tensors = apply_multi_snp_mutations(
                        variant_tensors,
                        [(idx, None) for idx in bg_indices],
                        is_multi_branch,
                    )
                else:
                    base_tensors = (variant_tensors.clone() if not is_multi_branch
                                    else {k: v.clone() for k, v in variant_tensors.items()})
                mut = apply_single_snp_mutation(base_tensors, snp_idx, cand, is_multi_branch)
                if is_multi_branch:
                    alt_tensors_per_snp[si_idx].append({k: v[0] for k, v in mut.items()})
                else:
                    alt_tensors_per_snp[si_idx].append(mut[0])

    # Step 3: batch predict all backgrounds
    if is_multi_branch:
        bg_batch = {k: torch.stack([t[k] for t in bg_tensors_list]) for k in bg_tensors_list[0]}
    else:
        bg_batch = torch.stack(bg_tensors_list)

    bg_preds_norm = predict_all_phenos(model, bg_batch, device, is_multi_branch)
    bg_si_vals = _si_from_preds(bg_preds_norm, direction, current_preds_norm)  # (n_mc,)

    # Step 4: batch predict all SNP+background combos per SNP
    combo_gains  = np.zeros(n_snps)
    snp_details: List[dict] = []
    batch_size = 256

    for si_idx, snp_idx in enumerate(snp_indices):
        tensors  = alt_tensors_per_snp[si_idx]
        n_tensors = len(tensors)
        n_cands  = n_tensors // n_mc  # should be ~3

        if is_multi_branch:
            batch_chunks = [tensors[p:p + batch_size] for p in range(0, n_tensors, batch_size)]
            preds_chunks = []
            for chunk in batch_chunks:
                batched = {k: torch.stack([t[k] for t in chunk]) for k in chunk[0]}
                preds_chunks.append(predict_all_phenos(model, batched, device, is_multi_branch))
            alt_preds_norm = np.concatenate(preds_chunks, axis=0)
        else:
            batch_chunks = [tensors[p:p + batch_size] for p in range(0, n_tensors, batch_size)]
            preds_chunks = [predict_all_phenos(torch.stack(c), device, is_multi_branch) for c in batch_chunks]
            alt_preds_norm = np.concatenate(preds_chunks, axis=0)

        alt_si = _si_from_preds(alt_preds_norm, direction, current_preds_norm)
        alt_si = alt_si.reshape(n_mc, n_cands)  # (M, n_cands)

        deltas    = alt_si - bg_si_vals[:, np.newaxis]  # (M, n_cands)
        best_per_bg = deltas.max(axis=1)  # (M,)
        gain = float(best_per_bg.mean())

        combo_gains[si_idx] = gain
        snp_details.append({
            'mean':     gain,
            'std':      float(best_per_bg.std()),
            'median':   float(np.median(best_per_bg)),
            'max':      float(best_per_bg.max()),
            'min':      float(best_per_bg.min()),
            'n_samples': int(n_mc),
        })

    return combo_gains, snp_details


###############################################################################
# Per-SNP contribution decomposition
###############################################################################

def per_snp_contribution_in_round(rr: RoundResult) -> Dict[int, float]:
    """
    Decompose the round SI gain into per-SNP marginal contributions.

    For each SNP i in the accepted combo:
        contrib_i = SI(full_combo) - SI(full_combo_minus_i)
                  = rr.si_evolved - rr.combo_results[full_mask ^ (1<<pos_i)]['_gain']

    Falls back to rr.snp_gains[i] when the minus-mask is not in combo_results.

    Returns dict: snp_idx -> SI contribution for this round.
    """
    contribs: Dict[int, float] = {}
    if not rr.accepted or not rr.combo_results:
        return contribs

    snp_to_pos: Dict[int, int] = {}
    for pos, (snp_idx, _, _) in enumerate(rr.mutations):
        snp_to_pos[snp_idx] = pos
    if not snp_to_pos:
        return contribs

    k = len(snp_to_pos)
    full_mask = (1 << k) - 1
    full_si = rr.si_evolved

    for snp_idx, pos in snp_to_pos.items():
        minus_mask = full_mask ^ (1 << pos)
        minus_si = rr.combo_results.get(minus_mask, {}).get('_gain')
        if minus_si is not None:
            contribs[snp_idx] = full_si - minus_si
        else:
            contribs[snp_idx] = rr.snp_gains.get(snp_idx, 0.0)

    return contribs


###############################################################################
# Strategy: screening
###############################################################################

def _screening_evaluate_one_snp(
    model, variant_tensors, variant_info, snp_idx,
    encoding_dim, device, is_multi_branch,
    regression_tasks, norm_stats, use_selection_index,
    direction, current_preds_norm, panel_genotypes,
    original_preds_norm=None,
    homozygous_only: bool = False,
) -> Optional[dict]:
    """Evaluate all mutation candidates for a single SNP. Returns best candidate info."""
    if is_multi_branch:
        orig = variant_tensors['snp'][0, snp_idx]
    else:
        orig = variant_tensors[0, snp_idx]

    ref = variant_info['snp']['refs'][snp_idx] if snp_idx < len(variant_info['snp']['refs']) else 'A'
    alt = variant_info['snp']['alts'][snp_idx] if snp_idx < len(variant_info['snp']['alts']) else 'T'
    avail_gts = panel_genotypes[snp_idx] if panel_genotypes else None

    candidates = build_mutation_candidates_for_snp(orig, encoding_dim, ref, alt, avail_gts, homozygous_only)
    if not candidates:
        return None

    mutated_batch = []
    for cand in candidates:
        mut = apply_single_snp_mutation(variant_tensors, snp_idx, cand, is_multi_branch)
        if is_multi_branch:
            mutated_batch.append({k: v[0] for k, v in mut.items()})
        else:
            mutated_batch.append(mut[0])

    if is_multi_branch:
        batch = {k: torch.stack([m[k] for m in mutated_batch] + [variant_tensors[k][0]])
                 for k in mutated_batch[0].keys()}
    else:
        batch = torch.stack(mutated_batch + [variant_tensors[0]])

    preds_all_norm = predict_all_phenos(model, batch, device, is_multi_branch)
    n_cands = len(candidates)

    if use_selection_index and current_preds_norm is not None:
        alt_preds_norm = preds_all_norm[:n_cands]
        orig_preds_norm = preds_all_norm[n_cands:]

        # Compute directional SI gain using full SI logic (supports all modes).
        # We compute SI(baseline) once, then SI(baseline + candidate) for each candidate.
        baseline_si = compute_selection_index_normalized(
            current_preds_norm.reshape(1, -1), direction, current_preds_norm.reshape(1, -1)
        )[0]

        # Build (n_cands + 1, n_tasks) array: first n_cands rows = candidate predictions,
        # last row = original (unchanged) predictions.
        stacked = np.vstack([alt_preds_norm, orig_preds_norm])
        alt_si_vals = compute_selection_index_normalized(
            stacked, direction, current_preds_norm.reshape(1, -1)
        )
        # alt_si_vals[0:n_cands] = SI for each candidate; alt_si_vals[n_cands] = SI for original
        gains = alt_si_vals[:n_cands] - baseline_si
        best_gain = float(gains.max())
        best_idx  = int(gains.argmax())
    else:
        target_preds = preds_all_norm[:n_cands]
        orig_preds   = preds_all_norm[n_cands:]
        best_idx     = int(target_preds.mean(axis=1).argmax())
        best_gain    = float(target_preds[best_idx, 0] - orig_preds[0, 0])
        

    # Per-phenotype predictions for the best candidate
    best_enc = candidates[best_idx]
    mut_best = apply_single_snp_mutation(variant_tensors, snp_idx, best_enc, is_multi_branch)
    if is_multi_branch:
        pred_batch = {k: v[0:1].to(device) for k, v in mut_best.items()}
    else:
        pred_batch = mut_best[0:1].to(device)
    preds_norm = predict_all_phenos(model, pred_batch, device, is_multi_branch)

    log_tasks = norm_stats.get('log_transformed_tasks', []) if isinstance(norm_stats, dict) else []
    if isinstance(norm_stats, dict) and 'regression_means' in norm_stats:
        preds_denorm = denormalize_predictions(preds_norm, norm_stats, regression_tasks, log_tasks)[0]
    else:
        preds_denorm = preds_norm[0]

    # For SI mode: compute SI of the original (VCF) genotype as reference.
    # This allows computing the contribution of each SNP relative to the
    # ORIGINAL genotype (not the cumulative genotype at start of this round).
    si_original_contribution = None
    if use_selection_index and original_preds_norm is not None:
        # SI of original genotype
        si_original = float(compute_selection_index_normalized(
            original_preds_norm.reshape(1, -1), direction, original_preds_norm.reshape(1, -1)
        )[0])
        # SI of this SNP (best candidate) relative to ORIGINAL genotype
        si_snp_on_original = float(compute_selection_index_normalized(
            alt_preds_norm[best_idx:best_idx+1], direction, original_preds_norm.reshape(1, -1)
        )[0])
        si_original_contribution = si_snp_on_original - si_original
    elif original_preds_norm is not None:
        # Non-SI mode: phenotype difference from original genotype
        si_original_contribution = float(target_preds[best_idx, 0] - orig_preds[0, 0])
    else:
        # Fallback: no original preds provided, use current baseline as reference
        si_original_contribution = best_gain

    all_gains = None
    if use_selection_index and current_preds_norm is not None:
        all_gains = (alt_si_vals[:n_cands] - baseline_si).tolist()

    return {
        'gain':       best_gain,
        'best_cand':  best_enc,
        'best_idx':   best_idx,
        'all_gains':  all_gains,
        'predictions': preds_denorm,
        'si_original_contribution': si_original_contribution,
    }


def homozygize_genotype_tensors(
    model,
    variant_tensors,
    variant_info: dict,
    device: str,
    is_multi_branch: bool,
    encoding_dim: int,
    direction: Optional[DirectionModes],
    current_preds_norm: np.ndarray,
    regression_tasks: List[str],
    norm_stats: Optional[dict],
) -> torch.Tensor:
    """
    Convert all heterozygous genotypes in variant_tensors to homozygous (REF/REF or ALT/ALT).

    For each SNP that is currently heterozygous, evaluates both REF/REF and ALT/ALT
    and picks whichever gives the higher SI gain relative to baseline.
    Works for all encoding dimensions (8, 4, 3, 1).

    Returns a new tensor with all heterozygous genotypes replaced.
    """
    if is_multi_branch:
        result = {k: v.clone() for k, v in variant_tensors.items()}
        n_snps = variant_tensors['snp'].shape[1]
    else:
        result = variant_tensors.clone()
        n_snps = variant_tensors.shape[1]

    ref_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    n_het_found = 0
    n_het_fixed = 0

    for snp_idx in range(n_snps):
        if is_multi_branch:
            orig = variant_tensors['snp'][0, snp_idx]
        else:
            orig = variant_tensors[0, snp_idx]

        vec = orig.cpu().numpy()
        ref = variant_info['snp']['refs'][snp_idx] if snp_idx < len(variant_info['snp']['refs']) else 'A'
        alt = variant_info['snp']['alts'][snp_idx] if snp_idx < len(variant_info['snp']['alts']) else 'T'
        ref_idx = ref_map.get(ref, 0)
        alt_idx = ref_map.get(alt, 1)

        # Check if this SNP is currently heterozygous
        is_het = False
        if encoding_dim == 8:
            vec_to_allele = [0, 1, 2, 3, 0, 1, 2, 3]
            hap_a = vec_to_allele[int(np.argmax(vec[:4]))]
            hap_b = vec_to_allele[4 + int(np.argmax(vec[4:]))]
            is_het = (hap_a != hap_b)
        elif encoding_dim == 4:
            if vec.sum() > 0:
                cls = int(np.argmax(vec))
                is_het = (cls in (1, 2))
        elif encoding_dim == 3:
            if vec.sum() > 0:
                cls = int(np.sum(np.arange(3) * vec))
                is_het = (cls == 1)
        elif encoding_dim == 1:
            val = int(round(float(vec[0])))
            is_het = (val == 1)

        if not is_het:
            continue
        n_het_found += 1

        # Build the two homozygous candidates
        cand_ref = _make_homozygous_encoding(ref_idx, alt_idx, encoding_dim, is_ref=True)
        cand_alt = _make_homozygous_encoding(ref_idx, alt_idx, encoding_dim, is_ref=False)

        # Apply both candidates to the current tensor to get properly shaped batch dicts
        mut_ref = apply_single_snp_mutation(variant_tensors, snp_idx, cand_ref, is_multi_branch)
        mut_alt = apply_single_snp_mutation(variant_tensors, snp_idx, cand_alt, is_multi_branch)

        # Collect all variants in this batch
        if is_multi_branch:
            variant_keys = list(mut_ref.keys())
            batch = {k: torch.stack([mut_ref[k][0], mut_alt[k][0], variant_tensors[k][0]])
                     for k in variant_keys}
        else:
            batch = torch.stack([mut_ref[0], mut_alt[0], variant_tensors[0]])

        preds_norm = predict_all_phenos(model, batch, device, is_multi_branch)

        # Pick the one with higher SI
        if direction is not None:
            gains = _si_from_preds(preds_norm, direction, current_preds_norm)
            # cand_ref=idx0 (REF/REF), cand_alt=idx1 (ALT/ALT), original=idx2
            best_cand = cand_ref if gains[0] >= gains[1] else cand_alt
            pick_label = 'REF/REF' if gains[0] >= gains[1] else 'ALT/ALT'
        else:
            # Fall back: pick REF/REF
            best_cand = cand_ref
            pick_label = 'REF/REF'

        # Apply the best homozygous encoding to the result tensor
        if is_multi_branch:
            result['snp'][0, snp_idx] = best_cand
            stored = result['snp'][0, snp_idx].cpu()
        else:
            result[0, snp_idx] = best_cand
            stored = result[0, snp_idx]
        n_het_fixed += 1

        gain_str = f"(gain_REF={gains[0]:.4f}, gain_ALT={gains[1]:.4f})" if direction else "(gain=N/A)"
        print(f"  [homozygize] SNP idx={snp_idx} ref={ref} alt={alt}: "
              f"orig={vec} → pick={pick_label} {gain_str} → stored={stored.numpy()}")

    print(f"  [homozygize] Done: {n_het_found} het SNPs found, {n_het_fixed} fixed")

    return result


def _make_homozygous_encoding(
    ref_idx: int, alt_idx: int, encoding_dim: int, is_ref: bool
) -> torch.Tensor:
    """Create a homozygous (REF/REF or ALT/ALT) encoding tensor."""
    allele_idx = ref_idx if is_ref else alt_idx
    if encoding_dim == 8:
        vec = np.zeros(8)
        vec[allele_idx] = 1.0
        vec[4 + allele_idx] = 1.0
        result = torch.from_numpy(vec).float()
        print(f"    [_make_homo] is_ref={is_ref} allele_idx={allele_idx} → {result.cpu().numpy()}")
        return result
    elif encoding_dim == 4:
        vec = np.zeros(4)
        vec[0 if is_ref else 3] = 1.0
        return torch.from_numpy(vec).float()
    elif encoding_dim == 3:
        vec = np.zeros(3)
        vec[0 if is_ref else 2] = 1.0
        return torch.from_numpy(vec).float()
    elif encoding_dim == 1:
        return torch.tensor([[float(1 if is_ref else 2)]], dtype=torch.float32)
    return torch.zeros(encoding_dim)


def strategy_screening(
    model, variant_tensors, variant_info, current_score,
    encoding_dim, device, is_multi_branch, pheno_idx, mode, top_k, min_improve,
    verbose, regression_tasks, norm_stats,
    current_preds, current_preds_norm,
    use_selection_index, direction, panel_genotypes,
    evolvable_indices, snp_priority_order, mc_samples,
    original_preds_norm=None,
    apply_all_positive_gain=True,
    homozygous_only: bool = False,
    seed: int = 42,
) -> Tuple:
    """
    Strategy 1: SNP screening — evaluate candidates then apply beneficial mutations.

    Two modes:
      (a) priority order: SNPs from target VCF comparison, mutate one per round
      (b) full screening: evaluate all evolvable SNPs, apply all with positive gain

    apply_all_positive_gain (default True): apply ALL SNPs with positive gain.
      If False (set when --top-k is explicitly used in screening), cap at top_k.

    Supports --mc-samples >= 2 for Monte-Carlo interaction-aware gain estimation.
    """
    if snp_priority_order:
        # Mode (a): target-VCF guided — mutate in priority order
        candidates_to_evaluate = snp_priority_order
    else:
        # Mode (b): full screening of evolvable pool
        pool = evolvable_indices.copy() if evolvable_indices is not None \
               else list(range(variant_tensors.shape[1] if not is_multi_branch
                               else variant_tensors['snp'].shape[1]))
        rng = np.random.RandomState(seed)
        rng.shuffle(pool)
        candidates_to_evaluate = pool

    eval_results = {}
    for snp_idx in candidates_to_evaluate:
        result = _screening_evaluate_one_snp(
            model, variant_tensors, variant_info, snp_idx,
            encoding_dim, device, is_multi_branch,
            regression_tasks, norm_stats, use_selection_index,
            direction, current_preds_norm, panel_genotypes,
            original_preds_norm=original_preds_norm,
            homozygous_only=homozygous_only,
        )
        if result is not None:
            eval_results[snp_idx] = result

    # Select SNPs to apply: all positive-gain SNPs, or top_k if apply_all_positive_gain=False
    sorted_snps = sorted(eval_results.keys(),
                         key=lambda i: eval_results[i]['gain'], reverse=True)
    if apply_all_positive_gain:
        # For pure screening: apply all SNPs with positive individual gain,
        # regardless of whether cumulative evolved score passes the acceptance gate.
        # This mirrors combinatorial mode (which always applies the best combo).
        selected_snps = [i for i in sorted_snps if eval_results[i]['gain'] > 0]
    else:
        selected_snps = sorted_snps[:top_k]

    # Apply mutations for selected SNPs
    current_tensors = clone_genotype_tensors(variant_tensors, is_multi_branch)
    accepted_mutations = []
    for snp_idx in selected_snps:
        result = eval_results[snp_idx]
        if result['gain'] > 0:
            current_tensors = apply_single_snp_mutation(
                current_tensors, snp_idx, result['best_cand'], is_multi_branch,
            )
            accepted_mutations.append((snp_idx, result['best_cand'], result['gain']))

    # Final prediction for accepted mutations
    if is_multi_branch:
        final_batch = {k: v[0:1].to(device) for k, v in current_tensors.items()}
    else:
        final_batch = current_tensors[0:1].to(device)
    evolved_preds_norm = predict_all_phenos(model, final_batch, device, is_multi_branch)

    log_tasks = norm_stats.get('log_transformed_tasks', []) if isinstance(norm_stats, dict) else []
    if isinstance(norm_stats, dict) and 'regression_means' in norm_stats:
        evolved_preds = denormalize_predictions(
            evolved_preds_norm, norm_stats, regression_tasks, log_tasks
        )[0]
    else:
        evolved_preds = evolved_preds_norm[0]

    if use_selection_index and direction is not None:
        evolved_score = float(
            compute_selection_index_normalized(
                evolved_preds_norm, direction, current_preds_norm,
            )[0]
        )
    else:
        evolved_score = float(evolved_preds[pheno_idx])

    accepted = evolved_score >= current_score + min_improve

    rr = RoundResult(
        round_num=0, si_baseline=current_score, si_evolved=evolved_score,
        si_gain=evolved_score - current_score, accepted=accepted,
        mutations=[(s, e, g) for s, e, g in accepted_mutations],
        snp_gains={i: eval_results[i]['gain'] for i in eval_results},
    )

    log_dict = {
        'strategy':   'screening',
        'eval_results': eval_results,
        'mutations':  accepted_mutations,
        'predictions': evolved_preds,
    }
    return current_tensors, evolved_score, accepted_mutations, log_dict


###############################################################################
# Strategy: combinatorial
###############################################################################

def strategy_combinatorial(
    model, variant_tensors, variant_info, current_score,
    encoding_dim, device, is_multi_branch, pheno_idx, mode, top_k, min_improve,
    seed, verbose, regression_tasks, norm_stats,
    current_preds, current_preds_norm,
    use_selection_index, direction, panel_genotypes,
    evolvable_indices, mc_samples,
    homozygous_only: bool = False,
) -> Tuple:
    """
    Strategy 2: Random SNP selection + exhaustive 2^K combinatorial search.

    1. Randomly sample top_k SNPs from the evolvable pool
    2. Evaluate all 2^K combinations by batch prediction
    3. Return the best combination

    With --mc-samples >= 2: uses Monte-Carlo Shapley-style gain estimation
    to prioritize SNPs with more stable interaction-aware contributions.
    """
    rng = np.random.RandomState(seed)

    pool = evolvable_indices if evolvable_indices is not None \
           else list(range(variant_tensors.shape[1] if not is_multi_branch
                           else variant_tensors['snp'].shape[1]))
    selected_indices = rng.choice(pool, size=min(top_k, len(pool)), replace=False).tolist()

    best_mutations: List[Tuple[int, object, float]] = []
    snp_sets:       List[Tuple[int, float]] = []
    per_snp_gains   = np.zeros(len(selected_indices))
    snp_screening   = np.zeros((len(selected_indices), len(regression_tasks)))

    for si_idx, snp_idx in enumerate(selected_indices):
        result = _screening_evaluate_one_snp(
            model, variant_tensors, variant_info, snp_idx,
            encoding_dim, device, is_multi_branch,
            regression_tasks, norm_stats, use_selection_index,
            direction, current_preds_norm, panel_genotypes,
            homozygous_only=homozygous_only,
        )
        if result is None:
            continue
        gain = result['gain']
        best_mutations.append((snp_idx, result['best_cand'], gain))
        snp_sets.append((snp_idx, gain))
        per_snp_gains[si_idx] = gain
        snp_screening[si_idx] = result['predictions']

    if not best_mutations:
        return None, current_score, [], {}

    best_mutations.sort(key=lambda x: x[2], reverse=True)
    best_mutations = best_mutations[:top_k]
    k = len(best_mutations)

    # Early exit: single SNP — just apply it
    if k == 0:
        return None, current_score, [], {}

    if k == 1:
        snp_idx, best_enc, gain = best_mutations[0]
        evolved_tensors = apply_single_snp_mutation(
            variant_tensors, snp_idx, best_enc, is_multi_branch,
        )
        if is_multi_branch:
            final_batch = {k: v[0:1].to(device) for k, v in evolved_tensors.items()}
        else:
            final_batch = evolved_tensors[0:1].to(device)
        evolved_preds_norm = predict_all_phenos(model, final_batch, device, is_multi_branch)
        log_tasks = norm_stats.get('log_transformed_tasks', []) if isinstance(norm_stats, dict) else []
        if isinstance(norm_stats, dict) and 'regression_means' in norm_stats:
            evolved_preds = denormalize_predictions(
                evolved_preds_norm, norm_stats, regression_tasks, log_tasks
            )[0]
        else:
            evolved_preds = evolved_preds_norm[0]

        if use_selection_index and direction is not None:
            evolved_score = float(
                compute_selection_index_normalized(
                    evolved_preds_norm, direction, current_preds_norm,
                )[0]
            )
        else:
            evolved_score = float(evolved_preds[pheno_idx])

        accepted = evolved_score >= current_score + min_improve
        rr = RoundResult(
            round_num=0, si_baseline=current_score, si_evolved=evolved_score,
            si_gain=evolved_score - current_score, accepted=accepted,
            mutations=[(snp_idx, best_enc, gain)],
            snp_gains={snp_idx: gain},
        )
        log_dict = {
            'strategy': 'combinatorial', 'mutations': [(snp_idx, best_enc, gain)],
            'combo_masks': np.zeros((2, 1), dtype=int),
            'combo_results': {}, 'combo_meta': None,
        }
        return evolved_tensors, evolved_score, [(snp_idx, best_enc, gain)], log_dict

    full_mask = (1 << k) - 1
    n_combos  = 2 ** k
    combo_batch = []

    for mask in range(n_combos):
        mutations = [
            (snp_idx, best_enc)
            for i, (snp_idx, best_enc, _) in enumerate(best_mutations)
            if mask & (1 << i)
        ]
        mut_tensors = apply_multi_snp_mutations(variant_tensors, mutations, is_multi_branch)
        if is_multi_branch:
            combo_batch.append({kk: vv[0] for kk, vv in mut_tensors.items()})
        else:
            combo_batch.append(mut_tensors[0])

    if is_multi_branch:
        combo_tensor = {k: torch.stack([b[k] for b in combo_batch]) for k in combo_batch[0].keys()}
    else:
        combo_tensor = torch.stack(combo_batch)

    combo_preds_norm = predict_all_phenos(model, combo_tensor, device, is_multi_branch)

    log_tasks = norm_stats.get('log_transformed_tasks', []) if isinstance(norm_stats, dict) else []
    if isinstance(norm_stats, dict) and 'regression_means' in norm_stats:
        combo_preds_denorm = denormalize_predictions(
            combo_preds_norm, norm_stats, regression_tasks, log_tasks,
        )
    else:
        combo_preds_denorm = combo_preds_norm

    # Determine best combo
    if use_selection_index and current_preds_norm is not None:
        si_vals         = compute_selection_index_normalized(combo_preds_norm, direction, current_preds_norm)
        best_combo_idx  = int(np.argmax(si_vals))
        best_gain       = float(si_vals[best_combo_idx])
        evolved_absolute = current_score + best_gain
    elif mode == 'maximize':
        combo_preds_target = combo_preds_denorm[:, pheno_idx]
        best_combo_idx     = int(np.argmax(combo_preds_target))
        best_gain          = float(combo_preds_target[best_combo_idx] - current_score)
        evolved_absolute   = best_gain
    else:
        combo_preds_target = combo_preds_denorm[:, pheno_idx]
        best_combo_idx     = int(np.argmin(combo_preds_target))
        best_gain          = float(current_score - combo_preds_target[best_combo_idx])
        evolved_absolute   = best_gain

    # Build combo_results dict: mask_int -> {trait: score, _gain: SI_gain}
    combo_masks = np.zeros((n_combos, k), dtype=int)
    for mask in range(n_combos):
        for i in range(k):
            if mask & (1 << i):
                combo_masks[mask, i] = 1

    combo_results: Dict[int, Dict[str, float]] = {}
    combo_preds_target_norm = combo_preds_norm[:, pheno_idx]

    for mask_idx in range(n_combos):
        combo_results[mask_idx] = {
            t: float(combo_preds_denorm[mask_idx, i])
            for i, t in enumerate(regression_tasks)
        }
        if use_selection_index and current_preds_norm is not None:
            combo_results[mask_idx]['_gain'] = float(si_vals[mask_idx])
        else:
            combo_results[mask_idx]['_gain'] = float(combo_preds_target_norm[mask_idx] - current_score)

    # Apply accepted mutations (2-tuple for apply_multi_snp_mutations)
    accepted_mutations = [
        (snp_idx, best_enc)
        for i, (snp_idx, best_enc, _) in enumerate(best_mutations)
        if best_combo_idx & (1 << i)
    ]
    evolved_tensors = (
        apply_multi_snp_mutations(variant_tensors, accepted_mutations, is_multi_branch)
        if accepted_mutations
        else variant_tensors
    )
    mutations_for_result = [
        (snp_idx, best_enc, gain)
        for i, (snp_idx, best_enc, gain) in enumerate(best_mutations)
        if best_combo_idx & (1 << i)
    ]

    accepted = evolved_absolute >= current_score + min_improve

    rr = RoundResult(
        round_num=0, si_baseline=current_score, si_evolved=evolved_absolute,
        si_gain=best_gain, accepted=accepted,
        combo_results=combo_results,
        snp_gains={s: g for s, g in snp_sets},
        mutations=mutations_for_result,
    )
    contrib = per_snp_contribution_in_round(rr)

    # Build flat gains array (for log writers that expect this format)
    combo_gains_arr = np.array([combo_results.get(i, {}).get('_gain', 0.0) for i in range(n_combos)])
    snp_sets_list = [(s, g) for s, g in snp_sets]

    log_dict = {
        'strategy':      'combinatorial',
        'mutations':     mutations_for_result,
        'combo_masks':   combo_masks,
        'combo_results': combo_results,
        # Flat arrays for log writers
        'gains':         combo_gains_arr,
        'all_preds':     combo_preds_denorm,
        'snp_sets':      snp_sets_list,
        'target_preds':  combo_preds_denorm[:, pheno_idx] if combo_preds_denorm is not None and combo_preds_denorm.ndim > 1 else np.array([]),
        'per_snp_gains': per_snp_gains,
        'snp_screening': snp_screening,
        'per_round_contrib': contrib,
        'combo_meta': {
            'full_mask': full_mask,
            'snp_pos': {snp_idx: pos for pos, (snp_idx, _, _) in enumerate(best_mutations)},
        },
    }
    return evolved_tensors, evolved_absolute, mutations_for_result, log_dict
