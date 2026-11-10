#!/usr/bin/env python3
"""Build a parent-specific QTN-to-proxy SNP mapping with a fold ensemble."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from aquila.evolve import (
    DirectionModes,
    compute_selection_index_normalized,
    get_encoding_dim,
    load_direction_file,
    load_vcf_data,
    normalize_trait_value,
)
from aquila.qtn_proxy import (
    ProxyCandidate,
    evolution_edit,
    marker_coordinate,
    opposite_homozygous_flip,
    resolve_qtn_marker,
    select_proxy_candidate,
)
from aquila.scripts.aquila_evolve_ensemble import (
    EvolutionEnsembleBackend,
    EvolutionFoldEnsemble,
)


SUPPORTED_COSINE_MODES = {"maximize", "minimize", "maintain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map each homozygous QTN to the strongest same-direction proxy "
            "within a parent-specific genomic window"
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing fold_0/best_model.pt through fold_9",
    )
    parser.add_argument("--vcf", required=True, help="Single-parent VCF")
    parser.add_argument(
        "--qtn-list",
        required=True,
        help="One model/VCF SNP ID per line",
    )
    parser.add_argument(
        "--direction-file",
        required=True,
        help="Trait direction file used by subsequent directed evolution",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory for qtn_proxy_map.{tsv,json} and proxy_sites.list",
    )
    parser.add_argument(
        "--window-kb",
        type=float,
        default=500.0,
        help="Symmetric proxy search window in kb (default: %(default)s)",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.8,
        help="Minimum multi-trait effect cosine (default: %(default)s)",
    )
    parser.add_argument(
        "--si-epsilon",
        type=float,
        default=1e-4,
        help="Absolute SI/effect norm treated as zero (default: %(default)s)",
    )
    parser.add_argument(
        "--cosine-modes",
        default="maximize,minimize,maintain",
        help=(
            "Comma-separated direction modes included in cosine; allowed: "
            "maximize,minimize,maintain"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of single-marker mutants per inference batch",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def read_id_list(path: str | Path) -> list[str]:
    identifiers = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            identifier = line.strip()
            if not identifier or identifier.startswith("#") or identifier in seen:
                continue
            seen.add(identifier)
            identifiers.append(identifier)
    if not identifiers:
        raise ValueError(f"No QTN IDs found in {path}")
    return identifiers


def file_signature(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_direction(
    direction_file: str,
    tasks: list[str],
    baseline_original: np.ndarray,
    norm_stats: dict | None,
) -> DirectionModes:
    names, weights, modes, numeric_values = load_direction_file(direction_file)
    weights_by_name = dict(zip(names, weights))
    modes_by_name = dict(zip(names, modes))
    numeric_by_name = dict(zip(names, numeric_values))
    aligned_weights = np.asarray(
        [weights_by_name.get(task, 1.0) for task in tasks], dtype=float
    )
    aligned_modes = np.asarray(
        [modes_by_name.get(task, "neutral") for task in tasks], dtype=object
    )
    numeric_targets = np.zeros(len(tasks), dtype=float)
    has_numeric = False
    for index, (task, mode) in enumerate(zip(tasks, aligned_modes)):
        if mode != "numeric_target":
            continue
        target_original = baseline_original[index] * (
            1.0 + float(numeric_by_name.get(task, 0.0)) / 100.0
        )
        numeric_targets[index] = normalize_trait_value(
            target_original, norm_stats, task
        )
        has_numeric = True
    return DirectionModes(
        modes=aligned_modes,
        weights=aligned_weights,
        numeric_targets_norm=numeric_targets,
        _is_numeric=has_numeric,
    )


def parse_cosine_modes(value: str) -> set[str]:
    modes = {part.strip().lower() for part in value.split(",") if part.strip()}
    invalid = sorted(modes - SUPPORTED_COSINE_MODES)
    if invalid:
        raise ValueError(
            "Unsupported --cosine-modes: " + ", ".join(invalid)
        )
    if not modes:
        raise ValueError("--cosine-modes must include at least one mode")
    return modes


def _single_sample_batch(variant_tensors, is_multi_branch: bool, device: str):
    if is_multi_branch:
        return {key: value[0:1].to(device) for key, value in variant_tensors.items()}
    return variant_tensors[0:1].to(device)


def initialize_ensemble(
    backend: EvolutionEnsembleBackend,
    config: dict,
    seq_length,
    variant_tensors,
    is_multi_branch: bool,
    device: str,
) -> EvolutionFoldEnsemble:
    model = backend.create_model(
        config, seq_length, device, backend.canonical_norm_stats
    )
    sample_batch = _single_sample_batch(
        variant_tensors, is_multi_branch, device
    )
    model.eval()
    with torch.no_grad():
        model(sample_batch)
    model, _ = backend.load_checkpoint(model, backend.checkpoints[0], device)
    return model


def predict_single_flips(
    model: EvolutionFoldEnsemble,
    variant_tensors,
    flips: dict[int, torch.Tensor],
    is_multi_branch: bool,
    device: str,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Predict canonical-Z traits for unique single-marker flips."""
    indices = sorted(flips)
    predictions: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            if is_multi_branch:
                batch = {
                    key: value[0:1].repeat(
                        len(chunk), *([1] * (value.ndim - 1))
                    )
                    for key, value in variant_tensors.items()
                }
                for row, marker_index in enumerate(chunk):
                    batch["snp"][row, marker_index] = flips[marker_index]
                device_batch = {
                    key: value.to(device) for key, value in batch.items()
                }
            else:
                batch = variant_tensors[0:1].repeat(
                    len(chunk), *([1] * (variant_tensors.ndim - 1))
                )
                for row, marker_index in enumerate(chunk):
                    batch[row, marker_index] = flips[marker_index]
                device_batch = batch.to(device)
            components = model.predict_components(device_batch)
            values = components.canonical_normalized.detach().cpu().numpy()
            predictions.update(
                {marker_index: values[row] for row, marker_index in enumerate(chunk)}
            )
    return predictions


def chromosome_index(
    chroms: Iterable,
    positions: Iterable,
) -> dict[str, tuple[list[int], list[int]]]:
    grouped: dict[str, list[tuple[int, int]]] = {}
    for index, (chrom, position) in enumerate(zip(chroms, positions)):
        grouped.setdefault(str(chrom), []).append((int(position), index))
    result = {}
    for chrom, values in grouped.items():
        values.sort()
        result[chrom] = (
            [position for position, _ in values],
            [index for _, index in values],
        )
    return result


def indices_in_window(
    index: dict[str, tuple[list[int], list[int]]],
    chrom: str,
    position: int,
    window_bp: int,
) -> list[int]:
    if chrom not in index:
        return []
    positions, marker_indices = index[chrom]
    left = bisect.bisect_left(positions, position - window_bp)
    right = bisect.bisect_right(positions, position + window_bp)
    return marker_indices[left:right]


def run_mapping(args: argparse.Namespace) -> dict:
    if args.window_kb < 0:
        raise ValueError("--window-kb must be non-negative")
    if not -1.0 <= args.min_cosine <= 1.0:
        raise ValueError("--min-cosine must be in [-1, 1]")
    if args.si_epsilon < 0:
        raise ValueError("--si-epsilon must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    cosine_modes = parse_cosine_modes(args.cosine_modes)
    qtn_ids = read_id_list(args.qtn_list)
    backend = EvolutionEnsembleBackend(args.model_dir)
    config, _, norm_stats = backend.load_model_and_config(args.model_dir)
    encoding_type = config.get("data", {}).get(
        "encoding_type", "diploid_onehot"
    )
    if encoding_type in {"snp_vcf", "snp_indel_vcf", "snp_indel_sv_vcf"}:
        encoding_type = "diploid_onehot"
    (
        variant_tensors,
        variant_info,
        sample_ids,
        seq_length,
        is_multi_branch,
    ) = load_vcf_data(args.vcf, encoding_type, config)
    if len(sample_ids) != 1:
        raise ValueError(
            f"Proxy mapping requires exactly one parent sample, found {len(sample_ids)}"
        )
    model = initialize_ensemble(
        backend,
        config,
        seq_length,
        variant_tensors,
        is_multi_branch,
        args.device,
    )
    baseline_batch = _single_sample_batch(
        variant_tensors, is_multi_branch, args.device
    )
    with torch.no_grad():
        baseline_components = model.predict_components(baseline_batch)
    baseline_norm = (
        baseline_components.canonical_normalized[0].detach().cpu().numpy()
    )
    baseline_original = (
        baseline_components.mean_original[0].detach().cpu().numpy()
    )
    tasks = backend.tasks
    direction = build_direction(
        args.direction_file, tasks, baseline_original, norm_stats
    )
    cosine_task_indices = [
        index
        for index, mode in enumerate(direction.modes)
        if str(mode) in cosine_modes
    ]
    if not cosine_task_indices:
        raise ValueError(
            "No regression tasks match --cosine-modes in the direction file"
        )

    snp_info = variant_info.get("snp", {})
    ids = [str(value) for value in snp_info.get("ids", [])]
    refs = [str(value) for value in snp_info.get("refs", [])]
    alts = [str(value) for value in snp_info.get("alts", [])]
    chroms = [str(value) for value in snp_info.get("chroms", [])]
    positions = [int(value) for value in snp_info.get("positions", [])]
    if not (len(ids) == len(refs) == len(alts) == len(chroms) == len(positions)):
        raise ValueError("Incomplete or inconsistent SNP metadata in the VCF")
    duplicate_ids = pd.Series(ids)[pd.Series(ids).duplicated()].unique().tolist()
    if duplicate_ids:
        raise ValueError(
            "VCF/model marker IDs are not unique: "
            + ", ".join(duplicate_ids[:5])
        )
    if backend.variant_ids is not None and ids != backend.variant_ids:
        raise ValueError(
            "Input VCF marker IDs/order do not match the fold checkpoints"
        )
    id_to_index = {identifier: index for index, identifier in enumerate(ids)}
    coordinate_to_indices: dict[tuple[str, int], list[int]] = {}
    for marker_index, (chrom, position) in enumerate(zip(chroms, positions)):
        coordinate_to_indices.setdefault(
            marker_coordinate(chrom, position), []
        ).append(marker_index)
    qtn_resolution = {
        qtn_id: resolve_qtn_marker(
            qtn_id, id_to_index, coordinate_to_indices
        )
        for qtn_id in qtn_ids
    }
    by_chromosome = chromosome_index(chroms, positions)
    encoding_dim = get_encoding_dim(variant_tensors)
    snp_tensor = (
        variant_tensors["snp"] if is_multi_branch else variant_tensors
    )

    flips: dict[int, torch.Tensor] = {}
    flip_metadata: dict[int, tuple[str, str]] = {}
    for marker_index in range(len(ids)):
        flip = opposite_homozygous_flip(
            snp_tensor[0, marker_index],
            encoding_dim,
            refs[marker_index],
            alts[marker_index],
        )
        if flip is not None:
            flips[marker_index] = flip.target_encoding
            flip_metadata[marker_index] = (flip.current_gt, flip.target_gt)

    window_bp = int(round(args.window_kb * 1000.0))
    qtn_windows: dict[str, list[int]] = {}
    needed_indices = set()
    for qtn_id in qtn_ids:
        qtn_index, _ = qtn_resolution[qtn_id]
        if qtn_index is None:
            continue
        if qtn_index in flips:
            needed_indices.add(qtn_index)
        candidates = [
            marker_index
            for marker_index in indices_in_window(
                by_chromosome,
                chroms[qtn_index],
                positions[qtn_index],
                window_bp,
            )
            if marker_index != qtn_index and marker_index in flips
        ]
        qtn_windows[qtn_id] = candidates
        needed_indices.update(candidates)

    print(
        f"[Proxy map] Predicting {len(needed_indices)} unique homozygous flips "
        f"for {len(qtn_ids)} QTN IDs"
    )
    flip_predictions = predict_single_flips(
        model,
        variant_tensors,
        {index: flips[index] for index in needed_indices},
        is_multi_branch,
        args.device,
        args.batch_size,
    )
    prediction_matrix = np.stack(
        [flip_predictions[index] for index in sorted(flip_predictions)]
    ) if flip_predictions else np.empty((0, len(tasks)))
    score_by_index = {}
    if len(prediction_matrix):
        scores = compute_selection_index_normalized(
            prediction_matrix,
            direction,
            baseline_norm.reshape(1, -1),
        )
        score_by_index = dict(zip(sorted(flip_predictions), scores))
    baseline_score = float(
        compute_selection_index_normalized(
            baseline_norm.reshape(1, -1),
            direction,
            baseline_norm.reshape(1, -1),
        )[0]
    )

    rows = []
    mappings = {}
    for qtn_id in qtn_ids:
        qtn_index, qtn_match = qtn_resolution[qtn_id]
        base_row = {
            "qtn_id": qtn_id,
            "qtn_marker_id": "",
            "qtn_match": "",
            "qtn_chr": "",
            "qtn_pos": "",
            "qtn_current_gt": "",
            "qtn_target_gt": "",
            "delta_si_qtn": np.nan,
            "proxy_id": "",
            "proxy_chr": "",
            "proxy_pos": "",
            "distance_bp": np.nan,
            "proxy_current_gt": "",
            "proxy_target_gt": "",
            "delta_si_proxy": np.nan,
            "cosine": np.nan,
            "effect_ratio": np.nan,
            "proxy_is_self": False,
            "status": "",
        }
        if qtn_index is None:
            base_row["qtn_match"] = qtn_match
            base_row["status"] = f"no_proxy:{qtn_match}"
            rows.append(base_row)
            mappings[qtn_id] = dict(base_row)
            continue
        base_row.update(
            {
                "qtn_marker_id": ids[qtn_index],
                "qtn_match": qtn_match,
                "qtn_chr": chroms[qtn_index],
                "qtn_pos": positions[qtn_index],
            }
        )
        if qtn_index not in flips:
            base_row["status"] = "no_proxy:qtn_not_homozygous_biallelic"
            rows.append(base_row)
            mappings[qtn_id] = dict(base_row)
            continue

        qtn_current_gt, qtn_target_gt = flip_metadata[qtn_index]
        qtn_prediction = flip_predictions[qtn_index]
        qtn_delta_vector = qtn_prediction - baseline_norm
        qtn_effect_vector = qtn_delta_vector[cosine_task_indices]
        qtn_delta_si = float(score_by_index[qtn_index] - baseline_score)
        base_row.update(
            {
                "qtn_current_gt": qtn_current_gt,
                "qtn_target_gt": qtn_target_gt,
                "delta_si_qtn": qtn_delta_si,
            }
        )

        candidates = []
        for marker_index in qtn_windows.get(qtn_id, []):
            current_gt, target_gt = flip_metadata[marker_index]
            delta_vector = flip_predictions[marker_index] - baseline_norm
            candidates.append(
                ProxyCandidate(
                    snp_id=ids[marker_index],
                    index=marker_index,
                    chrom=chroms[marker_index],
                    position=positions[marker_index],
                    distance_bp=abs(
                        positions[marker_index] - positions[qtn_index]
                    ),
                    current_gt=current_gt,
                    target_gt=target_gt,
                    delta_si=float(score_by_index[marker_index] - baseline_score),
                    effect_vector=delta_vector[cosine_task_indices],
                )
            )
        selected, cosine, status = select_proxy_candidate(
            qtn_delta_si,
            qtn_effect_vector,
            candidates,
            args.min_cosine,
            args.si_epsilon,
        )
        base_row["status"] = status
        mapping_payload = dict(base_row)
        mapping_payload["qtn_trait_effect_z"] = {
            tasks[index]: float(qtn_delta_vector[index])
            for index in cosine_task_indices
        }
        if selected is not None:
            proxy_delta_vector = (
                flip_predictions[selected.index] - baseline_norm
            )
            ratio = (
                abs(selected.delta_si) / abs(qtn_delta_si)
                if abs(qtn_delta_si) > args.si_epsilon
                else np.nan
            )
            base_row.update(
                {
                    "proxy_id": selected.snp_id,
                    "proxy_chr": selected.chrom,
                    "proxy_pos": selected.position,
                    "distance_bp": selected.distance_bp,
                    "proxy_current_gt": selected.current_gt,
                    "proxy_target_gt": selected.target_gt,
                    "delta_si_proxy": selected.delta_si,
                    "cosine": cosine,
                    "effect_ratio": ratio,
                    "proxy_is_self": False,
                }
            )
            mapping_payload = dict(base_row)
            mapping_payload["qtn_trait_effect_z"] = {
                tasks[index]: float(qtn_delta_vector[index])
                for index in cosine_task_indices
            }
            mapping_payload["proxy_trait_effect_z"] = {
                tasks[index]: float(proxy_delta_vector[index])
                for index in cosine_task_indices
            }
        else:
            base_row.update(
                {
                    "proxy_id": ids[qtn_index],
                    "proxy_chr": chroms[qtn_index],
                    "proxy_pos": positions[qtn_index],
                    "distance_bp": 0,
                    "proxy_current_gt": qtn_current_gt,
                    "proxy_target_gt": qtn_target_gt,
                    "delta_si_proxy": qtn_delta_si,
                    "cosine": 1.0,
                    "effect_ratio": 1.0,
                    "proxy_is_self": True,
                }
            )
            mapping_payload = dict(base_row)
            mapping_payload["qtn_trait_effect_z"] = {
                tasks[index]: float(qtn_delta_vector[index])
                for index in cosine_task_indices
            }
            mapping_payload["proxy_trait_effect_z"] = mapping_payload[
                "qtn_trait_effect_z"
            ]
        rows.append(base_row)
        mappings[qtn_id] = mapping_payload

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_dir / "qtn_proxy_map.tsv",
        sep="\t",
        index=False,
        float_format="%.8g",
    )
    selected_sites = sorted(
        {
            edit["site_id"]
            for row in rows
            if (edit := evolution_edit(row)) is not None
        }
    )
    with (output_dir / "proxy_sites.list").open("w", encoding="utf-8") as handle:
        for identifier in selected_sites:
            handle.write(identifier + "\n")

    metadata = {
        "schema_version": 1,
        "parent_sample": sample_ids[0],
        "parent_vcf": file_signature(args.vcf),
        "qtn_list": file_signature(args.qtn_list),
        "direction_file": file_signature(args.direction_file),
        "direction_sha256": file_sha256(args.direction_file),
        "model": backend.metadata(),
        "window_kb": args.window_kb,
        "min_cosine": args.min_cosine,
        "si_epsilon": args.si_epsilon,
        "cosine_modes": sorted(cosine_modes),
        "cosine_tasks": [tasks[index] for index in cosine_task_indices],
        "n_qtn_requested": len(qtn_ids),
        "n_mapped": int((frame["status"] == "mapped").sum()),
        "n_self_proxy": int(frame["proxy_is_self"].sum()),
        "n_unique_proxy_sites": len(selected_sites),
    }
    payload = json_safe({"metadata": metadata, "mappings": mappings})
    with (output_dir / "qtn_proxy_map.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        f"[Proxy map] mapped {metadata['n_mapped']}/{len(qtn_ids)} QTN IDs "
        f"to external proxies; {metadata['n_self_proxy']} QTNs remain their "
        f"own sites; {len(selected_sites)} unique evolution sites"
    )
    print(f"[Proxy map] {output_dir / 'qtn_proxy_map.tsv'}")
    print(f"[Proxy map] {output_dir / 'qtn_proxy_map.json'}")
    return payload


def main() -> None:
    run_mapping(parse_args())


if __name__ == "__main__":
    main()
