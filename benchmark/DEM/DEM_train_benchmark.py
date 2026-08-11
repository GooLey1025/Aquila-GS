#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Leakage-safe single-trait nested-CV training for DEM-SNP and DEM-Vars."""

# Migrated from: https://github.com/cma2015/DEM

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
DEM_SOURCE_ROOT = SCRIPT_DIRECTORY / "src"
for import_path in (
    SOURCE_ROOT,
    DEM_SOURCE_ROOT / "biodem" / "dem",
    DEM_SOURCE_ROOT / "biodem" / "utils",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.training.distributed import derive_seed
from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import (
    CandidateResult,
    InnerFoldResult,
    generate_grid_candidates,
    half_up_median_epoch,
    merge_config,
)
from data_ncv_benchmark import (
    CHANNELS_PER_MODALITY,
    ENCODING_NAMES,
    MODALITY_ORDER,
    SelectedSplits,
    load_metadata,
    load_split,
    load_target_tensors,
    resolve_traits,
    select_split_features,
)
from model_benchmark import DEMBenchmark, predict_dem, train_dem


@dataclass(frozen=True)
class OuterFoldJob:
    """One independently tuned trait and outer-fold job."""

    job_id: int
    trait_index: int
    trait_name: str
    outer_fold: int


@dataclass(frozen=True)
class WorkerContext:
    """Spawn-safe inputs shared by DEM workers."""

    data_directory: str
    output_directory: str
    metadata: dict[str, Any]
    target_mask: torch.Tensor
    raw_targets: torch.Tensor
    inner_count: int
    base_config: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    modalities: tuple[str, ...]
    model_name: str
    seed: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def expand_gpu_workers(gpu_ids: Sequence[int], jobs_per_gpu: int) -> list[int]:
    if jobs_per_gpu < 1:
        raise ValueError("--jobs-per-gpu must be at least 1")
    return [gpu_id for gpu_id in gpu_ids for _ in range(jobs_per_gpu)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-trait nested CV for DEM-SNP or DEM-Vars."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIRECTORY / "configs" / "DEM-SNP_nested_cv.yaml"),
    )
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Regression trait names; default: all regression traits.",
    )
    parser.add_argument("--outer-folds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=None,
        help="GPU IDs; omit for detection or pass no IDs for CPU.",
    )
    parser.add_argument("--jobs-per-gpu", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-candidates", type=positive_int, default=None)
    parser.add_argument("--max-inner-folds", type=positive_int, default=None)
    return parser.parse_args(argv)


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, allow_nan=True)
        handle.write("\n")


def _safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Trait name cannot form an output directory: {value!r}")
    return safe


def validate_config(config: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    model_name = str(config.get("benchmark", {}).get("model_name", ""))
    mode = str(config.get("benchmark", {}).get("mode", "")).lower()
    modalities = tuple(
        str(value).lower() for value in config.get("data", {}).get("modalities", ())
    )
    contracts = {
        "dem-snp": ("DEM-SNP", ("snp",)),
        "dem-vars": ("DEM-Vars", MODALITY_ORDER),
    }
    if mode not in contracts:
        raise ValueError("benchmark.mode must be dem-snp or dem-vars")
    expected_name, expected_modalities = contracts[mode]
    if model_name != expected_name or modalities != expected_modalities:
        raise ValueError(
            f"{expected_name} requires model_name={expected_name!r} and "
            f"modalities={expected_modalities}, got {model_name!r} and {modalities}"
        )
    return model_name, modalities


def select_candidate(candidates: Sequence[CandidateResult]) -> CandidateResult:
    finite = [candidate for candidate in candidates if math.isfinite(candidate.objective)]
    if not finite:
        raise ValueError("All DEM grid candidates produced non-finite objectives")
    return max(finite, key=lambda candidate: (candidate.objective, -candidate.candidate_id))


def aggregate_outer_metrics(
    fold_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not fold_metrics:
        raise ValueError("No outer-fold metrics were supplied")
    aggregate: dict[str, Any] = {}
    for scale in ("normalized", "original"):
        scale_values = [item[scale] for item in fold_metrics]
        summary: dict[str, Any] = {}
        for metric in ("avg_pearson", "avg_r2", "avg_mse", "avg_rmse", "avg_mae"):
            values = np.asarray([item.get(metric, np.nan) for item in scale_values])
            finite = values[np.isfinite(values)]
            summary[metric] = {
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "std": float(finite.std(ddof=0)) if finite.size else float("nan"),
                "values": values.tolist(),
            }
        aggregate[scale] = summary
    aggregate["primary"] = {
        "metric": "normalized.avg_pearson",
        "outer_fold_mean": aggregate["normalized"]["avg_pearson"]["mean"],
        "outer_fold_std": aggregate["normalized"]["avg_pearson"]["std"],
    }
    return aggregate


def _candidate_payload(candidate: CandidateResult) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "parameters": dict(candidate.parameters),
        "objective": candidate.objective,
        "best_epochs": list(candidate.best_epochs),
        "final_epoch": candidate.final_epoch,
        "inner_results": [
            {
                "inner_fold": result.inner_fold,
                "metric": result.metric,
                "best_epoch": result.best_epoch,
                "metrics": dict(result.metrics),
            }
            for result in candidate.inner_results
        ],
    }


def _selected_variant_payload(
    selected: SelectedSplits,
) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for branch_index, name in enumerate(selected.modality_names):
        rows = []
        for rank, (index, variant, importance) in enumerate(
            zip(
                selected.selected_indices[branch_index],
                selected.selected_variants[branch_index],
                selected.importances[branch_index],
            ),
            start=1,
        ):
            chrom, position, identifier, reference, alternate = variant
            rows.append(
                {
                    "rank": rank,
                    "original_index": int(index),
                    "chrom": chrom,
                    "position": position,
                    "id": identifier,
                    "ref": reference,
                    "alt": alternate,
                    "importance": float(importance),
                }
            )
        payload[name] = rows
    return payload


def _split_audit(split: Any) -> dict[str, Any]:
    return {
        "retained_count": len(split.sample_ids),
        "retained_sample_ids": list(split.sample_ids),
        "discarded_count": len(split.discarded_sample_ids),
        "discarded_sample_ids": list(split.discarded_sample_ids),
    }


def _selection_metadata(selected: SelectedSplits, rf_enabled: bool) -> dict[str, Any]:
    branches = {}
    variants = _selected_variant_payload(selected)
    for index, name in enumerate(selected.modality_names):
        branches[name] = {
            "encoding": ENCODING_NAMES[name],
            "channels_per_marker": CHANNELS_PER_MODALITY[name],
            "selection_unit": f"{name.upper()}_marker",
            "marker_count": len(selected.selected_indices[index]),
            "input_dim": selected.train.modalities[index].shape[1],
            "variants": variants[name],
        }
    return {
        "rf_enabled": rf_enabled,
        "modality_order": list(selected.modality_names),
        "branches": branches,
    }


def _load_pair(
    context: WorkerContext,
    trait_index: int,
    outer_fold: int,
    inner_fold: int | None,
    config: Mapping[str, Any],
) -> SelectedSplits:
    role = "test" if inner_fold is None else "valid"
    data_config = config["data"]
    arguments = (
        context.data_directory,
        context.metadata,
        context.target_mask,
        context.raw_targets,
        trait_index,
        outer_fold,
        inner_fold,
    )
    train = load_split(
        *arguments,
        "train",
        int(data_config["min_observed"]),
        context.modalities,
    )
    held_out = load_split(
        *arguments,
        role,
        int(data_config["min_observed"]),
        context.modalities,
    )
    return select_split_features(
        train,
        held_out,
        bool(data_config.get("rf_enabled", True)),
        data_config["selected_markers"],
        int(data_config["rf_estimators"]),
        tuple(data_config["rf_random_states"]),
        int(data_config["rf_n_jobs"]),
    )


def _load_model(
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    input_dims: Sequence[int],
    device: torch.device,
) -> DEMBenchmark:
    model_config = config["model"]
    model = DEMBenchmark(
        input_dims,
        1,
        int(model_config["n_heads"]),
        int(model_config["n_encoders"]),
        int(model_config["hidden_dim"]),
        float(model_config["dropout"]),
        model_config.get("single_hidden", (512, 128)),
        model_config.get("conc_hidden", (1536, 512)),
        model_config.get("integrated_hidden", (1024, 256, 128)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _inverse_trait(
    predictions: np.ndarray,
    processor: PerTraitPreprocessor,
    trait_index: int,
    trait_count: int,
) -> np.ndarray:
    full = np.zeros((len(predictions), trait_count), dtype=np.float32)
    mask = np.zeros_like(full, dtype=bool)
    full[:, trait_index] = predictions[:, 0]
    mask[:, trait_index] = True
    return np.asarray(processor.inverse(full, mask))[:, [trait_index]]


def run_outer_fold(
    job: OuterFoldJob,
    context: WorkerContext,
    device: torch.device,
) -> dict[str, Any]:
    start = time.time()
    trait_path = Path(context.output_directory) / _safe_name(job.trait_name)
    fold_path = trait_path / f"fold_{job.outer_fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    inner_results: dict[int, list[InnerFoldResult]] = {
        index: [] for index in range(len(context.candidates))
    }
    inner_audit = []
    for inner_fold in range(context.inner_count):
        selected = _load_pair(
            context, job.trait_index, job.outer_fold, inner_fold, context.base_config
        )
        inner_audit.append(
            {
                "inner_fold": inner_fold,
                "train": _split_audit(selected.train),
                "valid": _split_audit(selected.held_out),
                "selection": _selection_metadata(
                    selected, bool(context.base_config["data"].get("rf_enabled", True))
                ),
            }
        )
        for candidate_id, parameters in enumerate(context.candidates):
            candidate_config = merge_config(context.base_config, parameters)
            fit = train_dem(
                selected.train.modalities,
                selected.train.targets,
                selected.held_out.modalities,
                selected.held_out.targets,
                candidate_config,
                device,
                derive_seed(
                    context.seed,
                    job.trait_index,
                    job.outer_fold,
                    candidate_id,
                    inner_fold,
                ),
                evaluate_regression,
                job.trait_name,
            )
            inner_results[candidate_id].append(
                InnerFoldResult(
                    inner_fold, fit.best_metric, fit.best_epoch, fit.metrics
                )
            )
    candidates = []
    for candidate_id, parameters in enumerate(context.candidates):
        results = tuple(inner_results[candidate_id])
        values = np.asarray([result.metric for result in results], dtype=float)
        objective = float(values.mean()) if np.isfinite(values).all() else float("nan")
        candidates.append(
            CandidateResult(candidate_id, dict(parameters), objective, results)
        )
    best = select_candidate(candidates)
    best_config = merge_config(context.base_config, best.parameters)
    final_epoch = half_up_median_epoch(best.best_epochs)
    selected = _load_pair(
        context, job.trait_index, job.outer_fold, None, best_config
    )
    final_fit = train_dem(
        selected.train.modalities,
        selected.train.targets,
        None,
        None,
        best_config,
        device,
        derive_seed(
            context.seed, job.trait_index, job.outer_fold, best.candidate_id, 999
        ),
        evaluate_regression,
        job.trait_name,
        fixed_epochs=final_epoch,
    )
    input_dims = [values.shape[1] for values in selected.train.modalities]
    model = _load_model(final_fit.state_dict, best_config, input_dims, device)
    predictions, test_loss = predict_dem(
        model,
        selected.held_out.modalities,
        selected.held_out.targets,
        int(best_config["train"]["batch_size"]),
        device,
    )
    mask = np.ones_like(selected.held_out.targets, dtype=bool)
    normalized = evaluate_regression(
        predictions, selected.held_out.targets, mask, [job.trait_name]
    )
    preprocessor_path = (
        Path(context.data_directory)
        / "cv"
        / f"outer_fold_{job.outer_fold}"
        / "final"
        / "preprocessing.json"
    )
    processor = PerTraitPreprocessor.load_json(preprocessor_path)
    original_predictions = _inverse_trait(
        predictions,
        processor,
        job.trait_index,
        len(context.metadata["trait_names"]),
    )
    original = evaluate_regression(
        original_predictions,
        selected.held_out.raw_targets,
        mask,
        [job.trait_name],
    )
    selection = _selection_metadata(
        selected, bool(best_config["data"].get("rf_enabled", True))
    )
    torch.save(
        {
            "model_state_dict": final_fit.state_dict,
            "model_name": context.model_name,
            "config": best_config,
            "trait": job.trait_name,
            "trait_index": job.trait_index,
            "outer_fold": job.outer_fold,
            "final_epoch": final_epoch,
            "input_dims": input_dims,
            "output_dim": 1,
            "selection": selection,
        },
        fold_path / "best_model.pt",
    )
    _write_json(
        fold_path / "hpo_results.json",
        {
            "method": "grid",
            "direction": "maximize",
            "metric": "normalized Pearson correlation",
            "best_candidate_id": best.candidate_id,
            "best_parameters": dict(best.parameters),
            "best_valid_pearson_mean": best.objective,
            "final_epoch": final_epoch,
            "candidates": [_candidate_payload(candidate) for candidate in candidates],
        },
    )
    metrics = {
        "normalized": normalized.metrics,
        "original": original.metrics,
        "test_mse_loss": test_loss,
    }
    _write_json(fold_path / "metrics.json", metrics)
    _write_json(fold_path / "training_history.json", {"final": list(final_fit.history)})
    _write_json(
        fold_path / "sample_audit.json",
        {
            "trait": job.trait_name,
            "outer_train": _split_audit(selected.train),
            "outer_test": _split_audit(selected.held_out),
            "inner_folds": inner_audit,
        },
    )
    _write_json(fold_path / "selected_variants.json", selection)
    with (fold_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)
    shutil.copy2(preprocessor_path, fold_path / "preprocessing.json")
    with (fold_path / "predictions_original_scale.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["SampleID", "Trait", "Predicted", "Observed"])
        for sample_id, predicted, observed in zip(
            selected.held_out.sample_ids,
            original_predictions[:, 0],
            selected.held_out.raw_targets[:, 0],
        ):
            writer.writerow(
                [sample_id, job.trait_name, float(predicted), float(observed)]
            )
    return {
        "trait": job.trait_name,
        "trait_index": job.trait_index,
        "outer_fold": job.outer_fold,
        "best_candidate_id": best.candidate_id,
        "best_parameters": dict(best.parameters),
        "best_valid_pearson_mean": best.objective,
        "final_epoch": final_epoch,
        "metrics": metrics,
        "elapsed_seconds": time.time() - start,
    }


def _run_outer_fold_job(
    job: OuterFoldJob,
    device_name: str,
    context: WorkerContext,
) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return run_outer_fold(job, context, device)


def main() -> None:
    args = parse_args()
    from aquila.training.distributed import detect_gpu_ids, execute_gpu_jobs

    data_directory = Path(args.data_dir).resolve()
    output_directory = Path(args.output_dir).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_directory}")
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_name, modalities = validate_config(config)
    metadata = load_metadata(data_directory)
    raw_targets, target_mask = load_target_tensors(data_directory)
    traits = resolve_traits(metadata, args.traits)
    outer_count = int(metadata["outer_folds"])
    inner_count = int(metadata["inner_folds"])
    outer_folds = args.outer_folds or list(range(outer_count))
    if any(fold < 0 or fold >= outer_count for fold in outer_folds):
        raise ValueError(f"Outer folds must be in 0..{outer_count - 1}")
    if args.max_inner_folds is not None:
        inner_count = min(inner_count, args.max_inner_folds)
    candidates = generate_grid_candidates(config["hpo"]["parameters"])
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ValueError("DEM configuration generated no HPO candidates")
    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    worker_gpu_ids = expand_gpu_workers(gpu_ids, args.jobs_per_gpu)
    jobs = [
        OuterFoldJob(position, trait_index, trait_name, outer_fold)
        for position, (trait_index, trait_name, outer_fold) in enumerate(
            (trait_index, trait_name, outer_fold)
            for trait_index, trait_name in traits
            for outer_fold in outer_folds
        )
    ]
    context = WorkerContext(
        str(data_directory),
        str(output_directory),
        metadata,
        target_mask,
        raw_targets,
        inner_count,
        config,
        tuple(candidates),
        modalities,
        model_name,
        args.seed,
    )
    devices = (
        [f"cuda:{gpu_id}" for gpu_id in worker_gpu_ids]
        if worker_gpu_ids
        else ["cpu"]
    )
    print(
        f"[INFO] {model_name}: {len(jobs)} trait-by-fold jobs across {devices}; "
        f"{args.jobs_per_gpu} concurrent job(s) per GPU"
    )
    work_results = execute_gpu_jobs(
        jobs,
        _run_outer_fold_job,
        worker_gpu_ids,
        worker_args=(context,),
        raise_on_error=True,
    )
    results = [result.value for result in work_results]
    trait_results = {}
    for _, trait_name in traits:
        fold_results = [
            result for result in results if result["trait"] == trait_name
        ]
        trait_results[trait_name] = {
            "outer_folds": outer_folds,
            "results": fold_results,
            "outer_fold_summary": aggregate_outer_metrics(
                [result["metrics"] for result in fold_results]
            ),
        }
        _write_json(
            output_directory / _safe_name(trait_name) / "summary.json",
            trait_results[trait_name],
        )
    _write_json(
        output_directory / "summary.json",
        {
            "model_name": model_name,
            "mode": config["benchmark"]["mode"],
            "data_dir": str(data_directory),
            "config": str(config_path),
            "modalities": list(modalities),
            "traits": [name for _, name in traits],
            "outer_folds": outer_folds,
            "results": trait_results,
        },
    )


if __name__ == "__main__":
    main()
