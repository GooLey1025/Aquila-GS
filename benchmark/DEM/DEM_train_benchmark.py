#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Leakage-safe nested-CV training for multi-output DEM benchmarks."""

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
BENCHMARK_MODEL_ROOT = DEM_SOURCE_ROOT / "biodem" / "dem"
BENCHMARK_DATA_ROOT = DEM_SOURCE_ROOT / "biodem" / "utils"
for import_path in (
    str(SOURCE_ROOT),
    str(BENCHMARK_MODEL_ROOT),
    str(BENCHMARK_DATA_ROOT),
):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

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
    load_complete_case_split,
    load_metadata,
    load_target_tensors,
    regression_trait_indices,
    select_split_features,
)
from model_benchmark import DEMBenchmark, predict_dem, train_dem


def positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI value."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def expand_gpu_workers(gpu_ids: Sequence[int], jobs_per_gpu: int) -> list[int]:
    """Create one scheduler slot per concurrent job allowed on each GPU."""

    if jobs_per_gpu < 1:
        raise ValueError("--jobs-per-gpu must be at least 1")
    return [
        gpu_id
        for gpu_id in gpu_ids
        for _ in range(jobs_per_gpu)
    ]


@dataclass(frozen=True)
class OuterFoldJob:
    """One independently tuned multi-output DEM outer fold."""

    job_id: int
    outer_fold: int


@dataclass(frozen=True)
class OuterFoldContext:
    """Spawn-safe inputs shared by DEM outer-fold workers."""

    data_directory: str
    output_directory: str
    metadata: dict[str, Any]
    target_mask: torch.Tensor
    raw_targets: torch.Tensor
    trait_indices: tuple[int, ...]
    trait_names: tuple[str, ...]
    inner_count: int
    base_config: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    seed: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe nested CV for multi-output DEM."
    )
    parser.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "benchmark" / "test_v2"),
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIRECTORY / "configs" / "DEM_nested_cv.yaml"),
    )
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--outer-folds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=None,
        help="GPU IDs to use; omit to use all detected GPUs, or pass no IDs for CPU.",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=positive_int,
        default=1,
        help="Maximum concurrent outer-fold jobs per GPU (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-inner-folds", type=int, default=None)
    return parser.parse_args(argv)


def _write_json(path: Path, values: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, allow_nan=True)
        handle.write("\n")


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


def select_candidate(candidates: Sequence[CandidateResult]) -> CandidateResult:
    """Select maximum finite objective with stable candidate-ID tie break."""
    finite = [candidate for candidate in candidates if math.isfinite(candidate.objective)]
    if not finite:
        raise ValueError("All DEM grid candidates produced non-finite objectives")
    return max(finite, key=lambda candidate: (candidate.objective, -candidate.candidate_id))


def aggregate_outer_metrics(
    fold_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate scalar and per-trait metrics across outer folds."""
    if not fold_metrics:
        raise ValueError("No outer-fold metrics were supplied")
    scalar_names = ("avg_pearson", "avg_r2", "avg_rmse", "avg_mae")
    aggregate: dict[str, Any] = {}
    for scale in ("normalized", "original"):
        scale_metrics = [values[scale] for values in fold_metrics]
        summary = {}
        for name in scalar_names:
            values = np.asarray([item[name] for item in scale_metrics], dtype=float)
            finite = values[np.isfinite(values)]
            summary[name] = {
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "std": float(finite.std(ddof=0)) if finite.size else float("nan"),
                "values": values.tolist(),
            }
        trait_names = tuple(scale_metrics[0]["per_trait"])
        per_trait = {}
        for trait in trait_names:
            per_trait[trait] = {}
            for metric in ("pearson", "r2", "rmse", "mae"):
                values = np.asarray(
                    [item["per_trait"][trait][metric] for item in scale_metrics],
                    dtype=float,
                )
                finite = values[np.isfinite(values)]
                per_trait[trait][metric] = {
                    "mean": float(finite.mean()) if finite.size else float("nan"),
                    "std": float(finite.std(ddof=0)) if finite.size else float("nan"),
                    "values": values.tolist(),
                }
        aggregate[scale] = {**summary, "per_trait": per_trait}
    aggregate["primary"] = {
        "metric": "normalized.avg_pearson",
        "outer_fold_mean": aggregate["normalized"]["avg_pearson"]["mean"],
        "outer_fold_std": aggregate["normalized"]["avg_pearson"]["std"],
    }
    return aggregate


def _selected_variant_payload(selected: Any) -> list[dict[str, Any]]:
    rows = []
    for rank, (index, variant, importance) in enumerate(
        zip(
            selected.selected_indices,
            selected.selected_variants,
            selected.importances,
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
    return rows


def _load_pair(
    data_directory: Path,
    metadata: Mapping[str, Any],
    target_mask: torch.Tensor,
    raw_targets: torch.Tensor,
    trait_indices: Sequence[int],
    outer_fold: int,
    inner_fold: int | None,
    config: Mapping[str, Any],
) -> Any:
    held_role = "test" if inner_fold is None else "valid"
    data_config = config["data"]
    train = load_complete_case_split(
        data_directory,
        metadata,
        target_mask,
        raw_targets,
        trait_indices,
        outer_fold,
        inner_fold,
        "train",
        float(data_config["missing_genotype_value"]),
        int(data_config["min_complete_cases"]),
    )
    held = load_complete_case_split(
        data_directory,
        metadata,
        target_mask,
        raw_targets,
        trait_indices,
        outer_fold,
        inner_fold,
        held_role,
        float(data_config["missing_genotype_value"]),
        int(data_config["min_complete_cases"]),
    )
    return select_split_features(
        train,
        held,
        int(data_config["selected_snps"]),
        int(data_config["rf_estimators"]),
        tuple(data_config["rf_random_states"]),
        int(data_config["rf_n_jobs"]),
    )


def _load_model(
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    input_dim: int,
    output_dim: int,
    device: torch.device,
) -> DEMBenchmark:
    model_config = config["model"]
    model = DEMBenchmark(
        [input_dim],
        output_dim,
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


def run_outer_fold(
    data_directory: Path,
    output_directory: Path,
    metadata: Mapping[str, Any],
    target_mask: torch.Tensor,
    raw_targets: torch.Tensor,
    trait_indices: Sequence[int],
    trait_names: Sequence[str],
    outer_fold: int,
    inner_count: int,
    base_config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    fold_start = time.time()
    fold_path = output_directory / f"fold_{outer_fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    inner_results: dict[int, list[InnerFoldResult]] = {
        index: [] for index in range(len(candidates))
    }
    split_audit = []
    for inner_fold in range(inner_count):
        selected = _load_pair(
            data_directory,
            metadata,
            target_mask,
            raw_targets,
            trait_indices,
            outer_fold,
            inner_fold,
            base_config,
        )
        split_audit.append(
            {
                "inner_fold": inner_fold,
                "train_complete_cases": len(selected.train.sample_ids),
                "train_discarded": list(selected.train.discarded_sample_ids),
                "valid_complete_cases": len(selected.held_out.sample_ids),
                "valid_discarded": list(selected.held_out.discarded_sample_ids),
                "selected_snps": len(selected.selected_indices),
            }
        )
        for candidate_id, parameters in enumerate(candidates):
            config = merge_config(base_config, parameters)
            result = train_dem(
                selected.train.genotypes,
                selected.train.targets,
                selected.held_out.genotypes,
                selected.held_out.targets,
                config,
                device,
                derive_seed(seed, outer_fold, candidate_id, inner_fold),
                evaluate_regression,
                trait_names,
            )
            inner_results[candidate_id].append(
                InnerFoldResult(
                    inner_fold,
                    result.best_metric,
                    result.best_epoch,
                    result.metrics,
                )
            )
    candidate_results = []
    for candidate_id, parameters in enumerate(candidates):
        results = tuple(inner_results[candidate_id])
        values = np.asarray([result.metric for result in results], dtype=float)
        objective = float(values.mean()) if np.isfinite(values).all() else float("nan")
        candidate_results.append(
            CandidateResult(candidate_id, dict(parameters), objective, results)
        )
    best = select_candidate(candidate_results)
    best_config = merge_config(base_config, best.parameters)
    final_epoch = half_up_median_epoch(best.best_epochs)
    selected = _load_pair(
        data_directory,
        metadata,
        target_mask,
        raw_targets,
        trait_indices,
        outer_fold,
        None,
        best_config,
    )
    final_fit = train_dem(
        selected.train.genotypes,
        selected.train.targets,
        None,
        None,
        best_config,
        device,
        derive_seed(seed, outer_fold, best.candidate_id, 999),
        evaluate_regression,
        trait_names,
        fixed_epochs=final_epoch,
    )
    model = _load_model(
        final_fit.state_dict,
        best_config,
        selected.train.genotypes.shape[1],
        len(trait_names),
        device,
    )
    predictions, test_loss = predict_dem(
        model,
        selected.held_out.genotypes,
        selected.held_out.targets,
        int(best_config["train"]["batch_size"]),
        device,
    )
    normalized = evaluate_regression(
        predictions,
        selected.held_out.targets,
        np.ones_like(selected.held_out.targets, dtype=bool),
        trait_names,
    )
    preprocessor_path = (
        data_directory
        / "cv"
        / f"outer_fold_{outer_fold}"
        / "final"
        / "preprocessing.json"
    )
    from aquila.data.preprocessing import PerTraitPreprocessor

    processor = PerTraitPreprocessor.load_json(preprocessor_path)
    full_predictions = np.zeros(
        (len(predictions), len(metadata["trait_names"])), dtype=np.float32
    )
    full_mask = np.zeros_like(full_predictions, dtype=bool)
    full_predictions[:, trait_indices] = predictions
    full_mask[:, trait_indices] = True
    inverted = processor.inverse(full_predictions, full_mask)
    original_predictions = np.asarray(inverted)[:, trait_indices]
    original = evaluate_regression(
        original_predictions,
        selected.held_out.raw_targets,
        np.ones_like(selected.held_out.raw_targets, dtype=bool),
        trait_names,
    )
    torch.save(
        {
            "model_state_dict": final_fit.state_dict,
            "config": best_config,
            "outer_fold": outer_fold,
            "final_epoch": final_epoch,
            "trait_names": list(trait_names),
            "selected_indices": selected.selected_indices,
            "selected_variants": selected.selected_variants,
        },
        fold_path / "best_model.pt",
    )
    _write_json(
        fold_path / "hpo_results.json",
        {
            "method": "grid",
            "direction": "maximize",
            "best_candidate_id": best.candidate_id,
            "best_parameters": dict(best.parameters),
            "best_valid_pearson_mean": best.objective,
            "final_epoch": final_epoch,
            "candidates": [_candidate_payload(item) for item in candidate_results],
        },
    )
    metrics_payload = {
        "normalized": normalized.metrics,
        "original": original.metrics,
        "test_loss": test_loss,
    }
    _write_json(fold_path / "metrics.json", metrics_payload)
    _write_json(
        fold_path / "training_history.json",
        {"final": list(final_fit.history)},
    )
    with (fold_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)
    shutil.copy2(preprocessor_path, fold_path / "preprocessing.json")
    _write_json(
        fold_path / "sample_audit.json",
        {
            "outer_train_complete_cases": len(selected.train.sample_ids),
            "outer_train_discarded": list(selected.train.discarded_sample_ids),
            "outer_test_complete_cases": len(selected.held_out.sample_ids),
            "outer_test_discarded": list(selected.held_out.discarded_sample_ids),
            "inner_folds": split_audit,
        },
    )
    _write_json(
        fold_path / "selected_variants.json",
        {"variants": _selected_variant_payload(selected)},
    )
    with (fold_path / "predictions_original_scale.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["SampleID", *[f"Predicted_{name}" for name in trait_names], *[f"Observed_{name}" for name in trait_names]])
        for sample_id, predicted, observed in zip(
            selected.held_out.sample_ids,
            original_predictions,
            selected.held_out.raw_targets,
        ):
            writer.writerow([sample_id, *predicted.tolist(), *observed.tolist()])
    return {
        "outer_fold": outer_fold,
        "best_candidate_id": best.candidate_id,
        "best_parameters": dict(best.parameters),
        "best_valid_pearson_mean": best.objective,
        "final_epoch": final_epoch,
        "metrics": metrics_payload,
        "elapsed_seconds": time.time() - fold_start,
    }


def _run_outer_fold_job(
    job: OuterFoldJob,
    device_name: str,
    worker_context: OuterFoldContext,
) -> dict[str, Any]:
    """Run one multi-output DEM outer fold on its assigned device."""

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return run_outer_fold(
        Path(worker_context.data_directory),
        Path(worker_context.output_directory),
        worker_context.metadata,
        worker_context.target_mask,
        worker_context.raw_targets,
        worker_context.trait_indices,
        worker_context.trait_names,
        job.outer_fold,
        worker_context.inner_count,
        worker_context.base_config,
        worker_context.candidates,
        device,
        worker_context.seed,
    )


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
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    metadata = load_metadata(data_directory)
    raw_targets, target_mask = load_target_tensors(data_directory)
    trait_indices = regression_trait_indices(metadata)
    trait_names = [metadata["trait_names"][index] for index in trait_indices]
    outer_count = int(metadata["outer_folds"])
    inner_count = int(metadata["inner_folds"])
    outer_folds = args.outer_folds or list(range(outer_count))
    if any(fold < 0 or fold >= outer_count for fold in outer_folds):
        raise ValueError(f"Outer folds must be in 0..{outer_count - 1}")
    if args.max_inner_folds is not None:
        inner_count = min(inner_count, args.max_inner_folds)
    candidates = generate_grid_candidates(config["hpo"]["parameters"])
    if len(candidates) != 32:
        raise ValueError(f"DEM grid must contain 32 candidates, got {len(candidates)}")
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    worker_gpu_ids = expand_gpu_workers(gpu_ids, args.jobs_per_gpu)
    jobs = [
        OuterFoldJob(job_id=position, outer_fold=outer_fold)
        for position, outer_fold in enumerate(outer_folds)
    ]
    worker_context = OuterFoldContext(
        data_directory=str(data_directory),
        output_directory=str(output_directory),
        metadata=metadata,
        target_mask=target_mask,
        raw_targets=raw_targets,
        trait_indices=tuple(trait_indices),
        trait_names=tuple(trait_names),
        inner_count=inner_count,
        base_config=config,
        candidates=tuple(candidates),
        seed=args.seed,
    )
    devices = (
        [f"cuda:{gpu_id}" for gpu_id in worker_gpu_ids]
        if worker_gpu_ids
        else ["cpu"]
    )
    print(
        f"[INFO] DEM {len(outer_folds)} independent multi-output outer folds "
        f"across {devices}; {args.jobs_per_gpu} concurrent job(s) per GPU"
    )
    work_results = execute_gpu_jobs(
        jobs,
        _run_outer_fold_job,
        worker_gpu_ids,
        worker_args=(worker_context,),
        raise_on_error=True,
    )
    results = [result.value for result in work_results]
    fold_metrics = [result["metrics"] for result in results]
    _write_json(
        output_directory / "summary.json",
        {
            "data_dir": str(data_directory),
            "config": str(Path(args.config).resolve()),
            "trait_names": trait_names,
            "outer_folds": outer_folds,
            "results": results,
            "outer_fold_summary": aggregate_outer_metrics(fold_metrics),
        },
    )


if __name__ == "__main__":
    main()
