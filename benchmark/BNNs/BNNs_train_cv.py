#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/GSBreeder/BNNs

"""Leakage-safe nested cross-validation runner for single-trait BNN models."""

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
BNN_DIRECTORY = SCRIPT_DIRECTORY
BENCHMARK_SOURCE = BNN_DIRECTORY / "src_benchmark"
PROJECT_ROOT = BNN_DIRECTORY.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_path in (str(SOURCE_ROOT), str(BENCHMARK_SOURCE)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from aquila.data import load_prepared_data
from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.training.distributed import (
    derive_seed,
    detect_gpu_ids,
    execute_gpu_jobs,
)
from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import (
    CandidateResult,
    InnerFoldResult,
    generate_grid_candidates,
    half_up_median_epoch,
    merge_config,
)
from bnn_data_benchmark import MarkerPipeline, PreparedPair, load_trait_split, prepare_pair
from bnn_model_benchmark import build_model, predict_bnn, train_bnn


@dataclass(frozen=True)
class TraitFoldJob:
    """One independently scheduled trait and outer-fold run."""

    job_id: int
    trait_name: str
    trait_index: int
    outer_fold: int


@dataclass(frozen=True)
class TraitFoldContext:
    """Spawn-safe inputs shared by BNN GPU workers."""

    data_directory: str
    output_directory: str
    base_config: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    inner_count: int
    seed: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe nested CV for single-trait Bayesian MLPs."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIRECTORY / "configs" / "BNNs_nested_cv.yaml"),
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help=(
            "Optional regression trait subset. Defaults to all regression "
            "traits stored in the prepared-data metadata."
        ),
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
        help="Maximum concurrent trait/fold jobs per GPU (default: 1).",
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
    """Select maximum finite mean validation Pearson with a stable tie break."""

    finite = [candidate for candidate in candidates if math.isfinite(candidate.objective)]
    if not finite:
        raise ValueError("All BNN grid candidates produced non-finite objectives")
    return max(finite, key=lambda candidate: (candidate.objective, -candidate.candidate_id))


def _metric_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std(ddof=0)) if finite.size else float("nan"),
        "values": array.tolist(),
    }


def aggregate_outer_metrics(
    results: Sequence[Mapping[str, Any]],
    trait_names: Sequence[str],
) -> dict[str, Any]:
    """Aggregate all five outer-test folds independently for every trait."""

    aggregated = {}
    for trait in trait_names:
        trait_results = [item for item in results if item["trait"] == trait]
        scales = {}
        for scale in ("normalized", "original"):
            scales[scale] = {
                metric: _metric_summary(
                    [item["metrics"][scale][f"avg_{metric}"] for item in trait_results]
                )
                for metric in ("pearson", "r2", "mse", "rmse", "mae")
            }
        aggregated[trait] = {
            **scales,
            "primary": {
                "metric": "normalized.avg_pearson",
                "outer_fold_mean": scales["normalized"]["pearson"]["mean"],
                "outer_fold_std": scales["normalized"]["pearson"]["std"],
            },
        }
    return aggregated


def _inverse_trait(
    values: np.ndarray,
    preprocessor: PerTraitPreprocessor,
    trait_index: int,
) -> np.ndarray:
    params = preprocessor.traits[trait_index]
    restored = np.asarray(values, dtype=np.float64) * params.std + params.mean
    if params.use_log1p:
        restored = np.expm1(restored) - params.log_shift
    return restored.astype(np.float32)


def _pipeline_payload(pipeline: MarkerPipeline) -> dict[str, Any]:
    return {
        "imputation_values": pipeline.imputation_values,
        "scale_min": pipeline.scale_min,
        "scale_range": pipeline.scale_range,
        "selected_indices": pipeline.selected_indices,
        "selected_variants": pipeline.selected_variants,
        "lasso_coefficients": pipeline.lasso_coefficients,
    }


def _selected_variant_payload(pair: PreparedPair) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank + 1,
            "source_index": int(index),
            "chrom": variant[0],
            "position": variant[1],
            "id": variant[2],
            "ref": variant[3],
            "alt": variant[4],
            "lasso_coefficient": float(coefficient),
        }
        for rank, (index, variant, coefficient) in enumerate(
            zip(
                pair.pipeline.selected_indices,
                pair.pipeline.selected_variants,
                pair.pipeline.lasso_coefficients,
            )
        )
    ]


def _load_pair(
    data_directory: Path,
    prepared: Any,
    raw_targets: torch.Tensor,
    trait_index: int,
    outer_fold: int,
    inner_fold: int | None,
    config: Mapping[str, Any],
    seed: int,
) -> PreparedPair:
    held_role = "test" if inner_fold is None else "valid"
    train = load_trait_split(
        data_directory,
        prepared.metadata,
        prepared.target_mask,
        raw_targets,
        trait_index,
        outer_fold,
        inner_fold,
        "train",
    )
    held_out = load_trait_split(
        data_directory,
        prepared.metadata,
        prepared.target_mask,
        raw_targets,
        trait_index,
        outer_fold,
        inner_fold,
        held_role,
    )
    return prepare_pair(
        train,
        held_out,
        float(config["data"]["lasso_alpha"]),
        int(config["data"]["max_features"]),
        seed,
    )


def run_outer_fold(
    data_directory: Path,
    output_directory: Path,
    prepared: Any,
    raw_targets: torch.Tensor,
    trait_name: str,
    trait_index: int,
    outer_fold: int,
    base_config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    inner_count: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    fold_start = time.time()
    fold_path = output_directory / trait_name / f"fold_{outer_fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    inner_results: dict[int, list[InnerFoldResult]] = {
        index: [] for index in range(len(candidates))
    }
    split_audit = []
    for inner_fold in range(inner_count):
        for candidate_id, parameters in enumerate(candidates):
            config = merge_config(base_config, parameters)
            candidate_seed = derive_seed(seed, outer_fold, candidate_id, inner_fold)
            pair = _load_pair(
                data_directory,
                prepared,
                raw_targets,
                trait_index,
                outer_fold,
                inner_fold,
                config,
                candidate_seed,
            )
            if candidate_id == 0:
                split_audit.append(
                    {
                        "inner_fold": inner_fold,
                        "train_observed": len(pair.train.sample_ids),
                        "train_discarded": list(pair.train.discarded_sample_ids),
                        "valid_observed": len(pair.held_out.sample_ids),
                        "valid_discarded": list(pair.held_out.discarded_sample_ids),
                    }
                )
            fit = train_bnn(
                pair.train_features,
                pair.train.targets,
                pair.held_out_features,
                pair.held_out.targets,
                config,
                device,
                candidate_seed,
                evaluate_regression,
                trait_name,
            )
            inner_results[candidate_id].append(
                InnerFoldResult(
                    inner_fold,
                    fit.best_metric,
                    fit.best_epoch,
                    fit.metrics,
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
    final_seed = derive_seed(seed, outer_fold, best.candidate_id, 999)
    pair = _load_pair(
        data_directory,
        prepared,
        raw_targets,
        trait_index,
        outer_fold,
        None,
        best_config,
        final_seed,
    )
    final_fit = train_bnn(
        pair.train_features,
        pair.train.targets,
        None,
        None,
        best_config,
        device,
        final_seed,
        evaluate_regression,
        trait_name,
        fixed_epochs=final_epoch,
    )
    model = build_model(pair.train_features.shape[1], best_config, device)
    model.load_state_dict(final_fit.state_dict)
    predictions, uncertainty = predict_bnn(
        model,
        pair.held_out_features,
        device,
        int(best_config["inference"]["test_samples"]),
        derive_seed(seed + 1, outer_fold, best.candidate_id, 999),
    )
    normalized = evaluate_regression(
        predictions[:, None],
        pair.held_out.targets[:, None],
        np.ones((len(predictions), 1), dtype=bool),
        [trait_name],
    ).metrics
    preprocessor_path = (
        data_directory
        / "cv"
        / f"outer_fold_{outer_fold}"
        / "final"
        / "preprocessing.json"
    )
    preprocessor = PerTraitPreprocessor.load_json(preprocessor_path)
    original_predictions = _inverse_trait(predictions, preprocessor, trait_index)
    original = evaluate_regression(
        original_predictions[:, None],
        pair.held_out.raw_targets[:, None],
        np.ones((len(predictions), 1), dtype=bool),
        [trait_name],
    ).metrics
    metrics_payload = {"normalized": normalized, "original": original}
    torch.save(
        {
            "model_state_dict": final_fit.state_dict,
            "config": best_config,
            "trait": trait_name,
            "outer_fold": outer_fold,
            "final_epoch": final_epoch,
            "variant_schema": pair.train.variants,
            "marker_pipeline": _pipeline_payload(pair.pipeline),
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
            "outer_train_observed": len(pair.train.sample_ids),
            "outer_train_discarded": list(pair.train.discarded_sample_ids),
            "outer_test_observed": len(pair.held_out.sample_ids),
            "outer_test_discarded": list(pair.held_out.discarded_sample_ids),
            "outer_test_sample_ids": list(pair.held_out.sample_ids),
            "inner_folds": split_audit,
        },
    )
    _write_json(
        fold_path / "selected_variants.json",
        {"variants": _selected_variant_payload(pair)},
    )
    with (fold_path / "predictions_normalized_scale.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["SampleID", "Prediction", "Observed", "PosteriorStd"])
        for sample_id, prediction, observed, std in zip(
            pair.held_out.sample_ids,
            predictions,
            pair.held_out.targets,
            uncertainty,
        ):
            writer.writerow(
                [sample_id, float(prediction), float(observed), float(std)]
            )
    with (fold_path / "predictions_original_scale.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["SampleID", "Prediction", "Observed"])
        for sample_id, prediction, observed in zip(
            pair.held_out.sample_ids,
            original_predictions,
            pair.held_out.raw_targets,
        ):
            writer.writerow([sample_id, float(prediction), float(observed)])
    return {
        "trait": trait_name,
        "outer_fold": outer_fold,
        "best_candidate_id": best.candidate_id,
        "best_parameters": dict(best.parameters),
        "best_valid_pearson_mean": best.objective,
        "final_epoch": final_epoch,
        "metrics": metrics_payload,
        "elapsed_seconds": time.time() - fold_start,
    }


def _run_trait_fold(
    job: TraitFoldJob,
    device_name: str,
    context: TraitFoldContext,
) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    prepared = load_prepared_data(Path(context.data_directory))
    raw_targets = prepared.targets
    print(
        f"[INFO] BNN trait={job.trait_name} outer_fold={job.outer_fold} "
        f"candidates={len(context.candidates)} "
        f"inner_folds={context.inner_count} device={device}"
    )
    return run_outer_fold(
        Path(context.data_directory),
        Path(context.output_directory),
        prepared,
        raw_targets,
        job.trait_name,
        job.trait_index,
        job.outer_fold,
        context.base_config,
        context.candidates,
        context.inner_count,
        device,
        context.seed,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_directory = Path(args.data_dir).resolve()
    output_directory = Path(args.output_dir).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_directory}")
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    prepared = load_prepared_data(data_directory)
    raw_targets = prepared.targets
    trait_names = list(prepared.metadata["trait_names"])
    configured_regression = prepared.metadata.get("regression_tasks")
    if isinstance(configured_regression, list):
        regression_traits = [str(trait) for trait in configured_regression]
    else:
        trait_tasks = prepared.metadata.get("trait_tasks")
        if isinstance(trait_tasks, list) and len(trait_tasks) == len(trait_names):
            regression_traits = [
                trait
                for trait, task in zip(trait_names, trait_tasks)
                if task == "regression"
            ]
        else:
            regression_traits = trait_names
    traits = list(args.traits or regression_traits)
    if not traits:
        raise ValueError("Prepared data contain no regression traits")
    invalid = [trait for trait in traits if trait not in regression_traits]
    if invalid:
        raise ValueError(f"Unknown traits: {invalid}")
    outer_count = int(prepared.metadata["outer_folds"])
    inner_count = int(prepared.metadata["inner_folds"])
    outer_folds = args.outer_folds or list(range(outer_count))
    if any(fold < 0 or fold >= outer_count for fold in outer_folds):
        raise ValueError(f"Outer folds must be in 0..{outer_count - 1}")
    if args.max_inner_folds is not None:
        inner_count = min(inner_count, args.max_inner_folds)
    candidates = generate_grid_candidates(config["hpo"]["parameters"])
    if len(candidates) != 32:
        raise ValueError(f"BNN grid must contain 32 candidates, got {len(candidates)}")
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    worker_gpu_ids = expand_gpu_workers(gpu_ids, args.jobs_per_gpu)
    jobs = [
        TraitFoldJob(
            job_id=trait_position * len(outer_folds) + fold_position,
            trait_name=trait_name,
            trait_index=trait_names.index(trait_name),
            outer_fold=outer_fold,
        )
        for trait_position, trait_name in enumerate(traits)
        for fold_position, outer_fold in enumerate(outer_folds)
    ]
    worker_context = TraitFoldContext(
        data_directory=str(data_directory),
        output_directory=str(output_directory),
        base_config=config,
        candidates=tuple(dict(candidate) for candidate in candidates),
        inner_count=inner_count,
        seed=args.seed,
    )
    work_results = execute_gpu_jobs(
        jobs,
        _run_trait_fold,
        worker_gpu_ids,
        worker_args=(worker_context,),
        raise_on_error=True,
    )
    results = [work_result.value for work_result in work_results]
    _write_json(
        output_directory / "summary.json",
        {
            "data_dir": str(data_directory),
            "config": str(Path(args.config).resolve()),
            "traits": traits,
            "outer_folds": outer_folds,
            "results": results,
            "outer_fold_summary": aggregate_outer_metrics(results, traits),
        },
    )


if __name__ == "__main__":
    main()
