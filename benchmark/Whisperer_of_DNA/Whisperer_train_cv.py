#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/Marxin1992/Whisperer_of_DNA.git

"""Leakage-safe nested cross-validation for multi-trait DNA Whisper."""

from __future__ import annotations

import argparse
import json
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
WHISPERER_DIRECTORY = SCRIPT_DIRECTORY
BENCHMARK_SOURCE = WHISPERER_DIRECTORY / "src_benchmark"
PROJECT_ROOT = WHISPERER_DIRECTORY.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_path in (str(SOURCE_ROOT), str(BENCHMARK_SOURCE)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from aquila.benchmark.common import (
    aggregate_outer_folds,
    build_sample_audit,
    sanitize_json,
    serialize_candidate,
    write_json,
    write_predictions_csv,
)
from aquila.training.distributed import (
    derive_seed,
    detect_gpu_ids,
    execute_gpu_jobs,
)
from aquila.training.cuda_runtime import configure_cuda_runtime
from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import (
    CandidateResult,
    InnerFoldResult,
    generate_grid_candidates,
    half_up_median_epoch,
    select_best_candidate,
)
from whisperer_data import MultiTraitSplit, WhispererPreparedBenchmark
from whisperer_model import (
    apply_candidate_overrides,
    predict_model,
    train_model,
)


@dataclass(frozen=True)
class OuterFoldJob:
    """One independently scheduled outer-fold multi-trait run."""

    job_id: int
    outer_fold: int


@dataclass(frozen=True)
class OuterFoldContext:
    """Spawn-safe inputs shared by DNA Whisper workers."""

    data_directory: str
    output_directory: str
    config: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    inner_folds: tuple[int, ...]
    trait_names: tuple[str, ...]
    max_epochs: int
    budget: dict[str, Any]
    live_metrics_log: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def expand_gpu_workers(gpu_ids: Sequence[int], jobs_per_gpu: int) -> list[int]:
    """Create one scheduler slot per concurrent job allowed on each GPU."""

    if jobs_per_gpu < 1:
        raise ValueError("--jobs-per-gpu must be at least 1")
    return [gpu_id for gpu_id in gpu_ids for _ in range(jobs_per_gpu)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe nested CV for joint multi-trait DNA Whisper."
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "benchmark" / "test"))
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIRECTORY / "configs" / "Whisperer_nested_cv.yaml"),
    )
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Regression traits trained jointly in one model (default: all).",
    )
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-inner-folds", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--live-metrics-log",
        action="store_true",
        help=(
            "Append per-epoch metrics JSONL under "
            "{output}/fold_*/candidate_*/inner_*/metrics.jsonl and "
            "{output}/fold_*/outer_refit/metrics.jsonl."
        ),
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_path = (path.parent / config["model_config"]).resolve()
    with model_path.open("r", encoding="utf-8") as handle:
        config["model"] = json.load(handle)
    return config


def _candidate_parameters(raw: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "optimizer.learning_rate": "learning_rate",
        "optimizer.weight_decay": "weight_decay",
        "model.dropout": "dropout",
        "model.encoder_layers": "encoder_layers",
    }
    return {
        aliases.get(key, key.rsplit(".", 1)[-1]): value for key, value in raw.items()
    }


def _select_traits(
    benchmark: WhispererPreparedBenchmark,
    requested: Sequence[str] | None,
) -> list[str]:
    regression = [
        name
        for name, task in zip(benchmark.trait_names, benchmark.metadata["trait_tasks"])
        if task == "regression"
    ]
    selected = list(requested) if requested else regression
    invalid = [name for name in selected if name not in regression]
    if invalid:
        raise ValueError(f"Unknown regression traits: {invalid}")
    if not selected:
        raise ValueError("DNAWhisper multi-trait training requires at least one trait")
    return selected


def _slice_scale(scale: Mapping[str, Any], trait_name: str) -> dict[str, Any]:
    per_trait = dict(scale["per_trait"][trait_name])
    sliced = {
        "per_trait": {trait_name: per_trait},
        "aggregate": {
            "pearson": per_trait["pearson"],
            "r2": per_trait["r2"],
            "mse": per_trait["mse"],
            "rmse": per_trait["rmse"],
            "mae": per_trait["mae"],
            "n_traits": 1,
            "n_observations": per_trait["n"],
            "within_accession_pearson": float("nan"),
            "n_accessions_within_accession": 0,
        },
    }
    for metric_name in ("pearson", "r2", "mse", "rmse", "mae"):
        sliced[f"avg_{metric_name}"] = per_trait[metric_name]
    sliced["avg_within_accession_pearson"] = float("nan")
    sliced["n_accessions_within_accession"] = 0
    return sliced


def _slice_metrics(metrics: Mapping[str, Any], trait_name: str) -> dict[str, Any]:
    return {
        "normalized": _slice_scale(metrics["normalized"], trait_name),
        "original": _slice_scale(metrics["original"], trait_name),
        "test_loss": metrics["normalized"]["per_trait"][trait_name]["mse"],
    }


def _observation_counts(split: MultiTraitSplit) -> dict[str, int]:
    return {
        name: int(split.observed_mask[:, index].sum())
        for index, name in enumerate(split.trait_names)
    }


def _write_trait_predictions(
    path: Path,
    split: MultiTraitSplit,
    predictions_processed: np.ndarray,
    predictions_original: np.ndarray,
    trait_name: str,
    outer_fold: int,
) -> None:
    column = split.trait_names.index(trait_name)
    observed = np.asarray(split.observed_mask[:, column], dtype=bool)
    sample_ids = tuple(
        sample_id for sample_id, keep in zip(split.sample_ids, observed) if keep
    )
    write_predictions_csv(
        path,
        sample_ids,
        split.processed_targets[observed, column],
        predictions_processed[observed, column],
        split.raw_targets[observed, column],
        predictions_original[observed, column],
        trait_name=trait_name,
        outer_fold=outer_fold,
    )


def run_outer_fold(
    benchmark: WhispererPreparedBenchmark,
    output_directory: Path,
    trait_names: Sequence[str],
    outer_fold: int,
    config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    inner_folds: Sequence[int],
    device: torch.device,
    max_epochs: int,
    budget: Mapping[str, Any],
    live_metrics_log: bool = False,
) -> dict[str, Any]:
    started = time.time()
    names = tuple(str(name) for name in trait_names)
    fold_path = output_directory / f"fold_{outer_fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    block_length = int(config["model"]["embedding"]["Block_length"])
    patience = int(config["training"]["patience"])
    inner_results: dict[int, list[InnerFoldResult]] = {
        index: [] for index in range(len(candidates))
    }
    histories: dict[str, Any] = {}
    inner_audit = []
    global_variants = None
    variant_schema = None
    for inner_fold in inner_folds:
        train, valid, schema = benchmark.load_multi_trait_fold(
            names,
            outer_fold,
            inner_fold,
            block_length=block_length,
            expected_variants=global_variants,
        )
        if global_variants is None:
            global_variants = tuple(tuple(value) for value in schema["variants"])
            variant_schema = schema
        inner_audit.append(
            {
                "inner_fold": inner_fold,
                **build_sample_audit(train, valid, held_out_name="valid"),
                "train_observations_per_trait": _observation_counts(train),
                "valid_observations_per_trait": _observation_counts(valid),
            }
        )
        for candidate_id, raw_parameters in enumerate(candidates):
            parameters = _candidate_parameters(raw_parameters)
            parameters["batch_size"] = int(config["training"]["batch_size"])
            training_seed = derive_seed(
                int(config["seed"]),
                outer_fold,
                candidate_id,
                inner_fold,
            )
            model_config = apply_candidate_overrides(config["model"], parameters, names)
            metrics_log_path = (
                fold_path
                / f"candidate_{candidate_id}"
                / f"inner_{inner_fold}"
                / "metrics.jsonl"
                if live_metrics_log
                else None
            )
            result = train_model(
                train.genotypes,
                train.processed_targets,
                valid.genotypes,
                valid.processed_targets,
                model_config,
                parameters,
                device,
                training_seed,
                max_epochs=max_epochs,
                patience=patience,
                train_mask=train.observed_mask,
                valid_mask=valid.observed_mask,
                trait_names=names,
                metrics_log_path=metrics_log_path,
            )
            inner_results[candidate_id].append(
                InnerFoldResult(
                    inner_fold,
                    result.best_metric,
                    result.best_epoch,
                    {**result.best_metrics, "training_seed": training_seed},
                )
            )
            histories[f"candidate_{candidate_id}/inner_{inner_fold}"] = list(
                result.history
            )
    candidate_results = []
    for candidate_id, raw_parameters in enumerate(candidates):
        results = tuple(inner_results[candidate_id])
        metrics = np.asarray([result.metric for result in results], dtype=float)
        objective = (
            float(metrics.mean()) if np.isfinite(metrics).all() else float("nan")
        )
        parameters = _candidate_parameters(raw_parameters)
        parameters["batch_size"] = int(config["training"]["batch_size"])
        candidate_results.append(
            CandidateResult(
                candidate_id,
                parameters,
                objective,
                results,
            )
        )
    hpo = select_best_candidate(candidate_results, "maximize", "grid")
    best = hpo.best
    final_epoch = half_up_median_epoch(best.best_epochs)
    final_parameters = dict(best.parameters)
    final_config = apply_candidate_overrides(config["model"], final_parameters, names)
    final_seed = derive_seed(
        int(config["seed"]),
        outer_fold,
        best.candidate_id,
        999,
    )
    final_config["random_seed"] = final_seed
    outer_train, outer_test, final_schema = benchmark.load_multi_trait_fold(
        names,
        outer_fold,
        None,
        block_length=block_length,
        expected_variants=global_variants,
    )
    variant_schema = final_schema
    final_result = train_model(
        outer_train.genotypes,
        outer_train.processed_targets,
        None,
        None,
        final_config,
        final_parameters,
        device,
        final_seed,
        max_epochs=final_epoch,
        patience=final_epoch,
        fixed_epochs=final_epoch,
        train_mask=outer_train.observed_mask,
        trait_names=names,
        metrics_log_path=(
            fold_path / "outer_refit" / "metrics.jsonl" if live_metrics_log else None
        ),
    )
    predictions, observed, test_loss = predict_model(
        final_result.state_dict,
        outer_test.genotypes,
        outer_test.processed_targets,
        final_config,
        final_parameters,
        device,
        outer_test.observed_mask,
    )
    predictions_original = benchmark.inverse_selected_traits(
        predictions,
        observed,
        names,
        outer_fold,
    )
    processed_metrics = evaluate_regression(
        predictions,
        outer_test.processed_targets,
        observed,
        names,
    ).metrics
    original_metrics = evaluate_regression(
        predictions_original,
        outer_test.raw_targets,
        observed,
        names,
    ).metrics
    checkpoint = {
        "state_dict": final_result.state_dict,
        "config": final_config,
        "parameters": final_parameters,
        "traits": list(names),
        "outer_fold": outer_fold,
        "final_epoch": final_epoch,
        "training_seed": final_seed,
        "retained_variants": outer_train.variants,
    }
    torch.save(checkpoint, fold_path / "best_model.ckpt")
    with (fold_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "model": final_config,
                "optimizer": final_parameters,
                "training": config["training"],
                "budget": dict(budget),
                "traits": list(names),
            },
            handle,
            sort_keys=False,
        )
    shutil.copy2(
        benchmark.resolve_fold_paths(outer_fold).preprocessing,
        fold_path / "preprocessing.json",
    )
    write_json(
        fold_path / "hpo_results.json",
        {
            "method": hpo.method,
            "direction": hpo.direction,
            "best_candidate_id": best.candidate_id,
            "best_parameters": best.parameters,
            "best_valid_pearson_mean": best.objective,
            "final_epoch": final_epoch,
            "candidates": [serialize_candidate(item) for item in candidate_results],
            "actual_budget": dict(budget),
        },
    )
    metrics = {
        "normalized": processed_metrics,
        "original": original_metrics,
        "test_loss": test_loss,
    }
    write_json(fold_path / "metrics.json", metrics)
    write_json(
        fold_path / "training_history.json",
        {**histories, "outer_refit": list(final_result.history)},
    )
    write_json(
        fold_path / "sample_audit.json",
        {
            "outer": {
                **build_sample_audit(outer_train, outer_test, held_out_name="test"),
                "train_observations_per_trait": _observation_counts(outer_train),
                "test_observations_per_trait": _observation_counts(outer_test),
            },
            "inner_folds": inner_audit,
        },
    )
    write_json(fold_path / "variant_schema.json", variant_schema)
    for trait_name in names:
        _write_trait_predictions(
            fold_path / f"predictions_{trait_name}_original_scale.csv",
            outer_test,
            predictions,
            predictions_original,
            trait_name,
            outer_fold,
        )
        trait_fold = output_directory / trait_name / f"fold_{outer_fold}"
        trait_fold.mkdir(parents=True, exist_ok=True)
        write_json(trait_fold / "metrics.json", _slice_metrics(metrics, trait_name))
        _write_trait_predictions(
            trait_fold / "predictions_original_scale.csv",
            outer_test,
            predictions,
            predictions_original,
            trait_name,
            outer_fold,
        )
    runtime = {
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "training_seed": final_seed,
        "actual_budget": dict(budget),
        "outer_test_evaluations": 1,
        "traits": list(names),
    }
    write_json(fold_path / "runtime.json", runtime)
    return {
        "outer_fold": outer_fold,
        "traits": list(names),
        "best_candidate_id": best.candidate_id,
        "best_parameters": best.parameters,
        "best_valid_pearson_mean": best.objective,
        "final_epoch": final_epoch,
        "metrics": sanitize_json(metrics),
        "runtime": runtime,
    }


def _run_outer_fold(
    job: OuterFoldJob,
    device_name: str,
    context: OuterFoldContext,
) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda":
        configure_cuda_runtime(device_name, deterministic=False)
    benchmark = WhispererPreparedBenchmark(Path(context.data_directory))
    print(
        f"[INFO] traits={list(context.trait_names)} outer_fold={job.outer_fold} "
        f"candidates={len(context.candidates)} "
        f"inner_folds={len(context.inner_folds)} device={device}"
    )
    return run_outer_fold(
        benchmark,
        Path(context.output_directory),
        context.trait_names,
        job.outer_fold,
        context.config,
        context.candidates,
        context.inner_folds,
        device,
        context.max_epochs,
        context.budget,
        context.live_metrics_log,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_directory = Path(args.data_dir).resolve()
    config_path = Path(args.config).resolve()
    output_directory = Path(args.output_dir).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_directory}")
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    benchmark = WhispererPreparedBenchmark(data_directory)
    traits = _select_traits(benchmark, args.traits)
    outer_folds = args.outer_folds or list(range(benchmark.outer_fold_count))
    if any(fold < 0 or fold >= benchmark.outer_fold_count for fold in outer_folds):
        raise ValueError("Requested outer fold is outside the prepared fold range")
    inner_count = benchmark.inner_fold_count
    if args.max_inner_folds is not None:
        inner_count = min(inner_count, args.max_inner_folds)
    inner_folds = list(range(inner_count))
    raw_candidates = generate_grid_candidates(config["hpo"]["parameters"])
    if not raw_candidates:
        raise ValueError("DNA Whisper grid must contain at least one candidate")
    candidates = (
        raw_candidates[: args.max_candidates] if args.max_candidates else raw_candidates
    )
    max_epochs = int(config["training"]["max_epochs"])
    if args.max_epochs is not None:
        max_epochs = min(max_epochs, args.max_epochs)
    budget = {
        "planned_outer_folds": benchmark.outer_fold_count,
        "planned_inner_folds": benchmark.inner_fold_count,
        "planned_candidates": len(raw_candidates),
        "planned_max_epochs": int(config["training"]["max_epochs"]),
        "actual_outer_folds": list(outer_folds),
        "actual_inner_folds": inner_folds,
        "actual_candidates": len(candidates),
        "actual_max_epochs": max_epochs,
        "actual_traits": list(traits),
        "smoke_reduced": any(
            (
                len(outer_folds) < benchmark.outer_fold_count,
                inner_count < benchmark.inner_fold_count,
                len(candidates) < len(raw_candidates),
                max_epochs < int(config["training"]["max_epochs"]),
            )
        ),
    }
    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    worker_gpu_ids = expand_gpu_workers(gpu_ids, args.jobs_per_gpu)
    jobs = [
        OuterFoldJob(job_id=fold_index, outer_fold=outer_fold)
        for fold_index, outer_fold in enumerate(outer_folds)
    ]
    worker_context = OuterFoldContext(
        data_directory=str(data_directory),
        output_directory=str(output_directory),
        config=config,
        candidates=tuple(dict(candidate) for candidate in candidates),
        inner_folds=tuple(inner_folds),
        trait_names=tuple(traits),
        max_epochs=max_epochs,
        budget=budget,
        live_metrics_log=args.live_metrics_log,
    )
    work_results = execute_gpu_jobs(
        jobs,
        _run_outer_fold,
        worker_gpu_ids,
        worker_args=(worker_context,),
        raise_on_error=True,
    )
    results = [work_result.value for work_result in work_results]
    run_index = []
    for trait_name in traits:
        fold_results = [
            {
                "trait": trait_name,
                "outer_fold": result["outer_fold"],
                "best_candidate_id": result["best_candidate_id"],
                "best_parameters": result["best_parameters"],
                "best_valid_pearson_mean": result["best_valid_pearson_mean"],
                "final_epoch": result["final_epoch"],
                "metrics": _slice_metrics(result["metrics"], trait_name),
                "runtime": result["runtime"],
            }
            for result in results
        ]
        write_json(
            output_directory / trait_name / "summary.json",
            {
                "trait": trait_name,
                "folds": fold_results,
                "metrics": aggregate_outer_folds(
                    [result["metrics"] for result in fold_results]
                ),
                "actual_budget": budget,
            },
        )
        run_index.append(
            {
                "trait": trait_name,
                "status": "completed",
                "error": None,
                "completed_outer_folds": [
                    result["outer_fold"] for result in fold_results
                ],
            }
        )
    write_json(
        output_directory / "summary.json",
        {
            "data_dir": data_directory,
            "config": config_path,
            "traits": traits,
            "actual_budget": budget,
            "folds": results,
            "metrics": aggregate_outer_folds([result["metrics"] for result in results]),
            "runs": run_index,
        },
    )


if __name__ == "__main__":
    main()
