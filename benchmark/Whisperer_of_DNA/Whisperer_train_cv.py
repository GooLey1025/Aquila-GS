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
from threadpoolctl import threadpool_limits

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
from aquila.data import resolve_outer_folds
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
class InnerHPOJob:
    """One independently scheduled inner-fold HPO candidate."""

    job_id: int
    outer_fold: int
    candidate_id: int
    inner_fold: int


@dataclass(frozen=True)
class OuterRefitJob:
    """Outer-train refit after all inner HPO jobs for a fold complete."""

    job_id: int
    outer_fold: int
    started: float
    best_candidate_id: int
    best_parameters: dict[str, Any]
    final_epoch: int
    best_valid_pearson_mean: float
    candidate_results: tuple[CandidateResult, ...]
    histories: dict[str, Any]
    inner_audit: tuple[dict[str, Any], ...]
    variant_schema: dict[str, Any] | None


@dataclass(frozen=True)
class WorkerContext:
    """Spawn-safe inputs shared by DNA Whisper GPU workers."""

    data_directory: str
    output_directory: str
    config: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    inner_folds: tuple[int, ...]
    trait_names: tuple[str, ...]
    max_epochs: int
    budget: dict[str, Any]
    live_metrics_log: bool
    resume: bool


_LOADED_SPLITS: dict[tuple[Any, ...], tuple[MultiTraitSplit, MultiTraitSplit, dict[str, Any]]] = {}


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


def _inner_hpo_job_id(outer_fold: int, candidate_id: int, inner_fold: int) -> int:
    return int(outer_fold) * 1_000_000 + int(inner_fold) * 1_000 + int(candidate_id)


def _outer_refit_job_id(outer_fold: int) -> int:
    return int(outer_fold) * 1_000_000 + 999_999


def _build_inner_hpo_jobs(
    outer_folds: Sequence[int],
    candidate_count: int,
    inner_folds: Sequence[int],
) -> list[InnerHPOJob]:
    """Order jobs by inner fold then candidate so GPU workers reuse loaded VCFs."""

    jobs = []
    for outer_fold in outer_folds:
        for inner_fold in inner_folds:
            for candidate_id in range(candidate_count):
                jobs.append(
                    InnerHPOJob(
                        job_id=_inner_hpo_job_id(outer_fold, candidate_id, inner_fold),
                        outer_fold=int(outer_fold),
                        candidate_id=int(candidate_id),
                        inner_fold=int(inner_fold),
                    )
                )
    return jobs


def _limit_worker_threads() -> Any:
    limiter = threadpool_limits(limits=1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return limiter


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
        help="Maximum concurrent inner-HPO jobs per GPU (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete the output directory before running. Without this flag, "
            "completed folds and completed inner metrics logs are reused."
        ),
    )
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


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_completed_outer_fold(
    output_directory: Path,
    outer_fold: int,
    trait_names: Sequence[str],
) -> dict[str, Any] | None:
    """Recover a fully written outer-fold result without retraining it."""

    fold_path = output_directory / f"fold_{outer_fold}"
    required = (
        fold_path / "best_model.ckpt",
        fold_path / "hpo_results.json",
        fold_path / "metrics.json",
        fold_path / "runtime.json",
    )
    if not all(path.is_file() for path in required):
        return None
    try:
        hpo = _read_json(fold_path / "hpo_results.json")
        metrics = _read_json(fold_path / "metrics.json")
        runtime = _read_json(fold_path / "runtime.json")
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    completed_traits = tuple(str(name) for name in runtime.get("traits", ()))
    if completed_traits and completed_traits != tuple(trait_names):
        return None
    try:
        return {
            "outer_fold": outer_fold,
            "traits": list(trait_names),
            "best_candidate_id": hpo["best_candidate_id"],
            "best_parameters": hpo["best_parameters"],
            "best_valid_pearson_mean": hpo["best_valid_pearson_mean"],
            "final_epoch": hpo["final_epoch"],
            "metrics": metrics,
            "runtime": runtime,
        }
    except (KeyError, TypeError):
        return None


def _load_completed_inner_result(
    path: Path,
    inner_fold: int,
    expected_seed: int,
    max_epochs: int,
) -> tuple[InnerFoldResult, list[dict[str, Any]]] | None:
    """Recover one completed inner run from its append-only metrics log."""

    if not path.is_file():
        return None
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return None
                if not isinstance(row, dict):
                    return None
                rows.append(row)
    except OSError:
        return None
    if not rows:
        return None
    epochs = [row.get("epoch") for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        return None
    if any(row.get("seed") != expected_seed for row in rows):
        return None
    last = rows[-1]
    try:
        last_epoch = int(last["epoch"])
        best_epoch = int(last["best_epoch"])
        best_metric = float(last["best_valid_r"])
    except (KeyError, TypeError, ValueError):
        return None
    if not bool(last.get("early_stop")) and last_epoch < max_epochs:
        return None
    if not np.isfinite(best_metric):
        return None
    history = [
        {
            "epoch": row["epoch"],
            "train_loss": row.get("train_loss"),
            "valid_loss": row.get("valid_loss"),
            "valid_avg_pearson": row.get("valid_r"),
        }
        for row in rows
    ]
    return (
        InnerFoldResult(
            inner_fold,
            best_metric,
            best_epoch,
            {"training_seed": expected_seed, "recovered_from_metrics_log": True},
        ),
        history,
    )


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


def _load_inner_split(
    context: WorkerContext,
    outer_fold: int,
    inner_fold: int,
) -> tuple[MultiTraitSplit, MultiTraitSplit, dict[str, Any]]:
    key = (context.data_directory, outer_fold, inner_fold, context.trait_names)
    cached = _LOADED_SPLITS.get(key)
    if cached is not None:
        return cached
    _LOADED_SPLITS.clear()
    benchmark = WhispererPreparedBenchmark(Path(context.data_directory))
    train, valid, schema = benchmark.load_multi_trait_fold(
        context.trait_names,
        outer_fold,
        inner_fold,
        block_length=int(context.config["model"]["embedding"]["Block_length"]),
    )
    _LOADED_SPLITS[key] = (train, valid, schema)
    return train, valid, schema


def _run_inner_hpo_job(
    job: InnerHPOJob,
    device_name: str,
    context: WorkerContext,
) -> dict[str, Any]:
    limiter = _limit_worker_threads()
    try:
        device = torch.device(device_name)
        if device.type == "cuda":
            configure_cuda_runtime(device_name, deterministic=False)
        names = tuple(context.trait_names)
        fold_path = Path(context.output_directory) / f"fold_{job.outer_fold}"
        fold_path.mkdir(parents=True, exist_ok=True)
        parameters = _candidate_parameters(context.candidates[job.candidate_id])
        parameters["batch_size"] = int(context.config["training"]["batch_size"])
        training_seed = derive_seed(
            int(context.config["seed"]),
            job.outer_fold,
            job.candidate_id,
            job.inner_fold,
        )
        metrics_log_path = (
            fold_path
            / f"candidate_{job.candidate_id}"
            / f"inner_{job.inner_fold}"
            / "metrics.jsonl"
            if context.live_metrics_log or context.resume
            else None
        )
        recovered = None
        if context.resume:
            recovered = _load_completed_inner_result(
                metrics_log_path,
                job.inner_fold,
                training_seed,
                context.max_epochs,
            )
        if recovered is not None:
            recovered_result, recovered_history = recovered
            print(
                f"[INFO] traits={list(names)} outer_fold={job.outer_fold} "
                f"inner_fold={job.inner_fold} candidate={job.candidate_id + 1}/"
                f"{len(context.candidates)} recovered best_valid_pearson="
                f"{recovered_result.metric:.6f} "
                f"best_epoch={recovered_result.best_epoch} device={device}",
                flush=True,
            )
            return {
                "outer_fold": job.outer_fold,
                "candidate_id": job.candidate_id,
                "inner_fold": job.inner_fold,
                "metric": recovered_result.metric,
                "best_epoch": recovered_result.best_epoch,
                "metrics": dict(recovered_result.metrics),
                "history": recovered_history,
                "audit": None,
                "variant_schema": None,
                "recovered": True,
            }
        if context.resume and metrics_log_path is not None and metrics_log_path.exists():
            metrics_log_path.unlink()
        train, valid, schema = _load_inner_split(context, job.outer_fold, job.inner_fold)
        print(
            f"[INFO] traits={list(names)} outer_fold={job.outer_fold} "
            f"inner_fold={job.inner_fold} candidate={job.candidate_id + 1}/"
            f"{len(context.candidates)} device={device}",
            flush=True,
        )
        result = train_model(
            train.genotypes,
            train.processed_targets,
            valid.genotypes,
            valid.processed_targets,
            apply_candidate_overrides(context.config["model"], parameters, names),
            parameters,
            device,
            training_seed,
            max_epochs=context.max_epochs,
            patience=int(context.config["training"]["patience"]),
            train_mask=train.observed_mask,
            valid_mask=valid.observed_mask,
            trait_names=names,
            metrics_log_path=metrics_log_path,
        )
        print(
            f"[INFO] traits={list(names)} outer_fold={job.outer_fold} "
            f"inner_fold={job.inner_fold} candidate={job.candidate_id + 1}/"
            f"{len(context.candidates)} "
            f"best_valid_pearson={result.best_metric:.6f} "
            f"best_epoch={result.best_epoch} device={device}",
            flush=True,
        )
        return {
            "outer_fold": job.outer_fold,
            "candidate_id": job.candidate_id,
            "inner_fold": job.inner_fold,
            "metric": result.best_metric,
            "best_epoch": result.best_epoch,
            "metrics": {**result.best_metrics, "training_seed": training_seed},
            "history": list(result.history),
            "audit": {
                "inner_fold": job.inner_fold,
                **build_sample_audit(train, valid, held_out_name="valid"),
                "train_observations_per_trait": _observation_counts(train),
                "valid_observations_per_trait": _observation_counts(valid),
            },
            "variant_schema": schema,
            "recovered": False,
        }
    finally:
        limiter.restore_original_limits()


def _assemble_outer_hpo(
    payloads: Sequence[Mapping[str, Any]],
    context: WorkerContext,
    outer_fold: int,
) -> dict[str, Any]:
    by_candidate: dict[int, dict[int, Mapping[str, Any]]] = {}
    for payload in payloads:
        if int(payload["outer_fold"]) != int(outer_fold):
            continue
        by_candidate.setdefault(int(payload["candidate_id"]), {})[
            int(payload["inner_fold"])
        ] = payload
    histories: dict[str, Any] = {}
    audits: dict[int, dict[str, Any]] = {}
    variant_schema = None
    candidate_results = []
    for candidate_id, raw_parameters in enumerate(context.candidates):
        folds = by_candidate.get(candidate_id, {})
        missing = [inner_fold for inner_fold in context.inner_folds if inner_fold not in folds]
        if missing:
            raise ValueError(
                f"outer_fold={outer_fold} candidate={candidate_id} missing inner folds {missing}"
            )
        inner_results = []
        for inner_fold in context.inner_folds:
            payload = folds[inner_fold]
            inner_results.append(
                InnerFoldResult(
                    inner_fold,
                    float(payload["metric"]),
                    int(payload["best_epoch"]),
                    dict(payload["metrics"]),
                )
            )
            histories[f"candidate_{candidate_id}/inner_{inner_fold}"] = payload["history"]
            if payload.get("audit") is not None:
                audits[inner_fold] = dict(payload["audit"])
            if payload.get("variant_schema") is not None:
                schema = dict(payload["variant_schema"])
                if variant_schema is None:
                    variant_schema = schema
                elif variant_schema.get("variants") != schema.get("variants"):
                    raise ValueError(
                        "Fold VCF variant schema differs from the global schema"
                    )
        metrics = np.asarray([result.metric for result in inner_results], dtype=float)
        parameters = _candidate_parameters(raw_parameters)
        parameters["batch_size"] = int(context.config["training"]["batch_size"])
        candidate_results.append(
            CandidateResult(
                candidate_id,
                parameters,
                float(metrics.mean()) if np.isfinite(metrics).all() else float("nan"),
                tuple(inner_results),
            )
        )
    hpo = select_best_candidate(candidate_results, "maximize", "grid")
    inner_audit = tuple(
        audits.get(
            inner_fold,
            {"inner_fold": inner_fold, "recovered_from_metrics_logs": True},
        )
        for inner_fold in context.inner_folds
    )
    return {
        "hpo": hpo,
        "candidate_results": tuple(candidate_results),
        "histories": histories,
        "inner_audit": inner_audit,
        "variant_schema": variant_schema,
    }


def _run_outer_refit_job(
    job: OuterRefitJob,
    device_name: str,
    context: WorkerContext,
) -> dict[str, Any]:
    limiter = _limit_worker_threads()
    try:
        device = torch.device(device_name)
        if device.type == "cuda":
            configure_cuda_runtime(device_name, deterministic=False)
        names = tuple(context.trait_names)
        output_directory = Path(context.output_directory)
        fold_path = output_directory / f"fold_{job.outer_fold}"
        fold_path.mkdir(parents=True, exist_ok=True)
        final_parameters = dict(job.best_parameters)
        final_config = apply_candidate_overrides(
            context.config["model"],
            final_parameters,
            names,
        )
        final_seed = derive_seed(
            int(context.config["seed"]),
            job.outer_fold,
            job.best_candidate_id,
            999,
        )
        final_config["random_seed"] = final_seed
        expected_variants = None
        if job.variant_schema is not None:
            expected_variants = tuple(
                tuple(value) for value in job.variant_schema["variants"]
            )
        benchmark = WhispererPreparedBenchmark(Path(context.data_directory))
        print(
            f"[INFO] traits={list(names)} outer_fold={job.outer_fold} "
            f"outer_refit candidate={job.best_candidate_id} "
            f"final_epoch={job.final_epoch} device={device}",
            flush=True,
        )
        outer_train, outer_test, variant_schema = benchmark.load_multi_trait_fold(
            names,
            job.outer_fold,
            None,
            block_length=int(context.config["model"]["embedding"]["Block_length"]),
            expected_variants=expected_variants,
        )
        final_result = train_model(
            outer_train.genotypes,
            outer_train.processed_targets,
            None,
            None,
            final_config,
            final_parameters,
            device,
            final_seed,
            max_epochs=job.final_epoch,
            patience=job.final_epoch,
            fixed_epochs=job.final_epoch,
            train_mask=outer_train.observed_mask,
            trait_names=names,
            metrics_log_path=(
                fold_path / "outer_refit" / "metrics.jsonl"
                if context.live_metrics_log
                else None
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
            job.outer_fold,
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
            "outer_fold": job.outer_fold,
            "final_epoch": job.final_epoch,
            "training_seed": final_seed,
            "retained_variants": outer_train.variants,
        }
        torch.save(checkpoint, fold_path / "best_model.ckpt")
        with (fold_path / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "model": final_config,
                    "optimizer": final_parameters,
                    "training": context.config["training"],
                    "budget": dict(context.budget),
                    "traits": list(names),
                },
                handle,
                sort_keys=False,
            )
        shutil.copy2(
            benchmark.resolve_fold_paths(job.outer_fold).preprocessing,
            fold_path / "preprocessing.json",
        )
        write_json(
            fold_path / "hpo_results.json",
            {
                "method": "grid",
                "direction": "maximize",
                "best_candidate_id": job.best_candidate_id,
                "best_parameters": job.best_parameters,
                "best_valid_pearson_mean": job.best_valid_pearson_mean,
                "final_epoch": job.final_epoch,
                "candidates": [
                    serialize_candidate(item) for item in job.candidate_results
                ],
                "actual_budget": dict(context.budget),
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
            {**job.histories, "outer_refit": list(final_result.history)},
        )
        write_json(
            fold_path / "sample_audit.json",
            {
                "outer": {
                    **build_sample_audit(outer_train, outer_test, held_out_name="test"),
                    "train_observations_per_trait": _observation_counts(outer_train),
                    "test_observations_per_trait": _observation_counts(outer_test),
                },
                "inner_folds": list(job.inner_audit),
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
                job.outer_fold,
            )
            trait_fold = output_directory / trait_name / f"fold_{job.outer_fold}"
            trait_fold.mkdir(parents=True, exist_ok=True)
            write_json(trait_fold / "metrics.json", _slice_metrics(metrics, trait_name))
            _write_trait_predictions(
                trait_fold / "predictions_original_scale.csv",
                outer_test,
                predictions,
                predictions_original,
                trait_name,
                job.outer_fold,
            )
        runtime = {
            "elapsed_seconds": time.time() - job.started,
            "device": str(device),
            "training_seed": final_seed,
            "actual_budget": dict(context.budget),
            "outer_test_evaluations": 1,
            "traits": list(names),
        }
        write_json(fold_path / "runtime.json", runtime)
        return {
            "outer_fold": job.outer_fold,
            "traits": list(names),
            "best_candidate_id": job.best_candidate_id,
            "best_parameters": job.best_parameters,
            "best_valid_pearson_mean": job.best_valid_pearson_mean,
            "final_epoch": job.final_epoch,
            "metrics": sanitize_json(metrics),
            "runtime": runtime,
        }
    finally:
        limiter.restore_original_limits()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_directory = Path(args.data_dir).resolve()
    config_path = Path(args.config).resolve()
    output_directory = Path(args.output_dir).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_directory)
        else:
            print(
                f"[INFO] Resuming from existing results in {output_directory}",
                flush=True,
            )
    output_directory.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    benchmark = WhispererPreparedBenchmark(data_directory)
    traits = _select_traits(benchmark, args.traits)
    outer_folds = resolve_outer_folds(args.outer_folds, benchmark.metadata)
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
    completed_results = []
    pending_outer_folds = []
    for outer_fold in outer_folds:
        completed = (
            _load_completed_outer_fold(output_directory, outer_fold, traits)
            if not args.overwrite
            else None
        )
        if completed is None:
            pending_outer_folds.append(outer_fold)
        else:
            completed_results.append(completed)
            print(
                f"[INFO] outer_fold={outer_fold} recovered completed fold",
                flush=True,
            )
    worker_context = WorkerContext(
        data_directory=str(data_directory),
        output_directory=str(output_directory),
        config=config,
        candidates=tuple(dict(candidate) for candidate in candidates),
        inner_folds=tuple(inner_folds),
        trait_names=tuple(traits),
        max_epochs=max_epochs,
        budget=budget,
        live_metrics_log=args.live_metrics_log,
        resume=not args.overwrite,
    )
    inner_jobs = _build_inner_hpo_jobs(
        pending_outer_folds,
        len(candidates),
        inner_folds,
    )
    print(
        f"[INFO] traits={list(traits)} outer_folds={pending_outer_folds} "
        f"candidates={len(candidates)} inner_folds={inner_folds} "
        f"inner_hpo_jobs={len(inner_jobs)} gpus={gpu_ids or ['cpu']} "
        f"jobs_per_gpu={args.jobs_per_gpu}",
        flush=True,
    )
    fold_started = {outer_fold: time.time() for outer_fold in pending_outer_folds}
    inner_payloads = [
        work_result.value
        for work_result in execute_gpu_jobs(
            inner_jobs,
            _run_inner_hpo_job,
            worker_gpu_ids,
            worker_args=(worker_context,),
            raise_on_error=True,
        )
    ]
    refit_jobs = []
    for outer_fold in pending_outer_folds:
        assembled = _assemble_outer_hpo(inner_payloads, worker_context, outer_fold)
        best = assembled["hpo"].best
        refit_jobs.append(
            OuterRefitJob(
                job_id=_outer_refit_job_id(outer_fold),
                outer_fold=int(outer_fold),
                started=fold_started[outer_fold],
                best_candidate_id=int(best.candidate_id),
                best_parameters=dict(best.parameters),
                final_epoch=half_up_median_epoch(best.best_epochs),
                best_valid_pearson_mean=float(best.objective),
                candidate_results=assembled["candidate_results"],
                histories=assembled["histories"],
                inner_audit=assembled["inner_audit"],
                variant_schema=assembled["variant_schema"],
            )
        )
        print(
            f"[INFO] outer_fold={outer_fold} HPO complete "
            f"best_candidate={best.candidate_id} "
            f"best_valid_pearson_mean={best.objective:.6f} "
            f"final_epoch={half_up_median_epoch(best.best_epochs)}",
            flush=True,
        )
    work_results = execute_gpu_jobs(
        refit_jobs,
        _run_outer_refit_job,
        worker_gpu_ids,
        worker_args=(worker_context,),
        raise_on_error=True,
    )
    results = completed_results + [
        work_result.value for work_result in work_results
    ]
    results.sort(key=lambda result: result["outer_fold"])
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
