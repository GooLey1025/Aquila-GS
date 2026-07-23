#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Run leakage-free nested cross-validation on a prepared Aquila dataset."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from aquila.data.cv import load_fold_indices, parse_fold_selector
from aquila.data.dataset import PreparedData, load_prepared_data
from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.models.registry import create_model
from aquila.training.distributed import (
    FoldJob,
    FoldJobResult,
    PersistentGPUPool,
    derive_seed,
    detect_gpu_ids,
    execute_gpu_jobs,
    share_memory_tensors,
)
from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import (
    CandidateResult,
    HPOResult,
    InnerFoldResult,
    evaluate_candidate,
    generate_grid_candidates,
    merge_config,
    normalize_hpo_config,
    run_hpo,
    select_best_candidate,
)
from aquila.training.trainer import NestedCVTrainer
from aquila.utils import load_config


@dataclass(frozen=True)
class OuterFoldPayload:
    """Serializable inputs shared by one outer-fold worker."""

    prepared_data: PreparedData
    config: Dict[str, Any]
    output_directory: str
    gpu_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class HPOCandidateJob:
    """One grid-search candidate evaluated on a single GPU."""

    job_id: int
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class HPOCandidateContext:
    """Shared immutable inputs for within-fold multi-GPU HPO workers."""

    prepared_data: PreparedData
    config: Dict[str, Any]
    fold_id: int
    worker_seed: int
    regression_tasks: tuple[str, ...]
    classification_tasks: tuple[str, ...]
    inner_folds: tuple[int, ...]
    metric: str
    direction: str
    patience: int
    loader_options: Dict[str, Any]
    output_directory: str | None = None
    live_metrics_log: bool = False


@dataclass(frozen=True)
class NestedCVJob:
    """One GPU work item: a single inner fold or an outer final refit."""

    job_id: int
    kind: str  # "inner" | "final"
    fold_id: int
    candidate_id: int = -1
    inner_fold: int = -1
    parameters: Dict[str, Any] = field(default_factory=dict)
    final_epoch: int = 0
    hpo_result: HPOResult | None = None


@dataclass(frozen=True)
class NestedCVPoolContext:
    """Shared spawn-safe context for the keep-busy GPU pool."""

    prepared_data: PreparedData
    config: Dict[str, Any]
    output_directory: str
    global_seed: int
    regression_tasks: tuple[str, ...]
    classification_tasks: tuple[str, ...]
    metric: str
    direction: str
    patience: int
    loader_options: Dict[str, Any]
    live_metrics_log: bool = False


def _inner_job_id(fold_id: int, candidate_id: int, inner_fold: int) -> int:
    return int(fold_id) * 1_000_000 + int(candidate_id) * 1_000 + int(inner_fold)


def _final_job_id(fold_id: int) -> int:
    return int(fold_id) * 1_000_000 + 999_999


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Aquila models with prepared nested CV folds."
    )
    parser.add_argument("--data-dir", required=True, help="Prepared dataset directory.")
    parser.add_argument("--config", required=True, help="Nested-CV YAML configuration.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="experiments",
        help="Experiment output directory.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Zero-based outer folds to train; omission trains every fold.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        help="GPU IDs to use. Pass with no IDs to force CPU execution.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directories for selected folds.",
    )
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp32", "float32"),
        default="bf16",
        help="Training floating-point precision (default: bf16).",
    )
    parser.add_argument(
        "--live-metrics-log",
        action="store_true",
        help=(
            "Append per-epoch metrics JSONL under "
            "{output}/fold_*/candidate_*/inner_*/metrics.jsonl "
            "(overrides train.live_metrics_log=false in the config)."
        ),
    )
    return parser.parse_args()


def _sequence_lengths(features: Any) -> int | Dict[str, int]:
    if isinstance(features, dict):
        return {name: int(tensor.shape[1]) for name, tensor in features.items()}
    return int(features.shape[1])


def _trainer_kwargs(
    config: Mapping[str, Any],
    device: str,
    trait_names: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    train = config.get("train", {})
    scheduler_params = train.get("scheduler_params")
    return {
        "device": device,
        "learning_rate": float(train.get("learning_rate", 1e-4)),
        "weight_decay": float(train.get("weight_decay", 1e-5)),
        "loss_type": str(train.get("loss_type", "mse")),
        "uncertainty_weighting": bool(
            train.get("uncertainty_weighting", False)
        ),
        "huber_delta": float(train.get("huber_delta", 1.0)),
        "gradient_clip_norm": train.get("gradient_clip_norm", 1.0),
        "use_bf16": _use_bf16(train),
        "trait_names": list(trait_names),
        "scheduler_type": train.get("scheduler_type", "cosine_warmup"),
        "scheduler_params": (
            dict(scheduler_params) if isinstance(scheduler_params, Mapping) else {}
        ),
        "seed": int(seed),
    }


def _use_bf16(train: Mapping[str, Any]) -> bool:
    """Resolve AMP bf16 from ``train.precision`` (preferred) or legacy flag."""
    if "precision" in train:
        precision = str(train.get("precision", "bf16")).lower()
        if precision in {"bf16", "bfloat16"}:
            return True
        if precision in {"fp32", "float32", "fp16", "float16"}:
            return False
    # Legacy: mixed_precision true/false only when precision is absent.
    if "mixed_precision" in train:
        return bool(train["mixed_precision"])
    return True


def _loader_kwargs(train_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build DataLoader options for host-backed tensors.

    Nested-CV with ``gpu_resident`` (default) bypasses DataLoader workers via
    ``GpuResidentLoader``; this path is only used when tensors stay on CPU.
    """
    del train_config  # worker count is fixed; ignore user overrides
    return {
        "num_workers": 0,
        "pin_memory": False,
    }


def _task_lists(
    prepared_data: PreparedData,
) -> tuple[list[str], list[str]]:
    """Resolve regression/classification task names from prepared metadata."""
    metadata = prepared_data.metadata
    trait_names = list(metadata.get("trait_names", []))
    regression_tasks = metadata.get("regression_tasks")
    classification_tasks = metadata.get("classification_tasks")
    if isinstance(regression_tasks, list) and isinstance(
        classification_tasks, list
    ):
        return [str(name) for name in regression_tasks], [
            str(name) for name in classification_tasks
        ]
    return trait_names, []


def _make_model(
    config: Mapping[str, Any],
    prepared_data: PreparedData,
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str] | None = None,
) -> torch.nn.Module:
    model_config = config.get("model", {})
    return create_model(
        name=str(model_config.get("name", "aquila")),
        config=config,
        seq_length=_sequence_lengths(prepared_data.features),
        regression_tasks=list(regression_tasks),
        classification_tasks=list(classification_tasks or []),
    )


def _subset_features(features: Any, indices: np.ndarray) -> Any:
    if isinstance(features, dict):
        return {
            name: tensor[indices].contiguous()
            for name, tensor in features.items()
        }
    return features[indices].contiguous()


def _processed_view(
    prepared_data: PreparedData,
    indices: np.ndarray,
    target_path: Path,
) -> PreparedData:
    processed_targets = torch.load(
        target_path,
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(processed_targets, torch.Tensor)
        or processed_targets.ndim != 2
        or processed_targets.shape[0] != len(indices)
        or processed_targets.shape[1] != prepared_data.targets.shape[1]
    ):
        raise ValueError(f"Invalid processed target cache: {target_path}")
    metadata = copy.deepcopy(prepared_data.metadata)
    metadata["sample_ids"] = [
        prepared_data.metadata["sample_ids"][int(index)]
        for index in indices
    ]
    return PreparedData(
        features=_subset_features(prepared_data.features, indices),
        targets=processed_targets,
        target_mask=prepared_data.target_mask[indices].contiguous(),
        metadata=metadata,
        directory=prepared_data.directory,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _candidate_records(result: HPOResult) -> list[Dict[str, Any]]:
    records = []
    for candidate in result.candidates:
        records.append(
            {
                "candidate_id": candidate.candidate_id,
                "parameters": dict(candidate.parameters),
                "objective": candidate.objective,
                "best_epochs": list(candidate.best_epochs),
                "final_epoch": candidate.final_epoch,
                "inner_results": [
                    {
                        "inner_fold": inner.inner_fold,
                        "metric": inner.metric,
                        "best_epoch": inner.best_epoch,
                        "metrics": dict(inner.metrics),
                    }
                    for inner in candidate.inner_results
                ],
            }
        )
    return records


def _validate_preprocessing_cache(
    prepared: PreparedData,
    config: Mapping[str, Any],
) -> None:
    """Ensure training uses the preprocessing policy used to build caches."""
    cached = prepared.metadata.get("preprocessing", {})
    requested = config.get("preprocessing", {})
    cached_skew = float(cached.get("skew_threshold", 2.0))
    requested_skew = float(requested.get("skew_threshold", 2.0))
    cached_epsilon = float(cached.get("epsilon", 1e-8))
    requested_epsilon = float(requested.get("epsilon", 1e-8))
    if not np.isclose(cached_skew, requested_skew) or not np.isclose(
        cached_epsilon,
        requested_epsilon,
    ):
        raise ValueError(
            "Training preprocessing configuration does not match prepared "
            "target caches. Re-run aquila_data_cv.py with the requested "
            "skew_threshold and preprocessing epsilon."
        )


def _tensor_tree_to_device(value: Any, device: str | torch.device) -> Any:
    """Move nested CPU tensors to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {
            key: _tensor_tree_to_device(item, device)
            for key, item in value.items()
        }
    return value


def _prepared_on_device(prepared: PreparedData, device: str) -> PreparedData:
    """Return a PreparedData view with tensors resident on ``device``."""
    return PreparedData(
        features=_tensor_tree_to_device(prepared.features, device),
        targets=prepared.targets.to(device, non_blocking=True),
        target_mask=prepared.target_mask.to(device, non_blocking=True),
        metadata=prepared.metadata,
        directory=prepared.directory,
    )


def _train_inner_fold(
    *,
    prepared: PreparedData,
    config: Mapping[str, Any],
    fold_id: int,
    inner_fold: int,
    candidate_id: int,
    parameters: Mapping[str, Any],
    device: str,
    worker_seed: int,
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str],
    metric: str,
    direction: str,
    patience: int,
    loader_options: Mapping[str, Any],
    metrics_log_path: str | Path | None = None,
):
    """Train one inner fold for one HPO candidate on ``device``."""
    candidate_config = merge_config(config, parameters)
    split = load_fold_indices(prepared.directory, fold_id, inner_fold)
    inner_path = (
        prepared.directory
        / "cv"
        / f"outer_fold_{fold_id}"
        / f"inner_fold_{inner_fold}"
    )
    train_data = _processed_view(
        prepared,
        split["train"],
        inner_path / "Y_train_processed.pt",
    )
    valid_data = _processed_view(
        prepared,
        split["valid"],
        inner_path / "Y_valid_processed.pt",
    )
    train_cfg = candidate_config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    # Small prepared tensors fit in GPU memory; keep them device-resident so
    # training is not gated by host DataLoader workers / H2D copies.
    gpu_resident = bool(train_cfg.get("gpu_resident", True)) and str(
        device
    ).startswith("cuda")
    effective_loader_options = dict(loader_options)
    if gpu_resident:
        train_data = _prepared_on_device(train_data, device)
        valid_data = _prepared_on_device(valid_data, device)
        effective_loader_options = {
            "num_workers": 0,
            "pin_memory": False,
        }
    train_loader = train_data.loader(
        np.arange(len(split["train"])),
        batch_size=batch_size,
        shuffle=True,
        **effective_loader_options,
    )
    valid_loader = valid_data.loader(
        np.arange(len(split["valid"])),
        batch_size=batch_size,
        shuffle=False,
        **effective_loader_options,
    )
    inner_seed = derive_seed(worker_seed, fold_id, candidate_id, inner_fold)
    trainer = NestedCVTrainer(
        _make_model(
            candidate_config,
            prepared,
            regression_tasks,
            classification_tasks,
        ),
        num_regression_tasks=len(regression_tasks),
        num_classification_tasks=len(classification_tasks),
        **_trainer_kwargs(
            candidate_config,
            device,
            regression_tasks,
            inner_seed,
        ),
    )
    return trainer.train_inner(
        train_loader,
        valid_loader,
        max_epochs=int(train_cfg.get("num_epochs", 300)),
        patience=int(train_cfg.get("early_stopping_patience", patience)),
        metric=metric,
        direction=direction,
        min_delta=float(train_cfg.get("early_stopping_min_delta", 1e-4)),
        metrics_log_path=metrics_log_path,
    )


def _evaluate_hpo_candidate_job(
    job: HPOCandidateJob,
    device: str,
    context: HPOCandidateContext,
) -> CandidateResult:
    """Spawn-safe worker: evaluate one grid candidate on all inner folds."""
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":", 1)[1]))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def run_inner(
        parameters: Mapping[str, Any],
        inner_fold: int,
        candidate_id: int,
    ):
        metrics_log_path = None
        if context.live_metrics_log and context.output_directory:
            metrics_log_path = (
                Path(context.output_directory)
                / f"fold_{context.fold_id}"
                / f"candidate_{candidate_id}"
                / f"inner_{inner_fold}"
                / "metrics.jsonl"
            )
        return _train_inner_fold(
            prepared=context.prepared_data,
            config=context.config,
            fold_id=context.fold_id,
            inner_fold=inner_fold,
            candidate_id=candidate_id,
            parameters=parameters,
            device=device,
            worker_seed=context.worker_seed,
            regression_tasks=context.regression_tasks,
            classification_tasks=context.classification_tasks,
            metric=context.metric,
            direction=context.direction,
            patience=context.patience,
            loader_options=context.loader_options,
            metrics_log_path=metrics_log_path,
        )

    return evaluate_candidate(
        job.job_id,
        job.parameters,
        context.inner_folds,
        run_inner,
        metric=context.metric,
    )


def _run_grid_hpo_on_gpus(
    hpo_config: Mapping[str, Any],
    context: HPOCandidateContext,
    gpu_ids: Sequence[int],
) -> HPOResult:
    """Evaluate grid candidates in parallel across GPUs."""
    normalized = normalize_hpo_config(hpo_config)
    parameter_sets = generate_grid_candidates(normalized["parameters"])
    if not parameter_sets:
        raise ValueError("Grid HPO parameter space is empty")
    print(
        f"[fold {context.fold_id}] Grid HPO: {len(parameter_sets)} candidates "
        f"across GPUs {list(gpu_ids)}"
    )
    jobs = [
        HPOCandidateJob(job_id=index, parameters=parameters)
        for index, parameters in enumerate(parameter_sets)
    ]
    # This prepared dataset is already memory-resident and small. Benchmarks on
    # the 705-sample workload show that nested DataLoader workers add IPC and
    # collation overhead, so each GPU worker batches directly in-process.
    worker_loader_options = dict(context.loader_options)
    worker_loader_options["num_workers"] = 0
    worker_loader_options.pop("persistent_workers", None)
    worker_loader_options.pop("prefetch_factor", None)
    worker_context = HPOCandidateContext(
        prepared_data=context.prepared_data,
        config=context.config,
        fold_id=context.fold_id,
        worker_seed=context.worker_seed,
        regression_tasks=context.regression_tasks,
        classification_tasks=context.classification_tasks,
        inner_folds=context.inner_folds,
        metric=context.metric,
        direction=context.direction,
        patience=context.patience,
        loader_options=worker_loader_options,
        output_directory=context.output_directory,
        live_metrics_log=context.live_metrics_log,
    )
    work_results = execute_gpu_jobs(
        jobs,
        _evaluate_hpo_candidate_job,
        gpu_ids,
        worker_args=(worker_context,),
        raise_on_error=True,
    )
    candidates = [result.value for result in work_results]
    return select_best_candidate(
        candidates,
        normalized["direction"],
        method="grid",
    )


def _configure_worker_device(device: str) -> None:
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":", 1)[1]))
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def _nested_cv_gpu_worker(
    job: NestedCVJob,
    device: str,
    context: NestedCVPoolContext,
) -> Dict[str, Any]:
    """Spawn-safe worker for one inner fold or one outer final refit."""
    _configure_worker_device(device)
    worker_seed = derive_seed(context.global_seed, int(job.fold_id))
    if job.kind == "inner":
        metrics_log_path = None
        if context.live_metrics_log:
            metrics_log_path = (
                Path(context.output_directory)
                / f"fold_{job.fold_id}"
                / f"candidate_{job.candidate_id}"
                / f"inner_{job.inner_fold}"
                / "metrics.jsonl"
            )
        training = _train_inner_fold(
            prepared=context.prepared_data,
            config=context.config,
            fold_id=int(job.fold_id),
            inner_fold=int(job.inner_fold),
            candidate_id=int(job.candidate_id),
            parameters=job.parameters,
            device=device,
            worker_seed=worker_seed,
            regression_tasks=context.regression_tasks,
            classification_tasks=context.classification_tasks,
            metric=context.metric,
            direction=context.direction,
            patience=context.patience,
            loader_options=context.loader_options,
            metrics_log_path=metrics_log_path,
        )
        return {
            "kind": "inner",
            "fold_id": int(job.fold_id),
            "candidate_id": int(job.candidate_id),
            "inner_fold": int(job.inner_fold),
            "best_epoch": int(training.best_epoch),
            "best_metrics": dict(training.best_metrics),
        }
    if job.kind == "final":
        if job.hpo_result is None:
            raise ValueError("Final NestedCVJob requires hpo_result")
        return _finalize_outer_fold(
            prepared=context.prepared_data,
            config=context.config,
            fold_id=int(job.fold_id),
            output_directory=Path(context.output_directory),
            device=device,
            worker_seed=worker_seed,
            hpo_result=job.hpo_result,
            regression_tasks=context.regression_tasks,
            classification_tasks=context.classification_tasks,
            loader_options=context.loader_options,
        )
    raise ValueError(f"Unknown NestedCVJob kind: {job.kind!r}")


def _build_candidate_result(
    candidate_id: int,
    parameters: Mapping[str, Any],
    inner_by_fold: Mapping[int, Mapping[str, Any]],
    metric: str,
) -> CandidateResult:
    """Aggregate completed inner-fold worker payloads into one candidate."""
    inner_results = []
    for inner_fold in sorted(inner_by_fold):
        payload = inner_by_fold[inner_fold]
        metrics = dict(payload["best_metrics"])
        # Prefer avg_pearson; fall back to metric path aliases used by HPO yaml.
        if "avg_pearson" in metrics:
            score = float(metrics["avg_pearson"])
        else:
            score = float(metrics.get(metric.replace("/", ".").split(".")[-1], float("nan")))
        inner_results.append(
            InnerFoldResult(
                int(inner_fold),
                score,
                int(payload["best_epoch"]),
                metrics,
            )
        )
    if not inner_results:
        raise ValueError(f"Candidate {candidate_id} has no inner results")
    values = np.asarray([item.metric for item in inner_results], dtype=float)
    objective = (
        float("nan")
        if not np.all(np.isfinite(values))
        else float(np.mean(values))
    )
    return CandidateResult(
        candidate_id=int(candidate_id),
        parameters=copy.deepcopy(dict(parameters)),
        objective=objective,
        inner_results=tuple(inner_results),
    )


def _run_pipelined_grid_cv(
    *,
    prepared: PreparedData,
    config: Mapping[str, Any],
    output_directory: Path,
    selected_folds: Sequence[int],
    gpu_ids: Sequence[int],
    global_seed: int,
) -> list[Dict[str, Any]]:
    """Keep GPUs busy: inner-fold jobs across folds, then per-fold final refits."""
    train_config = config.get("train", {})
    hpo_config = config.get("hpo", {})
    normalized_hpo = normalize_hpo_config(hpo_config)
    if normalized_hpo["method"] != "grid":
        raise ValueError("Pipelined scheduler currently supports grid HPO only")
    parameter_sets = generate_grid_candidates(normalized_hpo["parameters"])
    if not parameter_sets:
        raise ValueError("Grid HPO parameter space is empty")

    regression_tasks, classification_tasks = _task_lists(prepared)
    if not regression_tasks:
        raise ValueError(
            "Prepared data contains no regression tasks; nested CV HPO "
            "currently requires at least one regression trait."
        )
    inner_count = int(prepared.metadata["inner_folds"])
    patience = int(train_config.get("early_stopping_patience", 20))
    loader_options = _loader_kwargs(train_config)
    live_metrics_log = bool(train_config.get("live_metrics_log", False))
    metric = str(normalized_hpo["metric"])
    direction = str(normalized_hpo["direction"])

    for fold_id in selected_folds:
        (output_directory / f"fold_{fold_id}").mkdir(parents=True, exist_ok=True)

    pool_context = NestedCVPoolContext(
        prepared_data=prepared,
        config=dict(config),
        output_directory=str(output_directory),
        global_seed=int(global_seed),
        regression_tasks=tuple(regression_tasks),
        classification_tasks=tuple(classification_tasks),
        metric=metric,
        direction=direction,
        patience=patience,
        loader_options=loader_options,
        live_metrics_log=live_metrics_log,
    )

    inner_jobs: list[NestedCVJob] = []
    expected_inners = {
        int(fold_id): inner_count * len(parameter_sets)
        for fold_id in selected_folds
    }
    for fold_id in selected_folds:
        for candidate_id, parameters in enumerate(parameter_sets):
            for inner_fold in range(inner_count):
                inner_jobs.append(
                    NestedCVJob(
                        job_id=_inner_job_id(fold_id, candidate_id, inner_fold),
                        kind="inner",
                        fold_id=int(fold_id),
                        candidate_id=int(candidate_id),
                        inner_fold=int(inner_fold),
                        parameters=dict(parameters),
                    )
                )

    print(
        f"[INFO] Persistent GPU pool on {list(gpu_ids)}; "
        f"{len(inner_jobs)} inner-fold jobs across folds {list(selected_folds)}; "
        "final refit queued per fold when that fold's HPO completes"
    )

    # fold_id -> candidate_id -> inner_fold -> payload
    inners: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]] = {
        int(fold_id): {} for fold_id in selected_folds
    }
    completed_inners = {int(fold_id): 0 for fold_id in selected_folds}
    final_submitted = {int(fold_id): False for fold_id in selected_folds}
    fold_summaries: Dict[int, Dict[str, Any]] = {}

    with PersistentGPUPool(
        gpu_ids,
        _nested_cv_gpu_worker,
        worker_args=(pool_context,),
    ) as pool:
        pool.submit_many(inner_jobs)
        while len(fold_summaries) < len(selected_folds):
            work = pool.get()
            if not work.succeeded:
                raise RuntimeError(
                    f"Nested CV GPU job {work.job_id} failed on {work.device}: "
                    f"{work.error}\n{work.traceback or ''}"
                )
            payload = work.value
            kind = payload["kind"]
            fold_id = int(payload["fold_id"])
            if kind == "inner":
                candidate_id = int(payload["candidate_id"])
                inner_fold = int(payload["inner_fold"])
                inners[fold_id].setdefault(candidate_id, {})[inner_fold] = payload
                completed_inners[fold_id] += 1
                if (
                    completed_inners[fold_id] == expected_inners[fold_id]
                    and not final_submitted[fold_id]
                ):
                    candidates = [
                        _build_candidate_result(
                            candidate_id,
                            parameter_sets[candidate_id],
                            inners[fold_id][candidate_id],
                            metric,
                        )
                        for candidate_id in range(len(parameter_sets))
                    ]
                    hpo_result = select_best_candidate(
                        candidates,
                        direction,
                        method="grid",
                    )
                    print(
                        f"[fold {fold_id}] HPO complete "
                        f"(best candidate={hpo_result.best.candidate_id}, "
                        f"final_epoch={hpo_result.best.final_epoch}); "
                        "queueing final refit"
                    )
                    pool.submit(
                        NestedCVJob(
                            job_id=_final_job_id(fold_id),
                            kind="final",
                            fold_id=fold_id,
                            candidate_id=int(hpo_result.best.candidate_id),
                            parameters=dict(hpo_result.best.parameters),
                            final_epoch=int(hpo_result.best.final_epoch),
                            hpo_result=hpo_result,
                        )
                    )
                    final_submitted[fold_id] = True
            elif kind == "final":
                fold_summaries[fold_id] = payload["summary"]
            else:
                raise RuntimeError(f"Unexpected worker payload kind: {kind!r}")

    return [fold_summaries[int(fold_id)] for fold_id in selected_folds]


def _finalize_outer_fold(
    *,
    prepared: PreparedData,
    config: Mapping[str, Any],
    fold_id: int,
    output_directory: Path,
    device: str,
    worker_seed: int,
    hpo_result: HPOResult,
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str],
    loader_options: Mapping[str, Any],
    started: float | None = None,
) -> Dict[str, Any]:
    """Refit on outer train with selected HPO, evaluate test, write fold artifacts."""
    if started is None:
        started = time.time()
    fold_directory = Path(output_directory) / f"fold_{fold_id}"
    fold_directory.mkdir(parents=True, exist_ok=True)
    trait_names = list(prepared.metadata["trait_names"])
    outer = load_fold_indices(prepared.directory, fold_id)
    selected_config = merge_config(config, hpo_result.best.parameters)
    final_epoch = hpo_result.best.final_epoch
    final_path = (
        prepared.directory
        / "cv"
        / f"outer_fold_{fold_id}"
        / "final"
    )
    final_processor = PerTraitPreprocessor.load_json(
        final_path / "preprocessing.json"
    )
    final_train_data = _processed_view(
        prepared,
        outer["train"],
        final_path / "Y_train_processed.pt",
    )
    final_test_data = _processed_view(
        prepared,
        outer["test"],
        final_path / "Y_test_processed.pt",
    )
    final_batch_size = int(
        selected_config.get("train", {}).get("batch_size", 32)
    )
    final_loader_options = dict(loader_options)
    if bool(selected_config.get("train", {}).get("gpu_resident", True)) and str(
        device
    ).startswith("cuda"):
        final_train_data = _prepared_on_device(final_train_data, device)
        final_test_data = _prepared_on_device(final_test_data, device)
        final_loader_options = {"num_workers": 0, "pin_memory": False}
    final_train_loader = final_train_data.loader(
        np.arange(len(outer["train"])),
        batch_size=final_batch_size,
        shuffle=True,
        **final_loader_options,
    )
    outer_test_loader = final_test_data.loader(
        np.arange(len(outer["test"])),
        batch_size=final_batch_size,
        shuffle=False,
        **final_loader_options,
    )
    final_seed = derive_seed(worker_seed, fold_id, hpo_result.best.candidate_id)
    final_trainer = NestedCVTrainer(
        _make_model(
            selected_config,
            prepared,
            regression_tasks,
            classification_tasks,
        ),
        num_regression_tasks=len(regression_tasks),
        num_classification_tasks=len(classification_tasks),
        **_trainer_kwargs(
            selected_config,
            device,
            regression_tasks,
            final_seed,
        ),
    )
    final_training = final_trainer.train_fixed_epochs(
        final_train_loader,
        epochs=final_epoch,
    )
    normalized = final_trainer.evaluate(outer_test_loader)
    n_regression = len(regression_tasks)
    prediction_mask = np.ones_like(normalized.predictions, dtype=bool)
    full_predictions = np.zeros(
        (normalized.predictions.shape[0], len(trait_names)),
        dtype=np.float64,
    )
    full_mask = np.zeros_like(full_predictions, dtype=bool)
    full_predictions[:, :n_regression] = normalized.predictions
    full_mask[:, :n_regression] = prediction_mask
    inverted = final_processor.inverse(full_predictions, full_mask)
    original_predictions = np.asarray(inverted)[:, :n_regression]
    raw_targets = prepared.targets[outer["test"]].cpu().numpy()[:, :n_regression]
    raw_mask = prepared.target_mask[outer["test"]].cpu().numpy()[:, :n_regression]
    original = evaluate_regression(
        original_predictions,
        raw_targets,
        raw_mask,
        regression_tasks,
    )

    selected_config.setdefault("data", {})
    selected_config["data"].update(
        {
            "encoding_type": prepared.metadata["encoding_type"],
            "variant_type": prepared.metadata.get("variant_type"),
            "regression_tasks": list(regression_tasks),
            "classification_tasks": list(classification_tasks),
        }
    )
    checkpoint_metadata = copy.deepcopy(prepared.metadata)
    checkpoint_metadata["sequence_lengths"] = _sequence_lengths(
        prepared.features
    )
    checkpoint = {
        **final_training.checkpoint_state,
        "config": selected_config,
        "metadata": checkpoint_metadata,
        "preprocessing": final_processor.to_dict(),
        "outer_fold": fold_id,
        "selected_hyperparameters": dict(hpo_result.best.parameters),
        "final_epoch": final_epoch,
    }
    torch.save(checkpoint, fold_directory / "best_model.pt")
    final_processor.save_json(fold_directory / "preprocessing.json")
    with (fold_directory / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(selected_config, handle, sort_keys=False)
    _write_json(
        fold_directory / "hpo_results.json",
        {
            "method": hpo_result.method,
            "direction": hpo_result.direction,
            "best_candidate_id": hpo_result.best.candidate_id,
            "best_parameters": dict(hpo_result.best.parameters),
            "best_valid_r_mean": hpo_result.best.objective,
            "final_epoch": final_epoch,
            "candidates": _candidate_records(hpo_result),
        },
    )
    metrics = {
        "normalized": normalized.metrics,
        "original": original.metrics,
    }
    _write_json(fold_directory / "metrics.json", metrics)
    _write_json(
        fold_directory / "training_history.json",
        final_training.history,
    )

    prediction_values: Dict[str, Any] = {
        "SampleID": [
            prepared.metadata["sample_ids"][int(index)]
            for index in outer["test"]
        ]
    }
    for trait_index, trait_name in enumerate(trait_names):
        prediction_values[f"{trait_name}_prediction"] = original_predictions[
            :, trait_index
        ]
        prediction_values[f"{trait_name}_observed"] = np.where(
            raw_mask[:, trait_index],
            raw_targets[:, trait_index],
            np.nan,
        )
    pd.DataFrame(prediction_values).to_csv(
        fold_directory / "predictions_original_scale.csv",
        index=False,
    )

    runtime = {
        "outer_fold": fold_id,
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda")
            else None
        ),
        "worker_seed": worker_seed,
        "final_seed": final_seed,
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    _write_json(fold_directory / "runtime.json", runtime)
    summary = {
        "outer_fold": fold_id,
        "status": "completed",
        "best_valid_r_mean": hpo_result.best.objective,
        "selected_hyperparameters": dict(hpo_result.best.parameters),
        "inner_best_epochs": list(hpo_result.best.best_epochs),
        "final_epoch": final_epoch,
        "metrics": metrics,
        "runtime": runtime,
    }
    return {"kind": "final", "fold_id": int(fold_id), "summary": summary}


def execute_outer_fold(job: FoldJob, device: str, worker_seed: int) -> Dict[str, Any]:
    """Train, select, refit, and evaluate one outer fold."""
    payload: OuterFoldPayload = job.payload
    prepared = payload.prepared_data
    config = copy.deepcopy(payload.config)
    fold_id = int(job.fold_id)
    fold_directory = Path(payload.output_directory) / f"fold_{fold_id}"
    fold_directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    regression_tasks, classification_tasks = _task_lists(prepared)
    if not regression_tasks:
        raise ValueError(
            "Prepared data contains no regression tasks; nested CV HPO "
            "currently requires at least one regression trait."
        )
    inner_count = int(prepared.metadata["inner_folds"])
    train_config = config.get("train", {})
    hpo_config = config.get("hpo", {})
    metric = str(hpo_config.get("metric", "avg_pearson"))
    direction = str(hpo_config.get("direction", "maximize"))
    patience = int(train_config.get("early_stopping_patience", 20))
    loader_options = _loader_kwargs(train_config)
    hpo_gpu_ids = list(payload.gpu_ids)
    if not hpo_gpu_ids and device.startswith("cuda:"):
        hpo_gpu_ids = [int(device.split(":", 1)[1])]
    normalized_hpo = normalize_hpo_config(hpo_config)
    parallel_grid = (
        normalized_hpo["method"] == "grid" and len(hpo_gpu_ids) > 1
    )
    # Do not create a parent CUDA context before multi-GPU HPO. An idle parent
    # context on cuda:0 otherwise depresses GPU-0 utilization in nvidia-smi.
    if device.startswith("cuda") and not parallel_grid:
        torch.cuda.set_device(int(device.split(":", 1)[1]))
        torch.backends.cudnn.benchmark = True
    live_metrics_log = bool(train_config.get("live_metrics_log", False))
    candidate_context = HPOCandidateContext(
        prepared_data=prepared,
        config=config,
        fold_id=fold_id,
        worker_seed=worker_seed,
        regression_tasks=tuple(regression_tasks),
        classification_tasks=tuple(classification_tasks),
        inner_folds=tuple(range(inner_count)),
        metric=str(normalized_hpo["metric"]),
        direction=str(normalized_hpo["direction"]),
        patience=patience,
        loader_options=loader_options,
        output_directory=str(payload.output_directory),
        live_metrics_log=live_metrics_log,
    )
    if parallel_grid:
        hpo_result = _run_grid_hpo_on_gpus(
            hpo_config,
            candidate_context,
            hpo_gpu_ids,
        )
        if device.startswith("cuda"):
            torch.cuda.set_device(int(device.split(":", 1)[1]))
            torch.backends.cudnn.benchmark = True
    else:
        def run_inner(
            parameters: Mapping[str, Any],
            inner_fold: int,
            candidate_id: int,
        ):
            metrics_log_path = None
            if live_metrics_log:
                metrics_log_path = (
                    fold_directory
                    / f"candidate_{candidate_id}"
                    / f"inner_{inner_fold}"
                    / "metrics.jsonl"
                )
            return _train_inner_fold(
                prepared=prepared,
                config=config,
                fold_id=fold_id,
                inner_fold=inner_fold,
                candidate_id=candidate_id,
                parameters=parameters,
                device=device,
                worker_seed=worker_seed,
                regression_tasks=regression_tasks,
                classification_tasks=classification_tasks,
                metric=metric,
                direction=direction,
                patience=patience,
                loader_options=loader_options,
                metrics_log_path=metrics_log_path,
            )

        hpo_result = run_hpo(
            hpo_config,
            list(range(inner_count)),
            run_inner,
        )
    finalized = _finalize_outer_fold(
        prepared=prepared,
        config=config,
        fold_id=fold_id,
        output_directory=Path(payload.output_directory),
        device=device,
        worker_seed=worker_seed,
        hpo_result=hpo_result,
        regression_tasks=regression_tasks,
        classification_tasks=classification_tasks,
        loader_options=loader_options,
        started=started,
    )
    return finalized["summary"]


def _aggregate_outer_test_metrics(
    completed: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Mean/std of outer-test avg_pearson on the normalized (training) scale.

    Pearson is never aggregated from inverse-transformed / original-scale
    predictions here; those remain available under each fold's metrics.json.
    """
    values: list[float] = []
    for item in completed:
        scale_metrics = item.get("metrics", {}).get("normalized", {})
        value = scale_metrics.get("avg_pearson")
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "scale": "normalized",
        "n_folds": int(array.size),
        "test_pearsonr_per_fold": values,
        "test_pearsonr_mean": float(np.mean(array)),
        "test_pearsonr_std": (
            float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        ),
    }


def _selected_per_fold(
    completed: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Compact per-fold HPO selection (hyperparams + final epoch)."""
    rows: list[Dict[str, Any]] = []
    for item in completed:
        rows.append(
            {
                "outer_fold": int(item["outer_fold"]),
                "selected_hyperparameters": dict(
                    item.get("selected_hyperparameters") or {}
                ),
                "final_epoch": int(item["final_epoch"]),
                "inner_best_epochs": [
                    int(epoch) for epoch in item.get("inner_best_epochs") or []
                ],
                "best_valid_r_mean": float(
                    item.get(
                        "best_valid_r_mean",
                        item.get("best_valid_r", item.get("best_objective")),
                    )
                ),
            }
        )
    return rows


def _prepare_output(
    output_directory: Path,
    selected_folds: Sequence[int],
    overwrite: bool,
) -> None:
    import shutil

    output_directory.mkdir(parents=True, exist_ok=True)
    for fold_id in selected_folds:
        fold_directory = output_directory / f"fold_{fold_id}"
        if fold_directory.exists() and any(fold_directory.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Fold output already exists: {fold_directory}; "
                    "use --overwrite to replace it"
                )
            shutil.rmtree(fold_directory)


def _copy_params_yaml(config_path: Path, output_directory: Path) -> Path:
    """Copy the training params YAML into the run output directory."""
    import shutil

    source = Path(config_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Config not found: {source}")
    destination = Path(output_directory) / source.name
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    args = parse_args()
    prepared = load_prepared_data(args.data_dir)
    config = load_config(args.config)
    config.setdefault("train", {})
    config["train"]["precision"] = args.precision
    if args.live_metrics_log:
        config["train"]["live_metrics_log"] = True
    if "mixed_precision" in config["train"]:
        # Prefer explicit precision; drop redundant legacy key.
        config["train"].pop("mixed_precision", None)
    _validate_preprocessing_cache(prepared, config)
    fold_count = int(prepared.metadata["outer_folds"])
    selected_folds = (
        list(range(fold_count))
        if args.folds is None
        else parse_fold_selector(args.folds, fold_count)
    )
    output_directory = Path(args.output_dir)
    _prepare_output(output_directory, selected_folds, args.overwrite)
    params_copy = _copy_params_yaml(Path(args.config), output_directory)

    prepared = PreparedData(
        features=share_memory_tensors(prepared.features),
        targets=prepared.targets.share_memory_(),
        target_mask=prepared.target_mask.share_memory_(),
        metadata=prepared.metadata,
        directory=prepared.directory,
    )
    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    global_seed = int(prepared.metadata["seed"])
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
    normalized_hpo = normalize_hpo_config(config.get("hpo", {}))
    use_pipeline = bool(gpu_ids) and normalized_hpo["method"] == "grid"

    results: list[FoldJobResult] = []
    if use_pipeline:
        print(
            f"[INFO] Persistent GPU pool on {gpu_ids}; "
            "inner-fold jobs pipelined across outer folds; "
            "final refit queued per fold when HPO completes"
        )
        try:
            completed = _run_pipelined_grid_cv(
                prepared=prepared,
                config=config,
                output_directory=output_directory,
                selected_folds=selected_folds,
                gpu_ids=gpu_ids,
                global_seed=global_seed,
            )
            for item in completed:
                fold_id = int(item["outer_fold"])
                results.append(
                    FoldJobResult(
                        fold_id=fold_id,
                        value=item,
                        device="pool",
                        worker_seed=derive_seed(global_seed, fold_id),
                    )
                )
        except Exception as error:
            import traceback

            results.append(
                FoldJobResult(
                    fold_id=-1,
                    device="pool",
                    worker_seed=global_seed,
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            )
    else:
        print(
            f"[INFO] Outer folds run serially; within-fold HPO uses GPUs "
            f"{gpu_ids or ['cpu']} (final refit on {device})"
        )
        for fold_id in selected_folds:
            worker_seed = derive_seed(global_seed, int(fold_id))
            try:
                value = execute_outer_fold(
                    FoldJob(
                        fold_id=int(fold_id),
                        payload=OuterFoldPayload(
                            prepared_data=prepared,
                            config=config,
                            output_directory=str(output_directory),
                            gpu_ids=tuple(gpu_ids),
                        ),
                    ),
                    device,
                    worker_seed,
                )
                results.append(
                    FoldJobResult(
                        fold_id=int(fold_id),
                        value=value,
                        device=device,
                        worker_seed=worker_seed,
                    )
                )
            except Exception as error:
                import traceback

                results.append(
                    FoldJobResult(
                        fold_id=int(fold_id),
                        device=device,
                        worker_seed=worker_seed,
                        error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                    )
                )
    completed = [result.value for result in results if result.succeeded]
    failed = [
        {
            "outer_fold": result.fold_id,
            "error": result.error,
            "traceback": result.traceback,
        }
        for result in results
        if not result.succeeded
    ]
    # Top-level summary: aggregates + compact HPO selections (no full metrics).
    outer_test_summary = _aggregate_outer_test_metrics(completed)
    selected_per_fold = _selected_per_fold(completed)
    summary: Dict[str, Any] = {
        "requested_folds": list(selected_folds),
        "completed_folds": [item["outer_fold"] for item in completed],
        "failed_folds": [item["outer_fold"] for item in failed],
        "selected_per_fold": selected_per_fold,
        "outer_test_summary": outer_test_summary,
        "data_dir": str(Path(args.data_dir).resolve()),
        "config": str(Path(args.config).resolve()),
        "params_yaml": str(params_copy.resolve()),
    }
    if failed:
        summary["failures"] = failed
    _write_json(output_directory / "summary.json", summary)
    if failed:
        raise RuntimeError(
            "Nested CV failed for folds: "
            + ", ".join(str(item["outer_fold"]) for item in failed)
        )
    print(
        f"Completed nested CV for folds {selected_folds}; "
        f"summary: {output_directory / 'summary.json'}"
    )
    if outer_test_summary:
        print(
            "[outer-test normalized] "
            f"test_pearsonr_mean={outer_test_summary['test_pearsonr_mean']:.6f} "
            f"test_pearsonr_std={outer_test_summary['test_pearsonr_std']:.6f} "
            f"(n={outer_test_summary['n_folds']})"
        )


if __name__ == "__main__":
    main()
