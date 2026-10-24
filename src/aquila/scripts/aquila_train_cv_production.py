#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Run per-fold production HPO or legacy K-fold HPO with full-data refit."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import yaml

from aquila.data.dataset import PreparedData, load_prepared_data
from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.scripts.aquila_data_cv_production import (
    DeploymentFold,
    validate_deployment_folds,
)
from aquila.scripts.aquila_train_cv import (
    _candidate_records,
    _json_safe,
    _loader_kwargs,
    _make_seeded_model,
    _prepared_on_device,
    _sequence_lengths,
    _task_lists,
    _trainer_kwargs,
    _validate_preprocessing_cache,
)
from aquila.training.cuda_runtime import (
    configure_cuda_runtime,
    resolve_train_deterministic,
    train_deterministic_enabled,
)
from aquila.training.distributed import (
    PersistentGPUPool,
    detect_gpu_ids,
    execute_gpu_jobs,
    share_memory_tensors,
)
from aquila.training.hpo import (
    CandidateResult,
    HPOResult,
    InnerFoldResult,
    evaluate_candidate,
    finite_parameter_candidates,
    generate_grid_candidates,
    merge_config,
    normalize_hpo_config,
    parameter_key,
    run_hpo,
    select_best_candidate,
    unique_bayesian_trial,
)
from aquila.training.trainer import (
    NestedCVTrainer,
    TrainingResult,
    resolve_training_seed,
)
from aquila.utils import load_config


@dataclass(frozen=True)
class DeploymentCandidateJob:
    """One grid-search candidate evaluated across every deployment CV fold."""

    job_id: int
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class DeploymentFoldJob:
    """One Bayesian trial evaluated on one deployment validation fold."""

    job_id: int
    candidate_id: int
    fold_id: int
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class PerFoldHPOJob:
    """One complete, independent Bayesian study for a deployment fold."""

    job_id: int
    fold_id: int


@dataclass(frozen=True)
class DeploymentHPOContext:
    """Spawn-safe inputs shared by multi-GPU deployment HPO workers."""

    prepared_data: PreparedData
    config: Dict[str, Any]
    folds: tuple[DeploymentFold, ...]
    regression_tasks: tuple[str, ...]
    classification_tasks: tuple[str, ...]
    metric: str
    direction: str
    patience: int
    loader_options: Dict[str, Any]
    output_directory: str
    live_metrics_log: bool = False
    resume: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independent Bayesian HPO on train/validation folds, or use "
            "the legacy cross-fold HPO plus full-data refit workflow."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Prepared dataset directory.")
    parser.add_argument("--config", required=True, help="Training/HPO YAML configuration.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="production_experiment",
        help="Production run output directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("per-fold", "cv-refit"),
        default="cv-refit",
        help=(
            "cv-refit preserves the legacy cross-fold HPO and full-data refit "
            "workflow (default); per-fold runs an independent study on each "
            "selected train/valid split and saves its early-stopping checkpoint."
        ),
    )
    parser.add_argument(
        "--fold",
        "--folds",
        dest="folds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "In per-fold mode, run these zero-based prepared folds, for "
            "example --fold 0 1 2 3. Omit to run every fold."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            "Override hpo.n_trials. Per-fold mode defaults to 500 when this "
            "option is omitted; cv-refit otherwise keeps the YAML value."
        ),
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
        help="Replace an existing nonempty production output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume per-fold Bayesian studies from output-dir SQLite storage. "
            "Completed trials and the current best checkpoint are reused."
        ),
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
        help="Write per-epoch validation metrics for every candidate and fold.",
    )
    parser.add_argument(
        "--use-deterministic",
        action="store_true",
        help="Force deterministic CUDA algorithms.",
    )
    return parser.parse_args()


def load_deployment_folds(prepared: PreparedData) -> list[DeploymentFold]:
    """Load fixed deployment CV indices created during data preparation."""
    fold_count = int(prepared.metadata["cv_folds"])
    folds = []
    for fold_id in range(fold_count):
        fold_path = prepared.directory / "cv" / f"fold_{fold_id}"
        folds.append(
            DeploymentFold(
                fold=fold_id,
                train=np.load(fold_path / "train_idx.npy", allow_pickle=False),
                valid=np.load(fold_path / "valid_idx.npy", allow_pickle=False),
            )
        )
    if str(prepared.metadata.get("data_mode", "")).lower() == "hpo":
        if len(folds) != 1:
            raise ValueError("HPO prepared data must contain exactly one fold")
        split = folds[0]
        train = np.asarray(split.train, dtype=np.int64)
        valid = np.asarray(split.valid, dtype=np.int64)
        all_indices = np.arange(prepared.sample_count, dtype=np.int64)
        if np.intersect1d(train, valid).size:
            raise ValueError("HPO train and validation sets overlap")
        if not np.array_equal(np.sort(np.concatenate((train, valid))), all_indices):
            raise ValueError("HPO split does not cover every prepared sample")
        return folds
    validate_deployment_folds(folds, prepared.sample_count)
    return folds


def select_deployment_folds(
    folds: Sequence[DeploymentFold],
    fold_ids: Sequence[int] | None,
) -> list[DeploymentFold]:
    """Select prepared folds or return all folds in their persisted order."""
    if fold_ids is None:
        return list(folds)
    selected_ids = [int(fold_id) for fold_id in fold_ids]
    if not selected_ids:
        raise ValueError("At least one fold must be selected")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Selected folds must be unique")
    invalid = [
        fold_id
        for fold_id in selected_ids
        if fold_id < 0 or fold_id >= len(folds)
    ]
    if invalid:
        raise ValueError(
            f"folds must be between 0 and {len(folds) - 1}, got {invalid}"
        )
    return [folds[fold_id] for fold_id in selected_ids]


def configure_hpo_mode(
    config: Mapping[str, Any],
    *,
    mode: str,
    n_trials: int | None,
) -> Dict[str, Any]:
    """Apply CLI HPO defaults while preserving legacy cv-refit behavior."""
    configured = copy.deepcopy(dict(config))
    hpo = configured.setdefault("hpo", {})
    if not isinstance(hpo, dict):
        raise TypeError("hpo configuration must be a mapping")
    normalized = normalize_hpo_config(hpo)
    if mode == "per-fold":
        if normalized["method"] != "bayesian":
            raise ValueError("per-fold mode requires hpo.method: bayesian")
        resolved_trials = 500 if n_trials is None else int(n_trials)
        if resolved_trials < 1:
            raise ValueError("n-trials must be positive")
        hpo["enabled"] = True
        hpo["method"] = "bayesian"
        hpo["n_trials"] = resolved_trials
    elif mode == "cv-refit":
        if n_trials is not None:
            if int(n_trials) < 1:
                raise ValueError("n-trials must be positive")
            hpo["n_trials"] = int(n_trials)
    else:
        raise ValueError(f"Unsupported production mode: {mode}")
    return configured


def _processed_subset(
    prepared: PreparedData,
    indices: np.ndarray,
    processed_targets: torch.Tensor,
    *,
    targets_are_subset: bool = False,
) -> PreparedData:
    metadata = copy.deepcopy(prepared.metadata)
    metadata["sample_ids"] = [
        prepared.metadata["sample_ids"][int(index)] for index in indices
    ]
    features = prepared.features
    if isinstance(features, dict):
        selected_features = {
            name: tensor[indices].contiguous() for name, tensor in features.items()
        }
    else:
        selected_features = features[indices].contiguous()
    return PreparedData(
        features=selected_features,
        targets=(
            processed_targets.contiguous()
            if targets_are_subset
            else processed_targets[indices].contiguous()
        ),
        target_mask=prepared.target_mask[indices].contiguous(),
        metadata=metadata,
        directory=prepared.directory,
    )


def _load_processed_targets(
    path: Path,
    expected_rows: int,
    expected_columns: int,
) -> torch.Tensor:
    targets = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(targets, torch.Tensor)
        or targets.ndim != 2
        or tuple(targets.shape) != (expected_rows, expected_columns)
    ):
        raise ValueError(f"Invalid processed target cache: {path}")
    return targets


def _prepare_deployment_fold_data(
    *,
    prepared: PreparedData,
    split: DeploymentFold,
    device: str,
    gpu_resident: bool,
) -> tuple[PreparedData, PreparedData]:
    """Load, subset, and optionally transfer one fold exactly once."""
    fold_path = prepared.directory / "cv" / f"fold_{split.fold}"
    train_data = _processed_subset(
        prepared,
        split.train,
        _load_processed_targets(
            fold_path / "Y_train_processed.pt",
            len(split.train),
            prepared.targets.shape[1],
        ),
        targets_are_subset=True,
    )
    valid_data = _processed_subset(
        prepared,
        split.valid,
        _load_processed_targets(
            fold_path / "Y_valid_processed.pt",
            len(split.valid),
            prepared.targets.shape[1],
        ),
        targets_are_subset=True,
    )
    if gpu_resident and str(device).startswith("cuda"):
        train_data = _prepared_on_device(train_data, device)
        valid_data = _prepared_on_device(valid_data, device)
    return train_data, valid_data


def _train_deployment_fold(
    *,
    prepared: PreparedData,
    config: Mapping[str, Any],
    split: DeploymentFold,
    candidate_id: int,
    parameters: Mapping[str, Any],
    device: str,
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str],
    metric: str,
    direction: str,
    patience: int,
    loader_options: Mapping[str, Any],
    metrics_log_path: str | Path | None = None,
    fold_data: tuple[PreparedData, PreparedData] | None = None,
    materialize_checkpoint: bool = True,
):
    """Train one HPO candidate on one runtime-generated CV split."""
    candidate_config = merge_config(config, parameters)
    train_config = candidate_config.get("train", {})
    batch_size = int(train_config.get("batch_size", 32))
    effective_loader_options = dict(loader_options)
    gpu_resident = bool(train_config.get("gpu_resident", True)) and str(
        device
    ).startswith("cuda")
    if fold_data is None:
        train_data, valid_data = _prepare_deployment_fold_data(
            prepared=prepared,
            split=split,
            device=device,
            gpu_resident=gpu_resident,
        )
    else:
        train_data, valid_data = fold_data
    if gpu_resident:
        effective_loader_options = {"num_workers": 0, "pin_memory": False}
    train_loader = train_data.loader(
        np.arange(len(split.train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **effective_loader_options,
    )
    valid_loader = valid_data.loader(
        np.arange(len(split.valid)),
        batch_size=batch_size,
        shuffle=False,
        **effective_loader_options,
    )
    training_seed = resolve_training_seed(
        candidate_config,
        fallback=int(prepared.metadata.get("seed", 42)),
    )
    trainer = NestedCVTrainer(
        _make_seeded_model(
            candidate_config,
            prepared,
            regression_tasks,
            classification_tasks,
            training_seed,
        ),
        num_regression_tasks=len(regression_tasks),
        num_classification_tasks=len(classification_tasks),
        **_trainer_kwargs(
            candidate_config,
            device,
            regression_tasks,
            training_seed,
        ),
    )
    return trainer.train_inner(
        train_loader,
        valid_loader,
        max_epochs=int(train_config.get("num_epochs", 300)),
        patience=int(train_config.get("early_stopping_patience", patience)),
        metric=metric,
        direction=direction,
        min_delta=float(train_config.get("early_stopping_min_delta", 1e-4)),
        metrics_log_path=metrics_log_path,
        materialize_checkpoint=materialize_checkpoint,
    )


def _evaluate_candidate(
    candidate_id: int,
    parameters: Mapping[str, Any],
    context: DeploymentHPOContext,
    device: str,
) -> CandidateResult:
    def run_fold(
        current_parameters: Mapping[str, Any],
        fold_id: int,
        current_candidate_id: int,
    ):
        metrics_log_path = None
        if context.live_metrics_log:
            metrics_log_path = (
                Path(context.output_directory)
                / f"candidate_{current_candidate_id}"
                / f"fold_{fold_id}"
                / "metrics.jsonl"
            )
        return _train_deployment_fold(
            prepared=context.prepared_data,
            config=context.config,
            split=context.folds[fold_id],
            candidate_id=current_candidate_id,
            parameters=current_parameters,
            device=device,
            regression_tasks=context.regression_tasks,
            classification_tasks=context.classification_tasks,
            metric=context.metric,
            direction=context.direction,
            patience=context.patience,
            loader_options=context.loader_options,
            metrics_log_path=metrics_log_path,
        )

    return evaluate_candidate(
        candidate_id,
        parameters,
        range(len(context.folds)),
        run_fold,
        metric=context.metric,
    )


def _deployment_candidate_worker(
    job: DeploymentCandidateJob,
    device: str,
    context: DeploymentHPOContext,
) -> CandidateResult:
    configure_cuda_runtime(
        device,
        deterministic=train_deterministic_enabled(context.config),
    )
    return _evaluate_candidate(job.job_id, job.parameters, context, device)


def _deployment_fold_worker(
    job: DeploymentFoldJob,
    device: str,
    context: DeploymentHPOContext,
) -> Dict[str, Any]:
    """Train one Bayesian candidate on one fold in a persistent GPU worker."""
    metrics_log_path = None
    if context.live_metrics_log:
        metrics_log_path = (
            Path(context.output_directory)
            / f"candidate_{job.candidate_id}"
            / f"fold_{job.fold_id}"
            / "metrics.jsonl"
        )
    training = _train_deployment_fold(
        prepared=context.prepared_data,
        config=context.config,
        split=context.folds[job.fold_id],
        candidate_id=job.candidate_id,
        parameters=job.parameters,
        device=device,
        regression_tasks=context.regression_tasks,
        classification_tasks=context.classification_tasks,
        metric=context.metric,
        direction=context.direction,
        patience=context.patience,
        loader_options=context.loader_options,
        metrics_log_path=metrics_log_path,
    )
    metrics = copy.deepcopy(dict(training.best_metrics))
    metric_value = _metric_from_metrics(metrics, context.metric)
    return {
        "candidate_id": int(job.candidate_id),
        "fold_id": int(job.fold_id),
        "metric": float(metric_value),
        "best_epoch": int(training.best_epoch),
        "metrics": metrics,
    }


def _metric_from_metrics(metrics: Mapping[str, Any], metric: str) -> float:
    """Resolve configured HPO metric names using the shared legacy aliases."""
    metric_value = metrics.get("avg_pearson")
    if metric_value is None:
        metric_path = metric.replace("/", ".")
        current: Any = metrics
        for part in metric_path.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                aliases = {
                    "best.val_r": "avg_pearson",
                    "val_r": "avg_pearson",
                    "pearson": "avg_pearson",
                }
                alias = aliases.get(metric_path) or aliases.get(part)
                if alias not in metrics:
                    raise KeyError(f"Metric path '{metric}' is not present")
                current = metrics[alias]
                break
        metric_value = current
    return float(metric_value)


def _per_fold_checkpoint(
    *,
    prepared: PreparedData,
    split: DeploymentFold,
    config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    training: TrainingResult,
    processor: PerTraitPreprocessor,
    cv_seed: int,
) -> Dict[str, Any]:
    """Build a prediction-compatible checkpoint for one validation-selected model."""
    selected_config = merge_config(config, parameters)
    regression_tasks, classification_tasks = _task_lists(prepared)
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
    checkpoint_metadata["sequence_lengths"] = _sequence_lengths(prepared.features)
    checkpoint_metadata["deployment_cv"] = {
        "folds": int(prepared.metadata["cv_folds"]),
        "fold": int(split.fold),
        "seed": int(cv_seed),
        "evaluation_set": "validation",
        "train_samples": int(len(split.train)),
        "validation_samples": int(len(split.valid)),
        "refit_full_dataset": False,
    }
    return {
        **training.checkpoint_state,
        "config": selected_config,
        "metadata": checkpoint_metadata,
        "preprocessing": processor.to_dict(),
        "deployment_model": True,
        "deployment_cv_folds": int(prepared.metadata["cv_folds"]),
        "deployment_cv_fold": int(split.fold),
        "deployment_cv_seed": int(cv_seed),
        "selected_hyperparameters": dict(parameters),
        "final_epoch": int(training.best_epoch),
        "scheduler_epochs": int(
            selected_config.get("train", {}).get(
                "num_epochs", training.best_epoch
            )
        ),
        "evaluation_set": "validation",
        "refit_full_dataset": False,
    }


def _materialize_best_training(training: TrainingResult) -> TrainingResult:
    """Copy only a winning trial's model state from GPU to CPU."""
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in training.best_state_dict.items()
    }
    checkpoint_state = {
        "epoch": int(training.best_epoch),
        "model_state_dict": state,
        "metrics": copy.deepcopy(dict(training.best_metrics)),
    }
    return TrainingResult(
        best_epoch=int(training.best_epoch),
        best_metrics=copy.deepcopy(dict(training.best_metrics)),
        best_state_dict=state,
        history=copy.deepcopy(training.history),
        checkpoint_state=checkpoint_state,
    )


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Atomically replace a checkpoint so interruption leaves a valid winner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _save_running_per_fold_best(
    *,
    context: DeploymentHPOContext,
    split: DeploymentFold,
    candidate: CandidateResult,
    training: TrainingResult,
) -> None:
    """Persist the current study winner immediately for safe resume."""
    fold_directory = Path(context.output_directory) / f"fold_{split.fold}"
    processor = PerTraitPreprocessor.load_json(
        context.prepared_data.directory
        / "cv"
        / f"fold_{split.fold}"
        / "preprocessing.json"
    )
    cv_seed = int(
        context.prepared_data.metadata.get(
            "cv_seed", context.prepared_data.metadata.get("seed", 42)
        )
    )
    checkpoint = _per_fold_checkpoint(
        prepared=context.prepared_data,
        split=split,
        config=context.config,
        parameters=candidate.parameters,
        training=training,
        processor=processor,
        cv_seed=cv_seed,
    )
    checkpoint["hpo_candidate_id"] = int(candidate.candidate_id)
    checkpoint["hpo_objective"] = float(candidate.objective)
    _atomic_torch_save(checkpoint, fold_directory / "best_model.pt")
    _write_json(fold_directory / "training_history.json", training.history)


def _candidate_from_optuna_trial(
    trial: Any,
    *,
    fold_id: int,
) -> CandidateResult:
    """Reconstruct one completed candidate from persistent Optuna attributes."""
    metrics = copy.deepcopy(dict(trial.user_attrs.get("metrics") or {}))
    best_epoch = int(trial.user_attrs["best_epoch"])
    objective = float(trial.value)
    return CandidateResult(
        candidate_id=int(trial.number),
        parameters=copy.deepcopy(dict(trial.params)),
        objective=objective,
        inner_results=(
            InnerFoldResult(
                inner_fold=int(fold_id),
                metric=objective,
                best_epoch=best_epoch,
                metrics=metrics,
            ),
        ),
    )


def _load_running_best_training(
    fold_output: Path,
) -> tuple[int, TrainingResult] | None:
    """Load the checkpoint required to continue a previously interrupted study."""
    checkpoint_path = fold_output / "best_model.pt"
    history_path = fold_output / "training_history.json"
    if not checkpoint_path.is_file():
        return None
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    candidate_id = int(checkpoint["hpo_candidate_id"])
    state = dict(checkpoint["model_state_dict"])
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.is_file()
        else []
    )
    training = TrainingResult(
        best_epoch=int(checkpoint["epoch"]),
        best_metrics=copy.deepcopy(dict(checkpoint.get("metrics") or {})),
        best_state_dict=state,
        history=history,
        checkpoint_state={
            "epoch": int(checkpoint["epoch"]),
            "model_state_dict": state,
            "metrics": copy.deepcopy(dict(checkpoint.get("metrics") or {})),
        },
    )
    return candidate_id, training


def _run_per_fold_study(
    *,
    context: DeploymentHPOContext,
    split: DeploymentFold,
    device: str,
) -> tuple[HPOResult, TrainingResult]:
    """Run or resume one TPE study with fold data resident on its GPU."""
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Bayesian HPO requires optuna") from error

    normalized = normalize_hpo_config(context.config.get("hpo", {}))
    if normalized["method"] != "bayesian":
        raise ValueError("per-fold mode requires Bayesian HPO")
    sampler = optuna.samplers.TPESampler(seed=normalized["seed"])
    fold_output = Path(context.output_directory) / f"fold_{split.fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    storage = None
    study_name = None
    if context.resume:
        storage = f"sqlite:///{(fold_output / 'optuna.db').resolve()}"
        study_name = f"aquila-production-fold-{split.fold}"
    study = optuna.create_study(
        direction=normalized["direction"],
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=context.resume,
    )
    finite_candidates = finite_parameter_candidates(normalized["parameters"])
    if (
        finite_candidates is not None
        and normalized["n_trials"] > len(finite_candidates)
    ):
        raise ValueError(
            f"Requested {normalized['n_trials']} unique Bayesian trials, but "
            f"the finite search space contains only {len(finite_candidates)} "
            "combinations"
        )
    if finite_candidates is not None:
        rng = np.random.default_rng(normalized["seed"])
        rng.shuffle(finite_candidates)

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and "best_epoch" in trial.user_attrs
    ]
    candidates = [
        _candidate_from_optuna_trial(trial, fold_id=split.fold)
        for trial in completed_trials
    ]
    seen = {parameter_key(candidate.parameters) for candidate in candidates}
    best_training: TrainingResult | None = None
    best_candidate = (
        select_best_candidate(
            candidates,
            normalized["direction"],
            method="bayesian",
        ).best
        if candidates
        else None
    )
    if best_candidate is not None:
        loaded = _load_running_best_training(fold_output)
        if loaded is None or loaded[0] != best_candidate.candidate_id:
            raise RuntimeError(
                f"Fold {split.fold} has persistent trials but no matching "
                "best_model.pt; restart with --overwrite"
            )
        best_training = loaded[1]
    maximize = normalized["direction"] == "maximize"
    base_train = context.config.get("train", {})
    fold_data = _prepare_deployment_fold_data(
        prepared=context.prepared_data,
        split=split,
        device=device,
        gpu_resident=bool(base_train.get("gpu_resident", True)),
    )
    print(
        f"[INFO] Fold {split.fold}: {len(candidates)}/"
        f"{normalized['n_trials']} completed TPE trials; fold data cached on "
        f"{device}",
        flush=True,
    )
    while len(candidates) < normalized["n_trials"]:
        trial, parameters, duplicate_count = unique_bayesian_trial(
            study,
            normalized["parameters"],
            seen,
            finite_candidates,
        )
        metrics_log_path = None
        if context.live_metrics_log:
            metrics_log_path = (
                fold_output / f"candidate_{trial.number}" / "metrics.jsonl"
            )
        training = _train_deployment_fold(
            prepared=context.prepared_data,
            config=context.config,
            split=split,
            candidate_id=int(trial.number),
            parameters=parameters,
            device=device,
            regression_tasks=context.regression_tasks,
            classification_tasks=context.classification_tasks,
            metric=context.metric,
            direction=context.direction,
            patience=context.patience,
            loader_options=context.loader_options,
            metrics_log_path=metrics_log_path,
            fold_data=fold_data,
            materialize_checkpoint=False,
        )
        objective = _metric_from_metrics(training.best_metrics, context.metric)
        if not math.isfinite(objective):
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise ValueError(
                f"Non-finite HPO objective in fold {split.fold}, "
                f"trial {trial.number}"
            )
        inner = InnerFoldResult(
            inner_fold=int(split.fold),
            metric=float(objective),
            best_epoch=int(training.best_epoch),
            metrics=copy.deepcopy(dict(training.best_metrics)),
        )
        candidate = CandidateResult(
            candidate_id=int(trial.number),
            parameters=copy.deepcopy(parameters),
            objective=float(objective),
            inner_results=(inner,),
        )
        candidates.append(candidate)
        seen.add(parameter_key(parameters))
        trial.set_user_attr("best_epoch", int(training.best_epoch))
        trial.set_user_attr(
            "metrics",
            _json_safe(dict(training.best_metrics)),
        )
        study.tell(trial, objective)
        is_better = best_candidate is None or (
            objective > best_candidate.objective
            if maximize
            else objective < best_candidate.objective
        )
        if is_better:
            best_candidate = candidate
            best_training = _materialize_best_training(training)
            _save_running_per_fold_best(
                context=context,
                split=split,
                candidate=candidate,
                training=best_training,
            )
        duplicate_note = (
            f" skipped_duplicates={duplicate_count}" if duplicate_count else ""
        )
        print(
            f"[Fold {split.fold} Bayesian HPO] unique={len(candidates)}/"
            f"{normalized['n_trials']} optuna_trial={trial.number + 1} "
            f"objective={objective:.6f} best_epoch={training.best_epoch}"
            f"{duplicate_note}",
            flush=True,
        )
    result = select_best_candidate(
        candidates,
        normalized["direction"],
        method="bayesian",
    )
    if best_training is None or best_candidate is None:
        raise RuntimeError(f"Fold {split.fold} HPO completed without a model")
    if result.best.candidate_id != best_candidate.candidate_id:
        raise RuntimeError("In-memory best checkpoint does not match HPO winner")
    return result, best_training


def _persist_per_fold_result(
    *,
    context: DeploymentHPOContext,
    split: DeploymentFold,
    hpo_result: HPOResult,
    training: TrainingResult,
    device: str,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Persist one fold's best validation-selected checkpoint and reports."""
    fold_directory = Path(context.output_directory) / f"fold_{split.fold}"
    fold_directory.mkdir(parents=True, exist_ok=True)
    selected_config = merge_config(context.config, hpo_result.best.parameters)
    selected_config.setdefault("data", {})
    selected_config["data"].update(
        {
            "encoding_type": context.prepared_data.metadata["encoding_type"],
            "variant_type": context.prepared_data.metadata.get("variant_type"),
            "regression_tasks": list(context.regression_tasks),
            "classification_tasks": list(context.classification_tasks),
        }
    )
    processor = PerTraitPreprocessor.load_json(
        context.prepared_data.directory
        / "cv"
        / f"fold_{split.fold}"
        / "preprocessing.json"
    )
    cv_seed = int(
        context.prepared_data.metadata.get(
            "cv_seed", context.prepared_data.metadata.get("seed", 42)
        )
    )
    checkpoint = _per_fold_checkpoint(
        prepared=context.prepared_data,
        split=split,
        config=context.config,
        parameters=hpo_result.best.parameters,
        training=training,
        processor=processor,
        cv_seed=cv_seed,
    )
    checkpoint["hpo_candidate_id"] = int(hpo_result.best.candidate_id)
    checkpoint["hpo_objective"] = float(hpo_result.best.objective)
    _atomic_torch_save(checkpoint, fold_directory / "best_model.pt")
    processor.save_json(fold_directory / "preprocessing.json")
    with (fold_directory / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(selected_config, handle, sort_keys=False)
    _write_json(fold_directory / "training_history.json", training.history)
    normalized = normalize_hpo_config(context.config.get("hpo", {}))
    _write_json(
        fold_directory / "hpo_results.json",
        {
            "method": hpo_result.method,
            "metric": normalized["metric"],
            "direction": hpo_result.direction,
            "fold": int(split.fold),
            "n_trials": len(hpo_result.candidates),
            "best_candidate_id": hpo_result.best.candidate_id,
            "best_parameters": dict(hpo_result.best.parameters),
            "best_validation": hpo_result.best.objective,
            "best_epoch": training.best_epoch,
            "candidates": _candidate_records(hpo_result),
        },
    )
    training_seed = resolve_training_seed(
        selected_config,
        fallback=int(context.prepared_data.metadata.get("seed", 42)),
    )
    runtime = {
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda")
            else None
        ),
        "cv_seed": cv_seed,
        "training_seed": training_seed,
        "elapsed_seconds": elapsed_seconds,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    _write_json(fold_directory / "runtime.json", runtime)
    summary = {
        "status": "completed",
        "mode": "per-fold",
        "model_type": "production",
        "evaluation_set": "validation",
        "refit_full_dataset": False,
        "fold": int(split.fold),
        "folds": int(context.prepared_data.metadata["cv_folds"]),
        "train_samples": int(len(split.train)),
        "validation_samples": int(len(split.valid)),
        "hpo_trials": len(hpo_result.candidates),
        "hpo_metric": normalized["metric"],
        "hpo_direction": hpo_result.direction,
        "best_validation": hpo_result.best.objective,
        "best_epoch": int(training.best_epoch),
        "selected_hyperparameters": dict(hpo_result.best.parameters),
        "checkpoint": str((fold_directory / "best_model.pt").resolve()),
        "runtime": runtime,
    }
    _write_json(fold_directory / "summary.json", summary)
    return summary


def _per_fold_hpo_worker(
    job: PerFoldHPOJob,
    device: str,
    context: DeploymentHPOContext,
) -> Dict[str, Any]:
    """Run and persist one complete fold-local study."""
    started = time.time()
    configure_cuda_runtime(
        device,
        deterministic=train_deterministic_enabled(context.config),
    )
    split_by_id = {split.fold: split for split in context.folds}
    split = split_by_id[job.fold_id]
    hpo_result, training = _run_per_fold_study(
        context=context,
        split=split,
        device=device,
    )
    return _persist_per_fold_result(
        context=context,
        split=split,
        hpo_result=hpo_result,
        training=training,
        device=device,
        elapsed_seconds=time.time() - started,
    )


def _run_per_fold_mode(
    context: DeploymentHPOContext,
    gpu_ids: Sequence[int],
) -> list[Dict[str, Any]]:
    """Schedule independent fold studies across available GPUs."""
    jobs = [
        PerFoldHPOJob(job_id=int(split.fold), fold_id=int(split.fold))
        for split in context.folds
    ]
    results = execute_gpu_jobs(
        jobs,
        _per_fold_hpo_worker,
        gpu_ids,
        worker_args=(context,),
        raise_on_error=True,
        deterministic=train_deterministic_enabled(context.config),
    )
    return [dict(result.value) for result in results]


def _run_bayesian_hpo_on_gpus(
    context: DeploymentHPOContext,
    gpu_ids: Sequence[int],
) -> HPOResult:
    """Run strict sequential TPE trials with folds spread across all GPUs."""
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Bayesian HPO requires optuna") from error

    normalized = normalize_hpo_config(context.config.get("hpo", {}))
    sampler = optuna.samplers.TPESampler(seed=normalized["seed"])
    study = optuna.create_study(
        direction=normalized["direction"],
        sampler=sampler,
    )
    candidates: list[CandidateResult] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    finite_candidates = finite_parameter_candidates(normalized["parameters"])
    if finite_candidates is not None:
        if normalized["n_trials"] > len(finite_candidates):
            raise ValueError(
                f"Requested {normalized['n_trials']} unique Bayesian trials, "
                f"but the finite search space contains only "
                f"{len(finite_candidates)} combinations"
            )
        rng = np.random.default_rng(normalized["seed"])
        rng.shuffle(finite_candidates)
    fold_count = len(context.folds)
    print(
        f"[INFO] Persistent Bayesian GPU pool on {list(gpu_ids)}; "
        f"{normalized['n_trials']} sequential TPE trials, "
        f"{fold_count} fold jobs per trial",
        flush=True,
    )
    with PersistentGPUPool(
        gpu_ids,
        _deployment_fold_worker,
        worker_args=(context,),
        deterministic=train_deterministic_enabled(context.config),
    ) as pool:
        while len(candidates) < normalized["n_trials"]:
            trial, parameters, duplicate_count = unique_bayesian_trial(
                study,
                normalized["parameters"],
                seen,
                finite_candidates,
            )
            if duplicate_count:
                print(
                    f"[Bayesian HPO] skipped {duplicate_count} duplicate "
                    f"suggestion(s) before unique candidate "
                    f"{len(candidates) + 1}/{normalized['n_trials']}",
                    flush=True,
                )
            jobs = [
                DeploymentFoldJob(
                    job_id=int(trial.number) * fold_count + fold_id,
                    candidate_id=int(trial.number),
                    fold_id=fold_id,
                    parameters=copy.deepcopy(parameters),
                )
                for fold_id in range(fold_count)
            ]
            pool.submit_many(jobs)
            by_fold: Dict[int, InnerFoldResult] = {}
            while len(by_fold) < fold_count:
                try:
                    work_result = pool.get(timeout=1.0)
                except Empty:
                    if pool.workers_alive:
                        continue
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise RuntimeError(
                        f"All GPU workers exited during Bayesian trial {trial.number}"
                    )
                if not work_result.succeeded:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise RuntimeError(
                        f"Bayesian trial {trial.number} GPU job "
                        f"{work_result.job_id} failed on {work_result.device}: "
                        f"{work_result.error}\n{work_result.traceback or ''}"
                    )
                payload = dict(work_result.value)
                if int(payload.get("candidate_id", -1)) != int(trial.number):
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise RuntimeError(
                        f"Received result for candidate "
                        f"{payload.get('candidate_id')} while waiting for "
                        f"trial {trial.number}"
                    )
                fold_id = int(payload["fold_id"])
                if fold_id < 0 or fold_id >= fold_count or fold_id in by_fold:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise RuntimeError(
                        f"Invalid or duplicate fold result {fold_id} "
                        f"for Bayesian trial {trial.number}"
                    )
                metric = float(payload["metric"])
                if not math.isfinite(metric):
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    raise RuntimeError(
                        f"Non-finite fold metric in Bayesian trial "
                        f"{trial.number}, fold {fold_id}"
                    )
                by_fold[fold_id] = InnerFoldResult(
                    inner_fold=fold_id,
                    metric=metric,
                    best_epoch=int(payload["best_epoch"]),
                    metrics=copy.deepcopy(dict(payload["metrics"])),
                )
            inner_results = tuple(by_fold[index] for index in range(fold_count))
            objective = float(
                np.mean([result.metric for result in inner_results])
            )
            candidate = CandidateResult(
                candidate_id=int(trial.number),
                parameters=copy.deepcopy(parameters),
                objective=objective,
                inner_results=inner_results,
            )
            candidates.append(candidate)
            seen.add(parameter_key(parameters))
            trial.set_user_attr(
                "inner_best_epochs", list(candidate.best_epochs)
            )
            study.tell(trial, objective)
            print(
                f"[Bayesian HPO] unique={len(candidates)}/"
                f"{normalized['n_trials']} optuna_trial={trial.number + 1} "
                f"objective={objective:.6f} "
                f"final_epoch={candidate.final_epoch}",
                flush=True,
            )
    return select_best_candidate(
        candidates,
        normalized["direction"],
        method="bayesian",
    )


def _run_hpo(
    context: DeploymentHPOContext,
    gpu_ids: Sequence[int],
) -> HPOResult:
    normalized = normalize_hpo_config(context.config.get("hpo", {}))
    if normalized["method"] == "bayesian" and len(gpu_ids) > 1:
        return _run_bayesian_hpo_on_gpus(context, gpu_ids)
    if normalized["method"] == "grid" and len(gpu_ids) > 1:
        parameter_sets = generate_grid_candidates(normalized["parameters"])
        jobs = [
            DeploymentCandidateJob(job_id=index, parameters=parameters)
            for index, parameters in enumerate(parameter_sets)
        ]
        results = execute_gpu_jobs(
            jobs,
            _deployment_candidate_worker,
            gpu_ids,
            worker_args=(context,),
            raise_on_error=True,
            deterministic=train_deterministic_enabled(context.config),
        )
        return select_best_candidate(
            [result.value for result in results],
            normalized["direction"],
            method="grid",
        )

    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
    configure_cuda_runtime(
        device,
        deterministic=train_deterministic_enabled(context.config),
    )

    def run_fold(
        parameters: Mapping[str, Any],
        fold_id: int,
        candidate_id: int,
    ):
        metrics_log_path = None
        if context.live_metrics_log:
            metrics_log_path = (
                Path(context.output_directory)
                / f"candidate_{candidate_id}"
                / f"fold_{fold_id}"
                / "metrics.jsonl"
            )
        return _train_deployment_fold(
            prepared=context.prepared_data,
            config=context.config,
            split=context.folds[fold_id],
            candidate_id=candidate_id,
            parameters=parameters,
            device=device,
            regression_tasks=context.regression_tasks,
            classification_tasks=context.classification_tasks,
            metric=context.metric,
            direction=context.direction,
            patience=context.patience,
            loader_options=context.loader_options,
            metrics_log_path=metrics_log_path,
        )

    return run_hpo(
        context.config.get("hpo", {}),
        list(range(len(context.folds))),
        run_fold,
    )


def _prepare_output(
    output_directory: Path,
    overwrite: bool,
    *,
    resume: bool = False,
) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if output_directory.exists() and any(output_directory.iterdir()):
        if resume:
            return
        if not overwrite:
            raise FileExistsError(
                f"Production output already exists: {output_directory}; "
                "use --overwrite to replace it or --resume to continue it"
            )
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _refit_full_dataset(
    *,
    prepared: PreparedData,
    config: Mapping[str, Any],
    hpo_result: HPOResult | None,
    device: str,
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str],
    loader_options: Mapping[str, Any],
    fold_count: int,
    cv_seed: int,
    training_seed: int | None = None,
    metrics_log_path: str | Path | None = None,
):
    if hpo_result is None:
        selected_config = copy.deepcopy(dict(config))
        selected_parameters: Dict[str, Any] = {}
    else:
        selected_config = merge_config(config, hpo_result.best.parameters)
        selected_parameters = dict(hpo_result.best.parameters)
    if training_seed is not None:
        selected_config.setdefault("train", {})["seed"] = int(training_seed)
    all_indices = np.arange(prepared.sample_count, dtype=np.int64)
    full_path = prepared.directory / "full"
    processor = PerTraitPreprocessor.load_json(
        full_path / "preprocessing.json"
    )
    processed_targets = _load_processed_targets(
        full_path / "Y_processed.pt",
        prepared.sample_count,
        prepared.targets.shape[1],
    )
    full_data = _processed_subset(prepared, all_indices, processed_targets)
    train_config = selected_config.get("train", {})
    batch_size = int(train_config.get("batch_size", 32))
    effective_loader_options = dict(loader_options)
    if bool(train_config.get("gpu_resident", True)) and device.startswith("cuda"):
        full_data = _prepared_on_device(full_data, device)
        effective_loader_options = {"num_workers": 0, "pin_memory": False}
    loader = full_data.loader(
        all_indices,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **effective_loader_options,
    )
    training_seed = resolve_training_seed(
        selected_config,
        fallback=int(prepared.metadata.get("seed", 42)),
    )
    trainer = NestedCVTrainer(
        _make_seeded_model(
            selected_config,
            prepared,
            regression_tasks,
            classification_tasks,
            training_seed,
        ),
        num_regression_tasks=len(regression_tasks),
        num_classification_tasks=len(classification_tasks),
        **_trainer_kwargs(
            selected_config,
            device,
            regression_tasks,
            training_seed,
        ),
    )
    if hpo_result is not None:
        final_epoch = int(hpo_result.best.final_epoch)
    elif train_config.get("fixed_epochs") is not None:
        final_epoch = int(train_config["fixed_epochs"])
    else:
        final_epoch = int(train_config["num_epochs"])
    scheduler_epochs = int(train_config.get("num_epochs", final_epoch))
    training = trainer.train_fixed_epochs(
        loader,
        epochs=final_epoch,
        scheduler_epochs=scheduler_epochs,
        metrics_log_path=metrics_log_path,
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
    checkpoint_metadata["sequence_lengths"] = _sequence_lengths(prepared.features)
    checkpoint_metadata["deployment_cv"] = {
        "folds": int(fold_count),
        "seed": int(cv_seed),
        "evaluation_set": "none",
    }
    checkpoint = {
        **training.checkpoint_state,
        "config": selected_config,
        "metadata": checkpoint_metadata,
        "preprocessing": processor.to_dict(),
        "deployment_model": True,
        "deployment_cv_folds": int(fold_count),
        "deployment_cv_seed": int(cv_seed),
        "selected_hyperparameters": selected_parameters,
        "final_epoch": final_epoch,
        "scheduler_epochs": scheduler_epochs,
    }
    return selected_config, processor, training, checkpoint, training_seed


def train_hpo_prepared_model(
    *,
    data_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    seed: int,
    precision: str = "bf16",
    live_metrics_log: bool = False,
    use_deterministic: bool = False,
    device: str | None = None,
) -> None:
    """Train the one fixed split stored by aquila_data_hpo.py.

    Features are uploaded once when ``train.gpu_resident`` is true. The
    validation Pearson correlation is written to ``best_metrics.json`` so
    ``aquila_train_hpo.py`` can score the trial.
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    prepared = load_prepared_data(data_dir)
    if str(prepared.metadata.get("data_mode", "")).lower() != "hpo":
        raise ValueError(
            "HPO split training requires data prepared by aquila_data_hpo.py"
        )
    config = load_config(config_path)
    config.setdefault("train", {})
    config["train"]["precision"] = precision
    config["train"]["seed"] = int(seed)
    resolve_train_deterministic(
        config,
        True if use_deterministic else None,
    )
    if live_metrics_log:
        config["train"]["live_metrics_log"] = True
    config["train"].pop("mixed_precision", None)
    _validate_preprocessing_cache(prepared, config)
    regression_tasks, classification_tasks = _task_lists(prepared)
    folds = load_deployment_folds(prepared)
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if resolved_device == "cuda":
        resolved_device = "cuda:0"
    configure_cuda_runtime(
        resolved_device,
        deterministic=train_deterministic_enabled(config),
    )
    normalized_hpo = normalize_hpo_config(config.get("hpo", {}))
    training = _train_deployment_fold(
        prepared=prepared,
        config=config,
        split=folds[0],
        candidate_id=0,
        parameters={},
        device=resolved_device,
        regression_tasks=regression_tasks,
        classification_tasks=classification_tasks,
        metric=str(normalized_hpo.get("metric", "best/val_r")),
        direction=str(normalized_hpo.get("direction", "maximize")),
        patience=int(config["train"].get("early_stopping_patience", 40)),
        loader_options=_loader_kwargs(config.get("train", {})),
        metrics_log_path=(
            output_directory / "metrics.jsonl" if live_metrics_log else None
        ),
    )
    metrics = _json_safe(dict(training.best_metrics))
    if "avg_pearson" not in metrics:
        raise KeyError("HPO training did not return avg_pearson")
    metrics["val_r"] = float(metrics["avg_pearson"])
    metrics["val_pearson"] = float(metrics["avg_pearson"])
    _write_json(output_directory / "best_metrics.json", metrics)
    torch.save(
        {
            "model_state_dict": training.best_state_dict,
            "best_metrics": metrics,
            "best_epoch": int(training.best_epoch),
            "config": config,
            "metadata": prepared.metadata,
        },
        output_directory / "best_model.pt",
    )
    print(
        f"HPO split fold 0 finished at epoch {training.best_epoch} "
        f"with val_r={metrics['val_r']:.6f}"
    )


def train_prepared_model(
    *,
    data_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    seed: int,
    precision: str = "bf16",
    live_metrics_log: bool = False,
    use_deterministic: bool = False,
    overwrite: bool = False,
    device: str | None = None,
) -> None:
    """Train one full-data model from a prepared dataset and a fixed config."""
    output_directory = Path(output_dir)
    if output_directory.exists() and any(output_directory.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_directory}; "
                "use --overwrite to replace it"
            )
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    prepared = load_prepared_data(data_dir)
    if prepared.metadata.get("data_mode") != "deployment":
        raise ValueError(
            "Full-data training requires data prepared by "
            "aquila-data-cv-production"
        )
    config = load_config(config_path)
    config.setdefault("train", {})
    config["train"]["precision"] = precision
    resolve_train_deterministic(config, True if use_deterministic else None)
    if live_metrics_log:
        config["train"]["live_metrics_log"] = True
    config["train"].pop("mixed_precision", None)
    _validate_preprocessing_cache(prepared, config)
    regression_tasks, classification_tasks = _task_lists(prepared)
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if resolved_device == "cuda":
        resolved_device = "cuda:0"
    configure_cuda_runtime(
        resolved_device,
        deterministic=train_deterministic_enabled(config),
    )
    if train_deterministic_enabled(config):
        print("[INFO] CUDA deterministic algorithms enabled")
    metrics_log_path = (
        output_directory / "metrics.jsonl" if live_metrics_log else None
    )
    selected_config, processor, training, checkpoint, training_seed = (
        _refit_full_dataset(
            prepared=prepared,
            config=config,
            hpo_result=None,
            device=resolved_device,
            regression_tasks=regression_tasks,
            classification_tasks=classification_tasks,
            loader_options=_loader_kwargs(config.get("train", {})),
            fold_count=int(
                prepared.metadata.get(
                    "cv_folds", prepared.metadata.get("folds", 0)
                )
            ),
            cv_seed=int(
                prepared.metadata.get(
                    "cv_seed", prepared.metadata.get("seed", 42)
                )
            ),
            training_seed=int(seed),
            metrics_log_path=metrics_log_path,
        )
    )
    if int(training_seed) != int(seed):
        raise RuntimeError(
            f"Seed {seed} resolved to training seed {training_seed}"
        )
    torch.save(checkpoint, output_directory / "best_model.pt")
    processor.save_json(output_directory / "preprocessing.json")
    with (output_directory / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(selected_config, handle, sort_keys=False)
    _write_json(output_directory / "training_history.json", training.history)
    runtime = {
        "device": resolved_device,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(resolved_device))
            if resolved_device.startswith("cuda")
            else None
        ),
        "cv_seed": int(checkpoint["deployment_cv_seed"]),
        "training_seed": int(training_seed),
        "final_epoch": int(checkpoint["final_epoch"]),
        "scheduler_epochs": int(checkpoint["scheduler_epochs"]),
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    _write_json(output_directory / "runtime.json", runtime)
    _write_json(
        output_directory / "summary.json",
        {
            "status": "completed",
            "model_type": "production",
            "evaluation_set": "none",
            "sample_count": prepared.sample_count,
            "folds": int(checkpoint["deployment_cv_folds"]),
            "cv_seed": int(checkpoint["deployment_cv_seed"]),
            "training_seed": int(training_seed),
            "final_epoch": int(checkpoint["final_epoch"]),
            "scheduler_epochs": int(checkpoint["scheduler_epochs"]),
            "data_dir": str(Path(data_dir).resolve()),
            "config": str(Path(config_path).resolve()),
            "checkpoint": str((output_directory / "best_model.pt").resolve()),
            "runtime": runtime,
        },
    )
    print(
        f"Completed full-data training with seed {training_seed}; "
        f"checkpoint: {output_directory / 'best_model.pt'}"
    )


def _run_cv_refit_mode(
    *,
    args: argparse.Namespace,
    prepared: PreparedData,
    config: Mapping[str, Any],
    folds: Sequence[DeploymentFold],
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str],
    output_directory: Path,
    gpu_ids: Sequence[int],
    started: float,
) -> None:
    """Preserve the original cross-fold HPO and full-data refit workflow."""
    cv_seed = int(prepared.metadata.get("cv_seed", prepared.metadata["seed"]))
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
    train_config = config.get("train", {})
    normalized_hpo = normalize_hpo_config(config.get("hpo", {}))
    context = DeploymentHPOContext(
        prepared_data=prepared,
        config=copy.deepcopy(dict(config)),
        folds=tuple(folds),
        regression_tasks=tuple(regression_tasks),
        classification_tasks=tuple(classification_tasks),
        metric=str(normalized_hpo["metric"]),
        direction=str(normalized_hpo["direction"]),
        patience=int(train_config.get("early_stopping_patience", 20)),
        loader_options=_loader_kwargs(train_config),
        output_directory=str(output_directory),
        live_metrics_log=bool(train_config.get("live_metrics_log", False)),
    )
    hpo_result = _run_hpo(context, gpu_ids)
    configure_cuda_runtime(
        device,
        deterministic=train_deterministic_enabled(config),
    )
    selected_config, processor, training, checkpoint, training_seed = (
        _refit_full_dataset(
            prepared=prepared,
            config=config,
            hpo_result=hpo_result,
            device=device,
            regression_tasks=regression_tasks,
            classification_tasks=classification_tasks,
            loader_options=_loader_kwargs(train_config),
            fold_count=len(folds),
            cv_seed=cv_seed,
        )
    )
    torch.save(checkpoint, output_directory / "best_model.pt")
    processor.save_json(output_directory / "preprocessing.json")
    with (output_directory / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(selected_config, handle, sort_keys=False)
    _write_json(
        output_directory / "hpo_results.json",
        {
            "method": hpo_result.method,
            "metric": normalized_hpo["metric"],
            "direction": hpo_result.direction,
            "folds": len(folds),
            "cv_seed": cv_seed,
            "best_candidate_id": hpo_result.best.candidate_id,
            "best_parameters": dict(hpo_result.best.parameters),
            "best_validation_mean": hpo_result.best.objective,
            "final_epoch": hpo_result.best.final_epoch,
            "candidates": _candidate_records(hpo_result),
        },
    )
    _write_json(output_directory / "training_history.json", training.history)
    runtime = {
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda")
            else None
        ),
        "cv_seed": cv_seed,
        "training_seed": training_seed,
        "scheduler_epochs": checkpoint["scheduler_epochs"],
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    _write_json(output_directory / "runtime.json", runtime)
    _write_json(
        output_directory / "summary.json",
        {
            "status": "completed",
            "mode": "cv-refit",
            "model_type": "production",
            "evaluation_set": "none",
            "refit_full_dataset": True,
            "sample_count": prepared.sample_count,
            "folds": len(folds),
            "cv_seed": cv_seed,
            "hpo_metric": normalized_hpo["metric"],
            "hpo_direction": hpo_result.direction,
            "best_validation_mean": hpo_result.best.objective,
            "selected_hyperparameters": dict(hpo_result.best.parameters),
            "fold_best_epochs": list(hpo_result.best.best_epochs),
            "final_epoch": hpo_result.best.final_epoch,
            "scheduler_epochs": checkpoint["scheduler_epochs"],
            "data_dir": str(Path(args.data_dir).resolve()),
            "config": str(Path(args.config).resolve()),
            "checkpoint": str((output_directory / "best_model.pt").resolve()),
            "runtime": runtime,
        },
    )
    print(
        f"Completed production CV ({len(folds)} folds) and full-data refit; "
        f"checkpoint: {output_directory / 'best_model.pt'}"
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    prepared = load_prepared_data(args.data_dir)
    config = configure_hpo_mode(
        load_config(args.config),
        mode=args.mode,
        n_trials=args.n_trials,
    )
    config.setdefault("train", {})
    config["train"]["precision"] = args.precision
    resolve_train_deterministic(
        config,
        True if args.use_deterministic else None,
    )
    if args.live_metrics_log:
        config["train"]["live_metrics_log"] = True
    config["train"].pop("mixed_precision", None)
    _validate_preprocessing_cache(prepared, config)

    regression_tasks, classification_tasks = _task_lists(prepared)
    if not regression_tasks:
        raise ValueError(
            "Prepared data contains no regression tasks; production HPO "
            "currently requires at least one regression trait"
        )
    if prepared.metadata.get("data_mode") != "deployment":
        raise ValueError(
            "aquila-train-cv-production requires data prepared by "
            "aquila-data-cv-production"
        )
    all_folds = load_deployment_folds(prepared)
    if args.mode == "cv-refit" and args.folds is not None:
        raise ValueError("--fold is only supported in per-fold mode")
    selected_folds = (
        select_deployment_folds(all_folds, args.folds)
        if args.mode == "per-fold"
        else all_folds
    )

    output_directory = Path(args.output_dir)
    if args.resume and args.mode != "per-fold":
        raise ValueError("--resume is only supported in per-fold mode")
    _prepare_output(
        output_directory,
        args.overwrite,
        resume=args.resume,
    )
    shutil.copy2(Path(args.config).resolve(), output_directory / Path(args.config).name)
    shutil.copy2(
        prepared.directory / "fold_assignments.tsv",
        output_directory / "fold_assignments.tsv",
    )
    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    if train_deterministic_enabled(config):
        print("[INFO] CUDA deterministic algorithms enabled")
    if len(gpu_ids) > 1:
        prepared = PreparedData(
            features=share_memory_tensors(prepared.features),
            targets=prepared.targets.share_memory_(),
            target_mask=prepared.target_mask.share_memory_(),
            metadata=prepared.metadata,
            directory=prepared.directory,
        )

    if args.mode == "cv-refit":
        _run_cv_refit_mode(
            args=args,
            prepared=prepared,
            config=config,
            folds=all_folds,
            regression_tasks=regression_tasks,
            classification_tasks=classification_tasks,
            output_directory=output_directory,
            gpu_ids=gpu_ids,
            started=started,
        )
        return

    train_config = config.get("train", {})
    normalized_hpo = normalize_hpo_config(config.get("hpo", {}))
    context = DeploymentHPOContext(
        prepared_data=prepared,
        config=copy.deepcopy(config),
        folds=tuple(selected_folds),
        regression_tasks=tuple(regression_tasks),
        classification_tasks=tuple(classification_tasks),
        metric=str(normalized_hpo["metric"]),
        direction=str(normalized_hpo["direction"]),
        patience=int(train_config.get("early_stopping_patience", 20)),
        loader_options=_loader_kwargs(train_config),
        output_directory=str(output_directory),
        live_metrics_log=bool(train_config.get("live_metrics_log", False)),
        resume=bool(args.resume),
    )
    fold_summaries = _run_per_fold_mode(context, gpu_ids)
    aggregate = {
        "status": "completed",
        "mode": "per-fold",
        "evaluation_set": "validation",
        "refit_full_dataset": False,
        "prepared_folds": len(all_folds),
        "selected_folds": [int(split.fold) for split in selected_folds],
        "hpo_trials_per_fold": int(normalized_hpo["n_trials"]),
        "elapsed_seconds": time.time() - started,
        "folds": fold_summaries,
        "warning": (
            "Validation metrics were used adaptively for HPO and are not "
            "unbiased test estimates."
        ),
    }
    _write_json(output_directory / "summary.json", aggregate)
    print(
        f"Completed independent HPO for folds "
        f"{aggregate['selected_folds']}; results: {output_directory}"
    )


if __name__ == "__main__":
    main()
