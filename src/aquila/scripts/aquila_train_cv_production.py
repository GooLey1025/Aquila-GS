#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Select hyperparameters by K-fold CV and refit a production model on all data."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import yaml

from aquila.data.dataset import PreparedData, load_prepared_data
from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.scripts.aquila_data_cv_production import (
    DeploymentFold,
    generate_deployment_folds,
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
    detect_gpu_ids,
    execute_gpu_jobs,
    share_memory_tensors,
)
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
from aquila.training.trainer import (
    NestedCVTrainer,
    resolve_training_seed,
)
from aquila.utils import load_config


@dataclass(frozen=True)
class DeploymentCandidateJob:
    """One grid-search candidate evaluated across every deployment CV fold."""

    job_id: int
    parameters: Dict[str, Any]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select Aquila hyperparameters with K-fold CV, then train one "
            "production model on the complete prepared dataset."
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
    validate_deployment_folds(folds, prepared.sample_count)
    return folds


def _trait_tasks(
    prepared: PreparedData,
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str],
) -> list[str]:
    trait_names = list(prepared.metadata["trait_names"])
    tasks = ["regression"] * len(regression_tasks) + ["classification"] * len(
        classification_tasks
    )
    if len(tasks) != len(trait_names):
        raise ValueError("Prepared trait task metadata does not match target columns")
    return tasks


def _fit_processor(
    prepared: PreparedData,
    config: Mapping[str, Any],
    indices: np.ndarray,
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str],
) -> PerTraitPreprocessor:
    preprocessing = config.get("preprocessing", {})
    return PerTraitPreprocessor(
        skew_threshold=float(preprocessing.get("skew_threshold", 2.0)),
        epsilon=float(preprocessing.get("epsilon", 1e-8)),
    ).fit(
        prepared.targets,
        prepared.target_mask,
        indices,
        prepared.metadata["trait_names"],
        trait_tasks=_trait_tasks(
            prepared,
            regression_tasks,
            classification_tasks,
        ),
    )


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
):
    """Train one HPO candidate on one runtime-generated CV split."""
    candidate_config = merge_config(config, parameters)
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
    train_config = candidate_config.get("train", {})
    batch_size = int(train_config.get("batch_size", 32))
    effective_loader_options = dict(loader_options)
    gpu_resident = bool(train_config.get("gpu_resident", True)) and str(
        device
    ).startswith("cuda")
    if gpu_resident:
        train_data = _prepared_on_device(train_data, device)
        valid_data = _prepared_on_device(valid_data, device)
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


def _run_hpo(
    context: DeploymentHPOContext,
    gpu_ids: Sequence[int],
) -> HPOResult:
    normalized = normalize_hpo_config(context.config.get("hpo", {}))
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


def _prepare_output(output_directory: Path, overwrite: bool) -> None:
    if output_directory.exists() and any(output_directory.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Production output already exists: {output_directory}; "
                "use --overwrite to replace it"
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


def main() -> None:
    args = parse_args()
    started = time.time()
    prepared = load_prepared_data(args.data_dir)
    config = load_config(args.config)
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
            "Prepared data contains no regression tasks; production CV HPO "
            "currently requires at least one regression trait"
        )
    if prepared.metadata.get("data_mode") != "deployment":
        raise ValueError(
            "aquila-train-cv-production requires data prepared by "
            "aquila-data-cv-production"
        )
    folds = load_deployment_folds(prepared)
    cv_seed = int(prepared.metadata.get("cv_seed", prepared.metadata["seed"]))

    output_directory = Path(args.output_dir)
    _prepare_output(output_directory, args.overwrite)
    shutil.copy2(Path(args.config).resolve(), output_directory / Path(args.config).name)
    shutil.copy2(
        prepared.directory / "fold_assignments.tsv",
        output_directory / "fold_assignments.tsv",
    )

    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu"
    if train_deterministic_enabled(config):
        print("[INFO] CUDA deterministic algorithms enabled")
    if len(gpu_ids) > 1 and normalize_hpo_config(config.get("hpo", {}))[
        "method"
    ] == "grid":
        prepared = PreparedData(
            features=share_memory_tensors(prepared.features),
            targets=prepared.targets.share_memory_(),
            target_mask=prepared.target_mask.share_memory_(),
            metadata=prepared.metadata,
            directory=prepared.directory,
        )

    train_config = config.get("train", {})
    normalized_hpo = normalize_hpo_config(config.get("hpo", {}))
    context = DeploymentHPOContext(
        prepared_data=prepared,
        config=copy.deepcopy(config),
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
            "model_type": "production",
            "evaluation_set": "none",
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


if __name__ == "__main__":
    main()
