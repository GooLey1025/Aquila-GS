#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Bayesian HPO on one 4:1 split, then a single outer-test prediction.

For every selected outer fold this runs ``n_trials`` sequential TPE trials
(default 200). Each trial trains on the prepared HPO training split and
selects its epoch on the validation split. The winning early-stopping
checkpoint is applied once to the untouched outer test. The model is not
refit on the full outer training set, and there is no inner cross-validation.

A current-fold GWAS dataset is bound to its outer fold, so one Rice655
directory contributes 200 trainings and one outer-test prediction.
"""

from __future__ import annotations

import argparse
import copy
import math
import platform
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from aquila.data.cv import resolve_outer_folds
from aquila.data.dataset import PreparedData, load_prepared_data
from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.scripts.aquila_train_cv import (
    _aggregate_outer_test_metrics,
    _json_safe,
    _loader_kwargs,
    _make_seeded_model,
    _prepared_on_device,
    _sequence_lengths,
    _task_lists,
    _trainer_kwargs,
    _validate_preprocessing_cache,
)
from aquila.scripts.aquila_train_cv_production import (
    _atomic_torch_save,
    _candidate_from_optuna_trial,
    _load_processed_targets,
    _load_running_best_training,
    _materialize_best_training,
    _metric_from_metrics,
    _processed_subset,
    _train_deployment_fold,
    _write_json,
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
from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import (
    CandidateResult,
    HPOResult,
    InnerFoldResult,
    finite_parameter_candidates,
    merge_config,
    normalize_hpo_config,
    parameter_key,
    select_best_candidate,
    unique_bayesian_trial,
)
from aquila.training.trainer import (
    NestedCVTrainer,
    TrainingResult,
    resolve_training_seed,
)
from aquila.utils import load_config


DEFAULT_TRIALS = 200


@dataclass(frozen=True)
class BenchmarkSplit:
    """One outer test plus the single train/validation holdout."""

    fold: int
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class BenchmarkFoldJob:
    """One independent outer-fold study."""

    job_id: int
    fold_id: int


@dataclass(frozen=True)
class BenchmarkHPOContext:
    """Spawn-safe inputs shared by outer-fold workers."""

    prepared_data: PreparedData
    config: Dict[str, Any]
    splits: tuple[BenchmarkSplit, ...]
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
            "Run 200-trial Bayesian HPO on one 4:1 train/validation split "
            "per outer fold, then predict that fold's outer test once."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Benchmark prepared-data directory.")
    parser.add_argument("--config", required=True, help="Training/HPO YAML configuration.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="benchmark_experiment",
        help="Benchmark run output directory.",
    )
    parser.add_argument(
        "--fold",
        "--folds",
        dest="folds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Zero-based outer folds to run, for example --fold 0 1 2. "
            "Omit to run every prepared outer fold. A current-fold GWAS "
            "dataset can only run its bound fold."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=f"Bayesian trial budget per outer fold (default: {DEFAULT_TRIALS}).",
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
        help="Replace an existing nonempty output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each outer fold from its SQLite study and best checkpoint.",
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
        help="Write per-epoch validation metrics for every trial.",
    )
    parser.add_argument(
        "--use-deterministic",
        action="store_true",
        help="Force deterministic CUDA algorithms.",
    )
    return parser.parse_args()


def configure_benchmark_hpo(
    config: Mapping[str, Any],
    n_trials: int | None,
) -> Dict[str, Any]:
    """Force the benchmark protocol: enabled Bayesian search, 200 trials."""
    configured = copy.deepcopy(dict(config))
    hpo = configured.setdefault("hpo", {})
    if not isinstance(hpo, dict):
        raise TypeError("hpo configuration must be a mapping")
    normalized = normalize_hpo_config(hpo)
    if normalized["method"] != "bayesian":
        raise ValueError(
            "aquila_train_cv_benchmark requires hpo.method: bayesian"
        )
    resolved = DEFAULT_TRIALS if n_trials is None else int(n_trials)
    if resolved < 1:
        raise ValueError("n-trials must be positive")
    hpo["enabled"] = True
    hpo["method"] = "bayesian"
    hpo["n_trials"] = resolved
    return configured


def load_benchmark_splits(prepared: PreparedData) -> list[BenchmarkSplit]:
    """Load the persisted outer-test and 4:1 holdout indices."""
    if prepared.metadata.get("data_mode") != "benchmark":
        raise ValueError(
            "aquila_train_cv_benchmark requires data prepared by "
            "aquila_data_cv_benchmark.py"
        )
    fold_count = int(prepared.metadata["outer_folds"])
    splits = []
    for fold_id in range(fold_count):
        fold_path = prepared.directory / "cv" / f"outer_fold_{fold_id}"
        splits.append(
            BenchmarkSplit(
                fold=fold_id,
                train=np.load(fold_path / "train_idx.npy", allow_pickle=False),
                valid=np.load(fold_path / "valid_idx.npy", allow_pickle=False),
                test=np.load(fold_path / "test_idx.npy", allow_pickle=False),
            )
        )
    return splits


def _fold_directory(output_directory: str | Path, fold_id: int) -> Path:
    return Path(output_directory) / f"fold_{int(fold_id)}"


def _cache_hpo_splits(
    *,
    prepared: PreparedData,
    split: BenchmarkSplit,
    device: str,
    gpu_resident: bool,
) -> tuple[PreparedData, PreparedData]:
    """Load HPO train/validation tensors once and optionally keep them on GPU."""
    fold_path = prepared.directory / "cv" / f"outer_fold_{split.fold}"
    n_traits = int(prepared.targets.shape[1])
    train_data = _processed_subset(
        prepared,
        split.train,
        _load_processed_targets(
            fold_path / "Y_train_processed.pt",
            len(split.train),
            n_traits,
        ),
        targets_are_subset=True,
    )
    valid_data = _processed_subset(
        prepared,
        split.valid,
        _load_processed_targets(
            fold_path / "Y_valid_processed.pt",
            len(split.valid),
            n_traits,
        ),
        targets_are_subset=True,
    )
    if gpu_resident and str(device).startswith("cuda"):
        train_data = _prepared_on_device(train_data, device)
        valid_data = _prepared_on_device(valid_data, device)
    return train_data, valid_data


def _benchmark_checkpoint(
    *,
    prepared: PreparedData,
    split: BenchmarkSplit,
    config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    training: TrainingResult,
    processor: PerTraitPreprocessor,
) -> Dict[str, Any]:
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
    checkpoint_metadata["benchmark_split"] = {
        "outer_fold": int(split.fold),
        "train_samples": int(len(split.train)),
        "validation_samples": int(len(split.valid)),
        "test_samples": int(len(split.test)),
        "evaluation_set": "outer_test",
        "refit_full_dataset": False,
    }
    return {
        **training.checkpoint_state,
        "config": selected_config,
        "metadata": checkpoint_metadata,
        "preprocessing": processor.to_dict(),
        "benchmark_model": True,
        "outer_fold": int(split.fold),
        "selected_hyperparameters": dict(parameters),
        "final_epoch": int(training.best_epoch),
        "scheduler_epochs": int(
            selected_config.get("train", {}).get("num_epochs", training.best_epoch)
        ),
        "evaluation_set": "outer_test",
        "refit_full_dataset": False,
    }


def _save_running_best(
    *,
    context: BenchmarkHPOContext,
    split: BenchmarkSplit,
    candidate: CandidateResult,
    training: TrainingResult,
) -> None:
    fold_directory = _fold_directory(context.output_directory, split.fold)
    processor = PerTraitPreprocessor.load_json(
        context.prepared_data.directory
        / "cv"
        / f"outer_fold_{split.fold}"
        / "preprocessing.json"
    )
    checkpoint = _benchmark_checkpoint(
        prepared=context.prepared_data,
        split=split,
        config=context.config,
        parameters=candidate.parameters,
        training=training,
        processor=processor,
    )
    checkpoint["hpo_candidate_id"] = int(candidate.candidate_id)
    checkpoint["hpo_objective"] = float(candidate.objective)
    _atomic_torch_save(checkpoint, fold_directory / "best_model.pt")
    _write_json(fold_directory / "training_history.json", training.history)


def _run_benchmark_study(
    *,
    context: BenchmarkHPOContext,
    split: BenchmarkSplit,
    device: str,
) -> tuple[HPOResult, TrainingResult]:
    """Run or resume one TPE study on a single train/validation holdout."""
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Bayesian HPO requires optuna") from error

    normalized = normalize_hpo_config(context.config.get("hpo", {}))
    if normalized["method"] != "bayesian":
        raise ValueError("benchmark HPO requires hpo.method: bayesian")
    sampler = optuna.samplers.TPESampler(seed=normalized["seed"])
    fold_output = _fold_directory(context.output_directory, split.fold)
    fold_output.mkdir(parents=True, exist_ok=True)
    storage = None
    study_name = None
    if context.resume:
        storage = f"sqlite:///{(fold_output / 'optuna.db').resolve()}"
        study_name = f"aquila-benchmark-fold-{split.fold}"
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
                f"Outer fold {split.fold} has persistent trials but no matching "
                "best_model.pt; restart with --overwrite"
            )
        best_training = loaded[1]
    maximize = normalized["direction"] == "maximize"
    base_train = context.config.get("train", {})
    fold_data = _cache_hpo_splits(
        prepared=context.prepared_data,
        split=split,
        device=device,
        gpu_resident=bool(base_train.get("gpu_resident", True)),
    )
    print(
        f"[INFO] Outer fold {split.fold}: {len(candidates)}/"
        f"{normalized['n_trials']} completed TPE trials; "
        f"train={len(split.train)} valid={len(split.valid)} "
        f"test={len(split.test)} held out; data cached on {device}",
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
                f"Non-finite HPO objective in outer fold {split.fold}, "
                f"trial {trial.number}"
            )
        candidate = CandidateResult(
            candidate_id=int(trial.number),
            parameters=copy.deepcopy(parameters),
            objective=float(objective),
            inner_results=(
                InnerFoldResult(
                    inner_fold=int(split.fold),
                    metric=float(objective),
                    best_epoch=int(training.best_epoch),
                    metrics=copy.deepcopy(dict(training.best_metrics)),
                ),
            ),
        )
        candidates.append(candidate)
        seen.add(parameter_key(parameters))
        trial.set_user_attr("best_epoch", int(training.best_epoch))
        trial.set_user_attr("metrics", _json_safe(dict(training.best_metrics)))
        study.tell(trial, objective)
        is_better = best_candidate is None or (
            objective > best_candidate.objective
            if maximize
            else objective < best_candidate.objective
        )
        if is_better:
            best_candidate = candidate
            best_training = _materialize_best_training(training)
            _save_running_best(
                context=context,
                split=split,
                candidate=candidate,
                training=best_training,
            )
        duplicate_note = (
            f" skipped_duplicates={duplicate_count}" if duplicate_count else ""
        )
        print(
            f"[Outer fold {split.fold} Bayesian HPO] "
            f"unique={len(candidates)}/{normalized['n_trials']} "
            f"optuna_trial={trial.number + 1} objective={objective:.6f} "
            f"best_epoch={training.best_epoch}{duplicate_note}",
            flush=True,
        )
    result = select_best_candidate(
        candidates,
        normalized["direction"],
        method="bayesian",
    )
    if best_training is None or best_candidate is None:
        raise RuntimeError(f"Outer fold {split.fold} HPO completed without a model")
    if result.best.candidate_id != best_candidate.candidate_id:
        raise RuntimeError("In-memory best checkpoint does not match HPO winner")
    return result, best_training


def _evaluate_outer_test(
    *,
    context: BenchmarkHPOContext,
    split: BenchmarkSplit,
    hpo_result: HPOResult,
    training: TrainingResult,
    device: str,
) -> Dict[str, Any]:
    """Score the validation-selected checkpoint once on the outer test."""
    selected_config = merge_config(context.config, hpo_result.best.parameters)
    fold_path = (
        context.prepared_data.directory / "cv" / f"outer_fold_{split.fold}"
    )
    processor = PerTraitPreprocessor.load_json(fold_path / "preprocessing.json")
    n_traits = int(context.prepared_data.targets.shape[1])
    test_data = _processed_subset(
        context.prepared_data,
        split.test,
        _load_processed_targets(
            fold_path / "Y_test_processed.pt",
            len(split.test),
            n_traits,
        ),
        targets_are_subset=True,
    )
    train_config = selected_config.get("train", {})
    batch_size = int(train_config.get("batch_size", 32))
    loader_options = dict(context.loader_options)
    if bool(train_config.get("gpu_resident", True)) and str(device).startswith("cuda"):
        test_data = _prepared_on_device(test_data, device)
        loader_options = {"num_workers": 0, "pin_memory": False}
    test_loader = test_data.loader(
        np.arange(len(split.test)),
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )
    training_seed = resolve_training_seed(
        selected_config,
        fallback=int(context.prepared_data.metadata.get("seed", 42)),
    )
    trainer = NestedCVTrainer(
        _make_seeded_model(
            selected_config,
            test_data,
            context.regression_tasks,
            context.classification_tasks,
            training_seed,
        ),
        num_regression_tasks=len(context.regression_tasks),
        num_classification_tasks=len(context.classification_tasks),
        **_trainer_kwargs(
            selected_config,
            device,
            context.regression_tasks,
            training_seed,
        ),
    )
    trainer.model.load_state_dict(training.best_state_dict)
    normalized = trainer.evaluate(test_loader)
    n_regression = len(context.regression_tasks)
    trait_names = list(context.prepared_data.metadata["trait_names"])
    full_predictions = np.zeros(
        (normalized.predictions.shape[0], len(trait_names)),
        dtype=np.float64,
    )
    full_mask = np.zeros_like(full_predictions, dtype=bool)
    full_predictions[:, :n_regression] = normalized.predictions
    full_mask[:, :n_regression] = True
    original_predictions = np.asarray(
        processor.inverse(full_predictions, full_mask)
    )[:, :n_regression]
    raw_targets = (
        context.prepared_data.targets[split.test].cpu().numpy()[:, :n_regression]
    )
    raw_mask = (
        context.prepared_data.target_mask[split.test].cpu().numpy()[:, :n_regression]
    )
    original = evaluate_regression(
        original_predictions,
        raw_targets,
        raw_mask,
        context.regression_tasks,
    )
    return {
        "metrics": {
            "normalized": normalized.metrics,
            "original": original.metrics,
        },
        "predictions": original_predictions,
        "raw_targets": raw_targets,
        "raw_mask": raw_mask,
        "processor": processor,
        "selected_config": selected_config,
        "training_seed": training_seed,
    }


def _persist_benchmark_fold(
    *,
    context: BenchmarkHPOContext,
    split: BenchmarkSplit,
    hpo_result: HPOResult,
    training: TrainingResult,
    evaluation: Mapping[str, Any],
    device: str,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    fold_directory = _fold_directory(context.output_directory, split.fold)
    fold_directory.mkdir(parents=True, exist_ok=True)
    processor = evaluation["processor"]
    selected_config = evaluation["selected_config"]
    checkpoint = _benchmark_checkpoint(
        prepared=context.prepared_data,
        split=split,
        config=context.config,
        parameters=hpo_result.best.parameters,
        training=training,
        processor=processor,
    )
    checkpoint["hpo_candidate_id"] = int(hpo_result.best.candidate_id)
    checkpoint["hpo_objective"] = float(hpo_result.best.objective)
    checkpoint["outer_test_metrics"] = _json_safe(evaluation["metrics"])
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
            "outer_fold": int(split.fold),
            "n_trials": len(hpo_result.candidates),
            "best_candidate_id": hpo_result.best.candidate_id,
            "best_parameters": dict(hpo_result.best.parameters),
            "best_validation": hpo_result.best.objective,
            "best_epoch": int(training.best_epoch),
            "refit_full_dataset": False,
            "candidates": [
                {
                    "candidate_id": int(candidate.candidate_id),
                    "parameters": dict(candidate.parameters),
                    "objective": float(candidate.objective),
                    "best_epoch": int(candidate.inner_results[0].best_epoch),
                }
                for candidate in hpo_result.candidates
            ],
        },
    )
    _write_json(fold_directory / "metrics.json", evaluation["metrics"])
    trait_names = list(context.prepared_data.metadata["trait_names"])
    prediction_values: Dict[str, Any] = {
        "SampleID": [
            context.prepared_data.metadata["sample_ids"][int(index)]
            for index in split.test
        ]
    }
    original_predictions = np.asarray(evaluation["predictions"])
    raw_targets = np.asarray(evaluation["raw_targets"])
    raw_mask = np.asarray(evaluation["raw_mask"])
    for trait_index, trait_name in enumerate(context.regression_tasks):
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
    del trait_names
    training_seed = int(evaluation["training_seed"])
    runtime = {
        "outer_fold": int(split.fold),
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda")
            else None
        ),
        "training_seed": training_seed,
        "elapsed_seconds": elapsed_seconds,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    _write_json(fold_directory / "runtime.json", runtime)
    summary = {
        "outer_fold": int(split.fold),
        "status": "completed",
        "mode": "benchmark",
        "evaluation_set": "outer_test",
        "refit_full_dataset": False,
        "train_samples": int(len(split.train)),
        "validation_samples": int(len(split.valid)),
        "test_samples": int(len(split.test)),
        "hpo_trials": len(hpo_result.candidates),
        "hpo_metric": normalized["metric"],
        "best_validation": float(hpo_result.best.objective),
        "best_epoch": int(training.best_epoch),
        "selected_hyperparameters": dict(hpo_result.best.parameters),
        "metrics": evaluation["metrics"],
        "checkpoint": str((fold_directory / "best_model.pt").resolve()),
        "runtime": runtime,
    }
    _write_json(fold_directory / "summary.json", summary)
    return summary


def _benchmark_worker(
    job: BenchmarkFoldJob,
    device: str,
    context: BenchmarkHPOContext,
) -> Dict[str, Any]:
    started = time.time()
    configure_cuda_runtime(
        device,
        deterministic=train_deterministic_enabled(context.config),
    )
    split = {item.fold: item for item in context.splits}[job.fold_id]
    hpo_result, training = _run_benchmark_study(
        context=context,
        split=split,
        device=device,
    )
    evaluation = _evaluate_outer_test(
        context=context,
        split=split,
        hpo_result=hpo_result,
        training=training,
        device=device,
    )
    return _persist_benchmark_fold(
        context=context,
        split=split,
        hpo_result=hpo_result,
        training=training,
        evaluation=evaluation,
        device=device,
        elapsed_seconds=time.time() - started,
    )


def _prepare_output(output_directory: Path, overwrite: bool, *, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if output_directory.exists() and any(output_directory.iterdir()):
        if resume:
            return
        if not overwrite:
            raise FileExistsError(
                f"Benchmark output already exists: {output_directory}; "
                "use --overwrite to replace it or --resume to continue it"
            )
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    started = time.time()
    prepared = load_prepared_data(args.data_dir)
    config = configure_benchmark_hpo(load_config(args.config), args.n_trials)
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
            "Prepared data contains no regression tasks; benchmark HPO "
            "currently requires at least one regression trait"
        )
    selected_ids = resolve_outer_folds(args.folds, prepared.metadata, option_name="--fold")
    splits = [
        split
        for split in load_benchmark_splits(prepared)
        if split.fold in set(selected_ids)
    ]
    output_directory = Path(args.output_dir)
    _prepare_output(output_directory, args.overwrite, resume=args.resume)
    shutil.copy2(Path(args.config).resolve(), output_directory / Path(args.config).name)
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
    train_config = config.get("train", {})
    normalized_hpo = normalize_hpo_config(config.get("hpo", {}))
    context = BenchmarkHPOContext(
        prepared_data=prepared,
        config=copy.deepcopy(config),
        splits=tuple(splits),
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
    jobs = [
        BenchmarkFoldJob(job_id=int(split.fold), fold_id=int(split.fold))
        for split in splits
    ]
    results = execute_gpu_jobs(
        jobs,
        _benchmark_worker,
        gpu_ids,
        worker_args=(context,),
        raise_on_error=True,
        deterministic=train_deterministic_enabled(config),
    )
    fold_summaries = [dict(result.value) for result in results]
    outer_test_summary = _aggregate_outer_test_metrics(fold_summaries)
    aggregate = {
        "status": "completed",
        "mode": "benchmark",
        "protocol": "10-outer-test/single-4:1-holdout/bayesian-hpo",
        "evaluation_set": "outer_test",
        "refit_full_dataset": False,
        "selected_folds": [int(split.fold) for split in splits],
        "hpo_trials_per_fold": int(normalized_hpo["n_trials"]),
        "model_trainings": int(normalized_hpo["n_trials"]) * len(splits),
        "outer_test_predictions": len(splits),
        "elapsed_seconds": time.time() - started,
        "outer_test_summary": outer_test_summary,
        "folds": fold_summaries,
    }
    _write_json(output_directory / "summary.json", aggregate)
    print(
        f"Completed benchmark HPO for outer folds {aggregate['selected_folds']}; "
        f"{aggregate['model_trainings']} trainings and "
        f"{aggregate['outer_test_predictions']} outer-test predictions; "
        f"results: {output_directory}"
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
