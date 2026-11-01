#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Train outer-holdout Aquila models and score the early-stopped checkpoint.

Each outer fold searches the grid on one inner train/validation split. The
winning early-stopped weights are applied directly to that fold's outer test.
The search does not average inner folds and does not refit on the full outer
training set.
"""

from __future__ import annotations

import argparse
import json
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

from aquila.data.cv import load_fold_indices, resolve_outer_folds
from aquila.data.dataset import PreparedData, load_prepared_data
from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.scripts.aquila_train_cv import (
    _aggregate_outer_test_metrics,
    _candidate_records,
    _copy_params_yaml,
    _features_for_split,
    _loader_kwargs,
    _make_seeded_model,
    _prepare_output,
    _prepared_on_device,
    _processed_view,
    _selected_per_fold,
    _sequence_lengths,
    _task_lists,
    _train_inner_fold,
    _trainer_kwargs,
    _validate_preprocessing_cache,
    _write_json,
)
from aquila.training.cuda_runtime import (
    resolve_train_deterministic,
    train_deterministic_enabled,
)
from aquila.training.distributed import (
    PersistentGPUPool,
    derive_seed,
    detect_gpu_ids,
)
from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import (
    CandidateResult,
    InnerFoldResult,
    merge_config,
    normalize_hpo_config,
    select_best_candidate,
)
from aquila.training.trainer import NestedCVTrainer, resolve_training_seed
from aquila.utils import load_config


@dataclass(frozen=True)
class HoldoutJob:
    """One candidate fit or one outer-test evaluation."""

    job_id: int
    kind: str  # "candidate" | "test"
    fold_id: int
    candidate_id: int = -1
    parameters: Dict[str, Any] | None = None


@dataclass(frozen=True)
class HoldoutPoolContext:
    """Spawn-safe inputs shared by holdout GPU workers."""

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


def require_holdout_data(prepared: PreparedData) -> None:
    """Reject nested-CV directories that still expect an outer-train refit."""
    protocol = prepared.metadata.get("protocol")
    if protocol != "outer_holdout":
        raise ValueError(
            "aquila_train_cv_holdout.py requires data prepared by "
            "aquila_data_cv_holdout.py (metadata protocol 'outer_holdout'); "
            f"got {protocol!r}"
        )
    if int(prepared.metadata.get("inner_folds", 0)) != 1:
        raise ValueError("outer_holdout data must contain exactly one inner split")


def select_holdout_winner(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    direction: str,
):
    """Pick the early-stopped candidate with the best inner-validation score.

    Ties keep the lowest candidate id, matching grid HPO selection.
    """
    candidates = []
    for record in records:
        metrics = dict(record["best_metrics"])
        if "avg_pearson" in metrics:
            score = float(metrics["avg_pearson"])
        else:
            from aquila.training.trainer import _metric_value

            score = float(_metric_value(metrics, metric))
        candidates.append(
            CandidateResult(
                candidate_id=int(record["candidate_id"]),
                parameters=dict(record["parameters"]),
                objective=score,
                inner_results=(
                    InnerFoldResult(
                        0,
                        score,
                        int(record["best_epoch"]),
                        metrics,
                    ),
                ),
            )
        )
    return select_best_candidate(candidates, direction, method="grid")


def _candidate_directory(output_directory: Path, fold_id: int, candidate_id: int) -> Path:
    return output_directory / f"fold_{fold_id}" / f"candidate_{candidate_id}"


def _candidate_job_id(fold_id: int, candidate_id: int) -> int:
    return int(fold_id) * 1_000_000 + int(candidate_id)


def _test_job_id(fold_id: int) -> int:
    return int(fold_id) * 1_000_000 + 900_000


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _completed_candidate(directory: Path) -> Dict[str, Any] | None:
    result_path = directory / "result.json"
    checkpoint_path = directory / "best_model.pt"
    if result_path.is_file() and checkpoint_path.is_file():
        return _read_json(result_path)
    return None


def _completed_fold_summary(fold_directory: Path) -> Dict[str, Any] | None:
    summary_path = fold_directory / "fold_summary.json"
    metrics_path = fold_directory / "metrics.json"
    if summary_path.is_file() and metrics_path.is_file():
        return _read_json(summary_path)
    return None


def _configure_worker_device(device: str, config: Mapping[str, Any]) -> None:
    from aquila.scripts.aquila_train_cv import _configure_worker_device as configure

    configure(device, config)


def _save_candidate(
    *,
    training: Any,
    candidate_config: Mapping[str, Any],
    directory: Path,
    fold_id: int,
    candidate_id: int,
    parameters: Mapping[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": training.best_state_dict,
        "config": dict(candidate_config),
        "best_epoch": int(training.best_epoch),
        "best_metrics": dict(training.best_metrics),
        "parameters": dict(parameters),
        "candidate_id": int(candidate_id),
        "outer_fold": int(fold_id),
        "protocol": "outer_holdout",
        "refit": False,
    }
    torch.save(checkpoint, directory / "best_model.pt")
    _write_json(directory / "training_history.json", training.history)
    # result.json is written last so a resume can require both artifacts.
    _write_json(
        directory / "result.json",
        {
            "candidate_id": int(candidate_id),
            "outer_fold": int(fold_id),
            "best_epoch": int(training.best_epoch),
            "best_metrics": dict(training.best_metrics),
            "parameters": dict(parameters),
        },
    )


def _train_candidate(job: HoldoutJob, device: str, context: HoldoutPoolContext) -> None:
    if job.parameters is None:
        raise ValueError("Candidate job is missing parameters")
    worker_seed = derive_seed(context.global_seed, int(job.fold_id))
    directory = _candidate_directory(
        Path(context.output_directory),
        int(job.fold_id),
        int(job.candidate_id),
    )
    metrics_log_path = None
    if context.live_metrics_log:
        metrics_log_path = directory / "inner_0" / "metrics.jsonl"
    candidate_config = merge_config(context.config, job.parameters)
    training = _train_inner_fold(
        prepared=context.prepared_data,
        config=context.config,
        fold_id=int(job.fold_id),
        inner_fold=0,
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
    _save_candidate(
        training=training,
        candidate_config=candidate_config,
        directory=directory,
        fold_id=int(job.fold_id),
        candidate_id=int(job.candidate_id),
        parameters=job.parameters,
    )


def _score_outer_test(
    *,
    prepared: PreparedData,
    config: Mapping[str, Any],
    fold_id: int,
    output_directory: Path,
    device: str,
    worker_seed: int,
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str],
    loader_options: Mapping[str, Any],
    metric: str,
    direction: str,
    started: float,
) -> Dict[str, Any]:
    """Score the selected early-stopped checkpoint on the outer test split.

    This loads the winning candidate weights and evaluates them. It does not
    train for a fixed number of epochs on the outer training set.
    """
    fold_directory = output_directory / f"fold_{fold_id}"
    fold_directory.mkdir(parents=True, exist_ok=True)
    records = []
    candidate_dirs = sorted(
        path
        for path in fold_directory.glob("candidate_*")
        if path.is_dir() and (path / "result.json").is_file()
    )
    for candidate_dir in candidate_dirs:
        record = _completed_candidate(candidate_dir)
        if record is None:
            raise FileNotFoundError(
                f"Candidate checkpoint is incomplete: {candidate_dir}"
            )
        records.append(record)
    if not records:
        raise RuntimeError(f"Fold {fold_id} finished without candidate results")
    hpo_result = select_holdout_winner(
        records,
        metric=metric,
        direction=direction,
    )
    winner = hpo_result.best
    winner_dir = _candidate_directory(
        output_directory,
        fold_id,
        int(winner.candidate_id),
    )
    checkpoint = torch.load(
        winner_dir / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("candidate_id", -1)) != int(winner.candidate_id):
        raise RuntimeError(
            f"Checkpoint candidate id does not match the selected winner "
            f"for fold {fold_id}"
        )
    selected_config = merge_config(config, winner.parameters)
    inner_path = (
        prepared.directory
        / "cv"
        / f"outer_fold_{fold_id}"
        / "inner_fold_0"
    )
    processor = PerTraitPreprocessor.load_json(inner_path / "preprocessing.json")
    outer = load_fold_indices(prepared.directory, fold_id)
    test_data = _processed_view(
        prepared,
        outer["test"],
        inner_path / "Y_test_processed.pt",
        features=_features_for_split(
            prepared,
            outer["test"],
            fold_id,
            0,
        ),
    )
    batch_size = int(selected_config.get("train", {}).get("batch_size", 32))
    eval_loader_options = dict(loader_options)
    if bool(selected_config.get("train", {}).get("gpu_resident", True)) and str(
        device
    ).startswith("cuda"):
        test_data = _prepared_on_device(test_data, device)
        eval_loader_options = {"num_workers": 0, "pin_memory": False}
    test_loader = test_data.loader(
        np.arange(len(outer["test"])),
        batch_size=batch_size,
        shuffle=False,
        **eval_loader_options,
    )
    training_seed = resolve_training_seed(
        selected_config,
        fallback=int(prepared.metadata.get("seed", 42)),
    )
    trainer = NestedCVTrainer(
        _make_seeded_model(
            selected_config,
            test_data,
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
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    normalized = trainer.evaluate(test_loader)

    trait_names = list(prepared.metadata["trait_names"])
    n_regression = len(regression_tasks)
    full_predictions = np.zeros(
        (normalized.predictions.shape[0], len(trait_names)),
        dtype=np.float64,
    )
    full_mask = np.zeros_like(full_predictions, dtype=bool)
    prediction_mask = np.ones_like(normalized.predictions, dtype=bool)
    full_predictions[:, :n_regression] = normalized.predictions
    full_mask[:, :n_regression] = prediction_mask
    inverted = processor.inverse(full_predictions, full_mask)
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
    best_epoch = int(winner.inner_results[0].best_epoch)
    checkpoint_metadata = dict(prepared.metadata)
    checkpoint_metadata["sequence_lengths"] = _sequence_lengths(test_data.features)
    saved_checkpoint = {
        "model_state_dict": checkpoint["model_state_dict"],
        "config": selected_config,
        "metadata": checkpoint_metadata,
        "preprocessing": processor.to_dict(),
        "outer_fold": fold_id,
        "candidate_id": int(winner.candidate_id),
        "selected_hyperparameters": dict(winner.parameters),
        "best_epoch": best_epoch,
        "final_epoch": best_epoch,
        "protocol": "outer_holdout",
        "refit": False,
    }
    torch.save(saved_checkpoint, fold_directory / "best_model.pt")
    processor.save_json(fold_directory / "preprocessing.json")
    with (fold_directory / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(selected_config, handle, sort_keys=False)
    _write_json(
        fold_directory / "hpo_results.json",
        {
            "method": hpo_result.method,
            "protocol": "outer_holdout",
            "direction": hpo_result.direction,
            "best_candidate_id": winner.candidate_id,
            "best_parameters": dict(winner.parameters),
            "best_valid_r": winner.objective,
            "best_epoch": best_epoch,
            "refit": False,
            "candidates": _candidate_records(hpo_result),
        },
    )
    metrics = {
        "normalized": normalized.metrics,
        "original": original.metrics,
    }
    _write_json(fold_directory / "metrics.json", metrics)
    history_path = winner_dir / "training_history.json"
    if history_path.is_file():
        shutil.copy2(history_path, fold_directory / "training_history.json")

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
        "protocol": "outer_holdout",
        "refit": False,
        "device": device,
        "gpu_name": (
            torch.cuda.get_device_name(torch.device(device))
            if str(device).startswith("cuda")
            else None
        ),
        "worker_seed": worker_seed,
        "training_seed": training_seed,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    _write_json(fold_directory / "runtime.json", runtime)
    summary = {
        "outer_fold": fold_id,
        "status": "completed",
        "protocol": "outer_holdout",
        "refit": False,
        "best_valid_r": winner.objective,
        "best_valid_r_mean": winner.objective,
        "selected_hyperparameters": dict(winner.parameters),
        "inner_best_epochs": [best_epoch],
        "best_epoch": best_epoch,
        "final_epoch": best_epoch,
        "metrics": metrics,
        "runtime": runtime,
    }
    _write_json(fold_directory / "fold_summary.json", summary)
    return summary


def _holdout_gpu_worker(
    job: HoldoutJob,
    device: str,
    context: HoldoutPoolContext,
) -> Dict[str, Any]:
    """Train one grid candidate, or score the selected checkpoint on outer test."""
    _configure_worker_device(device, context.config)
    worker_seed = derive_seed(context.global_seed, int(job.fold_id))
    if job.kind == "candidate":
        _train_candidate(job, device, context)
        return {
            "kind": "candidate",
            "fold_id": int(job.fold_id),
            "candidate_id": int(job.candidate_id),
        }
    if job.kind == "test":
        summary = _score_outer_test(
            prepared=context.prepared_data,
            config=context.config,
            fold_id=int(job.fold_id),
            output_directory=Path(context.output_directory),
            device=device,
            worker_seed=worker_seed,
            regression_tasks=context.regression_tasks,
            classification_tasks=context.classification_tasks,
            loader_options=context.loader_options,
            metric=context.metric,
            direction=context.direction,
            started=time.time(),
        )
        return {"kind": "test", "fold_id": int(job.fold_id), "summary": summary}
    raise ValueError(f"Unknown holdout job kind: {job.kind!r}")


def _queue_fold_test(
    submit,
    *,
    fold_id: int,
    completed: Mapping[int, int],
    expected: Mapping[int, int],
    submitted: Dict[int, bool],
) -> None:
    if completed[fold_id] != expected[fold_id] or submitted[fold_id]:
        return
    submit(
        HoldoutJob(
            job_id=_test_job_id(fold_id),
            kind="test",
            fold_id=int(fold_id),
        )
    )
    submitted[fold_id] = True


def _run_holdout_grid(
    *,
    prepared: PreparedData,
    config: Mapping[str, Any],
    output_directory: Path,
    selected_folds: Sequence[int],
    gpu_ids: Sequence[int],
    global_seed: int,
    resume: bool,
) -> list[Dict[str, Any]]:
    train_config = config.get("train", {})
    normalized_hpo = normalize_hpo_config(config.get("hpo", {}))
    if normalized_hpo["method"] != "grid":
        raise ValueError("Outer-holdout training supports grid HPO only")
    from aquila.training.hpo import generate_grid_candidates

    parameter_sets = generate_grid_candidates(normalized_hpo["parameters"])
    if not parameter_sets:
        raise ValueError("Grid HPO parameter space is empty")
    if len(parameter_sets) >= 900_000:
        raise ValueError("Grid HPO has too many candidates to schedule")

    regression_tasks, classification_tasks = _task_lists(prepared)
    if not regression_tasks:
        raise ValueError(
            "Prepared data contains no regression tasks; holdout HPO "
            "requires at least one regression trait."
        )
    patience = int(train_config.get("early_stopping_patience", 5))
    context = HoldoutPoolContext(
        prepared_data=prepared,
        config=dict(config),
        output_directory=str(output_directory),
        global_seed=int(global_seed),
        regression_tasks=tuple(regression_tasks),
        classification_tasks=tuple(classification_tasks),
        metric=str(normalized_hpo["metric"]),
        direction=str(normalized_hpo["direction"]),
        patience=patience,
        loader_options=_loader_kwargs(train_config),
        live_metrics_log=bool(train_config.get("live_metrics_log", False)),
    )

    fold_summaries: Dict[int, Dict[str, Any]] = {}
    expected = {
        int(fold_id): len(parameter_sets) for fold_id in selected_folds
    }
    completed = {int(fold_id): 0 for fold_id in selected_folds}
    jobs: list[HoldoutJob] = []
    for fold_id in selected_folds:
        fold_directory = output_directory / f"fold_{fold_id}"
        fold_directory.mkdir(parents=True, exist_ok=True)
        if resume:
            finished = _completed_fold_summary(fold_directory)
            if finished is not None:
                fold_summaries[int(fold_id)] = finished
                completed[int(fold_id)] = expected[int(fold_id)]
                continue
        for candidate_id, parameters in enumerate(parameter_sets):
            candidate_dir = _candidate_directory(
                output_directory,
                int(fold_id),
                int(candidate_id),
            )
            if resume and _completed_candidate(candidate_dir) is not None:
                completed[int(fold_id)] += 1
                continue
            if resume and candidate_dir.exists():
                shutil.rmtree(candidate_dir)
            jobs.append(
                HoldoutJob(
                    job_id=_candidate_job_id(fold_id, candidate_id),
                    kind="candidate",
                    fold_id=int(fold_id),
                    candidate_id=int(candidate_id),
                    parameters=dict(parameters),
                )
            )

    pending_folds = [
        int(fold_id)
        for fold_id in selected_folds
        if int(fold_id) not in fold_summaries
    ]
    print(
        f"[INFO] Outer-holdout grid: {len(parameter_sets)} candidates, "
        f"{len(jobs)} candidate jobs, folds {list(selected_folds)}; "
        "the early-stopped winner is scored on outer test with no refit"
    )

    def consume(payload: Mapping[str, Any], submit) -> None:
        kind = payload["kind"]
        fold_id = int(payload["fold_id"])
        if kind == "candidate":
            completed[fold_id] += 1
            _queue_fold_test(
                submit,
                fold_id=fold_id,
                completed=completed,
                expected=expected,
                submitted=test_submitted,
            )
            return
        if kind == "test":
            fold_summaries[fold_id] = dict(payload["summary"])
            return
        raise RuntimeError(f"Unexpected holdout payload kind: {kind!r}")

    test_submitted = {
        int(fold_id): int(fold_id) in fold_summaries for fold_id in selected_folds
    }
    if gpu_ids:
        with PersistentGPUPool(
            gpu_ids,
            _holdout_gpu_worker,
            worker_args=(context,),
            deterministic=train_deterministic_enabled(context.config),
        ) as pool:
            pool.submit_many(jobs)
            for fold_id in pending_folds:
                _queue_fold_test(
                    pool.submit,
                    fold_id=fold_id,
                    completed=completed,
                    expected=expected,
                    submitted=test_submitted,
                )
            while len(fold_summaries) < len(selected_folds):
                work = pool.get()
                if not work.succeeded:
                    pool.abort(wait=True)
                    raise RuntimeError(
                        f"Holdout GPU job {work.job_id} failed on {work.device}: "
                        f"{work.error}\n{work.traceback or ''}"
                    )
                consume(work.value, pool.submit)
    else:
        submitted_ids = {job.job_id for job in jobs}

        def submit(job: HoldoutJob) -> None:
            if job.job_id in submitted_ids:
                raise ValueError(f"Duplicate holdout job id: {job.job_id}")
            submitted_ids.add(job.job_id)
            jobs.append(job)

        for fold_id in pending_folds:
            _queue_fold_test(
                submit,
                fold_id=fold_id,
                completed=completed,
                expected=expected,
                submitted=test_submitted,
            )
        pending = list(jobs)
        while pending:
            job = pending.pop(0)
            payload = _holdout_gpu_worker(job, "cpu", context)
            extra = len(jobs)
            consume(payload, submit)
            if len(jobs) > extra:
                pending.extend(jobs[extra:])

    return [fold_summaries[int(fold_id)] for fold_id in selected_folds]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Aquila outer-holdout models. Grid search uses one inner "
            "validation split; the early-stopped winner is scored on the "
            "outer test set."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Prepared holdout dataset.")
    parser.add_argument("--config", required=True, help="Grid HPO YAML configuration.")
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
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted run. Finished candidates and folds are "
            "kept. Do not combine with --overwrite."
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
        help="Append per-epoch metrics JSONL under each candidate directory.",
    )
    parser.add_argument(
        "--use-deterministic",
        action="store_true",
        help="Use CUDA deterministic algorithms, overriding train.deterministic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume cannot be combined with --overwrite")
    prepared = load_prepared_data(args.data_dir)
    require_holdout_data(prepared)
    config = load_config(args.config)
    config.setdefault("train", {})
    config["train"]["precision"] = args.precision
    resolve_train_deterministic(
        config,
        True if args.use_deterministic else None,
    )
    if args.live_metrics_log or args.resume:
        config["train"]["live_metrics_log"] = True
    if "mixed_precision" in config["train"]:
        config["train"].pop("mixed_precision", None)
    _validate_preprocessing_cache(prepared, config)
    selected_folds = resolve_outer_folds(
        args.folds,
        prepared.metadata,
        option_name="--folds",
    )
    output_directory = Path(args.output_dir)
    _prepare_output(
        output_directory,
        selected_folds,
        args.overwrite,
        resume=args.resume,
    )
    params_copy = _copy_params_yaml(Path(args.config), output_directory)
    from aquila.training.distributed import share_memory_tensors

    prepared = PreparedData(
        features=share_memory_tensors(prepared.features),
        targets=prepared.targets.share_memory_(),
        target_mask=prepared.target_mask.share_memory_(),
        metadata=prepared.metadata,
        directory=prepared.directory,
    )
    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    global_seed = int(prepared.metadata["seed"])
    deterministic = train_deterministic_enabled(config)
    if deterministic:
        print("[INFO] CUDA deterministic algorithms enabled")
    completed = _run_holdout_grid(
        prepared=prepared,
        config=config,
        output_directory=output_directory,
        selected_folds=selected_folds,
        gpu_ids=gpu_ids,
        global_seed=global_seed,
        resume=args.resume,
    )
    outer_test_summary = _aggregate_outer_test_metrics(completed)
    summary = {
        "protocol": "outer_holdout",
        "refit": False,
        "requested_folds": list(selected_folds),
        "completed_folds": [item["outer_fold"] for item in completed],
        "failed_folds": [],
        "selected_per_fold": _selected_per_fold(completed),
        "outer_test_summary": outer_test_summary,
        "data_dir": str(Path(args.data_dir).resolve()),
        "config": str(Path(args.config).resolve()),
        "params_yaml": str(params_copy.resolve()),
        "deterministic": deterministic,
    }
    _write_json(output_directory / "summary.json", summary)
    print(
        f"Completed outer-holdout CV for folds {list(selected_folds)}; "
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
