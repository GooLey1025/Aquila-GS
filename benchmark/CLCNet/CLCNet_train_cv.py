#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/SuppurNewer/CLCNet

"""Leakage-safe nested cross-validation training for CLCNet."""

from __future__ import annotations

import argparse
import csv
import gc
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CLCNET_DIRECTORY = SCRIPT_DIRECTORY
BENCHMARK_SOURCE = CLCNET_DIRECTORY / "src_benchmark"
PROJECT_ROOT = CLCNET_DIRECTORY.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_path in (str(SOURCE_ROOT), str(CLCNET_DIRECTORY), str(BENCHMARK_SOURCE)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from aquila.benchmark.common import (
    PreparedBenchmark,
    SingleTraitSplit,
    aggregate_outer_folds,
    build_sample_audit,
    evaluate_two_scales,
    serialize_hpo,
    write_json,
    write_predictions_csv,
)
from aquila.training.distributed import derive_seed
from aquila.training.distributed import detect_gpu_ids, execute_gpu_jobs
from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import (
    CandidateResult,
    HPOResult,
    InnerFoldResult,
    generate_grid_candidates,
    merge_config,
    select_best_candidate,
)
from src_benchmark.model_benchmark import CLCNetBenchmark, estimate_model_size


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Training-only marker schema after optional LightGBM selection."""

    selected_indices: np.ndarray
    global_indices: np.ndarray
    chromosome_indices: Mapping[str, np.ndarray]
    global_importances: np.ndarray
    chromosome_importances: Mapping[str, np.ndarray]
    seed: int
    num_boost_round: int
    method: str


@dataclass(frozen=True)
class TrainingResult:
    """Selected state and validation evidence for one CLCNet fit."""

    state_dict: Mapping[str, torch.Tensor]
    best_epoch: int
    best_metrics: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class TraitFoldJob:
    """One independently tuned trait and outer-fold combination."""

    job_id: int
    trait_name: str
    outer_fold: int


@dataclass(frozen=True)
class TraitFoldContext:
    """Spawn-safe inputs shared by CLCNet GPU workers."""

    data_directory: str
    output_directory: str
    base_config: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    inner_folds: tuple[int, ...]
    seed: int


class DeterministicPairDataset(Dataset):
    """CLCNet pairs regenerated deterministically for each training epoch."""

    def __init__(self, genotypes: np.ndarray, targets: np.ndarray, seed: int) -> None:
        inputs = np.asarray(genotypes, dtype=np.float32)
        labels = np.asarray(targets, dtype=np.float32).reshape(-1)
        if inputs.ndim != 2 or labels.ndim != 1 or len(inputs) != len(labels):
            raise ValueError("Pair data must be aligned sample-major arrays")
        if len(inputs) < 2:
            raise ValueError("CLCNet training requires at least two observed samples")
        self.genotypes = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(labels)
        self.seed = int(seed)
        self.partners = np.zeros(len(inputs), dtype=np.int64)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        generator = np.random.default_rng(derive_seed(self.seed, inner_fold_id=epoch))
        self.partners = generator.integers(0, len(self.genotypes), len(self.genotypes))

    def __len__(self) -> int:
        return len(self.genotypes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        partner = int(self.partners[index])
        first_target = self.targets[index]
        second_target = self.targets[partner]
        difference = torch.abs(first_target - second_target).square()
        return (
            self.genotypes[index],
            self.genotypes[partner],
            first_target.reshape(1),
            second_target.reshape(1),
            difference,
        )


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
        description="Run leakage-safe nested CV for single-trait CLCNet models."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIRECTORY / "configs" / "CLCNet_nested_cv.yaml"),
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
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Testing only: cap training epochs without changing the formal grid.",
    )
    parser.add_argument(
        "--lightgbm-selection",
        action="store_true",
        help=(
            "Enable chromosome-aware LightGBM marker selection. "
            "Disabled by default; the network then uses every marker."
        ),
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_upstream_genotypes(values: Any, missing_value: float = 3.0) -> np.ndarray:
    """Preserve VCF dosage and map missing calls to upstream CLCNet's value 3."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("CLCNet genotypes must be a two-dimensional array")
    finite = array[np.isfinite(array)]
    if finite.size and not np.isin(finite, (0.0, 1.0, 2.0)).all():
        raise ValueError("CLCNet benchmark requires diploid dosage values 0, 1, or 2")
    return np.where(np.isfinite(array), array, float(missing_value)).astype(
        np.float32
    )


def chromosome_groups(
    variants: Sequence[tuple[str, str, str, str, str]],
) -> dict[str, np.ndarray]:
    """Map chromosome labels to absolute marker columns without offset arithmetic."""
    groups: dict[str, list[int]] = {}
    for index, variant in enumerate(variants):
        groups.setdefault(str(variant[0]), []).append(index)
    if not groups:
        raise ValueError("CLCNet feature selection requires at least one variant")
    return {
        chromosome: np.asarray(indices, dtype=np.int64)
        for chromosome, indices in groups.items()
    }


def lightgbm_selection_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether chromosome-aware LightGBM marker selection is enabled."""
    return bool(config.get("enabled", False))


def identity_feature_selection(
    variants: Sequence[tuple[str, str, str, str, str]],
    seed: int,
) -> FeatureSelectionResult:
    """Retain every marker when LightGBM selection is disabled."""
    if not variants:
        raise ValueError("CLCNet feature selection requires at least one variant")
    indices = np.arange(len(variants), dtype=np.int64)
    return FeatureSelectionResult(
        selected_indices=indices,
        global_indices=indices,
        chromosome_indices={},
        global_importances=np.zeros(len(variants), dtype=np.float64),
        chromosome_importances={},
        seed=int(seed),
        num_boost_round=0,
        method="identity",
    )


def apply_feature_selection_override(
    config: dict[str, Any],
    *,
    enabled: bool | None,
) -> dict[str, Any]:
    """Apply CLI LightGBM selection override onto a loaded config."""
    selection = dict(config.get("feature_selection") or {})
    selection.setdefault("enabled", False)
    if enabled:
        selection["enabled"] = True
    config["feature_selection"] = selection
    return config


def fit_feature_selector(
    train_genotypes: np.ndarray,
    train_targets: np.ndarray,
    variants: Sequence[tuple[str, str, str, str, str]],
    config: Mapping[str, Any],
    seed: int,
) -> FeatureSelectionResult:
    """Select markers on training data only; LightGBM is optional and off by default."""
    inputs = encode_upstream_genotypes(
        train_genotypes, float(config.get("missing_genotype_value", 3.0))
    )
    targets = np.asarray(train_targets, dtype=np.float32).reshape(-1)
    if len(inputs) != len(targets) or len(inputs) < 2:
        raise ValueError("Feature selection requires aligned nonempty training data")
    if inputs.shape[1] != len(variants):
        raise ValueError("Feature selection inputs do not match the variant schema")
    if not lightgbm_selection_enabled(config):
        return identity_feature_selection(variants, seed)

    try:
        import lightgbm as lgb
    except ImportError as error:
        raise ImportError("CLCNet chromosome-aware selection requires lightgbm") from error

    rounds = int(config.get("num_boost_round", 100))
    parameters = {
        "objective": "regression",
        "metric": "mse",
        "force_col_wise": True,
        "device": "cpu",
        "verbosity": -1,
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        "deterministic": True,
        "num_threads": int(config.get("num_threads", 1)),
    }

    def train_importances(matrix: np.ndarray, local_seed: int) -> np.ndarray:
        local_parameters = dict(parameters)
        for name in ("seed", "feature_fraction_seed", "bagging_seed", "data_random_seed"):
            local_parameters[name] = int(local_seed)
        model = lgb.train(
            local_parameters,
            lgb.Dataset(matrix, label=targets),
            num_boost_round=rounds,
        )
        return np.asarray(model.feature_importance(importance_type="gain"))

    global_importances = train_importances(inputs, seed)
    global_indices = np.flatnonzero(global_importances > 0).astype(np.int64)
    chromosome_selected: dict[str, np.ndarray] = {}
    chromosome_importances: dict[str, np.ndarray] = {}
    for group_id, (chromosome, absolute_indices) in enumerate(
        chromosome_groups(variants).items()
    ):
        local_seed = derive_seed(seed, trial_id=group_id + 1)
        importances = train_importances(inputs[:, absolute_indices], local_seed)
        chromosome_importances[chromosome] = importances
        chromosome_selected[chromosome] = absolute_indices[importances > 0]
    selected_parts = [global_indices, *chromosome_selected.values()]
    selected = np.unique(np.concatenate(selected_parts)).astype(np.int64)
    if selected.size == 0:
        raise RuntimeError("CLCNet LightGBM selectors did not retain any marker")
    return FeatureSelectionResult(
        selected_indices=selected,
        global_indices=global_indices,
        chromosome_indices=chromosome_selected,
        global_importances=global_importances,
        chromosome_importances=chromosome_importances,
        seed=int(seed),
        num_boost_round=rounds,
        method="lightgbm",
    )


def _shared_dimensions(config: Mapping[str, Any]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in config["model"]["shared_dimensions"])
    if len(values) != 3 or any(value < 1 for value in values):
        raise ValueError("model.shared_dimensions must be three positive integers")
    return values


def _loader(
    genotypes: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(genotypes, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.float32).reshape(-1, 1)),
    )
    if not len(dataset):
        raise ValueError("CLCNet evaluation split is empty")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _predict(
    model: CLCNetBenchmark,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            prediction, _ = model(inputs.to(device, non_blocking=True))
            predictions.append(prediction.detach().cpu())
            targets.append(target)
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    result = evaluate_regression(
        predictions,
        targets,
        np.ones_like(targets, dtype=bool),
        ["trait"],
    )
    return result.metrics


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _train_epoch(
    model: CLCNetBenchmark,
    dataset: DeterministicPairDataset,
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    seed: int,
) -> float:
    dataset.set_epoch(epoch)
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, inner_fold_id=epoch))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    model.train()
    losses: list[float] = []
    mse = nn.MSELoss()
    contrastive_weight = float(config["contrastive_weight"])
    for first, second, first_y, second_y, difference in loader:
        first = first.to(device, non_blocking=True)
        second = second.to(device, non_blocking=True)
        first_y = first_y.to(device, non_blocking=True)
        second_y = second_y.to(device, non_blocking=True)
        difference = difference.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        first_prediction, first_representation = model(first)
        second_prediction, second_representation = model(second)
        representation_distance = F.pairwise_distance(
            first_representation, second_representation
        )
        loss = (
            mse(first_prediction, first_y)
            + mse(second_prediction, second_y)
            + contrastive_weight
            * mse(representation_distance, difference)
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def train_clcnet(
    train_genotypes: np.ndarray,
    train_targets: np.ndarray,
    valid_genotypes: np.ndarray | None,
    valid_targets: np.ndarray | None,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    *,
    fixed_epochs: int | None = None,
) -> TrainingResult:
    """Train CLCNet and select inner epochs only by validation Pearson."""
    set_seed(seed)
    training = config["train"]
    model = CLCNetBenchmark(
        int(train_genotypes.shape[1]), _shared_dimensions(config)
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
        min_lr=float(training["min_learning_rate"]),
    )
    pair_dataset = DeterministicPairDataset(train_genotypes, train_targets, seed)
    validation_loader = (
        _loader(valid_genotypes, valid_targets, int(training["batch_size"]))
        if valid_genotypes is not None and valid_targets is not None
        else None
    )
    epoch_count = int(fixed_epochs or training["max_epochs"])
    patience = int(training.get("patience", epoch_count))
    min_delta = float(training.get("min_delta", 0.0))
    best_metric = -float("inf")
    best_epoch = epoch_count
    best_metrics: dict[str, Any] = {}
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    stale_epochs = 0
    for epoch in range(1, epoch_count + 1):
        training_loss = _train_epoch(
            model, pair_dataset, training, optimizer, device, epoch, seed
        )
        scheduler.step(training_loss)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": training_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if validation_loader is None:
            history.append(record)
            continue
        predictions, targets = _predict(model, validation_loader, device)
        metrics = _metrics(predictions, targets)
        pearson = float(metrics["avg_pearson"])
        record["valid_metrics"] = metrics
        history.append(record)
        if math.isfinite(pearson) and pearson > best_metric + min_delta:
            best_metric = pearson
            best_epoch = epoch
            best_metrics = metrics
            best_state = _cpu_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        best_state = _cpu_state_dict(model)
    return TrainingResult(
        state_dict=best_state,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        history=tuple(history),
    )


def predict_clcnet(
    genotypes: np.ndarray,
    targets: np.ndarray,
    config: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    model = CLCNetBenchmark(
        int(genotypes.shape[1]), _shared_dimensions(config)
    ).to(device)
    model.load_state_dict(state_dict)
    predictions, _ = _predict(
        model,
        _loader(genotypes, targets, int(config["train"]["batch_size"])),
        device,
    )
    return predictions.reshape(-1)


def feature_selection_payload(
    result: FeatureSelectionResult,
    variants: Sequence[tuple[str, str, str, str, str]],
    train_sample_ids: Sequence[str],
) -> dict[str, Any]:
    enabled = result.method == "lightgbm"
    if enabled:
        selection_rule = "importance > 0; union global and chromosome selectors"
    else:
        selection_rule = "retain all markers; LightGBM selection disabled"
    return {
        "enabled": enabled,
        "method": result.method,
        "fit_scope": "training samples only",
        "train_sample_ids": list(train_sample_ids),
        "seed": result.seed,
        "num_boost_round": result.num_boost_round,
        "importance_type": "gain" if enabled else None,
        "selection_rule": selection_rule,
        "total_variants": len(variants),
        "global_selected_count": int(len(result.global_indices)),
        "chromosome_selected_counts": {
            chromosome: int(len(indices))
            for chromosome, indices in result.chromosome_indices.items()
        },
        "union_selected_count": int(len(result.selected_indices)),
    }


def write_selected_variants(
    path: Path,
    selected_indices: np.ndarray,
    variants: Sequence[tuple[str, str, str, str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["feature_index", "CHROM", "POS", "ID", "REF", "ALT"])
        for index in selected_indices:
            writer.writerow([int(index), *variants[int(index)]])


def _selected_matrices(
    train: SingleTraitSplit,
    held_out: SingleTraitSplit,
    selection: FeatureSelectionResult,
    missing_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_values = encode_upstream_genotypes(train.genotypes, missing_value)
    held_out_values = encode_upstream_genotypes(held_out.genotypes, missing_value)
    return (
        train_values[:, selection.selected_indices],
        held_out_values[:, selection.selected_indices],
    )


def run_outer_fold(
    benchmark: PreparedBenchmark,
    output_directory: Path,
    trait_name: str,
    outer_fold: int,
    base_config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    inner_folds: Sequence[int],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    fold_start = time.perf_counter()
    fold_path = output_directory / trait_name / f"fold_{outer_fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    missing_value = float(base_config["data"]["missing_genotype_value"])
    inner_results: dict[int, list[InnerFoldResult]] = {
        candidate_id: [] for candidate_id in range(len(candidates))
    }
    inner_histories: dict[str, Any] = {}
    inner_audits = []
    feature_selection_seconds = 0.0
    hpo_training_seconds = 0.0

    for inner_fold in inner_folds:
        train, valid = benchmark.load_single_trait_fold(
            trait_name, outer_fold, inner_fold
        )
        inner_audits.append(
            {
                "inner_fold": inner_fold,
                **build_sample_audit(train, valid, held_out_name="valid"),
            }
        )
        selector_start = time.perf_counter()
        selector_seed = derive_seed(seed, outer_fold, 0, inner_fold)
        selection = fit_feature_selector(
            train.genotypes,
            train.processed_targets,
            train.variants,
            base_config["feature_selection"],
            selector_seed,
        )
        feature_selection_seconds += time.perf_counter() - selector_start
        train_genotypes, valid_genotypes = _selected_matrices(
            train, valid, selection, missing_value
        )
        print(
            f"[INFO] CLCNet trait={trait_name} outer={outer_fold} inner={inner_fold} "
            f"selection={selection.method} "
            f"selected_variants={len(selection.selected_indices)} "
            f"model={estimate_model_size(train_genotypes.shape[1], _shared_dimensions(base_config))}"
        )
        for candidate_id, parameters in enumerate(candidates):
            candidate_config = merge_config(base_config, parameters)
            candidate_start = time.perf_counter()
            result = train_clcnet(
                train_genotypes,
                train.processed_targets,
                valid_genotypes,
                valid.processed_targets,
                candidate_config,
                device,
                derive_seed(seed, outer_fold, candidate_id, inner_fold),
            )
            hpo_training_seconds += time.perf_counter() - candidate_start
            metric = float(result.best_metrics.get("avg_pearson", float("nan")))
            inner_results[candidate_id].append(
                InnerFoldResult(
                    inner_fold=inner_fold,
                    metric=metric,
                    best_epoch=result.best_epoch,
                    metrics=result.best_metrics,
                )
            )
            inner_histories[f"candidate_{candidate_id}_inner_{inner_fold}"] = list(
                result.history
            )
            del result
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    candidate_results = []
    for candidate_id, parameters in enumerate(candidates):
        results = tuple(inner_results[candidate_id])
        metrics = np.asarray([result.metric for result in results], dtype=np.float64)
        objective = float(metrics.mean()) if np.isfinite(metrics).all() else float("nan")
        candidate_results.append(
            CandidateResult(candidate_id, dict(parameters), objective, results)
        )
    hpo = select_best_candidate(candidate_results, "maximize", "grid")
    best_config = merge_config(base_config, hpo.best.parameters)

    outer_train, outer_test = benchmark.load_single_trait_fold(
        trait_name, outer_fold, None
    )
    selector_start = time.perf_counter()
    final_selection = fit_feature_selector(
        outer_train.genotypes,
        outer_train.processed_targets,
        outer_train.variants,
        best_config["feature_selection"],
        derive_seed(seed, outer_fold, hpo.best.candidate_id, 999),
    )
    feature_selection_seconds += time.perf_counter() - selector_start
    outer_train_genotypes, outer_test_genotypes = _selected_matrices(
        outer_train, outer_test, final_selection, missing_value
    )
    size_estimate = estimate_model_size(
        outer_train_genotypes.shape[1], _shared_dimensions(best_config)
    )
    print(
        f"[INFO] CLCNet trait={trait_name} outer={outer_fold} final "
        f"selection={final_selection.method} "
        f"selected_variants={len(final_selection.selected_indices)} "
        f"model={size_estimate}"
    )

    final_train_start = time.perf_counter()
    final_result = train_clcnet(
        outer_train_genotypes,
        outer_train.processed_targets,
        None,
        None,
        best_config,
        device,
        derive_seed(seed, outer_fold, hpo.best.candidate_id, 999),
        fixed_epochs=hpo.best.final_epoch,
    )
    final_training_seconds = time.perf_counter() - final_train_start
    evaluation_start = time.perf_counter()
    processed_predictions = predict_clcnet(
        outer_test_genotypes,
        outer_test.processed_targets,
        best_config,
        final_result.state_dict,
        device,
    )
    original_predictions = benchmark.inverse_trait(
        processed_predictions, trait_name, outer_fold
    )
    evaluation = evaluate_two_scales(
        processed_predictions,
        outer_test.processed_targets,
        original_predictions,
        outer_test.raw_targets,
        trait_name=trait_name,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start

    torch.save(
        {
            "model_state_dict": final_result.state_dict,
            "trait": trait_name,
            "outer_fold": outer_fold,
            "input_dim": int(outer_train_genotypes.shape[1]),
            "shared_dimensions": list(_shared_dimensions(best_config)),
            "variant_schema": [
                outer_train.variants[int(index)]
                for index in final_selection.selected_indices
            ],
            "selected_feature_indices": final_selection.selected_indices,
            "missing_genotype_value": missing_value,
            "config": best_config,
            "best_candidate_id": hpo.best.candidate_id,
            "final_epoch": hpo.best.final_epoch,
        },
        fold_path / "best_model.pt",
    )
    with (fold_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)
    write_json(
        fold_path / "hpo_results.json",
        {
            **serialize_hpo(HPOResult(
                hpo.best, tuple(candidate_results), "maximize", "grid"
            )),
            "candidate_budget": len(candidates),
            "inner_fold_count": len(inner_folds),
        },
    )
    metrics = {
        "processed": evaluation.processed,
        "original": evaluation.original,
    }
    write_json(fold_path / "metrics.json", metrics)
    benchmark.load_preprocessor(outer_fold).save_json(
        fold_path / "preprocessing.json"
    )
    write_json(
        fold_path / "feature_selection.json",
        feature_selection_payload(
            final_selection, outer_train.variants, outer_train.sample_ids
        ),
    )
    write_selected_variants(
        fold_path / "selected_variants.tsv",
        final_selection.selected_indices,
        outer_train.variants,
    )
    write_json(
        fold_path / "sample_audit.json",
        {
            "outer": build_sample_audit(
                outer_train, outer_test, held_out_name="test"
            ),
            "inner_folds": inner_audits,
        },
    )
    write_json(
        fold_path / "training_history.json",
        {
            "inner": inner_histories,
            "final": list(final_result.history),
        },
    )
    runtime = {
        "feature_selection_seconds": feature_selection_seconds,
        "hpo_training_seconds": hpo_training_seconds,
        "final_training_seconds": final_training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": time.perf_counter() - fold_start,
        "model_size_estimate": size_estimate,
    }
    write_json(fold_path / "runtime.json", runtime)
    write_predictions_csv(
        fold_path / "predictions.csv",
        outer_test.sample_ids,
        evaluation.targets_processed,
        evaluation.predictions_processed,
        evaluation.targets_original,
        evaluation.predictions_original,
        trait_name=trait_name,
        outer_fold=outer_fold,
    )
    return {
        "trait": trait_name,
        "outer_fold": outer_fold,
        "best_candidate_id": hpo.best.candidate_id,
        "best_parameters": dict(hpo.best.parameters),
        "best_inner_pearson": hpo.best.objective,
        "final_epoch": hpo.best.final_epoch,
        "metrics": metrics,
        "runtime_seconds": runtime["total_seconds"],
    }


def _run_trait_fold(
    job: TraitFoldJob,
    device_name: str,
    worker_context: TraitFoldContext,
) -> dict[str, Any]:
    """Run one complete CLCNet trait/fold pipeline on one device."""

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(int(device_name.split(":", 1)[1]))
        torch.backends.cuda.matmul.allow_tf32 = True
    benchmark = PreparedBenchmark(Path(worker_context.data_directory))
    print(
        f"[INFO] CLCNet trait={job.trait_name} outer_fold={job.outer_fold} "
        f"candidates={len(worker_context.candidates)} "
        f"inner_folds={len(worker_context.inner_folds)} device={device_name}"
    )
    return run_outer_fold(
        benchmark,
        Path(worker_context.output_directory),
        job.trait_name,
        job.outer_fold,
        worker_context.base_config,
        worker_context.candidates,
        worker_context.inner_folds,
        device,
        worker_context.seed,
    )


def main() -> None:
    args = parse_args()
    data_directory = Path(args.data_dir).resolve()
    output_directory = Path(args.output_dir).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_directory}")
        import shutil

        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    apply_feature_selection_override(
        config, enabled=True if args.lightgbm_selection else None
    )
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            raise ValueError("--max-epochs must be positive")
        config["train"]["max_epochs"] = min(
            int(config["train"]["max_epochs"]), args.max_epochs
        )
        config["train"]["patience"] = min(
            int(config["train"]["patience"]), args.max_epochs
        )
    benchmark = PreparedBenchmark(data_directory)
    traits = list(args.traits or benchmark.regression_traits)
    if not traits:
        raise ValueError("Prepared data contain no regression traits")
    invalid_traits = [trait for trait in traits if trait not in benchmark.regression_traits]
    if invalid_traits:
        raise ValueError(f"Unknown traits: {invalid_traits}")
    outer_folds = args.outer_folds or list(range(benchmark.outer_fold_count))
    if any(fold < 0 or fold >= benchmark.outer_fold_count for fold in outer_folds):
        raise ValueError(
            f"Outer folds must be in 0..{benchmark.outer_fold_count - 1}"
        )
    inner_folds = list(range(benchmark.inner_fold_count))
    if args.max_inner_folds is not None:
        inner_folds = inner_folds[: args.max_inner_folds]
    candidates = generate_grid_candidates(config["hpo"]["parameters"])
    if len(candidates) != 32:
        raise ValueError(f"CLCNet HPO grid must contain exactly 32 candidates, got {len(candidates)}")
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates or not inner_folds:
        raise ValueError("At least one HPO candidate and inner fold are required")
    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    worker_gpu_ids = expand_gpu_workers(gpu_ids, args.jobs_per_gpu)
    jobs = [
        TraitFoldJob(
            job_id=trait_index * len(outer_folds) + fold_index,
            trait_name=trait,
            outer_fold=outer_fold,
        )
        for trait_index, trait in enumerate(traits)
        for fold_index, outer_fold in enumerate(outer_folds)
    ]
    worker_context = TraitFoldContext(
        data_directory=str(data_directory),
        output_directory=str(output_directory),
        base_config=config,
        candidates=tuple(candidates),
        inner_folds=tuple(inner_folds),
        seed=args.seed,
    )
    work_results = execute_gpu_jobs(
        jobs,
        _run_trait_fold,
        worker_gpu_ids,
        worker_args=(worker_context,),
        raise_on_error=True,
    )
    summaries = [result.value for result in work_results]
    by_trait = {}
    for trait in traits:
        trait_results = [item for item in summaries if item["trait"] == trait]
        by_trait[trait] = {
            "folds": trait_results,
            "outer_fold_summary": aggregate_outer_folds(
                [item["metrics"] for item in trait_results]
            ),
        }
    write_json(
        output_directory / "summary.json",
        {
            "model": "CLCNet",
            "data_dir": data_directory,
            "config": Path(args.config).resolve(),
            "traits": traits,
            "outer_folds": outer_folds,
            "candidate_budget": len(candidates),
            "inner_folds": inner_folds,
            "results": by_trait,
        },
    )


if __name__ == "__main__":
    main()
