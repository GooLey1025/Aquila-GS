#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/Zhoushuchang-lab/CropARNet

"""Leakage-safe single-trait nested-CV adapter for CropARNet."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class GenotypeSplit:
    """A trait-filtered split supplied by ``aquila.benchmark``."""

    genotypes: np.ndarray
    targets: np.ndarray
    sample_ids: tuple[str, ...]
    discarded_sample_ids: tuple[str, ...]
    variants: tuple[Any, ...]
    missing_mask: np.ndarray | None = None


@dataclass(frozen=True)
class FitResult:
    """State and validation trace from one training run."""

    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_pearson: float
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class TraitFoldJob:
    """One independently tuned trait and outer-fold combination."""

    job_id: int
    trait_index: int
    trait: str
    outer_fold: int


@dataclass(frozen=True)
class TraitFoldContext:
    """Spawn-safe inputs shared by CropARNet GPU workers."""

    data_dir: str
    base_config: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    inner_count: int
    output_dir: str
    seed: int


class CropARNet(nn.Module):
    """Original SNP-weighting MLP with its residual weighted input."""

    def __init__(
        self,
        num_markers: int,
        weights_units: Sequence[int],
        regressor_units: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = self._attention(num_markers, weights_units)
        self.regressor = self._regressor(
            num_markers, regressor_units, dropout
        )

    @staticmethod
    def _attention(num_markers: int, units: Sequence[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = num_markers
        for index, width in enumerate(units):
            layers.append(nn.Linear(previous, int(width)))
            if index < len(units) - 1:
                layers.append(nn.GELU())
            previous = int(width)
        layers.extend((nn.Linear(previous, num_markers), nn.Sigmoid()))
        return nn.Sequential(*layers)

    @staticmethod
    def _regressor(
        num_markers: int, units: Sequence[int], dropout: float
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous = num_markers
        for index, width in enumerate(units):
            width = int(width)
            layers.append(nn.Linear(previous, width))
            if index < len(units) - 1:
                layers.extend(
                    (nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))
                )
            previous = width
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.attention[:-1](inputs)
        weights = self.attention(inputs)
        predictions = self.regressor(inputs * weights + inputs).reshape(-1)
        return predictions, logits


def encode_diploid_gt(genotype: str) -> tuple[float, bool]:
    """Encode biallelic diploid GT while retaining a distinct missing mask."""

    alleles = genotype.replace("|", "/").split("/")
    if len(alleles) != 2 or "." in alleles:
        return np.nan, True
    values = tuple(int(value) for value in alleles)
    if any(value not in {0, 1} for value in values):
        raise ValueError("CropARNet requires biallelic diploid genotypes")
    return float(sum(values) - 1), False


def fit_train_only_scaler(
    train: np.ndarray,
    held_out: np.ndarray,
    train_missing: np.ndarray | None = None,
    held_missing: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Median-impute and MinMax-scale using training genotypes only."""

    train_values = np.asarray(train, dtype=np.float32).copy()
    held_values = np.asarray(held_out, dtype=np.float32).copy()
    train_mask = (
        np.isnan(train_values)
        if train_missing is None
        else np.asarray(train_missing, dtype=bool)
    )
    held_mask = (
        np.isnan(held_values)
        if held_missing is None
        else np.asarray(held_missing, dtype=bool)
    )
    observed = np.where(train_mask, np.nan, train_values)
    medians = np.nanmedian(observed, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_values[train_mask] = np.broadcast_to(medians, train_values.shape)[
        train_mask
    ]
    held_values[held_mask] = np.broadcast_to(medians, held_values.shape)[
        held_mask
    ]
    scaler = MinMaxScaler()
    return (
        scaler.fit_transform(train_values).astype(np.float32),
        scaler.transform(held_values).astype(np.float32),
        scaler,
    )


def pearson(targets: np.ndarray, predictions: np.ndarray) -> float:
    if len(targets) < 2 or np.std(targets) == 0 or np.std(predictions) == 0:
        return 0.0
    value = float(np.corrcoef(targets, predictions)[0, 1])
    return value if math.isfinite(value) else 0.0


def half_up_median_epoch(epochs: Sequence[int]) -> int:
    value = Decimal(str(float(np.median(np.asarray(epochs, dtype=float)))))
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def generate_candidates(parameters: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    names = list(parameters)
    candidates = []
    for values in itertools.product(*(parameters[name] for name in names)):
        candidate: dict[str, Any] = {}
        for name, value in zip(names, values, strict=True):
            path = name.split(".")
            destination = candidate
            for part in path[:-1]:
                destination = destination.setdefault(part, {})
            destination[path[-1]] = value
        candidates.append(candidate)
    return candidates


def expand_gpu_workers(gpu_ids: Sequence[int], jobs_per_gpu: int) -> list[int]:
    """Create one scheduler slot per concurrent job allowed on each GPU."""

    if jobs_per_gpu < 1:
        raise ValueError("--jobs-per-gpu must be at least 1")
    return [
        gpu_id
        for gpu_id in gpu_ids
        for _ in range(jobs_per_gpu)
    ]


def _seed(seed: int, *parts: int) -> int:
    value = int(seed)
    for part in parts:
        value = (value * 1_000_003 + int(part) + 97) % (2**31 - 1)
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model(num_markers: int, config: Mapping[str, Any]) -> CropARNet:
    model = config["model"]
    return CropARNet(
        num_markers,
        model["weights_units"],
        model["regressor_units"],
        float(model["dropout"]),
    )


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray | None,
    valid_y: np.ndarray | None,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    fixed_epochs: int | None = None,
) -> FitResult:
    """Train from scratch; validation Pearson selects inner-fold epoch."""

    _set_seed(seed)
    model = _model(train_x.shape[1], config).to(device)
    training = config["training"]
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_x, dtype=torch.float32),
            torch.as_tensor(train_y, dtype=torch.float32),
        ),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.MSELoss()
    max_epochs = fixed_epochs or int(training["max_epochs"])
    patience = int(training["patience"])
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_score, stale = 1, -math.inf, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = 0.0
        if valid_x is not None and valid_y is not None:
            model.eval()
            with torch.no_grad():
                predictions = model(
                    torch.as_tensor(valid_x, dtype=torch.float32, device=device)
                )[0].cpu().numpy()
            score = pearson(valid_y, predictions)
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        history.append(
            {"epoch": float(epoch), "train_loss": float(np.mean(losses)), "pearson": score}
        )
        if fixed_epochs is None and stale >= patience:
            break
    if fixed_epochs is not None:
        best_state, best_epoch, best_score = (
            copy.deepcopy(model.state_dict()),
            fixed_epochs,
            0.0,
        )
    return FitResult(best_state, best_epoch, best_score, tuple(history))


def _json(path: Path, payload: Any) -> None:
    from aquila.benchmark import sanitize_json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            sanitize_json(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def _resolve_api() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    try:
        from aquila.benchmark import (  # type: ignore[attr-defined]
            evaluate_trait_predictions,
            load_nested_cv_context,
            load_trait_split,
        )
    except ImportError as error:
        raise ImportError(
            "CropARNet expects aquila.benchmark.load_nested_cv_context, "
            "load_trait_split, and evaluate_trait_predictions"
        ) from error
    return load_nested_cv_context, load_trait_split, evaluate_trait_predictions


def _split(payload: Any) -> GenotypeSplit:
    if isinstance(payload, GenotypeSplit):
        return payload
    return GenotypeSplit(
        np.asarray(payload.genotypes),
        np.asarray(payload.targets, dtype=np.float32).reshape(-1),
        tuple(payload.sample_ids),
        tuple(payload.discarded_sample_ids),
        tuple(payload.variants),
        None
        if getattr(payload, "missing_mask", None) is None
        else np.asarray(payload.missing_mask, dtype=bool),
    )


def _run_trait_fold(
    job: TraitFoldJob,
    device_name: str,
    worker_context: TraitFoldContext,
) -> dict[str, Any]:
    """Tune and refit one single-trait outer fold on one device."""

    if device_name.startswith("cuda:"):
        torch.cuda.set_device(int(device_name.split(":", 1)[1]))
        torch.backends.cudnn.benchmark = True
    device = torch.device(device_name)
    load_context, load_split, evaluate = _resolve_api()
    context = load_context(worker_context.data_dir)
    trait = job.trait
    outer_fold = job.outer_fold
    started = time.time()
    candidate_records = []
    for candidate_id, candidate in enumerate(worker_context.candidates):
        config = copy.deepcopy(worker_context.base_config)
        config["model"].update(candidate.get("model", {}))
        config["training"].update(candidate.get("training", {}))
        inner_records = []
        for inner_fold in range(worker_context.inner_count):
            train = _split(load_split(context, trait, outer_fold, inner_fold, "train"))
            valid = _split(load_split(context, trait, outer_fold, inner_fold, "valid"))
            if train.variants != valid.variants:
                raise ValueError("Inner train/valid variant schemas differ")
            train_x, valid_x, _ = fit_train_only_scaler(
                train.genotypes,
                valid.genotypes,
                train.missing_mask,
                valid.missing_mask,
            )
            fit = train_model(
                train_x,
                train.targets,
                valid_x,
                valid.targets,
                config,
                device,
                _seed(
                    worker_context.seed,
                    job.trait_index,
                    outer_fold,
                    candidate_id,
                    inner_fold,
                ),
            )
            inner_records.append(
                {
                    "inner_fold": inner_fold,
                    "pearson": fit.best_pearson,
                    "best_epoch": fit.best_epoch,
                    "history": list(fit.history),
                }
            )
        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "parameters": candidate,
                "mean_pearson": float(
                    np.mean([item["pearson"] for item in inner_records])
                ),
                "final_epoch": half_up_median_epoch(
                    [item["best_epoch"] for item in inner_records]
                ),
                "inner_results": inner_records,
            }
        )
    best = max(
        candidate_records,
        key=lambda item: (item["mean_pearson"], -item["candidate_id"]),
    )
    selected = copy.deepcopy(worker_context.base_config)
    selected["model"].update(best["parameters"].get("model", {}))
    selected["training"].update(best["parameters"].get("training", {}))
    train = _split(load_split(context, trait, outer_fold, None, "train"))
    test = _split(load_split(context, trait, outer_fold, None, "test"))
    if train.variants != test.variants:
        raise ValueError("Outer train/test variant schemas differ")
    train_x, test_x, scaler = fit_train_only_scaler(
        train.genotypes,
        test.genotypes,
        train.missing_mask,
        test.missing_mask,
    )
    final_fit = train_model(
        train_x,
        train.targets,
        None,
        None,
        selected,
        device,
        _seed(
            worker_context.seed,
            job.trait_index,
            outer_fold,
            best["candidate_id"],
            999,
        ),
        fixed_epochs=best["final_epoch"],
    )
    model = _model(train_x.shape[1], selected).to(device)
    model.load_state_dict(final_fit.state_dict)
    model.eval()
    with torch.no_grad():
        predictions = model(
            torch.as_tensor(test_x, dtype=torch.float32, device=device)
        )[0].cpu().numpy()
    metrics = evaluate(context, trait, outer_fold, test.targets, predictions)
    fold_path = Path(worker_context.output_dir) / trait / f"fold_{outer_fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": final_fit.state_dict,
            "config": selected,
            "trait": trait,
            "outer_fold": outer_fold,
            "epoch": best["final_epoch"],
            "scaler": {
                "min": scaler.min_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
        },
        fold_path / "checkpoint.pt",
    )
    _json(fold_path / "hpo.json", {"candidates": candidate_records, "best": best})
    _json(fold_path / "history.json", list(final_fit.history))
    _json(fold_path / "metrics.json", metrics)
    _json(
        fold_path / "audit.json",
        {
            "train_samples": list(train.sample_ids),
            "test_samples": list(test.sample_ids),
            "train_discarded": list(train.discarded_sample_ids),
            "test_discarded": list(test.discarded_sample_ids),
            "device": device_name,
            "test_evaluations": 1,
            "runtime_seconds": time.time() - started,
        },
    )
    with (fold_path / "predictions.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("sample_id", "observed_processed", "predicted_processed"))
        writer.writerows(zip(test.sample_ids, test.targets, predictions, strict=True))
    return {
        "trait": trait,
        "outer_fold": outer_fold,
        "candidate_id": best["candidate_id"],
        "epoch": best["final_epoch"],
        "device": device_name,
        "metrics": metrics,
    }


def run_nested_cv(args: argparse.Namespace) -> dict[str, Any]:
    """Run independent per-trait HPO and untouched outer-test evaluation."""

    load_context, _, _ = _resolve_api()
    from aquila.benchmark import aggregate_outer_folds
    from aquila.training.distributed import detect_gpu_ids, execute_gpu_jobs

    with Path(args.config).open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    context = load_context(args.data_dir)
    traits = args.traits or list(context.regression_traits)
    outer_folds = args.outer_folds or list(range(int(context.outer_folds)))
    inner_count = int(context.inner_folds)
    if args.max_inner_folds is not None:
        inner_count = min(inner_count, args.max_inner_folds)
    candidates = generate_candidates(base["hpo"]["grid"])
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    output = Path(args.output_dir or base["output"]["directory"])
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    gpu_ids = [] if args.gpus == [] else detect_gpu_ids(args.gpus)
    worker_gpu_ids = expand_gpu_workers(gpu_ids, args.jobs_per_gpu)
    jobs = [
        TraitFoldJob(
            job_id=trait_index * len(outer_folds) + fold_index,
            trait_index=trait_index,
            trait=trait,
            outer_fold=outer_fold,
        )
        for trait_index, trait in enumerate(traits)
        for fold_index, outer_fold in enumerate(outer_folds)
    ]
    worker_context = TraitFoldContext(
        data_dir=str(Path(args.data_dir).resolve()),
        base_config=base,
        candidates=tuple(candidates),
        inner_count=inner_count,
        output_dir=str(output.resolve()),
        seed=args.seed,
    )
    devices = (
        [f"cuda:{gpu_id}" for gpu_id in worker_gpu_ids]
        if worker_gpu_ids
        else ["cpu"]
    )
    print(
        f"CropARNet: {len(traits)} traits x {len(outer_folds)} outer folds; "
        f"one independent model per trait/fold across {devices}; "
        f"{args.jobs_per_gpu} concurrent job(s) per GPU"
    )
    work_results = execute_gpu_jobs(
        jobs,
        _run_trait_fold,
        worker_gpu_ids,
        worker_args=(worker_context,),
        raise_on_error=True,
    )
    all_results = [result.value for result in work_results]
    aggregate = {}
    for trait in traits:
        fold_metrics = [
            result["metrics"] for result in all_results if result["trait"] == trait
        ]
        if fold_metrics:
            aggregate[trait] = aggregate_outer_folds(fold_metrics)
    summary = {
        "model": "CropARNet",
        "traits": traits,
        "outer_folds": outer_folds,
        "results": all_results,
        "aggregate": aggregate,
    }
    _json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("-o", "--output-dir")
    parser.add_argument("--traits", nargs="+")
    parser.add_argument("--outer-folds", nargs="+", type=int)
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=None,
        help="GPU IDs to use; omit to use all detected GPUs, or pass no IDs for CPU.",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Maximum concurrent trait/fold jobs per GPU (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-inner-folds", type=int)
    return parser.parse_args(argv)


def main() -> None:
    run_nested_cv(parse_args())


if __name__ == "__main__":
    main()
