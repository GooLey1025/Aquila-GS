#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/dmlc/xgboost

"""Leakage-safe XGBoost nested cross-validation benchmark adapter."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

try:
    from aquila.benchmark import (  # type: ignore[attr-defined]
        PerTraitPreprocessor,
        derive_seed,
        evaluate_regression,
        generate_grid_candidates,
        half_up_median_epoch,
        load_prepared_data,
    )
except ImportError:
    from aquila.data import PerTraitPreprocessor, load_prepared_data
    from aquila.training.distributed import derive_seed
    from aquila.training.evaluator import evaluate_regression
    from aquila.training.hpo import (
        generate_grid_candidates,
        half_up_median_epoch,
    )


@dataclass(frozen=True)
class VCFGenotypes:
    """Dosage genotypes and VCF alignment metadata."""

    genotypes: np.ndarray
    sample_ids: tuple[str, ...]
    variants: tuple[tuple[str, str, str, str, str], ...]


@dataclass(frozen=True)
class SplitData:
    """One observed-trait split before genotype imputation."""

    genotypes: np.ndarray
    targets: np.ndarray
    sample_ids: tuple[str, ...]
    discarded_sample_ids: tuple[str, ...]
    variants: tuple[tuple[str, str, str, str, str], ...]


@dataclass(frozen=True)
class ImputationResult:
    """Training-fitted marker means and transformed matrices."""

    train: np.ndarray
    held_out: np.ndarray
    marker_means: np.ndarray
    all_missing_markers: tuple[int, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe nested CV for single-trait XGBoost models."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIRECTORY / "configs" / "xgboost_nested_cv.yaml"),
    )
    parser.add_argument("--traits", nargs="+", default=None)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--outer-folds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-inner-folds", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args(argv)


def require_xgboost() -> Any:
    """Import XGBoost with an actionable optional-dependency error."""
    try:
        import xgboost
    except ImportError as error:
        raise RuntimeError(
            "The XGBoost nested-CV adapter requires the optional 'xgboost' "
            "package. Install it with `python -m pip install xgboost`."
        ) from error
    return xgboost


def _open_vcf(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def encode_dosage(sample_field: str, gt_index: int) -> float:
    """Encode diploid biallelic GT as ALT dosage 0, 1, or 2."""
    fields = sample_field.split(":")
    if gt_index >= len(fields):
        return float("nan")
    alleles = fields[gt_index].replace("|", "/").split("/")
    if len(alleles) != 2 or "." in alleles:
        return float("nan")
    try:
        values = [int(allele) for allele in alleles]
    except ValueError:
        return float("nan")
    if any(allele not in {0, 1} for allele in values):
        raise ValueError("XGBoost benchmark requires biallelic diploid genotypes")
    return float(sum(values))


def load_vcf_genotypes(path: str | Path) -> VCFGenotypes:
    """Read a fold-local VCF as samples by markers without imputing it."""
    vcf_path = Path(path)
    sample_ids: tuple[str, ...] | None = None
    variants = []
    marker_rows = []
    with _open_vcf(vcf_path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                columns = line.rstrip("\n").split("\t")
                sample_ids = tuple(columns[9:])
                if not sample_ids or len(set(sample_ids)) != len(sample_ids):
                    raise ValueError(f"Invalid VCF sample header: {vcf_path}")
                continue
            if line.startswith("#"):
                continue
            if sample_ids is None:
                raise ValueError(f"VCF has no #CHROM header: {vcf_path}")
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9 + len(sample_ids):
                raise ValueError(f"VCF sample count mismatch: {vcf_path}")
            if "," in columns[4]:
                raise ValueError("XGBoost benchmark requires biallelic variants")
            format_fields = columns[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record lacks GT: {columns[0]}:{columns[1]}")
            gt_index = format_fields.index("GT")
            variants.append(
                (columns[0], columns[1], columns[2], columns[3], columns[4])
            )
            marker_rows.append(
                [encode_dosage(field, gt_index) for field in columns[9:]]
            )
    if sample_ids is None or not marker_rows:
        raise ValueError(f"VCF contains no genotype records: {vcf_path}")
    matrix = np.asarray(marker_rows, dtype=np.float32).T
    return VCFGenotypes(matrix, sample_ids, tuple(variants))


def _load_tensor(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _split_paths(
    data_directory: Path,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
) -> tuple[Path, Path, Path]:
    outer_path = data_directory / "cv" / f"outer_fold_{outer_fold}"
    raw_path = data_directory / "raw_genotype" / f"outer_fold_{outer_fold}"
    if inner_fold is None:
        if role not in {"train", "test"}:
            raise ValueError(f"Invalid final split role: {role}")
        split_path = outer_path / "final"
        index_path = outer_path / f"{role}_idx.npy"
        vcf_path = raw_path / f"{role}.vcf.gz"
    else:
        if role not in {"train", "valid"}:
            raise ValueError(f"Invalid inner split role: {role}")
        split_path = outer_path / f"inner_fold_{inner_fold}"
        index_path = split_path / f"{role}_idx.npy"
        vcf_path = raw_path / f"inner_fold_{inner_fold}" / f"{role}.vcf.gz"
    return split_path, index_path, vcf_path


def load_split_data(
    data_directory: Path,
    metadata: Mapping[str, Any],
    target_mask: Any,
    trait_index: int,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
) -> SplitData:
    """Align VCF rows to cached fold-local processed phenotypes."""
    split_path, index_path, vcf_path = _split_paths(
        data_directory, outer_fold, inner_fold, role
    )
    absolute_indices = np.load(index_path, allow_pickle=False)
    processed = _load_tensor(split_path / f"Y_{role}_processed.pt")
    processed_array = np.asarray(processed.detach().cpu(), dtype=np.float32)
    if len(absolute_indices) != len(processed_array):
        raise ValueError(f"Processed targets do not align with {index_path}")
    mask_array = np.asarray(target_mask.detach().cpu(), dtype=bool)
    sample_ids = metadata["sample_ids"]
    by_sample = {
        str(sample_ids[int(index)]): (
            float(processed_array[position, trait_index]),
            bool(mask_array[int(index), trait_index]),
        )
        for position, index in enumerate(absolute_indices)
    }
    vcf = load_vcf_genotypes(vcf_path)
    if set(vcf.sample_ids) != set(by_sample):
        raise ValueError(f"VCF samples do not match fold indices: {vcf_path}")
    observed_positions = [
        position
        for position, sample_id in enumerate(vcf.sample_ids)
        if by_sample[sample_id][1]
    ]
    if len(observed_positions) < 2:
        raise ValueError(f"Trait has fewer than two observed samples in {vcf_path}")
    observed_ids = tuple(vcf.sample_ids[position] for position in observed_positions)
    discarded = tuple(
        sample_id for sample_id in vcf.sample_ids if not by_sample[sample_id][1]
    )
    targets = np.asarray(
        [by_sample[sample_id][0] for sample_id in observed_ids], dtype=np.float32
    )
    if not np.isfinite(targets).all() or np.any(targets == -999):
        raise ValueError("Missing phenotype sentinel entered XGBoost targets")
    return SplitData(
        genotypes=vcf.genotypes[observed_positions].copy(),
        targets=targets,
        sample_ids=observed_ids,
        discarded_sample_ids=discarded,
        variants=vcf.variants,
    )


def validate_variant_schema(first: SplitData, second: SplitData) -> None:
    if first.variants != second.variants:
        raise ValueError("Training and held-out VCF variant schemas differ")


def impute_from_training(
    train: np.ndarray,
    held_out: np.ndarray,
) -> ImputationResult:
    """Fit per-marker means on training rows and apply them to both matrices."""
    train_array = np.asarray(train, dtype=np.float32)
    held_array = np.asarray(held_out, dtype=np.float32)
    if train_array.ndim != 2 or held_array.ndim != 2:
        raise ValueError("Genotype matrices must be two-dimensional")
    if train_array.shape[1] != held_array.shape[1]:
        raise ValueError("Genotype matrices must have the same marker count")
    valid_counts = np.sum(np.isfinite(train_array), axis=0)
    sums = np.nansum(train_array, axis=0, dtype=np.float64)
    means = np.divide(
        sums,
        valid_counts,
        out=np.zeros(train_array.shape[1], dtype=np.float64),
        where=valid_counts > 0,
    ).astype(np.float32)
    all_missing = tuple(np.flatnonzero(valid_counts == 0).astype(int).tolist())
    train_result = np.where(np.isfinite(train_array), train_array, means)
    held_result = np.where(np.isfinite(held_array), held_array, means)
    return ImputationResult(
        train=train_result.astype(np.float32, copy=False),
        held_out=held_result.astype(np.float32, copy=False),
        marker_means=means,
        all_missing_markers=all_missing,
    )


def _predict(booster: Any, matrix: Any, iteration_count: int | None = None) -> np.ndarray:
    if iteration_count is None:
        return np.asarray(booster.predict(matrix), dtype=np.float32)
    try:
        values = booster.predict(matrix, iteration_range=(0, iteration_count))
    except TypeError:
        values = booster.predict(matrix, ntree_limit=iteration_count)
    return np.asarray(values, dtype=np.float32)


def _best_round(booster: Any, maximum_rounds: int) -> int:
    best_iteration = getattr(booster, "best_iteration", None)
    if best_iteration is None:
        return int(maximum_rounds)
    return min(int(maximum_rounds), int(best_iteration) + 1)


def _metric_payload(predictions: np.ndarray, targets: np.ndarray, trait: str) -> dict:
    return evaluate_regression(
        predictions[:, None],
        targets[:, None],
        np.ones((len(targets), 1), dtype=bool),
        [trait],
    ).metrics


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


def _candidate_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "parameters": candidate["parameters"],
        "objective": candidate["objective"],
        "best_iterations": [
            result["best_iteration"] for result in candidate["inner_results"]
        ],
        "final_rounds": candidate["final_rounds"],
        "inner_results": candidate["inner_results"],
    }


def train_inner_candidate(
    xgboost: Any,
    train: SplitData,
    valid: SplitData,
    config: Mapping[str, Any],
    seed: int,
    n_jobs: int,
    trait_name: str,
) -> dict[str, Any]:
    validate_variant_schema(train, valid)
    imputed = impute_from_training(train.genotypes, valid.genotypes)
    train_matrix = xgboost.DMatrix(imputed.train, label=train.targets)
    valid_matrix = xgboost.DMatrix(imputed.held_out, label=valid.targets)
    train_config = config["train"]
    params = {
        **dict(config["model"]),
        "seed": int(seed),
        "nthread": int(n_jobs),
    }
    maximum_rounds = int(train_config["max_rounds"])
    booster = xgboost.train(
        params=params,
        dtrain=train_matrix,
        num_boost_round=maximum_rounds,
        evals=[(valid_matrix, "valid")],
        early_stopping_rounds=int(train_config["early_stopping_rounds"]),
        verbose_eval=False,
    )
    best_round = _best_round(booster, maximum_rounds)
    predictions = _predict(booster, valid_matrix, best_round)
    metrics = _metric_payload(predictions, valid.targets, trait_name)
    return {
        "best_iteration": best_round,
        "metrics": metrics,
        "objective": float(metrics["avg_pearson"]),
        "imputation": {
            "all_missing_marker_indices": list(imputed.all_missing_markers),
            "marker_count": int(imputed.marker_means.size),
        },
    }


def _select_best(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    finite = [item for item in candidates if math.isfinite(float(item["objective"]))]
    if not finite:
        raise ValueError("All XGBoost grid candidates produced non-finite objectives")
    return max(finite, key=lambda item: (float(item["objective"]), -item["candidate_id"]))


def _write_json(path: Path, payload: Any) -> None:
    try:
        from aquila.benchmark import sanitize_json
    except ImportError:
        def sanitize_json(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): sanitize_json(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [sanitize_json(item) for item in value]
            if isinstance(value, np.ndarray):
                return sanitize_json(value.tolist())
            if isinstance(value, (float, np.floating)):
                number = float(value)
                return number if math.isfinite(number) else None
            if isinstance(value, np.integer):
                return int(value)
            return value

    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def run_outer_fold(
    xgboost: Any,
    data_directory: Path,
    output_directory: Path,
    prepared: Any,
    trait_name: str,
    trait_index: int,
    outer_fold: int,
    base_config: Mapping[str, Any],
    candidate_parameters: Sequence[Mapping[str, Any]],
    inner_count: int,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    started = time.time()
    fold_path = output_directory / trait_name / f"fold_{outer_fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    candidate_results = []
    split_audit = []

    inner_splits = []
    for inner_fold in range(inner_count):
        train = load_split_data(
            data_directory,
            prepared.metadata,
            prepared.target_mask,
            trait_index,
            outer_fold,
            inner_fold,
            "train",
        )
        valid = load_split_data(
            data_directory,
            prepared.metadata,
            prepared.target_mask,
            trait_index,
            outer_fold,
            inner_fold,
            "valid",
        )
        validate_variant_schema(train, valid)
        inner_splits.append((train, valid))
        split_audit.append(
            {
                "inner_fold": inner_fold,
                "train_observed": len(train.sample_ids),
                "train_discarded": list(train.discarded_sample_ids),
                "valid_observed": len(valid.sample_ids),
                "valid_discarded": list(valid.discarded_sample_ids),
            }
        )

    for candidate_id, parameters in enumerate(candidate_parameters):
        config = {
            **base_config,
            "model": {**base_config["model"], **dict(parameters)},
        }
        inner_results = []
        for inner_fold, (train, valid) in enumerate(inner_splits):
            result = train_inner_candidate(
                xgboost,
                train,
                valid,
                config,
                derive_seed(seed, outer_fold, candidate_id, inner_fold),
                n_jobs,
                trait_name,
            )
            inner_results.append({"inner_fold": inner_fold, **result})
        objectives = np.asarray(
            [result["objective"] for result in inner_results], dtype=float
        )
        objective = (
            float(objectives.mean()) if np.isfinite(objectives).all() else float("nan")
        )
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "parameters": dict(parameters),
                "objective": objective,
                "final_rounds": half_up_median_epoch(
                    [result["best_iteration"] for result in inner_results]
                ),
                "inner_results": inner_results,
            }
        )

    best = _select_best(candidate_results)
    best_config = {
        **base_config,
        "model": {**base_config["model"], **dict(best["parameters"])},
    }
    final_rounds = int(best["final_rounds"])
    outer_train = load_split_data(
        data_directory,
        prepared.metadata,
        prepared.target_mask,
        trait_index,
        outer_fold,
        None,
        "train",
    )
    outer_test = load_split_data(
        data_directory,
        prepared.metadata,
        prepared.target_mask,
        trait_index,
        outer_fold,
        None,
        "test",
    )
    validate_variant_schema(outer_train, outer_test)
    imputed = impute_from_training(outer_train.genotypes, outer_test.genotypes)
    train_matrix = xgboost.DMatrix(imputed.train, label=outer_train.targets)
    test_matrix = xgboost.DMatrix(imputed.held_out)
    final_params = {
        **best_config["model"],
        "seed": derive_seed(seed, outer_fold, int(best["candidate_id"]), 999),
        "nthread": int(n_jobs),
    }
    booster = xgboost.train(
        params=final_params,
        dtrain=train_matrix,
        num_boost_round=final_rounds,
        evals=[],
        verbose_eval=False,
    )
    normalized_predictions = _predict(booster, test_matrix)
    normalized_metrics = _metric_payload(
        normalized_predictions, outer_test.targets, trait_name
    )
    preprocessor = PerTraitPreprocessor.load_json(
        data_directory
        / "cv"
        / f"outer_fold_{outer_fold}"
        / "final"
        / "preprocessing.json"
    )
    original_predictions = _inverse_trait(
        normalized_predictions, preprocessor, trait_index
    )
    original_targets = _inverse_trait(outer_test.targets, preprocessor, trait_index)
    original_metrics = _metric_payload(
        original_predictions, original_targets, trait_name
    )

    booster.save_model(str(fold_path / "booster.json"))
    np.save(fold_path / "imputation_means.npy", imputed.marker_means)
    _write_json(
        fold_path / "hpo_results.json",
        {
            "method": "grid",
            "direction": "maximize",
            "metric": "avg_pearson",
            "best_candidate_id": best["candidate_id"],
            "best_parameters": best["parameters"],
            "best_valid_pearson_mean": best["objective"],
            "final_rounds": final_rounds,
            "candidates": [_candidate_payload(item) for item in candidate_results],
        },
    )
    _write_json(
        fold_path / "metrics.json",
        {"standardized": normalized_metrics, "original": original_metrics},
    )
    _write_json(
        fold_path / "audit.json",
        {
            "dosage_encoding": "ALT allele count: 0/0=0, heterozygous=1, 1/1=2",
            "imputation_fit": "outer_train_only",
            "imputation_marker_count": int(imputed.marker_means.size),
            "all_missing_marker_indices": list(imputed.all_missing_markers),
            "outer_train_observed": len(outer_train.sample_ids),
            "outer_train_sample_ids": list(outer_train.sample_ids),
            "outer_train_discarded": list(outer_train.discarded_sample_ids),
            "outer_test_observed": len(outer_test.sample_ids),
            "outer_test_sample_ids": list(outer_test.sample_ids),
            "outer_test_discarded": list(outer_test.discarded_sample_ids),
            "test_evaluation_count": 1,
            "inner_folds": split_audit,
        },
    )
    with (fold_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)
    preprocessor.save_json(fold_path / "preprocessing.json")
    with (fold_path / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "prediction_standardized",
                "observed_standardized",
                "prediction_original",
                "observed_original",
            ]
        )
        for values in zip(
            outer_test.sample_ids,
            normalized_predictions,
            outer_test.targets,
            original_predictions,
            original_targets,
        ):
            writer.writerow(
                [values[0], *(float(value) for value in values[1:])]
            )
    return {
        "outer_fold": outer_fold,
        "best_candidate_id": best["candidate_id"],
        "best_parameters": best["parameters"],
        "best_valid_pearson_mean": best["objective"],
        "final_rounds": final_rounds,
        "test_metrics_standardized": normalized_metrics,
        "test_metrics_original": original_metrics,
        "elapsed_seconds": time.time() - started,
    }


def _validate_positive_limit(value: int | None, name: str) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be positive")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_positive_limit(args.max_candidates, "--max-candidates")
    _validate_positive_limit(args.max_inner_folds, "--max-inner-folds")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    xgboost = require_xgboost()
    data_directory = Path(args.data_dir).resolve()
    output_directory = Path(args.output_dir).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_directory}")
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    prepared = load_prepared_data(data_directory)
    trait_names = list(prepared.metadata["trait_names"])
    traits = args.traits or trait_names
    invalid_traits = [trait for trait in traits if trait not in trait_names]
    if invalid_traits:
        raise ValueError(f"Unknown traits: {invalid_traits}")
    trait_tasks = prepared.metadata.get(
        "trait_tasks", ["regression"] * len(trait_names)
    )
    non_regression = [
        trait for trait in traits if trait_tasks[trait_names.index(trait)] != "regression"
    ]
    if non_regression:
        raise ValueError(f"XGBoost adapter supports regression traits only: {non_regression}")
    outer_count = int(prepared.metadata["outer_folds"])
    inner_count = int(prepared.metadata["inner_folds"])
    outer_folds = args.outer_folds or list(range(outer_count))
    if any(fold < 0 or fold >= outer_count for fold in outer_folds):
        raise ValueError(f"Outer folds must be in 0..{outer_count - 1}")
    if len(set(outer_folds)) != len(outer_folds):
        raise ValueError("Outer folds must be unique")
    if args.max_inner_folds is not None:
        inner_count = min(inner_count, args.max_inner_folds)
    candidates = generate_grid_candidates(config["hpo"]["parameters"])
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ValueError("The XGBoost HPO grid is empty")

    summaries = []
    for trait_name in traits:
        trait_index = trait_names.index(trait_name)
        for outer_fold in outer_folds:
            print(
                f"[INFO] XGBoost trait={trait_name} outer_fold={outer_fold} "
                f"candidates={len(candidates)} inner_folds={inner_count}"
            )
            summaries.append(
                {
                    "trait": trait_name,
                    **run_outer_fold(
                        xgboost,
                        data_directory,
                        output_directory,
                        prepared,
                        trait_name,
                        trait_index,
                        outer_fold,
                        config,
                        candidates,
                        inner_count,
                        args.seed,
                        args.n_jobs,
                    ),
                }
            )
    from aquila.benchmark import aggregate_outer_folds

    aggregate = {}
    for trait_name in traits:
        fold_metrics = [
            {
                "standardized": result["test_metrics_standardized"],
                "original": result["test_metrics_original"],
            }
            for result in summaries
            if result["trait"] == trait_name
        ]
        if fold_metrics:
            aggregate[trait_name] = aggregate_outer_folds(fold_metrics)
    _write_json(
        output_directory / "summary.json",
        {
            "adapter": "xgboost_nested_cv",
            "data_dir": str(data_directory),
            "config": str(Path(args.config).resolve()),
            "traits": traits,
            "outer_folds": outer_folds,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
            "results": summaries,
            "aggregate": aggregate,
        },
    )


if __name__ == "__main__":
    main()
