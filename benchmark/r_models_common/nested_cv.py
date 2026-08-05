#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/GooLey1025/Aquila-GS

"""Leakage-safe nested-CV orchestration shared by R benchmark models."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    upstream_url: str
    worker: Path
    required_r_packages: tuple[str, ...]
    grid_keys: tuple[str, ...]


@dataclass(frozen=True)
class VCFData:
    values: np.ndarray
    sample_ids: tuple[str, ...]
    variants: tuple[tuple[str, str, str, str, str], ...]


@dataclass(frozen=True)
class SplitData:
    values: np.ndarray
    targets: np.ndarray
    sample_ids: tuple[str, ...]
    discarded_sample_ids: tuple[str, ...]
    variants: tuple[tuple[str, str, str, str, str], ...]


def load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_genotype(sample_field: str, gt_index: int) -> float:
    fields = sample_field.split(":")
    if gt_index >= len(fields):
        return math.nan
    alleles = fields[gt_index].replace("|", "/").split("/")
    if len(alleles) != 2 or "." in alleles:
        return math.nan
    try:
        dosage = [int(allele) for allele in alleles]
    except ValueError:
        return math.nan
    if any(allele not in {0, 1} for allele in dosage):
        raise ValueError("R benchmarks require diploid biallelic genotypes")
    return float(sum(dosage))


def load_vcf(path: Path) -> VCFData:
    opener = gzip.open if path.suffix == ".gz" else open
    sample_ids: tuple[str, ...] | None = None
    variants: list[tuple[str, str, str, str, str]] = []
    rows: list[list[float]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            columns = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                sample_ids = tuple(columns[9:])
                if not sample_ids or len(set(sample_ids)) != len(sample_ids):
                    raise ValueError(f"Invalid VCF sample header: {path}")
                continue
            if line.startswith("#"):
                continue
            if sample_ids is None:
                raise ValueError(f"VCF has no #CHROM header: {path}")
            if len(columns) != 9 + len(sample_ids):
                raise ValueError(f"VCF sample count mismatch: {path}")
            if "," in columns[4]:
                raise ValueError("R benchmarks require biallelic variants")
            format_fields = columns[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record lacks GT: {columns[0]}:{columns[1]}")
            gt_index = format_fields.index("GT")
            variants.append(tuple(columns[index] for index in (0, 1, 2, 3, 4)))
            rows.append(
                [parse_genotype(field, gt_index) for field in columns[9:]]
            )
    if sample_ids is None or not rows:
        raise ValueError(f"VCF contains no genotype records: {path}")
    return VCFData(
        np.asarray(rows, dtype=np.float64).T,
        sample_ids,
        tuple(variants),
    )


def split_paths(
    data_dir: Path,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
) -> tuple[Path, Path, Path]:
    outer_path = data_dir / "cv" / f"outer_fold_{outer_fold}"
    raw_path = data_dir / "raw_genotype" / f"outer_fold_{outer_fold}"
    if inner_fold is None:
        split_path = outer_path / "final"
        index_path = outer_path / f"{role}_idx.npy"
        vcf_path = raw_path / f"{role}.vcf.gz"
    else:
        split_path = outer_path / f"inner_fold_{inner_fold}"
        index_path = split_path / f"{role}_idx.npy"
        vcf_path = raw_path / f"inner_fold_{inner_fold}" / f"{role}.vcf.gz"
    return split_path, index_path, vcf_path


def load_split(
    data_dir: Path,
    metadata: Mapping[str, Any],
    target_mask: torch.Tensor,
    trait_index: int,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
) -> SplitData:
    split_path, index_path, vcf_path = split_paths(
        data_dir, outer_fold, inner_fold, role
    )
    indices = np.load(index_path, allow_pickle=False)
    targets = load_tensor(split_path / f"Y_{role}_processed.pt")
    if len(indices) != len(targets):
        raise ValueError(f"Processed targets do not align with {index_path}")
    sample_ids = metadata["sample_ids"]
    by_sample = {
        str(sample_ids[int(index)]): (
            float(targets[position, trait_index]),
            bool(target_mask[int(index), trait_index]),
        )
        for position, index in enumerate(indices)
    }
    vcf = load_vcf(vcf_path)
    if set(vcf.sample_ids) != set(by_sample):
        raise ValueError(f"VCF samples do not match fold indices: {vcf_path}")
    observed = [
        position
        for position, sample_id in enumerate(vcf.sample_ids)
        if by_sample[sample_id][1]
    ]
    discarded = tuple(
        sample_id
        for sample_id in vcf.sample_ids
        if not by_sample[sample_id][1]
    )
    if len(observed) < 3:
        raise ValueError(f"Trait has fewer than three observed samples: {vcf_path}")
    y = np.asarray(
        [by_sample[vcf.sample_ids[position]][0] for position in observed],
        dtype=np.float64,
    )
    if not np.isfinite(y).all() or np.any(y == -999):
        raise ValueError("Missing phenotype sentinel entered R benchmark targets")
    return SplitData(
        vcf.values[observed],
        y,
        tuple(vcf.sample_ids[position] for position in observed),
        discarded,
        vcf.variants,
    )


def impute_from_training(
    train: SplitData, held_out: SplitData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train.variants != held_out.variants:
        raise ValueError("Training and held-out VCF variant schemas differ")
    means = np.nanmean(train.values, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train_values = np.where(np.isnan(train.values), means, train.values)
    held_values = np.where(np.isnan(held_out.values), means, held_out.values)
    return train_values, held_values, means


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 1:
        raise ValueError("Observed and predicted values must be aligned vectors")
    residual = observed - predicted
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    pearson = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if len(observed) >= 2
        and np.std(observed) > 0
        and np.std(predicted) > 0
        else math.nan
    )
    return {
        "pearson": pearson,
        "r2": 1.0 - float(np.sum(residual**2)) / denominator
        if denominator > 0
        else math.nan,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
    }


def sanitize_json(value: Any) -> Any:
    """Recursively replace non-finite numbers with JSON null."""
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value: Any) -> None:
    """Write standards-compliant, deterministic benchmark JSON."""
    path.write_text(
        json.dumps(
            sanitize_json(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def mean_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else math.nan


def inverse_trait(
    values: np.ndarray, preprocessing_path: Path, trait_index: int
) -> np.ndarray:
    with preprocessing_path.open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)["traits"][trait_index]
    restored = np.asarray(values, dtype=np.float64) * float(parameters["std"])
    restored += float(parameters["mean"])
    if parameters["use_log1p"]:
        restored = np.expm1(restored) - float(parameters["log_shift"])
    return restored


def expand_grid(spec: ModelSpec, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = config.get("grid", {})
    unexpected = sorted(set(grid) - set(spec.grid_keys))
    if unexpected:
        raise ValueError(f"Unexpected grid keys for {spec.name}: {unexpected}")
    if not spec.grid_keys:
        if grid:
            raise ValueError(f"{spec.name} is configured as a singleton model")
        return [dict(config.get("parameters", {}))]
    missing = [key for key in spec.grid_keys if key not in grid]
    if missing:
        raise ValueError(f"Missing grid keys for {spec.name}: {missing}")
    candidates = []
    base = dict(config.get("parameters", {}))
    for values in itertools.product(*(grid[key] for key in spec.grid_keys)):
        candidate = dict(base)
        candidate.update(dict(zip(spec.grid_keys, values)))
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"{spec.name} grid is empty")
    return candidates


def derive_seed(base: int, trait: str, outer: int, candidate: int, inner: int) -> int:
    payload = f"{base}|{trait}|{outer}|{candidate}|{inner}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 2147483647


def write_matrix(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values, delimiter="\t", fmt="%.17g")


def write_vector(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values.reshape(-1, 1), delimiter="\t", fmt="%.17g")


def read_predictions(path: Path, expected: int) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64, ndmin=1)
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid worker predictions: {path}")
    return values


class RWorker:
    def __init__(
        self,
        spec: ModelSpec,
        rscript: str,
        runner: Any = subprocess.run,
    ) -> None:
        self.spec = spec
        self.rscript = rscript
        self.runner = runner

    def check_environment(self) -> None:
        executable = shutil.which(self.rscript)
        if executable is None:
            raise RuntimeError(
                f"Rscript executable not found: {self.rscript}. "
                "Install R or pass --rscript."
            )
        expression = (
            "missing <- c("
            + ",".join(
                f"'{package}'" for package in self.spec.required_r_packages
            )
            + ")[!vapply(c("
            + ",".join(
                f"'{package}'" for package in self.spec.required_r_packages
            )
            + "), requireNamespace, logical(1), quietly=TRUE)];"
            "if(length(missing)) {"
            "cat(paste(missing, collapse=',')); quit(status=17)}"
        )
        result = self.runner(
            [executable, "-e", expression],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            missing = result.stdout.strip() or result.stderr.strip() or "unknown"
            raise RuntimeError(
                f"Missing R package(s) for {self.spec.name}: {missing}"
            )

    def run(
        self,
        work_dir: Path,
        train_x: np.ndarray,
        train_y: np.ndarray,
        predict_x: np.ndarray,
        parameters: Mapping[str, Any],
        seed: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        work_dir.mkdir(parents=True, exist_ok=True)
        write_matrix(work_dir / "train_x.tsv", train_x)
        write_vector(work_dir / "train_y.tsv", train_y)
        write_matrix(work_dir / "predict_x.tsv", predict_x)
        with (work_dir / "parameters.json").open("w", encoding="utf-8") as handle:
            json.dump(parameters, handle, indent=2)
        command = [
            self.rscript,
            str(self.spec.worker),
            "--train-x",
            str(work_dir / "train_x.tsv"),
            "--train-y",
            str(work_dir / "train_y.tsv"),
            "--predict-x",
            str(work_dir / "predict_x.tsv"),
            "--parameters",
            str(work_dir / "parameters.json"),
            "--output-dir",
            str(work_dir),
            "--seed",
            str(seed),
        ]
        result = self.runner(command, text=True, capture_output=True, check=False)
        (work_dir / "worker.stdout.log").write_text(
            result.stdout or "", encoding="utf-8"
        )
        (work_dir / "worker.stderr.log").write_text(
            result.stderr or "", encoding="utf-8"
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{self.spec.name} R worker failed ({result.returncode}). "
                f"See {work_dir / 'worker.stderr.log'}"
            )
        metadata_path = work_dir / "worker_metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        return read_predictions(work_dir / "predictions.tsv", len(predict_x)), metadata


def select_candidate(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    finite = [
        result
        for result in results
        if math.isfinite(float(result["mean_valid_pearson"]))
    ]
    if not finite:
        raise ValueError("All candidates produced non-finite validation Pearson")
    return max(
        finite,
        key=lambda result: (
            float(result["mean_valid_pearson"]),
            -int(result["candidate_id"]),
        ),
    )


def write_predictions(
    path: Path,
    sample_ids: Sequence[str],
    observed_normalized: np.ndarray,
    predicted_normalized: np.ndarray,
    observed_original: np.ndarray,
    predicted_original: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "SampleID",
                "ObservedNormalized",
                "PredictedNormalized",
                "ObservedOriginal",
                "PredictedOriginal",
            ]
        )
        writer.writerows(
            zip(
                sample_ids,
                observed_normalized,
                predicted_normalized,
                observed_original,
                predicted_original,
            )
        )


def run_outer_fold(
    spec: ModelSpec,
    worker: RWorker,
    data_dir: Path,
    output_dir: Path,
    metadata: Mapping[str, Any],
    target_mask: torch.Tensor,
    trait: str,
    trait_index: int,
    outer_fold: int,
    inner_count: int,
    candidates: Sequence[Mapping[str, Any]],
    base_seed: int,
) -> dict[str, Any]:
    started = time.time()
    fold_dir = output_dir / trait / f"fold_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    candidate_results = []
    inner_audit = []
    for candidate_id, parameters in enumerate(candidates):
        fold_results = []
        for inner_fold in range(inner_count):
            train = load_split(
                data_dir,
                metadata,
                target_mask,
                trait_index,
                outer_fold,
                inner_fold,
                "train",
            )
            valid = load_split(
                data_dir,
                metadata,
                target_mask,
                trait_index,
                outer_fold,
                inner_fold,
                "valid",
            )
            train_x, valid_x, means = impute_from_training(train, valid)
            prediction, worker_metadata = worker.run(
                fold_dir
                / "inner"
                / f"candidate_{candidate_id}"
                / f"inner_fold_{inner_fold}",
                train_x,
                train.targets,
                valid_x,
                parameters,
                derive_seed(
                    base_seed, trait, outer_fold, candidate_id, inner_fold
                ),
            )
            metrics = regression_metrics(valid.targets, prediction)
            fold_results.append(
                {
                    "inner_fold": inner_fold,
                    "metrics": metrics,
                    "worker_metadata": worker_metadata,
                }
            )
            if candidate_id == 0:
                inner_audit.append(
                    {
                        "inner_fold": inner_fold,
                        "train_observed": len(train.sample_ids),
                        "train_discarded": list(train.discarded_sample_ids),
                        "valid_observed": len(valid.sample_ids),
                        "valid_discarded": list(valid.discarded_sample_ids),
                        "imputation_means": means.tolist(),
                    }
                )
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "parameters": dict(parameters),
                "mean_valid_pearson": mean_finite(
                    [result["metrics"]["pearson"] for result in fold_results]
                ),
                "inner_results": fold_results,
            }
        )
    best = select_candidate(candidate_results)
    train = load_split(
        data_dir,
        metadata,
        target_mask,
        trait_index,
        outer_fold,
        None,
        "train",
    )
    test = load_split(
        data_dir,
        metadata,
        target_mask,
        trait_index,
        outer_fold,
        None,
        "test",
    )
    train_x, test_x, means = impute_from_training(train, test)
    prediction, worker_metadata = worker.run(
        fold_dir / "final",
        train_x,
        train.targets,
        test_x,
        best["parameters"],
        derive_seed(base_seed, trait, outer_fold, int(best["candidate_id"]), 999),
    )
    normalized_metrics = regression_metrics(test.targets, prediction)
    preprocessing_path = (
        data_dir
        / "cv"
        / f"outer_fold_{outer_fold}"
        / "final"
        / "preprocessing.json"
    )
    observed_original = inverse_trait(test.targets, preprocessing_path, trait_index)
    predicted_original = inverse_trait(prediction, preprocessing_path, trait_index)
    original_metrics = regression_metrics(observed_original, predicted_original)
    write_json(
        fold_dir / "hpo_results.json",
        {
            "selection_metric": "mean inner-validation Pearson",
            "best_candidate_id": best["candidate_id"],
            "best_parameters": best["parameters"],
            "candidates": candidate_results,
        },
    )
    write_json(
        fold_dir / "metrics.json",
        {"normalized": normalized_metrics, "original": original_metrics},
    )
    write_json(
        fold_dir / "sample_audit.json",
        {
            "outer_train_observed": len(train.sample_ids),
            "outer_train_discarded": list(train.discarded_sample_ids),
            "outer_test_observed": len(test.sample_ids),
            "outer_test_discarded": list(test.discarded_sample_ids),
            "outer_test_sample_ids": list(test.sample_ids),
            "inner_folds": inner_audit,
            "final_imputation_means": means.tolist(),
        },
    )
    write_predictions(
        fold_dir / "predictions.csv",
        test.sample_ids,
        test.targets,
        prediction,
        observed_original,
        predicted_original,
    )
    shutil.copy2(preprocessing_path, fold_dir / "preprocessing.json")
    write_json(fold_dir / "selected_parameters.json", best["parameters"])
    return {
        "trait": trait,
        "outer_fold": outer_fold,
        "best_candidate_id": best["candidate_id"],
        "best_parameters": best["parameters"],
        "best_valid_pearson_mean": best["mean_valid_pearson"],
        "metrics": {
            "normalized": normalized_metrics,
            "original": original_metrics,
        },
        "worker_metadata": worker_metadata,
        "elapsed_seconds": time.time() - started,
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for trait in sorted({str(result["trait"]) for result in results}):
        trait_results = [result for result in results if result["trait"] == trait]
        aggregate[trait] = {}
        for scale in ("normalized", "original"):
            aggregate[trait][scale] = {}
            for metric in ("pearson", "r2", "mse", "rmse", "mae"):
                values = np.asarray(
                    [result["metrics"][scale][metric] for result in trait_results],
                    dtype=np.float64,
                )
                aggregate[trait][scale][metric] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values, ddof=1))
                    if np.count_nonzero(np.isfinite(values)) > 1
                    else math.nan,
                }
    return aggregate


def build_parser(spec: ModelSpec, default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run leakage-safe nested CV for {spec.name}."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--traits", nargs="+", default=None)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--outer-folds", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-inner-folds", type=int, default=None)
    return parser


def cli(spec: ModelSpec, default_config: Path) -> None:
    args = build_parser(spec, default_config).parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Prepared data directory not found: {data_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("raw_genotype_saved", False):
        raise ValueError(
            "Prepared data must include fold-specific raw_genotype VCF files"
        )
    target_mask = load_tensor(data_dir / "Y_mask.pt").bool()
    if target_mask.ndim != 2:
        raise ValueError("Y_mask.pt must be a [samples, traits] matrix")
    if target_mask.shape[0] != len(metadata["sample_ids"]):
        raise ValueError("Y_mask.pt sample dimension does not match metadata")
    if target_mask.shape[1] != len(metadata["trait_names"]):
        raise ValueError("Y_mask.pt trait dimension does not match metadata")
    regression_traits = metadata.get("regression_tasks")
    if not isinstance(regression_traits, list):
        trait_tasks = metadata.get("trait_tasks")
        if isinstance(trait_tasks, list):
            regression_traits = [
                trait
                for trait, task in zip(metadata["trait_names"], trait_tasks)
                if task == "regression"
            ]
        else:
            regression_traits = list(metadata["trait_names"])
    traits = args.traits or regression_traits
    invalid = [trait for trait in traits if trait not in metadata["trait_names"]]
    if invalid:
        raise ValueError(f"Unknown traits: {invalid}")
    non_regression = [trait for trait in traits if trait not in regression_traits]
    if non_regression:
        raise ValueError(
            f"{spec.name} only supports regression traits: {non_regression}"
        )
    outer_count = int(metadata["outer_folds"])
    inner_count = int(metadata["inner_folds"])
    outer_folds = args.outer_folds or list(range(outer_count))
    if any(fold < 0 or fold >= outer_count for fold in outer_folds):
        raise ValueError(f"Outer folds must be in 0..{outer_count - 1}")
    if args.max_inner_folds is not None:
        inner_count = min(inner_count, args.max_inner_folds)
    candidates = expand_grid(spec, config)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    worker = RWorker(spec, args.rscript)
    worker.check_environment()
    results = []
    for trait in traits:
        trait_index = list(metadata["trait_names"]).index(trait)
        for outer_fold in outer_folds:
            print(
                f"[INFO] {spec.name} trait={trait} outer_fold={outer_fold} "
                f"candidates={len(candidates)} inner_folds={inner_count}",
                flush=True,
            )
            results.append(
                run_outer_fold(
                    spec,
                    worker,
                    data_dir,
                    output_dir,
                    metadata,
                    target_mask,
                    trait,
                    trait_index,
                    outer_fold,
                    inner_count,
                    candidates,
                    seed,
                )
            )
    write_json(
        output_dir / "summary.json",
        {
            "model": spec.name,
            "data_dir": str(data_dir),
            "config": str(Path(args.config).resolve()),
            "seed": seed,
            "traits": traits,
            "outer_folds": outer_folds,
            "results": results,
            "aggregate": aggregate_results(results),
        },
    )


def launch(
    name: str,
    upstream_url: str,
    worker: Path,
    packages: Sequence[str],
    grid_keys: Sequence[str],
    default_config: Path,
) -> None:
    cli(
        ModelSpec(
            name,
            upstream_url,
            worker,
            tuple(packages),
            tuple(grid_keys),
        ),
        default_config,
    )


if __name__ == "__main__":
    sys.exit("Use a model-specific benchmark entry point.")
