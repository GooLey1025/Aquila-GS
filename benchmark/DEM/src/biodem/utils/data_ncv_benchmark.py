# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Prepared nested-CV data adapter for the DEM benchmark."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor


@dataclass(frozen=True)
class VCFDosage:
    """Dosage matrix and stable variant schema loaded from one VCF."""

    genotypes: np.ndarray
    sample_ids: tuple[str, ...]
    variants: tuple[tuple[str, str, str, str, str], ...]


@dataclass(frozen=True)
class DEMSplit:
    """One complete-case split aligned to its fold-specific raw VCF."""

    genotypes: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    sample_ids: tuple[str, ...]
    discarded_sample_ids: tuple[str, ...]
    variants: tuple[tuple[str, str, str, str, str], ...]


@dataclass(frozen=True)
class SelectedSplits:
    """Training and held-out splits after train-only SNP selection."""

    train: DEMSplit
    held_out: DEMSplit
    selected_indices: np.ndarray
    selected_variants: tuple[tuple[str, str, str, str, str], ...]
    importances: np.ndarray


def _torch_load(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_prepared_directory(data_directory: str | Path) -> None:
    """Require the common benchmark artifacts used by every DEM split."""
    directory = Path(data_directory)
    required = (
        directory / "metadata.json",
        directory / "Y_raw.pt",
        directory / "Y_mask.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DEM benchmark prepared data are incomplete. Missing: "
            + ", ".join(missing)
        )


def load_metadata(data_directory: str | Path) -> dict[str, Any]:
    """Load and validate common nested-CV metadata."""
    validate_prepared_directory(data_directory)
    path = Path(data_directory) / "metadata.json"
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    if int(metadata.get("outer_folds", 0)) < 1:
        raise ValueError("metadata.json has no outer folds")
    if int(metadata.get("inner_folds", 0)) < 1:
        raise ValueError("metadata.json has no inner folds")
    if not metadata.get("trait_names"):
        raise ValueError("metadata.json has no phenotype traits")
    return metadata


def regression_trait_indices(metadata: Mapping[str, Any]) -> tuple[int, ...]:
    """Return all regression output columns in metadata order."""
    names = tuple(metadata["trait_names"])
    tasks = metadata.get("trait_tasks")
    if tasks is None:
        count = int(metadata.get("n_regression_tasks", len(names)))
        indices = tuple(range(count))
    else:
        indices = tuple(
            index for index, task in enumerate(tasks) if task == "regression"
        )
    if not indices:
        raise ValueError("DEM benchmark requires at least one regression trait")
    return indices


def _encode_gt(sample_field: str, gt_index: int, missing_value: float) -> float:
    fields = sample_field.split(":")
    if gt_index >= len(fields):
        return missing_value
    alleles = fields[gt_index].replace("|", "/").split("/")
    if len(alleles) != 2 or "." in alleles:
        return missing_value
    try:
        dosage = [int(allele) for allele in alleles]
    except ValueError:
        return missing_value
    if any(allele not in {0, 1} for allele in dosage):
        raise ValueError("DEM benchmark requires biallelic diploid genotypes")
    return float(sum(dosage))


def load_vcf_dosage(
    path: str | Path,
    missing_value: float = 1.0,
) -> VCFDosage:
    """Read GT fields as 0/1/2 alternate-allele dosage."""
    vcf_path = Path(path)
    if not vcf_path.is_file():
        raise FileNotFoundError(f"Raw genotype VCF not found: {vcf_path}")
    sample_ids: tuple[str, ...] | None = None
    variants = []
    marker_rows = []
    with gzip.open(vcf_path, "rt", encoding="utf-8") as handle:
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
                raise ValueError("DEM benchmark requires biallelic variants")
            format_fields = columns[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record lacks GT: {columns[0]}:{columns[1]}")
            gt_index = format_fields.index("GT")
            variants.append(
                (columns[0], columns[1], columns[2], columns[3], columns[4])
            )
            marker_rows.append(
                [
                    _encode_gt(field, gt_index, missing_value)
                    for field in columns[9:]
                ]
            )
    if sample_ids is None or not marker_rows:
        raise ValueError(f"VCF contains no genotype records: {vcf_path}")
    matrix = np.asarray(marker_rows, dtype=np.float32).T
    return VCFDosage(
        np.ascontiguousarray(matrix),
        sample_ids,
        tuple(variants),
    )


def _split_paths(
    data_directory: Path,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
) -> tuple[Path, Path, Path]:
    outer_path = data_directory / "cv" / f"outer_fold_{outer_fold}"
    raw_path = data_directory / "raw_genotype" / f"outer_fold_{outer_fold}"
    valid_roles = {"train", "test"} if inner_fold is None else {"train", "valid"}
    if role not in valid_roles:
        raise ValueError(f"Invalid split role {role!r}")
    if inner_fold is None:
        split_path = outer_path / "final"
        index_path = outer_path / f"{role}_idx.npy"
        vcf_path = raw_path / f"{role}.vcf.gz"
    else:
        split_path = outer_path / f"inner_fold_{inner_fold}"
        index_path = split_path / f"{role}_idx.npy"
        vcf_path = raw_path / f"inner_fold_{inner_fold}" / f"{role}.vcf.gz"
    return split_path, index_path, vcf_path


def load_complete_case_split(
    data_directory: str | Path,
    metadata: Mapping[str, Any],
    target_mask: torch.Tensor,
    raw_targets: torch.Tensor,
    trait_indices: Sequence[int],
    outer_fold: int,
    inner_fold: int | None,
    role: str,
    missing_genotype: float,
    min_samples: int = 3,
) -> DEMSplit:
    """Align one prepared split and retain rows observed for every output."""
    directory = Path(data_directory)
    split_path, index_path, vcf_path = _split_paths(
        directory, outer_fold, inner_fold, role
    )
    target_path = split_path / f"Y_{role}_processed.pt"
    required = (index_path, target_path, vcf_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DEM benchmark split artifacts are incomplete. Missing: "
            + ", ".join(missing)
        )
    absolute_indices = np.load(index_path, allow_pickle=False)
    processed = _torch_load(target_path)
    if len(absolute_indices) != len(processed):
        raise ValueError(f"Processed targets do not align with {index_path}")
    trait_array = np.asarray(trait_indices, dtype=np.int64)
    sample_ids = metadata["sample_ids"]
    by_sample = {}
    for position, absolute_index in enumerate(absolute_indices):
        index = int(absolute_index)
        complete = bool(target_mask[index, trait_array].all())
        by_sample[str(sample_ids[index])] = (
            processed[position, trait_array].numpy().astype(np.float32),
            raw_targets[index, trait_array].numpy().astype(np.float32),
            complete,
        )
    vcf = load_vcf_dosage(vcf_path, missing_genotype)
    if set(vcf.sample_ids) != set(by_sample):
        raise ValueError(f"VCF samples do not match fold indices: {vcf_path}")
    retained = [
        position
        for position, sample_id in enumerate(vcf.sample_ids)
        if by_sample[sample_id][2]
    ]
    discarded = tuple(
        sample_id for sample_id in vcf.sample_ids if not by_sample[sample_id][2]
    )
    if len(retained) < min_samples:
        raise ValueError(
            "Complete-case filtering left fewer than "
            f"{min_samples} samples in {vcf_path}: {len(retained)}"
        )
    targets = np.stack(
        [by_sample[vcf.sample_ids[position]][0] for position in retained]
    )
    original = np.stack(
        [by_sample[vcf.sample_ids[position]][1] for position in retained]
    )
    if not np.isfinite(targets).all() or np.any(targets == -999):
        raise ValueError("Missing phenotype sentinel entered DEM targets")
    return DEMSplit(
        genotypes=np.ascontiguousarray(vcf.genotypes[retained]),
        targets=np.ascontiguousarray(targets),
        raw_targets=np.ascontiguousarray(original),
        sample_ids=tuple(vcf.sample_ids[position] for position in retained),
        discarded_sample_ids=discarded,
        variants=vcf.variants,
    )


def validate_variant_schema(first: DEMSplit, second: DEMSplit) -> None:
    """Require identical ordered markers between train and held-out splits."""
    if first.variants != second.variants:
        raise ValueError("Training and held-out VCF variant schemas differ")


def fit_rf_selector(
    genotypes: np.ndarray,
    targets: np.ndarray,
    n_features: int,
    n_estimators: int,
    random_states: Sequence[int],
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average multi-output RF importance across deterministic seeds."""
    if genotypes.ndim != 2 or targets.ndim != 2:
        raise ValueError("RF inputs must be two-dimensional")
    if len(genotypes) != len(targets):
        raise ValueError("RF genotypes and targets must align")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    importance = np.zeros((len(random_states), genotypes.shape[1]), dtype=np.float64)
    if len(random_states) == 0:
        raise ValueError("At least one RF random state is required")
    for row, random_state in enumerate(random_states):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=int(random_state),
            n_jobs=n_jobs,
        )
        model.fit(genotypes, targets)
        importance[row] = model.feature_importances_
    averaged = importance.mean(axis=0)
    count = min(int(n_features), genotypes.shape[1])
    selected = np.argsort(-averaged, kind="stable")[:count].astype(np.int64)
    return selected, averaged[selected].astype(np.float32)


def apply_selected_features(split: DEMSplit, indices: np.ndarray) -> DEMSplit:
    """Apply a training-selected marker schema to another split."""
    selected = np.asarray(indices, dtype=np.int64)
    return DEMSplit(
        genotypes=np.ascontiguousarray(split.genotypes[:, selected]),
        targets=split.targets,
        raw_targets=split.raw_targets,
        sample_ids=split.sample_ids,
        discarded_sample_ids=split.discarded_sample_ids,
        variants=tuple(split.variants[index] for index in selected),
    )


def select_split_features(
    train: DEMSplit,
    held_out: DEMSplit,
    n_features: int,
    n_estimators: int,
    random_states: Sequence[int],
    n_jobs: int,
) -> SelectedSplits:
    """Fit RF selection on train only and transform both splits."""
    validate_variant_schema(train, held_out)
    indices, importances = fit_rf_selector(
        train.genotypes,
        train.targets,
        n_features,
        n_estimators,
        random_states,
        n_jobs,
    )
    selected_train = apply_selected_features(train, indices)
    selected_held_out = apply_selected_features(held_out, indices)
    return SelectedSplits(
        train=selected_train,
        held_out=selected_held_out,
        selected_indices=indices,
        selected_variants=selected_train.variants,
        importances=importances,
    )


def load_target_tensors(data_directory: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load original targets and their observation mask."""
    directory = Path(data_directory)
    validate_prepared_directory(directory)
    raw = _torch_load(directory / "Y_raw.pt")
    mask = _torch_load(directory / "Y_mask.pt")
    if raw.ndim != 2 or mask.ndim != 2 or raw.shape != mask.shape:
        raise ValueError("Y_raw.pt and Y_mask.pt must be aligned matrices")
    if mask.dtype != torch.bool:
        raise ValueError("Y_mask.pt must contain a bool tensor")
    return raw, mask
