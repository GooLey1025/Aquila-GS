# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/GSBreeder/BNNs

"""Leakage-safe data loading and marker selection for the BNN benchmark."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler


Variant = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class VCFDosage:
    genotypes: np.ndarray
    sample_ids: tuple[str, ...]
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class BNNSplit:
    genotypes: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    sample_ids: tuple[str, ...]
    discarded_sample_ids: tuple[str, ...]
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class MarkerPipeline:
    imputation_values: np.ndarray
    scale_min: np.ndarray
    scale_range: np.ndarray
    selected_indices: np.ndarray
    selected_variants: tuple[Variant, ...]
    lasso_coefficients: np.ndarray


@dataclass(frozen=True)
class PreparedPair:
    train: BNNSplit
    held_out: BNNSplit
    train_features: np.ndarray
    held_out_features: np.ndarray
    pipeline: MarkerPipeline


def _torch_load(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _encode_gt(sample_field: str, gt_index: int) -> float:
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
    if any(value not in {0, 1} for value in values):
        raise ValueError("BNN benchmark requires biallelic diploid genotypes")
    return float(sum(values))


def load_vcf_dosage(path: str | Path) -> VCFDosage:
    """Read biallelic GT fields as 0/1/2 dosage and NaN missing values."""

    vcf_path = Path(path)
    if not vcf_path.is_file():
        raise FileNotFoundError(f"Raw genotype VCF not found: {vcf_path}")
    sample_ids: tuple[str, ...] | None = None
    variants = []
    rows = []
    with gzip.open(vcf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                sample_ids = tuple(line.rstrip("\n").split("\t")[9:])
                if not sample_ids or len(sample_ids) != len(set(sample_ids)):
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
                raise ValueError("BNN benchmark requires biallelic variants")
            format_fields = columns[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record lacks GT: {columns[0]}:{columns[1]}")
            gt_index = format_fields.index("GT")
            variants.append(
                (columns[0], columns[1], columns[2], columns[3], columns[4])
            )
            rows.append([_encode_gt(value, gt_index) for value in columns[9:]])
    if sample_ids is None or not rows:
        raise ValueError(f"VCF contains no genotype records: {vcf_path}")
    return VCFDosage(
        np.ascontiguousarray(np.asarray(rows, dtype=np.float32).T),
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
        return (
            outer_path / "final",
            outer_path / f"{role}_idx.npy",
            raw_path / f"{role}.vcf.gz",
        )
    split_path = outer_path / f"inner_fold_{inner_fold}"
    return (
        split_path,
        split_path / f"{role}_idx.npy",
        raw_path / f"inner_fold_{inner_fold}" / f"{role}.vcf.gz",
    )


def load_trait_split(
    data_directory: str | Path,
    metadata: Mapping[str, Any],
    target_mask: torch.Tensor,
    raw_targets: torch.Tensor,
    trait_index: int,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
    min_samples: int = 3,
) -> BNNSplit:
    """Load one fixed split and independently remove missing target rows."""

    split_path, index_path, vcf_path = _split_paths(
        Path(data_directory), outer_fold, inner_fold, role
    )
    target_path = split_path / f"Y_{role}_processed.pt"
    missing = [
        str(path)
        for path in (index_path, target_path, vcf_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "BNN benchmark split artifacts are incomplete. Missing: "
            + ", ".join(missing)
        )
    absolute_indices = np.load(index_path, allow_pickle=False)
    processed = _torch_load(target_path)
    if len(absolute_indices) != len(processed):
        raise ValueError(f"Processed targets do not align with {index_path}")
    sample_ids = metadata["sample_ids"]
    by_sample = {
        str(sample_ids[int(index)]): (
            float(processed[position, trait_index]),
            float(raw_targets[int(index), trait_index]),
            bool(target_mask[int(index), trait_index]),
        )
        for position, index in enumerate(absolute_indices)
    }
    vcf = load_vcf_dosage(vcf_path)
    if set(vcf.sample_ids) != set(by_sample):
        raise ValueError(f"VCF samples do not match fold indices: {vcf_path}")
    retained = [
        position
        for position, sample_id in enumerate(vcf.sample_ids)
        if by_sample[sample_id][2]
    ]
    if len(retained) < min_samples:
        raise ValueError(
            f"Trait has fewer than {min_samples} observed samples in {vcf_path}"
        )
    targets = np.asarray(
        [by_sample[vcf.sample_ids[position]][0] for position in retained],
        dtype=np.float32,
    )
    raw = np.asarray(
        [by_sample[vcf.sample_ids[position]][1] for position in retained],
        dtype=np.float32,
    )
    if not np.isfinite(targets).all() or np.any(targets == -999):
        raise ValueError("Missing phenotype sentinel entered BNN targets")
    retained_ids = tuple(vcf.sample_ids[position] for position in retained)
    retained_set = set(retained_ids)
    return BNNSplit(
        np.ascontiguousarray(vcf.genotypes[retained]),
        targets,
        raw,
        retained_ids,
        tuple(sample_id for sample_id in vcf.sample_ids if sample_id not in retained_set),
        vcf.variants,
    )


def validate_variant_schema(train: BNNSplit, held_out: BNNSplit) -> None:
    if train.variants != held_out.variants:
        raise ValueError("Training and held-out VCF variant schemas differ")


def fit_marker_pipeline(
    genotypes: np.ndarray,
    targets: np.ndarray,
    variants: tuple[Variant, ...],
    alpha: float,
    max_features: int,
    seed: int,
) -> MarkerPipeline:
    """Fit imputation, scaling, and LASSO selection on training rows only."""

    values = np.asarray(genotypes, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("Training genotypes and targets must align")
    observed = np.isfinite(values)
    counts = observed.sum(axis=0)
    sums = np.where(observed, values, 0.0).sum(axis=0)
    means = np.divide(
        sums,
        counts,
        out=np.ones(values.shape[1], dtype=np.float64),
        where=counts > 0,
    ).astype(np.float32)
    imputed = np.where(observed, values, means)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(imputed)
    lasso = Lasso(
        alpha=float(alpha),
        random_state=int(seed),
        max_iter=50000,
        tol=1e-3,
    )
    selector = SelectFromModel(
        lasso,
        threshold=1e-12,
        max_features=min(int(max_features), values.shape[1]),
    )
    selector.fit(scaled, np.asarray(targets, dtype=np.float32))
    selected = selector.get_support(indices=True).astype(np.int64)
    coefficients = np.asarray(selector.estimator_.coef_, dtype=np.float32)
    if selected.size == 0:
        ranked = np.argsort(-np.abs(coefficients), kind="stable")
        selected = ranked[: min(int(max_features), len(ranked))].astype(np.int64)
    return MarkerPipeline(
        imputation_values=means,
        scale_min=np.asarray(scaler.data_min_, dtype=np.float32),
        scale_range=np.asarray(scaler.data_range_, dtype=np.float32),
        selected_indices=selected,
        selected_variants=tuple(variants[index] for index in selected),
        lasso_coefficients=coefficients[selected],
    )


def transform_markers(
    genotypes: np.ndarray,
    pipeline: MarkerPipeline,
) -> np.ndarray:
    """Apply immutable training-fitted preprocessing to another split."""

    values = np.asarray(genotypes, dtype=np.float32)
    if values.shape[1] != len(pipeline.imputation_values):
        raise ValueError("Genotype marker count does not match fitted pipeline")
    imputed = np.where(np.isfinite(values), values, pipeline.imputation_values)
    denominator = np.where(pipeline.scale_range > 0, pipeline.scale_range, 1.0)
    scaled = (imputed - pipeline.scale_min) / denominator
    return np.ascontiguousarray(
        scaled[:, pipeline.selected_indices].astype(np.float32)
    )


def prepare_pair(
    train: BNNSplit,
    held_out: BNNSplit,
    alpha: float,
    max_features: int,
    seed: int,
) -> PreparedPair:
    """Fit marker processing on train and apply it unchanged to held-out rows."""

    validate_variant_schema(train, held_out)
    pipeline = fit_marker_pipeline(
        train.genotypes,
        train.targets,
        train.variants,
        alpha,
        max_features,
        seed,
    )
    return PreparedPair(
        train,
        held_out,
        transform_markers(train.genotypes, pipeline),
        transform_markers(held_out.genotypes, pipeline),
        pipeline,
    )
