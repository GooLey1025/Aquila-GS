# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Prepared nested-CV data adapter for DEM-SNP and DEM-Vars."""

# Migrated from: https://github.com/cma2015/DEM

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

MODALITY_ORDER = ("snp", "indel", "sv")
GENOTYPE_COMBINATIONS = (
    ("A", "A"), ("C", "C"), ("G", "G"), ("T", "T"), ("A", "C"),
    ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T"),
)
GENOTYPE_CATEGORY = {
    pair: index for index, pair in enumerate(GENOTYPE_COMBINATIONS, start=1)
}
CHANNELS_PER_MODALITY = {"snp": 10, "indel": 4, "sv": 4}
ENCODING_NAMES = {
    "snp": "DEM_pregv_10class_onehot",
    "indel": "Aquila_Vars_4class_onehot",
    "sv": "Aquila_Vars_4class_onehot",
}
Variant = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class VCFBranch:
    marker_categories: np.ndarray
    sample_ids: tuple[str, ...]
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class VCFGenotypes:
    marker_categories: np.ndarray
    onehot_genotypes: np.ndarray
    sample_ids: tuple[str, ...]
    variants: tuple[Variant, ...]


@dataclass(frozen=True)
class DEMSplit:
    modalities: tuple[np.ndarray, ...]
    marker_categories: tuple[np.ndarray, ...]
    targets: np.ndarray
    raw_targets: np.ndarray
    sample_ids: tuple[str, ...]
    discarded_sample_ids: tuple[str, ...]
    variants: tuple[tuple[Variant, ...], ...]
    modality_names: tuple[str, ...]


@dataclass(frozen=True)
class SelectedSplits:
    train: DEMSplit
    held_out: DEMSplit
    selected_indices: tuple[np.ndarray, ...]
    selected_variants: tuple[tuple[Variant, ...], ...]
    importances: tuple[np.ndarray, ...]
    modality_names: tuple[str, ...]


def _torch_load(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_prepared_directory(data_directory: str | Path) -> None:
    directory = Path(data_directory)
    required = (
        directory / "metadata.json", directory / "Y_raw.pt",
        directory / "Y_mask.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "DEM benchmark prepared data are incomplete. Missing: "
            + ", ".join(missing)
        )


def load_metadata(data_directory: str | Path) -> dict[str, Any]:
    validate_prepared_directory(data_directory)
    with (Path(data_directory) / "metadata.json").open(
        "r", encoding="utf-8"
    ) as handle:
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
    names = tuple(metadata["trait_names"])
    tasks = metadata.get("trait_tasks")
    if tasks is None:
        indices = tuple(range(int(metadata.get("n_regression_tasks", len(names)))))
    else:
        indices = tuple(
            index for index, task in enumerate(tasks) if task == "regression"
        )
    if not indices:
        raise ValueError("DEM benchmark requires at least one regression trait")
    return indices


def resolve_traits(
    metadata: Mapping[str, Any], selected_traits: Sequence[str] | None
) -> tuple[tuple[int, str], ...]:
    names = tuple(str(name) for name in metadata["trait_names"])
    available = {
        names[index]: index for index in regression_trait_indices(metadata)
    }
    if selected_traits is None:
        return tuple((index, names[index]) for index in available.values())
    requested = tuple(dict.fromkeys(str(name) for name in selected_traits))
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown regression traits: {unknown}. Available: {list(available)}"
        )
    return tuple((available[name], name) for name in requested)


def onehot_encode_marker_categories(marker_categories: np.ndarray) -> np.ndarray:
    categories = np.asarray(marker_categories)
    if categories.ndim != 2:
        raise ValueError("Marker genotype categories must be two-dimensional")
    if np.any(categories < 0) or np.any(categories > 10):
        raise ValueError("SNP marker genotype categories must be in 0..10")
    encoded = np.eye(11, dtype=np.float32)[categories.astype(int)]
    return np.ascontiguousarray(encoded[:, :, 1:].reshape(len(categories), -1))


def onehot_encode_genotype_classes(marker_categories: np.ndarray) -> np.ndarray:
    categories = np.asarray(marker_categories)
    if categories.ndim != 2:
        raise ValueError("Marker genotype categories must be two-dimensional")
    if np.any(categories < 0) or np.any(categories > 4):
        raise ValueError("INDEL/SV genotype categories must be in 0..4")
    encoded = np.eye(5, dtype=np.float32)[categories.astype(int)]
    return np.ascontiguousarray(encoded[:, :, 1:].reshape(len(categories), -1))


def expand_marker_categories(
    marker_categories: np.ndarray, modality: str
) -> np.ndarray:
    if modality == "snp":
        return onehot_encode_marker_categories(marker_categories)
    if modality in {"indel", "sv"}:
        return onehot_encode_genotype_classes(marker_categories)
    raise ValueError(f"Unsupported DEM modality: {modality}")


def _classify_variant(identifier: str, reference: str, alternate: str) -> str:
    upper_id = identifier.upper()
    if "INDEL" in upper_id:
        return "indel"
    if "SV" in upper_id:
        return "sv"
    if "SNP" in upper_id:
        return "snp"
    alternates = alternate.split(",")
    if any(
        alt.startswith("<") or "[" in alt or "]" in alt or alt == "*"
        for alt in alternates
    ):
        return "sv"
    if (
        len(reference) == 1
        and reference.upper() in {"A", "C", "G", "T"}
        and all(
            len(alt) == 1 and alt.upper() in {"A", "C", "G", "T"}
            for alt in alternates
        )
    ):
        return "snp"
    return "indel"


def _gt_indices(
    sample_field: str, gt_index: int, location: str
) -> tuple[int, int] | None:
    fields = sample_field.split(":")
    if gt_index >= len(fields):
        raise ValueError(f"VCF sample field lacks GT at {location}")
    gt = fields[gt_index]
    separator = "|" if "|" in gt else "/"
    values = gt.split(separator)
    if len(values) != 2:
        raise ValueError(f"Invalid diploid GT at {location}")
    if "." in values:
        return None
    try:
        genotype = tuple(int(value) for value in values)
    except ValueError as error:
        raise ValueError(f"Non-integer GT allele at {location}") from error
    if any(index < 0 for index in genotype):
        raise ValueError(f"Negative GT allele at {location}")
    return genotype


def _encode_snp(
    genotype: tuple[int, int] | None,
    reference: str,
    alternate: str,
    location: str,
) -> int:
    if genotype is None:
        return 0
    alleles = tuple(
        allele.upper() for allele in (reference, *alternate.split(","))
    )
    if any(
        len(allele) != 1 or allele not in {"A", "C", "G", "T"}
        for allele in alleles
    ):
        raise ValueError(
            f"DEM SNP branch requires single-nucleotide A/C/G/T alleles at {location}"
        )
    if any(index >= len(alleles) for index in genotype):
        raise ValueError(f"GT allele index exceeds REF/ALT list at {location}")
    pair = tuple(sorted((alleles[genotype[0]], alleles[genotype[1]])))
    return GENOTYPE_CATEGORY[pair]


def _encode_genotype_class(
    genotype: tuple[int, int] | None, location: str
) -> int:
    if genotype is None:
        return 0
    if any(index not in {0, 1} for index in genotype):
        raise ValueError(
            f"DEM INDEL/SV branches require biallelic GT values at {location}"
        )
    return {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}[genotype]


def load_vcf_branches(
    path: str | Path, required_modalities: Sequence[str]
) -> dict[str, VCFBranch]:
    vcf_path = Path(path)
    if not vcf_path.is_file():
        raise FileNotFoundError(f"Raw genotype VCF not found: {vcf_path}")
    modalities = tuple(str(value).lower() for value in required_modalities)
    if (
        any(value not in MODALITY_ORDER for value in modalities)
        or len(set(modalities)) != len(modalities)
    ):
        raise ValueError(f"Invalid or duplicate DEM modalities: {modalities}")
    sample_ids: tuple[str, ...] | None = None
    variants: dict[str, list[Variant]] = {name: [] for name in modalities}
    rows: dict[str, list[list[int]]] = {name: [] for name in modalities}
    with gzip.open(vcf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                sample_ids = tuple(line.rstrip("\n").split("\t")[9:])
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
            modality = _classify_variant(columns[2], columns[3], columns[4])
            if modality not in rows:
                continue
            format_fields = columns[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record lacks GT: {columns[0]}:{columns[1]}")
            gt_index = format_fields.index("GT")
            variants[modality].append(
                (columns[0], columns[1], columns[2], columns[3], columns[4])
            )
            encoded = []
            for sample_id, field in zip(sample_ids, columns[9:]):
                location = f"{columns[0]}:{columns[1]} sample {sample_id!r}"
                genotype = _gt_indices(field, gt_index, location)
                encoded.append(
                    _encode_snp(genotype, columns[3], columns[4], location)
                    if modality == "snp"
                    else _encode_genotype_class(genotype, location)
                )
            rows[modality].append(encoded)
    if sample_ids is None:
        raise ValueError(f"VCF has no #CHROM header: {vcf_path}")
    missing = [name for name in modalities if not rows[name]]
    if missing:
        raise ValueError(
            f"VCF is missing required DEM branches {missing}: {vcf_path}. "
            "DEM-Vars requires SNP, INDEL, and SV records."
        )
    return {
        name: VCFBranch(
            np.ascontiguousarray(np.asarray(rows[name], dtype=np.int8).T),
            sample_ids,
            tuple(variants[name]),
        )
        for name in modalities
    }


def load_vcf_genotypes(path: str | Path) -> VCFGenotypes:
    branch = load_vcf_branches(path, ("snp",))["snp"]
    return VCFGenotypes(
        branch.marker_categories,
        onehot_encode_marker_categories(branch.marker_categories),
        branch.sample_ids,
        branch.variants,
    )


def load_vcf_dosage(path: str | Path) -> VCFGenotypes:
    """Compatibility alias for the original SNP-only benchmark loader."""

    return load_vcf_genotypes(path)


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


def load_split(
    data_directory: str | Path,
    metadata: Mapping[str, Any],
    target_mask: torch.Tensor,
    raw_targets: torch.Tensor,
    trait_index: int,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
    min_observed: int = 2,
    modalities: Sequence[str] = ("snp",),
) -> DEMSplit:
    trait_index = int(trait_index)
    directory = Path(data_directory)
    split_path, index_path, vcf_path = _split_paths(
        directory, outer_fold, inner_fold, role
    )
    target_path = split_path / f"Y_{role}_processed.pt"
    missing = [
        str(path) for path in (index_path, target_path, vcf_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "DEM benchmark split artifacts are incomplete. Missing: "
            + ", ".join(missing)
        )
    absolute_indices = np.load(index_path, allow_pickle=False)
    processed = _torch_load(target_path)
    if len(absolute_indices) != len(processed):
        raise ValueError(f"Processed targets do not align with {index_path}")
    cohort_ids = metadata["sample_ids"]
    by_sample = {
        str(cohort_ids[int(absolute_index)]): (
            float(processed[position, trait_index]),
            float(raw_targets[int(absolute_index), trait_index]),
            bool(target_mask[int(absolute_index), trait_index]),
        )
        for position, absolute_index in enumerate(absolute_indices)
    }
    modality_names = tuple(str(value).lower() for value in modalities)
    branches = load_vcf_branches(vcf_path, modality_names)
    first = branches[modality_names[0]]
    if set(first.sample_ids) != set(by_sample):
        raise ValueError(f"VCF samples do not match fold indices: {vcf_path}")
    observed = tuple(
        sample_id for sample_id in first.sample_ids if by_sample[sample_id][2]
    )
    discarded = tuple(
        sample_id for sample_id in first.sample_ids if not by_sample[sample_id][2]
    )
    if len(observed) < min_observed:
        trait_name = metadata["trait_names"][trait_index]
        raise ValueError(
            f"Split has fewer than {min_observed} observed values for "
            f"trait {trait_name!r}: {vcf_path}"
        )
    positions = np.asarray(
        [first.sample_ids.index(sample_id) for sample_id in observed],
        dtype=np.int64,
    )
    categories = tuple(
        np.ascontiguousarray(branches[name].marker_categories[positions])
        for name in modality_names
    )
    targets = np.asarray(
        [[by_sample[sample_id][0]] for sample_id in observed], dtype=np.float32
    )
    original = np.asarray(
        [[by_sample[sample_id][1]] for sample_id in observed], dtype=np.float32
    )
    if (
        not np.isfinite(targets).all()
        or not np.isfinite(original).all()
        or np.any(targets == -999)
        or np.any(original == -999)
    ):
        raise ValueError("Retained DEM targets must be finite and non-sentinel")
    return DEMSplit(
        tuple(
            expand_marker_categories(values, name)
            for name, values in zip(modality_names, categories)
        ),
        categories,
        targets,
        original,
        observed,
        discarded,
        tuple(branches[name].variants for name in modality_names),
        modality_names,
    )


def validate_variant_schema(first: DEMSplit, second: DEMSplit) -> None:
    if first.modality_names != second.modality_names:
        raise ValueError("Training and held-out DEM modality orders differ")
    for name, first_schema, second_schema in zip(
        first.modality_names, first.variants, second.variants
    ):
        if first_schema != second_schema:
            raise ValueError(
                f"Training and held-out VCF variant schemas differ for {name}"
            )


def fit_rf_selector(
    genotypes: np.ndarray,
    targets: np.ndarray,
    n_features: int,
    n_estimators: int,
    random_states: Sequence[int],
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a single-trait RF selector on retained training rows only."""
    x = np.asarray(genotypes)
    y = np.asarray(targets).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 2:
        raise ValueError("RF selection requires aligned single-trait training data")
    if int(n_features) < 1 or not random_states:
        raise ValueError("RF feature count and random states must be non-empty")
    seed_importances = []
    for random_state in random_states:
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )
        model.fit(x, y)
        values = np.asarray(model.feature_importances_, dtype=np.float64)
        total = float(values.sum())
        seed_importances.append(values / total if total > 0.0 else values)
    averaged = np.mean(seed_importances, axis=0)
    count = min(int(n_features), x.shape[1])
    selected = np.argsort(-averaged, kind="stable")[:count].astype(np.int64)
    return selected, averaged[selected].astype(np.float32)


def apply_selected_features(
    split: DEMSplit, indices: np.ndarray, branch_index: int = 0
) -> DEMSplit:
    selected = np.asarray(indices, dtype=np.int64)
    categories = list(split.marker_categories)
    modalities = list(split.modalities)
    schemas = list(split.variants)
    categories[branch_index] = np.ascontiguousarray(
        categories[branch_index][:, selected]
    )
    name = split.modality_names[branch_index]
    modalities[branch_index] = expand_marker_categories(
        categories[branch_index], name
    )
    schemas[branch_index] = tuple(schemas[branch_index][index] for index in selected)
    return DEMSplit(
        tuple(modalities), tuple(categories), split.targets, split.raw_targets,
        split.sample_ids, split.discarded_sample_ids, tuple(schemas),
        split.modality_names,
    )


def select_split_features(
    train: DEMSplit,
    held_out: DEMSplit,
    enabled: bool,
    n_features: int | Mapping[str, int],
    n_estimators: int,
    random_states: Sequence[int],
    n_jobs: int,
) -> SelectedSplits:
    validate_variant_schema(train, held_out)
    selected_train = train
    selected_held = held_out
    all_indices = []
    all_importances = []
    for branch_index, name in enumerate(train.modality_names):
        marker_count = train.marker_categories[branch_index].shape[1]
        requested = (
            int(n_features.get(name, marker_count))
            if isinstance(n_features, Mapping)
            else int(n_features)
        )
        if enabled:
            indices, importances = fit_rf_selector(
                train.marker_categories[branch_index],
                train.targets,
                requested,
                n_estimators,
                random_states,
                n_jobs,
            )
            selected_train = apply_selected_features(
                selected_train, indices, branch_index
            )
            selected_held = apply_selected_features(
                selected_held, indices, branch_index
            )
        else:
            indices = np.arange(marker_count, dtype=np.int64)
            importances = np.full(marker_count, np.nan, dtype=np.float32)
        all_indices.append(indices)
        all_importances.append(importances)
    return SelectedSplits(
        selected_train,
        selected_held,
        tuple(all_indices),
        selected_train.variants,
        tuple(all_importances),
        train.modality_names,
    )


def load_target_tensors(
    data_directory: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    directory = Path(data_directory)
    validate_prepared_directory(directory)
    raw = _torch_load(directory / "Y_raw.pt")
    mask = _torch_load(directory / "Y_mask.pt")
    if raw.ndim != 2 or mask.ndim != 2 or raw.shape != mask.shape:
        raise ValueError("Y_raw.pt and Y_mask.pt must be aligned matrices")
    if mask.dtype != torch.bool:
        raise ValueError("Y_mask.pt must contain a bool tensor")
    return raw, mask
