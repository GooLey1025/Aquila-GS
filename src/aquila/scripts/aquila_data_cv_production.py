#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Prepare aligned full-dataset artifacts for Aquila production CV training."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from sklearn.model_selection import KFold

from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.scripts.aquila_data_cv import (
    NestedCVDataPreparer,
    PreparationConfig,
)


@dataclass(frozen=True)
class DeploymentPreparationConfig:
    """Configuration for preparing one full-data deployment dataset."""

    genotype_file: Path
    phenotype_file: Path
    output_directory: Path
    encoding_type: str = "diploid_onehot"
    variant_type: str | None = None
    id_prefix: str | None = None
    normalize_heterozygous_order: bool = False
    sample_id_column: str | None = None
    traits: Sequence[str] | None = None
    classification_tasks: Sequence[str] | None = None
    missing_sentinel: float = -999.0
    folds: int = 5
    seed: int = 42
    skew_threshold: float = 2.0
    preprocessing_epsilon: float = 1e-8
    overwrite: bool = False


@dataclass(frozen=True)
class DeploymentFold:
    """One persisted train/validation split over the full reference dataset."""

    fold: int
    train: np.ndarray
    valid: np.ndarray


def generate_deployment_folds(
    sample_count: int,
    fold_count: int = 5,
    seed: int = 42,
) -> list[DeploymentFold]:
    """Generate deterministic K-fold splits for deployment model selection."""
    if sample_count < 2:
        raise ValueError("At least two samples are required for production CV")
    if not 2 <= int(fold_count) <= int(sample_count):
        raise ValueError(
            f"folds must be between 2 and {sample_count}, got {fold_count}"
        )
    indices = np.arange(sample_count, dtype=np.int64)
    splitter = KFold(
        n_splits=int(fold_count),
        shuffle=True,
        random_state=int(seed),
    )
    return [
        DeploymentFold(
            fold=fold_id,
            train=indices[train_positions].copy(),
            valid=indices[valid_positions].copy(),
        )
        for fold_id, (train_positions, valid_positions) in enumerate(
            splitter.split(indices)
        )
    ]


def validate_deployment_folds(
    folds: Sequence[DeploymentFold],
    sample_count: int,
) -> None:
    """Require validation folds to form a complete, disjoint partition."""
    if not folds:
        raise ValueError("At least one production fold is required")
    all_indices = np.arange(sample_count, dtype=np.int64)
    validation_parts = []
    for expected_fold, split in enumerate(folds):
        if int(split.fold) != expected_fold:
            raise ValueError("Deployment fold identifiers must be zero-based")
        train = np.asarray(split.train, dtype=np.int64)
        valid = np.asarray(split.valid, dtype=np.int64)
        if train.ndim != 1 or valid.ndim != 1 or not train.size or not valid.size:
            raise ValueError(f"Fold {split.fold} must have nonempty 1D splits")
        if np.intersect1d(train, valid).size:
            raise ValueError(f"Fold {split.fold} train and validation sets overlap")
        if not np.array_equal(np.sort(np.concatenate((train, valid))), all_indices):
            raise ValueError(f"Fold {split.fold} does not cover every sample")
        validation_parts.append(valid)
    if not np.array_equal(np.sort(np.concatenate(validation_parts)), all_indices):
        raise ValueError("Validation folds must partition every sample exactly once")


class DeploymentDataPreparer(NestedCVDataPreparer):
    """Prepare fixed K-fold caches and a full-data cache for rapid training."""

    _ARTIFACTS = (
        "X.pt",
        "Y_raw.pt",
        "Y_mask.pt",
        "metadata.json",
        "samples.tsv",
        "fold_assignments.tsv",
        "cv",
        "full",
    )

    def __init__(self, config: DeploymentPreparationConfig) -> None:
        self.deployment_config = config
        super().__init__(
            PreparationConfig(
                genotype_file=config.genotype_file,
                phenotype_file=config.phenotype_file,
                output_directory=config.output_directory,
                encoding_type=config.encoding_type,
                variant_type=config.variant_type,
                id_prefix=config.id_prefix,
                sample_id_column=config.sample_id_column,
                traits=config.traits,
                classification_tasks=config.classification_tasks,
                missing_sentinel=config.missing_sentinel,
                seed=config.seed,
                skew_threshold=config.skew_threshold,
                preprocessing_epsilon=config.preprocessing_epsilon,
                overwrite=config.overwrite,
            )
        )

    def prepare(self) -> Dict[str, Any]:
        """Parse, align, split, preprocess, and persist deployment CV data."""
        self._validate_deployment_inputs()
        self._prepare_deployment_output()
        parsed = self._parse_genotypes()
        genotypes = self._normalize_genotypes(parsed)
        phenotype = self._read_phenotypes()
        aligned = self._align(genotypes, phenotype)
        folds = generate_deployment_folds(
            len(aligned["sample_ids"]),
            fold_count=self.deployment_config.folds,
            seed=self.deployment_config.seed,
        )
        validate_deployment_folds(folds, len(aligned["sample_ids"]))
        metadata = self._build_metadata(genotypes, phenotype, aligned)
        metadata.update(
            {
                "data_mode": "deployment",
                "cv_folds": len(folds),
                "cv_seed": self.deployment_config.seed,
                "normalize_heterozygous_order": (
                    self.deployment_config.normalize_heterozygous_order
                ),
                "preprocessing": {
                    "skew_threshold": self.deployment_config.skew_threshold,
                    "epsilon": self.deployment_config.preprocessing_epsilon,
                    "fit_scope": "fold_train_and_full_dataset",
                },
            }
        )
        metadata.pop("outer_folds", None)
        metadata.pop("inner_folds", None)
        metadata.pop("raw_genotype_saved", None)
        self._save_deployment_artifacts(aligned, metadata, folds)
        return metadata

    def _parse_genotypes(self) -> Any:
        from aquila.encoding import (
            normalize_heterozygous_order,
            parse_genotype_file,
        )

        parsed = parse_genotype_file(
            str(self.deployment_config.genotype_file),
            encoding_type=self.deployment_config.encoding_type,
            variant_type=self.deployment_config.variant_type,
            id_prefix=self.deployment_config.id_prefix,
        )
        if self.deployment_config.normalize_heterozygous_order:
            normalized = normalize_heterozygous_order(parsed)
            print(
                f"Normalized {normalized} ALT/REF heterozygous call(s) "
                "to REF/ALT order"
            )
        return parsed

    def _validate_deployment_inputs(self) -> None:
        config = self.deployment_config
        if not config.genotype_file.is_file():
            raise FileNotFoundError(f"Genotype file not found: {config.genotype_file}")
        if not config.phenotype_file.is_file():
            raise FileNotFoundError(
                f"Phenotype file not found: {config.phenotype_file}"
            )
        if not math.isfinite(config.missing_sentinel):
            raise ValueError("missing_sentinel must be finite")
        if config.skew_threshold < 0:
            raise ValueError("skew_threshold must be nonnegative")
        if config.preprocessing_epsilon <= 0:
            raise ValueError("preprocessing_epsilon must be positive")
        if config.folds < 2:
            raise ValueError("folds must be at least two")

    def _prepare_deployment_output(self) -> None:
        output = self.deployment_config.output_directory
        existing = [name for name in self._ARTIFACTS if (output / name).exists()]
        if existing and not self.deployment_config.overwrite:
            raise FileExistsError(
                "Output artifacts already exist; use --overwrite to replace them: "
                + ", ".join(existing)
            )
        output.mkdir(parents=True, exist_ok=True)
        if self.deployment_config.overwrite:
            for name in self._ARTIFACTS:
                path = output / name
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()

    def _save_deployment_artifacts(
        self,
        aligned: Dict[str, Any],
        metadata: Dict[str, Any],
        folds: Sequence[DeploymentFold],
    ) -> None:
        output = self.deployment_config.output_directory
        torch.save(aligned["features"], output / "X.pt")
        torch.save(aligned["targets"], output / "Y_raw.pt")
        torch.save(aligned["target_mask"], output / "Y_mask.pt")
        with (output / "samples.tsv").open("w", encoding="utf-8") as handle:
            handle.write("SampleIndex\tSampleID\n")
            for index, sample_id in enumerate(aligned["sample_ids"]):
                handle.write(f"{index}\t{sample_id}\n")
        assignments = np.full(len(aligned["sample_ids"]), -1, dtype=np.int64)
        for split in folds:
            assignments[split.valid] = split.fold
            fold_path = output / "cv" / f"fold_{split.fold}"
            fold_path.mkdir(parents=True, exist_ok=True)
            np.save(fold_path / "train_idx.npy", split.train)
            np.save(fold_path / "valid_idx.npy", split.valid)
            processor = self._fit_deployment_processor(aligned, split.train)
            processed = processor.apply(
                aligned["targets"],
                aligned["target_mask"],
            )
            torch.save(
                processed[split.train].contiguous(),
                fold_path / "Y_train_processed.pt",
            )
            torch.save(
                processed[split.valid].contiguous(),
                fold_path / "Y_valid_processed.pt",
            )
            processor.save_json(fold_path / "preprocessing.json")
        with (output / "fold_assignments.tsv").open(
            "w", encoding="utf-8"
        ) as handle:
            handle.write("SampleIndex\tSampleID\tValidationFold\n")
            for index, sample_id in enumerate(aligned["sample_ids"]):
                handle.write(f"{index}\t{sample_id}\t{assignments[index]}\n")
        all_indices = np.arange(len(aligned["sample_ids"]), dtype=np.int64)
        full_processor = self._fit_deployment_processor(aligned, all_indices)
        full_targets = full_processor.apply(
            aligned["targets"],
            aligned["target_mask"],
        )
        full_path = output / "full"
        full_path.mkdir(parents=True, exist_ok=True)
        torch.save(full_targets.contiguous(), full_path / "Y_processed.pt")
        full_processor.save_json(full_path / "preprocessing.json")
        with (output / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")

    def _fit_deployment_processor(
        self,
        aligned: Dict[str, Any],
        indices: np.ndarray,
    ) -> PerTraitPreprocessor:
        return PerTraitPreprocessor(
            skew_threshold=self.deployment_config.skew_threshold,
            epsilon=self.deployment_config.preprocessing_epsilon,
        ).fit(
            aligned["targets"],
            aligned["target_mask"],
            indices,
            aligned["trait_names"],
            trait_tasks=aligned["trait_tasks"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare aligned K-fold data and preprocessing caches for "
            "aquila-train-cv-production."
        )
    )
    parser.add_argument(
        "--genotype",
        "--geno",
        "--vcf",
        dest="genotype",
        required=True,
        help="Input genotype or VCF file.",
    )
    parser.add_argument(
        "--phenotype",
        "--pheno",
        dest="phenotype",
        required=True,
        help="Input phenotype table.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--output-directory",
        dest="output_directory",
        required=True,
        help="Directory for production prepared-data artifacts.",
    )
    parser.add_argument(
        "--encoding",
        "--encoding-type",
        choices=("token", "diploid_onehot", "onehot", "10classed_onehot"),
        default="diploid_onehot",
        help="Genotype encoding passed to aquila.encoding.parse_genotype_file.",
    )
    parser.add_argument(
        "--variant-type",
        choices=("snp", "indel", "sv"),
        default=None,
        help=(
            "How to encode every kept VCF record: snp, indel, or sv. "
            "This does not filter by ID. When omitted, types are detected "
            "automatically and stored as separate branches."
        ),
    )
    parser.add_argument(
        "--id-prefix",
        "--variant-type-prefix",
        dest="id_prefix",
        default=None,
        help=(
            "Keep VCF records whose ID starts with one of the supplied "
            "prefixes, for example 'SNP-' or 'SNP- | INDEL-'. "
            "This filters records and does not choose the encoding."
        ),
    )
    parser.add_argument(
        "--normalize-heterozygous-order",
        action="store_true",
        help=(
            "Treat heterozygous allele order as unphased by normalizing 1/0 "
            "and 1|0 to the same REF/ALT encoding as 0/1 before saving X.pt. "
            "The input VCF is not modified."
        ),
    )
    parser.add_argument(
        "--sample-id-column",
        default=None,
        help="Phenotype sample ID column; defaults to the first column.",
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Phenotype columns to retain; defaults to all non-ID columns.",
    )
    parser.add_argument(
        "--classification-tasks",
        nargs="+",
        default=None,
        help="Selected phenotype columns treated as classification tasks.",
    )
    parser.add_argument(
        "--missing-sentinel",
        type=float,
        default=-999.0,
        help="Numeric phenotype value treated as missing.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of persisted K-fold validation splits (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Default reproducibility seed recorded in metadata.",
    )
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=2.0,
        help="Absolute skewness threshold used during runtime preprocessing.",
    )
    parser.add_argument(
        "--preprocessing-epsilon",
        type=float,
        default=1e-8,
        help="Numerical epsilon used during runtime phenotype preprocessing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing production prepared-data artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DeploymentPreparationConfig(
        genotype_file=Path(args.genotype),
        phenotype_file=Path(args.phenotype),
        output_directory=Path(args.output_directory),
        encoding_type=args.encoding,
        variant_type=args.variant_type,
        id_prefix=args.id_prefix,
        normalize_heterozygous_order=args.normalize_heterozygous_order,
        sample_id_column=args.sample_id_column,
        traits=args.traits,
        classification_tasks=args.classification_tasks,
        missing_sentinel=args.missing_sentinel,
        folds=args.folds,
        seed=args.seed,
        skew_threshold=args.skew_threshold,
        preprocessing_epsilon=args.preprocessing_epsilon,
        overwrite=args.overwrite,
    )
    metadata = DeploymentDataPreparer(config).prepare()
    print(
        f"Prepared production data with {metadata['n_samples']} samples and "
        f"{metadata['n_traits']} traits at {config.output_directory}"
    )
    print(
        f"Saved {metadata['cv_folds']} CV folds and full-data preprocessing "
        "caches for aquila-train-cv-production."
    )


if __name__ == "__main__":
    main()
