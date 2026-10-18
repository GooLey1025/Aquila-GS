#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Prepare outer-test folds with one 4:1 train/validation split.

Each outer test is held out for the final prediction. The matching outer
training samples are shuffled once into train and validation; there are no
inner CV folds. Phenotype preprocessing is fit on the HPO training split only
and then applied to validation and outer test.

A holdout JSON from ``aquila_cv.py`` is used as written. An older nested-CV
JSON still supplies the outer tests; its inner folds are replaced by one 4:1
split. Rice655 current-fold GWAS keeps ``--fold-specific-gwas-fold``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from aquila.data.cv import (
    generate_benchmark_folds,
    generate_benchmark_folds_from_assignments,
    validate_benchmark_folds,
    validate_outer_fold_observations,
)
from aquila.scripts.aquila_data_cv import (
    NestedCVDataPreparer,
    PreparationConfig,
)


@dataclass(frozen=True)
class BenchmarkPreparationConfig(PreparationConfig):
    """Configuration for one benchmark dataset."""

    valid_fraction: float | None = None


class BenchmarkDataPreparer(NestedCVDataPreparer):
    """Align genotypes and persist outer-test plus one HPO holdout."""

    def __init__(self, config: BenchmarkPreparationConfig) -> None:
        self.benchmark_config = config
        super().__init__(config)

    def prepare(self) -> Dict[str, Any]:
        self._validate_inputs()
        self._prepare_output_directory()
        genotypes = self._normalize_genotypes(self._parse_genotypes())
        phenotype = self._read_phenotypes()
        aligned = self._align(genotypes, phenotype)
        folds = self._benchmark_folds(aligned["sample_ids"])
        validate_benchmark_folds(folds, len(aligned["sample_ids"]))
        self._validate_fold_specific_gwas(folds)
        observed_counts = validate_outer_fold_observations(
            aligned["target_mask"].numpy(),
            folds,
            phenotype["trait_names"],
            min_observed=self.config.min_observed_per_fold,
        )
        metadata = self._build_metadata(genotypes, phenotype, aligned)
        metadata.pop("inner_folds", None)
        fraction = self._resolved_valid_fraction()
        metadata.update(
            {
                "data_mode": "benchmark",
                "outer_folds": len(folds),
                "valid_fraction": fraction,
                "hpo_split": {
                    "scheme": "single_random",
                    "train_valid_ratio": (
                        f"{1.0 - fraction:.4g}:{fraction:.4g}"
                    ),
                    "valid_fraction": fraction,
                    "seed_offset": "seed + outer_fold + 1",
                },
                "fold_mapping_file": (
                    str(self.config.fold_mapping.resolve())
                    if self.config.fold_mapping is not None
                    else None
                ),
                "min_observed_per_outer_fold": (
                    self.config.min_observed_per_fold
                ),
                "outer_fold_observed_counts": observed_counts,
                "preprocessing": {
                    "skew_threshold": self.config.skew_threshold,
                    "epsilon": self.config.preprocessing_epsilon,
                    "fit_scope": "hpo_train_only",
                },
            }
        )
        if self.config.fold_specific_gwas_fold is not None:
            metadata["fold_specific_gwas"] = {
                "enabled": True,
                "outer_fold": self.config.fold_specific_gwas_fold,
                "selection_scope": "outer_train_only",
                "source_vcf": str(self.config.genotype_file.resolve()),
                "source_vcf_sha256": metadata["source_checksums"][
                    "genotype_sha256"
                ],
                "fold_mapping_file": str(self.config.fold_mapping.resolve()),
                "fold_mapping_sha256": self._sha256(self.config.fold_mapping),
                "variant_counts": {
                    name: len(values)
                    for name, values in genotypes.variant_ids.items()
                },
                "feature_shapes": metadata["feature_shapes"],
            }
        self._save_benchmark_artifacts(aligned, metadata, folds)
        if self.config.save_raw_genotype:
            self._save_benchmark_raw_genotypes(aligned["sample_ids"], folds)
        return metadata

    def _validate_inputs(self) -> None:
        super()._validate_inputs()
        configured = self.benchmark_config.valid_fraction
        if configured is not None and not 0.0 < float(configured) < 1.0:
            raise ValueError("--valid-fraction must be between 0 and 1")

    def _resolved_valid_fraction(self) -> float:
        """Use a holdout JSON fraction, otherwise the CLI value or 0.2."""
        mapped = getattr(self, "_mapping_valid_fraction", None)
        if mapped is not None:
            return float(mapped)
        configured = self.benchmark_config.valid_fraction
        if configured is None:
            return 0.2
        return float(configured)

    def _benchmark_folds(
        self,
        aligned_sample_ids: Sequence[str],
    ) -> list[Dict[str, object]]:
        if self.config.fold_mapping is None:
            return generate_benchmark_folds(
                n_samples=len(aligned_sample_ids),
                outer_folds=self.config.outer_folds,
                valid_fraction=self._resolved_valid_fraction(),
                seed=self.config.seed,
            )
        mapping_path = self.config.fold_mapping
        with mapping_path.open("r", encoding="utf-8") as handle:
            first_character = next(
                (
                    character
                    for character in handle.read(4096)
                    if not character.isspace()
                ),
                "",
            )
        if first_character == "{":
            with mapping_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            schema = str(payload.get("schema", ""))
            if schema == "aquila_holdout_cv":
                return self._load_holdout_fold_mapping(
                    payload,
                    aligned_sample_ids,
                )
            if schema == "aquila_nested_cv":
                nested = self._load_nested_fold_mapping(
                    mapping_path,
                    aligned_sample_ids,
                )
                return [
                    self._replace_inner_folds(outer)
                    for outer in nested
                ]
            raise ValueError(
                "Fold mapping schema must be aquila_holdout_cv or "
                f"aquila_nested_cv, got {schema!r}"
            )
        assignments = self._load_outer_fold_assignments(
            mapping_path,
            aligned_sample_ids,
        )
        return generate_benchmark_folds_from_assignments(
            assignments,
            outer_folds=self.config.outer_folds,
            valid_fraction=self._resolved_valid_fraction(),
            seed=self.config.seed,
        )

    def _load_holdout_fold_mapping(
        self,
        payload: Dict[str, Any],
        aligned_sample_ids: Sequence[str],
    ) -> list[Dict[str, object]]:
        """Use the train, validation, and test IDs written by aquila_cv.py."""
        fraction = payload.get("valid_fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise ValueError("Holdout mapping valid_fraction must be a number")
        if not 0.0 < float(fraction) < 1.0:
            raise ValueError("Holdout mapping valid_fraction must be between 0 and 1")
        configured = self.benchmark_config.valid_fraction
        if (
            configured is not None
            and abs(float(configured) - float(fraction)) > 1e-12
        ):
            raise ValueError(
                f"--valid-fraction {configured} does not match the holdout "
                f"mapping valid_fraction {fraction}"
            )
        self._mapping_valid_fraction = float(fraction)
        positions = {
            sample_id: index
            for index, sample_id in enumerate(aligned_sample_ids)
        }
        mapped_samples = {str(value) for value in payload.get("sample_ids", [])}
        missing = sorted(set(positions) - mapped_samples)
        if missing:
            raise ValueError(
                "Aligned samples missing from fold mapping: "
                + ", ".join(missing[:10])
            )

        def indices(sample_ids: Sequence[object]) -> np.ndarray:
            return np.sort(
                np.asarray(
                    [positions[str(sample_id)] for sample_id in sample_ids],
                    dtype=np.int64,
                )
            )

        folds: list[Dict[str, object]] = []
        for outer in payload.get("folds", []):
            train = indices(outer.get("train", []))
            valid = indices(outer.get("valid", []))
            test = indices(outer.get("test", []))
            recorded_outer_train = outer.get("outer_train")
            outer_train = (
                indices(recorded_outer_train)
                if recorded_outer_train is not None
                else np.sort(np.concatenate([train, valid]))
            )
            folds.append(
                {
                    "fold": int(outer["fold"]),
                    "outer_train": outer_train,
                    "train": train,
                    "valid": valid,
                    "test": test,
                }
            )
        validate_benchmark_folds(folds, len(aligned_sample_ids))
        return sorted(folds, key=lambda fold: int(fold["fold"]))

    def _replace_inner_folds(
        self,
        outer: Dict[str, object],
    ) -> Dict[str, object]:
        """Keep a mapped outer partition and draw one new holdout inside it."""
        from aquila.data.cv import split_train_valid

        outer_train = np.asarray(outer["train"], dtype=np.int64)
        train, valid = split_train_valid(
            outer_train,
            valid_fraction=self._resolved_valid_fraction(),
            seed=self.config.seed + int(outer["fold"]) + 1,
        )
        return {
            "fold": int(outer["fold"]),
            "outer_train": outer_train.copy(),
            "train": train,
            "valid": valid,
            "test": np.asarray(outer["test"], dtype=np.int64).copy(),
        }

    def _save_benchmark_artifacts(
        self,
        aligned: Dict[str, Any],
        metadata: Dict[str, Any],
        folds: Sequence[Dict[str, object]],
    ) -> None:
        output = self.config.output_directory
        torch.save(aligned["features"], output / "X.pt")
        torch.save(aligned["targets"], output / "Y_raw.pt")
        torch.save(aligned["target_mask"], output / "Y_mask.pt")
        cv_directory = output / "cv"
        cv_directory.mkdir(parents=True, exist_ok=True)
        targets = aligned["targets"]
        target_mask = aligned["target_mask"]
        for outer in folds:
            fold_id = int(outer["fold"])
            fold_path = cv_directory / f"outer_fold_{fold_id}"
            fold_path.mkdir(parents=True, exist_ok=True)
            np.save(fold_path / "outer_train_idx.npy", outer["outer_train"])
            np.save(fold_path / "train_idx.npy", outer["train"])
            np.save(fold_path / "valid_idx.npy", outer["valid"])
            np.save(fold_path / "test_idx.npy", outer["test"])
            processor = self._fit_preprocessor(
                targets,
                target_mask,
                outer["train"],
                aligned["trait_names"],
                aligned.get("trait_tasks"),
            )
            processed = processor.apply(targets, target_mask)
            torch.save(
                processed[outer["train"]].contiguous(),
                fold_path / "Y_train_processed.pt",
            )
            torch.save(
                processed[outer["valid"]].contiguous(),
                fold_path / "Y_valid_processed.pt",
            )
            torch.save(
                processed[outer["test"]].contiguous(),
                fold_path / "Y_test_processed.pt",
            )
            processor.save_json(fold_path / "preprocessing.json")
        self._save_benchmark_fold_mapping(
            output / "sample_fold_mapping.txt",
            aligned["sample_ids"],
            folds,
        )
        with (output / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")

    @staticmethod
    def _save_benchmark_fold_mapping(
        path: Path,
        sample_ids: Sequence[str],
        folds: Sequence[Dict[str, object]],
    ) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("SampleIndex\tSampleID\tOuterFold\tRole\n")
            for outer in folds:
                roles = {}
                for index in np.asarray(outer["train"], dtype=np.int64):
                    roles[int(index)] = "train"
                for index in np.asarray(outer["valid"], dtype=np.int64):
                    roles[int(index)] = "valid"
                for index in np.asarray(outer["test"], dtype=np.int64):
                    roles[int(index)] = "test"
                outer_number = int(outer["fold"])
                for index, sample_id in enumerate(sample_ids):
                    handle.write(
                        f"{index}\t{sample_id}\t{outer_number}\t"
                        f"{roles[index]}\n"
                    )

    def _save_benchmark_raw_genotypes(
        self,
        sample_ids: Sequence[str],
        folds: Sequence[Dict[str, object]],
    ) -> None:
        """Reuse the nested VCF writer; inner_fold_0 is the single 4:1 split."""
        adapted = [
            {
                "fold": int(outer["fold"]),
                "train": outer["outer_train"],
                "test": outer["test"],
                "inner": [
                    {
                        "fold": 0,
                        "train": outer["train"],
                        "valid": outer["valid"],
                    }
                ],
            }
            for outer in folds
        ]
        self._save_raw_genotype_subsets(sample_ids, adapted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare aligned Aquila data with 10 outer tests and one 4:1 "
            "train/validation split inside each outer training set."
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
        help="Directory for prepared artifacts.",
    )
    parser.add_argument(
        "--encoding",
        "--encoding-type",
        choices=(
            "token",
            "diploid_onehot",
            "onehot",
            "10classed_onehot",
            "dem_10classed_onehot",
        ),
        default="diploid_onehot",
        help="Genotype encoding passed to aquila.encoding.parse_genotype_file.",
    )
    parser.add_argument(
        "--variant-type",
        choices=("snp", "indel", "sv"),
        default=None,
        help=(
            "Optional explicit encoding type for kept records. Every retained "
            "VCF record is encoded as SNP, INDEL, or SV. This does not filter "
            "by ID; use --id-prefix for that."
        ),
    )
    parser.add_argument(
        "--id-prefix",
        "--variant-type-prefix",
        dest="id_prefix",
        default=None,
        help=(
            "Keep VCF records whose ID starts with one of these prefixes. "
            "Separate alternatives with '|'. "
            "Examples: 'SNP-' or 'SNP- | INDEL-'."
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
        help="Phenotype columns treated as classification tasks.",
    )
    parser.add_argument(
        "--missing-sentinel",
        type=float,
        default=-999.0,
        help="Numeric phenotype value treated as missing.",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=None,
        help=(
            "Number of outer tests generated when --fold-mapping is omitted. "
            "Defaults to 10."
        ),
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=None,
        help=(
            "Validation fraction of each outer training set when this script "
            "draws the split. Default 0.2 is 4:1. A holdout JSON already "
            "records its own fraction."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic fold generation.",
    )
    parser.add_argument(
        "--fold-mapping",
        default=None,
        help=(
            "Holdout JSON from aquila_cv.py (train/valid/test), a legacy "
            "nested-CV JSON, or a SampleID/OuterFold table. A holdout JSON is "
            "used as written. Inner folds in a nested JSON are replaced by "
            "one 4:1 split."
        ),
    )
    parser.add_argument(
        "--fold-specific-gwas-fold",
        type=int,
        default=None,
        metavar="FOLD",
        help=(
            "Bind this prepared dataset to one zero-based outer fold whose "
            "GWAS marker panel was selected from that fold's outer training "
            "samples. Requires --fold-mapping."
        ),
    )
    parser.add_argument(
        "--min-observed-per-fold",
        type=int,
        default=10,
        help="Minimum observed values per trait in every outer test fold.",
    )
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=2.0,
        help="Absolute skewness threshold for per-trait log1p transforms.",
    )
    parser.add_argument(
        "--preprocessing-epsilon",
        type=float,
        default=1e-8,
        help="Numerical epsilon used by phenotype preprocessing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing prepared artifacts in the output directory.",
    )
    parser.add_argument(
        "--save-raw-genotype",
        action="store_true",
        help=(
            "Write sample-subset VCF.GZ files for outer train/test and the "
            "single train/validation split (stored as inner_fold_0)."
        ),
    )
    args = parser.parse_args()
    if args.fold_mapping and args.outer_folds is not None:
        parser.error("--fold-mapping cannot be combined with --outer-folds")
    return args


def main() -> None:
    args = parse_args()
    config = BenchmarkPreparationConfig(
        genotype_file=Path(args.genotype),
        phenotype_file=Path(args.phenotype),
        output_directory=Path(args.output_directory),
        encoding_type=args.encoding,
        variant_type=args.variant_type,
        id_prefix=args.id_prefix,
        sample_id_column=args.sample_id_column,
        traits=args.traits,
        classification_tasks=args.classification_tasks,
        missing_sentinel=args.missing_sentinel,
        outer_folds=args.outer_folds if args.outer_folds is not None else 10,
        inner_folds=0,
        seed=args.seed,
        fold_mapping=Path(args.fold_mapping) if args.fold_mapping else None,
        fold_specific_gwas_fold=args.fold_specific_gwas_fold,
        save_raw_genotype=args.save_raw_genotype,
        min_observed_per_fold=args.min_observed_per_fold,
        skew_threshold=args.skew_threshold,
        preprocessing_epsilon=args.preprocessing_epsilon,
        overwrite=args.overwrite,
        valid_fraction=args.valid_fraction,
    )
    metadata = BenchmarkDataPreparer(config).prepare()
    print(
        f"Prepared {metadata['n_samples']} samples and "
        f"{metadata['n_traits']} traits in {config.output_directory}"
    )
    print(
        f"  Outer tests: {metadata['outer_folds']}; "
        "train:validation = "
        f"{1.0 - metadata['valid_fraction']:.4g}:{metadata['valid_fraction']:.4g}"
    )
    if metadata.get("fold_specific_gwas"):
        print(
            "  Current GWAS fold: "
            f"{metadata['fold_specific_gwas']['outer_fold']}"
        )


if __name__ == "__main__":
    main()
