#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Prepare an outer-holdout dataset: 10x (1:9) outer, then one 4:1 inner split.

Each outer test fold is held out for final scoring. The matching outer train
set is split once into inner train and inner valid at 4:1. Phenotype
preprocessing is fit on inner train only and applied to inner valid and outer
test. There is no outer-train refit cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from aquila.data.cv import generate_outer_holdout_folds
from aquila.scripts.aquila_data_cv import (
    NestedCVDataPreparer,
    PreparationConfig,
)


class OuterHoldoutDataPreparer(NestedCVDataPreparer):
    """Persist one inner holdout inside each outer training partition."""

    def _build_folds(self, sample_ids: Sequence[str]) -> List[Dict[str, object]]:
        if self.config.fold_mapping is None:
            return generate_outer_holdout_folds(
                n_samples=len(sample_ids),
                outer_folds=self.config.outer_folds,
                seed=self.config.seed,
            )
        return self._load_holdout_mapping(sample_ids)

    def _load_holdout_mapping(
        self,
        aligned_sample_ids: Sequence[str],
    ) -> List[Dict[str, object]]:
        """Load the single inner split written by aquila_cv_holdout.py."""
        mapping_path = self.config.fold_mapping
        if mapping_path is None:
            raise RuntimeError("Fold mapping path is not configured")
        with mapping_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != "aquila_outer_holdout":
            raise ValueError(
                "outer-holdout data preparation requires a mapping from "
                "aquila_cv_holdout.py (schema aquila_outer_holdout)"
            )
        outer_count = int(payload.get("outer_folds", -1))
        inner_count = int(payload.get("inner_folds", -1))
        if outer_count < 2:
            raise ValueError("Holdout mapping must contain at least two outer folds")
        if inner_count != 1:
            raise ValueError("Holdout mapping must contain exactly one inner split")

        positions = {
            sample_id: index
            for index, sample_id in enumerate(aligned_sample_ids)
        }
        aligned_set = set(positions)
        mapped_samples = {str(value) for value in payload.get("sample_ids", [])}
        if self.config.fold_specific_gwas_fold is not None:
            missing_from_genotype = sorted(mapped_samples - aligned_set)
            if missing_from_genotype:
                raise ValueError(
                    "Fold-specific GWAS genotype must contain every sample in "
                    "the holdout mapping; missing genotype samples: "
                    + ", ".join(missing_from_genotype[:10])
                )
        missing = sorted(aligned_set - mapped_samples)
        if missing:
            raise ValueError(
                "Aligned samples missing from fold mapping: "
                + ", ".join(missing[:10])
            )

        def indices(sample_ids: Sequence[object]) -> np.ndarray:
            return np.asarray(
                [
                    positions[str(sample_id)]
                    for sample_id in sample_ids
                    if str(sample_id) in positions
                ],
                dtype=np.int64,
            )

        folds: List[Dict[str, object]] = []
        for outer in payload.get("folds", []):
            inner_splits = [
                {
                    "fold": int(inner["fold"]),
                    "train": indices(inner.get("train", [])),
                    "valid": indices(inner.get("valid", [])),
                }
                for inner in outer.get("inner", [])
            ]
            folds.append(
                {
                    "fold": int(outer["fold"]),
                    "train": indices(outer.get("train", [])),
                    "test": indices(outer.get("test", [])),
                    "inner": inner_splits,
                }
            )
        self._validate_nested_mapping(
            folds,
            len(aligned_sample_ids),
            outer_count,
            inner_count,
        )
        return sorted(folds, key=lambda fold: int(fold["fold"]))

    def _annotate_protocol(
        self,
        metadata: Dict[str, Any],
        folds: Sequence[Dict[str, object]],
    ) -> None:
        if not folds or any(len(outer["inner"]) != 1 for outer in folds):
            raise ValueError("outer_holdout preparation requires exactly one inner split")
        metadata["protocol"] = "outer_holdout"
        metadata["inner_folds"] = 1
        metadata["inner_train_valid_ratio"] = [4, 1]
        preprocessing = metadata.get("preprocessing")
        if isinstance(preprocessing, dict):
            preprocessing["fit_scope"] = "inner_train_only"

    def _save_processed_targets(
        self,
        aligned: Dict[str, Any],
        folds: Sequence[Dict[str, object]],
        cv_directory: Path,
    ) -> None:
        """Cache inner-train preprocessing for train, valid, and outer test."""
        targets = aligned["targets"]
        target_mask = aligned["target_mask"]
        trait_names = aligned["trait_names"]
        trait_tasks = aligned.get("trait_tasks")
        for outer in folds:
            outer_id = int(outer["fold"])
            outer_test = outer["test"]
            outer_path = cv_directory / f"outer_fold_{outer_id}"
            if len(outer["inner"]) != 1:
                raise ValueError(
                    f"Outer fold {outer_id} must contain exactly one inner holdout"
                )
            inner = outer["inner"][0]
            inner_id = int(inner["fold"])
            if inner_id != 0:
                raise ValueError(
                    f"Outer fold {outer_id} inner holdout must be inner_fold_0"
                )
            processor = self._fit_preprocessor(
                targets,
                target_mask,
                inner["train"],
                trait_names,
                trait_tasks,
            )
            processed = processor.apply(targets, target_mask)
            inner_path = outer_path / f"inner_fold_{inner_id}"
            inner_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                processed[inner["train"]].contiguous(),
                inner_path / "Y_train_processed.pt",
            )
            torch.save(
                processed[inner["valid"]].contiguous(),
                inner_path / "Y_valid_processed.pt",
            )
            torch.save(
                processed[outer_test].contiguous(),
                inner_path / "Y_test_processed.pt",
            )
            processor.save_json(inner_path / "preprocessing.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Aquila data with a 1:9 outer split and one 4:1 "
            "inner train/validation holdout per outer fold."
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
        dest="encoding",
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
        help="Optional explicit encoding type for kept records.",
    )
    parser.add_argument(
        "--id-prefix",
        dest="id_prefix",
        default=None,
        help="Keep VCF records whose ID starts with one of these prefixes.",
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
        help="Phenotype columns treated as binary classification tasks.",
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
            "Number of outer folds when --fold-mapping is omitted. "
            "Defaults to 10 (test:train = 1:9)."
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
            "Outer-holdout JSON written by aquila_cv_holdout.py. "
            "Outer and inner assignments in the file are used as stored."
        ),
    )
    parser.add_argument(
        "--fold-specific-gwas-fold",
        type=int,
        default=None,
        metavar="FOLD",
        help=(
            "Bind this prepared dataset to one zero-based outer fold. "
            "Requires --fold-mapping."
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
            "Write sample-subset VCF.GZ files for every outer train/test and "
            "the inner train/validation holdout."
        ),
    )
    args = parser.parse_args()
    if args.fold_mapping and args.outer_folds is not None:
        parser.error("--fold-mapping cannot be combined with --outer-folds")
    return args


def main() -> None:
    args = parse_args()
    config = PreparationConfig(
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
        inner_folds=1,
        seed=args.seed,
        fold_mapping=Path(args.fold_mapping) if args.fold_mapping else None,
        fold_specific_gwas_fold=args.fold_specific_gwas_fold,
        save_raw_genotype=args.save_raw_genotype,
        min_observed_per_fold=args.min_observed_per_fold,
        skew_threshold=args.skew_threshold,
        preprocessing_epsilon=args.preprocessing_epsilon,
        overwrite=args.overwrite,
    )
    metadata = OuterHoldoutDataPreparer(config).prepare()
    print(
        f"Prepared outer-holdout data for {metadata['n_samples']} samples and "
        f"{metadata['n_traits']} traits in {config.output_directory}"
    )
    print(
        "  protocol=outer_holdout outer_folds="
        f"{metadata['outer_folds']} inner_train:valid=4:1"
    )


if __name__ == "__main__":
    main()
