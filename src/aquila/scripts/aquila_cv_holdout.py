#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Create an outer-holdout sample mapping: 1:9 outer, then one 4:1 inner split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from aquila.data.cv import generate_outer_holdout_folds, validate_outer_fold_observations
from aquila.scripts.aquila_cv import load_phenotype_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an outer-holdout sample mapping. Each outer test is 1/10 "
            "of the cohort. Each outer training set is split once into inner "
            "train and inner valid at 4:1."
        )
    )
    parser.add_argument(
        "--phenotype",
        "--pheno",
        required=True,
        help="Phenotype table with sample IDs in the first column by default.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="holdout_cv_mapping.json",
        help="Output outer-holdout mapping in JSON format.",
    )
    parser.add_argument(
        "--sample-id-column",
        default=None,
        help="Sample ID column; defaults to the first column.",
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Traits to validate; defaults to all non-ID columns.",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=10,
        help="Number of outer folds. Default 10 (test:train = 1:9).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-sentinel", type=float, default=-999.0)
    parser.add_argument(
        "--min-observed",
        type=int,
        default=10,
        help="Minimum observed values required per trait in every outer test fold.",
    )
    return parser.parse_args()


def save_holdout_mapping(
    path: str | Path,
    sample_ids: List[str],
    folds: List[Dict[str, object]],
    seed: int,
) -> None:
    """Save one inner holdout per outer fold, addressed by sample ID."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if any(len(outer["inner"]) != 1 for outer in folds):
        raise ValueError("Holdout mapping requires exactly one inner split per outer fold")
    payload = {
        "schema": "aquila_outer_holdout",
        "version": 1,
        "seed": int(seed),
        "outer_folds": len(folds),
        "inner_folds": 1,
        "inner_train_valid_ratio": [4, 1],
        "sample_ids": list(sample_ids),
        "folds": [],
    }
    for outer in folds:
        inner = outer["inner"][0]
        payload["folds"].append(
            {
                "fold": int(outer["fold"]),
                "train": [
                    sample_ids[index]
                    for index in np.asarray(outer["train"], dtype=np.int64)
                ],
                "test": [
                    sample_ids[index]
                    for index in np.asarray(outer["test"], dtype=np.int64)
                ],
                "inner": [
                    {
                        "fold": int(inner["fold"]),
                        "train": [
                            sample_ids[index]
                            for index in np.asarray(inner["train"], dtype=np.int64)
                        ],
                        "valid": [
                            sample_ids[index]
                            for index in np.asarray(inner["valid"], dtype=np.int64)
                        ],
                    }
                ],
            }
        )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.min_observed < 10:
        raise ValueError("--min-observed must be at least 10")
    phenotype = load_phenotype_mask(
        args.phenotype,
        sample_id_column=args.sample_id_column,
        traits=args.traits,
        missing_sentinel=args.missing_sentinel,
    )
    folds = generate_outer_holdout_folds(
        n_samples=len(phenotype["sample_ids"]),
        outer_folds=args.outer_folds,
        seed=args.seed,
    )
    validate_outer_fold_observations(
        phenotype["mask"],
        folds,
        phenotype["trait_names"],
        min_observed=args.min_observed,
    )
    save_holdout_mapping(
        args.output,
        phenotype["sample_ids"],
        folds,
        args.seed,
    )
    print(
        f"Saved {args.outer_folds} outer folds and one 4:1 inner holdout for "
        f"{len(phenotype['sample_ids'])} samples to {args.output}"
    )
    print(
        f"Validated at least {args.min_observed} observed values per trait "
        "in every outer test fold"
    )


if __name__ == "__main__":
    main()
