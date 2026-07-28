#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Create reusable nested-CV sample mappings from a phenotype table."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from aquila.data.cv import generate_nested_folds, validate_outer_fold_observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create phenotype-aware nested CV sample mappings."
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
        default="nested_cv_mapping.json",
        help="Output nested-CV mapping in JSON format.",
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
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
        help="Number of inner folds within every outer training partition.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-sentinel", type=float, default=-999.0)
    parser.add_argument(
        "--min-observed",
        type=int,
        default=30,
        help="Minimum observed values required per trait in every fold.",
    )
    return parser.parse_args()


def load_phenotype_mask(
    phenotype_path: str | Path,
    sample_id_column: str | None = None,
    traits: List[str] | None = None,
    missing_sentinel: float = -999.0,
) -> Dict[str, Any]:
    """Load sample IDs and an observed-value mask without phenotype imputation."""
    frame = pd.read_csv(
        phenotype_path,
        sep=None,
        engine="python",
        dtype=str,
        keep_default_na=False,
    )
    if frame.empty or len(frame.columns) < 2:
        raise ValueError(
            "Phenotype file must contain samples and at least one trait"
        )
    id_column = sample_id_column or str(frame.columns[0])
    if id_column not in frame.columns:
        raise ValueError(f"Sample ID column not found: {id_column!r}")
    trait_names = list(
        traits
        or [str(column) for column in frame.columns if column != id_column]
    )
    missing_traits = [name for name in trait_names if name not in frame.columns]
    if missing_traits:
        raise ValueError(f"Phenotype traits not found: {missing_traits}")

    sample_ids = [str(value).strip() for value in frame[id_column]]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("Sample IDs must be nonempty")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique")

    mask = np.zeros((len(frame), len(trait_names)), dtype=bool)
    missing_text = {"", "nan", "na", "n/a", "null"}
    for trait_index, trait_name in enumerate(trait_names):
        for row_index, raw_value in enumerate(frame[trait_name]):
            text = str(raw_value).strip()
            if text.lower() in missing_text:
                continue
            try:
                value = float(text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid phenotype value {text!r} for sample "
                    f"{sample_ids[row_index]!r}, trait {trait_name!r}"
                ) from error
            if not math.isfinite(value):
                continue
            mask[row_index, trait_index] = value != missing_sentinel
    return {
        "sample_ids": sample_ids,
        "trait_names": trait_names,
        "mask": mask,
        "sample_id_column": id_column,
    }


def save_mapping(
    path: str | Path,
    sample_ids: List[str],
    folds: List[Dict[str, object]],
    seed: int,
) -> None:
    """Save complete nested folds using sample IDs instead of row positions."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "aquila_nested_cv",
        "version": 1,
        "seed": seed,
        "outer_folds": len(folds),
        "inner_folds": len(folds[0]["inner"]) if folds else 0,
        "sample_ids": sample_ids,
        "folds": [],
    }
    for outer in folds:
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
                    for inner in outer["inner"]
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
    folds = generate_nested_folds(
        n_samples=len(phenotype["sample_ids"]),
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    validate_outer_fold_observations(
        phenotype["mask"],
        folds,
        phenotype["trait_names"],
        min_observed=args.min_observed,
    )
    save_mapping(
        args.output,
        phenotype["sample_ids"],
        folds,
        args.seed,
    )
    print(
        f"Saved {args.outer_folds} outer folds and "
        f"{args.inner_folds} inner folds for "
        f"{len(phenotype['sample_ids'])} samples to {args.output}"
    )
    print(
        f"Validated at least {args.min_observed} observed values per trait "
        "in every fold"
    )


if __name__ == "__main__":
    main()
