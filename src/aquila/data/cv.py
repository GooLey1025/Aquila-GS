# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Deterministic nested cross-validation split utilities."""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.model_selection import KFold


NestedFold = Dict[str, object]


def generate_nested_folds(
    n_samples: int,
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = 42,
) -> List[NestedFold]:
    """Generate nested K-fold splits using absolute dataset indices."""
    if n_samples < 2:
        raise ValueError("At least two aligned samples are required")
    if not 2 <= outer_folds <= n_samples:
        raise ValueError(
            f"outer_folds must be between 2 and {n_samples}, got {outer_folds}"
        )

    absolute_indices = np.arange(n_samples, dtype=np.int64)
    outer_splitter = KFold(
        n_splits=outer_folds, shuffle=True, random_state=seed
    )
    folds: List[NestedFold] = []

    for outer_index, (train_positions, test_positions) in enumerate(
        outer_splitter.split(absolute_indices)
    ):
        outer_train = absolute_indices[train_positions]
        outer_test = absolute_indices[test_positions]
        if not 2 <= inner_folds <= len(outer_train):
            raise ValueError(
                "inner_folds must be between 2 and the smallest outer training "
                f"set size ({len(outer_train)}), got {inner_folds}"
            )

        inner_splitter = KFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + outer_index + 1,
        )
        inner_splits = []
        for inner_index, (inner_train_pos, valid_pos) in enumerate(
            inner_splitter.split(outer_train)
        ):
            inner_splits.append(
                {
                    "fold": inner_index,
                    "train": outer_train[inner_train_pos].copy(),
                    "valid": outer_train[valid_pos].copy(),
                }
            )

        folds.append(
            {
                "fold": outer_index,
                "train": outer_train.copy(),
                "test": outer_test.copy(),
                "inner": inner_splits,
            }
        )

    return folds


def generate_nested_folds_from_assignments(
    outer_test_folds: Sequence[int] | np.ndarray,
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = 42,
) -> List[NestedFold]:
    """Generate inner folds while preserving predefined outer test assignments."""
    assignments = np.asarray(outer_test_folds, dtype=np.int64)
    if assignments.ndim != 1 or assignments.size < 2:
        raise ValueError("outer_test_folds must be a one-dimensional sample mapping")
    invalid = assignments[(assignments < 0) | (assignments >= outer_folds)]
    if invalid.size:
        raise ValueError(
            f"Outer fold assignments must be in 0..{outer_folds - 1}"
        )

    absolute_indices = np.arange(assignments.size, dtype=np.int64)
    folds: List[NestedFold] = []
    for outer_index in range(outer_folds):
        outer_test = absolute_indices[assignments == outer_index]
        outer_train = absolute_indices[assignments != outer_index]
        if outer_test.size == 0:
            raise ValueError(f"Outer fold {outer_index} has no test samples")
        if not 2 <= inner_folds <= outer_train.size:
            raise ValueError(
                f"inner_folds must be between 2 and {outer_train.size}, "
                f"got {inner_folds}"
            )

        splitter = KFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + outer_index + 1,
        )
        inner_splits = []
        for inner_index, (train_pos, valid_pos) in enumerate(
            splitter.split(outer_train)
        ):
            inner_splits.append(
                {
                    "fold": inner_index,
                    "train": outer_train[train_pos].copy(),
                    "valid": outer_train[valid_pos].copy(),
                }
            )
        folds.append(
            {
                "fold": outer_index,
                "train": outer_train.copy(),
                "test": outer_test.copy(),
                "inner": inner_splits,
            }
        )
    return folds


def validate_outer_fold_observations(
    target_mask: np.ndarray,
    folds: Sequence[NestedFold],
    trait_names: Sequence[str],
    min_observed: int = 10,
) -> Dict[str, Dict[str, int]]:
    """Require enough observed values for every trait in every outer test fold."""
    mask = np.asarray(target_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != len(trait_names):
        raise ValueError("target_mask and trait_names have incompatible shapes")
    if min_observed < 1:
        raise ValueError("min_observed must be positive")

    counts: Dict[str, Dict[str, int]] = {}
    failures = []
    for outer in folds:
        fold_id = int(outer["fold"])
        test_indices = np.asarray(outer["test"], dtype=np.int64)
        fold_counts = mask[test_indices].sum(axis=0)
        counts[str(fold_id)] = {
            str(name): int(fold_counts[index])
            for index, name in enumerate(trait_names)
        }
        for index, name in enumerate(trait_names):
            observed = int(fold_counts[index])
            if observed < min_observed:
                failures.append(
                    f"trait {name!r}, fold {fold_id}: "
                    f"{observed} observed values (minimum {min_observed})"
                )
    if failures:
        raise ValueError(
            "Phenotype coverage requirement failed:\n  - "
            + "\n  - ".join(failures)
        )
    return counts


def save_nested_folds(folds: Iterable[NestedFold], cv_directory: str | Path) -> None:
    """Persist nested folds in a directory-per-fold layout."""
    cv_path = Path(cv_directory)
    cv_path.mkdir(parents=True, exist_ok=True)

    for outer in folds:
        outer_path = cv_path / f"outer_fold_{outer['fold']}"
        outer_path.mkdir(parents=True, exist_ok=True)
        np.save(outer_path / "train_idx.npy", outer["train"])
        np.save(outer_path / "test_idx.npy", outer["test"])

        for inner in outer["inner"]:
            inner_path = outer_path / f"inner_fold_{inner['fold']}"
            inner_path.mkdir(parents=True, exist_ok=True)
            np.save(inner_path / "train_idx.npy", inner["train"])
            np.save(inner_path / "valid_idx.npy", inner["valid"])


def load_fold_indices(
    prepared_directory: str | Path,
    outer_fold: int,
    inner_fold: int | None = None,
) -> Dict[str, np.ndarray]:
    """Load one outer or inner split from a prepared-data directory."""
    outer_path = Path(prepared_directory) / "cv" / f"outer_fold_{outer_fold}"
    if inner_fold is None:
        names = ("train_idx", "test_idx")
        split_path = outer_path
    else:
        names = ("train_idx", "valid_idx")
        split_path = outer_path / f"inner_fold_{inner_fold}"

    missing = [name for name in names if not (split_path / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files in {split_path}: {', '.join(missing)}"
        )
    return {
        name.removesuffix("_idx"): np.load(
            split_path / f"{name}.npy", allow_pickle=False
        )
        for name in names
    }


def parse_fold_selector(
    selector: str | int | Sequence[int],
    fold_count: int,
) -> List[int]:
    """Validate one or more unique, zero-based integer fold identifiers."""
    if fold_count < 1:
        raise ValueError("fold_count must be positive")

    if isinstance(selector, (bool, np.bool_)):
        raise ValueError("Fold identifiers must be integers")
    if isinstance(selector, (int, np.integer)):
        selected = [int(selector)]
    elif isinstance(selector, str):
        tokens = [token.strip() for token in selector.split(",")]
        if not tokens or any(not token for token in tokens):
            raise ValueError("At least one fold identifier is required")
        try:
            selected = [int(token) for token in tokens]
        except ValueError as error:
            raise ValueError("Fold identifiers must be integers") from error
    else:
        selected = list(selector)
        if not selected:
            raise ValueError("At least one fold identifier is required")
        if any(isinstance(fold, bool) or not isinstance(fold, (int, np.integer))
               for fold in selected):
            raise ValueError("Fold identifiers must be integers")
        selected = [int(fold) for fold in selected]

    if len(set(selected)) != len(selected):
        raise ValueError("Fold identifiers must not contain duplicates")
    invalid = sorted(fold for fold in selected if fold < 0 or fold >= fold_count)
    if invalid:
        raise ValueError(
            f"Fold numbers must be in 0..{fold_count - 1}; got {invalid}"
        )
    return selected
