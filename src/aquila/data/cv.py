# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Deterministic nested cross-validation split utilities."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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


def split_train_valid(
    indices: Sequence[int] | np.ndarray,
    valid_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one index vector once into deterministic train and validation sets.

    ``valid_fraction`` 0.2 is the 4:1 train:validation holdout. The validation
    count is the rounded fraction, with at least one sample kept in each side.
    """
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("A train/validation split requires at least two samples")
    fraction = float(valid_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"valid_fraction must be between 0 and 1, got {valid_fraction}"
        )
    n_valid = int(round(values.size * fraction))
    n_valid = min(max(n_valid, 1), int(values.size) - 1)
    order = np.random.default_rng(int(seed)).permutation(values.size)
    valid = np.sort(values[order[:n_valid]])
    train = np.sort(values[order[n_valid:]])
    return train, valid


def _benchmark_fold(
    fold_id: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    valid_fraction: float,
    seed: int,
) -> NestedFold:
    train, valid = split_train_valid(
        outer_train,
        valid_fraction=valid_fraction,
        seed=int(seed) + int(fold_id) + 1,
    )
    return {
        "fold": int(fold_id),
        "outer_train": np.asarray(outer_train, dtype=np.int64).copy(),
        "train": train,
        "valid": valid,
        "test": np.asarray(outer_test, dtype=np.int64).copy(),
    }


def generate_benchmark_folds(
    n_samples: int,
    outer_folds: int = 10,
    valid_fraction: float = 0.2,
    seed: int = 42,
) -> List[NestedFold]:
    """Generate outer K-fold tests and one 4:1 split inside each outer train set."""
    if n_samples < 2:
        raise ValueError("At least two aligned samples are required")
    if not 2 <= int(outer_folds) <= int(n_samples):
        raise ValueError(
            f"outer_folds must be between 2 and {n_samples}, got {outer_folds}"
        )
    absolute_indices = np.arange(n_samples, dtype=np.int64)
    outer_splitter = KFold(
        n_splits=int(outer_folds),
        shuffle=True,
        random_state=int(seed),
    )
    folds: List[NestedFold] = []
    for outer_index, (train_positions, test_positions) in enumerate(
        outer_splitter.split(absolute_indices)
    ):
        folds.append(
            _benchmark_fold(
                outer_index,
                absolute_indices[train_positions],
                absolute_indices[test_positions],
                valid_fraction,
                seed,
            )
        )
    return folds


def generate_benchmark_folds_from_assignments(
    outer_test_folds: Sequence[int] | np.ndarray,
    outer_folds: int = 10,
    valid_fraction: float = 0.2,
    seed: int = 42,
) -> List[NestedFold]:
    """Keep predefined outer tests and draw one train/validation split in each."""
    assignments = np.asarray(outer_test_folds, dtype=np.int64)
    if assignments.ndim != 1 or assignments.size < 2:
        raise ValueError("outer_test_folds must be a one-dimensional sample mapping")
    invalid = assignments[(assignments < 0) | (assignments >= int(outer_folds))]
    if invalid.size:
        raise ValueError(
            f"Outer fold assignments must be in 0..{int(outer_folds) - 1}"
        )
    absolute_indices = np.arange(assignments.size, dtype=np.int64)
    folds: List[NestedFold] = []
    for outer_index in range(int(outer_folds)):
        outer_test = absolute_indices[assignments == outer_index]
        outer_train = absolute_indices[assignments != outer_index]
        if outer_test.size == 0:
            raise ValueError(f"Outer fold {outer_index} has no test samples")
        if outer_train.size < 2:
            raise ValueError(
                f"Outer fold {outer_index} needs at least two training samples"
            )
        folds.append(
            _benchmark_fold(
                outer_index,
                outer_train,
                outer_test,
                valid_fraction,
                seed,
            )
        )
    return folds


def validate_benchmark_folds(
    folds: Sequence[NestedFold],
    sample_count: int,
) -> None:
    """Require 10-style outer tests and one disjoint train/validation holdout."""
    if len(folds) < 2:
        raise ValueError("At least two benchmark outer folds are required")
    all_indices = set(range(int(sample_count)))
    observed_test: set[int] = set()
    expected_ids = set(range(len(folds)))
    if {int(fold["fold"]) for fold in folds} != expected_ids:
        raise ValueError("Benchmark outer fold identifiers must be zero-based")
    for outer in folds:
        fold_id = int(outer["fold"])
        outer_train = set(np.asarray(outer["outer_train"], dtype=np.int64).tolist())
        train = set(np.asarray(outer["train"], dtype=np.int64).tolist())
        valid = set(np.asarray(outer["valid"], dtype=np.int64).tolist())
        test = set(np.asarray(outer["test"], dtype=np.int64).tolist())
        if not train or not valid or not test:
            raise ValueError(f"Outer fold {fold_id} has an empty split")
        if outer_train & test or outer_train | test != all_indices:
            raise ValueError(
                f"Outer fold {fold_id} is not a complete disjoint partition"
            )
        if train & valid or train | valid != outer_train:
            raise ValueError(
                f"Outer fold {fold_id} train/validation split does not "
                "partition its outer training samples"
            )
        if observed_test & test:
            raise ValueError("Outer test folds overlap")
        observed_test.update(test)
    if observed_test != all_indices:
        raise ValueError("Outer test folds do not cover every sample")


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


def is_single_holdout_fold(prepared_directory: str | Path, outer_fold: int) -> bool:
    """Return whether one outer fold stores a single 4:1 train/validation holdout."""
    outer_path = Path(prepared_directory) / "cv" / f"outer_fold_{int(outer_fold)}"
    return (outer_path / "valid_idx.npy").is_file() and (
        outer_path / "outer_train_idx.npy"
    ).is_file()


def resolve_prepared_split_paths(
    prepared_directory: str | Path,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
) -> tuple[Path, Path, Path]:
    """Return processed-target directory, index file, and raw VCF for one role.

    Benchmark holdout data keeps the HPO train split, validation split, and
    outer test at the outer-fold root. The train VCF is the 4:1 training
    subset. Nested-CV data keeps its historical inner-fold and ``final`` layout.
    """
    root = Path(prepared_directory)
    outer_path = root / "cv" / f"outer_fold_{int(outer_fold)}"
    raw_path = root / "raw_genotype" / f"outer_fold_{int(outer_fold)}"
    if is_single_holdout_fold(root, outer_fold):
        if inner_fold not in (None, 0):
            raise ValueError(
                "Benchmark holdout data has one validation split; "
                f"inner fold {inner_fold} is not available"
            )
        if inner_fold is None and role not in {"train", "test"}:
            raise ValueError(f"Invalid outer-test role: {role}")
        if inner_fold is not None and role not in {"train", "valid"}:
            raise ValueError(f"Invalid holdout role: {role}")
        vcf_path = (
            raw_path / "test.vcf.gz"
            if role == "test"
            else raw_path / "inner_fold_0" / f"{role}.vcf.gz"
        )
        return outer_path, outer_path / f"{role}_idx.npy", vcf_path
    if inner_fold is None:
        if role not in {"train", "test"}:
            raise ValueError(f"Invalid final split role: {role}")
        return (
            outer_path / "final",
            outer_path / f"{role}_idx.npy",
            raw_path / f"{role}.vcf.gz",
        )
    if role not in {"train", "valid"}:
        raise ValueError(f"Invalid inner split role: {role}")
    split_path = outer_path / f"inner_fold_{int(inner_fold)}"
    return (
        split_path,
        split_path / f"{role}_idx.npy",
        raw_path / f"inner_fold_{int(inner_fold)}" / f"{role}.vcf.gz",
    )


def load_fold_indices(
    prepared_directory: str | Path,
    outer_fold: int,
    inner_fold: int | None = None,
) -> Dict[str, np.ndarray]:
    """Load one outer or inner split from a prepared-data directory."""
    roles = ("train", "test") if inner_fold is None else ("train", "valid")
    loaded: Dict[str, np.ndarray] = {}
    for role in roles:
        _processed, index_path, _vcf = resolve_prepared_split_paths(
            prepared_directory,
            outer_fold,
            inner_fold,
            role,
        )
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing split file: {index_path}")
        loaded[role] = np.load(index_path, allow_pickle=False)
    return loaded


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


def resolve_outer_folds(
    requested_folds: Sequence[int] | None,
    metadata: Mapping[str, Any],
    *,
    option_name: str = "--outer-folds",
) -> List[int]:
    """Resolve outer folds while enforcing prepared-data feature binding."""
    fold_count = int(metadata["outer_folds"])
    bindings = (
        ("fold_specific_features", "split-local markers"),
        ("fold_specific_gwas", "fold-specific GWAS markers"),
    )
    binding_name = ""
    selection_label = ""
    binding: Any = None
    for name, label in bindings:
        if metadata.get(name) is not None:
            binding_name = name
            selection_label = label
            binding = metadata[name]
            break
    if binding is None:
        return (
            list(range(fold_count))
            if requested_folds is None
            else parse_fold_selector(requested_folds, fold_count)
        )
    if not isinstance(binding, Mapping):
        raise ValueError(f"metadata {binding_name} must be an object")
    if binding.get("enabled") is not True:
        raise ValueError(f"metadata {binding_name}.enabled must be true")
    bound_fold = binding.get("outer_fold")
    if isinstance(bound_fold, bool) or not isinstance(bound_fold, int):
        raise ValueError(
            f"metadata {binding_name}.outer_fold must be an integer"
        )
    if bound_fold < 0 or bound_fold >= fold_count:
        raise ValueError(
            f"metadata {binding_name}.outer_fold is outside the prepared "
            f"fold range: {bound_fold}"
        )
    if requested_folds is None:
        return [bound_fold]
    selected = parse_fold_selector(requested_folds, fold_count)
    if selected != [bound_fold]:
        raise ValueError(
            f"This prepared dataset contains {selection_label} for outer fold "
            f"{bound_fold}; requested folds were {selected}. Use {option_name} "
            f"{bound_fold} or omit {option_name}."
        )
    return selected
