# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Shared leakage-safe utilities for prepared nested-CV benchmarks."""

from __future__ import annotations

import csv
import gzip
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from aquila.data import PerTraitPreprocessor, load_fold_indices, load_prepared_data
from aquila.training.evaluator import RegressionEvaluator


VariantKey = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class FoldPaths:
    """Resolved files for one prepared inner or outer-final split."""

    prepared_directory: Path
    outer_fold: int
    inner_fold: int | None
    train_indices: Path
    held_out_indices: Path
    train_vcf: Path
    held_out_vcf: Path
    train_targets: Path
    held_out_targets: Path
    preprocessing: Path

    @property
    def held_out_name(self) -> str:
        return "valid" if self.inner_fold is not None else "test"

    def require_complete(self) -> "FoldPaths":
        """Fail with one complete report when any required artifact is absent."""
        missing = [
            f"{name}: {path}"
            for name, path in self.files().items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Prepared benchmark fold is incomplete:\n  - " + "\n  - ".join(missing)
            )
        return self

    def files(self) -> dict[str, Path]:
        return {
            "train_indices": self.train_indices,
            f"{self.held_out_name}_indices": self.held_out_indices,
            "train_vcf": self.train_vcf,
            f"{self.held_out_name}_vcf": self.held_out_vcf,
            "train_targets": self.train_targets,
            f"{self.held_out_name}_targets": self.held_out_targets,
            "preprocessing": self.preprocessing,
        }


@dataclass(frozen=True)
class DosageVCF:
    """Sample-major alternate-allele dosages and ordered VCF schema."""

    dosages: np.ndarray
    sample_ids: tuple[str, ...]
    variants: tuple[VariantKey, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.dosages)
        if values.ndim != 2:
            raise ValueError("VCF dosages must be a two-dimensional array")
        if values.shape != (len(self.sample_ids), len(self.variants)):
            raise ValueError("VCF dosage dimensions do not match samples and variants")

    @property
    def genotypes(self) -> np.ndarray:
        """Compatibility alias used by benchmark model adapters."""
        return self.dosages

    def align_samples(
        self,
        expected_sample_ids: Sequence[str],
        *,
        require_exact: bool = True,
    ) -> "DosageVCF":
        expected = _unique_strings(expected_sample_ids, "expected sample IDs")
        observed = _unique_strings(self.sample_ids, "VCF sample IDs")
        observed_index = {sample_id: index for index, sample_id in enumerate(observed)}
        missing = [sample_id for sample_id in expected if sample_id not in observed_index]
        extras = [sample_id for sample_id in observed if sample_id not in set(expected)]
        if missing or (require_exact and extras):
            details = []
            if missing:
                details.append(f"missing={missing[:10]}")
            if require_exact and extras:
                details.append(f"unexpected={extras[:10]}")
            raise ValueError("VCF sample schema mismatch: " + ", ".join(details))
        order = np.asarray([observed_index[sample_id] for sample_id in expected])
        return DosageVCF(
            np.asarray(self.dosages)[order].copy(),
            expected,
            self.variants,
        )

    def validate_variants(self, expected: Sequence[VariantKey]) -> None:
        validate_ordered_variants(expected, self.variants)


@dataclass(frozen=True)
class SingleTraitSplit:
    """Observed rows for one trait in one prepared partition."""

    genotypes: np.ndarray
    processed_targets: np.ndarray
    raw_targets: np.ndarray
    sample_ids: tuple[str, ...]
    discarded_sample_ids: tuple[str, ...]
    absolute_indices: np.ndarray
    variants: tuple[VariantKey, ...]

    @property
    def targets(self) -> np.ndarray:
        """Compatibility alias for model adapters."""
        return self.processed_targets

    @property
    def missing_mask(self) -> np.ndarray:
        """Return the genotype-missing mask without conflating dosage zero."""
        return ~np.isfinite(self.genotypes)


@dataclass(frozen=True)
class TwoScaleEvaluation:
    """Regression metrics on processed and original phenotype scales."""

    processed: dict[str, Any]
    original: dict[str, Any]
    predictions_processed: np.ndarray
    predictions_original: np.ndarray
    targets_processed: np.ndarray
    targets_original: np.ndarray


class GenotypePreprocessor:
    """Train-only column mean imputation followed by optional scaling."""

    def __init__(self, scaler: str = "none", epsilon: float = 1e-12) -> None:
        normalized = str(scaler).lower()
        if normalized not in {"none", "standard", "minmax"}:
            raise ValueError("scaler must be one of: none, standard, minmax")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.scaler = normalized
        self.epsilon = float(epsilon)
        self.mean_: np.ndarray | None = None
        self.offset_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    def fit(self, train_genotypes: Any) -> "GenotypePreprocessor":
        values = _as_float_matrix(train_genotypes, "train_genotypes")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("train_genotypes must be nonempty")
        observed = np.isfinite(values)
        counts = observed.sum(axis=0)
        if np.any(counts == 0):
            columns = np.flatnonzero(counts == 0).tolist()
            raise ValueError(f"Training variants contain only missing values: {columns}")
        means = np.nansum(values, axis=0) / counts
        imputed = np.where(observed, values, means)
        if self.scaler == "standard":
            offset = imputed.mean(axis=0)
            scale = imputed.std(axis=0, ddof=0)
        elif self.scaler == "minmax":
            offset = imputed.min(axis=0)
            scale = imputed.max(axis=0) - offset
        else:
            offset = np.zeros(values.shape[1], dtype=np.float64)
            scale = np.ones(values.shape[1], dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        scale[~np.isfinite(scale) | (np.abs(scale) <= self.epsilon)] = 1.0
        self.mean_ = np.asarray(means, dtype=np.float64)
        self.offset_ = np.asarray(offset, dtype=np.float64)
        self.scale_ = scale
        return self

    def transform(self, genotypes: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Genotype preprocessor has not been fitted")
        values = _as_float_matrix(genotypes, "genotypes")
        assert self.mean_ is not None
        assert self.offset_ is not None
        assert self.scale_ is not None
        if values.shape[1] != self.mean_.size:
            raise ValueError("Genotype columns do not match fitted preprocessor")
        imputed = np.where(np.isfinite(values), values, self.mean_)
        return ((imputed - self.offset_) / self.scale_).astype(np.float32)

    def fit_transform(self, train_genotypes: Any) -> np.ndarray:
        return self.fit(train_genotypes).transform(train_genotypes)

    def to_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Genotype preprocessor has not been fitted")
        assert self.mean_ is not None
        assert self.offset_ is not None
        assert self.scale_ is not None
        return {
            "scaler": self.scaler,
            "epsilon": self.epsilon,
            "imputation_mean": self.mean_.tolist(),
            "offset": self.offset_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GenotypePreprocessor":
        instance = cls(str(values["scaler"]), float(values.get("epsilon", 1e-12)))
        instance.mean_ = np.asarray(values["imputation_mean"], dtype=np.float64)
        instance.offset_ = np.asarray(values["offset"], dtype=np.float64)
        instance.scale_ = np.asarray(values["scale"], dtype=np.float64)
        if not (
            instance.mean_.ndim
            == instance.offset_.ndim
            == instance.scale_.ndim
            == 1
            and instance.mean_.shape == instance.offset_.shape == instance.scale_.shape
        ):
            raise ValueError("Serialized genotype preprocessor arrays are inconsistent")
        return instance


class PreparedBenchmark:
    """Validated access to one Aquila prepared nested-CV directory."""

    def __init__(self, directory: str | Path, *, validate_all: bool = True) -> None:
        self.directory = Path(directory).resolve()
        self.prepared = load_prepared_data(self.directory)
        self.metadata = self.prepared.metadata
        self.sample_ids = tuple(str(value) for value in self.metadata["sample_ids"])
        self.trait_names = tuple(str(value) for value in self.metadata["trait_names"])
        if validate_all:
            self.validate_complete()

    @property
    def outer_fold_count(self) -> int:
        return int(self.metadata["outer_folds"])

    @property
    def inner_fold_count(self) -> int:
        return int(self.metadata["inner_folds"])

    @property
    def outer_folds(self) -> int:
        """Compatibility alias for the configured outer-fold count."""
        return self.outer_fold_count

    @property
    def inner_folds(self) -> int:
        """Compatibility alias for the configured inner-fold count."""
        return self.inner_fold_count

    @property
    def regression_traits(self) -> tuple[str, ...]:
        """Regression trait names in prepared target-column order."""
        configured = self.metadata.get("regression_tasks")
        if isinstance(configured, list):
            return tuple(str(value) for value in configured)
        tasks = self.metadata.get("trait_tasks")
        if isinstance(tasks, list) and len(tasks) == len(self.trait_names):
            return tuple(
                name
                for name, task in zip(self.trait_names, tasks)
                if task == "regression"
            )
        return self.trait_names

    def resolve_fold_paths(
        self,
        outer_fold: int,
        inner_fold: int | None = None,
        *,
        require_complete: bool = True,
    ) -> FoldPaths:
        self._validate_fold_number(outer_fold, self.outer_fold_count, "outer")
        outer_cv = self.directory / "cv" / f"outer_fold_{outer_fold}"
        outer_vcf = self.directory / "raw_genotype" / f"outer_fold_{outer_fold}"
        if inner_fold is None:
            processed = outer_cv / "final"
            paths = FoldPaths(
                self.directory,
                outer_fold,
                None,
                outer_cv / "train_idx.npy",
                outer_cv / "test_idx.npy",
                _resolve_vcf_path(outer_vcf, "train"),
                _resolve_vcf_path(outer_vcf, "test"),
                processed / "Y_train_processed.pt",
                processed / "Y_test_processed.pt",
                processed / "preprocessing.json",
            )
        else:
            self._validate_fold_number(inner_fold, self.inner_fold_count, "inner")
            inner_cv = outer_cv / f"inner_fold_{inner_fold}"
            inner_vcf = outer_vcf / f"inner_fold_{inner_fold}"
            paths = FoldPaths(
                self.directory,
                outer_fold,
                inner_fold,
                inner_cv / "train_idx.npy",
                inner_cv / "valid_idx.npy",
                _resolve_vcf_path(inner_vcf, "train"),
                _resolve_vcf_path(inner_vcf, "valid"),
                inner_cv / "Y_train_processed.pt",
                inner_cv / "Y_valid_processed.pt",
                inner_cv / "preprocessing.json",
            )
        return paths.require_complete() if require_complete else paths

    def validate_complete(self) -> None:
        if not bool(self.metadata.get("raw_genotype_saved")):
            raise ValueError(
                "Prepared benchmark requires raw fold VCFs; regenerate data with "
                "--save-raw-genotype"
            )
        failures = []
        for outer_fold in range(self.outer_fold_count):
            selections = [None, *range(self.inner_fold_count)]
            for inner_fold in selections:
                paths = self.resolve_fold_paths(
                    outer_fold, inner_fold, require_complete=False
                )
                failures.extend(
                    str(path) for path in paths.files().values() if not path.is_file()
                )
        if failures:
            raise FileNotFoundError(
                "Prepared benchmark is incomplete; missing artifacts:\n  - "
                + "\n  - ".join(failures)
            )

    def load_fold(
        self,
        outer_fold: int,
        inner_fold: int | None = None,
    ) -> tuple[SingleTraitSplit, SingleTraitSplit]:
        raise RuntimeError("Use load_single_trait_fold() with an explicit trait")

    def load_single_trait_fold(
        self,
        trait: str | int,
        outer_fold: int,
        inner_fold: int | None = None,
    ) -> tuple[SingleTraitSplit, SingleTraitSplit]:
        paths = self.resolve_fold_paths(outer_fold, inner_fold)
        trait_index = self.trait_index(trait)
        split = load_fold_indices(self.directory, outer_fold, inner_fold)
        held_out_name = paths.held_out_name
        train_indices = np.asarray(split["train"], dtype=np.int64)
        held_out_indices = np.asarray(split[held_out_name], dtype=np.int64)
        train_vcf = load_vcf_dosage(paths.train_vcf).align_samples(
            self.sample_ids_for(train_indices)
        )
        held_out_vcf = load_vcf_dosage(paths.held_out_vcf).align_samples(
            self.sample_ids_for(held_out_indices)
        )
        validate_ordered_variants(train_vcf.variants, held_out_vcf.variants)
        train_processed = load_processed_targets(paths.train_targets, len(train_indices))
        held_out_processed = load_processed_targets(
            paths.held_out_targets, len(held_out_indices)
        )
        train = self._single_trait_partition(
            train_vcf, train_indices, train_processed, trait_index
        )
        held_out = self._single_trait_partition(
            held_out_vcf, held_out_indices, held_out_processed, trait_index
        )
        return train, held_out

    def load_preprocessor(
        self,
        outer_fold: int,
        inner_fold: int | None = None,
    ) -> PerTraitPreprocessor:
        paths = self.resolve_fold_paths(outer_fold, inner_fold)
        processor = PerTraitPreprocessor.load_json(paths.preprocessing)
        names = tuple(item.name for item in processor.traits)
        if names != self.trait_names:
            raise ValueError(
                "Fold preprocessor trait order does not match prepared metadata"
            )
        return processor

    def inverse_trait(
        self,
        values: Any,
        trait: str | int,
        outer_fold: int,
        inner_fold: int | None = None,
    ) -> np.ndarray:
        trait_index = self.trait_index(trait)
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        full = np.zeros((array.size, len(self.trait_names)), dtype=np.float64)
        mask = np.zeros_like(full, dtype=bool)
        full[:, trait_index] = array
        mask[:, trait_index] = True
        restored = self.load_preprocessor(outer_fold, inner_fold).inverse(full, mask)
        return np.asarray(restored)[:, trait_index]

    def trait_index(self, trait: str | int) -> int:
        if isinstance(trait, bool):
            raise ValueError("trait must be a name or integer column index")
        if isinstance(trait, (int, np.integer)):
            index = int(trait)
            if not 0 <= index < len(self.trait_names):
                raise IndexError(f"Trait index is out of range: {index}")
            return index
        try:
            return self.trait_names.index(str(trait))
        except ValueError as error:
            raise ValueError(f"Unknown trait {trait!r}") from error

    def sample_ids_for(self, indices: Sequence[int]) -> tuple[str, ...]:
        return tuple(self.sample_ids[int(index)] for index in indices)

    def _single_trait_partition(
        self,
        vcf: DosageVCF,
        absolute_indices: np.ndarray,
        processed: np.ndarray,
        trait_index: int,
    ) -> SingleTraitSplit:
        raw = self.prepared.targets.detach().cpu().numpy()[absolute_indices, trait_index]
        mask = self.prepared.target_mask.detach().cpu().numpy()[
            absolute_indices, trait_index
        ].astype(bool)
        observed = mask & np.isfinite(raw) & np.isfinite(processed[:, trait_index])
        sample_array = np.asarray(vcf.sample_ids, dtype=object)
        return SingleTraitSplit(
            np.asarray(vcf.dosages)[observed].copy(),
            np.asarray(processed[observed, trait_index], dtype=np.float32),
            np.asarray(raw[observed], dtype=np.float32),
            tuple(sample_array[observed].tolist()),
            tuple(sample_array[~observed].tolist()),
            absolute_indices[observed].copy(),
            vcf.variants,
        )

    @staticmethod
    def _validate_fold_number(value: int, count: int, kind: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{kind}_fold must be an integer")
        if not 0 <= int(value) < count:
            raise ValueError(f"{kind}_fold must be in 0..{count - 1}")


def load_nested_cv_context(data_directory: str | Path) -> PreparedBenchmark:
    """Open and fully validate a prepared nested-CV benchmark directory."""
    return PreparedBenchmark(data_directory, validate_all=True)


def load_trait_split(
    context: PreparedBenchmark,
    trait: str | int,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
) -> SingleTraitSplit:
    """Load one observed-only trait split through the shared context."""
    if not isinstance(context, PreparedBenchmark):
        raise TypeError("context must be a PreparedBenchmark")
    expected_roles = {"train", "valid"} if inner_fold is not None else {
        "train",
        "test",
    }
    if role not in expected_roles:
        raise ValueError(
            f"role must be one of {sorted(expected_roles)}, received {role!r}"
        )
    train, held_out = context.load_single_trait_fold(
        trait, outer_fold, inner_fold
    )
    return train if role == "train" else held_out


def evaluate_trait_predictions(
    context: PreparedBenchmark,
    trait: str | int,
    outer_fold: int,
    targets_processed: Any,
    predictions_processed: Any,
) -> dict[str, Any]:
    """Evaluate outer-test predictions on processed and original scales."""
    if not isinstance(context, PreparedBenchmark):
        raise TypeError("context must be a PreparedBenchmark")
    trait_name = context.trait_names[context.trait_index(trait)]
    targets = np.asarray(targets_processed, dtype=np.float64).reshape(-1)
    predictions = np.asarray(
        predictions_processed, dtype=np.float64
    ).reshape(-1)
    targets_original = context.inverse_trait(
        targets, trait, outer_fold, inner_fold=None
    )
    predictions_original = context.inverse_trait(
        predictions, trait, outer_fold, inner_fold=None
    )
    result = evaluate_two_scales(
        predictions,
        targets,
        predictions_original,
        targets_original,
        trait_name=trait_name,
    )
    return {
        "normalized": result.processed,
        "original": result.original,
    }


def load_vcf_dosage(path: str | Path) -> DosageVCF:
    """Load diploid GT fields as 0/1/2 ALT dosage, preserving missing as NaN."""
    vcf_path = Path(path)
    if not vcf_path.is_file():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    opener = gzip.open if vcf_path.suffix.lower() == ".gz" else open
    sample_ids: tuple[str, ...] | None = None
    variants: list[VariantKey] = []
    variant_dosages: list[list[float]] = []
    with opener(vcf_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##"):
                continue
            columns = line.rstrip("\r\n").split("\t")
            if line.startswith("#CHROM"):
                if len(columns) < 10:
                    raise ValueError("VCF must contain at least one sample")
                sample_ids = _unique_strings(columns[9:], "VCF sample IDs")
                continue
            if line.startswith("#"):
                continue
            if sample_ids is None:
                raise ValueError("VCF data appeared before the #CHROM header")
            if len(columns) != 9 + len(sample_ids):
                raise ValueError(
                    f"VCF record {line_number} has an inconsistent sample count"
                )
            if "," in columns[4]:
                raise ValueError(
                    f"VCF record {line_number} is multiallelic; dosage requires "
                    "biallelic variants"
                )
            format_fields = columns[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record {line_number} has no GT FORMAT field")
            gt_index = format_fields.index("GT")
            dosages = [
                _parse_diploid_dosage(value, gt_index, line_number)
                for value in columns[9:]
            ]
            variants.append(
                (columns[0], columns[1], columns[2], columns[3], columns[4])
            )
            variant_dosages.append(dosages)
    if sample_ids is None:
        raise ValueError("VCF does not contain a #CHROM header")
    if not variants:
        raise ValueError("VCF does not contain any variant records")
    matrix = np.asarray(variant_dosages, dtype=np.float32).T
    return DosageVCF(matrix, sample_ids, tuple(variants))


def validate_ordered_variants(
    expected: Sequence[VariantKey],
    observed: Sequence[VariantKey],
) -> None:
    """Require identical variant count, identity, alleles, and order."""
    expected_tuple = tuple(tuple(map(str, value)) for value in expected)
    observed_tuple = tuple(tuple(map(str, value)) for value in observed)
    if len(expected_tuple) != len(observed_tuple):
        raise ValueError(
            "Variant schema length mismatch: "
            f"expected {len(expected_tuple)}, observed {len(observed_tuple)}"
        )
    for index, (left, right) in enumerate(zip(expected_tuple, observed_tuple)):
        if left != right:
            raise ValueError(
                f"Ordered variant schema mismatch at column {index}: "
                f"expected {left}, observed {right}"
            )


def load_processed_targets(path: str | Path, expected_rows: int) -> np.ndarray:
    """Load and validate one fold-local processed phenotype tensor."""
    values = _torch_load(Path(path))
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError(f"Processed targets must be a 2D tensor: {path}")
    if values.shape[0] != expected_rows:
        raise ValueError(
            f"Processed target row count mismatch in {path}: "
            f"expected {expected_rows}, observed {values.shape[0]}"
        )
    return values.detach().cpu().numpy()


def evaluate_two_scales(
    predictions_processed: Any,
    targets_processed: Any,
    predictions_original: Any,
    targets_original: Any,
    *,
    trait_name: str,
) -> TwoScaleEvaluation:
    """Evaluate the same observed predictions on processed and original scales."""
    processed_prediction = np.asarray(predictions_processed, dtype=np.float64).reshape(-1)
    processed_target = np.asarray(targets_processed, dtype=np.float64).reshape(-1)
    original_prediction = np.asarray(predictions_original, dtype=np.float64).reshape(-1)
    original_target = np.asarray(targets_original, dtype=np.float64).reshape(-1)
    sizes = {
        processed_prediction.size,
        processed_target.size,
        original_prediction.size,
        original_target.size,
    }
    if len(sizes) != 1:
        raise ValueError("Predictions and targets on both scales must have equal length")
    evaluator = RegressionEvaluator([trait_name])
    processed = evaluator.evaluate(processed_prediction, processed_target)
    original = evaluator.evaluate(original_prediction, original_target)
    return TwoScaleEvaluation(
        sanitize_json(processed.metrics),
        sanitize_json(original.metrics),
        processed_prediction,
        original_prediction,
        processed_target,
        original_target,
    )


def sanitize_json(value: Any) -> Any:
    """Recursively convert values to strict finite JSON-compatible objects."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, torch.Tensor):
        return sanitize_json(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, int)) or value is None:
        return value
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, value: Any) -> None:
    """Write strict JSON after applying the shared finite-value policy."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def serialize_candidate(candidate: Any) -> dict[str, Any]:
    """Serialize Aquila or model-specific candidate results consistently."""
    inner_results = []
    for result in getattr(candidate, "inner_results", ()):
        inner_results.append(
            {
                "inner_fold": int(result.inner_fold),
                "metric": result.metric,
                "best_epoch": int(result.best_epoch),
                "metrics": result.metrics,
            }
        )
    payload = {
        "candidate_id": int(candidate.candidate_id),
        "parameters": dict(candidate.parameters),
        "objective": candidate.objective,
        "best_epochs": list(getattr(candidate, "best_epochs", ())),
        "final_epoch": getattr(candidate, "final_epoch", None),
        "inner_results": inner_results,
    }
    return sanitize_json(payload)


def serialize_hpo(result: Any) -> dict[str, Any]:
    """Serialize a complete HPO result with its selected candidate."""
    return {
        "method": str(result.method),
        "direction": str(result.direction),
        "best_candidate_id": int(result.best.candidate_id),
        "best": serialize_candidate(result.best),
        "candidates": [serialize_candidate(item) for item in result.candidates],
    }


def build_sample_audit(
    train: SingleTraitSplit,
    held_out: SingleTraitSplit,
    *,
    held_out_name: str,
) -> dict[str, Any]:
    """Describe retained and discarded single-trait samples."""
    return {
        "train": {
            "observed_count": len(train.sample_ids),
            "sample_ids": list(train.sample_ids),
            "discarded_count": len(train.discarded_sample_ids),
            "discarded_sample_ids": list(train.discarded_sample_ids),
        },
        held_out_name: {
            "observed_count": len(held_out.sample_ids),
            "sample_ids": list(held_out.sample_ids),
            "discarded_count": len(held_out.discarded_sample_ids),
            "discarded_sample_ids": list(held_out.discarded_sample_ids),
        },
    }


def write_predictions_csv(
    path: str | Path,
    sample_ids: Sequence[str],
    targets_processed: Any,
    predictions_processed: Any,
    targets_original: Any,
    predictions_original: Any,
    *,
    trait_name: str,
    outer_fold: int,
) -> None:
    """Write one row per observed outer-test prediction."""
    columns = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in (
            targets_processed,
            predictions_processed,
            targets_original,
            predictions_original,
        )
    ]
    ids = tuple(str(value) for value in sample_ids)
    if any(column.size != len(ids) for column in columns):
        raise ValueError("Prediction columns and sample IDs must have equal lengths")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "trait",
                "outer_fold",
                "target_processed",
                "prediction_processed",
                "target_original",
                "prediction_original",
            ]
        )
        for index, sample_id in enumerate(ids):
            writer.writerow(
                [
                    sample_id,
                    trait_name,
                    int(outer_fold),
                    *[float(column[index]) for column in columns],
                ]
            )


def aggregate_outer_folds(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate nested metric dictionaries across outer folds using ddof=1."""
    if not fold_metrics:
        raise ValueError("At least one outer fold metric is required")
    return _aggregate_metric_node(list(fold_metrics))


def _aggregate_metric_node(values: list[Any]) -> Any:
    if all(isinstance(value, Mapping) for value in values):
        keys = list(values[0])
        for value in values[1:]:
            if set(value) != set(keys):
                raise ValueError("Outer fold metric schemas are inconsistent")
        return {
            key: _aggregate_metric_node([value[key] for value in values])
            for key in keys
        }
    numeric = []
    for value in values:
        if value is None:
            numeric.append(float("nan"))
        elif isinstance(value, (int, float, np.integer, np.floating)):
            numeric.append(float(value))
        else:
            return sanitize_json(values)
    finite = np.asarray(numeric, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "values": sanitize_json(numeric),
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std(ddof=1)) if finite.size >= 2 else None,
        "n": int(finite.size),
    }


def _parse_diploid_dosage(value: str, gt_index: int, line_number: int) -> float:
    fields = value.split(":")
    if gt_index >= len(fields):
        raise ValueError(f"VCF record {line_number} has a truncated sample field")
    genotype = fields[gt_index]
    if genotype in {".", "./.", ".|."}:
        return float("nan")
    alleles = genotype.replace("|", "/").split("/")
    if len(alleles) != 2:
        raise ValueError(
            f"VCF record {line_number} contains non-diploid genotype {genotype!r}"
        )
    if "." in alleles:
        return float("nan")
    if any(allele not in {"0", "1"} for allele in alleles):
        raise ValueError(
            f"VCF record {line_number} contains unsupported genotype {genotype!r}"
        )
    return float(sum(allele == "1" for allele in alleles))


def _unique_strings(values: Iterable[Any], name: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must be nonempty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return normalized


def _as_float_matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return array


def _resolve_vcf_path(directory: Path, stem: str) -> Path:
    candidates = (directory / f"{stem}.vcf.gz", directory / f"{stem}.vcf")
    return next((path for path in candidates if path.is_file()), candidates[0])


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
