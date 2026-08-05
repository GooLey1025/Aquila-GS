# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Serializable train-only preprocessing for masked phenotype traits."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from scipy.stats import skew


@dataclass
class TraitPreprocessing:
    """Fitted preprocessing parameters for one phenotype trait."""

    name: str
    task: str
    skewness: float = 0.0
    use_log1p: bool = False
    log_shift: float = 0.0
    mean: float = 0.0
    std: float = 1.0
    valid_count: int = 0
    zero_variance: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TraitPreprocessing":
        return cls(**values)


class PerTraitPreprocessor:
    """Fit log1p and z-score parameters from masked training rows only."""

    def __init__(
        self,
        skew_threshold: float = 2.0,
        epsilon: float = 1e-8,
    ) -> None:
        if skew_threshold < 0:
            raise ValueError("skew_threshold must be nonnegative")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.skew_threshold = float(skew_threshold)
        self.epsilon = float(epsilon)
        self.traits: List[TraitPreprocessing] = []

    @property
    def is_fitted(self) -> bool:
        return bool(self.traits)

    def fit(
        self,
        y: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray,
        train_indices: Sequence[int] | np.ndarray,
        trait_names: Sequence[str],
        trait_tasks: Sequence[str] | None = None,
    ) -> "PerTraitPreprocessor":
        """Fit each regression trait using only valid selected training values."""
        y_array = _as_numpy_2d(y, "y").astype(np.float64, copy=False)
        mask_array = _as_numpy_2d(mask, "mask").astype(bool, copy=False)
        if y_array.shape != mask_array.shape:
            raise ValueError("y and mask must have identical shapes")
        if len(trait_names) != y_array.shape[1]:
            raise ValueError("trait_names length does not match target columns")

        tasks = list(trait_tasks or ["regression"] * len(trait_names))
        if len(tasks) != len(trait_names):
            raise ValueError("trait_tasks length does not match trait_names")
        invalid_tasks = sorted(set(tasks) - {"regression", "classification"})
        if invalid_tasks:
            raise ValueError(f"Unsupported task types: {invalid_tasks}")

        indices = np.asarray(train_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("train_indices must be one-dimensional")
        if indices.size and (indices.min() < 0 or indices.max() >= len(y_array)):
            raise IndexError("train_indices contain out-of-range values")

        fitted: List[TraitPreprocessing] = []
        for trait_index, (name, task) in enumerate(zip(trait_names, tasks)):
            valid = mask_array[indices, trait_index]
            values = y_array[indices, trait_index][valid]
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Masked training values for trait {name!r} must be finite"
                )
            params = TraitPreprocessing(
                name=str(name),
                task=task,
                valid_count=int(values.size),
            )

            if task == "classification" or values.size == 0:
                fitted.append(params)
                continue

            if (
                values.size >= 3
                and self.skew_threshold > 0
                and float(np.ptp(values)) > self.epsilon
            ):
                skewness = float(skew(values, bias=False))
                params.skewness = skewness if np.isfinite(skewness) else 0.0
                params.use_log1p = (
                    np.isfinite(skewness) and abs(skewness) > self.skew_threshold
                )
            if params.use_log1p:
                minimum = float(values.min())
                params.log_shift = max(0.0, -minimum + self.epsilon)
                values = np.log1p(values + params.log_shift)

            params.mean = float(values.mean())
            observed_std = float(values.std(ddof=0))
            params.zero_variance = (
                not np.isfinite(observed_std) or observed_std <= self.epsilon
            )
            params.std = 1.0 if params.zero_variance else observed_std
            fitted.append(params)

        self.traits = fitted
        return self

    def apply(
        self,
        y: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        """Apply fitted transforms while leaving masked values unchanged."""
        return self._transform(y, mask, inverse=False)

    def inverse(
        self,
        y: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray,
    ) -> torch.Tensor | np.ndarray:
        """Invert fitted transforms while leaving masked values unchanged."""
        return self._transform(y, mask, inverse=True)

    def _transform(
        self,
        y: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray,
        inverse: bool,
    ) -> torch.Tensor | np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted")
        is_tensor = isinstance(y, torch.Tensor)
        y_array = _as_numpy_2d(y, "y").astype(np.float64, copy=True)
        mask_array = _as_numpy_2d(mask, "mask").astype(bool, copy=False)
        if y_array.shape != mask_array.shape:
            raise ValueError("y and mask must have identical shapes")
        if y_array.shape[1] != len(self.traits):
            raise ValueError("Target columns do not match fitted traits")

        for index, params in enumerate(self.traits):
            valid = mask_array[:, index]
            if not valid.any() or params.task == "classification":
                continue
            values = y_array[valid, index]
            if inverse:
                values = values * params.std + params.mean
                if params.use_log1p:
                    values = np.expm1(values) - params.log_shift
            else:
                if params.use_log1p:
                    argument = values + params.log_shift
                    if np.any(argument <= -1.0):
                        raise ValueError(
                            f"Trait {params.name!r} contains values outside the "
                            "fitted log1p domain"
                        )
                    values = np.log1p(argument)
                values = (values - params.mean) / params.std
            y_array[valid, index] = values

        if is_tensor:
            return torch.as_tensor(y_array, dtype=y.dtype, device=y.device)
        original = np.asarray(y)
        return y_array.astype(original.dtype, copy=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "skew_threshold": self.skew_threshold,
            "epsilon": self.epsilon,
            "traits": [trait.to_dict() for trait in self.traits],
        }

    def save_json(self, path: str | Path) -> None:
        """Write fitted preprocessing parameters as JSON."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "PerTraitPreprocessor":
        instance = cls(
            skew_threshold=values["skew_threshold"],
            epsilon=values["epsilon"],
        )
        instance.traits = [
            TraitPreprocessing.from_dict(item) for item in values.get("traits", [])
        ]
        return instance

    @classmethod
    def load_json(cls, path: str | Path) -> "PerTraitPreprocessor":
        """Restore preprocessing parameters from JSON."""
        with Path(path).open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Preprocessor JSON must contain an object")
        return cls.from_dict(values)


def _as_numpy_2d(
    values: torch.Tensor | np.ndarray,
    name: str,
) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return array
