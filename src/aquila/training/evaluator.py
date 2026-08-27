# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Masked regression evaluation for nested cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and aligned arrays produced by regression evaluation."""

    metrics: dict[str, Any]
    predictions: np.ndarray
    targets: np.ndarray
    mask: np.ndarray


class RegressionEvaluator:
    """Calculate per-trait and macro masked regression metrics."""

    METRIC_NAMES = ("pearson", "r2", "mse", "rmse", "mae")

    def __init__(self, trait_names: Sequence[str] | None = None) -> None:
        self.trait_names = tuple(trait_names) if trait_names is not None else None

    def evaluate(
        self,
        predictions: Any,
        targets: Any,
        mask: Any | None = None,
    ) -> EvaluationResult:
        """Evaluate predictions while excluding missing labels per trait."""
        prediction_array = self._as_matrix(predictions, "predictions")
        target_array = self._as_matrix(targets, "targets")
        if prediction_array.shape != target_array.shape:
            raise ValueError(
                "Predictions and targets must have identical shapes, got "
                f"{prediction_array.shape} and {target_array.shape}"
            )

        if mask is None:
            mask_array = np.isfinite(target_array)
        else:
            mask_array = self._as_matrix(mask, "mask").astype(bool, copy=True)
            if mask_array.shape != target_array.shape:
                raise ValueError("Mask must have the same shape as targets")
            mask_array &= np.isfinite(target_array)
        mask_array &= np.isfinite(prediction_array)

        names = self._trait_names(prediction_array.shape[1])
        per_trait: dict[str, dict[str, float | int]] = {}
        aggregate_values = {metric: [] for metric in self.METRIC_NAMES}

        for index, name in enumerate(names):
            valid = mask_array[:, index]
            predicted = prediction_array[valid, index]
            observed = target_array[valid, index]
            trait_metrics = self._trait_metrics(predicted, observed)
            per_trait[name] = {"n": int(valid.sum()), **trait_metrics}
            for metric_name in self.METRIC_NAMES:
                value = trait_metrics[metric_name]
                if np.isfinite(value):
                    aggregate_values[metric_name].append(value)

        aggregate = {
            metric_name: self._finite_mean(values)
            for metric_name, values in aggregate_values.items()
        }
        aggregate["n_traits"] = sum(
            bool(np.isfinite(metrics["rmse"])) for metrics in per_trait.values()
        )
        aggregate["n_observations"] = int(mask_array.sum())
        within_accession = _within_accession_pearson(
            prediction_array, target_array, mask_array
        )
        aggregate["within_accession_pearson"] = within_accession["mean"]
        aggregate["n_accessions_within_accession"] = within_accession["n_accessions"]

        metrics: dict[str, Any] = {
            "per_trait": per_trait,
            "aggregate": aggregate,
        }
        for metric_name in self.METRIC_NAMES:
            metrics[f"avg_{metric_name}"] = aggregate[metric_name]
        metrics["avg_within_accession_pearson"] = within_accession["mean"]
        metrics["n_accessions_within_accession"] = within_accession["n_accessions"]

        return EvaluationResult(
            metrics=metrics,
            predictions=prediction_array,
            targets=target_array,
            mask=mask_array,
        )

    def _trait_names(self, trait_count: int) -> tuple[str, ...]:
        if self.trait_names is None:
            return tuple(f"trait_{index}" for index in range(trait_count))
        if len(self.trait_names) != trait_count:
            raise ValueError(
                f"Expected {len(self.trait_names)} traits, received {trait_count}"
            )
        return self.trait_names

    @staticmethod
    def _trait_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        if predictions.size == 0:
            return {name: float("nan") for name in RegressionEvaluator.METRIC_NAMES}

        residual = predictions - targets
        mse = float(np.mean(np.square(residual)))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residual)))

        target_ss = float(np.sum(np.square(targets - np.mean(targets))))
        r2 = (
            float(1.0 - np.sum(np.square(residual)) / target_ss)
            if predictions.size >= 2 and target_ss > 0.0
            else float("nan")
        )
        if (
            predictions.size >= 2
            and float(np.std(predictions)) > 0.0
            and float(np.std(targets)) > 0.0
        ):
            pearson = float(np.corrcoef(predictions, targets)[0, 1])
        else:
            pearson = float("nan")
        return {
            "pearson": pearson,
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }

    @staticmethod
    def _as_matrix(values: Any, name: str) -> np.ndarray:
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        array = np.asarray(values)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError(f"{name} must be a one- or two-dimensional array")
        return array

    @staticmethod
    def _finite_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")


def _within_accession_pearson(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    """Mean Pearson r between observed and predicted trait vectors per accession.

    An accession contributes only when it has at least two observed traits and
    both vectors have nonzero variance. This is undefined for single-trait
    evaluations.
    """
    correlations: list[float] = []
    for index in range(predictions.shape[0]):
        valid = mask[index]
        if int(valid.sum()) < 2:
            continue
        predicted = np.asarray(predictions[index, valid], dtype=np.float64)
        observed = np.asarray(targets[index, valid], dtype=np.float64)
        if float(np.std(predicted)) <= 0.0 or float(np.std(observed)) <= 0.0:
            continue
        value = float(np.corrcoef(predicted, observed)[0, 1])
        if np.isfinite(value):
            correlations.append(value)
    return {
        "mean": float(np.mean(correlations)) if correlations else float("nan"),
        "n_accessions": len(correlations),
    }


def evaluate_regression(
    predictions: Any,
    targets: Any,
    mask: Any | None = None,
    trait_names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Convenience wrapper for masked regression evaluation."""
    return RegressionEvaluator(trait_names).evaluate(predictions, targets, mask)
