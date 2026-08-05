# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Deterministic inner-fold hyperparameter search and aggregation."""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np


InnerFoldCallback = Callable[[Mapping[str, Any], int, int], Any]


@dataclass(frozen=True)
class InnerFoldResult:
    """Objective and selected epoch from one inner validation fold."""

    inner_fold: int
    metric: float
    best_epoch: int
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateResult:
    """Aggregated result for one HPO candidate."""

    candidate_id: int
    parameters: Mapping[str, Any]
    objective: float
    inner_results: tuple[InnerFoldResult, ...]

    @property
    def best_epochs(self) -> tuple[int, ...]:
        return tuple(result.best_epoch for result in self.inner_results)

    @property
    def final_epoch(self) -> int:
        return half_up_median_epoch(self.best_epochs)


@dataclass(frozen=True)
class HPOResult:
    """Complete search outcome with deterministic winner selection."""

    best: CandidateResult
    candidates: tuple[CandidateResult, ...]
    direction: str
    method: str


def normalize_hpo_config(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Normalize a missing, disabled, Bayesian, or grid HPO section."""
    values = dict(config or {})
    enabled = bool(values.get("enabled", bool(values)))
    method = str(values.get("method", values.get("strategy", "bayesian"))).lower()
    aliases = {
        "optuna": "bayesian",
        "tpe": "bayesian",
        "grid_search": "grid",
        "none": "disabled",
    }
    method = aliases.get(method, method)
    if not enabled:
        method = "disabled"
    if method not in {"disabled", "bayesian", "grid"}:
        raise ValueError("HPO method must be disabled, bayesian, or grid")
    direction = str(values.get("direction", "maximize")).lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("HPO direction must be maximize or minimize")
    return {
        **values,
        "enabled": method != "disabled",
        "method": method,
        "direction": direction,
        "metric": values.get("metric", "avg_pearson"),
        "n_trials": int(values.get("n_trials", 50)),
        "seed": int(values.get("seed", 42)),
        "parameters": copy.deepcopy(values.get("parameters", {})),
    }


def set_config_path(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set one dot-separated dictionary/list path in place."""
    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError(f"Invalid configuration path: {path!r}")
    current: Any = config
    for position, part in enumerate(parts[:-1]):
        next_is_index = parts[position + 1].isdigit()
        if part.isdigit():
            if not isinstance(current, list):
                raise TypeError(f"Expected a list before index {part} in '{path}'")
            index = int(part)
            if index >= len(current):
                raise IndexError(f"List index {index} is out of range in '{path}'")
            current = current[index]
            continue
        if not isinstance(current, dict):
            raise TypeError(f"Expected a mapping before key {part!r} in '{path}'")
        if part not in current:
            current[part] = [] if next_is_index else {}
        current = current[part]

    final = parts[-1]
    if final.isdigit():
        if not isinstance(current, list):
            raise TypeError(f"Expected a list before index {final} in '{path}'")
        index = int(final)
        if index >= len(current):
            raise IndexError(f"List index {index} is out of range in '{path}'")
        current[index] = value
    else:
        if not isinstance(current, dict):
            raise TypeError(f"Expected a mapping before key {final!r} in '{path}'")
        current[final] = value


def merge_config(
    base_config: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """Deep-copy a config and apply dot-path candidate parameters."""
    merged = copy.deepcopy(dict(base_config))
    for path, value in parameters.items():
        set_config_path(merged, str(path), copy.deepcopy(value))
    return merged


merge_hpo_config = merge_config


def generate_grid_candidates(
    parameter_space: Mapping[str, Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Generate a stable Cartesian grid from existing HPO definitions."""
    names = sorted(parameter_space)
    value_sets = [
        _grid_values(name, parameter_space[name]) for name in names
    ]
    return [
        dict(zip(names, combination))
        for combination in itertools.product(*value_sets)
    ]


def suggest_parameters(
    trial: Any,
    parameter_space: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Map the existing Aquila parameter schema to an Optuna trial."""
    suggested: Dict[str, Any] = {}
    for name in sorted(parameter_space):
        definition = parameter_space[name]
        kind = str(definition.get("type", "categorical")).lower()
        if kind == "categorical":
            suggested[name] = trial.suggest_categorical(
                name, list(definition["choices"])
            )
        elif kind in {"uniform", "float"}:
            suggested[name] = trial.suggest_float(
                name,
                float(definition["low"]),
                float(definition["high"]),
                step=definition.get("step"),
            )
        elif kind in {"log_uniform", "loguniform"}:
            suggested[name] = trial.suggest_float(
                name,
                float(definition["low"]),
                float(definition["high"]),
                log=True,
            )
        elif kind in {"int_uniform", "int"}:
            suggested[name] = trial.suggest_int(
                name,
                int(definition["low"]),
                int(definition["high"]),
                step=int(definition.get("step", 1)),
                log=bool(definition.get("log", False)),
            )
        else:
            raise ValueError(f"Unsupported HPO type {kind!r} for {name!r}")
    return suggested


def evaluate_candidate(
    candidate_id: int,
    parameters: Mapping[str, Any],
    inner_folds: Iterable[int],
    callback: InnerFoldCallback,
    *,
    metric: str = "avg_pearson",
) -> CandidateResult:
    """Evaluate one parameter set on every inner fold."""
    results = []
    for inner_fold in inner_folds:
        raw = callback(parameters, int(inner_fold), int(candidate_id))
        results.append(_coerce_inner_result(raw, int(inner_fold), metric))
    if not results:
        raise ValueError("At least one inner fold is required")
    objective_values = np.asarray([item.metric for item in results], dtype=float)
    if not np.all(np.isfinite(objective_values)):
        objective = float("nan")
    else:
        objective = float(np.mean(objective_values))
    return CandidateResult(
        candidate_id=int(candidate_id),
        parameters=copy.deepcopy(dict(parameters)),
        objective=objective,
        inner_results=tuple(results),
    )


def run_hpo(
    hpo_config: Mapping[str, Any] | None,
    inner_folds: Sequence[int],
    callback: InnerFoldCallback,
) -> HPOResult:
    """Run disabled/default, deterministic grid, or Optuna Bayesian HPO."""
    config = normalize_hpo_config(hpo_config)
    method = config["method"]
    if method == "disabled":
        candidate = evaluate_candidate(
            0, {}, inner_folds, callback, metric=config["metric"]
        )
        return HPOResult(candidate, (candidate,), config["direction"], method)
    if method == "grid":
        candidates = [
            evaluate_candidate(
                index,
                parameters,
                inner_folds,
                callback,
                metric=config["metric"],
            )
            for index, parameters in enumerate(
                generate_grid_candidates(config["parameters"])
            )
        ]
        return _finish_search(candidates, config["direction"], method)
    return run_bayesian_hpo(config, inner_folds, callback)


def run_bayesian_hpo(
    hpo_config: Mapping[str, Any],
    inner_folds: Sequence[int],
    callback: InnerFoldCallback,
) -> HPOResult:
    """Run seeded Optuna TPE while retaining all inner-fold details."""
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Bayesian HPO requires optuna") from error

    config = normalize_hpo_config(hpo_config)
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    study = optuna.create_study(
        direction=config["direction"],
        sampler=sampler,
    )
    candidates: list[CandidateResult] = []

    def objective(trial: Any) -> float:
        parameters = suggest_parameters(trial, config["parameters"])
        result = evaluate_candidate(
            trial.number,
            parameters,
            inner_folds,
            callback,
            metric=config["metric"],
        )
        candidates.append(result)
        trial.set_user_attr(
            "inner_best_epochs", list(result.best_epochs)
        )
        if not math.isfinite(result.objective):
            raise ValueError("HPO objective is non-finite")
        return result.objective

    study.optimize(objective, n_trials=config["n_trials"])
    return _finish_search(candidates, config["direction"], "bayesian")


def aggregate_inner_results(
    results: Sequence[InnerFoldResult],
) -> tuple[float, int]:
    """Return mean objective and half-up rounded median best epoch."""
    if not results:
        raise ValueError("At least one inner result is required")
    metrics = np.asarray([result.metric for result in results], dtype=float)
    return float(np.mean(metrics)), half_up_median_epoch(
        [result.best_epoch for result in results]
    )


def half_up_mean_epoch(epochs: Sequence[int]) -> int:
    """Round a positive mean to an integer using decimal half-up rounding."""
    if not epochs:
        raise ValueError("At least one best epoch is required")
    if any(int(epoch) < 1 for epoch in epochs):
        raise ValueError("Best epochs must be positive")
    total = sum(Decimal(int(epoch)) for epoch in epochs)
    mean = total / Decimal(len(epochs))
    return int(mean.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def half_up_median_epoch(epochs: Sequence[int]) -> int:
    """Round a positive median to an integer using decimal half-up rounding."""
    if not epochs:
        raise ValueError("At least one best epoch is required")
    if any(int(epoch) < 1 for epoch in epochs):
        raise ValueError("Best epochs must be positive")
    values = sorted(int(epoch) for epoch in epochs)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        median = Decimal(values[mid])
    else:
        median = (Decimal(values[mid - 1]) + Decimal(values[mid])) / Decimal(2)
    return max(1, int(median.quantize(Decimal("1"), rounding=ROUND_HALF_UP)))


round_mean_epoch = half_up_mean_epoch
round_median_epoch = half_up_median_epoch


def _grid_values(name: str, definition: Mapping[str, Any]) -> list[Any]:
    if "values" in definition:
        values = list(definition["values"])
    else:
        kind = str(definition.get("type", "categorical")).lower()
        if kind == "categorical":
            values = list(definition.get("choices", ()))
        elif kind in {"int_uniform", "int"}:
            low = int(definition["low"])
            high = int(definition["high"])
            step = int(definition.get("step", 1))
            values = list(range(low, high + 1, step))
        elif "step" in definition:
            low = Decimal(str(definition["low"]))
            high = Decimal(str(definition["high"]))
            step = Decimal(str(definition["step"]))
            values = []
            current = low
            while current <= high:
                values.append(float(current))
                current += step
        else:
            raise ValueError(
                f"Grid parameter {name!r} requires choices, values, or step"
            )
    if not values:
        raise ValueError(f"Grid parameter {name!r} has no candidate values")
    return values


def _coerce_inner_result(
    value: Any,
    inner_fold: int,
    metric_path: str,
) -> InnerFoldResult:
    if isinstance(value, InnerFoldResult):
        return value
    if hasattr(value, "best_epoch") and hasattr(value, "best_metrics"):
        metrics = value.best_metrics
        return InnerFoldResult(
            inner_fold,
            _nested_metric(metrics, metric_path),
            int(value.best_epoch),
            copy.deepcopy(metrics),
        )
    if isinstance(value, Mapping):
        metrics = value.get("metrics", value)
        best_epoch = value.get("best_epoch", value.get("epoch"))
        if best_epoch is None:
            raise KeyError("Inner-fold callback result requires best_epoch")
        metric_value = value.get("metric")
        if metric_value is None:
            metric_value = _nested_metric(metrics, metric_path)
        return InnerFoldResult(
            inner_fold,
            float(metric_value),
            int(best_epoch),
            copy.deepcopy(dict(metrics)),
        )
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return InnerFoldResult(inner_fold, float(value[0]), int(value[1]))
    raise TypeError("Unsupported inner-fold callback result")


def _nested_metric(values: Mapping[str, Any], path: str) -> float:
    current: Any = values
    for part in path.replace("/", ".").split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            aliases = {
                "best.val_r": "avg_pearson",
                "val_r": "avg_pearson",
                "pearson": "avg_pearson",
            }
            alias = aliases.get(path.replace("/", ".")) or aliases.get(part)
            if alias in values:
                return float(values[alias])
            raise KeyError(f"Metric path '{path}' is not present")
    return float(current)


def select_best_candidate(
    candidates: Sequence[CandidateResult],
    direction: str,
    method: str = "grid",
) -> HPOResult:
    """Select the deterministic winner from evaluated candidates."""
    return _finish_search(candidates, direction, method)


def _finish_search(
    candidates: Sequence[CandidateResult],
    direction: str,
    method: str,
) -> HPOResult:
    if not candidates:
        raise RuntimeError("HPO completed without candidates")
    maximize = direction == "maximize"

    def rank(item: CandidateResult) -> tuple[float, int]:
        score = item.objective
        if not math.isfinite(score):
            score = -float("inf") if maximize else float("inf")
        return ((-score if maximize else score), item.candidate_id)

    best = min(candidates, key=rank)
    if not math.isfinite(best.objective):
        raise RuntimeError("No HPO candidate produced a finite objective")
    return HPOResult(best, tuple(candidates), direction, method)
