# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Single-holdout Bayesian search shared by benchmark model runners."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping

import numpy as np

from aquila.training.hpo import (
    finite_parameter_candidates,
    normalize_hpo_config,
    parameter_key,
    unique_bayesian_trial,
)


def is_holdout_protocol(metadata: Mapping[str, Any]) -> bool:
    """Return whether prepared data uses one 4:1 split instead of inner CV."""
    return str(metadata.get("data_mode", "")).lower() == "benchmark"


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze_choices(space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(space))
    for definition in frozen.values():
        if str(definition.get("type", "")).lower() != "categorical":
            continue
        definition["choices"] = [
            tuple(item) if isinstance(item, list) else item
            for item in definition["choices"]
        ]
    return frozen


def run_holdout_search(
    hpo_config: Mapping[str, Any],
    evaluate: Callable[[Mapping[str, Any], int], tuple[float, Any]],
    *,
    default_trials: int = 200,
) -> dict[str, Any]:
    """Run sequential TPE on one train/validation split and keep the best payload.

    ``evaluate`` receives thawed parameters and the Optuna trial number. It
    returns the validation objective and an arbitrary payload. The winning
    payload is the early-stopping result that must be reused for outer-test
    prediction; this search does not refit the winner.
    """
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Bayesian HPO requires optuna") from error

    configured = copy.deepcopy(dict(hpo_config or {}))
    if configured.get("n_trials") is None:
        configured["n_trials"] = int(default_trials)
    configured["method"] = "bayesian"
    configured["enabled"] = True
    normalized = normalize_hpo_config(configured)
    if normalized["method"] != "bayesian":
        raise ValueError("Holdout benchmark HPO requires method: bayesian")
    space = _freeze_choices(normalized["parameters"])
    if not space:
        raise ValueError("Bayesian HPO requires a parameter space")
    sampler = optuna.samplers.TPESampler(seed=int(normalized["seed"]))
    study = optuna.create_study(
        direction=normalized["direction"],
        sampler=sampler,
    )
    finite_candidates = finite_parameter_candidates(space)
    if (
        finite_candidates is not None
        and int(normalized["n_trials"]) > len(finite_candidates)
    ):
        raise ValueError(
            f"Requested {normalized['n_trials']} Bayesian trials, but the "
            f"finite search space contains only {len(finite_candidates)} "
            "combinations"
        )
    if finite_candidates is not None:
        rng = np.random.default_rng(int(normalized["seed"]))
        rng.shuffle(finite_candidates)
    maximize = normalized["direction"] == "maximize"
    seen: set[tuple[tuple[str, Any], ...]] = set()
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    while len(trials) < int(normalized["n_trials"]):
        trial, parameters, _duplicate_count = unique_bayesian_trial(
            study,
            space,
            seen,
            finite_candidates,
        )
        thawed = {key: _thaw(value) for key, value in parameters.items()}
        objective, payload = evaluate(thawed, int(trial.number))
        objective = float(objective)
        if not math.isfinite(objective):
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise ValueError(
                f"Non-finite holdout objective at trial {trial.number}"
            )
        study.tell(trial, objective)
        seen.add(parameter_key(parameters))
        record = {
            "candidate_id": int(trial.number),
            "parameters": thawed,
            "objective": objective,
        }
        trials.append(record)
        improved = best is None or (
            objective > float(best["objective"])
            if maximize
            else objective < float(best["objective"])
        )
        if improved:
            best = {**record, "payload": payload}
        print(
            f"[Holdout Bayesian HPO] unique={len(trials)}/"
            f"{normalized['n_trials']} trial={trial.number + 1} "
            f"objective={objective:.6f}",
            flush=True,
        )
    if best is None:
        raise RuntimeError("Holdout Bayesian search produced no model")
    return {
        "method": "bayesian",
        "metric": normalized["metric"],
        "direction": normalized["direction"],
        "n_trials": len(trials),
        "best": best,
        "trials": trials,
    }
