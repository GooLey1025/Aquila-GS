# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Nested cross-validation training APIs."""

from importlib import import_module
from typing import Any

__all__ = [
    "CandidateResult",
    "EvaluationResult",
    "FoldJob",
    "FoldJobResult",
    "GPUFoldQueue",
    "PersistentGPUPool",
    "HPOResult",
    "InnerFoldResult",
    "NestedCVTrainer",
    "RegressionEvaluator",
    "TrainingResult",
    "aggregate_inner_results",
    "derive_seed",
    "detect_gpu_ids",
    "evaluate_candidate",
    "evaluate_regression",
    "execute_fold_jobs",
    "generate_grid_candidates",
    "half_up_mean_epoch",
    "half_up_median_epoch",
    "merge_config",
    "normalize_hpo_config",
    "run_bayesian_hpo",
    "run_hpo",
    "set_config_path",
    "resolve_training_seed",
    "set_training_seed",
    "share_memory_tensors",
    "suggest_parameters",
    "supports_bf16",
    "train_final_model",
    "train_inner_fold",
]

_MODULE_EXPORTS = {
    "distributed": {
        "FoldJob",
        "FoldJobResult",
        "GPUFoldQueue",
        "PersistentGPUPool",
        "derive_seed",
        "detect_gpu_ids",
        "execute_fold_jobs",
        "share_memory_tensors",
    },
    "evaluator": {
        "EvaluationResult",
        "RegressionEvaluator",
        "evaluate_regression",
    },
    "hpo": {
        "CandidateResult",
        "HPOResult",
        "InnerFoldResult",
        "aggregate_inner_results",
        "evaluate_candidate",
        "generate_grid_candidates",
        "half_up_mean_epoch",
        "half_up_median_epoch",
        "merge_config",
        "normalize_hpo_config",
        "run_bayesian_hpo",
        "run_hpo",
        "set_config_path",
        "suggest_parameters",
    },
    "trainer": {
        "NestedCVTrainer",
        "TrainingResult",
        "resolve_training_seed",
        "set_training_seed",
        "supports_bf16",
        "train_final_model",
        "train_inner_fold",
    },
}


def __getattr__(name: str) -> Any:
    """Load training components lazily so spawn workers can mask CUDA first."""
    for module_name, exports in _MODULE_EXPORTS.items():
        if name in exports:
            value = getattr(import_module(f"{__name__}.{module_name}"), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
