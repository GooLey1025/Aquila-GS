# -*- coding: utf-8 -*-

from __future__ import annotations

import torch

from aquila.training.cuda_runtime import (
    configure_cuda_runtime,
    resolve_train_deterministic,
    train_deterministic_enabled,
)
from aquila.training.trainer import resolve_training_seed


def test_deterministic_defaults_off_when_missing() -> None:
    assert train_deterministic_enabled({}) is False
    assert train_deterministic_enabled({"train": {}}) is False
    assert train_deterministic_enabled(None) is False


def test_yaml_true_without_cli_override() -> None:
    config: dict = {"train": {"deterministic": True}}
    assert resolve_train_deterministic(config, None) is True
    assert config["train"]["deterministic"] is True


def test_cli_use_deterministic_forces_on() -> None:
    config: dict = {"train": {"deterministic": False}}
    assert resolve_train_deterministic(config, True) is True
    config = {"train": {"deterministic": True}}
    assert resolve_train_deterministic(config, None) is True


def test_missing_yaml_key_stays_false_without_cli() -> None:
    config: dict = {"train": {"precision": "bf16"}}
    assert resolve_train_deterministic(config, None) is False
    assert config["train"]["deterministic"] is False


def test_shared_training_seed_prefers_train_then_hpo() -> None:
    assert resolve_training_seed({"train": {"seed": 7}, "hpo": {"seed": 9}}) == 7
    assert resolve_training_seed({"hpo": {"seed": 9}}, fallback=42) == 9
    assert resolve_training_seed({}, fallback=42) == 42


def test_configure_cuda_runtime_toggles_flags() -> None:
    configure_cuda_runtime(deterministic=True)
    assert torch.are_deterministic_algorithms_enabled() is True
    configure_cuda_runtime(deterministic=False)
    assert torch.are_deterministic_algorithms_enabled() is False
    if torch.cuda.is_available():
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False
