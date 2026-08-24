# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""CUDA / cuDNN runtime flags for nested-CV workers."""

from __future__ import annotations

import os
from typing import Any, Mapping, MutableMapping

import torch

CUBLAS_DETERMINISTIC_WORKSPACE = ":4096:8"


def train_deterministic_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return ``train.deterministic``, defaulting to off when unset."""
    if not config:
        return False
    train = config.get("train")
    if not isinstance(train, Mapping):
        return False
    return bool(train.get("deterministic", False))


def resolve_train_deterministic(
    config: MutableMapping[str, Any],
    cli_override: bool | None,
) -> bool:
    """Apply CLI override onto ``train.deterministic``; default is false.

    ``cli_override`` is ``None`` when ``--use-deterministic`` was omitted,
    so YAML wins. Passing ``True`` forces deterministic algorithms on.
    """
    train = config.setdefault("train", {})
    if not isinstance(train, dict):
        raise TypeError("config['train'] must be a mapping")
    if cli_override is not None:
        train["deterministic"] = bool(cli_override)
    else:
        train.setdefault("deterministic", False)
    return bool(train["deterministic"])


def configure_cuda_runtime(
    device: str | None = None,
    *,
    deterministic: bool = False,
) -> None:
    """Bind the process to ``device`` and set cuDNN / TF32 / det flags.

    Deterministic mode must set ``CUBLAS_WORKSPACE_CONFIG`` before the first
    CUDA call in this process.
    """
    enabled = bool(deterministic)
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_DETERMINISTIC_WORKSPACE)
    resolved = None if device is None else str(device)
    if resolved is not None and resolved.startswith("cuda:"):
        torch.cuda.set_device(int(resolved.split(":", 1)[1]))
    if enabled:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        return
    torch.use_deterministic_algorithms(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
