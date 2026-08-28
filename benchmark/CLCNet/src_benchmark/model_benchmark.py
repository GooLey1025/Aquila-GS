# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/SuppurNewer/CLCNet

"""CLCNet architecture preserved for the nested-CV benchmark."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ORIGINAL_SHARED_DIMENSIONS = (4096, 2048, 1024)


class CLCNetBenchmark(nn.Module):
    """Original CLCNet network without the upstream broken shuffle branch."""

    def __init__(
        self,
        input_dim: int,
        shared_dimensions: tuple[int, int, int] = ORIGINAL_SHARED_DIMENSIONS,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("CLCNet input_dim must be positive")
        dims = tuple(int(value) for value in shared_dimensions)
        if len(dims) != 3 or any(value < 1 for value in dims):
            raise ValueError("shared_dimensions must be three positive integers")
        first, second, third = dims
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, third),
        )
        self.shared_layer_1 = nn.Linear(input_dim, third)
        self.main_task_layer = nn.Sequential(
            nn.Linear(third, second),
            nn.ReLU(),
            nn.Linear(second, second),
            nn.ReLU(),
            nn.Linear(second, 1),
        )
        self.aux_task_layer = nn.Sequential(
            nn.Linear(third, second),
            nn.ReLU(),
            nn.Linear(second, second),
            nn.ReLU(),
            nn.Linear(second, first),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_layer(inputs) + self.shared_layer_1(inputs)
        prediction = self.main_task_layer(shared)
        representation = F.normalize(self.aux_task_layer(shared), p=2, dim=1)
        return prediction, representation


def estimate_model_size(
    input_dim: int,
    shared_dimensions: tuple[int, int, int] = ORIGINAL_SHARED_DIMENSIONS,
) -> dict[str, float | int]:
    """Estimate upstream model parameters without allocating the network."""
    first, second, third = tuple(int(value) for value in shared_dimensions)
    parameter_count = (
        input_dim * first
        + first
        + first * second
        + second
        + second * third
        + third
        + input_dim * third
        + third
        + third * second
        + second
        + second * second
        + second
        + second
        + 1
        + third * second
        + second
        + second * second
        + second
        + second * first
        + first
    )
    return {
        "input_dim": int(input_dim),
        "parameter_count": int(parameter_count),
        "fp32_weight_gib": float(parameter_count * 4 / 1024**3),
        "fp32_sgd_training_state_gib": float(parameter_count * 12 / 1024**3),
    }
