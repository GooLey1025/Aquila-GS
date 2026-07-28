# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/ganlab/MENET

"""Trait-specific MENET encoder adapted for benchmark isolation."""

from __future__ import annotations

import torch
from torch import nn


class EncoderResidualBlock(nn.Module):
    """Residual encoder block from MENET's RepGeno stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        pool_size: int,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels,
                out_channels,
                groups=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.residual(inputs) + self.shortcut(inputs))


class TraitSpecificEncoderBenchmark(nn.Module):
    """Learn a phenotype-supervised SNP embedding for RepGeno."""

    def __init__(
        self,
        marker_count: int,
        stride: int,
        output_dimension: int,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            EncoderResidualBlock(1, 8, 5, stride, stride),
            EncoderResidualBlock(8, 16, 5, stride, stride),
            EncoderResidualBlock(16, 32, 5, stride, stride),
            nn.Flatten(),
        )
        with torch.no_grad():
            flattened = self.backbone(torch.ones(2, 1, marker_count)).shape[1]
        self.embedding = nn.Linear(flattened, output_dimension)

    def forward_once(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.backbone(inputs))

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.forward_once(anchor),
            self.forward_once(positive),
            self.forward_once(negative),
        )
