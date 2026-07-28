# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/ganlab/MENET

"""Benchmark-safe MENET models with configurable regularization."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Faithful residual convolution block used by the original MENET."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
        )
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.residual(inputs) + self.shortcut(inputs))


class VariantEffectEncoder(nn.Module):
    """Encode scalar marker sequences through MENET's VE branch."""

    def __init__(
        self,
        marker_count: int,
        config: dict,
        dropout: float,
    ) -> None:
        super().__init__()
        conv_channels, kernel_size, stride = config["conv"]
        res1_channels, res1_kernel, res1_pool = config["res1"]
        res2_channels, res2_kernel, res2_pool = config["res2"]
        self.features = nn.Sequential(
            nn.Conv1d(
                1,
                conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
            ),
            ResidualBlock(
                conv_channels,
                res1_channels,
                res1_kernel,
                res1_pool,
            ),
            ResidualBlock(
                res1_channels,
                res2_channels,
                res2_kernel,
                res2_pool,
            ),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(int(config["adaptive"]))
        with torch.no_grad():
            dummy = torch.zeros(2, 1, marker_count)
            feature_size = self._flatten_features(dummy).shape[1]
        hidden_dim, output_dim = config["embedding_dim"]
        self.projection = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _flatten_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).permute(0, 2, 1)
        return self.adaptive_pool(features).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(self._flatten_features(inputs))


class RelatednessEncoder(nn.Module):
    """Encode relationships to a fixed training-only reference bank."""

    def __init__(
        self,
        reference_count: int,
        config: dict,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim, output_dim = config["embedding_dim"]
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reference_count, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class CrossInformationFusion(nn.Module):
    """Exchange information between VE and RepGeno representations."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.field = nn.Sequential(
            nn.Linear(2 * dimension, 4 * dimension),
            nn.BatchNorm1d(4 * dimension),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(dropout),
            nn.Linear(4 * dimension, dimension),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.field(inputs)


class MeNetBenchmark(nn.Module):
    """Single-chromosome MENET variant for leakage-safe benchmarking."""

    def __init__(
        self,
        marker_count: int,
        reference_count: int,
        config: dict,
        dropout: float,
    ) -> None:
        super().__init__()
        self.variant_effect = VariantEffectEncoder(
            marker_count,
            config["VE"],
            dropout,
        )
        self.relatedness = RelatednessEncoder(
            reference_count,
            config["RepGeno"],
            dropout,
        )
        fusion_dimension = int(
            config["output"]["VE_and_RepGeno_embedding_dim"]
        )
        self.variant_fusion = CrossInformationFusion(
            fusion_dimension,
            dropout,
        )
        self.relatedness_fusion = CrossInformationFusion(
            fusion_dimension,
            dropout,
        )
        self.output = nn.Sequential(
            nn.Linear(2 * fusion_dimension, config["output"]["embedding_dim"]),
            nn.BatchNorm1d(config["output"]["embedding_dim"]),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(dropout),
            nn.Linear(config["output"]["embedding_dim"], 1),
        )

    def forward(
        self,
        genotype: torch.Tensor,
        relatedness: torch.Tensor,
    ) -> torch.Tensor:
        variant_features = self.variant_effect(genotype)
        relatedness_features = self.relatedness(relatedness)
        combined = torch.cat((variant_features, relatedness_features), dim=1)
        variant_fused = self.variant_fusion(combined) + variant_features
        relatedness_fused = (
            self.relatedness_fusion(combined) + relatedness_features
        )
        return self.output(torch.cat((variant_fused, relatedness_fused), dim=1))
