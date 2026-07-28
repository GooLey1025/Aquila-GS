# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/ganlab/MENET

"""MENET benchmark losses isolated from the original source tree."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class TripletLossBenchmark(nn.Module):
    """Phenotype-adaptive triplet loss used by MENET RepGeno."""

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(
        self,
        anchor: torch.Tensor,
        anchor_target: torch.Tensor,
        first: torch.Tensor,
        first_target: torch.Tensor,
        second: torch.Tensor,
        second_target: torch.Tensor,
    ) -> torch.Tensor:
        first_distance = torch.abs(anchor_target - first_target)
        second_distance = torch.abs(anchor_target - second_target)
        first_is_positive = first_distance <= second_distance
        positive = torch.where(first_is_positive, first, second)
        negative = torch.where(first_is_positive, second, first)
        positive_target = torch.where(
            first_is_positive,
            first_target,
            second_target,
        )
        negative_target = torch.where(
            first_is_positive,
            second_target,
            first_target,
        )
        ratio = torch.abs(anchor_target - negative_target) / (
            torch.abs(anchor_target - positive_target) + 1e-2
        )
        adaptive_margin = torch.clip(self.margin * ratio, 0.1, 0.9).reshape(-1)
        anchor = functional.normalize(anchor, p=2, dim=1)
        positive = functional.normalize(positive, p=2, dim=1)
        negative = functional.normalize(negative, p=2, dim=1)
        positive_distance = functional.pairwise_distance(anchor, positive, p=2)
        negative_distance = functional.pairwise_distance(anchor, negative, p=2)
        return functional.softplus(
            positive_distance - negative_distance + adaptive_margin
        ).mean()
