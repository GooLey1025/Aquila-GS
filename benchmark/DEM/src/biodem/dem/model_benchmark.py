# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Single-output DEM model and benchmark training primitives."""

# Migrated from: https://github.com/cma2015/DEM

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DEMFitResult:
    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_metric: float
    metrics: dict[str, Any]
    predictions: np.ndarray
    history: tuple[dict[str, float], ...]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encoder(
    input_dim: int,
    n_heads: int,
    hidden_dim: int,
    dropout: float,
    n_encoders: int,
) -> nn.TransformerEncoder:
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
        ),
        n_encoders,
    )


class ExtractionBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_heads: int,
        n_encoders: int,
        hidden_dim: int,
        dropout: float,
        dense_dims: Sequence[int],
        concatenated: bool,
    ) -> None:
        super().__init__()
        self.encoders = _encoder(
            input_dim, n_heads, hidden_dim, dropout, n_encoders
        )
        first, second = (int(value) for value in dense_dims)
        if concatenated:
            self.linears = nn.Sequential(
                nn.Linear(input_dim, first),
                nn.Mish(),
                nn.LayerNorm(first),
                nn.Linear(first, second),
                nn.Linear(second, output_dim),
            )
        else:
            self.linears = nn.Sequential(
                nn.Linear(input_dim, first),
                nn.LayerNorm(first),
                nn.Linear(first, second),
                nn.Mish(),
                nn.Linear(second, output_dim),
            )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.encoders(values)
        values = torch.flatten(values, start_dim=1)
        return self.linears(values), self.linears[0](values)


class IntegrationBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_heads: int,
        n_encoders: int,
        hidden_dim: int,
        dropout: float,
        dense_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.encoders = _encoder(
            input_dim, n_heads, hidden_dim, dropout, n_encoders
        )
        first, second, third = (int(value) for value in dense_dims)
        self.linears = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.LayerNorm(first),
            nn.Linear(first, second),
            nn.Mish(),
            nn.Linear(second, third),
            nn.Linear(third, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linears(torch.flatten(self.encoders(values), start_dim=1))


class DEMBenchmark(nn.Module):
    """Faithful DEM dual-extraction architecture with configurable modalities."""

    def __init__(
        self,
        omics_dims: Sequence[int],
        output_dim: int,
        n_heads: int,
        n_encoders: int,
        hidden_dim: int,
        dropout: float,
        single_hidden: Sequence[int] = (512, 128),
        conc_hidden: Sequence[int] = (1536, 512),
        integrated_hidden: Sequence[int] = (1024, 256, 128),
    ) -> None:
        super().__init__()
        dimensions = tuple(int(value) for value in omics_dims)
        integrated_input = int(conc_hidden[0]) + int(single_hidden[0]) * len(
            dimensions
        )
        for dimension in (*dimensions, sum(dimensions), integrated_input):
            if dimension < 1 or dimension % n_heads != 0:
                raise ValueError(
                    f"DEM dimension {dimension} is incompatible with "
                    f"n_heads={n_heads}"
                )
        self.omics_dims = dimensions
        self.output_dim = int(output_dim)
        self.extract_conc = ExtractionBranch(
            sum(dimensions), output_dim, n_heads, n_encoders, hidden_dim,
            dropout, conc_hidden, True,
        )
        self.extract_each_omics = nn.ModuleList(
            [
                ExtractionBranch(
                    dimension, output_dim, n_heads, n_encoders, hidden_dim,
                    dropout, single_hidden, False,
                )
                for dimension in dimensions
            ]
        )
        self.integrate_extractions = IntegrationBranch(
            integrated_input, output_dim, n_heads, n_encoders, hidden_dim,
            dropout, integrated_hidden,
        )
        self.weights_each_omics = nn.ParameterList(
            [nn.Parameter(torch.ones(1) / len(dimensions)) for _ in dimensions]
        )
        self.weight_conc = nn.Parameter(torch.ones(1))
        self.weight_integrated = nn.Parameter(torch.ones(1))

    def forward(self, omics: Sequence[torch.Tensor]) -> torch.Tensor:
        values = tuple(omics)
        if len(values) != len(self.omics_dims):
            raise ValueError(
                f"DEM expected {len(self.omics_dims)} modalities, got {len(values)}"
            )
        predicted_conc, hidden_conc = self.extract_conc(torch.cat(values, 1))
        predictions = []
        hidden = []
        for index, extractor in enumerate(self.extract_each_omics):
            predicted, representation = extractor(values[index])
            predictions.append(self.weights_each_omics[index] * predicted)
            hidden.append(representation)
        predicted_each = torch.stack(predictions).sum(dim=0)
        predicted_integrated = self.integrate_extractions(
            torch.cat([hidden_conc, *hidden], dim=1)
        )
        return (
            self.weight_conc * predicted_conc
            + self.weight_integrated * predicted_integrated
            + predicted_each
        )


class _ModalDataset(Dataset):
    def __init__(
        self, features: Sequence[np.ndarray], targets: np.ndarray
    ) -> None:
        self.features = tuple(torch.from_numpy(value).float() for value in features)
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        return tuple(value[index] for value in self.features), self.targets[index]


def _as_modalities(features: np.ndarray | Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    if isinstance(features, np.ndarray):
        values = (features,)
    else:
        values = tuple(features)
    values = tuple(np.asarray(value, dtype=np.float32) for value in values)
    if not values or any(value.ndim != 2 for value in values):
        raise ValueError("DEM modalities must be non-empty two-dimensional arrays")
    if len({len(value) for value in values}) != 1:
        raise ValueError("DEM modalities must contain the same samples")
    return values


def _loader(
    features: np.ndarray | Sequence[np.ndarray],
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    modalities = _as_modalities(features)
    if targets.ndim != 2 or targets.shape[1] != 1:
        raise ValueError("DEM targets must have shape (samples, 1)")
    if len(targets) != len(modalities[0]):
        raise ValueError("DEM features and targets must align")
    if shuffle and len(targets) < batch_size:
        raise ValueError(
            f"DEM training requires at least one complete batch: "
            f"{len(targets)} samples < batch_size {batch_size}"
        )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        _ModalDataset(modalities, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        generator=generator if shuffle else None,
    )


def predict_dem(
    model: DEMBenchmark,
    features: np.ndarray | Sequence[np.ndarray],
    targets: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    modalities = _as_modalities(features)
    if len(modalities[0]) == 0:
        raise ValueError("DEM prediction requires at least one sample")
    dummy = (
        np.zeros((len(modalities[0]), 1), dtype=np.float32)
        if targets is None
        else targets
    )
    loader = _loader(modalities, dummy, batch_size, False, 0)
    criterion = nn.MSELoss()
    predictions = []
    losses = []
    model.eval()
    with torch.no_grad():
        for batch_features, observed in loader:
            output = model([value.to(device) for value in batch_features])
            predictions.append(output.cpu().numpy())
            if targets is not None:
                losses.append(float(criterion(output, observed.to(device))))
    prediction_array = np.concatenate(predictions, axis=0)
    return prediction_array, float(np.mean(losses)) if losses else float("nan")


def train_dem(
    train_x: np.ndarray | Sequence[np.ndarray],
    train_y: np.ndarray,
    valid_x: np.ndarray | Sequence[np.ndarray] | None,
    valid_y: np.ndarray | None,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    evaluator: Any,
    trait_name: str,
    fixed_epochs: int | None = None,
) -> DEMFitResult:
    set_seed(seed)
    train_modalities = _as_modalities(train_x)
    model_config = config["model"]
    train_config = config["train"]
    model = DEMBenchmark(
        [value.shape[1] for value in train_modalities],
        1,
        int(model_config["n_heads"]),
        int(model_config["n_encoders"]),
        int(model_config["hidden_dim"]),
        float(model_config["dropout"]),
        model_config.get("single_hidden", (512, 128)),
        model_config.get("conc_hidden", (1536, 512)),
        model_config.get("integrated_hidden", (1024, 256, 128)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 5, 2
    )
    criterion = nn.MSELoss()
    batch_size = int(train_config["batch_size"])
    train_loader = _loader(
        train_modalities, train_y, batch_size, True, seed
    )
    max_epochs = (
        int(fixed_epochs)
        if fixed_epochs is not None
        else int(train_config["max_epochs"])
    )
    patience = int(train_config["patience"])
    min_delta = float(train_config.get("min_delta", 0.0))
    best_metric = -float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    best_predictions = np.empty((0, 1), dtype=np.float32)
    best_metrics: dict[str, Any] = {}
    history = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch_features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model([value.to(device) for value in batch_features])
            loss = criterion(prediction, targets.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if valid_x is not None and valid_y is not None:
            predictions, valid_loss = predict_dem(
                model, valid_x, valid_y, batch_size, device
            )
            mask = np.ones_like(valid_y, dtype=bool)
            metrics = evaluator(
                predictions, valid_y, mask, [trait_name]
            ).metrics
            metric = float(metrics["avg_pearson"])
            record.update(valid_loss=valid_loss, valid_pearson=metric)
            if np.isfinite(metric) and metric > best_metric + min_delta:
                best_metric = metric
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_predictions = predictions
                best_metrics = metrics
                stale = 0
            else:
                stale += 1
            if fixed_epochs is None and stale >= patience:
                history.append(record)
                break
        else:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(record)
    if valid_x is None or valid_y is None:
        best_metric = float("nan")
    return DEMFitResult(
        best_state, best_epoch, best_metric, best_metrics,
        best_predictions, tuple(history),
    )
