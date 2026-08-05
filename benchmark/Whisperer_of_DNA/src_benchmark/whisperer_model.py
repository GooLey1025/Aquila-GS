# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/Marxin1992/Whisperer_of_DNA.git

"""Training orchestration around the preserved DNAWhisper model."""

from __future__ import annotations

import copy
import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from aquila.training.evaluator import evaluate_regression


WHISPERER_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainingResult:
    """Selected model state and inner-validation history."""

    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_metric: float
    best_metrics: dict[str, Any]
    history: tuple[dict[str, Any], ...]


def import_dna_whisper() -> type:
    """Import the upstream Lightning module without exposing generic packages."""
    if str(WHISPERER_ROOT) not in sys.path:
        sys.path.insert(0, str(WHISPERER_ROOT))
    namespace = "aquila_whisperer_upstream"
    training_path = WHISPERER_ROOT / "training"
    models_path = training_path / "models"
    if namespace not in sys.modules:
        package = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(namespace, loader=None, is_package=True)
        )
        package.__path__ = [str(WHISPERER_ROOT)]
        sys.modules[namespace] = package
    training_name = f"{namespace}.training"
    if training_name not in sys.modules:
        package = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(training_name, loader=None, is_package=True)
        )
        package.__path__ = [str(training_path)]
        sys.modules[training_name] = package
    models_name = f"{training_name}.models"
    if models_name not in sys.modules:
        package = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(models_name, loader=None, is_package=True)
        )
        package.__path__ = [str(models_path)]
        sys.modules[models_name] = package
    existing_training = sys.modules.get("training")
    sys.modules["training"] = sys.modules[training_name]
    try:
        module = importlib.import_module(f"{models_name}.DNAWhisper")
    except ImportError as error:
        raise ImportError(
            "DNA Whisper requires its upstream dependencies, including "
            "pytorch-lightning, einops, and entmax."
        ) from error
    finally:
        if existing_training is None:
            sys.modules.pop("training", None)
        else:
            sys.modules["training"] = existing_training
    _patch_standard_attention_mask_semantics(models_name)
    return module.DNAWhisper


def _patch_standard_attention_mask_semantics(models_name: str) -> None:
    """Make the no-flash MHA fallback consume the upstream valid-position mask."""
    attention_module = importlib.import_module(f"{models_name}.attention_types")
    attention_class = attention_module.StandardAttention
    if getattr(attention_class, "_aquila_valid_mask_patch", False):
        return
    original_forward = attention_class.forward

    def forward_with_valid_mask(
        self: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        proj_q: torch.Tensor | None = None,
        proj_k: torch.Tensor | None = None,
        batch_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mha_mask = mask
        if (
            mask is not None
            and mask.dtype == torch.bool
            and mask.ndim == 2
            and mask.shape == (q.shape[0], k.shape[1])
        ):
            mha_mask = ~mask
        return original_forward(
            self,
            q,
            k,
            v,
            mask=mha_mask,
            proj_q=proj_q,
            proj_k=proj_k,
            batch_idx=batch_idx,
        )

    attention_class.forward = forward_with_valid_mask
    attention_class._aquila_valid_mask_patch = True


def apply_candidate_overrides(
    base_config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    trait_name: str,
) -> dict[str, Any]:
    """Apply all benchmark model overrides to a deep-copied upstream config."""
    config = copy.deepcopy(dict(base_config))
    dropout = float(parameters.get("dropout", config["embedding"]["dropout_rate"]))
    layers = int(
        parameters.get(
            "encoder_layers",
            config["GFI_FormerBLOCKS"]["blocks"][0]["encoder"]["num_layers"],
        )
    )
    config["embedding"]["input_type"] = "SNP"
    config["embedding"]["input_dims"] = 10
    config["embedding"]["dropout_rate"] = dropout
    config["output_layer"]["phenotype_dim"] = 1
    config["output_layer"]["phenotype_name"] = [trait_name]
    config["output_layer"]["dropout_rate"] = dropout
    for block in config["GFI_FormerBLOCKS"]["blocks"][
        : config["GFI_FormerBLOCKS"]["num_blocks"]
    ]:
        block["encoder"]["num_layers"] = layers
        block["encoder"]["attention"]["dropout_rate"] = dropout
        block["encoder"]["dropout_rate"] = dropout
        block["decoder"]["cross_attention"]["dropout_rate"] = dropout
        block["decoder"]["MOE"]["dropout_rate"] = dropout
        block["decoder"]["pooling"]["dropout_rate"] = dropout
    auxiliary = config["loss_config"]["auxiliary_losses"]
    auxiliary["Deep_Supervision"]["enabled"] = False
    auxiliary["PWCosSim"]["enabled"] = False
    auxiliary["correlation"]["enabled"] = False
    return config


def set_seed(seed: int) -> None:
    """Seed all RNGs used by the adapter and upstream model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    config: Mapping[str, Any],
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    *,
    random_seed: int | None = None,
) -> torch.nn.Module:
    """Instantiate the preserved DNAWhisper Lightning module from scratch."""
    dna_whisper = import_dna_whisper()
    model_config = copy.deepcopy(dict(config))
    if random_seed is not None:
        model_config["random_seed"] = int(random_seed)
    model = dna_whisper(
        config=model_config,
        optimizer_config={
            "name": "adamw",
            "params": {
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            },
        },
        scheduler_config={"name": "none", "params": {}},
    )
    return model.to(device)


def _loader(
    genotypes: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(genotypes, dtype=np.float32)),
        torch.from_numpy(np.asarray(targets, dtype=np.float32)).reshape(-1, 1),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
    )


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for features, phenotype in loader:
            output = model(features.to(device))["final_pred"]
            predictions.append(output.detach().cpu())
            targets.append(phenotype)
    prediction_array = torch.cat(predictions).numpy().reshape(-1)
    target_array = torch.cat(targets).numpy().reshape(-1)
    return (
        prediction_array,
        target_array,
        float(np.mean(np.square(prediction_array - target_array))),
    )


def train_model(
    train_genotypes: np.ndarray,
    train_targets: np.ndarray,
    valid_genotypes: np.ndarray | None,
    valid_targets: np.ndarray | None,
    config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    device: torch.device,
    seed: int,
    *,
    max_epochs: int,
    patience: int,
    fixed_epochs: int | None = None,
) -> TrainingResult:
    """Train with validation-only selection or a fixed outer-refit duration."""
    set_seed(seed)
    model = build_model(
        config,
        float(parameters["learning_rate"]),
        float(parameters["weight_decay"]),
        device,
        random_seed=seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    batch_size = int(parameters["batch_size"])
    train_loader = _loader(train_genotypes, train_targets, batch_size, True)
    valid_loader = (
        _loader(valid_genotypes, valid_targets, batch_size, False)
        if valid_genotypes is not None and valid_targets is not None
        else None
    )
    epoch_count = int(fixed_epochs or max_epochs)
    best_epoch = epoch_count
    best_metric = -float("inf")
    best_metrics: dict[str, Any] = {}
    best_state = copy.deepcopy(model.state_dict())
    history = []
    stale = 0
    for epoch in range(1, epoch_count + 1):
        model.train()
        train_losses = []
        for features, phenotype in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = features.to(device)
            phenotype = phenotype.to(device)
            outputs = model(features)
            losses = model.compute_loss(outputs, phenotype)
            loss = losses["total_loss"]
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
        }
        if valid_loader is None:
            best_state = copy.deepcopy(model.state_dict())
            history.append(row)
            continue
        predictions, targets, valid_loss = _predict(model, valid_loader, device)
        metrics = evaluate_regression(
            predictions, targets, np.ones_like(targets, dtype=bool), ["trait"]
        ).metrics
        metric = float(metrics["avg_pearson"])
        row.update(
            {
                "valid_loss": valid_loss,
                "valid_avg_pearson": metric,
                "valid_metrics": metrics,
            }
        )
        history.append(row)
        if np.isfinite(metric) and metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    return TrainingResult(
        {key: value.detach().cpu() for key, value in best_state.items()},
        best_epoch,
        best_metric,
        best_metrics,
        tuple(history),
    )


def predict_model(
    state_dict: Mapping[str, torch.Tensor],
    genotypes: np.ndarray,
    targets: np.ndarray,
    config: Mapping[str, Any],
    parameters: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Predict one held-out split from a saved DNAWhisper state."""
    model = build_model(
        config,
        float(parameters["learning_rate"]),
        float(parameters["weight_decay"]),
        device,
    )
    model.load_state_dict(state_dict)
    predictions, _, loss = _predict(
        model,
        _loader(genotypes, targets, int(parameters["batch_size"]), False),
        device,
    )
    return predictions, loss
