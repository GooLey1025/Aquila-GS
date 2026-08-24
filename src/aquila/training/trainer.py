# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Training primitives for leakage-free nested cross-validation."""

from __future__ import annotations

import copy
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aquila.metrics import MultiTaskLoss

from .evaluator import EvaluationResult, RegressionEvaluator


@dataclass
class TrainingResult:
    """State returned by inner-fold and final model fitting."""

    best_epoch: int
    best_metrics: Dict[str, Any]
    best_state_dict: Dict[str, torch.Tensor]
    history: list[Dict[str, Any]] = field(default_factory=list)
    checkpoint_state: Dict[str, Any] = field(default_factory=dict)

    @property
    def epochs_trained(self) -> int:
        return len(self.history)


def set_training_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without requiring CUDA."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_seed(
    config: Mapping[str, Any] | None,
    *,
    fallback: int = 42,
) -> int:
    """Return the shared init/train seed for every nested-CV fit.

    Preference: ``train.seed`` > ``hpo.seed`` > ``fallback`` (prepared-data seed).
    """
    if not config:
        return int(fallback)
    train = config.get("train")
    if isinstance(train, Mapping) and train.get("seed") is not None:
        return int(train["seed"])
    hpo = config.get("hpo")
    if isinstance(hpo, Mapping) and hpo.get("seed") is not None:
        return int(hpo["seed"])
    return int(fallback)


def supports_bf16(device: torch.device | str) -> bool:
    """Return whether CUDA bf16 autocast is safe on the selected device."""
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(checker and checker())


class NestedCVTrainer:
    """Fit masked multi-trait regression models from tensor data loaders."""

    def __init__(
        self,
        model: nn.Module,
        *,
        num_regression_tasks: int,
        num_classification_tasks: int = 0,
        device: torch.device | str | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_type: str = "mse",
        uncertainty_weighting: bool = False,
        huber_delta: float = 1.0,
        gradient_clip_norm: float | None = 1.0,
        use_bf16: bool = True,
        trait_names: Sequence[str] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        scheduler_type: str | None = "cosine_warmup",
        scheduler_params: Mapping[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        if num_regression_tasks < 1 and num_classification_tasks < 1:
            raise ValueError(
                "At least one regression or classification task is required"
            )
        if num_regression_tasks < 0 or num_classification_tasks < 0:
            raise ValueError("Task counts must be nonnegative")
        self.num_regression_tasks = int(num_regression_tasks)
        self.num_classification_tasks = int(num_classification_tasks)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(seed)
        set_training_seed(self.seed)
        self.model = model.to(self.device)
        self.gradient_clip_norm = gradient_clip_norm
        self.use_bf16 = bool(use_bf16 and supports_bf16(self.device))
        self.learning_rate = float(learning_rate)
        self.scheduler_type = _normalize_scheduler_type(scheduler_type)
        self.scheduler_params = dict(scheduler_params or {})
        self.scheduler: Any | None = None
        self._step_scheduler_per_batch = False
        self.evaluator = RegressionEvaluator(trait_names)
        self.criterion = criterion or MultiTaskLoss(
            num_regression_tasks=self.num_regression_tasks,
            num_classification_tasks=self.num_classification_tasks,
            loss_type=loss_type,
            uncertainty_weighting=uncertainty_weighting,
            huber_delta=huber_delta,
        )
        self.criterion = self.criterion.to(self.device)
        if optimizer is None:
            parameters = list(self.model.parameters()) + list(
                self.criterion.parameters()
            )
            self.optimizer = torch.optim.AdamW(
                parameters,
                lr=self.learning_rate,
                weight_decay=float(weight_decay),
            )
        else:
            self.optimizer = optimizer

    def train_inner(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        *,
        max_epochs: int,
        patience: int = 20,
        metric: str = "avg_pearson",
        direction: str = "maximize",
        min_delta: float = 1e-4,
        metrics_log_path: str | Path | None = None,
    ) -> TrainingResult:
        """Early-stop on an inner validation fold and restore its best model."""
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if patience < 1:
            raise ValueError("patience must be positive")
        self._configure_scheduler(train_loader, num_epochs=int(max_epochs))
        maximize = _is_maximize(direction)
        best_score = -float("inf") if maximize else float("inf")
        best_epoch = 0
        best_metrics: Dict[str, Any] = {}
        best_state: Dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        history: list[Dict[str, Any]] = []
        log_path = Path(metrics_log_path) if metrics_log_path else None

        for epoch in range(1, int(max_epochs) + 1):
            # Keep train_loss on device until epoch end; only the scalar syncs.
            train_loss = self._train_epoch(train_loader, materialize_loss=True)
            row: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": self._current_lr(),
            }
            validation = self.evaluate(valid_loader)
            score = _metric_value(validation.metrics, metric)
            row["validation"] = validation.metrics
            self._step_scheduler_on_validation(validation.metrics, score)
            if best_state is None:
                best_epoch = epoch
                best_metrics = validation.metrics
                # Keep the best weights on-device during the loop; a full
                # D2H clone on every improvement stalls the SM pipeline.
                best_state = _device_state_dict(self.model)
            improved = _improved(score, best_score, maximize, min_delta)
            if improved:
                best_score = score
                best_epoch = epoch
                best_metrics = validation.metrics
                best_state = _device_state_dict(self.model)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    history.append(row)
                    _append_metrics_log(
                        log_path,
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "learning_rate": row["learning_rate"],
                            "valid_r": score,
                            "best_epoch": best_epoch,
                            "best_valid_r": best_score,
                            "early_stop": True,
                            "seed": self.seed,
                        },
                    )
                    break
            history.append(row)
            _append_metrics_log(
                log_path,
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": row["learning_rate"],
                    "valid_r": score,
                    "best_epoch": best_epoch,
                    "best_valid_r": best_score
                    if np.isfinite(best_score)
                    else None,
                    "early_stop": False,
                    "seed": self.seed,
                },
            )

        if best_state is None:
            raise RuntimeError("Inner training produced no validated checkpoint")
        self.model.load_state_dict(best_state)
        best_state_cpu = {
            name: tensor.detach().cpu().clone()
            for name, tensor in best_state.items()
        }
        return TrainingResult(
            best_epoch=best_epoch,
            best_metrics=dict(best_metrics),
            best_state_dict=best_state_cpu,
            history=history,
            checkpoint_state=self.checkpoint_state(
                epoch=best_epoch, metrics=best_metrics
            ),
        )

    def train_fixed_epochs(
        self,
        train_loader: DataLoader,
        *,
        epochs: int,
    ) -> TrainingResult:
        """Train for exactly ``epochs`` passes without validation or stopping."""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        # Schedule over the actual refit length so cosine reaches min_lr at end.
        self._configure_scheduler(train_loader, num_epochs=int(epochs))
        history = []
        for epoch in range(1, int(epochs) + 1):
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": self._train_epoch(train_loader),
                    "learning_rate": self._current_lr(),
                }
            )
        state = _cpu_state_dict(self.model)
        metrics: Dict[str, Any] = {"train_loss": history[-1]["train_loss"]}
        return TrainingResult(
            best_epoch=int(epochs),
            best_metrics=metrics,
            best_state_dict=state,
            history=history,
            checkpoint_state=self.checkpoint_state(
                epoch=int(epochs), metrics=metrics
            ),
        )

    def evaluate(self, loader: DataLoader) -> EvaluationResult:
        """Predict a loader and compute masked regression metrics."""
        self.model.eval()
        predictions = []
        targets = []
        masks = []
        with torch.no_grad():
            for batch in loader:
                inputs, target, mask = _unpack_batch(batch, self.device)
                with self._autocast():
                    output = _regression_output(self.model(inputs))
                predictions.append(output.detach().float())
                targets.append(target.detach().float())
                masks.append(mask.detach().bool())
        if not predictions:
            raise ValueError("Cannot evaluate an empty data loader")
        # Single host sync after the full eval pass (not per-batch).
        return self.evaluator.evaluate(
            torch.cat(predictions),
            torch.cat(targets),
            torch.cat(masks),
        )

    def checkpoint_state(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Build a portable checkpoint dictionary."""
        state = {
            "epoch": int(epoch),
            "model_state_dict": _cpu_state_dict(self.model),
            "criterion_state_dict": _cpu_state_dict(self.criterion),
            "optimizer_state_dict": _to_cpu_copy(self.optimizer.state_dict()),
            "metrics": copy.deepcopy(dict(metrics or {})),
            "seed": self.seed,
            "precision": "bf16" if self.use_bf16 else "float32",
            "scheduler_type": self.scheduler_type,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = _to_cpu_copy(self.scheduler.state_dict())
        return state

    def _configure_scheduler(
        self,
        train_loader: DataLoader,
        *,
        num_epochs: int,
    ) -> None:
        """Build the LR schedule for a training run of ``num_epochs`` epochs."""
        self.scheduler = None
        self._step_scheduler_per_batch = False
        # Reset param groups to the configured peak/base LR before attaching.
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

        kind = self.scheduler_type
        if kind in {"none", "constant", "disabled"}:
            return

        steps_per_epoch = max(1, int(len(train_loader)))
        total_steps = int(num_epochs) * steps_per_epoch
        params = dict(self.scheduler_params)

        if kind == "cosine_warmup":
            warmup_epochs = int(params.get("warmup_epochs", 5))
            warmup_epochs = max(0, min(warmup_epochs, max(0, int(num_epochs) - 1)))
            warmup_steps = warmup_epochs * steps_per_epoch
            cosine_steps = max(1, total_steps - warmup_steps)
            start_factor = float(params.get("warmup_start_factor", 0.01))
            min_lr = float(params.get("min_lr", 1e-6))
            if warmup_steps <= 0:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_steps,
                    eta_min=min_lr,
                )
            else:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_steps,
                    eta_min=min_lr,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_steps],
                )
            self._step_scheduler_per_batch = True
            return

        if kind == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=float(params.get("max_lr", self.learning_rate)),
                total_steps=total_steps,
                pct_start=float(params.get("pct_start", 0.3)),
                anneal_strategy=str(params.get("anneal_strategy", "cos")),
                div_factor=float(params.get("div_factor", 25.0)),
                final_div_factor=float(params.get("final_div_factor", 1e4)),
            )
            self._step_scheduler_per_batch = True
            return

        if kind == "reduce_on_plateau":
            mode = str(params.get("mode", "max")).lower()
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=float(params.get("factor", 0.5)),
                patience=int(params.get("patience", 5)),
                min_lr=float(params.get("min_lr", 0.0)),
            )
            self._step_scheduler_per_batch = False
            return

        raise ValueError(
            f"Unknown scheduler_type={kind!r}; expected "
            "cosine_warmup, one_cycle, reduce_on_plateau, or none"
        )

    def _step_scheduler_on_validation(
        self,
        metrics: Mapping[str, Any],
        score: float,
    ) -> None:
        if self.scheduler is None or self._step_scheduler_per_batch:
            return
        if self.scheduler_type != "reduce_on_plateau":
            return
        mode = str(getattr(self.scheduler, "mode", "min")).lower()
        if mode == "max":
            self.scheduler.step(float(score))
            return
        value = metrics.get("avg_rmse")
        if value is None:
            value = -float(score)
        self.scheduler.step(float(value))

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _train_epoch(
        self,
        loader: DataLoader,
        *,
        materialize_loss: bool = True,
    ) -> float:
        self.model.train()
        total_loss: torch.Tensor | None = None
        batches = 0
        for batch in loader:
            inputs, targets, masks = _unpack_multitask_batch(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                outputs = _model_task_outputs(self.model(inputs))
                safe_targets = {
                    task: torch.where(
                        masks[task],
                        targets[task],
                        torch.zeros_like(targets[task]),
                    )
                    for task in targets
                }
                loss_value = self.criterion(
                    outputs,
                    safe_targets,
                    masks,
                    return_details=False,
                )
                loss = loss_value[0] if isinstance(loss_value, tuple) else loss_value
            # Do not torch.isfinite(loss) per step — that forces a device sync
            # every batch and caps sustained GPU utilization.
            loss.backward()
            # clip_grad_norm_ device-syncs; skip when unset (prefer yaml null
            # for max GPU util on small-batch HPO workloads).
            if self.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.gradient_clip_norm)
                )
            self.optimizer.step()
            if self._step_scheduler_per_batch and self.scheduler is not None:
                self.scheduler.step()
            if materialize_loss:
                detached_loss = loss.detach()
                total_loss = (
                    detached_loss
                    if total_loss is None
                    else total_loss + detached_loss
                )
            batches += 1
        if batches == 0:
            raise ValueError("Cannot train with an empty data loader")
        if not materialize_loss:
            return float("nan")
        if total_loss is None:
            raise RuntimeError("Training produced no loss values")
        mean_loss = total_loss / batches
        if not torch.isfinite(mean_loss):
            raise FloatingPointError("Non-finite training loss encountered")
        return float(mean_loss.detach().cpu())

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()


def _normalize_scheduler_type(value: str | None) -> str:
    if value is None:
        return "none"
    normalized = str(value).strip().lower()
    aliases = {
        "": "none",
        "null": "none",
        "none": "none",
        "constant": "none",
        "disabled": "none",
        "cosine": "cosine_warmup",
        "cosineannealingwarmup": "cosine_warmup",
        "cosine_annealing_warmup": "cosine_warmup",
        "reduceonplateau": "reduce_on_plateau",
        "plateau": "reduce_on_plateau",
        "onecycle": "one_cycle",
        "one_cycle_lr": "one_cycle",
    }
    return aliases.get(normalized, normalized)

def train_inner_fold(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    num_regression_tasks: int,
    max_epochs: int,
    trainer_kwargs: Mapping[str, Any] | None = None,
    **fit_kwargs: Any,
) -> TrainingResult:
    """Construct a trainer and run one early-stopped inner fold."""
    trainer = NestedCVTrainer(
        model,
        num_regression_tasks=num_regression_tasks,
        **dict(trainer_kwargs or {}),
    )
    return trainer.train_inner(
        train_loader, valid_loader, max_epochs=max_epochs, **fit_kwargs
    )


def train_final_model(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    num_regression_tasks: int,
    epochs: int,
    trainer_kwargs: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Construct a fresh trainer and perform the exact final-epoch refit."""
    trainer = NestedCVTrainer(
        model,
        num_regression_tasks=num_regression_tasks,
        **dict(trainer_kwargs or {}),
    )
    return trainer.train_fixed_epochs(train_loader, epochs=epochs)


def _unpack_batch(
    batch: Any,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    inputs, targets, masks = _unpack_multitask_batch(batch, device)
    if "regression" not in targets:
        raise KeyError("Batch must provide regression targets for evaluation")
    return inputs, targets["regression"], masks["regression"]


def _unpack_multitask_batch(
    batch: Any,
    device: torch.device,
) -> tuple[Any, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if isinstance(batch, Mapping):
        input_value = _first_present(batch, ("inputs", "x", "X"))
        if input_value is None:
            excluded = {
                "sample_id",
                "sample_ids",
                "index",
                "regression_targets",
                "classification_targets",
                "targets",
                "y",
                "Y",
                "y_raw",
                "Y_raw",
                "regression_mask",
                "classification_mask",
                "target_mask",
                "mask",
                "y_mask",
                "Y_mask",
            }
            branches = {
                key: value
                for key, value in batch.items()
                if key not in excluded and isinstance(value, torch.Tensor)
            }
            input_value = (
                branches if len(branches) != 1 else next(iter(branches.values()))
            )

        targets: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}
        if "regression_targets" in batch:
            targets["regression"] = _as_2d_tensor(
                batch["regression_targets"], device, torch.float32
            )
            mask_value = batch.get("regression_mask")
            if mask_value is None:
                masks["regression"] = torch.isfinite(targets["regression"])
            else:
                masks["regression"] = _as_2d_tensor(
                    mask_value, device, torch.bool
                )
        if "classification_targets" in batch:
            targets["classification"] = _as_2d_tensor(
                batch["classification_targets"], device, torch.float32
            )
            mask_value = batch.get("classification_mask")
            if mask_value is None:
                masks["classification"] = torch.isfinite(
                    targets["classification"]
                )
            else:
                masks["classification"] = _as_2d_tensor(
                    mask_value, device, torch.bool
                )
        if not targets:
            fallback_targets = _first_present(
                batch, ("targets", "y", "Y", "y_raw", "Y_raw")
            )
            fallback_mask = _first_present(
                batch, ("target_mask", "mask", "y_mask", "Y_mask")
            )
            if fallback_targets is None:
                raise KeyError("Batch must provide model targets")
            targets["regression"] = _as_2d_tensor(
                fallback_targets, device, torch.float32
            )
            if fallback_mask is None:
                masks["regression"] = torch.isfinite(targets["regression"])
            else:
                masks["regression"] = _as_2d_tensor(
                    fallback_mask, device, torch.bool
                )
    elif isinstance(batch, (tuple, list)) and len(batch) in (2, 3):
        input_value, fallback_targets = batch[:2]
        fallback_mask = batch[2] if len(batch) == 3 else None
        targets = {
            "regression": _as_2d_tensor(
                fallback_targets, device, torch.float32
            )
        }
        if fallback_mask is None:
            masks = {"regression": torch.isfinite(targets["regression"])}
        else:
            masks = {
                "regression": _as_2d_tensor(fallback_mask, device, torch.bool)
            }
    else:
        raise TypeError("Batch must be a mapping or a two/three-item sequence")

    if input_value is None:
        raise KeyError("Batch must provide model inputs")
    for task in list(targets):
        if targets[task].shape != masks[task].shape:
            raise ValueError(
                f"{task} targets and mask must have identical shapes"
            )
        masks[task] = masks[task] & torch.isfinite(targets[task])
    return _move_to_device(input_value, device), targets, masks


def _model_task_outputs(output: Any) -> Dict[str, torch.Tensor]:
    if isinstance(output, Mapping):
        resolved: Dict[str, torch.Tensor] = {}
        if "regression" in output:
            resolved["regression"] = _regression_output(output)
        if "classification" in output:
            classification = output["classification"]
            if not isinstance(classification, torch.Tensor):
                raise TypeError("Classification model output must be a tensor")
            resolved["classification"] = (
                classification[:, None]
                if classification.ndim == 1
                else classification
            )
        if not resolved:
            raise KeyError(
                "Model output must contain regression and/or classification"
            )
        return resolved
    return {"regression": _regression_output(output)}


def _first_present(values: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in values:
            return values[name]
    return None


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _as_2d_tensor(
    value: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
    return tensor[:, None] if tensor.ndim == 1 else tensor


def _regression_output(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping):
        if "regression" not in output:
            raise KeyError("Model output does not contain a 'regression' tensor")
        output = output["regression"]
    if not isinstance(output, torch.Tensor):
        raise TypeError("Regression model output must be a tensor")
    return output[:, None] if output.ndim == 1 else output


def _append_metrics_log(
    path: Path | None,
    record: Mapping[str, Any],
) -> None:
    """Append one JSONL metrics row and flush for live ``tail -f`` viewing."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in record.items()
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _device_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def _to_cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _metric_value(metrics: Mapping[str, Any], path: str) -> float:
    normalized = path.replace("/", ".")
    direct_aliases = {
        "best.val_r": "avg_pearson",
        "val_r": "avg_pearson",
        "pearson": "avg_pearson",
        "mse": "avg_mse",
        "rmse": "avg_rmse",
        "mae": "avg_mae",
    }
    direct_alias = direct_aliases.get(normalized)
    if direct_alias is not None and direct_alias in metrics:
        return float(metrics[direct_alias])
    current: Any = metrics
    for part in normalized.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            aliases = {
                "val_r": "avg_pearson",
                "pearson": "avg_pearson",
                "mse": "avg_mse",
                "rmse": "avg_rmse",
                "mae": "avg_mae",
            }
            alias = aliases.get(part)
            if alias and alias in metrics:
                current = metrics[alias]
                continue
            raise KeyError(f"Metric path '{path}' is not present")
    return float(current)


def _is_maximize(direction: str) -> bool:
    normalized = direction.lower()
    if normalized not in {"maximize", "minimize", "max", "min"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")
    return normalized in {"maximize", "max"}


def _improved(
    score: float,
    best_score: float,
    maximize: bool,
    min_delta: float,
) -> bool:
    if not np.isfinite(score):
        return False
    return (
        score > best_score + min_delta
        if maximize
        else score < best_score - min_delta
    )
