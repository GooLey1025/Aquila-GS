# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/GSBreeder/BNNs

"""Bayesian MLP and training primitives for the BNN benchmark."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass(frozen=True)
class BNNFitResult:
    """Selected posterior state, epoch, validation metrics, and history."""

    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_metric: float
    metrics: dict[str, Any]
    history: tuple[dict[str, float], ...]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for one independent fit."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ScaleMixturePrior:
    """Two-component zero-mean Gaussian scale-mixture prior."""

    def __init__(self, sigma1: float, sigma2: float, pi: float) -> None:
        if sigma1 <= 0 or sigma2 <= 0:
            raise ValueError("Prior standard deviations must be positive")
        if not 0 < pi < 1:
            raise ValueError("Prior mixture probability must be in (0, 1)")
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.pi = float(pi)

    def log_prob(self, values: torch.Tensor) -> torch.Tensor:
        first = Normal(
            torch.zeros((), device=values.device, dtype=values.dtype),
            torch.as_tensor(self.sigma1, device=values.device, dtype=values.dtype),
        ).log_prob(values)
        second = Normal(
            torch.zeros((), device=values.device, dtype=values.dtype),
            torch.as_tensor(self.sigma2, device=values.device, dtype=values.dtype),
        ).log_prob(values)
        mixture = torch.logaddexp(
            first + math.log(self.pi),
            second + math.log1p(-self.pi),
        )
        return mixture.sum()


class BayesLinear(nn.Module):
    """Factorized Gaussian Bayesian linear layer."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        prior: ScaleMixturePrior,
    ) -> None:
        super().__init__()
        self.weight_mu = nn.Parameter(
            torch.empty(output_dim, input_dim).uniform_(-0.2, 0.2)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(output_dim, input_dim).uniform_(-5.0, -4.0)
        )
        self.bias_mu = nn.Parameter(torch.empty(output_dim).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.empty(output_dim).uniform_(-5.0, -4.0))
        self.prior = prior

    @staticmethod
    def _sample(mu: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.nn.functional.softplus(rho)
        return mu + sigma * torch.randn_like(mu), sigma

    @staticmethod
    def _posterior_log_prob(
        values: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        return Normal(mu, sigma).log_prob(values).sum()

    def forward(
        self,
        values: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample:
            weight, weight_sigma = self._sample(self.weight_mu, self.weight_rho)
            bias, bias_sigma = self._sample(self.bias_mu, self.bias_rho)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            weight_sigma = torch.nn.functional.softplus(self.weight_rho)
            bias_sigma = torch.nn.functional.softplus(self.bias_rho)
        output = torch.nn.functional.linear(values, weight, bias)
        if not sample:
            zero = output.new_zeros(())
            return output, zero, zero
        log_prior = self.prior.log_prob(weight) + self.prior.log_prob(bias)
        log_posterior = self._posterior_log_prob(
            weight, self.weight_mu, weight_sigma
        ) + self._posterior_log_prob(bias, self.bias_mu, bias_sigma)
        return output, log_prior, log_posterior


class BayesMLP(nn.Module):
    """Bayesian multilayer perceptron preserved from the upstream design."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        sigma1: float,
        sigma2: float,
        pi: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        prior = ScaleMixturePrior(sigma1, sigma2, pi)
        dimensions = [input_dim, *(int(value) for value in hidden_dims), 1]
        self.layers = nn.ModuleList(
            BayesLinear(dimensions[index], dimensions[index + 1], prior)
            for index in range(len(dimensions) - 1)
        )
        activations = {
            "relu": torch.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported BNN activation: {activation}")
        self.activation = activations[activation]

    def forward_once(
        self,
        values: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_prior = values.new_zeros(())
        log_posterior = values.new_zeros(())
        hidden = values
        for index, layer in enumerate(self.layers):
            hidden, layer_prior, layer_posterior = layer(hidden, sample=sample)
            log_prior = log_prior + layer_prior
            log_posterior = log_posterior + layer_posterior
            if index + 1 < len(self.layers):
                hidden = self.activation(hidden)
        return hidden, log_prior, log_posterior

    def sample_predictions(
        self,
        values: torch.Tensor,
        sample_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        predictions = []
        log_priors = []
        log_posteriors = []
        for _ in range(sample_count):
            prediction, log_prior, log_posterior = self.forward_once(values, True)
            predictions.append(prediction)
            log_priors.append(log_prior)
            log_posteriors.append(log_posterior)
        return (
            torch.stack(predictions),
            torch.stack(log_priors).mean(),
            torch.stack(log_posteriors).mean(),
        )


def build_model(
    input_dim: int,
    config: Mapping[str, Any],
    device: torch.device,
) -> BayesMLP:
    """Construct one configured BNN on the requested device."""

    model_config = config["model"]
    prior_config = config["prior"]
    return BayesMLP(
        input_dim=input_dim,
        hidden_dims=model_config["hidden_dims"],
        sigma1=float(prior_config["sigma1"]),
        sigma2=float(prior_config["sigma2"]),
        pi=float(prior_config["pi"]),
        activation=str(model_config.get("activation", "relu")),
    ).to(device)


def elbo_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    log_prior: torch.Tensor,
    log_posterior: torch.Tensor,
    noise_std: float,
    kl_weight: float,
) -> torch.Tensor:
    """Negative evidence lower bound for Gaussian regression."""

    likelihood = Normal(predictions, float(noise_std)).log_prob(targets).sum(dim=(1, 2))
    return float(kl_weight) * (log_posterior - log_prior) - likelihood.mean()


def predict_bnn(
    model: BayesMLP,
    features: np.ndarray,
    device: torch.device,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and standard deviation with deterministic MC draws."""

    set_seed(seed)
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        draws, _, _ = model.sample_predictions(tensor, sample_count)
    values = draws.squeeze(-1).cpu().numpy()
    return values.mean(axis=0).astype(np.float32), values.std(axis=0).astype(np.float32)


def train_bnn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray | None,
    valid_y: np.ndarray | None,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    evaluator: Any,
    trait_name: str,
    fixed_epochs: int | None = None,
) -> BNNFitResult:
    """Fit one candidate using validation Pearson or a fixed final epoch count."""

    set_seed(seed)
    model = build_model(train_x.shape[1], config, device)
    train_config = config["train"]
    inference_config = config["inference"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    features = torch.from_numpy(np.asarray(train_x, dtype=np.float32)).to(device)
    targets = torch.from_numpy(np.asarray(train_y, dtype=np.float32)).reshape(-1, 1).to(
        device
    )
    max_epochs = (
        int(fixed_epochs)
        if fixed_epochs is not None
        else int(train_config["max_epochs"])
    )
    patience = int(train_config["patience"])
    min_delta = float(train_config.get("min_delta", 0.0))
    train_samples = int(train_config.get("mc_samples", 1))
    validation_samples = int(inference_config["validation_samples"])
    noise_std = float(train_config["noise_std"])
    kl_weight = float(train_config.get("kl_weight", 1.0))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_metric = -float("inf")
    best_metrics: dict[str, Any] = {}
    history = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        draws, log_prior, log_posterior = model.sample_predictions(
            features, train_samples
        )
        loss = elbo_loss(
            draws,
            targets.unsqueeze(0).expand_as(draws),
            log_prior,
            log_posterior,
            noise_std,
            kl_weight,
        )
        loss.backward()
        gradient_clip = float(train_config.get("gradient_clip", 0.0))
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        record = {"epoch": epoch, "train_loss": float(loss.detach())}
        if valid_x is not None and valid_y is not None:
            predictions, _ = predict_bnn(
                model,
                valid_x,
                device,
                validation_samples,
                seed + epoch,
            )
            metrics = evaluator(
                predictions[:, None],
                np.asarray(valid_y, dtype=np.float32)[:, None],
                np.ones((len(valid_y), 1), dtype=bool),
                [trait_name],
            ).metrics
            metric = float(metrics["avg_pearson"])
            record["valid_pearson"] = metric
            if np.isfinite(metric) and metric > best_metric + min_delta:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_metric = metric
                best_metrics = metrics
                stale = 0
            else:
                stale += 1
            if fixed_epochs is None and stale >= patience:
                history.append(record)
                break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        history.append(record)
    if valid_x is None or valid_y is None:
        best_metric = float("nan")
    return BNNFitResult(
        state_dict=best_state,
        best_epoch=best_epoch,
        best_metric=best_metric,
        metrics=best_metrics,
        history=tuple(history),
    )
