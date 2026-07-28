#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/ganlab/MENET

"""Leakage-safe nested cross-validation training for MENET benchmarks."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
MENET_DIRECTORY = SCRIPT_DIRECTORY.parent
PROJECT_ROOT = MENET_DIRECTORY.parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_path in (
    str(MENET_DIRECTORY),
    str(SOURCE_ROOT),
    str(SCRIPT_DIRECTORY),
):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from aquila.data import load_prepared_data
from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.training.distributed import derive_seed
from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import (
    CandidateResult,
    HPOResult,
    InnerFoldResult,
    generate_grid_candidates,
    half_up_median_epoch,
    merge_config,
)
from src_benchmark.loss_benchmark import TripletLossBenchmark
from src_benchmark.network.menet_benchmark import MeNetBenchmark
from src_benchmark.network.trait_encoder_benchmark import (
    TraitSpecificEncoderBenchmark,
)


@dataclass(frozen=True)
class VCFGenotypes:
    """MENET scalar genotypes and their VCF schema."""

    genotypes: torch.Tensor
    sample_ids: tuple[str, ...]
    variants: tuple[tuple[str, str, str, str, str], ...]


@dataclass(frozen=True)
class SplitData:
    """One observed-trait split aligned to a fold-specific VCF."""

    genotypes: torch.Tensor
    targets: torch.Tensor
    sample_ids: tuple[str, ...]
    discarded_sample_ids: tuple[str, ...]
    variants: tuple[tuple[str, str, str, str, str], ...]


@dataclass(frozen=True)
class EncoderResult:
    """Selected trait-encoder state and validation history."""

    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_valid_loss: float
    history: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class MeNetResult:
    """Selected MENET state and validation predictions."""

    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    metrics: dict[str, Any]
    history: tuple[dict[str, float], ...]


class DeterministicTripletDataset(Dataset):
    """Fixed triplets that never resample during validation or training."""

    def __init__(
        self,
        genotypes: torch.Tensor,
        targets: torch.Tensor,
        seed: int,
    ) -> None:
        if genotypes.ndim != 2:
            raise ValueError("Triplet genotypes must be [samples, markers]")
        if targets.ndim != 1 or len(targets) != len(genotypes):
            raise ValueError("Triplet targets must align with genotypes")
        if len(genotypes) < 3:
            raise ValueError("At least three observed samples are required")
        generator = np.random.default_rng(seed)
        triplets = []
        for anchor in range(len(genotypes)):
            candidates = np.delete(np.arange(len(genotypes)), anchor)
            selected = generator.choice(candidates, size=2, replace=False)
            triplets.append((anchor, int(selected[0]), int(selected[1])))
        self.genotypes = genotypes
        self.targets = targets
        self.triplets = tuple(triplets)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        anchor, first, second = self.triplets[index]
        return (
            self.genotypes[anchor].unsqueeze(0),
            self.targets[anchor].reshape(1),
            self.genotypes[first].unsqueeze(0),
            self.targets[first].reshape(1),
            self.genotypes[second].unsqueeze(0),
            self.targets[second].reshape(1),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe nested CV for single-trait MENET models."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--traits", nargs="+", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument(
        "--outer-folds",
        nargs="+",
        type=int,
        default=None,
        help="Zero-based outer folds; defaults to all folds.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Testing only: limit the 32-point grid.",
    )
    parser.add_argument(
        "--max-inner-folds",
        type=int,
        default=None,
        help="Testing only: limit inner folds.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _open_vcf(path: Path):
    return gzip.open(path, "rt", encoding="utf-8")


def _encode_gt(sample_field: str, gt_index: int, missing_value: float) -> float:
    fields = sample_field.split(":")
    if gt_index >= len(fields):
        return missing_value
    genotype = fields[gt_index].replace("|", "/")
    alleles = genotype.split("/")
    if len(alleles) != 2 or "." in alleles:
        return missing_value
    try:
        values = [int(allele) for allele in alleles]
    except ValueError:
        return missing_value
    if any(allele not in {0, 1} for allele in values):
        raise ValueError("MENET benchmark requires biallelic diploid genotypes")
    return float(sum(values) - 1)


def load_vcf_genotypes(
    path: str | Path,
    missing_value: float = 0.0,
) -> VCFGenotypes:
    """Convert VCF GT values to MENET's {-1, 0, 1} marker convention."""
    vcf_path = Path(path)
    sample_ids: tuple[str, ...] | None = None
    variants = []
    marker_rows = []
    with _open_vcf(vcf_path) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                columns = line.rstrip("\n").split("\t")
                sample_ids = tuple(columns[9:])
                if not sample_ids or len(set(sample_ids)) != len(sample_ids):
                    raise ValueError(f"Invalid VCF sample header: {vcf_path}")
                continue
            if line.startswith("#"):
                continue
            if sample_ids is None:
                raise ValueError(f"VCF has no #CHROM header: {vcf_path}")
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9 + len(sample_ids):
                raise ValueError(f"VCF sample count mismatch: {vcf_path}")
            alternate = columns[4]
            if "," in alternate:
                raise ValueError("MENET benchmark requires biallelic variants")
            format_fields = columns[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record lacks GT: {columns[0]}:{columns[1]}")
            gt_index = format_fields.index("GT")
            variants.append(
                (columns[0], columns[1], columns[2], columns[3], alternate)
            )
            marker_rows.append(
                [
                    _encode_gt(field, gt_index, missing_value)
                    for field in columns[9:]
                ]
            )
    if sample_ids is None or not marker_rows:
        raise ValueError(f"VCF contains no genotype records: {vcf_path}")
    matrix = torch.tensor(marker_rows, dtype=torch.float32).transpose(0, 1)
    return VCFGenotypes(matrix.contiguous(), sample_ids, tuple(variants))


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_split_data(
    data_directory: Path,
    metadata: Mapping[str, Any],
    target_mask: torch.Tensor,
    trait_index: int,
    outer_fold: int,
    inner_fold: int | None,
    role: str,
    missing_genotype: float,
) -> SplitData:
    """Align one VCF split to cached processed targets and discard missing labels."""
    outer_path = data_directory / "cv" / f"outer_fold_{outer_fold}"
    raw_path = data_directory / "raw_genotype" / f"outer_fold_{outer_fold}"
    valid_roles = {"train", "test"} if inner_fold is None else {"train", "valid"}
    if role not in valid_roles:
        raise ValueError(f"Invalid split role {role!r}")
    if inner_fold is None:
        split_path = outer_path / "final"
        index_path = outer_path / f"{role}_idx.npy"
        target_name = f"Y_{role}_processed.pt"
        vcf_path = raw_path / f"{role}.vcf.gz"
    else:
        split_path = outer_path / f"inner_fold_{inner_fold}"
        index_path = split_path / f"{role}_idx.npy"
        target_name = f"Y_{role}_processed.pt"
        vcf_path = (
            raw_path / f"inner_fold_{inner_fold}" / f"{role}.vcf.gz"
        )
    absolute_indices = np.load(index_path, allow_pickle=False)
    processed = _load_tensor(split_path / target_name)
    if len(absolute_indices) != len(processed):
        raise ValueError(f"Processed targets do not align with {index_path}")

    sample_ids = metadata["sample_ids"]
    by_sample = {
        str(sample_ids[int(index)]): (
            float(processed[position, trait_index]),
            bool(target_mask[int(index), trait_index]),
        )
        for position, index in enumerate(absolute_indices)
    }
    vcf = load_vcf_genotypes(vcf_path, missing_genotype)
    if set(vcf.sample_ids) != set(by_sample):
        raise ValueError(f"VCF samples do not match fold indices: {vcf_path}")
    observed_positions = [
        position
        for position, sample_id in enumerate(vcf.sample_ids)
        if by_sample[sample_id][1]
    ]
    discarded = tuple(
        sample_id
        for sample_id in vcf.sample_ids
        if not by_sample[sample_id][1]
    )
    if len(observed_positions) < 3:
        raise ValueError(
            f"Trait has fewer than three observed samples in {vcf_path}"
        )
    targets = torch.tensor(
        [by_sample[vcf.sample_ids[position]][0] for position in observed_positions],
        dtype=torch.float32,
    )
    if not torch.isfinite(targets).all() or torch.any(targets == -999):
        raise ValueError("Missing phenotype sentinel entered MENET targets")
    observed_ids = tuple(vcf.sample_ids[position] for position in observed_positions)
    expected_observed_order = tuple(
        sample_id for sample_id in vcf.sample_ids if by_sample[sample_id][1]
    )
    if observed_ids != expected_observed_order:
        raise RuntimeError("Observed VCF sample order changed unexpectedly")
    return SplitData(
        genotypes=vcf.genotypes[observed_positions].contiguous(),
        targets=targets,
        sample_ids=observed_ids,
        discarded_sample_ids=discarded,
        variants=vcf.variants,
    )


def validate_variant_schema(first: SplitData, second: SplitData) -> None:
    if first.variants != second.variants:
        raise ValueError("Training and held-out VCF variant schemas differ")


def _triplet_loss(
    model: TraitSpecificEncoderBenchmark,
    batch: Sequence[torch.Tensor],
    criterion: TripletLossBenchmark,
    device: torch.device,
) -> torch.Tensor:
    values = [value.to(device, non_blocking=True) for value in batch]
    anchor, anchor_y, first, first_y, second, second_y = values
    anchor_z, first_z, second_z = model(anchor, first, second)
    return criterion(
        anchor_z,
        anchor_y,
        first_z,
        first_y,
        second_z,
        second_y,
    )


def _evaluate_encoder(
    model: TraitSpecificEncoderBenchmark,
    loader: DataLoader,
    criterion: TripletLossBenchmark,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            losses.append(float(_triplet_loss(model, batch, criterion, device)))
    return float(np.mean(losses))


def train_encoder(
    train: SplitData,
    valid: SplitData | None,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    fixed_epochs: int | None = None,
) -> EncoderResult:
    """Train the phenotype-supervised encoder on observed training labels only."""
    set_seed(seed)
    encoder_config = config["encoder"]
    model = TraitSpecificEncoderBenchmark(
        train.genotypes.shape[1],
        int(encoder_config["stride"]),
        int(encoder_config["embedding_dim"]),
    ).to(device)
    criterion = TripletLossBenchmark(
        float(encoder_config["margin"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(encoder_config["learning_rate"]),
        weight_decay=float(encoder_config.get("weight_decay", 0.0)),
    )
    train_dataset = DeterministicTripletDataset(
        train.genotypes,
        train.targets,
        seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(encoder_config["batch_size"]),
        shuffle=True,
        drop_last=(
            len(train_dataset) > 1
            and len(train_dataset) % int(encoder_config["batch_size"]) == 1
        ),
    )
    valid_loader = None
    if valid is not None:
        valid_loader = DataLoader(
            DeterministicTripletDataset(
                valid.genotypes,
                valid.targets,
                seed + 1,
            ),
            batch_size=int(encoder_config["batch_size"]),
            shuffle=False,
        )

    epoch_count = int(fixed_epochs or encoder_config["max_epochs"])
    patience = int(encoder_config.get("patience", epoch_count))
    best_loss = float("inf")
    best_epoch = epoch_count
    best_state = copy.deepcopy(model.state_dict())
    history = []
    stale_epochs = 0
    for epoch in range(1, epoch_count + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _triplet_loss(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
        valid_loss = (
            _evaluate_encoder(model, valid_loader, criterion, device)
            if valid_loader is not None
            else float("nan")
        )
        history.append({"epoch": epoch, "valid_loss": valid_loss})
        if valid_loader is None:
            best_state = copy.deepcopy(model.state_dict())
            continue
        if valid_loss < best_loss - float(encoder_config.get("min_delta", 0.0)):
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return EncoderResult(
        state_dict={key: value.cpu() for key, value in best_state.items()},
        best_epoch=best_epoch,
        best_valid_loss=best_loss,
        history=tuple(history),
    )


def build_relatedness(
    train: SplitData,
    held_out: SplitData,
    encoder_config: Mapping[str, Any],
    encoder_state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Represent both partitions against a training-only embedding bank."""
    validate_variant_schema(train, held_out)
    model = TraitSpecificEncoderBenchmark(
        train.genotypes.shape[1],
        int(encoder_config["stride"]),
        int(encoder_config["embedding_dim"]),
    ).to(device)
    model.load_state_dict(encoder_state)
    model.eval()

    def embed(values: torch.Tensor) -> torch.Tensor:
        outputs = []
        with torch.no_grad():
            for batch in values.split(256):
                encoded = model.forward_once(batch.unsqueeze(1).to(device))
                outputs.append(nn.functional.normalize(encoded, dim=1).cpu())
        return torch.cat(outputs)

    train_embeddings = embed(train.genotypes)
    held_out_embeddings = embed(held_out.genotypes)
    train_distances = torch.cdist(train_embeddings, train_embeddings)
    maximum = float(train_distances.max())
    scale = maximum if maximum > 0 else 1.0
    train_relatedness = 1.0 - train_distances / scale
    held_out_relatedness = (
        1.0 - torch.cdist(held_out_embeddings, train_embeddings) / scale
    )
    return train_relatedness, held_out_relatedness, scale


def _menet_loader(
    split: SplitData,
    relatedness: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        split.genotypes.unsqueeze(1),
        relatedness.unsqueeze(1),
        split.targets.unsqueeze(1),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(dataset) % batch_size == 1,
    )


def _predict_menet(
    model: MeNetBenchmark,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions = []
    targets = []
    losses = []
    with torch.no_grad():
        for genotype, relatedness, target in loader:
            prediction = model(
                genotype.to(device, non_blocking=True),
                relatedness.to(device, non_blocking=True),
            )
            target_device = target.to(device, non_blocking=True)
            losses.append(float(nn.functional.l1_loss(prediction, target_device)))
            predictions.append(prediction.cpu())
            targets.append(target)
    return (
        torch.cat(predictions).numpy(),
        torch.cat(targets).numpy(),
        float(np.mean(losses)),
    )


def train_menet(
    train: SplitData,
    valid: SplitData | None,
    train_relatedness: torch.Tensor,
    valid_relatedness: torch.Tensor | None,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
    fixed_epochs: int | None = None,
) -> MeNetResult:
    """Train MENET and select epochs exclusively from inner validation."""
    set_seed(seed)
    train_config = config["train"]
    model = MeNetBenchmark(
        train.genotypes.shape[1],
        train_relatedness.shape[1],
        config["model"],
        float(train_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    batch_size = int(train_config["batch_size"])
    train_loader = _menet_loader(
        train,
        train_relatedness,
        batch_size,
        True,
    )
    valid_loader = (
        _menet_loader(valid, valid_relatedness, batch_size, False)
        if valid is not None and valid_relatedness is not None
        else None
    )
    epoch_count = int(fixed_epochs or train_config["max_epochs"])
    patience = int(train_config.get("patience", epoch_count))
    best_metric = -float("inf")
    best_epoch = epoch_count
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict[str, Any] = {}
    history = []
    stale_epochs = 0
    for epoch in range(1, epoch_count + 1):
        model.train()
        for genotype, relatedness, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                genotype.to(device, non_blocking=True),
                relatedness.to(device, non_blocking=True),
            )
            loss = nn.functional.l1_loss(
                prediction,
                target.to(device, non_blocking=True),
            )
            loss.backward()
            optimizer.step()
        if valid_loader is None:
            best_state = copy.deepcopy(model.state_dict())
            continue
        predictions, targets, valid_loss = _predict_menet(
            model,
            valid_loader,
            device,
        )
        metrics = evaluate_regression(
            predictions,
            targets,
            np.ones_like(targets, dtype=bool),
            ["trait"],
        ).metrics
        pearson = float(metrics["avg_pearson"])
        history.append(
            {"epoch": epoch, "valid_loss": valid_loss, "valid_pearson": pearson}
        )
        if math.isfinite(pearson) and pearson > best_metric + float(
            train_config.get("min_delta", 0.0)
        ):
            best_metric = pearson
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return MeNetResult(
        state_dict={key: value.cpu() for key, value in best_state.items()},
        best_epoch=best_epoch,
        metrics=best_metrics,
        history=tuple(history),
    )


def evaluate_model(
    train: SplitData,
    test: SplitData,
    train_relatedness: torch.Tensor,
    test_relatedness: torch.Tensor,
    config: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device,
    trait_name: str,
) -> tuple[dict[str, Any], np.ndarray]:
    model = MeNetBenchmark(
        train.genotypes.shape[1],
        train_relatedness.shape[1],
        config["model"],
        float(config["train"]["dropout"]),
    ).to(device)
    model.load_state_dict(state_dict)
    loader = _menet_loader(
        test,
        test_relatedness,
        int(config["train"]["batch_size"]),
        False,
    )
    predictions, targets, _ = _predict_menet(model, loader, device)
    result = evaluate_regression(
        predictions,
        targets,
        np.ones_like(targets, dtype=bool),
        [trait_name],
    )
    return result.metrics, result.predictions


def _candidate_payload(candidate: CandidateResult) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "parameters": dict(candidate.parameters),
        "objective": candidate.objective,
        "best_epochs": list(candidate.best_epochs),
        "final_epoch": candidate.final_epoch,
        "inner_results": [
            {
                "inner_fold": result.inner_fold,
                "metric": result.metric,
                "best_epoch": result.best_epoch,
                "metrics": dict(result.metrics),
            }
            for result in candidate.inner_results
        ],
    }


def _select_candidate(
    candidates: Sequence[CandidateResult],
) -> CandidateResult:
    finite = [candidate for candidate in candidates if math.isfinite(candidate.objective)]
    if not finite:
        raise ValueError("All MENET grid candidates produced non-finite objectives")
    return max(finite, key=lambda candidate: (candidate.objective, -candidate.candidate_id))


def _inverse_trait(
    values: np.ndarray,
    preprocessor: PerTraitPreprocessor,
    trait_index: int,
) -> np.ndarray:
    params = preprocessor.traits[trait_index]
    restored = np.asarray(values, dtype=np.float64) * params.std + params.mean
    if params.use_log1p:
        restored = np.expm1(restored) - params.log_shift
    return restored.astype(np.float32)


def run_outer_fold(
    data_directory: Path,
    output_directory: Path,
    prepared: Any,
    trait_name: str,
    trait_index: int,
    outer_fold: int,
    base_config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    inner_count: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    fold_start = time.time()
    fold_path = output_directory / trait_name / f"fold_{outer_fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    inner_results_by_candidate: dict[int, list[InnerFoldResult]] = {
        index: [] for index in range(len(candidates))
    }
    encoder_epoch_by_candidate: dict[int, list[int]] = {
        index: [] for index in range(len(candidates))
    }
    split_audit = []
    candidate_configs = [
        merge_config(base_config, parameters) for parameters in candidates
    ]
    encoder_keys = sorted(
        {
            (
                float(config["encoder"]["learning_rate"]),
                int(config["encoder"]["embedding_dim"]),
            )
            for config in candidate_configs
        }
    )
    encoder_group_ids = {
        encoder_key: group_id
        for group_id, encoder_key in enumerate(encoder_keys)
    }
    missing_value = float(base_config["data"]["missing_genotype_value"])

    for inner_fold in range(inner_count):
        train = load_split_data(
            data_directory,
            prepared.metadata,
            prepared.target_mask,
            trait_index,
            outer_fold,
            inner_fold,
            "train",
            missing_value,
        )
        valid = load_split_data(
            data_directory,
            prepared.metadata,
            prepared.target_mask,
            trait_index,
            outer_fold,
            inner_fold,
            "valid",
            missing_value,
        )
        validate_variant_schema(train, valid)
        split_audit.append(
            {
                "inner_fold": inner_fold,
                "train_observed": len(train.sample_ids),
                "train_discarded": list(train.discarded_sample_ids),
                "valid_observed": len(valid.sample_ids),
                "valid_discarded": list(valid.discarded_sample_ids),
            }
        )
        encoder_cache: dict[
            tuple[float, int],
            tuple[EncoderResult, torch.Tensor, torch.Tensor],
        ] = {}
        for candidate_id, parameters in enumerate(candidates):
            config = candidate_configs[candidate_id]
            encoder_key = (
                float(config["encoder"]["learning_rate"]),
                int(config["encoder"]["embedding_dim"]),
            )
            if encoder_key not in encoder_cache:
                encoder_seed = derive_seed(
                    seed,
                    outer_fold,
                    encoder_group_ids[encoder_key],
                    inner_fold,
                )
                encoder = train_encoder(
                    train,
                    valid,
                    config,
                    device,
                    encoder_seed,
                )
                train_rep, valid_rep, _ = build_relatedness(
                    train,
                    valid,
                    config["encoder"],
                    encoder.state_dict,
                    device,
                )
                encoder_cache[encoder_key] = (encoder, train_rep, valid_rep)
            encoder, train_rep, valid_rep = encoder_cache[encoder_key]
            menet_seed = derive_seed(
                seed + 1,
                outer_fold,
                candidate_id,
                inner_fold,
            )
            result = train_menet(
                train,
                valid,
                train_rep,
                valid_rep,
                config,
                device,
                menet_seed,
            )
            metric = float(result.metrics.get("avg_pearson", float("nan")))
            inner_results_by_candidate[candidate_id].append(
                InnerFoldResult(
                    inner_fold=inner_fold,
                    metric=metric,
                    best_epoch=result.best_epoch,
                    metrics=result.metrics,
                )
            )
            encoder_epoch_by_candidate[candidate_id].append(encoder.best_epoch)

    candidate_results = []
    for candidate_id, parameters in enumerate(candidates):
        results = tuple(inner_results_by_candidate[candidate_id])
        objective_values = np.asarray([result.metric for result in results])
        objective = (
            float(objective_values.mean())
            if np.isfinite(objective_values).all()
            else float("nan")
        )
        candidate_results.append(
            CandidateResult(
                candidate_id,
                dict(parameters),
                objective,
                results,
            )
        )
    best = _select_candidate(candidate_results)
    hpo = HPOResult(best, tuple(candidate_results), "maximize", "grid")
    best_config = merge_config(base_config, best.parameters)
    final_encoder_epoch = half_up_median_epoch(
        encoder_epoch_by_candidate[best.candidate_id]
    )
    final_menet_epoch = best.final_epoch

    outer_train = load_split_data(
        data_directory,
        prepared.metadata,
        prepared.target_mask,
        trait_index,
        outer_fold,
        None,
        "train",
        missing_value,
    )
    outer_test = load_split_data(
        data_directory,
        prepared.metadata,
        prepared.target_mask,
        trait_index,
        outer_fold,
        None,
        "test",
        missing_value,
    )
    validate_variant_schema(outer_train, outer_test)
    final_encoder = train_encoder(
        outer_train,
        None,
        best_config,
        device,
        derive_seed(seed, outer_fold, best.candidate_id, 999),
        fixed_epochs=final_encoder_epoch,
    )
    train_rep, test_rep, relation_scale = build_relatedness(
        outer_train,
        outer_test,
        best_config["encoder"],
        final_encoder.state_dict,
        device,
    )
    final_menet = train_menet(
        outer_train,
        None,
        train_rep,
        None,
        best_config,
        device,
        derive_seed(seed + 1, outer_fold, best.candidate_id, 999),
        fixed_epochs=final_menet_epoch,
    )
    normalized_metrics, normalized_predictions = evaluate_model(
        outer_train,
        outer_test,
        train_rep,
        test_rep,
        best_config,
        final_menet.state_dict,
        device,
        trait_name,
    )
    preprocessor = PerTraitPreprocessor.load_json(
        data_directory
        / "cv"
        / f"outer_fold_{outer_fold}"
        / "final"
        / "preprocessing.json"
    )
    original_predictions = _inverse_trait(
        normalized_predictions,
        preprocessor,
        trait_index,
    )
    original_targets = _inverse_trait(
        outer_test.targets.numpy()[:, None],
        preprocessor,
        trait_index,
    )
    original_metrics = evaluate_regression(
        original_predictions,
        original_targets,
        np.ones_like(original_targets, dtype=bool),
        [trait_name],
    ).metrics

    torch.save(
        {
            "encoder_state_dict": final_encoder.state_dict,
            "menet_state_dict": final_menet.state_dict,
            "reference_sample_ids": outer_train.sample_ids,
            "relation_scale": relation_scale,
            "variant_schema": outer_train.variants,
            "config": best_config,
            "trait": trait_name,
            "outer_fold": outer_fold,
            "encoder_epoch": final_encoder_epoch,
            "menet_epoch": final_menet_epoch,
        },
        fold_path / "best_model.pt",
    )
    with (fold_path / "hpo_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": hpo.method,
                "direction": hpo.direction,
                "best_candidate_id": best.candidate_id,
                "best_parameters": dict(best.parameters),
                "best_valid_pearson_mean": best.objective,
                "final_encoder_epoch": final_encoder_epoch,
                "final_menet_epoch": final_menet_epoch,
                "candidates": [
                    _candidate_payload(candidate)
                    for candidate in candidate_results
                ],
            },
            handle,
            indent=2,
            allow_nan=True,
        )
    with (fold_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"normalized": normalized_metrics, "original": original_metrics},
            handle,
            indent=2,
            allow_nan=True,
        )
    with (fold_path / "training_history.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "final_encoder": list(final_encoder.history),
                "final_menet": list(final_menet.history),
            },
            handle,
            indent=2,
            allow_nan=True,
        )
    with (fold_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(best_config, handle, sort_keys=False)
    preprocessor.save_json(fold_path / "preprocessing.json")
    with (fold_path / "sample_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "outer_train_observed": len(outer_train.sample_ids),
                "outer_train_discarded": list(outer_train.discarded_sample_ids),
                "outer_test_observed": len(outer_test.sample_ids),
                "outer_test_sample_ids": list(outer_test.sample_ids),
                "outer_test_discarded": list(outer_test.discarded_sample_ids),
                "reference_sample_ids": list(outer_train.sample_ids),
                "inner_folds": split_audit,
            },
            handle,
            indent=2,
        )
    with (fold_path / "predictions_original_scale.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["SampleID", "Prediction", "Observed"])
        for sample_id, prediction, observed in zip(
            outer_test.sample_ids,
            original_predictions[:, 0],
            original_targets[:, 0],
        ):
            writer.writerow([sample_id, float(prediction), float(observed)])
    return {
        "outer_fold": outer_fold,
        "best_candidate_id": best.candidate_id,
        "best_parameters": dict(best.parameters),
        "best_valid_pearson_mean": best.objective,
        "test_pearson": normalized_metrics["avg_pearson"],
        "elapsed_seconds": time.time() - fold_start,
    }


def main() -> None:
    args = parse_args()
    data_directory = Path(args.data_dir).resolve()
    output_directory = Path(args.output_dir).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_directory}"
            )
        import shutil

        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    prepared = load_prepared_data(data_directory)
    trait_names = list(prepared.metadata["trait_names"])
    invalid_traits = [trait for trait in args.traits if trait not in trait_names]
    if invalid_traits:
        raise ValueError(f"Unknown traits: {invalid_traits}")
    outer_count = int(prepared.metadata["outer_folds"])
    inner_count = int(prepared.metadata["inner_folds"])
    outer_folds = args.outer_folds or list(range(outer_count))
    if any(fold < 0 or fold >= outer_count for fold in outer_folds):
        raise ValueError(f"Outer folds must be in 0..{outer_count - 1}")
    if args.max_inner_folds is not None:
        inner_count = min(inner_count, args.max_inner_folds)
    candidates = generate_grid_candidates(config["hpo"]["parameters"])
    if len(candidates) != 32:
        raise ValueError(f"MENET grid must contain 32 candidates, got {len(candidates)}")
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    summaries = []
    for trait_name in args.traits:
        trait_index = trait_names.index(trait_name)
        for outer_fold in outer_folds:
            print(
                f"[INFO] MENET trait={trait_name} outer_fold={outer_fold} "
                f"candidates={len(candidates)} inner_folds={inner_count}"
            )
            summaries.append(
                {
                    "trait": trait_name,
                    **run_outer_fold(
                        data_directory,
                        output_directory,
                        prepared,
                        trait_name,
                        trait_index,
                        outer_fold,
                        config,
                        candidates,
                        inner_count,
                        device,
                        args.seed,
                    ),
                }
            )
    with (output_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_dir": str(data_directory),
                "config": str(Path(args.config).resolve()),
                "traits": args.traits,
                "outer_folds": outer_folds,
                "results": summaries,
            },
            handle,
            indent=2,
            allow_nan=True,
        )


if __name__ == "__main__":
    main()
