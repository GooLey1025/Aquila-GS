#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/ganlab/MENET

"""Focused tests for the leakage-safe MENET benchmark integration."""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import torch
import yaml

TEST_DIRECTORY = Path(__file__).resolve().parent
BENCHMARK_SOURCE = TEST_DIRECTORY.parent
MENET_DIRECTORY = BENCHMARK_SOURCE.parent
PROJECT_ROOT = MENET_DIRECTORY.parents[1]
for import_path in (
    str(PROJECT_ROOT / "src"),
    str(BENCHMARK_SOURCE),
    str(MENET_DIRECTORY),
):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from MENET_train_cv import (
    DeterministicTripletDataset,
    SplitData,
    _encode_gt,
    _inverse_trait,
    _menet_loader,
    build_relatedness,
    expand_gpu_workers,
    load_vcf_genotypes,
    parse_args,
    validate_variant_schema,
)
from aquila.data.preprocessing import PerTraitPreprocessor, TraitPreprocessing
from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import generate_grid_candidates
from src_benchmark.network.trait_encoder_benchmark import (
    TraitSpecificEncoderBenchmark,
)


def test_gt_encoding() -> None:
    assert _encode_gt("0/0", 0, 0.0) == -1.0
    assert _encode_gt("0|1", 0, 0.0) == 0.0
    assert _encode_gt("1/1", 0, 0.0) == 1.0
    assert _encode_gt("./.", 0, 0.25) == 0.25
    assert _encode_gt("12:1/0", 1, 0.0) == 0.0


def test_vcf_conversion(tmp_path: Path) -> None:
    path = tmp_path / "tiny.vcf.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
        )
        handle.write("1\t1\trs1\tA\tG\t.\tPASS\t.\tGT\t0/0\t1/1\n")
        handle.write("1\t2\trs2\tC\tT\t.\tPASS\t.\tGT:DP\t0/1:4\t./.:0\n")
    parsed = load_vcf_genotypes(path, missing_value=0.0)
    assert parsed.sample_ids == ("A", "B")
    assert parsed.genotypes.tolist() == [[-1.0, 0.0], [1.0, 0.0]]
    assert len(parsed.variants) == 2


def test_deterministic_triplets() -> None:
    genotype = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    target = torch.arange(5, dtype=torch.float32)
    first = DeterministicTripletDataset(genotype, target, seed=42)
    second = DeterministicTripletDataset(genotype, target, seed=42)
    assert first.triplets == second.triplets
    assert all(len(set(triplet)) == 3 for triplet in first.triplets)


def test_training_only_relatedness() -> None:
    marker_count = 128
    variants = (("1", "1", "rs1", "A", "G"),) * marker_count
    train = SplitData(
        genotypes=torch.randint(
            -1, 2, (5, marker_count), dtype=torch.int64
        ).float(),
        targets=torch.arange(5, dtype=torch.float32),
        sample_ids=tuple(f"T{index}" for index in range(5)),
        discarded_sample_ids=(),
        variants=variants,
    )
    held_out = SplitData(
        genotypes=torch.randint(
            -1, 2, (3, marker_count), dtype=torch.int64
        ).float(),
        targets=torch.arange(3, dtype=torch.float32),
        sample_ids=tuple(f"V{index}" for index in range(3)),
        discarded_sample_ids=(),
        variants=variants,
    )
    validate_variant_schema(train, held_out)
    config = {"stride": 2, "embedding_dim": 8}
    model = TraitSpecificEncoderBenchmark(marker_count, 2, 8)
    train_relation, held_relation, scale = build_relatedness(
        train,
        held_out,
        config,
        model.state_dict(),
        torch.device("cpu"),
    )
    assert train_relation.shape == (5, 5)
    assert held_relation.shape == (3, 5)
    assert scale > 0
    assert torch.allclose(torch.diag(train_relation), torch.ones(5))


def test_grid_budget() -> None:
    config_path = (
        Path(__file__).parent.parent.parent / "configs" / "MeNet_nested_cv.yaml"
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    candidates = generate_grid_candidates(config["hpo"]["parameters"])
    assert len(candidates) == 64


def test_menet_training_loader_discards_incomplete_final_batch() -> None:
    split = SplitData(
        genotypes=torch.zeros(5, 4),
        targets=torch.zeros(5),
        sample_ids=tuple(f"S{index}" for index in range(5)),
        discarded_sample_ids=(),
        variants=(("1", "1", "v", "A", "G"),) * 4,
    )
    relatedness = torch.zeros(5, 5)
    train_loader = _menet_loader(split, relatedness, batch_size=4, shuffle=True)
    valid_loader = _menet_loader(split, relatedness, batch_size=4, shuffle=False)
    assert sum(batch[0].shape[0] for batch in train_loader) == 4
    assert sum(batch[0].shape[0] for batch in valid_loader) == 5


def test_single_trait_inverse_transform() -> None:
    preprocessor = PerTraitPreprocessor()
    preprocessor.traits = [
        TraitPreprocessing(name="A", task="regression", mean=10.0, std=2.0),
        TraitPreprocessing(
            name="B",
            task="regression",
            mean=1.0,
            std=0.5,
            use_log1p=True,
            log_shift=2.0,
        ),
    ]
    restored = _inverse_trait(
        torch.tensor([[0.0], [2.0]]).numpy(),
        preprocessor,
        0,
    )
    assert restored[:, 0].tolist() == [10.0, 14.0]


def test_regression_metrics_include_mse() -> None:
    result = evaluate_regression(
        [[1.0], [3.0]],
        [[2.0], [2.0]],
        [[True], [True]],
        ["trait"],
    )
    metrics = result.metrics
    assert metrics["per_trait"]["trait"]["mse"] == 1.0
    assert metrics["avg_mse"] == 1.0
    assert metrics["avg_rmse"] == 1.0


def test_gpu_cli_defaults_to_detection() -> None:
    arguments = parse_args(
        [
            "--data-dir",
            "data",
            "--config",
            "config",
            "-o",
            "output",
        ]
    )
    assert arguments.gpus is None
    assert arguments.jobs_per_gpu == 1
    assert arguments.traits is None


def test_gpu_cli_supports_selection_and_cpu_fallback() -> None:
    selected = parse_args(
        [
            "--data-dir",
            "data",
            "--config",
            "config",
            "--traits",
            "trait",
            "-o",
            "output",
            "--gpus",
            "0",
            "2",
            "--jobs-per-gpu",
            "3",
        ]
    )
    cpu = parse_args(
        [
            "--data-dir",
            "data",
            "--config",
            "config",
            "--traits",
            "trait",
            "-o",
            "output",
            "--gpus",
        ]
    )
    assert selected.gpus == [0, 2]
    assert selected.jobs_per_gpu == 3
    assert cpu.gpus == []


def test_gpu_worker_slots_allow_multiple_jobs_per_device() -> None:
    assert expand_gpu_workers([0, 2], 3) == [0, 0, 0, 2, 2, 2]
    assert expand_gpu_workers([], 4) == []
