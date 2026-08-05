# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/GSBreeder/BNNs

"""Focused tests for the leakage-safe BNN benchmark integration."""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

TEST_DIRECTORY = Path(__file__).resolve().parent
BENCHMARK_SOURCE = TEST_DIRECTORY.parent
BNN_DIRECTORY = BENCHMARK_SOURCE.parent
PROJECT_ROOT = BENCHMARK_SOURCE.parents[2]
for import_path in (PROJECT_ROOT / "src", BENCHMARK_SOURCE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from aquila.training.evaluator import evaluate_regression
from aquila.training.hpo import CandidateResult, InnerFoldResult, generate_grid_candidates
from bnn_data_benchmark import (
    BNNSplit,
    fit_marker_pipeline,
    load_vcf_dosage,
    prepare_pair,
    transform_markers,
    validate_variant_schema,
)
from bnn_model_benchmark import build_model, predict_bnn

SPEC = importlib.util.spec_from_file_location(
    "bnn_train_cv", BNN_DIRECTORY / "BNNs_train_cv.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_gpu_cli_default_selection_and_cpu() -> None:
    base = ["--data-dir", "data", "-o", "output"]
    default = RUNNER.parse_args(base)
    selected = RUNNER.parse_args([*base, "--gpus", "1", "3"])
    cpu = RUNNER.parse_args([*base, "--gpus"])
    assert default.traits is None
    assert default.gpus is None
    assert default.jobs_per_gpu == 1
    assert selected.gpus == [1, 3]
    assert cpu.gpus == []


def test_gpu_slot_expansion_and_positive_job_count() -> None:
    assert RUNNER.expand_gpu_workers([0, 2], 3) == [0, 0, 0, 2, 2, 2]
    with pytest.raises(SystemExit):
        RUNNER.parse_args(
            [
                "--data-dir",
                "data",
                "--traits",
                "trait",
                "-o",
                "output",
                "--jobs-per-gpu",
                "0",
            ]
        )


def test_vcf_dosage_and_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "tiny.vcf.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n")
        handle.write("1\t10\trs1\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n")
        handle.write("1\t20\trs2\tC\tT\t.\tPASS\t.\tGT:DP\t1/1:8\t./.:0\n")
    loaded = load_vcf_dosage(path)
    np.testing.assert_array_equal(loaded.genotypes[0], np.asarray([0.0, 2.0]))
    assert loaded.genotypes[1, 0] == 1.0
    assert np.isnan(loaded.genotypes[1, 1])
    assert loaded.sample_ids == ("A", "B")


def _split(values: np.ndarray, targets: np.ndarray, prefix: str) -> BNNSplit:
    variants = tuple(
        ("1", str(index), f"v{index}", "A", "G") for index in range(values.shape[1])
    )
    return BNNSplit(
        values.astype(np.float32),
        targets.astype(np.float32),
        targets.astype(np.float32),
        tuple(f"{prefix}{index}" for index in range(len(values))),
        (),
        variants,
    )


def test_marker_pipeline_is_training_only() -> None:
    train = _split(
        np.asarray([[0.0, np.nan, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0]]),
        np.asarray([0.0, 1.0, 2.0]),
        "T",
    )
    held = _split(
        np.asarray([[20.0, np.nan, 1.0], [30.0, 30.0, 0.0]]),
        np.asarray([1.0, 0.0]),
        "V",
    )
    pair = prepare_pair(train, held, alpha=1e-6, max_features=2, seed=42)
    np.testing.assert_allclose(pair.pipeline.imputation_values, [1.0, 1.5, 1 / 3])
    assert np.max(pair.held_out_features) > 1.0
    assert pair.pipeline.selected_indices.size <= 2


def test_pipeline_round_trip_and_schema_validation() -> None:
    values = np.asarray([[0.0, 1.0], [2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    split = _split(values, np.asarray([0.0, 1.0, 2.0]), "T")
    pipeline = fit_marker_pipeline(
        split.genotypes, split.targets, split.variants, 1e-6, 2, 7
    )
    transformed = transform_markers(split.genotypes, pipeline)
    assert transformed.shape[0] == 3
    validate_variant_schema(split, split)
    changed = BNNSplit(
        split.genotypes,
        split.targets,
        split.raw_targets,
        split.sample_ids,
        (),
        tuple(reversed(split.variants)),
    )
    try:
        validate_variant_schema(split, changed)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected variant schema mismatch")


def test_grid_contains_exactly_32_candidates() -> None:
    path = BENCHMARK_SOURCE.parent / "configs" / "BNNs_nested_cv.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    candidates = generate_grid_candidates(config["hpo"]["parameters"])
    assert len(candidates) == 32


def test_candidate_selection_epoch_and_metric_aggregation() -> None:
    first = CandidateResult(
        0,
        {"x": 1},
        0.5,
        (
            InnerFoldResult(0, 0.4, 3, {}),
            InnerFoldResult(1, 0.6, 4, {}),
        ),
    )
    second = CandidateResult(
        1,
        {"x": 2},
        0.5,
        (
            InnerFoldResult(0, 0.5, 8, {}),
            InnerFoldResult(1, 0.5, 10, {}),
        ),
    )
    assert RUNNER.select_candidate([second, first]).candidate_id == 0
    assert first.final_epoch == 4
    metric = evaluate_regression(
        [[1.0], [3.0]], [[2.0], [2.0]], [[True], [True]], ["trait"]
    ).metrics
    results = [
        {
            "trait": "trait",
            "metrics": {"normalized": metric, "original": metric},
        }
        for _ in range(5)
    ]
    summary = RUNNER.aggregate_outer_metrics(results, ["trait"])
    assert summary["trait"]["normalized"]["mse"]["mean"] == 1.0


def test_mc_prediction_is_seed_reproducible() -> None:
    config = {
        "model": {"hidden_dims": [4], "activation": "relu"},
        "prior": {"sigma1": 1.0, "sigma2": 0.001, "pi": 0.5},
    }
    device = torch.device("cpu")
    torch.manual_seed(2)
    model = build_model(3, config, device)
    features = np.ones((4, 3), dtype=np.float32)
    first_mean, first_std = predict_bnn(model, features, device, 5, 42)
    second_mean, second_std = predict_bnn(model, features, device, 5, 42)
    np.testing.assert_array_equal(first_mean, second_mean)
    np.testing.assert_array_equal(first_std, second_std)
    assert np.all(first_std >= 0)
