#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/GooLey1025/Aquila-GS

"""Tests for shared R-model nested-CV orchestration."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from benchmark.r_models_common.nested_cv import (
    ModelSpec,
    RWorker,
    SplitData,
    aggregate_results,
    expand_grid,
    impute_from_training,
    load_vcf,
    regression_metrics,
    write_json,
)


def write_vcf(path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        )
        handle.write("1\t1\tv1\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n")
        handle.write("1\t2\tv2\tC\tT\t.\tPASS\t.\tGT\t./.\t1|1\n")


def test_vcf_encoding_and_training_only_imputation(tmp_path: Path) -> None:
    vcf_path = tmp_path / "split.vcf.gz"
    write_vcf(vcf_path)
    loaded = load_vcf(vcf_path)
    assert loaded.sample_ids == ("S1", "S2")
    assert np.allclose(loaded.values[:, 0], [0.0, 1.0])
    assert np.isnan(loaded.values[0, 1])
    train = SplitData(
        np.array([[0.0, np.nan], [2.0, 1.0]]),
        np.array([0.0, 1.0]),
        ("S1", "S2"),
        (),
        loaded.variants,
    )
    held = SplitData(
        np.array([[1.0, np.nan]]),
        np.array([0.5]),
        ("S3",),
        (),
        loaded.variants,
    )
    train_x, held_x, means = impute_from_training(train, held)
    assert np.allclose(means, [1.0, 1.0])
    assert np.allclose(train_x, [[0.0, 1.0], [2.0, 1.0]])
    assert np.allclose(held_x, [[1.0, 1.0]])


def test_explicit_grids() -> None:
    lasso = ModelSpec("Lasso", "", Path("worker.R"), (), ("lambda",))
    elastic = ModelSpec(
        "ElasticNet", "", Path("worker.R"), (), ("alpha", "lambda")
    )
    assert expand_grid(lasso, {"grid": {"lambda": [0.1, 1.0]}}) == [
        {"lambda": 0.1},
        {"lambda": 1.0},
    ]
    candidates = expand_grid(
        elastic,
        {"parameters": {"intercept": True}, "grid": {
            "alpha": [0.25, 0.5],
            "lambda": [0.01, 0.1, 1.0],
        }},
    )
    assert len(candidates) == 6
    assert candidates[0] == {
        "intercept": True,
        "alpha": 0.25,
        "lambda": 0.01,
    }


def test_five_metrics() -> None:
    metrics = regression_metrics(
        np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
    )
    assert set(metrics) == {"pearson", "r2", "mse", "rmse", "mae"}
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_strict_json_and_sample_standard_deviation(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"
    write_json(output, {"finite": 1.0, "missing": float("nan")})
    raw = output.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw)["missing"] is None

    results = [
        {
            "trait": "Trait1",
            "metrics": {
                scale: {
                    metric: float(fold + 1)
                    for metric in ("pearson", "r2", "mse", "rmse", "mae")
                }
                for scale in ("normalized", "original")
            },
        }
        for fold in range(2)
    ]
    aggregate = aggregate_results(results)
    assert aggregate["Trait1"]["normalized"]["pearson"]["std"] == pytest.approx(
        np.std([1.0, 2.0], ddof=1)
    )


def test_mock_r_worker_contract(tmp_path: Path) -> None:
    def runner(command, **kwargs):
        output_dir = Path(command[command.index("--output-dir") + 1])
        prediction_x = np.loadtxt(
            command[command.index("--predict-x") + 1],
            delimiter="\t",
            ndmin=2,
        )
        np.savetxt(
            output_dir / "predictions.tsv",
            np.arange(len(prediction_x), dtype=float),
        )
        (output_dir / "worker_metadata.json").write_text(
            json.dumps({"warnings": [], "session": ["mock"]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    spec = ModelSpec("mock", "", Path("worker.R"), (), ())
    worker = RWorker(spec, "Rscript", runner=runner)
    predictions, metadata = worker.run(
        tmp_path,
        np.ones((3, 2)),
        np.arange(3, dtype=float),
        np.ones((2, 2)),
        {"value": 1},
        42,
    )
    assert predictions.tolist() == [0.0, 1.0]
    assert metadata["session"] == ["mock"]
