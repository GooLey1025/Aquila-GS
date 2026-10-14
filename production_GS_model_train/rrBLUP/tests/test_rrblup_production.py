#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/cran/rrBLUP

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RRBLUP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RRBLUP_DIR))

from rrblup_production import (  # noqa: E402
    _alt_alleles,
    _marker_ids,
    dosages_from_diploid_onehot,
    run_trait,
)


class _RecordingWorker:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, work_dir, train_x, train_y, predict_x, parameters, seed):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        prediction = np.asarray(predict_x[:, 0], dtype=np.float64)
        np.savetxt(work_dir / "predictions.tsv", prediction, fmt="%.17g")
        (work_dir / "model.rds").write_text("model\n", encoding="utf-8")
        (work_dir / "worker_metadata.json").write_text("{}\n", encoding="utf-8")
        self.calls.append(
            {
                "train_rows": len(train_y),
                "predict_rows": len(predict_x),
                "parameters": dict(parameters),
                "seed": seed,
            }
        )
        return prediction, {}


def _write_processed(path: Path, indices: list[int]) -> None:
    targets = np.asarray(indices, dtype=np.float32).reshape(-1, 1)
    torch.save(torch.tensor(targets), path)


def test_marker_ids_come_from_the_main_branch() -> None:
    assert _marker_ids({"variant_ids": {"main": ["SNP-1", "SNP-2"]}}) == [
        "SNP-1",
        "SNP-2",
    ]


def test_dosage_counts_the_vcf_alt_nucleotide() -> None:
    features = np.zeros((2, 1, 8), dtype=np.float64)
    features[0, 0, :] = [1, 0, 0, 0, 1, 0, 0, 0]
    features[1, 0, :] = [1, 0, 0, 0, 0, 1, 0, 0]
    dosage = dosages_from_diploid_onehot(features, [1])
    assert np.allclose(dosage[:, 0], [0.0, 1.0])


def test_missing_onehot_stays_missing() -> None:
    features = np.zeros((1, 1, 8), dtype=np.float64)
    dosage = dosages_from_diploid_onehot(features, [1])
    assert np.isnan(dosage[0, 0])


def test_non_nucleotide_markers_are_excluded_from_dosage(tmp_path: Path) -> None:
    vcf = tmp_path / "markers.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "1\t1\tSNP-valid\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1\n"
        "1\t2\tSNP-star\tA\t*\t.\tPASS\t.\tGT\t0/0\t0/1\n",
        encoding="utf-8",
    )
    from aquila.encoding import parse_genotype_file

    parsed = parse_genotype_file(str(vcf), encoding_type="diploid_onehot",
                                 variant_type="snp", id_prefix="SNP-")
    alts, indices, keep = _alt_alleles(
        str(vcf), parsed["variant_ids"], "SNP-"
    )
    assert alts == ["C"]
    assert indices.tolist() == [1]
    assert keep.tolist() == [True, False]
    dosage = dosages_from_diploid_onehot(parsed["matrix"][:, keep, :], indices)
    assert dosage.shape == (2, 1)
    assert np.allclose(dosage[:, 0], [0.0, 1.0])


def test_training_uses_all_observed_samples_without_folds(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    full_path = data_dir / "full"
    full_path.mkdir(parents=True)
    _write_processed(full_path / "Y_processed.pt", list(range(6)))
    (full_path / "preprocessing.json").write_text(
        json.dumps(
            {
                "traits": [
                    {
                        "std": 1.0,
                        "mean": 0.0,
                        "use_log1p": False,
                        "log_shift": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    worker = _RecordingWorker()
    summary = run_trait(
        worker=worker,
        dosage=np.arange(12, dtype=np.float64).reshape(6, 2),
        sample_ids=[f"S{index}" for index in range(6)],
        mask=np.ones((6, 1), dtype=bool),
        data_dir=data_dir,
        output_dir=tmp_path / "out",
        trait="PH_BLUP",
        trait_index=0,
        parameters={"method": "REML"},
        base_seed=42,
        variants=("m0", "m1"),
    )
    assert len(worker.calls) == 1
    assert worker.calls[0]["train_rows"] == 6
    assert worker.calls[0]["predict_rows"] == 6
    assert summary["parameters"] == {"method": "REML"}
    full_dir = tmp_path / "out" / "PH_BLUP" / "full"
    assert (full_dir / "model.rds").is_file()
    assert (full_dir / "insample_predictions.tsv").is_file()
    assert not (tmp_path / "out" / "PH_BLUP" / "hpo_results.json").exists()
