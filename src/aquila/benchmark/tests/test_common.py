# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Tests for the shared prepared nested-CV benchmark contract."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aquila.benchmark import (
    GenotypePreprocessor,
    PreparedBenchmark,
    aggregate_outer_folds,
    build_sample_audit,
    evaluate_two_scales,
    load_vcf_dosage,
    sanitize_json,
    validate_ordered_variants,
    write_json,
    write_predictions_csv,
)
from aquila.data import PerTraitPreprocessor


def _write_vcf(
    path: Path,
    sample_ids: list[str],
    genotypes: list[list[str]],
    *,
    gzip_output: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(sample_ids)
            + "\n"
        )
        variants = [("1", "10", "rs1", "A", "G"), ("1", "20", ".", "C", "T")]
        for variant, calls in zip(variants, genotypes):
            handle.write(
                "\t".join((*variant, ".", "PASS", ".", "GT:DP"))
                + "\t"
                + "\t".join(f"{call}:8" for call in calls)
                + "\n"
            )


def _make_prepared_directory(root: Path) -> Path:
    sample_ids = ["s0", "s1", "s2", "s3"]
    raw = torch.tensor([[1.0], [2.0], [-999.0], [4.0]], dtype=torch.float32)
    mask = torch.tensor([[True], [True], [False], [True]])
    torch.save(torch.zeros((4, 2), dtype=torch.float32), root / "X.pt")
    torch.save(raw, root / "Y_raw.pt")
    torch.save(mask, root / "Y_mask.pt")
    metadata = {
        "sample_ids": sample_ids,
        "trait_names": ["height"],
        "regression_tasks": ["height"],
        "classification_tasks": [],
        "trait_tasks": ["regression"],
        "n_samples": 4,
        "n_traits": 1,
        "outer_folds": 2,
        "inner_folds": 2,
        "raw_genotype_saved": True,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "sample_fold_mapping.txt").write_text("tiny\n", encoding="utf-8")

    outer_splits = [
        (np.asarray([2, 3]), np.asarray([0, 1])),
        (np.asarray([0, 1]), np.asarray([2, 3])),
    ]
    vcf_calls = {
        "s0": ["0/0", "0/1"],
        "s1": ["0/1", "1/1"],
        "s2": ["./.", "0/0"],
        "s3": ["1/1", "0/1"],
    }
    for outer_fold, (train_indices, test_indices) in enumerate(outer_splits):
        cv_path = root / "cv" / f"outer_fold_{outer_fold}"
        raw_path = root / "raw_genotype" / f"outer_fold_{outer_fold}"
        cv_path.mkdir(parents=True)
        np.save(cv_path / "train_idx.npy", train_indices)
        np.save(cv_path / "test_idx.npy", test_indices)
        _save_processed(
            cv_path / "final",
            raw,
            mask,
            train_indices,
            test_indices,
            "test",
        )
        _write_split_vcf(raw_path / "train.vcf.gz", train_indices, sample_ids, vcf_calls)
        _write_split_vcf(raw_path / "test.vcf.gz", test_indices, sample_ids, vcf_calls)

        for inner_fold in range(2):
            inner_train = train_indices[inner_fold : inner_fold + 1]
            inner_valid = train_indices[1 - inner_fold : 2 - inner_fold]
            inner_path = cv_path / f"inner_fold_{inner_fold}"
            inner_raw = raw_path / f"inner_fold_{inner_fold}"
            inner_path.mkdir()
            np.save(inner_path / "train_idx.npy", inner_train)
            np.save(inner_path / "valid_idx.npy", inner_valid)
            _save_processed(
                inner_path,
                raw,
                mask,
                inner_train,
                inner_valid,
                "valid",
            )
            _write_split_vcf(
                inner_raw / "train.vcf.gz", inner_train, sample_ids, vcf_calls
            )
            _write_split_vcf(
                inner_raw / "valid.vcf.gz", inner_valid, sample_ids, vcf_calls
            )
    return root


def _save_processed(
    path: Path,
    raw: torch.Tensor,
    mask: torch.Tensor,
    train_indices: np.ndarray,
    held_out_indices: np.ndarray,
    held_out_name: str,
) -> None:
    processor = PerTraitPreprocessor(skew_threshold=0).fit(
        raw, mask, train_indices, ["height"]
    )
    processed = processor.apply(raw, mask)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(processed[train_indices], path / "Y_train_processed.pt")
    torch.save(
        processed[held_out_indices],
        path / f"Y_{held_out_name}_processed.pt",
    )
    processor.save_json(path / "preprocessing.json")


def _write_split_vcf(
    path: Path,
    indices: np.ndarray,
    all_ids: list[str],
    calls: dict[str, list[str]],
) -> None:
    ids = [all_ids[int(index)] for index in indices]
    records = [[calls[sample_id][variant] for sample_id in ids] for variant in range(2)]
    _write_vcf(path, list(reversed(ids)), [list(reversed(row)) for row in records])


def test_vcf_loader_plain_gzip_missing_and_alignment(tmp_path: Path) -> None:
    for name, compressed in (("tiny.vcf", False), ("tiny.vcf.gz", True)):
        path = tmp_path / name
        _write_vcf(
            path,
            ["b", "a"],
            [["0/0", "0|1"], ["1/1", "./."]],
            gzip_output=compressed,
        )
        loaded = load_vcf_dosage(path).align_samples(["a", "b"])
        np.testing.assert_allclose(
            loaded.dosages,
            np.asarray([[1.0, np.nan], [0.0, 2.0]], dtype=np.float32),
            equal_nan=True,
        )
        assert loaded.sample_ids == ("a", "b")


def test_vcf_loader_rejects_non_diploid_and_variant_reordering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.vcf"
    _write_vcf(path, ["a"], [["0",], ["0/1"]], gzip_output=False)
    with pytest.raises(ValueError, match="non-diploid"):
        load_vcf_dosage(path)
    expected = (("1", "10", "x", "A", "G"), ("1", "20", "y", "C", "T"))
    with pytest.raises(ValueError, match="column 0"):
        validate_ordered_variants(expected, tuple(reversed(expected)))


@pytest.mark.parametrize("scaler", ["none", "standard", "minmax"])
def test_train_only_genotype_preprocessor(scaler: str) -> None:
    train = np.asarray([[0.0, np.nan], [2.0, 2.0]], dtype=np.float32)
    held_out = np.asarray([[np.nan, 100.0]], dtype=np.float32)
    processor = GenotypePreprocessor(scaler).fit(train)
    transformed = processor.transform(held_out)
    assert np.isfinite(transformed).all()
    assert processor.mean_ is not None
    np.testing.assert_allclose(processor.mean_, [1.0, 2.0])
    restored = GenotypePreprocessor.from_dict(processor.to_dict())
    np.testing.assert_allclose(restored.transform(held_out), transformed)


def test_prepared_benchmark_integrity_filter_inverse_and_audit(
    tmp_path: Path,
) -> None:
    benchmark = PreparedBenchmark(_make_prepared_directory(tmp_path))
    train, test = benchmark.load_single_trait_fold("height", outer_fold=1)
    assert train.sample_ids == ("s0", "s1")
    assert test.sample_ids == ("s3",)
    assert test.discarded_sample_ids == ("s2",)
    np.testing.assert_allclose(
        benchmark.inverse_trait(
            test.processed_targets, "height", outer_fold=1
        ),
        test.raw_targets,
    )
    audit = build_sample_audit(train, test, held_out_name="test")
    assert audit["test"]["discarded_sample_ids"] == ["s2"]

    missing = benchmark.resolve_fold_paths(0, 0).held_out_vcf
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        PreparedBenchmark(tmp_path)


def test_metrics_json_predictions_and_ddof_one(tmp_path: Path) -> None:
    evaluation = evaluate_two_scales(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [10.0, 20.0, 30.0],
        [10.0, 20.0, 30.0],
        trait_name="height",
    )
    assert evaluation.processed["avg_pearson"] == pytest.approx(1.0)
    assert evaluation.original["avg_rmse"] == pytest.approx(0.0)

    output = tmp_path / "predictions.csv"
    write_predictions_csv(
        output,
        ["a", "b", "c"],
        evaluation.targets_processed,
        evaluation.predictions_processed,
        evaluation.targets_original,
        evaluation.predictions_original,
        trait_name="height",
        outer_fold=0,
    )
    with output.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == ["a", "b", "c"]

    summary = aggregate_outer_folds(
        [{"processed": {"pearson": 1.0}}, {"processed": {"pearson": 3.0}}]
    )
    assert summary["processed"]["pearson"]["mean"] == pytest.approx(2.0)
    assert summary["processed"]["pearson"]["std"] == pytest.approx(np.sqrt(2.0))

    strict = sanitize_json({"nan": np.nan, "inf": torch.tensor(float("inf"))})
    assert strict == {"nan": None, "inf": None}
    json_path = tmp_path / "strict.json"
    write_json(json_path, strict)
    assert "NaN" not in json_path.read_text(encoding="utf-8")
