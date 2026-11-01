# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Outer-holdout split, cache, and winner-selection tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

from aquila.data.cv import (
    apply_inner_holdout,
    generate_nested_folds,
    generate_outer_holdout_folds,
    generate_outer_holdout_folds_from_assignments,
    outer_evaluation_paths,
    save_nested_folds,
)
from aquila.data.dataset import PreparedData, _validate_cv_artifacts
from aquila.scripts.aquila_cv import save_mapping
from aquila.scripts.aquila_cv_holdout import save_holdout_mapping
from aquila.scripts.aquila_data_cv import PreparationConfig
from aquila.scripts.aquila_data_cv_holdout import OuterHoldoutDataPreparer
from aquila.scripts.aquila_train_cv_holdout import (
    _holdout_gpu_worker,
    _score_outer_test,
    _train_candidate,
    require_holdout_data,
    select_holdout_winner,
)
from aquila.training.hpo import generate_grid_candidates, normalize_hpo_config
from aquila.utils import load_config


ROOT = Path(__file__).resolve().parents[3]


def _valid_count(n_train: int, n_splits: int = 5) -> int:
    """Size of sklearn KFold's first test part."""
    return n_train // n_splits + (1 if n_train % n_splits else 0)


def _assert_holdout_partition(folds, n_samples: int) -> None:
    observed_test = []
    for outer in folds:
        train = np.asarray(outer["train"], dtype=np.int64)
        test = np.asarray(outer["test"], dtype=np.int64)
        assert len(outer["inner"]) == 1
        inner = outer["inner"][0]
        inner_train = np.asarray(inner["train"], dtype=np.int64)
        inner_valid = np.asarray(inner["valid"], dtype=np.int64)
        assert set(train).isdisjoint(test)
        assert set(train) | set(test) == set(range(n_samples))
        assert set(inner_train).isdisjoint(inner_valid)
        assert set(inner_train) | set(inner_valid) == set(train.tolist())
        assert set(inner_train).isdisjoint(test)
        assert set(inner_valid).isdisjoint(test)
        assert inner_valid.size == _valid_count(train.size)
        assert inner_train.size == train.size - inner_valid.size
        if train.size % 5 == 0:
            assert inner_train.size * 1 == inner_valid.size * 4
        observed_test.append(test)
    covered = np.concatenate(observed_test)
    assert sorted(covered.tolist()) == list(range(n_samples))


def test_outer_holdout_matches_nested_outer_tests_and_uses_one_4_to_1_split() -> None:
    for n_samples in (100, 103):
        nested = generate_nested_folds(
            n_samples,
            outer_folds=10,
            inner_folds=4,
            seed=42,
        )
        holdout = generate_outer_holdout_folds(
            n_samples,
            outer_folds=10,
            seed=42,
        )
        assert len(holdout) == 10
        _assert_holdout_partition(holdout, n_samples)
        for nested_fold, holdout_fold in zip(nested, holdout):
            assert np.array_equal(nested_fold["test"], holdout_fold["test"])
            assert np.array_equal(nested_fold["train"], holdout_fold["train"])
            assert len(nested_fold["inner"]) == 4
            assert len(holdout_fold["inner"]) == 1


def test_assignment_holdout_keeps_outer_tests_and_drops_mapped_inner_folds() -> None:
    nested = generate_nested_folds(100, outer_folds=10, inner_folds=4, seed=42)
    assignments = np.empty(100, dtype=np.int64)
    for fold in nested:
        assignments[np.asarray(fold["test"])] = int(fold["fold"])
    from_assignments = generate_outer_holdout_folds_from_assignments(
        assignments,
        outer_folds=10,
        seed=42,
    )
    replaced = apply_inner_holdout(nested, seed=42)
    direct = generate_outer_holdout_folds(100, outer_folds=10, seed=42)
    for left, middle, right in zip(from_assignments, replaced, direct):
        assert np.array_equal(left["test"], right["test"])
        assert np.array_equal(left["train"], right["train"])
        assert np.array_equal(
            left["inner"][0]["train"],
            middle["inner"][0]["train"],
        )
        assert np.array_equal(
            left["inner"][0]["valid"],
            right["inner"][0]["valid"],
        )


def _holdout_preparer(mapping: Path) -> OuterHoldoutDataPreparer:
    return OuterHoldoutDataPreparer(
        PreparationConfig(
            genotype_file=mapping,
            phenotype_file=mapping,
            output_directory=mapping.parent,
            fold_mapping=mapping,
            outer_folds=10,
            inner_folds=1,
        )
    )


def test_holdout_mapping_is_loaded_as_written(tmp_path: Path) -> None:
    folds = generate_outer_holdout_folds(100, outer_folds=10, seed=42)
    sample_ids = [f"s{index}" for index in range(100)]
    mapping = tmp_path / "holdout_cv.json"
    save_holdout_mapping(mapping, sample_ids, folds, seed=42)
    loaded = _holdout_preparer(mapping)._build_folds(sample_ids)
    assert len(loaded) == 10
    for original, restored in zip(folds, loaded):
        assert np.array_equal(original["test"], restored["test"])
        assert np.array_equal(original["train"], restored["train"])
        assert len(restored["inner"]) == 1
        assert np.array_equal(
            original["inner"][0]["train"],
            restored["inner"][0]["train"],
        )
        assert np.array_equal(
            original["inner"][0]["valid"],
            restored["inner"][0]["valid"],
        )


def test_holdout_data_prep_rejects_nested_mapping(tmp_path: Path) -> None:
    folds = generate_nested_folds(100, outer_folds=10, inner_folds=4, seed=42)
    sample_ids = [f"s{index}" for index in range(100)]
    mapping = tmp_path / "nested_cv.json"
    save_mapping(mapping, sample_ids, folds, seed=42)
    with pytest.raises(ValueError, match="aquila_cv_holdout.py"):
        _holdout_preparer(mapping)._build_folds(sample_ids)


def test_cv_loader_accepts_single_inner_split_only_for_holdout(tmp_path: Path) -> None:
    folds = generate_outer_holdout_folds(100, outer_folds=10, seed=42)
    save_nested_folds(folds, tmp_path / "cv")
    _validate_cv_artifacts(
        tmp_path,
        {"outer_folds": 10, "inner_folds": 1, "protocol": "outer_holdout"},
        100,
    )
    with pytest.raises(ValueError, match="at least two"):
        _validate_cv_artifacts(
            tmp_path,
            {"outer_folds": 10, "inner_folds": 1},
            100,
        )


def test_processed_targets_use_inner_train_and_skip_refit_cache(tmp_path: Path) -> None:
    n_samples = 50
    folds = generate_outer_holdout_folds(n_samples, outer_folds=10, seed=42)
    config = PreparationConfig(
        genotype_file=tmp_path / "missing.vcf",
        phenotype_file=tmp_path / "missing.pheno",
        output_directory=tmp_path / "out",
        outer_folds=10,
        inner_folds=1,
        seed=42,
    )
    preparer = OuterHoldoutDataPreparer(config)
    generator = torch.Generator().manual_seed(0)
    aligned = {
        "targets": torch.randn(n_samples, 2, generator=generator),
        "target_mask": torch.ones(n_samples, 2, dtype=torch.bool),
        "trait_names": ["trait_a", "trait_b"],
        "trait_tasks": ["regression", "regression"],
    }
    cv_directory = tmp_path / "cv"
    preparer._save_processed_targets(aligned, folds, cv_directory)

    for outer in folds:
        outer_path = cv_directory / f"outer_fold_{int(outer['fold'])}"
        assert not (outer_path / "final").exists()
        inner_path = outer_path / "inner_fold_0"
        train = torch.load(inner_path / "Y_train_processed.pt", weights_only=True)
        valid = torch.load(inner_path / "Y_valid_processed.pt", weights_only=True)
        test = torch.load(inner_path / "Y_test_processed.pt", weights_only=True)
        inner = outer["inner"][0]
        assert tuple(train.shape) == (len(inner["train"]), 2)
        assert tuple(valid.shape) == (len(inner["valid"]), 2)
        assert tuple(test.shape) == (len(outer["test"]), 2)
        assert train.mean(dim=0).abs().max() < 1e-5
        assert (inner_path / "preprocessing.json").is_file()


def test_select_holdout_winner_uses_highest_validation_score() -> None:
    records = [
        {
            "candidate_id": 0,
            "parameters": {"model.d_model": 128},
            "best_metrics": {"avg_pearson": 0.2},
            "best_epoch": 3,
        },
        {
            "candidate_id": 1,
            "parameters": {"model.d_model": 256},
            "best_metrics": {"avg_pearson": 0.8},
            "best_epoch": 7,
        },
        {
            "candidate_id": 2,
            "parameters": {"model.d_model": 256},
            "best_metrics": {"avg_pearson": 0.8},
            "best_epoch": 4,
        },
    ]
    winner = select_holdout_winner(
        records,
        metric="best/val_r",
        direction="maximize",
    )
    assert winner.best.candidate_id == 1
    assert winner.best.objective == pytest.approx(0.8)
    assert winner.best.inner_results[0].best_epoch == 7
    assert len(winner.best.inner_results) == 1


def test_holdout_protocol_is_required() -> None:
    prepared = PreparedData(
        features=torch.zeros(2, 1),
        targets=torch.zeros(2, 1),
        target_mask=torch.ones(2, 1, dtype=torch.bool),
        metadata={"protocol": "nested", "inner_folds": 4},
        directory=Path("."),
    )
    with pytest.raises(ValueError, match="outer_holdout"):
        require_holdout_data(prepared)


def test_outer_test_path_does_not_refit() -> None:
    assert "train_fixed_epochs" not in _score_outer_test.__code__.co_names
    assert "train_fixed_epochs" not in _holdout_gpu_worker.__code__.co_names
    assert "train_fixed_epochs" not in _train_candidate.__code__.co_names


def test_outer_evaluation_paths_follow_holdout_cache(tmp_path: Path) -> None:
    outer = tmp_path / "cv" / "outer_fold_0"
    inner = outer / "inner_fold_0"
    final = outer / "final"
    raw = tmp_path / "raw_genotype" / "outer_fold_0"
    inner.mkdir(parents=True)
    raw.mkdir(parents=True)
    (inner / "Y_test_processed.pt").write_bytes(b"")
    (inner / "train_idx.npy").write_bytes(b"")
    (outer / "test_idx.npy").write_bytes(b"")

    train_dir, train_index, train_vcf = outer_evaluation_paths(tmp_path, 0, "train")
    test_dir, test_index, test_vcf = outer_evaluation_paths(tmp_path, 0, "test")
    assert train_dir == inner
    assert train_index == inner / "train_idx.npy"
    assert train_vcf == raw / "inner_fold_0" / "train.vcf.gz"
    assert test_dir == inner
    assert test_index == outer / "test_idx.npy"
    assert test_vcf == raw / "test.vcf.gz"

    final.mkdir()
    (final / "Y_test_processed.pt").write_bytes(b"")
    nested_dir, nested_index, nested_vcf = outer_evaluation_paths(tmp_path, 0, "test")
    assert nested_dir == final
    assert nested_index == outer / "test_idx.npy"
    assert nested_vcf == raw / "test.vcf.gz"


def test_species_holdout_config_has_16_budgets_and_early_stopping() -> None:
    config = load_config(
        ROOT / "benchmark" / "aquila-snp" / "16hpo_budgets.30-5.yaml"
    )
    normalized = normalize_hpo_config(config["hpo"])
    assert len(generate_grid_candidates(normalized["parameters"])) == 16
    assert config["train"]["num_epochs"] == 30
    assert config["train"]["early_stopping_patience"] == 5
