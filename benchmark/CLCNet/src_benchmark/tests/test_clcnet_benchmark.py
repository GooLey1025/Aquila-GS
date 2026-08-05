# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/SuppurNewer/CLCNet

"""Focused tests for the leakage-safe CLCNet benchmark adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
MODULE_PATH = SCRIPT_DIRECTORY.parent / "CLCNet_train_cv.py"
SPEC = importlib.util.spec_from_file_location("clcnet_train_cv_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from aquila.benchmark.common import DosageVCF
from aquila.training.hpo import generate_grid_candidates, half_up_median_epoch


def test_hpo_grid_has_exactly_32_stable_candidates() -> None:
    config_path = SCRIPT_DIRECTORY.parent / "configs" / "CLCNet_nested_cv.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    first = generate_grid_candidates(config["hpo"]["parameters"])
    second = generate_grid_candidates(config["hpo"]["parameters"])
    assert len(first) == 32
    assert first == second
    assert first[0]["train.batch_size"] == 16
    assert first[-1]["train.scheduler_factor"] == 0.8


def test_gpu_cli_defaults_to_detection() -> None:
    arguments = MODULE.parse_args(["--data-dir", "data", "-o", "output"])
    assert arguments.gpus is None
    assert arguments.jobs_per_gpu == 1
    assert arguments.traits is None


def test_gpu_cli_selection_and_cpu_mode() -> None:
    selected = MODULE.parse_args(
        [
            "--data-dir",
            "data",
            "--traits",
            "trait",
            "-o",
            "output",
            "--gpus",
            "0",
            "2",
        ]
    )
    cpu = MODULE.parse_args(
        [
            "--data-dir",
            "data",
            "--traits",
            "trait",
            "-o",
            "output",
            "--gpus",
        ]
    )
    assert selected.gpus == [0, 2]
    assert cpu.gpus == []


def test_gpu_slots_expand_and_jobs_per_gpu_is_positive() -> None:
    assert MODULE.expand_gpu_workers([1, 3], 2) == [1, 1, 3, 3]
    with pytest.raises(SystemExit):
        MODULE.parse_args(
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


def test_upstream_encoding_maps_only_missing_calls_to_three() -> None:
    values = np.asarray([[0.0, 1.0, np.nan], [2.0, np.nan, 0.0]])
    encoded = MODULE.encode_upstream_genotypes(values)
    np.testing.assert_array_equal(
        encoded,
        np.asarray([[0.0, 1.0, 3.0], [2.0, 3.0, 0.0]], dtype=np.float32),
    )


def test_sample_alignment_uses_ids_instead_of_position() -> None:
    variants = (("1", "10", "v1", "A", "G"),)
    vcf = DosageVCF(
        np.asarray([[2.0], [0.0]], dtype=np.float32),
        ("sample_b", "sample_a"),
        variants,
    )
    aligned = vcf.align_samples(("sample_a", "sample_b"))
    assert aligned.sample_ids == ("sample_a", "sample_b")
    np.testing.assert_array_equal(aligned.dosages[:, 0], [0.0, 2.0])


def test_chromosome_groups_preserve_absolute_variant_indices() -> None:
    variants = (
        ("2", "5", "a", "A", "C"),
        ("1", "9", "b", "A", "G"),
        ("2", "8", "c", "C", "T"),
        ("10", "1", "d", "G", "T"),
    )
    groups = MODULE.chromosome_groups(variants)
    assert list(groups) == ["2", "1", "10"]
    np.testing.assert_array_equal(groups["2"], [0, 2])
    np.testing.assert_array_equal(groups["1"], [1])
    np.testing.assert_array_equal(groups["10"], [3])


def test_pair_sampling_is_reproducible_by_seed_and_epoch() -> None:
    genotypes = np.arange(20, dtype=np.float32).reshape(5, 4)
    targets = np.arange(5, dtype=np.float32)
    first = MODULE.DeterministicPairDataset(genotypes, targets, seed=123)
    second = MODULE.DeterministicPairDataset(genotypes, targets, seed=123)
    np.testing.assert_array_equal(first.partners, second.partners)
    first.set_epoch(7)
    second.set_epoch(7)
    np.testing.assert_array_equal(first.partners, second.partners)


def test_training_loader_discards_incomplete_final_batch(monkeypatch) -> None:
    captured = {}
    original = MODULE.DataLoader

    def recording_loader(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(MODULE, "DataLoader", recording_loader)
    dataset = MODULE.DeterministicPairDataset(
        np.zeros((5, 3), dtype=np.float32),
        np.arange(5, dtype=np.float32),
        seed=42,
    )
    class TinyModel(MODULE.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.regression = MODULE.nn.Linear(3, 1)
            self.representation = MODULE.nn.Linear(3, 2)

        def forward(self, inputs):
            return self.regression(inputs), MODULE.F.normalize(
                self.representation(inputs), dim=1
            )

    model = TinyModel()
    optimizer = MODULE.torch.optim.SGD(model.parameters(), lr=0.01)
    MODULE._train_epoch(
        model,
        dataset,
        {"batch_size": 4, "contrastive_weight": 1.0},
        optimizer,
        MODULE.torch.device("cpu"),
        epoch=1,
        seed=42,
    )
    assert captured["drop_last"] is True


def test_half_up_median_epoch_matches_benchmark_contract() -> None:
    assert half_up_median_epoch([1, 2, 3, 4]) == 3
    assert half_up_median_epoch([2, 2, 5, 9]) == 4


def test_feature_selection_payload_records_training_scope_only() -> None:
    selection = MODULE.FeatureSelectionResult(
        selected_indices=np.asarray([0, 2]),
        global_indices=np.asarray([0]),
        chromosome_indices={"1": np.asarray([2])},
        global_importances=np.asarray([1.0, 0.0, 0.0]),
        chromosome_importances={"1": np.asarray([0.0, 1.0])},
        seed=42,
        num_boost_round=100,
    )
    variants = (
        ("1", "1", "a", "A", "C"),
        ("1", "2", "b", "A", "G"),
        ("1", "3", "c", "C", "T"),
    )
    payload = MODULE.feature_selection_payload(
        selection, variants, ("train_a", "train_b")
    )
    assert payload["fit_scope"] == "training samples only"
    assert payload["train_sample_ids"] == ["train_a", "train_b"]
    assert payload["union_selected_count"] == 2


def test_selected_matrices_apply_training_selected_columns_unchanged() -> None:
    variants = (
        ("1", "1", "a", "A", "C"),
        ("1", "2", "b", "A", "G"),
        ("2", "1", "c", "C", "T"),
    )
    train = MODULE.SingleTraitSplit(
        genotypes=np.asarray([[0.0, 1.0, np.nan], [2.0, 0.0, 1.0]]),
        processed_targets=np.asarray([0.0, 1.0]),
        raw_targets=np.asarray([10.0, 20.0]),
        sample_ids=("train_a", "train_b"),
        discarded_sample_ids=(),
        absolute_indices=np.asarray([0, 1]),
        variants=variants,
    )
    held_out = MODULE.SingleTraitSplit(
        genotypes=np.asarray([[1.0, 2.0, 0.0]]),
        processed_targets=np.asarray([0.5]),
        raw_targets=np.asarray([15.0]),
        sample_ids=("valid_a",),
        discarded_sample_ids=(),
        absolute_indices=np.asarray([2]),
        variants=variants,
    )
    selection = MODULE.FeatureSelectionResult(
        selected_indices=np.asarray([0, 2]),
        global_indices=np.asarray([0]),
        chromosome_indices={"2": np.asarray([2])},
        global_importances=np.asarray([1.0, 0.0, 0.0]),
        chromosome_importances={"2": np.asarray([1.0])},
        seed=42,
        num_boost_round=100,
    )
    train_selected, held_out_selected = MODULE._selected_matrices(
        train, held_out, selection, 3.0
    )
    np.testing.assert_array_equal(train_selected, [[0.0, 3.0], [2.0, 1.0]])
    np.testing.assert_array_equal(held_out_selected, [[1.0, 0.0]])


def test_two_scale_metrics_keep_pearson_and_restore_error_scale() -> None:
    evaluation = MODULE.evaluate_two_scales(
        predictions_processed=[-0.5, 0.5, 1.0],
        targets_processed=[-1.0, 0.0, 1.0],
        predictions_original=[15.0, 25.0, 30.0],
        targets_original=[10.0, 20.0, 30.0],
        trait_name="trait",
    )
    assert np.isclose(
        evaluation.processed["avg_pearson"],
        evaluation.original["avg_pearson"],
    )
    assert evaluation.original["avg_mse"] > evaluation.processed["avg_mse"]
