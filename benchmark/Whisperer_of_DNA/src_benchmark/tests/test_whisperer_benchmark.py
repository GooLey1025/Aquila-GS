#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/Marxin1992/Whisperer_of_DNA.git

"""Focused tests for the DNA Whisper nested-CV adapter."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

TEST_DIRECTORY = Path(__file__).resolve().parent
SOURCE_DIRECTORY = TEST_DIRECTORY.parent
WHISPERER_DIRECTORY = SOURCE_DIRECTORY.parent
PROJECT_ROOT = WHISPERER_DIRECTORY.parents[1]
for import_path in (
    str(PROJECT_ROOT / "src"),
    str(WHISPERER_DIRECTORY),
    str(SOURCE_DIRECTORY),
):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from aquila.benchmark.common import evaluate_two_scales
from aquila.data.preprocessing import PerTraitPreprocessor, TraitPreprocessing
from aquila.training.distributed import derive_seed
from aquila.training.hpo import generate_grid_candidates, half_up_median_epoch
from Whisperer_train_cv import _slice_metrics, expand_gpu_workers, parse_args
from whisperer_data import (
    GENOTYPE_CLASSES,
    WhispererVCF,
    WhispererPreparedBenchmark,
    encode_diploid_bases,
    load_whisperer_vcf,
    retained_variant_indices,
)
from whisperer_model import (
    _fill_missing_targets,
    _loader,
    apply_candidate_overrides,
    import_dna_whisper,
)


def test_gpu_cli_default_selection_and_cpu() -> None:
    base = ["-o", "output"]
    default = parse_args(base)
    selected = parse_args([*base, "--gpus", "1", "3"])
    cpu = parse_args([*base, "--gpus"])
    assert default.gpus is None
    assert default.jobs_per_gpu == 1
    assert default.traits is None
    assert selected.gpus == [1, 3]
    assert cpu.gpus == []


def test_gpu_slot_expansion_and_positive_job_count() -> None:
    assert expand_gpu_workers([0, 2], 3) == [0, 0, 0, 2, 2, 2]
    with pytest.raises(SystemExit):
        parse_args(["-o", "output", "--jobs-per-gpu", "0"])


def test_genotype_encoding_all_classes_and_missing() -> None:
    cases = [
        ("A", "T", "0/0", "AA"),
        ("A", "T", "0/1", "AT"),
        ("A", "C", "1/0", "AC"),
        ("A", "G", "0|1", "AG"),
        ("A", "T", "1/1", "TT"),
        ("T", "C", "0/1", "TC"),
        ("T", "G", "1/0", "TG"),
        ("A", "C", "1/1", "CC"),
        ("C", "G", "0/1", "CG"),
        ("A", "G", "1/1", "GG"),
    ]
    for reference, alternate, genotype, expected in cases:
        encoded = encode_diploid_bases(reference, alternate, genotype)
        assert encoded.argmax() == GENOTYPE_CLASSES.index(expected)
        assert encoded.sum() == 1.0
    assert encode_diploid_bases("A", "G", "./.").tolist() == [0.0] * 10


def test_vcf_reader_and_schema_validation(tmp_path: Path) -> None:
    path = tmp_path / "tiny.vcf.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tB\tA\n"
        )
        handle.write("1\t1\tchr1.s_1\tA\tG\t.\tPASS\t.\tGT\t1/1\t0/0\n")
        handle.write("1\t2\tSNP-rs2\tC\tT\t.\tPASS\t.\tGT:DP\t./.:0\t0/1:4\n")
        handle.write("1\t3\tSNP-rs3\tA\t*\t.\tPASS\t.\tGT\t0/0\t0/1\n")
        handle.write("1\t3\tINDEL-rs3\tCA\tC\t.\tPASS\t.\tGT\t0/0\t0/1\n")
    parsed = load_whisperer_vcf(path)
    assert parsed.genotypes.shape == (2, 4, 10)
    aligned = parsed.align_samples(("A", "B"))
    assert aligned.sample_ids == ("A", "B")
    assert aligned.genotypes[0, 0, GENOTYPE_CLASSES.index("AA")] == 1.0
    assert aligned.genotypes[1, 1].sum() == 0.0
    assert aligned.genotypes[:, 2].sum() == 0.0
    assert aligned.genotypes[:, 3].sum() == 0.0
    with pytest.raises(ValueError, match="sample schema"):
        parsed.align_samples(("A", "C"))


def test_vcf_accepts_nonprefixed_ids_and_rejects_multiallelic_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\n"
        "1\t1\tINDEL-rs1\tAT\tA\t.\tPASS\t.\tGT\t0/1\n"
        "1\t2\tSNP-rs2\tA\tG,T\t.\tPASS\t.\tGT\t0/1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="biallelic"):
        load_whisperer_vcf(path)


def test_alignment_reorders_without_changing_variants() -> None:
    values = np.arange(40, dtype=np.float32).reshape(2, 2, 10)
    variants = (("1", "1", "rs1", "A", "G"), ("1", "2", "rs2", "C", "T"))
    aligned = WhispererVCF(values, ("B", "A"), variants).align_samples(("A", "B"))
    assert np.array_equal(aligned.genotypes[0], values[1])
    assert aligned.variants == variants


def test_deterministic_block_trim() -> None:
    assert retained_variant_indices(7621, 32).size == 7616
    assert retained_variant_indices(7621, 32)[-1] == 7615
    assert np.array_equal(
        retained_variant_indices(64, 32), np.arange(64, dtype=np.int64)
    )


def test_adapter_does_not_use_supervised_marker_selection() -> None:
    adapter_files = [
        SOURCE_DIRECTORY.parent / "Whisperer_train_cv.py",
        SOURCE_DIRECTORY / "whisperer_data.py",
        SOURCE_DIRECTORY / "whisperer_model.py",
    ]
    forbidden = (
        "MICFilter",
        "mic_filter",
        "mic_filtering",
        "lightgbm",
        "LightGBM",
        "feature_selection",
        "snp_qc",
        "calc_mic",
    )
    for path in adapter_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} must not reference {token}"
    keep = retained_variant_indices(100, 32)
    assert np.array_equal(keep, np.arange(96, dtype=np.int64))



def test_split_local_missing_filter_contract() -> None:
    mask = np.array([True, False, True])
    sample_ids = np.array(["A", "B", "C"], dtype=object)
    assert tuple(sample_ids[mask]) == ("A", "C")
    assert tuple(sample_ids[~mask]) == ("B",)
    assert not np.any(np.array([1.0, -999.0, 2.0])[mask] == -999.0)


def test_multi_trait_partition_keeps_partially_observed_samples() -> None:
    benchmark = object.__new__(WhispererPreparedBenchmark)
    benchmark.prepared = type(
        "Prepared",
        (),
        {
            "targets": torch.tensor(
                [[1.0, -999.0], [-999.0, -999.0], [-999.0, 3.0]]
            ),
            "target_mask": torch.tensor(
                [[True, False], [False, False], [False, True]]
            ),
        },
    )()
    genotypes = np.zeros((3, 2, 10), dtype=np.float32)
    vcf = WhispererVCF(
        genotypes,
        ("A", "B", "C"),
        (("1", "1", "rs1", "A", "G"), ("1", "2", "rs2", "C", "T")),
    )
    split = benchmark._multi_trait_partition(
        vcf,
        np.array([0, 1, 2]),
        np.array(
            [[0.0, -999.0], [-999.0, -999.0], [-999.0, 1.0]],
            dtype=np.float32,
        ),
        ("TraitA", "TraitB"),
        np.array([0, 1]),
        np.array([0, 1]),
    )
    assert split.sample_ids == ("A", "C")
    assert split.discarded_sample_ids == ("B",)
    assert split.observed_mask.tolist() == [[True, False], [False, True]]
    assert split.processed_targets.shape == (2, 2)
    assert split.trait_names == ("TraitA", "TraitB")


def test_prepared_partition_filters_missing_trait_samples() -> None:
    benchmark = object.__new__(WhispererPreparedBenchmark)
    benchmark.prepared = type(
        "Prepared",
        (),
        {
            "targets": torch.tensor([[1.0], [-999.0], [3.0]]),
            "target_mask": torch.tensor([[True], [False], [True]]),
        },
    )()
    genotypes = np.zeros((3, 2, 10), dtype=np.float32)
    vcf = WhispererVCF(
        genotypes,
        ("A", "B", "C"),
        (("1", "1", "rs1", "A", "G"), ("1", "2", "rs2", "C", "T")),
    )
    split = benchmark._partition(
        vcf,
        np.array([0, 1, 2]),
        np.array([[0.0], [-999.0], [1.0]], dtype=np.float32),
        0,
        np.array([0, 1]),
    )
    assert split.sample_ids == ("A", "C")
    assert split.discarded_sample_ids == ("B",)
    assert split.processed_targets.tolist() == [0.0, 1.0]


def test_grid_contains_exactly_32_candidates() -> None:
    config_path = SOURCE_DIRECTORY.parent / "configs" / "Whisperer_nested_cv.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    candidates = generate_grid_candidates(config["hpo"]["parameters"])
    assert len(candidates) == 32


def test_dropout_and_encoder_override_are_uniform() -> None:
    model_path = WHISPERER_DIRECTORY / "training" / "config" / "model_config.json"
    base = json.loads(model_path.read_text(encoding="utf-8"))
    updated = apply_candidate_overrides(
        base, {"dropout": 0.2, "encoder_layers": 6}, ("TraitA", "TraitB")
    )
    assert updated["embedding"]["input_dims"] == 10
    assert updated["output_layer"]["phenotype_dim"] == 2
    assert updated["output_layer"]["phenotype_name"] == ["TraitA", "TraitB"]
    assert updated["output_layer"]["dropout_rate"] == 0.2
    assert updated["loss_config"]["auxiliary_losses"]["Deep_Supervision"]["enabled"] is True
    assert updated["loss_config"]["auxiliary_losses"]["PWCosSim"]["enabled"] is False
    assert updated["loss_config"]["auxiliary_losses"]["correlation"]["enabled"] is False
    for block in updated["GFI_FormerBLOCKS"]["blocks"][:2]:
        assert block["encoder"]["num_layers"] == 6
        assert block["encoder"]["attention"]["dropout_rate"] == 0.2
        assert block["decoder"]["cross_attention"]["dropout_rate"] == 0.2
        assert block["decoder"]["MOE"]["dropout_rate"] == 0.2
        assert block["decoder"]["pooling"]["dropout_rate"] == 0.2


def test_inverse_transform_and_two_scale_metrics() -> None:
    processor = PerTraitPreprocessor()
    processor.traits = [
        TraitPreprocessing(name="Trait", task="regression", mean=10.0, std=2.0)
    ]
    full = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    restored = processor.inverse(full, np.ones_like(full, dtype=bool))[:, 0]
    evaluation = evaluate_two_scales(
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        restored,
        restored,
        trait_name="Trait",
    )
    assert evaluation.processed["avg_pearson"] == pytest.approx(1.0)
    assert evaluation.original["avg_mse"] == pytest.approx(0.0)


def test_half_up_epoch_aggregation() -> None:
    assert half_up_median_epoch([1, 2, 3, 4]) == 3
    assert half_up_median_epoch([2, 8, 4, 6]) == 5


def test_nested_coordinates_produce_distinct_seeds() -> None:
    first = derive_seed(42, 0, 0, 0)
    assert first != derive_seed(42, 1, 0, 0)
    assert first != derive_seed(42, 0, 1, 0)
    assert first != derive_seed(42, 0, 0, 1)


def test_training_loader_discards_incomplete_final_batch() -> None:
    genotypes = np.zeros((17, 4, 10), dtype=np.float32)
    targets = np.zeros(17, dtype=np.float32)
    loader = _loader(genotypes, targets, batch_size=8, shuffle=True)
    assert sum(batch[0].shape[0] for batch in loader) == 16


def test_evaluation_loader_keeps_incomplete_final_batch() -> None:
    genotypes = np.zeros((17, 4, 10), dtype=np.float32)
    targets = np.zeros(17, dtype=np.float32)
    loader = _loader(genotypes, targets, batch_size=8, shuffle=False)
    assert sum(batch[0].shape[0] for batch in loader) == 17


def test_loader_keeps_two_dimensional_targets_and_observation_mask() -> None:
    genotypes = np.zeros((5, 4, 10), dtype=np.float32)
    targets = np.arange(10, dtype=np.float32).reshape(5, 2)
    mask = np.array(
        [[True, False], [True, True], [False, True], [True, True], [True, False]]
    )
    loader = _loader(genotypes, targets, batch_size=8, shuffle=False, mask=mask)
    features, phenotype, observed = next(iter(loader))
    assert features.shape == (5, 4, 10)
    assert phenotype.shape == (5, 2)
    assert observed.dtype == torch.bool
    assert observed.tolist() == mask.tolist()


def test_missing_labels_are_filled_with_detached_predictions() -> None:
    predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    targets = torch.tensor([[10.0, -999.0], [30.0, 40.0]])
    mask = torch.tensor([[True, False], [True, True]])
    filled = _fill_missing_targets(predictions, targets, mask)
    assert filled[0, 0].item() == 10.0
    assert filled[0, 1].item() == 2.0
    assert filled[1, 1].item() == 40.0
    assert not filled[0, 1].requires_grad
    residual = filled - predictions
    assert residual[0, 1].item() == 0.0


def test_trait_metric_slice_keeps_per_trait_pearson() -> None:
    metrics = {
        "normalized": {
            "per_trait": {
                "TraitA": {"n": 4, "pearson": 0.8, "r2": 0.5, "mse": 0.2, "rmse": 0.45, "mae": 0.3},
                "TraitB": {"n": 3, "pearson": 0.4, "r2": 0.1, "mse": 0.9, "rmse": 0.95, "mae": 0.7},
            },
            "avg_pearson": 0.6,
        },
        "original": {
            "per_trait": {
                "TraitA": {"n": 4, "pearson": 0.7, "r2": 0.4, "mse": 1.2, "rmse": 1.1, "mae": 0.8},
                "TraitB": {"n": 3, "pearson": 0.3, "r2": 0.0, "mse": 2.0, "rmse": 1.4, "mae": 1.1},
            },
            "avg_pearson": 0.5,
        },
        "test_loss": 0.55,
    }
    sliced = _slice_metrics(metrics, "TraitA")
    assert sliced["normalized"]["per_trait"]["TraitA"]["pearson"] == 0.8
    assert sliced["normalized"]["avg_pearson"] == 0.8
    assert sliced["test_loss"] == 0.2
    assert "TraitB" not in sliced["normalized"]["per_trait"]


def test_standard_attention_valid_mask_fallback_is_finite() -> None:
    import_dna_whisper()
    attention_module = sys.modules[
        "aquila_whisperer_upstream.training.models.attention_types"
    ]
    attention = attention_module.StandardAttention(8, 2, 0.0)
    values = torch.randn(3, 4, 8)
    valid_mask = torch.ones(3, 4, dtype=torch.bool)
    output, _ = attention(values, values, values, mask=valid_mask)
    assert torch.isfinite(output).all()
