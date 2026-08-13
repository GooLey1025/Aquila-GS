# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/Marxin1992/Whisperer_of_DNA.git

"""Strict Aquila prepared-data adapter for DNA Whisper benchmarks."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aquila.benchmark.common import PreparedBenchmark, SingleTraitSplit, VariantKey


GENOTYPE_CLASSES = ("AA", "AT", "AC", "AG", "TT", "TC", "TG", "CC", "CG", "GG")
GENOTYPE_CLASS_INDEX = {value: index for index, value in enumerate(GENOTYPE_CLASSES)}
BASE_ORDER = {"A": 0, "T": 1, "C": 2, "G": 3}


@dataclass(frozen=True)
class MultiTraitSplit:
    """Observed multi-trait rows for one prepared partition."""

    genotypes: np.ndarray
    processed_targets: np.ndarray
    raw_targets: np.ndarray
    observed_mask: np.ndarray
    sample_ids: tuple[str, ...]
    discarded_sample_ids: tuple[str, ...]
    absolute_indices: np.ndarray
    variants: tuple[VariantKey, ...]
    trait_names: tuple[str, ...]


@dataclass(frozen=True)
class WhispererVCF:
    """Sample-major ten-class one-hot genotypes and ordered VCF schema."""

    genotypes: np.ndarray
    sample_ids: tuple[str, ...]
    variants: tuple[VariantKey, ...]

    def align_samples(self, expected_sample_ids: Sequence[str]) -> "WhispererVCF":
        expected = tuple(str(value) for value in expected_sample_ids)
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("Expected sample IDs must be nonempty and unique")
        if set(expected) != set(self.sample_ids):
            missing = sorted(set(expected) - set(self.sample_ids))
            extras = sorted(set(self.sample_ids) - set(expected))
            raise ValueError(
                f"VCF sample schema mismatch: missing={missing[:10]}, "
                f"unexpected={extras[:10]}"
            )
        positions = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        order = np.asarray([positions[sample_id] for sample_id in expected])
        return WhispererVCF(
            self.genotypes[order].copy(),
            expected,
            self.variants,
        )


def encode_diploid_bases(
    reference: str,
    alternate: str,
    genotype: str,
) -> np.ndarray:
    """Encode an unphased diploid SNP genotype into Whisperer's ten classes."""
    result = np.zeros(10, dtype=np.float32)
    normalized = genotype.replace("|", "/")
    if normalized in {".", "./."}:
        return result
    alleles = normalized.split("/")
    if len(alleles) != 2:
        raise ValueError(f"Genotype must be diploid, received {genotype!r}")
    if "." in alleles:
        return result
    if any(allele not in {"0", "1"} for allele in alleles):
        raise ValueError(f"Only biallelic GT values are supported, received {genotype!r}")
    bases = [reference if allele == "0" else alternate for allele in alleles]
    bases.sort(key=BASE_ORDER.__getitem__)
    result[GENOTYPE_CLASS_INDEX["".join(bases)]] = 1.0
    return result


def load_whisperer_vcf(path: str | Path) -> WhispererVCF:
    """Read Aquila-selected SNP records as ten-class one-hot vectors."""
    vcf_path = Path(path)
    if not vcf_path.is_file():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    opener = gzip.open if vcf_path.suffix.lower() == ".gz" else open
    sample_ids: tuple[str, ...] | None = None
    variants: list[VariantKey] = []
    marker_rows: list[np.ndarray] = []
    with opener(vcf_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##"):
                continue
            columns = line.rstrip("\r\n").split("\t")
            if line.startswith("#CHROM"):
                if len(columns) < 10:
                    raise ValueError("VCF must contain at least one sample")
                sample_ids = tuple(columns[9:])
                if any(not value for value in sample_ids) or len(set(sample_ids)) != len(
                    sample_ids
                ):
                    raise ValueError("VCF sample IDs must be nonempty and unique")
                continue
            if line.startswith("#"):
                continue
            if sample_ids is None:
                raise ValueError("VCF data appeared before the #CHROM header")
            if len(columns) != 9 + len(sample_ids):
                raise ValueError(
                    f"VCF record {line_number} has an inconsistent sample count"
                )
            reference, alternate = columns[3].upper(), columns[4].upper()
            if "," in columns[4] or reference == alternate:
                raise ValueError(
                    f"VCF record {line_number} is not a biallelic SNP record"
                )
            format_fields = columns[8].split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record {line_number} has no GT FORMAT field")
            gt_index = format_fields.index("GT")
            valid_bases = (
                len(reference) == 1
                and len(alternate) == 1
                and reference in BASE_ORDER
                and alternate in BASE_ORDER
            )
            encoded = []
            for sample_field in columns[9:]:
                fields = sample_field.split(":")
                if gt_index >= len(fields):
                    raise ValueError(
                        f"VCF record {line_number} has a truncated sample field"
                    )
                encoded.append(
                    encode_diploid_bases(reference, alternate, fields[gt_index])
                    if valid_bases
                    else np.zeros(10, dtype=np.float32)
                )
            variants.append(
                (columns[0], columns[1], columns[2], reference, alternate)
            )
            marker_rows.append(np.stack(encoded))
    if sample_ids is None:
        raise ValueError("VCF does not contain a #CHROM header")
    if not marker_rows:
        raise ValueError("VCF does not contain any genotype records")
    genotypes = np.stack(marker_rows, axis=1).astype(np.float32, copy=False)
    return WhispererVCF(genotypes, sample_ids, tuple(variants))


def retained_variant_indices(variant_count: int, block_length: int) -> np.ndarray:
    """Keep a phenotype-independent prefix so the SNP sequence is block-divisible.

    This is not marker screening: leftover columns at the end of the ordered
    schema are dropped solely because DNAWhisper's embedding requires the
    sequence length to be a multiple of ``Block_length``.
    """
    if variant_count < 1 or block_length < 1:
        raise ValueError("Variant count and block length must be positive")
    retained_count = variant_count - variant_count % block_length
    if retained_count == 0:
        raise ValueError("Block length exceeds the available variant count")
    return np.arange(retained_count, dtype=np.int64)


class WhispererPreparedBenchmark(PreparedBenchmark):
    """Prepared benchmark that replaces dosage loading with DNA base-pair encoding."""

    def load_single_trait_fold(
        self,
        trait: str | int,
        outer_fold: int,
        inner_fold: int | None = None,
        *,
        block_length: int,
        expected_variants: Sequence[VariantKey] | None = None,
    ) -> tuple[SingleTraitSplit, SingleTraitSplit, dict[str, Any]]:
        paths = self.resolve_fold_paths(outer_fold, inner_fold)
        trait_index = self.trait_index(trait)
        train_indices = np.load(paths.train_indices, allow_pickle=False).astype(np.int64)
        held_indices = np.load(paths.held_out_indices, allow_pickle=False).astype(np.int64)
        train_vcf = load_whisperer_vcf(paths.train_vcf).align_samples(
            self.sample_ids_for(train_indices)
        )
        held_vcf = load_whisperer_vcf(paths.held_out_vcf).align_samples(
            self.sample_ids_for(held_indices)
        )
        if train_vcf.variants != held_vcf.variants:
            raise ValueError("Training and held-out VCF variant schemas differ")
        if expected_variants is not None and tuple(expected_variants) != train_vcf.variants:
            raise ValueError("Fold VCF variant schema differs from the global schema")
        keep = retained_variant_indices(len(train_vcf.variants), block_length)
        train_processed = self._load_processed(paths.train_targets, len(train_indices))
        held_processed = self._load_processed(paths.held_out_targets, len(held_indices))
        train = self._partition(
            train_vcf, train_indices, train_processed, trait_index, keep
        )
        held = self._partition(
            held_vcf, held_indices, held_processed, trait_index, keep
        )
        schema = {
            "full_variant_count": len(train_vcf.variants),
            "retained_variant_count": int(keep.size),
            "trimmed_remainder": len(train_vcf.variants) - int(keep.size),
            "block_length": block_length,
            "retained_indices": keep.tolist(),
            "variants": [list(value) for value in train_vcf.variants],
            "retained_variants": [list(train_vcf.variants[index]) for index in keep],
        }
        return train, held, schema

    def load_multi_trait_fold(
        self,
        trait_names: Sequence[str],
        outer_fold: int,
        inner_fold: int | None = None,
        *,
        block_length: int,
        expected_variants: Sequence[VariantKey] | None = None,
    ) -> tuple[MultiTraitSplit, MultiTraitSplit, dict[str, Any]]:
        """Load one joint multi-trait split with per-trait observation masks."""
        names = tuple(str(name) for name in trait_names)
        if not names:
            raise ValueError("DNAWhisper multi-trait training requires at least one trait")
        invalid = [name for name in names if name not in self.trait_names]
        if invalid:
            raise ValueError(f"Unknown traits: {invalid}")
        trait_indices = np.asarray(
            [self.trait_index(name) for name in names], dtype=np.int64
        )
        paths = self.resolve_fold_paths(outer_fold, inner_fold)
        train_indices = np.load(paths.train_indices, allow_pickle=False).astype(np.int64)
        held_indices = np.load(paths.held_out_indices, allow_pickle=False).astype(np.int64)
        train_vcf = load_whisperer_vcf(paths.train_vcf).align_samples(
            self.sample_ids_for(train_indices)
        )
        held_vcf = load_whisperer_vcf(paths.held_out_vcf).align_samples(
            self.sample_ids_for(held_indices)
        )
        if train_vcf.variants != held_vcf.variants:
            raise ValueError("Training and held-out VCF variant schemas differ")
        if expected_variants is not None and tuple(expected_variants) != train_vcf.variants:
            raise ValueError("Fold VCF variant schema differs from the global schema")
        keep = retained_variant_indices(len(train_vcf.variants), block_length)
        train_processed = self._load_processed(paths.train_targets, len(train_indices))
        held_processed = self._load_processed(paths.held_out_targets, len(held_indices))
        train = self._multi_trait_partition(
            train_vcf, train_indices, train_processed, names, trait_indices, keep
        )
        held = self._multi_trait_partition(
            held_vcf, held_indices, held_processed, names, trait_indices, keep
        )
        schema = {
            "full_variant_count": len(train_vcf.variants),
            "retained_variant_count": int(keep.size),
            "trimmed_remainder": len(train_vcf.variants) - int(keep.size),
            "block_length": block_length,
            "retained_indices": keep.tolist(),
            "variants": [list(value) for value in train_vcf.variants],
            "retained_variants": [list(train_vcf.variants[index]) for index in keep],
            "trait_names": list(names),
        }
        return train, held, schema

    def inverse_selected_traits(
        self,
        values: Any,
        mask: Any,
        trait_names: Sequence[str],
        outer_fold: int,
        inner_fold: int | None = None,
    ) -> np.ndarray:
        """Invert selected trait columns using the fold-local preprocessor."""
        names = tuple(str(name) for name in trait_names)
        array = np.asarray(values, dtype=np.float64)
        mask_array = np.asarray(mask, dtype=bool)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if mask_array.ndim == 1:
            mask_array = mask_array.reshape(-1, 1)
        if array.shape != mask_array.shape:
            raise ValueError("Inverse values and mask must have the same shape")
        if array.shape[1] != len(names):
            raise ValueError("Inverse values do not match the selected trait set")
        full = np.zeros((array.shape[0], len(self.trait_names)), dtype=np.float64)
        full_mask = np.zeros_like(full, dtype=bool)
        for column, name in enumerate(names):
            index = self.trait_index(name)
            full[:, index] = array[:, column]
            full_mask[:, index] = mask_array[:, column]
        restored = np.asarray(
            self.load_preprocessor(outer_fold, inner_fold).inverse(full, full_mask)
        )
        return restored[:, [self.trait_index(name) for name in names]]

    @staticmethod
    def _load_processed(path: Path, expected_rows: int) -> np.ndarray:
        import torch

        try:
            values = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            values = torch.load(path, map_location="cpu")
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise ValueError(f"Processed targets must be a two-dimensional tensor: {path}")
        if values.shape[0] != expected_rows:
            raise ValueError(f"Processed target row count mismatch: {path}")
        return values.detach().cpu().numpy()

    def _partition(
        self,
        vcf: WhispererVCF,
        absolute_indices: np.ndarray,
        processed: np.ndarray,
        trait_index: int,
        keep: np.ndarray,
    ) -> SingleTraitSplit:
        raw = self.prepared.targets.detach().cpu().numpy()[absolute_indices, trait_index]
        mask = self.prepared.target_mask.detach().cpu().numpy()[
            absolute_indices, trait_index
        ].astype(bool)
        observed = mask & np.isfinite(raw) & np.isfinite(processed[:, trait_index])
        sample_array = np.asarray(vcf.sample_ids, dtype=object)
        return SingleTraitSplit(
            vcf.genotypes[observed][:, keep].copy(),
            np.asarray(processed[observed, trait_index], dtype=np.float32),
            np.asarray(raw[observed], dtype=np.float32),
            tuple(sample_array[observed].tolist()),
            tuple(sample_array[~observed].tolist()),
            absolute_indices[observed].copy(),
            tuple(vcf.variants[index] for index in keep),
        )

    def _multi_trait_partition(
        self,
        vcf: WhispererVCF,
        absolute_indices: np.ndarray,
        processed: np.ndarray,
        trait_names: Sequence[str],
        trait_indices: np.ndarray,
        keep: np.ndarray,
    ) -> MultiTraitSplit:
        raw = self.prepared.targets.detach().cpu().numpy()[absolute_indices][
            :, trait_indices
        ]
        mask = self.prepared.target_mask.detach().cpu().numpy()[absolute_indices][
            :, trait_indices
        ].astype(bool)
        selected = np.asarray(processed[:, trait_indices], dtype=np.float32)
        observed = mask & np.isfinite(raw) & np.isfinite(selected)
        keep_samples = observed.any(axis=1)
        sample_array = np.asarray(vcf.sample_ids, dtype=object)
        return MultiTraitSplit(
            vcf.genotypes[keep_samples][:, keep].copy(),
            selected[keep_samples].copy(),
            np.asarray(raw[keep_samples], dtype=np.float32),
            observed[keep_samples].copy(),
            tuple(sample_array[keep_samples].tolist()),
            tuple(sample_array[~keep_samples].tolist()),
            absolute_indices[keep_samples].copy(),
            tuple(vcf.variants[index] for index in keep),
            tuple(trait_names),
        )
