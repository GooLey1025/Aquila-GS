# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Tests for explicit and automatic VCF variant typing."""

from __future__ import annotations

from pathlib import Path

from aquila.encoding import parse_genotype_file


def _write_vcf(path: Path) -> None:
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
        "1\t10\trs10\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n"
        "1\t20\t.\tAT\tA\t.\tPASS\t.\tGT\t0/1\t1/1\n"
        "1\t30\tmarker30\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/0\n",
        encoding="utf-8",
    )


def test_explicit_snp_does_not_require_id_prefix(tmp_path: Path) -> None:
    path = tmp_path / "input.vcf"
    _write_vcf(path)

    parsed = parse_genotype_file(
        str(path),
        encoding_type="diploid_onehot",
        variant_type="snp",
    )

    assert parsed["matrix"].shape == (2, 3, 8)
    assert parsed["variant_ids"] == ["rs10", ".", "marker30"]


def test_explicit_indel_and_sv_treat_every_record_as_selected_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.vcf"
    _write_vcf(path)

    indel = parse_genotype_file(
        str(path),
        encoding_type="diploid_onehot",
        variant_type="indel",
    )
    structural = parse_genotype_file(
        str(path),
        encoding_type="diploid_onehot",
        variant_type="sv",
    )

    assert indel["matrix"].shape == (2, 3, 4)
    assert structural["matrix"].shape == (2, 3, 4)
    assert indel["variant_ids"] == ["rs10", ".", "marker30"]
    assert structural["variant_ids"] == ["rs10", ".", "marker30"]


def test_omitted_variant_type_detects_types_automatically(tmp_path: Path) -> None:
    path = tmp_path / "input.vcf"
    _write_vcf(path)

    parsed = parse_genotype_file(
        str(path),
        encoding_type="diploid_onehot",
        variant_type=None,
    )

    assert parsed["SNP"]["variant_ids"] == ["rs10", "marker30"]
    assert parsed["INDEL"]["variant_ids"] == ["."]

