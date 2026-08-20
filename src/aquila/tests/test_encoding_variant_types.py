# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Tests for explicit and automatic VCF variant typing."""

from __future__ import annotations

from pathlib import Path

from aquila.encoding import parse_genotype_file, parse_id_prefix_spec


def _write_vcf(path: Path) -> None:
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
        "1\t10\trs10\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n"
        "1\t20\t.\tAT\tA\t.\tPASS\t.\tGT\t0/1\t1/1\n"
        "1\t30\tmarker30\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/0\n",
        encoding="utf-8",
    )


def test_explicit_snp_errors_on_non_acgt_sites(tmp_path: Path) -> None:
    path = tmp_path / "input.vcf"
    _write_vcf(path)

    try:
        parse_genotype_file(
            str(path),
            encoding_type="diploid_onehot",
            variant_type="snp",
        )
    except ValueError as exc:
        message = str(exc)
        assert "SNP mode" in message
        assert "non-ACGT" in message
        assert "." in message
    else:
        raise AssertionError("expected ValueError for non-ACGT indel record")


def test_explicit_snp_keeps_all_acgt_sites_without_id_prefix(tmp_path: Path) -> None:
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
        "1\t10\trs10\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n"
        "1\t30\tmarker30\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/0\n",
        encoding="utf-8",
    )
    parsed = parse_genotype_file(
        str(path),
        encoding_type="diploid_onehot",
        variant_type="snp",
    )
    assert parsed["variant_ids"] == ["rs10", "marker30"]
    assert parsed["matrix"].shape == (2, 2, 8)


def test_snp_mode_allows_star_allele_as_all_zero(tmp_path: Path) -> None:
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
        "1\t10\tSNP-1-10-1\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n"
        "1\t20\tSNP-1-20-1\tA\t*\t.\tPASS\t.\tGT\t0/0\t1/1\n",
        encoding="utf-8",
    )
    parsed = parse_genotype_file(
        str(path),
        encoding_type="10classed_onehot",
        variant_type="snp",
        id_prefix="SNP-",
    )
    assert parsed["variant_ids"] == ["SNP-1-10-1", "SNP-1-20-1"]
    assert parsed["matrix"].shape == (2, 2, 10)
    assert float(parsed["matrix"][:, 1].sum()) == 0.0


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


def test_10classed_onehot_is_unordered_diploid_over_acgt(tmp_path: Path) -> None:
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\tC\n"
        "1\t10\trs10\tA\tT\t.\tPASS\t.\tGT\t0/0\t0/1\t1/0\n"
        "1\t20\trs20\tC\tG\t.\tPASS\t.\tGT\t1/1\t./.\t0/1\n",
        encoding="utf-8",
    )
    parsed = parse_genotype_file(
        str(path),
        encoding_type="10classed_onehot",
        variant_type="snp",
    )
    matrix = parsed["matrix"]
    assert matrix.shape == (3, 2, 10)
    # AA, AT, AC, AG, TT, TC, TG, CC, CG, GG
    assert matrix[0, 0].tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # AA
    assert matrix[1, 0].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # AT
    assert matrix[2, 0].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # TA == AT
    assert matrix[0, 1].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # GG
    assert matrix[1, 1].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # missing
    assert matrix[2, 1].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  # CG


def test_parse_id_prefix_spec_splits_and_strips_caret() -> None:
    assert parse_id_prefix_spec("SNP-") == ("SNP-",)
    assert parse_id_prefix_spec("SNP- | INDEL-") == ("SNP-", "INDEL-")
    assert parse_id_prefix_spec(None) is None


def test_id_prefix_keeps_matching_vcf_ids(tmp_path: Path) -> None:
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tA\tB\n"
        "1\t10\tSNP-1-10-1\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n"
        "1\t20\tINDEL-1-20-1\tAT\tA\t.\tPASS\t.\tGT\t0/1\t1/1\n"
        "1\t30\tSV-1-30-1\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/0\n",
        encoding="utf-8",
    )
    snp_only = parse_genotype_file(
        str(path),
        encoding_type="10classed_onehot",
        variant_type="snp",
        id_prefix="SNP-",
    )
    assert snp_only["variant_ids"] == ["SNP-1-10-1"]
    assert snp_only["matrix"].shape == (2, 1, 10)
    try:
        parse_genotype_file(
            str(path),
            encoding_type="diploid_onehot",
            variant_type="snp",
            id_prefix="SNP- | INDEL-",
        )
    except ValueError as exc:
        assert "INDEL-1-20-1" in str(exc)
    else:
        raise AssertionError("expected ValueError when INDEL rows are kept as SNP")

