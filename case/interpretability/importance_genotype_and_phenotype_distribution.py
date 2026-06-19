#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract genotype + phenotype distribution data for top importance loci.

Workflow:
1. Read importance ranking file:
   ${PREFIX}.${VCF%.vcf.gz}.multi-task.${TRAIT}.position_importance/importance_ranking_${TRAIT}_mean.tsv
2. Keep top-K loci (default: 10)
3. Read VCF and find matching loci by ID (e.g. SNP-1-10000-1)
4. Convert phased/unphased GT to allele genotype labels using REF/ALT:
   0|0 -> REFREF, e.g. AA
   0|1 -> REFALT, e.g. AT
   1|0 -> ALTREF, normalized to alphabetical-like allele order? no, keep unordered by sorting alleles, so TA -> AT
   1|1 -> ALTALT, e.g. TT
5. Merge with phenotype table
6. Output:
   - long-format sample-level table
   - per-locus genotype count summary
"""

import argparse
import gzip
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract genotype + phenotype data for top importance loci"
    )
    parser.add_argument(
        "--importance",
        required=True,
        help="Path to importance_ranking_${TRAIT}_mean.tsv"
    )
    parser.add_argument(
        "--vcf",
        required=True,
        help="Path to bgzipped/gzipped VCF file"
    )
    parser.add_argument(
        "--pheno",
        required=True,
        help="Path to phenotype TSV file; first column is sample ID"
    )
    parser.add_argument(
        "--trait",
        required=True,
        help="Trait column name in phenotype table"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top K loci from importance ranking to extract (default: 10)"
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix; default: {TRAIT}.top{K}"
    )
    return parser.parse_args()


def normalize_genotype_label(ref: str, alt: str, gt: str) -> str:
    """
    Convert GT to genotype label using REF/ALT.
    Examples:
      REF=A ALT=T GT=0|0 -> AA
      GT=0|1 or 1|0 -> AT
      GT=1|1 -> TT
    Missing -> NA
    Multi-allelic / unsupported -> NA
    """
    if gt is None or gt in {".", "./.", ".|."}:
        return "NA"

    gt = gt.split(":")[0]

    if gt in {"./.", ".|."}:
        return "NA"

    if "|" in gt:
        alleles = gt.split("|")
    elif "/" in gt:
        alleles = gt.split("/")
    else:
        return "NA"

    if len(alleles) != 2:
        return "NA"

    allele_map = {"0": ref, "1": alt}
    base_calls = []
    for a in alleles:
        if a == ".":
            return "NA"
        if a not in allele_map:
            return "NA"
        base_calls.append(allele_map[a])

    # 对杂合型统一排序，避免 TA 和 AT 分开
    if len(base_calls) == 2 and base_calls[0] != base_calls[1]:
        base_calls = sorted(base_calls)

    return "".join(base_calls)


def read_importance_topk(path: str, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"rank", "locus_id", "importance_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Importance file missing columns: {missing}")

    df = df.sort_values(["rank", "importance_mean"], ascending=[True, False]).head(top_k).copy()

    # locus_id example: SNP-6-2790767-1
    parts = df["locus_id"].str.split("-", expand=True)
    if parts.shape[1] < 4:
        raise ValueError("Unexpected locus_id format; expected like SNP-6-2790767-1")

    df["variant_type"] = parts[0]
    df["chr"] = parts[1].astype(str)
    df["pos"] = parts[2].astype(int)
    df["allele_index"] = parts[3].astype(str)

    return df


def read_pheno(path: str, trait: str) -> pd.DataFrame:
    pheno = pd.read_csv(path, sep="\t", dtype=str)
    if pheno.shape[1] < 2:
        raise ValueError("Phenotype file must have at least two columns")

    sample_col = pheno.columns[0]
    if trait not in pheno.columns:
        raise ValueError(f"Trait '{trait}' not found in phenotype file columns")

    pheno = pheno[[sample_col, trait]].copy()
    pheno.columns = ["sample", "phenotype"]
    pheno["sample"] = pheno["sample"].astype(str)
    pheno["phenotype"] = pd.to_numeric(pheno["phenotype"], errors="coerce")
    return pheno


def scan_vcf_for_targets(vcf_path: str, target_ids: set) -> tuple[list[str], dict]:
    """
    Scan VCF once and extract only target loci.
    Returns:
      samples: sample IDs from VCF header
      records: dict[locus_id] = {
          "chrom", "pos", "id", "ref", "alt", "sample_gts": {sample: gt_string}
      }
    """
    opener = gzip.open if vcf_path.endswith(".gz") else open
    samples = None
    records = {}

    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                samples = header[9:]
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue

            chrom, pos, vid, ref, alt, qual, filt, info, fmt = fields[:9]
            if vid not in target_ids:
                continue

            sample_fields = fields[9:]
            gt_map = {}
            for s, sf in zip(samples, sample_fields):
                gt_map[s] = sf.split(":")[0]

            records[vid] = {
                "chrom": chrom,
                "pos": int(pos),
                "id": vid,
                "ref": ref,
                "alt": alt,
                "format": fmt,
                "sample_gts": gt_map,
            }

            if len(records) == len(target_ids):
                break

    if samples is None:
        raise ValueError("VCF header not found")

    missing = target_ids - set(records.keys())
    if missing:
        print(f"[WARN] {len(missing)} target loci were not found in VCF:")
        for x in sorted(missing):
            print(f"  - {x}")

    return samples, records


def build_long_table(importance_top: pd.DataFrame, pheno_df: pd.DataFrame, vcf_records: dict, trait: str) -> pd.DataFrame:
    rows = []

    pheno_map = dict(zip(pheno_df["sample"], pheno_df["phenotype"]))

    for _, row in importance_top.iterrows():
        locus_id = row["locus_id"]
        if locus_id not in vcf_records:
            continue

        rec = vcf_records[locus_id]
        ref = rec["ref"]
        alt = rec["alt"]

        for sample, gt in rec["sample_gts"].items():
            genotype_label = normalize_genotype_label(ref, alt, gt)
            phenotype_value = pheno_map.get(sample, np.nan)

            rows.append({
                "trait": trait,
                "rank": int(row["rank"]),
                "locus_id": locus_id,
                "chr": row["chr"],
                "pos": int(row["pos"]),
                "importance_mean": float(row["importance_mean"]),
                "importance_std": float(row["importance_std"]) if "importance_std" in row and pd.notna(row["importance_std"]) else np.nan,
                "ref": ref,
                "alt": alt,
                "sample": sample,
                "gt_raw": gt,
                "genotype": genotype_label,
                "phenotype": phenotype_value,
            })

    out = pd.DataFrame(rows)
    return out


def build_summary_table(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    valid = long_df[long_df["genotype"] != "NA"].copy()

    locus_totals = (
        valid.groupby("locus_id")["sample"]
        .nunique()
        .rename("n_total")
        .reset_index()
    )

    counts = (
        valid.groupby(["trait", "rank", "locus_id", "chr", "pos", "ref", "alt", "genotype"])["sample"]
        .nunique()
        .rename("n")
        .reset_index()
    )

    counts = counts.merge(locus_totals, on="locus_id", how="left")
    counts["prop"] = counts["n"] / counts["n_total"] * 100.0
    counts = counts.sort_values(["rank", "n"], ascending=[True, False])

    return counts


def main():
    args = parse_args()

    out_prefix = args.out_prefix or f"{args.trait}.top{args.top_k}"

    print(f"[1/5] Reading top-{args.top_k} importance loci")
    importance_top = read_importance_topk(args.importance, args.top_k)
    target_ids = set(importance_top["locus_id"].tolist())

    print(f"[2/5] Reading phenotype: {args.trait}")
    pheno_df = read_pheno(args.pheno, args.trait)

    print("[3/5] Scanning VCF for target loci")
    _, vcf_records = scan_vcf_for_targets(args.vcf, target_ids)

    print("[4/5] Building sample-level long table")
    long_df = build_long_table(importance_top, pheno_df, vcf_records, args.trait)

    print("[5/5] Building genotype count summary")
    summary_df = build_summary_table(long_df)

    long_out = f"{out_prefix}.long.tsv"
    summary_out = f"{out_prefix}.genotype_summary.tsv"

    long_df.to_csv(long_out, sep="\t", index=False)
    summary_df.to_csv(summary_out, sep="\t", index=False)

    print(f"[OK] Long table saved: {long_out}")
    print(f"[OK] Summary table saved: {summary_out}")


if __name__ == "__main__":
    main()