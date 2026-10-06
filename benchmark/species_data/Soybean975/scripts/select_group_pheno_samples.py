#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select VCF samples in a Group that have at least one non-NA phenotype, "
            "write bcftools -S sample list and filtered phenotype table."
        )
    )
    parser.add_argument("vcf_in", help="Input VCF (.vcf.gz)")
    parser.add_argument("info_xlsx", help="Sample info Excel (Line_info sheet)")
    parser.add_argument("pheno_in", help="Phenotype table (TSV, LINE column)")
    parser.add_argument("keep_list", help="Output sample list for bcftools -S")
    parser.add_argument("pheno_out", help="Output filtered phenotype table")
    parser.add_argument(
        "--group",
        default="Improved cultivar",
        help='Group value in info xlsx to keep (default: "Improved cultivar")',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    group_label = args.group

    info = pd.read_excel(args.info_xlsx, sheet_name="Line_info")
    group_ids = set(
        info.loc[info["Group"].astype(str) == group_label, "LINE"].astype(str)
    )

    pheno = pd.read_csv(args.pheno_in, sep="\t")
    pheno["LINE"] = pheno["LINE"].astype(str)
    trait_cols = [c for c in pheno.columns if c != "LINE"]
    pheno = pheno.loc[pheno[trait_cols].notna().any(axis=1)].copy()
    pheno_ids = set(pheno["LINE"])

    vcf_samples = subprocess.check_output(
        ["bcftools", "query", "-l", args.vcf_in], text=True
    ).splitlines()
    keep_vcf, keep_lines = [], []
    for sid in vcf_samples:
        line = sid.split("_")[0]
        if line in group_ids and line in pheno_ids:
            keep_vcf.append(sid)
            keep_lines.append(line)

    if not keep_vcf:
        raise SystemExit(
            f"No {group_label} samples with at least one phenotype found in the VCF."
        )

    Path(args.keep_list).write_text("\n".join(keep_vcf) + "\n")
    pheno.loc[pheno["LINE"].isin(keep_lines)].to_csv(
        args.pheno_out, sep="\t", index=False, na_rep="NA"
    )
    print(f"[INFO] VCF samples: {len(vcf_samples)}")
    print(f"[INFO] {group_label} in info: {len(group_ids)}")
    print(f"[INFO] Samples with >=1 phenotype: {len(pheno_ids)}")
    print(f"[INFO] Keep {group_label} + phenotype: {len(keep_vcf)} -> {args.keep_list}")
    print(f"[INFO] Wrote phenotype table: {args.pheno_out}")


if __name__ == "__main__":
    main()
