#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select VCF samples with at least one non-missing phenotype "
            "in Wheat_All_traits_Matrix.xlsx and write bcftools -S sample list."
        )
    )
    parser.add_argument("vcf_in", help="Input VCF (.vcf.gz)")
    parser.add_argument("keep_list", help="Output sample list for bcftools -S")
    parser.add_argument(
        "--pheno-xlsx",
        default="Wheat_All_traits_Matrix.xlsx",
        help="Phenotype Excel (All_Data sheet)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    vcf_samples = subprocess.check_output(
        ["bcftools", "query", "-l", args.vcf_in], text=True
    ).splitlines()

    pheno = pd.read_excel(args.pheno_xlsx, sheet_name="All_Data")
    pheno = pheno.rename(columns={pheno.columns[0]: "LINE"})
    pheno["LINE"] = pheno["LINE"].astype(str)
    trait_cols = [c for c in pheno.columns if c != "LINE"]
    has_pheno = set(pheno.loc[pheno[trait_cols].notna().any(axis=1), "LINE"])

    keep = sorted(set(vcf_samples) & has_pheno)
    if not keep:
        raise SystemExit(
            "No samples with at least one phenotype found in the VCF."
        )

    Path(args.keep_list).write_text("".join(f"{sid}\n" for sid in keep))
    print(f"[INFO] VCF samples: {len(vcf_samples)}")
    print(f"[INFO] accessions with >=1 phenotype: {len(has_pheno)}")
    print(f"[INFO] keep samples written to {args.keep_list}: {len(keep)}")
    print(
        f"[INFO] dropped (no phenotype or absent from xlsx): "
        f"{len(vcf_samples) - len(keep)}"
    )


if __name__ == "__main__":
    main()
