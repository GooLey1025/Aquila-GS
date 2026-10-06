#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write phenotype table for samples present in the LD-pruned VCF "
            "with at least one non-missing trait."
        )
    )
    parser.add_argument("prefix", help="Sample prefix, e.g. Wheat850")
    parser.add_argument(
        "--pheno-xlsx",
        default="Wheat_All_traits_Matrix.xlsx",
        help="Phenotype Excel (All_Data sheet)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vcf_in = Path(f"{args.prefix}.LD.vcf.gz")
    pheno_out = Path(f"{args.prefix}.pheno")

    if not Path(str(vcf_in) + ".tbi").exists() and not Path(str(vcf_in) + ".csi").exists():
        subprocess.check_call(["bcftools", "index", "--csi", "--force", str(vcf_in)])

    vcf_ids = set(
        subprocess.check_output(["bcftools", "query", "-l", str(vcf_in)], text=True).splitlines()
    )

    pheno = pd.read_excel(args.pheno_xlsx, sheet_name="All_Data")
    pheno = pheno.rename(columns={pheno.columns[0]: "LINE"})
    pheno["LINE"] = pheno["LINE"].astype(str)
    trait_cols = [c for c in pheno.columns if c != "LINE"]
    n_xlsx = len(pheno)
    pheno = pheno.loc[pheno[trait_cols].notna().any(axis=1)].copy()
    n_with_trait = len(pheno)
    pheno = pheno.loc[pheno["LINE"].isin(vcf_ids)].copy()
    pheno.to_csv(pheno_out, sep="\t", index=False, na_rep="NA")

    print(f"[INFO] xlsx samples: {n_xlsx}")
    print(f"[INFO] with >=1 trait: {n_with_trait}")
    print(f"[INFO] also in {vcf_in}: {len(pheno)}")
    print(f"[INFO] wrote {pheno_out}")


if __name__ == "__main__":
    main()
