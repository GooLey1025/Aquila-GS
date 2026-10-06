#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Strip CUBIC_ from LD-pruned VCF sample names, keep samples shared "
            "with the phenotype table, and write a filtered pheno file."
        )
    )
    parser.add_argument("vcf_in", help="Input LD-pruned VCF (.vcf.gz)")
    parser.add_argument("pheno_in", help="Phenotype TSV with LINE column")
    parser.add_argument("vcf_out", help="Output renamed VCF (.vcf.gz)")
    parser.add_argument("pheno_out", help="Output phenotype TSV of shared samples")
    return parser.parse_args()


def to_pheno_id(sample_id: str) -> str:
    name = sample_id
    if name.startswith("0_"):
        name = name[2:]
    if name.startswith("CUBIC_"):
        name = name[len("CUBIC_") :]
    return name


def main():
    args = parse_args()
    vcf_in = Path(args.vcf_in)
    vcf_out = Path(args.vcf_out)
    map_path = Path("CUBIC_to_LINE.sample_map")
    keep_path = Path("shared.samples.txt")

    vcf_samples = subprocess.check_output(
        ["bcftools", "query", "-l", str(vcf_in)], text=True
    ).splitlines()

    pheno = pd.read_csv(args.pheno_in, sep="\t")
    if "LINE" not in pheno.columns:
        raise SystemExit(f"Missing LINE column in {args.pheno_in}")
    pheno["LINE"] = pheno["LINE"].astype(str)
    trait_cols = [c for c in pheno.columns if c != "LINE"]
    pheno = pheno.loc[pheno[trait_cols].notna().any(axis=1)].copy()
    pheno_ids = set(pheno["LINE"])

    rows = []
    keep_new = []
    seen_new = set()
    for old in vcf_samples:
        new = to_pheno_id(old)
        if new not in pheno_ids:
            continue
        if new in seen_new:
            raise SystemExit(f"Duplicate renamed sample ID: {new} (from {old})")
        seen_new.add(new)
        rows.append((old, new))
        keep_new.append(new)

    if not rows:
        raise SystemExit(
            "No shared samples after stripping CUBIC_ against the phenotype table."
        )

    map_path.write_text("".join(f"{old}\t{new}\n" for old, new in rows))
    keep_path.write_text("".join(f"{sid}\n" for sid in keep_new))

    tmp_vcf = vcf_out.with_suffix(".tmp.vcf.gz")
    subprocess.check_call(
        [
            "bcftools",
            "reheader",
            "-s",
            str(map_path),
            "-o",
            str(tmp_vcf),
            str(vcf_in),
        ]
    )
    subprocess.check_call(
        [
            "bcftools",
            "view",
            "-S",
            str(keep_path),
            "--force-samples",
            "-Oz",
            "-o",
            str(vcf_out),
            str(tmp_vcf),
        ]
    )
    tmp_vcf.unlink(missing_ok=True)
    subprocess.check_call(["bcftools", "index", "--force", str(vcf_out)])

    n_before = len(pheno)
    pheno = pheno.loc[pheno["LINE"].isin(seen_new)].copy()
    pheno.to_csv(args.pheno_out, sep="\t", index=False, na_rep="NA")

    print(f"[INFO] VCF samples: {len(vcf_samples)}")
    print(f"[INFO] phenotype samples with >=1 trait: {len(pheno_ids)}")
    print(f"[INFO] shared after stripping CUBIC_: {len(rows)}")
    print(f"[INFO] dropped from VCF: {len(vcf_samples) - len(rows)}")
    print(f"[INFO] pheno kept {len(pheno)} / {n_before}")
    print(f"[INFO] wrote {vcf_out}, {args.pheno_out}, {map_path}, {keep_path}")


if __name__ == "__main__":
    main()
