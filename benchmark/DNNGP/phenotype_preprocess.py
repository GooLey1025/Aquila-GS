#!/usr/bin/env python3
"""
Split a phenotype TSV into per-trait TSV files.

Input pheno.tsv requirements:
- First row: header
- First column: sample IDs
- Subsequent columns: phenotype values

Outputs:
- One file per phenotype column under --out_dir
- Each output file has two columns:
    1) sample IDs (kept as-is)
    2) phenotype values (missing values imputed by the trait-wise mean)

Optional reference alignment:
- If --ref is provided, the output sample order will follow the first column of the ref file.
- By default, the reference file is assumed to HAVE a header.
- If the reference file has NO header, specify --no-header.

Missing handling:
- Missing values in phenotype: NA / NaN / empty / non-numeric -> treated as NaN, then filled with mean.
- If reference contains IDs not present in phenotype, those rows will be created and then filled by mean.
"""

import argparse
import os
import sys
import pandas as pd


def load_reference_ids(ref_path: str, no_header: bool) -> list[str]:
    """Load reference sample IDs from the first column."""
    try:
        if no_header:
            ref_df = pd.read_csv(ref_path, sep="\t", header=None, dtype=str)
            ref_ids = ref_df.iloc[:, 0].astype(str).tolist()
            print(f"[INFO] Loaded {len(ref_ids)} reference IDs (no header): {ref_path}")
        else:
            ref_df = pd.read_csv(ref_path, sep="\t", dtype=str)
            ref_ids = ref_df.iloc[:, 0].astype(str).tolist()
            print(f"[INFO] Loaded {len(ref_ids)} reference IDs (with header): {ref_path}")
        return ref_ids
    except Exception as e:
        raise RuntimeError(f"Failed to read reference file: {ref_path}\nError: {e}") from e


def safe_mean(series: pd.Series, trait_name: str) -> float:
    """Compute mean while guarding against all-NaN columns."""
    mean_val = series.mean()
    if pd.isna(mean_val):
        raise ValueError(
            f"Trait '{trait_name}' has all missing / non-numeric values after parsing; cannot impute mean."
        )
    return float(mean_val)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split phenotype TSV into per-trait files, optionally aligned to a reference sample order."
    )
    parser.add_argument("-i", "--input", required=True, help="Input phenotype TSV (header required).")
    parser.add_argument("-o", "--out_dir", required=True, help="Output directory for per-trait TSV files.")
    parser.add_argument(
        "-r", "--ref",
        default=None,
        help="Optional reference file for sample order (first column used). Default assumes header."
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Use this if the reference file has NO header (default: reference has header)."
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix for output filenames (e.g., 'pheno_')."
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix for output filenames (e.g., '_aligned')."
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------
    # Load phenotype table
    # -----------------------------
    try:
        df = pd.read_csv(args.input, sep="\t")
    except Exception as e:
        print(f"[ERROR] Failed to read phenotype file: {args.input}\n{e}", file=sys.stderr)
        return 1

    if df.shape[1] < 2:
        print("[ERROR] Phenotype TSV must have at least 2 columns: sample_id + >=1 phenotype.", file=sys.stderr)
        return 1

    sample_col = df.columns[0]
    trait_cols = list(df.columns[1:])

    # Ensure sample IDs are strings (avoid numeric coercion)
    df[sample_col] = df[sample_col].astype(str)

    print(f"[INFO] Phenotype file: {args.input}")
    print(f"[INFO] Sample ID column: '{sample_col}'")
    print(f"[INFO] Detected {len(trait_cols)} phenotype columns")

    # -----------------------------
    # Load reference IDs (optional)
    # -----------------------------
    ref_ids = None
    if args.ref:
        ref_ids = load_reference_ids(args.ref, args.no_header)

    # -----------------------------
    # Process each trait
    # -----------------------------
    for trait in trait_cols:
        sub = df[[sample_col, trait]].copy()

        # Convert phenotype to numeric; invalid values -> NaN
        sub[trait] = pd.to_numeric(sub[trait], errors="coerce")

        # Align to reference order if provided
        if ref_ids is not None:
            sub = sub.set_index(sample_col)
            sub = sub.reindex(ref_ids)  # IDs not found become NaN rows
            sub.reset_index(inplace=True)
            sub.columns = [sample_col, trait]

        # Impute missing with mean
        mean_val = safe_mean(sub[trait], trait)
        sub[trait] = sub[trait].fillna(mean_val)

        # Write output
        out_name = f"{args.prefix}{trait}{args.suffix}.tsv"
        out_path = os.path.join(args.out_dir, out_name)
        sub.to_csv(out_path, sep="\t", index=False)

        print(f"[OK] {trait}: mean={mean_val:.6f} -> {out_path}")

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())