#!/usr/bin/env python3
"""
Convert TSV / CSV / Eigenvec files to PKL format for DNNGP.

When eigenvec or tsv are converted to PKL, the pandas version and model version
should be the same as DNNGP.

Usage:
    python tsv2pkl_CLI.py <input_file> [output_file]
"""

import pandas as pd
import sys
import os
from pathlib import Path


def convert_to_pkl(inpath, outpath=None):
    if not os.path.exists(inpath):
        raise FileNotFoundError(f"Input file not found: {inpath}")

    # auto output path
    if outpath is None:
        outpath = str(Path(inpath).with_suffix(".pkl"))

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)

    inpath_lower = inpath.lower()

    # ===============================
    # Eigenvec file
    # ===============================
    if "eigenvec" in inpath_lower:
        Gene = pd.read_csv(inpath, sep="\t", header=0, index_col=1)

        # Remove FID / IID columns if present
        for col in ["FID", "#FID", "IID", "#IID"]:
            if col in Gene.columns:
                del Gene[col]

        Gene.to_pickle(outpath)
        print(f"[OK] Eigenvec converted → {outpath}")

    # ===============================
    # CSV file
    # ===============================
    elif inpath_lower.endswith(".csv"):
        Gene = pd.read_csv(inpath, sep=",", header=0, index_col=0)
        Gene.to_pickle(outpath)
        print(f"[OK] CSV converted → {outpath}")

    # ===============================
    # TSV file (default)
    # ===============================
    else:
        Gene = pd.read_csv(inpath, sep="\t", header=0, index_col=0)
        Gene.to_pickle(outpath)
        print(f"[OK] TSV converted → {outpath}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    inpath = sys.argv[1]
    outpath = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        convert_to_pkl(inpath, outpath)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()