#!/usr/bin/env python3
"""
Extract per-phenotype best metrics from training log
and report mean statistics.

Usage:
    python extract_best_metrics.py train.log out.tsv
"""

import re
import sys
from pathlib import Path

LOG = Path(sys.argv[1])
OUT = Path(sys.argv[2])

re_col = re.compile(r"Processing phenotype column\s+(\d+)", re.I)
re_es_epoch = re.compile(r"Early stopping triggered at epoch\s+(\d+)", re.I)

re_best_loss = re.compile(r"Best\s+Test\s+Loss:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)
re_best_r2   = re.compile(r"Best\s+R\s*[\u00b2^]?\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)
re_best_pcc  = re.compile(r"Best\s+PCC:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)


def mean_ignore_na(values):
    vals = [float(v) for v in values if v != "NA"]
    return f"{sum(vals)/len(vals):.4f}" if vals else "NA"


def main():
    rows = []
    cur = None

    with LOG.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re_col.search(line)
            if m:
                if cur:
                    rows.append(cur)
                cur = {
                    "phenotype_col": m.group(1),
                    "best_epoch": "NA",
                    "best_test_loss": "NA",
                    "best_r2": "NA",
                    "best_pcc": "NA",
                }
                continue

            if cur is None:
                continue

            if (m := re_es_epoch.search(line)):
                cur["best_epoch"] = m.group(1)
            elif (m := re_best_loss.search(line)):
                cur["best_test_loss"] = m.group(1)
            elif (m := re_best_r2.search(line)):
                cur["best_r2"] = m.group(1)
            elif (m := re_best_pcc.search(line)):
                cur["best_pcc"] = m.group(1)

    if cur:
        rows.append(cur)

    # sort by phenotype index
    rows.sort(key=lambda x: int(x["phenotype_col"]))

    # compute means
    mean_row = {
        "phenotype_col": "MEAN",
        "best_epoch": "NA",
        "best_test_loss": mean_ignore_na([r["best_test_loss"] for r in rows]),
        "best_r2": mean_ignore_na([r["best_r2"] for r in rows]),
        "best_pcc": mean_ignore_na([r["best_pcc"] for r in rows]),
    }

    # write output
    with OUT.open("w") as f:
        header = ["phenotype_col", "best_epoch", "best_test_loss", "best_r2", "best_pcc"]
        f.write("\t".join(header) + "\n")

        for r in rows:
            f.write("\t".join(r[k] for k in header) + "\n")

        f.write("\t".join(mean_row[k] for k in header) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_best_metrics.py train.log out.tsv")
        sys.exit(1)
    main()