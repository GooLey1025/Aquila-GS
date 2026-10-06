#!/usr/bin/env python3
"""Compare published soybean BLUPs with independently fitted line BLUPs."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def fit_blup(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    long = frame[["LINE", *columns]].melt(
        id_vars="LINE", var_name="ENVIRONMENT", value_name="PHENOTYPE"
    )
    long["PHENOTYPE"] = pd.to_numeric(long["PHENOTYPE"], errors="coerce")
    long = long.dropna(subset=["PHENOTYPE"])
    model = smf.mixedlm(
        "PHENOTYPE ~ C(ENVIRONMENT)",
        long,
        groups=long["LINE"],
        re_formula="1",
    )
    fit = None
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = model.fit(
                    reml=True, method=method, maxiter=3000, disp=False
                )
            # A nominally converged boundary solution may not permit BLUP
            # extraction. In that case try the next optimizer.
            _ = candidate.random_effects
            fit = candidate
            if candidate.converged:
                break
        except (np.linalg.LinAlgError, ValueError):
            continue
    if fit is None:
        raise RuntimeError("model fitting failed")
    environments = pd.DataFrame(
        {"ENVIRONMENT": sorted(long["ENVIRONMENT"].unique())}
    )
    mean = float(fit.predict(environments).mean())
    effects = {
        str(line): mean + float(np.asarray(effect)[0])
        for line, effect in fit.random_effects.items()
    }
    return frame["LINE"].map(effects)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, sep="\t")
    frame["LINE"] = frame["LINE"].astype(str)
    groups: dict[str, list[str]] = {}
    for column in frame.columns:
        if column == "LINE":
            continue
        groups.setdefault(column.split("_", 1)[0], []).append(column)

    rows = []
    for trait, columns in groups.items():
        published_column = f"{trait}_BLUP"
        environments = [
            column for column in columns if column != published_column
        ]
        if published_column not in frame.columns or len(environments) < 2:
            continue
        try:
            recalculated = fit_blup(frame, environments)
            published = pd.to_numeric(
                frame[published_column], errors="coerce"
            )
            paired = pd.concat(
                [published.rename("published"), recalculated.rename("new")],
                axis=1,
            ).dropna()
            offset = float((paired["published"] - paired["new"]).mean())
            centered_error = paired["published"] - (paired["new"] + offset)
            rows.append(
                {
                    "trait": trait,
                    "environment_count": len(environments),
                    "paired_samples": len(paired),
                    "pearson_r": float(
                        paired["published"].corr(paired["new"])
                    ),
                    "mean_offset_published_minus_new": offset,
                    "centered_mae": float(centered_error.abs().mean()),
                    "status": "validated",
                    "message": "",
                }
            )
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as error:
            rows.append(
                {
                    "trait": trait,
                    "environment_count": len(environments),
                    "paired_samples": 0,
                    "pearson_r": np.nan,
                    "mean_offset_published_minus_new": np.nan,
                    "centered_mae": np.nan,
                    "status": "failed",
                    "message": str(error),
                }
            )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False, na_rep="NA")
    print(f"[INFO] wrote BLUP validation for {len(result)} traits to {args.output}")
    if not result.empty:
        valid = result.loc[result["status"].eq("validated")]
        print(
            f"[INFO] validated={len(valid)}, "
            f"median Pearson r={valid['pearson_r'].median():.6f}"
        )


if __name__ == "__main__":
    main()
