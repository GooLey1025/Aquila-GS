#!/usr/bin/env python3
"""Multi-environment phenotype BLUPs for the 3605-rice trial table.

A trait is modelled only when it was measured in at least two locations and
two years. The mixed model is

    y = environment + genotype + residual

with environment fixed and genotype random. Variance components are
method-of-moments estimates, and genotype effects are the corresponding
Henderson BLUPs. The reported value is the
unweighted mean of the environment effects plus the genotype BLUP, on the
original measurement scale. Traits that are not multi-year and multi-location
are copied through unchanged.
"""

import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


COLUMN_PATTERN = re.compile(r"^(.*)_([^_]+)_(\d{4})$")


def _variance_components(y, ei, gi, n_env, n_geno, xy):
    """Method-of-moments variances after removing fixed environment means."""
    env_mean = xy / n_env
    adjusted = y - env_mean[ei]
    geno_sum = np.bincount(gi, weights=adjusted, minlength=len(n_geno))
    geno_mean = geno_sum / n_geno
    within = adjusted - geno_mean[gi]
    degrees = len(y) - len(n_geno)
    ve = float(within @ within) / degrees if degrees > 0 else float(np.var(y))
    ve = max(ve, 1e-8)
    weights = n_geno / n_geno.sum()
    centered = geno_mean - np.sum(weights * geno_mean)
    between = float(np.sum(n_geno * centered ** 2))
    denominator = float(n_geno.sum() - np.sum(n_geno ** 2) / n_geno.sum())
    vu = max((between - ve * (len(n_geno) - 1)) / denominator, 1e-8)
    return vu, ve


def _solve_blup(lam, ei, gi, n_env, n_geno, xz, xy, zy):
    p = len(n_env)
    q = len(n_geno)
    size = p + q
    coeff = np.empty((size, size), dtype=np.float64)
    coeff[:p, :p] = np.diag(n_env)
    coeff[:p, p:] = xz
    coeff[p:, :p] = xz.T
    coeff[p:, p:] = np.diag(n_geno + lam)
    rhs = np.concatenate([xy, zy])
    factor = cho_factor(coeff, lower=True, check_finite=False)
    theta = cho_solve(factor, rhs, check_finite=False)
    return theta[:p], theta[p:]


def genotype_blup(values, environments, genotypes):
    """Return genotype BLUPs and fitted variance components."""
    y = np.asarray(values, dtype=np.float64)
    env_levels, ei = np.unique(environments, return_inverse=True)
    geno_levels, gi = np.unique(genotypes, return_inverse=True)
    p = len(env_levels)
    q = len(geno_levels)
    if p < 2 or q < 3 or len(y) <= p + 2:
        raise ValueError("Too few observations for a multi-environment BLUP")

    n_env = np.bincount(ei, minlength=p).astype(np.float64)
    n_geno = np.bincount(gi, minlength=q).astype(np.float64)
    xz = np.zeros((p, q), dtype=np.float64)
    np.add.at(xz, (ei, gi), 1.0)
    xy = np.bincount(ei, weights=y, minlength=p).astype(np.float64)
    zy = np.bincount(gi, weights=y, minlength=q).astype(np.float64)
    vu, ve = _variance_components(y, ei, gi, n_env, n_geno, xy)
    beta, geno = _solve_blup(ve / vu, ei, gi, n_env, n_geno, xz, xy, zy)
    blup = pd.Series(float(np.mean(beta)) + geno, index=geno_levels, dtype=float)
    info = {
        "n_obs": int(len(y)),
        "n_genotypes": int(q),
        "n_environments": int(p),
        "Vu": float(vu),
        "Ve": float(ve),
    }
    return blup, info


def _column_groups(columns):
    grouped = defaultdict(list)
    for column in columns:
        match = COLUMN_PATTERN.match(str(column))
        if match:
            grouped[match.group(1)].append(column)
    return grouped


def _observed_design(frame, columns):
    pieces = []
    for column in columns:
        location, year = column.rsplit("_", 2)[-2:]
        piece = frame.loc[frame[column].notna(), ["SampleID", column]].copy()
        if piece.empty:
            continue
        piece = piece.rename(columns={column: "value"})
        piece["location"] = location
        piece["year"] = year
        piece["environment"] = location + "_" + year
        pieces.append(piece)
    if not pieces:
        return None
    return pd.concat(pieces, ignore_index=True)


def build_blup_table(phenotype):
    """Return the BLUP table and a log of which traits were modelled."""
    groups = _column_groups(phenotype.columns[1:])
    modelled = {}
    log_rows = []
    for trait, columns in groups.items():
        long = _observed_design(phenotype, columns)
        if long is None:
            log_rows.append(
                {"trait": trait, "action": "kept_raw", "reason": "no observed values"}
            )
            continue
        n_locations = long["location"].nunique()
        n_years = long["year"].nunique()
        if n_locations < 2 or n_years < 2:
            log_rows.append(
                {
                    "trait": trait,
                    "action": "kept_raw",
                    "reason": f"{n_locations} location(s), {n_years} year(s)",
                    "n_obs": int(len(long)),
                }
            )
            continue
        blup, info = genotype_blup(
            long["value"].to_numpy(),
            long["environment"].to_numpy(),
            long["SampleID"].to_numpy(),
        )
        modelled[trait] = blup
        log_rows.append({"trait": trait, "action": "blup", **info})
        print(
            f"[BLUP] {trait}: n={info['n_obs']} "
            f"genotypes={info['n_genotypes']} envs={info['n_environments']} "
            f"Vu={info['Vu']:.4g} Ve={info['Ve']:.4g}",
            flush=True,
        )

    output = pd.DataFrame({"Sample_ID": phenotype["SampleID"].astype(str)})
    for trait in modelled:
        output[f"{trait}_BLUP"] = (
            output["Sample_ID"].map(modelled[trait]).astype(float)
        )
    raw_columns = []
    for column in phenotype.columns[1:]:
        match = COLUMN_PATTERN.match(str(column))
        if match and match.group(1) in modelled:
            continue
        raw_columns.append(column)
    for column in raw_columns:
        output[column] = phenotype[column].to_numpy()
    return output, pd.DataFrame(log_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenotype", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--log",
        default=None,
        help="TSV log of modelled versus copied traits",
    )
    parser.add_argument(
        "--samples",
        default=None,
        help="Optional sample-ID list. Writes --extract for these IDs.",
    )
    parser.add_argument("--extract", default=None)
    args = parser.parse_args()

    phenotype = pd.read_csv(
        args.phenotype, sep="\t", na_values=["NA", "NaN", "nan", ""]
    )
    phenotype.columns = [str(column).strip() for column in phenotype.columns]
    if "SampleID" not in phenotype.columns:
        raise ValueError("Phenotype file must contain a SampleID column")
    phenotype["SampleID"] = phenotype["SampleID"].astype(str).str.strip()
    if phenotype["SampleID"].duplicated().any():
        raise ValueError("Phenotype SampleID values are not unique")
    for column in phenotype.columns[1:]:
        phenotype[column] = pd.to_numeric(phenotype[column], errors="coerce")

    table, log = build_blup_table(phenotype)
    table.to_csv(args.output, sep="\t", index=False, na_rep="NA")
    log_path = args.log or str(args.output) + ".log.tsv"
    log.to_csv(log_path, sep="\t", index=False)
    print(f"[WRITE] {args.output} ({table.shape[0]} samples, {table.shape[1] - 1} columns)")
    print(f"[WRITE] {log_path}")

    if args.samples or args.extract:
        if not args.samples or not args.extract:
            raise ValueError("--samples and --extract must be given together")
        wanted = [
            line.strip()
            for line in open(args.samples, encoding="utf-8")
            if line.strip()
        ]
        extracted = table.set_index("Sample_ID").reindex(wanted).reset_index()
        missing = int(extracted.drop(columns="Sample_ID").isna().all(axis=1).sum())
        if missing:
            raise ValueError(f"{missing} requested samples are absent from the phenotype")
        extracted.to_csv(args.extract, sep="\t", index=False, na_rep="NA")
        print(f"[WRITE] {args.extract} ({len(extracted)} samples)")


if __name__ == "__main__":
    main()
