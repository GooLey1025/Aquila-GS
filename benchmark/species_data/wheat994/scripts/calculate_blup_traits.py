#!/usr/bin/env python3
"""Calculate one environment-adjusted BLUP column per wheat trait."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group phenotype columns by the text before the first underscore. "
            "Multi-environment traits are converted to line BLUPs; traits with "
            "one source column are retained unchanged."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pheno-xlsx",
        required=True,
        type=Path,
        help=(
            "Wheat_All_traits_Matrix.xlsx; its All_Data sheet defines samples "
            "with at least one non-missing original phenotype."
        ),
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="Optional TSV containing source columns and model diagnostics.",
    )
    parser.add_argument("--min-lines", type=int, default=100)
    parser.add_argument("--min-observations-per-environment", type=int, default=30)
    parser.add_argument("--min-repeated-lines", type=int, default=30)
    parser.add_argument("--min-environment-overlap", type=int, default=20)
    parser.add_argument(
        "--min-line-variance-fraction",
        type=float,
        default=0.01,
        help=(
            "Minimum line_variance / (line_variance + residual_variance). "
            "Traits below this threshold are excluded."
        ),
    )
    return parser.parse_args()


def group_trait_columns(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for column in columns:
        trait = column.split("_", 1)[0]
        groups.setdefault(trait, []).append(column)
    return groups


def assess_observation_structure(
    phenotype: pd.DataFrame,
    environment_columns: list[str],
    *,
    min_lines: int,
    min_observations_per_environment: int,
    min_repeated_lines: int,
    min_environment_overlap: int,
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    numeric = phenotype[environment_columns].apply(pd.to_numeric, errors="coerce")
    observed = numeric.notna()
    environment_counts = observed.sum(axis=0).astype(int)
    line_counts = observed.sum(axis=1)
    lines_observed = int((line_counts > 0).sum())
    repeated_lines = int((line_counts >= 2).sum())

    adjacency = {column: set() for column in environment_columns}
    pair_overlaps: list[int] = []
    for index, left in enumerate(environment_columns):
        for right in environment_columns[index + 1 :]:
            overlap = int((observed[left] & observed[right]).sum())
            pair_overlaps.append(overlap)
            if overlap >= min_environment_overlap:
                adjacency[left].add(right)
                adjacency[right].add(left)

    visited: set[str] = set()
    pending = [environment_columns[0]]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    environment_connected = len(visited) == len(environment_columns)

    reasons: list[str] = []
    if lines_observed < min_lines:
        reasons.append(f"lines_observed<{min_lines}")
    low_environments = [
        f"{column}:{count}"
        for column, count in environment_counts.items()
        if count < min_observations_per_environment
    ]
    if low_environments:
        reasons.append(
            "environment_observations_below_"
            f"{min_observations_per_environment}="
            + ",".join(low_environments)
        )
    if repeated_lines < min_repeated_lines:
        reasons.append(f"repeated_lines<{min_repeated_lines}")
    if not environment_connected:
        reasons.append(
            f"environment_graph_disconnected_at_overlap_{min_environment_overlap}"
        )

    details: dict[str, object] = {
        "observations": int(observed.sum().sum()),
        "lines_observed": lines_observed,
        "environment_count": len(environment_columns),
        "min_environment_observations": int(environment_counts.min()),
        "repeated_lines": repeated_lines,
        "minimum_pairwise_overlap": min(pair_overlaps) if pair_overlaps else np.nan,
        "environment_connected": environment_connected,
    }
    return numeric, details, reasons


def fit_blup(
    phenotype: pd.DataFrame,
    trait: str,
    environment_columns: list[str],
    structure_details: dict[str, object],
) -> tuple[pd.Series, dict[str, object]]:
    long_table = phenotype[["LINE", *environment_columns]].melt(
        id_vars="LINE",
        var_name="ENVIRONMENT",
        value_name="PHENOTYPE",
    )
    long_table["PHENOTYPE"] = pd.to_numeric(
        long_table["PHENOTYPE"], errors="coerce"
    )
    long_table = long_table.dropna(subset=["PHENOTYPE"]).copy()
    if long_table.empty:
        raise ValueError(f"{trait}: all source phenotype values are missing")
    if long_table["LINE"].nunique() < 2:
        raise ValueError(f"{trait}: fewer than two lines have observations")

    # Environment is fixed and line is random:
    # y_ij = mu + environment_j + line_i + error_ij.
    model = smf.mixedlm(
        "PHENOTYPE ~ C(ENVIRONMENT)",
        long_table,
        groups=long_table["LINE"],
        re_formula="1",
    )
    fit = None
    errors: list[str] = []
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = model.fit(
                    reml=True,
                    method=method,
                    maxiter=3000,
                    disp=False,
                )
            fit = candidate
            if candidate.converged:
                break
        except (np.linalg.LinAlgError, ValueError) as error:
            errors.append(f"{method}: {error}")
    if fit is None:
        raise RuntimeError(f"{trait}: model fitting failed ({'; '.join(errors)})")

    # Average the fixed environment means so the BLUP remains on the original
    # phenotype scale, then add each line's shrunken random intercept.
    environments = pd.DataFrame(
        {"ENVIRONMENT": sorted(long_table["ENVIRONMENT"].unique())}
    )
    overall_mean = float(fit.predict(environments).mean())
    line_variance = float(np.asarray(fit.cov_re)[0, 0])
    residual_variance = float(fit.scale)
    denominator = line_variance + residual_variance
    line_variance_fraction = (
        line_variance / denominator if denominator > 0 else np.nan
    )
    if not np.isfinite(line_variance) or line_variance <= 1e-10:
        raise ValueError("line_variance_zero")
    line_effects = {
        str(line): overall_mean + float(np.asarray(effect)[0])
        for line, effect in fit.random_effects.items()
    }
    values = phenotype["LINE"].map(line_effects)
    values.index = phenotype.index
    details: dict[str, object] = {
        "trait": trait,
        "output_column": f"{trait}_BLUP",
        "method": "mixed_model_blup",
        "source_column_count": len(environment_columns),
        "source_columns": ";".join(environment_columns),
        "lines_output": int(values.notna().sum()),
        "overall_mean": overall_mean,
        "line_variance": line_variance,
        "residual_variance": residual_variance,
        "line_variance_fraction": line_variance_fraction,
        "converged": bool(fit.converged),
        "optimizer": str(fit.method),
        **structure_details,
    }
    return values, details


def main() -> None:
    args = parse_args()
    phenotype = pd.read_csv(
        args.input,
        sep="\t",
        na_values=["NA", "NaN", "nan", ""],
        keep_default_na=True,
    )
    if phenotype.empty or len(phenotype.columns) < 2:
        raise ValueError("Input must contain a sample ID and phenotype columns")
    phenotype = phenotype.rename(columns={phenotype.columns[0]: "LINE"})
    phenotype["LINE"] = phenotype["LINE"].astype(str)
    if phenotype["LINE"].duplicated().any():
        raise ValueError("Input phenotype contains duplicated LINE identifiers")

    original = pd.read_excel(args.pheno_xlsx, sheet_name="All_Data")
    original = original.rename(columns={original.columns[0]: "LINE"})
    original["LINE"] = original["LINE"].astype(str)
    original_traits = [column for column in original.columns if column != "LINE"]
    original[original_traits] = original[original_traits].apply(
        pd.to_numeric, errors="coerce"
    )
    keep_ids = set(
        original.loc[original[original_traits].notna().any(axis=1), "LINE"]
    )
    phenotype = phenotype.loc[phenotype["LINE"].isin(keep_ids)].copy()
    if phenotype.empty:
        raise ValueError(
            "No input samples have a non-missing phenotype in the All_Data sheet"
        )

    trait_columns = [column for column in phenotype.columns if column != "LINE"]
    trait_groups = group_trait_columns(trait_columns)
    output = pd.DataFrame({"LINE": phenotype["LINE"]})
    diagnostics: list[dict[str, object]] = []
    included_count = 0
    excluded_count = 0

    for trait, columns in trait_groups.items():
        if len(columns) == 1:
            column = columns[0]
            values = pd.to_numeric(phenotype[column], errors="coerce")
            lines_observed = int(values.notna().sum())
            included = lines_observed >= args.min_lines
            details = {
                "trait": trait,
                "output_column": column,
                "status": "included" if included else "excluded",
                "exclusion_reason": (
                    "" if included else f"lines_observed<{args.min_lines}"
                ),
                "method": "retained_single_column",
                "source_column_count": 1,
                "source_columns": column,
                "observations": lines_observed,
                "lines_observed": lines_observed,
                "lines_output": lines_observed if included else 0,
                "environment_count": 1,
                "min_environment_observations": lines_observed,
                "repeated_lines": 0,
                "minimum_pairwise_overlap": np.nan,
                "environment_connected": True,
                "overall_mean": float(values.mean()),
                "line_variance": np.nan,
                "residual_variance": np.nan,
                "line_variance_fraction": np.nan,
                "converged": True,
                "optimizer": "not_applicable",
            }
            diagnostics.append(details)
            if included:
                output[column] = values
                included_count += 1
                print(f"[INFO] {trait}: retained single column {column}")
            else:
                excluded_count += 1
                print(
                    f"[EXCLUDED] {trait}: {details['exclusion_reason']}"
                )
            continue

        output_column = f"{trait}_BLUP"
        _, structure_details, reasons = assess_observation_structure(
            phenotype,
            columns,
            min_lines=args.min_lines,
            min_observations_per_environment=(
                args.min_observations_per_environment
            ),
            min_repeated_lines=args.min_repeated_lines,
            min_environment_overlap=args.min_environment_overlap,
        )
        base_details = {
            "trait": trait,
            "output_column": output_column,
            "method": "mixed_model_blup",
            "source_column_count": len(columns),
            "source_columns": ";".join(columns),
            **structure_details,
        }
        if reasons:
            diagnostics.append(
                {
                    **base_details,
                    "status": "excluded",
                    "exclusion_reason": ";".join(reasons),
                    "lines_output": 0,
                    "overall_mean": np.nan,
                    "line_variance": np.nan,
                    "residual_variance": np.nan,
                    "line_variance_fraction": np.nan,
                    "converged": False,
                    "optimizer": "not_fitted",
                }
            )
            excluded_count += 1
            print(f"[EXCLUDED] {trait}: {';'.join(reasons)}")
            continue

        try:
            values, details = fit_blup(
                phenotype, trait, columns, structure_details
            )
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as error:
            diagnostics.append(
                {
                    **base_details,
                    "status": "excluded",
                    "exclusion_reason": f"model_failure:{error}",
                    "lines_output": 0,
                    "overall_mean": np.nan,
                    "line_variance": np.nan,
                    "residual_variance": np.nan,
                    "line_variance_fraction": np.nan,
                    "converged": False,
                    "optimizer": "failed",
                }
            )
            excluded_count += 1
            print(f"[EXCLUDED] {trait}: model_failure:{error}")
            continue

        postfit_reasons: list[str] = []
        if not details["converged"]:
            postfit_reasons.append("model_not_converged")
        line_variance = float(details["line_variance"])
        variance_fraction = float(details["line_variance_fraction"])
        if not np.isfinite(line_variance) or line_variance <= 1e-10:
            postfit_reasons.append("line_variance_zero")
        elif (
            not np.isfinite(variance_fraction)
            or variance_fraction < args.min_line_variance_fraction
        ):
            postfit_reasons.append(
                "line_variance_fraction<"
                f"{args.min_line_variance_fraction:g}"
            )

        details["status"] = "excluded" if postfit_reasons else "included"
        details["exclusion_reason"] = ";".join(postfit_reasons)
        diagnostics.append(details)
        if postfit_reasons:
            excluded_count += 1
            print(f"[EXCLUDED] {trait}: {';'.join(postfit_reasons)}")
        else:
            output[output_column] = values
            included_count += 1
            print(
                f"[INFO] {trait}: {len(columns)} columns -> {output_column}; "
                f"{details['lines_output']} lines"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False, na_rep="NA")
    diagnostics_path = (
        args.diagnostics
        if args.diagnostics is not None
        else args.output.with_suffix(args.output.suffix + ".blup_model.tsv")
    )
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(
        diagnostics_path, sep="\t", index=False, na_rep="NA"
    )
    print(
        f"[INFO] wrote {len(output)} samples and {len(output.columns) - 1} "
        f"traits to {args.output}"
    )
    print(
        f"[INFO] included traits: {included_count}; "
        f"excluded traits: {excluded_count}"
    )
    print(f"[INFO] wrote model diagnostics to {diagnostics_path}")


if __name__ == "__main__":
    main()
