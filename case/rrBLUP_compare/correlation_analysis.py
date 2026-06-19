#!/usr/bin/env python3
"""
rrBLUP Cross-Population Correlation Analysis

Correlates rrBLUP predictions on 120-inbred-line population
with BLUP phenotype values, and generates scatter + correlation plots.

Points are colored by subpopulation:
- Japonica: green
- Indica: pink
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy import stats
from matplotlib.ticker import MaxNLocator

import argparse
import os


# =============================================================================
# Trait mapping
# =============================================================================

TRAIT_MAP = {
    "PlantHeight": (
        "PlantHeight_BLUP",
        "Plant height"
    ),
    "HeadingDate": (
        "HeadingDate_BLUP",
        "Heading date"
    ),
    "YieldPerPlant": (
        "YieldPerPlant_BLUP",
        "Grain yield per plant"
    ),
}


# =============================================================================
# Colors
# =============================================================================

BLACK = "#000000"

JAPONICA_COLOR = "#6FA87C"
INDICA_COLOR = "#D9798A"
LINE_COLOR = "#6E6E6E"

GROUP_PALETTE = {
    "japonica": JAPONICA_COLOR,
    "indica": INDICA_COLOR,
}


# =============================================================================
# Matplotlib style
# =============================================================================

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,

    "xtick.color": BLACK,
    "ytick.color": BLACK,

    "text.color": BLACK,

    "axes.linewidth": 1.1,

    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,

    "xtick.major.size": 4,
    "ytick.major.size": 4,

    "xtick.direction": "out",
    "ytick.direction": "out",
})


# =============================================================================
# Data loading
# =============================================================================

def load_metadata(meta_path):
    """
    Metadata file:
    column 0 = Sample_ID
    column 4 = japonica / indica
    """

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta = pd.read_csv(meta_path, header=None)

    meta = meta.rename(columns={
        0: "Sample_ID",
        4: "Group"
    })

    meta["Sample_ID"] = (
        meta["Sample_ID"]
        .astype(str)
        .str.strip()
    )

    meta["Group"] = (
        meta["Group"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    meta = meta[
        meta["Group"].isin(["japonica", "indica"])
    ].copy()

    print(f"[DATA] Metadata: {len(meta)} samples")
    print("[DATA] Group counts:")
    print(meta["Group"].value_counts())

    return meta[["Sample_ID", "Group"]]


def load_data(preds_path, blup_path):
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    if not os.path.exists(blup_path):
        raise FileNotFoundError(f"BLUP file not found: {blup_path}")

    preds = pd.read_csv(preds_path, sep="\t")
    blup = pd.read_csv(blup_path, sep="\t")

    preds["Sample_ID"] = (
        preds["Sample_ID"]
        .astype(str)
        .str.strip()
    )

    blup["Sample_ID"] = (
        blup["Sample_ID"]
        .astype(str)
        .str.strip()
    )

    print(f"[DATA] Predictions: {len(preds)} samples")
    print(f"[DATA] BLUP: {len(blup)} samples")

    return preds, blup


def merge_data(preds, blup, meta):
    merged = (
        blup
        .merge(preds, on="Sample_ID", how="inner")
        .merge(meta, on="Sample_ID", how="inner")
    )

    print(f"[MERGE] Overlapping samples with group info: {len(merged)}")

    if len(merged) == 0:
        raise ValueError(
            "No overlapping samples found among predictions, BLUP file, and metadata. "
            "Please check Sample_ID matching."
        )

    print("[MERGE] Group counts:")
    print(merged["Group"].value_counts())

    return merged


# =============================================================================
# Axis helper
# =============================================================================

def nice_limits_and_ticks(values, n_ticks=5):
    """
    Generate natural-looking axis limits and ticks.
    Prefer integer-like / clean ticks instead of arbitrary decimals.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    vmin = values.min()
    vmax = values.max()

    if vmin == vmax:
        pad = abs(vmin) * 0.05 if vmin != 0 else 1.0
    else:
        pad = (vmax - vmin) * 0.08

    lower = vmin - pad
    upper = vmax + pad

    locator = MaxNLocator(
        nbins=n_ticks,
        steps=[1, 2, 2.5, 5, 10],
        integer=False
    )

    ticks = locator.tick_values(lower, upper)

    lower = ticks.min()
    upper = ticks.max()

    return lower, upper, ticks


# =============================================================================
# Plot
# =============================================================================

def plot_trait(merged, pred_col, blup_col, label, out_prefix):
    sub = merged[
        [blup_col, pred_col, "Group"]
    ].dropna().copy()

    x_raw = sub[pred_col].values
    y = sub[blup_col].values

    # Scale rrBLUP predictions to match original BLUP mean and std
    x_scaled = (
        (x_raw - x_raw.mean())
        / x_raw.std()
        * y.std()
        + y.mean()
    )

    sub["Pred_scaled"] = x_scaled

    r, p = stats.pearsonr(x_scaled, y)

    print(
        f"  {label:20s}  r = {r:+.4f}  "
        f"p = {p:.2e}  n = {len(sub)}"
    )

    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    # -------------------------------------------------------------------------
    # Scatter by group
    # -------------------------------------------------------------------------

    for group in ["japonica", "indica"]:

        gdf = sub[sub["Group"] == group]

        ax.scatter(
            gdf["Pred_scaled"],
            gdf[blup_col],

            s=28,
            alpha=0.60,

            color=GROUP_PALETTE[group],

            edgecolors="none",

            zorder=2,
        )

    # -------------------------------------------------------------------------
    # Regression line
    # -------------------------------------------------------------------------

    z = np.polyfit(x_scaled, y, 1)

    x_lower, x_upper, x_ticks = nice_limits_and_ticks(
        x_scaled,
        n_ticks=5
    )

    y_lower, y_upper, y_ticks = nice_limits_and_ticks(
        y,
        n_ticks=5
    )

    xp = np.linspace(x_lower, x_upper, 200)

    ax.plot(
        xp,
        np.polyval(z, xp),

        color=LINE_COLOR,

        lw=1.8,
        ls="--",

        zorder=3,
    )

    # -------------------------------------------------------------------------
    # Axis limits and ticks
    # -------------------------------------------------------------------------

    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xticklabels(
        [f"{v:g}" for v in x_ticks]
    )

    ax.set_yticklabels(
        [f"{v:g}" for v in y_ticks]
    )

    # -------------------------------------------------------------------------
    # Correlation annotation
    # -------------------------------------------------------------------------

    ax.text(
        0.04,
        0.92,

        f"$r$ = {r:.3f}",

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontsize=11,
        fontweight="bold",
    )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    ax.set_title(
        label,

        fontsize=13,
        fontweight="bold",

        pad=8
    )

    ax.set_xlabel(
        "Predicted BLUP value (rrBLUP)",

        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Observed BLUP value",

        fontsize=11,
        fontweight="bold"
    )

    # -------------------------------------------------------------------------
    # Style
    # -------------------------------------------------------------------------

    ax.tick_params(
        direction="out",
        length=4,
        width=1.0
    )

    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    ax.grid(False)

    plt.tight_layout()

    fig.savefig(
        f"{out_prefix}.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        f"{out_prefix}.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    return r, p, len(sub)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="rrBLUP cross-population correlation analysis"
    )

    parser.add_argument(
        "--preds",
        default="rrblup_output/preds.tsv",
        help="rrBLUP predictions TSV"
    )

    parser.add_argument(
        "--blup",
        default="blup_phenotype.tsv",
        help="BLUP reference phenotype TSV"
    )

    parser.add_argument(
        "--meta",
        default="inbred_line_120_to_download.csv",
        help="Metadata CSV; column 0 = Sample_ID, column 4 = japonica/indica"
    )

    parser.add_argument(
        "--output-dir",
        default="rrblup_output",
        help="Output directory"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    meta = load_metadata(args.meta)
    preds, blup = load_data(args.preds, args.blup)

    merged = merge_data(
        preds=preds,
        blup=blup,
        meta=meta
    )

    print("\n--- Correlation Results ---\n")

    results = []

    for trait, (blup_col, label) in TRAIT_MAP.items():

        if trait not in preds.columns:
            print(f"  SKIP {trait}: column not in predictions file")
            continue

        if blup_col not in blup.columns:
            print(
                f"  SKIP {trait}: BLUP column "
                f"'{blup_col}' not in BLUP file"
            )
            continue

        out_prefix = os.path.join(
            args.output_dir,
            f"fig_blup_correlation_{trait}"
        )

        r, p, n = plot_trait(
            merged=merged,
            pred_col=trait,
            blup_col=blup_col,
            label=label,
            out_prefix=out_prefix
        )

        results.append({
            "trait": label,
            "pearson_r": r,
            "p_value": p,
            "n_samples": n
        })

        print(f"  Saved: fig_blup_correlation_{trait}.pdf/.png")

    if results:
        res_df = pd.DataFrame(results)

        summary_file = os.path.join(
            args.output_dir,
            "correlation_summary.tsv"
        )

        res_df.to_csv(
            summary_file,
            sep="\t",
            index=False
        )

        print(f"\nSummary saved: {summary_file}")

        print("\n--- Summary Table ---")
        print(res_df.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()