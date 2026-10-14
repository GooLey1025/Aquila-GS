#!/usr/bin/env python3
"""Predicted versus observed BLUP correlation plots for the 120 rice lines.

Points are colored from 300lines_map.txt:
- japonica: green
- indica: pink

Predictions are shifted and scaled to the observed mean and standard deviation
before plotting. Pearson r is unchanged by that rescaling.
"""

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats


TRAIT_MAP = {
    "GYP_BLUP": ("YieldPerPlant_BLUP", "Grain yield per plant"),
    "HD_BLUP": ("HeadingDate_BLUP", "Heading date"),
}

BLACK = "#000000"
JAPONICA_COLOR = "#6FA87C"
INDICA_COLOR = "#D9798A"
LINE_COLOR = "#6E6E6E"
GROUP_PALETTE = {
    "japonica": JAPONICA_COLOR,
    "indica": INDICA_COLOR,
}

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


def load_metadata(meta_path):
    """Read indica/japonica labels for the 120-Rice samples."""
    meta = pd.read_csv(meta_path, sep="\t")
    required = {"Sample_ID", "Subspecies"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if "Rice-Pop" in meta.columns:
        meta = meta.loc[meta["Rice-Pop"].astype(str).str.strip() == "120-Rice"].copy()
    meta["Sample_ID"] = meta["Sample_ID"].astype(str).str.strip()
    meta["Group"] = meta["Subspecies"].astype(str).str.strip().str.lower()
    meta = meta.loc[meta["Group"].isin(["japonica", "indica"])].copy()
    print(f"[DATA] Metadata: {len(meta)} samples")
    print(meta["Group"].value_counts().to_string())
    return meta[["Sample_ID", "Group"]]


def prediction_column(preds, trait):
    if trait in preds.columns:
        return trait
    suffixed = f"{trait}_Pred"
    if suffixed in preds.columns:
        return suffixed
    return None


def load_tables(preds_path, blup_path):
    preds = pd.read_csv(preds_path, sep="\t")
    blup = pd.read_csv(blup_path, sep="\t")
    for frame, path in ((preds, preds_path), (blup, blup_path)):
        if "Sample_ID" not in frame.columns:
            raise ValueError(f"{path} has no Sample_ID column")
        frame["Sample_ID"] = frame["Sample_ID"].astype(str).str.strip()
    print(f"[DATA] Predictions: {len(preds)} samples")
    print(f"[DATA] BLUP: {len(blup)} samples")
    return preds, blup


def nice_limits_and_ticks(values, n_ticks=5):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    vmin = values.min()
    vmax = values.max()
    pad = abs(vmin) * 0.05 if vmin == vmax else (vmax - vmin) * 0.08
    if vmin == vmax and vmin == 0:
        pad = 1.0
    locator = MaxNLocator(nbins=n_ticks, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(vmin - pad, vmax + pad)
    return float(ticks.min()), float(ticks.max()), ticks


def plot_trait(merged, pred_col, blup_col, label, model_name, out_prefix):
    sub = merged[[blup_col, pred_col, "Group"]].dropna().copy()
    x_raw = sub[pred_col].to_numpy(dtype=float)
    y = sub[blup_col].to_numpy(dtype=float)
    x_scaled = (x_raw - x_raw.mean()) / x_raw.std(ddof=0) * y.std(ddof=0) + y.mean()
    sub["Pred_scaled"] = x_scaled
    r, p = stats.pearsonr(x_scaled, y)
    print(f"  {label:24s}  r = {r:+.4f}  p = {p:.2e}  n = {len(sub)}")

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    for group in ["japonica", "indica"]:
        group_frame = sub.loc[sub["Group"] == group]
        ax.scatter(
            group_frame["Pred_scaled"],
            group_frame[blup_col],
            s=28,
            alpha=0.60,
            color=GROUP_PALETTE[group],
            edgecolors="none",
            zorder=2,
            label=group.capitalize(),
        )

    slope = np.polyfit(x_scaled, y, 1)
    x_lower, x_upper, x_ticks = nice_limits_and_ticks(x_scaled)
    y_lower, y_upper, y_ticks = nice_limits_and_ticks(y)
    grid = np.linspace(x_lower, x_upper, 200)
    ax.plot(grid, np.polyval(slope, grid), color=LINE_COLOR, lw=1.8, ls="--", zorder=3)
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in x_ticks])
    ax.set_yticklabels([f"{tick:g}" for tick in y_ticks])
    ax.text(
        0.04, 0.92, f"$r$ = {r:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold",
    )
    ax.set_title(label, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel(f"Predicted BLUP value ({model_name})", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed BLUP value", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=9, markerscale=1.2)
    ax.tick_params(direction="out", length=4, width=1.0)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return r, p, len(sub)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True, help="Prediction TSV with Sample_ID")
    parser.add_argument("--blup", required=True, help="Observed BLUP TSV")
    parser.add_argument("--meta", required=True, help="300lines_map.txt")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    meta = load_metadata(args.meta)
    preds, blup = load_tables(args.preds, args.blup)
    merged = blup.merge(preds, on="Sample_ID", how="inner").merge(meta, on="Sample_ID")
    print(f"[MERGE] {len(merged)} samples")
    if merged.empty:
        raise ValueError("No samples overlap across predictions, BLUP, and the map")
    print(merged["Group"].value_counts().to_string())

    results = []
    for trait, (blup_col, label) in TRAIT_MAP.items():
        pred_col = prediction_column(preds, trait)
        if pred_col is None or blup_col not in blup.columns:
            print(f"  SKIP {trait}: prediction or BLUP column is missing")
            continue
        out_prefix = os.path.join(args.output_dir, f"fig_blup_correlation_{trait}")
        r, p, n = plot_trait(merged, pred_col, blup_col, label, args.model_name, out_prefix)
        results.append({
            "model": args.model_name,
            "trait": trait,
            "label": label,
            "pearson_r": r,
            "p_value": p,
            "n_samples": n,
        })
        print(f"  Saved: {out_prefix}.pdf/.png")

    if results:
        summary = pd.DataFrame(results)
        summary_path = os.path.join(args.output_dir, "correlation_summary.tsv")
        summary.to_csv(summary_path, sep="\t", index=False)
        print(summary.to_string(index=False))
        print(f"[WRITE] {summary_path}")


if __name__ == "__main__":
    main()
