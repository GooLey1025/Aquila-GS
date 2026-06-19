#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot model ablation results using automatically loaded results_summary.tsv files.
Journal-style two-panel bar plot for Pearson correlation and R².
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# Input / Output
# =============================================================================

BASE_DIR = "/home/gulei/projects/Aquila-GS/model_ablation/marker_ablation"

EXPERIMENTS = {
    "All Markers": "705rice_conv_mha.aquila-snp",
    "No GWAS": "705rice_conv_mha.noGWAS.aquila-snp",
    "No RiceNavi": "705rice_conv_mha.noRiceNavi.aquila-snp",
    "No GWAS & RiceNavi": "705rice_conv_mha.noGWAS_and_QTN.aquila-snp",
}

OUT_PREFIX = "model_ablation_journal_style"


# =============================================================================
# Data loading
# =============================================================================

def load_results_summary(exp_dir):
    """Load overall performance from results_summary.tsv."""
    summary_file = os.path.join(BASE_DIR, exp_dir, "results_summary.tsv")

    if not os.path.exists(summary_file):
        print(f"[WARNING] Missing file: {summary_file}")
        return None

    df = pd.read_csv(summary_file, sep="\t")

    if "type" not in df.columns:
        raise ValueError(f"'type' column not found in {summary_file}")

    overall = df[df["type"] == "overall"]

    if overall.empty:
        raise ValueError(f"No overall row found in {summary_file}")

    row = overall.iloc[0]

    required_cols = [
        "val_r_mean",
        "val_r_std",
        "val_r2_mean",
        "val_r2_std",
        "n_runs",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {summary_file}")

    return {
        "r": float(row["val_r_mean"]),
        "r_std": float(row["val_r_std"]),
        "r2": float(row["val_r2_mean"]),
        "r2_std": float(row["val_r2_std"]),
        "n_runs": int(row["n_runs"]),
    }


# =============================================================================
# Load all experiments
# =============================================================================

all_summary = {}

print("Loading ablation results...")
print("=" * 70)

for exp_name, exp_dir in EXPERIMENTS.items():
    result = load_results_summary(exp_dir)

    if result is not None:
        all_summary[exp_name] = result
        print(
            f"{exp_name:<22} "
            f"r = {result['r']:.4f} ± {result['r_std']:.4f}, "
            f"R² = {result['r2']:.4f} ± {result['r2_std']:.4f}, "
            f"n = {result['n_runs']}"
        )

print("=" * 70)

conditions = [c for c in EXPERIMENTS.keys() if c in all_summary]

if len(conditions) == 0:
    raise RuntimeError("No valid experiment results were loaded.")

if len(conditions) < len(EXPERIMENTS):
    missing = [c for c in EXPERIMENTS.keys() if c not in all_summary]
    print(f"[WARNING] Missing experiments: {missing}")


# =============================================================================
# Prepare plot data
# =============================================================================

labels = [
    "All\nmarkers",
    "No\nGWAS",
    "No\nRiceNavi",
    "No GWAS\n& RiceNavi",
]

labels = labels[:len(conditions)]

r_mean = np.array([all_summary[c]["r"] for c in conditions])
r_std = np.array([all_summary[c]["r_std"] for c in conditions])

r2_mean = np.array([all_summary[c]["r2"] for c in conditions])
r2_std = np.array([all_summary[c]["r2_std"] for c in conditions])


# =============================================================================
# Style
# =============================================================================

colors = [
    "#7FA6C3",
    "#2E6F95",
    "#74B39A",
    "#2E7F68",
]

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

mpl.rcParams["axes.linewidth"] = 1.4
mpl.rcParams["xtick.major.width"] = 1.3
mpl.rcParams["ytick.major.width"] = 1.3
mpl.rcParams["xtick.major.size"] = 5
mpl.rcParams["ytick.major.size"] = 5


# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(5.4, 5.8),
    sharex=True,
    gridspec_kw={"hspace": 0.12},
)

x = np.arange(len(conditions))
bar_width = 0.62


def smart_ylim(values, errors, lower_pad=0.015, upper_pad=0.012):
    ymin = np.min(values - errors)
    ymax = np.max(values + errors)
    span = ymax - ymin
    return ymin - span * lower_pad - 0.005, ymax + span * upper_pad + 0.008


panels = [
    {
        "ax": axes[0],
        "mean": r_mean,
        "std": r_std,
        "ylabel": "Pearson correlation (r)",
        "ylim": smart_ylim(r_mean, r_std),
    },
    {
        "ax": axes[1],
        "mean": r2_mean,
        "std": r2_std,
        "ylabel": r"Coefficient of determination ($R^2$)",
        "ylim": smart_ylim(r2_mean, r2_std),
    },
]

for panel in panels:
    ax = panel["ax"]
    mean = panel["mean"]
    std = panel["std"]
    ylabel = panel["ylabel"]
    ylim = panel["ylim"]

    ax.bar(
        x,
        mean,
        width=bar_width,
        color=colors[:len(x)],
        edgecolor="black",
        linewidth=1.1,
        zorder=3,
    )

    ax.errorbar(
        x,
        mean,
        yerr=std,
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=4,
    )

    offset = (ylim[1] - ylim[0]) * 0.018

    for i, v in enumerate(mean):
        ax.text(
            x[i],
            v + std[i] + offset,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(*ylim)

    ax.grid(
        axis="y",
        color="#E6E6E6",
        linewidth=0.9,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=10)
axes[1].set_xlabel("Marker ablation setting", fontsize=13, fontweight="bold")

plt.tight_layout()

png_file = os.path.join(BASE_DIR, f"{OUT_PREFIX}.png")
pdf_file = os.path.join(BASE_DIR, f"{OUT_PREFIX}.pdf")

plt.savefig(png_file, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(pdf_file, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\nSaved: {png_file}")
print(f"Saved: {pdf_file}")


# =============================================================================
# Print summary
# =============================================================================

print("\n" + "=" * 70)
print("Model Ablation Summary")
print("=" * 70)
print(f"{'Condition':<22} {'r':>10} {'r_SD':>10} {'R2':>10} {'R2_SD':>10} {'Delta_r':>10}")
print("-" * 70)

baseline = all_summary[conditions[0]]["r"]

for condition in conditions:
    r = all_summary[condition]["r"]
    r_sd = all_summary[condition]["r_std"]
    r2 = all_summary[condition]["r2"]
    r2_sd = all_summary[condition]["r2_std"]
    delta = r - baseline

    print(
        f"{condition:<22} "
        f"{r:>10.4f} "
        f"{r_sd:>10.4f} "
        f"{r2:>10.4f} "
        f"{r2_sd:>10.4f} "
        f"{delta:>+10.4f}"
    )

print("=" * 70)