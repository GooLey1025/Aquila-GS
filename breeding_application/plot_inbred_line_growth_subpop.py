#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


# =============================================================================
# Input / Output
# =============================================================================

META_FILE = "inbred_line_120_to_download.csv"
PRED_FILE = "120_inbred_line.predictions.tsv"

OUT_FIG = "GYP_BLUP_stage_by_subpop_pseudo_group_violin.pdf"

TRAIT_COL = "GYP_BLUP_Pred"
GROUP_COL = 4


# =============================================================================
# Colors
# =============================================================================

BLACK = "#000000"
WHITE = "#FFFFFF"

palette = {
    "japonica": "#6FA87C",
    "indica": "#D9798A"
}


# =============================================================================
# Matplotlib style
# =============================================================================

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

mpl.rcParams["axes.unicode_minus"] = False

sns.set_style("white")

mpl.rcParams.update({
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "legend.edgecolor": BLACK,
    "legend.labelcolor": BLACK,

    "axes.linewidth": 1.1,

    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,

    "xtick.major.size": 4,
    "ytick.major.size": 4,

    "xtick.direction": "out",
    "ytick.direction": "out",
})


# =============================================================================
# Load metadata
# =============================================================================

meta = pd.read_csv(META_FILE, header=None)

meta = meta.rename(columns={
    0: "Sample_ID",
    3: "Year",
    GROUP_COL: "Group"
})

meta["Sample_ID"] = meta["Sample_ID"].astype(str).str.strip()

meta["Year"] = pd.to_numeric(
    meta["Year"],
    errors="coerce"
)

meta["Group"] = (
    meta["Group"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# =============================================================================
# Load prediction
# =============================================================================

pred = pd.read_csv(
    PRED_FILE,
    sep="\t"
)

pred["Sample_ID"] = (
    pred["Sample_ID"]
    .astype(str)
    .str.strip()
)

if TRAIT_COL not in pred.columns:
    raise ValueError(
        f"Cannot find trait column: {TRAIT_COL}\n"
        f"Available columns:\n{list(pred.columns)}"
    )


# =============================================================================
# Keep Japonica / Indica
# =============================================================================

meta = meta[
    meta["Group"].isin(["japonica", "indica"])
].copy()


# =============================================================================
# Assign stage
# =============================================================================

def assign_stage(year):

    if pd.isna(year):
        return None

    elif 1976 <= year <= 1999:
        return "1976–1999"

    elif 2000 <= year <= 2010:
        return "2000–2010"

    elif 2011 <= year <= 2020:
        return "2011–2020"

    elif year > 2020:
        return "2021–2024"

    else:
        return None


stage_order = [
    "1976–1999",
    "2000–2010",
    "2011–2020",
    "2021–2024"
]

group_order = [
    "japonica",
    "indica"
]

meta["Stage"] = meta["Year"].apply(assign_stage)

meta = meta.dropna(subset=["Stage"]).copy()

meta["Stage"] = pd.Categorical(
    meta["Stage"],
    categories=stage_order,
    ordered=True
)

meta["Group"] = pd.Categorical(
    meta["Group"],
    categories=group_order,
    ordered=True
)


# =============================================================================
# Merge
# =============================================================================

df = pd.merge(
    meta[["Sample_ID", "Year", "Stage", "Group"]],
    pred[["Sample_ID", TRAIT_COL]],
    on="Sample_ID",
    how="inner"
)

df = df.dropna(subset=[TRAIT_COL]).copy()

if df.empty:
    raise ValueError(
        "Merged dataframe is empty.\n"
        "Please check Sample_ID matching and GROUP_COL."
    )


# =============================================================================
# Stage + Group order
# =============================================================================

pair_order = []

for stage in stage_order:
    for group in group_order:
        pair_order.append(f"{stage}|{group}")

df["Stage_Group"] = (
    df["Stage"].astype(str)
    + "|"
    + df["Group"].astype(str)
)

df = df[
    df["Stage_Group"].isin(pair_order)
].copy()

df["Stage_Group"] = pd.Categorical(
    df["Stage_Group"],
    categories=pair_order,
    ordered=True
)

pair_palette = {
    f"{stage}|{group}": palette[group]
    for stage in stage_order
    for group in group_order
}


# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 2.8))


# -----------------------------------------------------------------------------
# Violin
# -----------------------------------------------------------------------------

sns.violinplot(
    data=df,
    x="Stage_Group",
    y=TRAIT_COL,
    order=pair_order,
    palette=pair_palette,
    inner=None,
    cut=0,
    linewidth=1.0,
    width=0.82,
    saturation=1,
    ax=ax
)

for coll in ax.collections:
    try:
        coll.set_alpha(0.72)
        coll.set_edgecolor(BLACK)
        coll.set_linewidth(0.8)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Boxplot
# -----------------------------------------------------------------------------

sns.boxplot(
    data=df,
    x="Stage_Group",
    y=TRAIT_COL,
    order=pair_order,
    width=0.18,
    showcaps=True,
    showfliers=False,

    boxprops=dict(
        facecolor=WHITE,
        edgecolor=BLACK,
        linewidth=0.95,
        alpha=1.0
    ),

    whiskerprops=dict(
        color=BLACK,
        linewidth=0.95
    ),

    capprops=dict(
        color=BLACK,
        linewidth=0.95
    ),

    medianprops=dict(
        color=BLACK,
        linewidth=1.2
    ),

    ax=ax
)


# -----------------------------------------------------------------------------
# Strip plot
# -----------------------------------------------------------------------------

sns.stripplot(
    data=df,
    x="Stage_Group",
    y=TRAIT_COL,
    order=pair_order,

    hue="Group",
    palette=palette,

    dodge=False,

    size=3.0,
    alpha=0.30,

    jitter=0.18,

    edgecolor="none",

    ax=ax,
    zorder=3
)

if ax.legend_ is not None:
    ax.legend_.remove()


# -----------------------------------------------------------------------------
# Mean marker
# -----------------------------------------------------------------------------

mean_df = (
    df.groupby("Stage_Group", observed=True)[TRAIT_COL]
    .mean()
    .reindex(pair_order)
)

ax.scatter(
    range(len(pair_order)),
    mean_df.values,

    marker="D",
    s=38,

    color=WHITE,
    edgecolor=BLACK,
    linewidth=0.7,

    zorder=5
)


# =============================================================================
# X axis
# =============================================================================

ax.set_xticks(range(len(pair_order)))
ax.set_xticklabels([""] * len(pair_order))

centers = [0.5, 2.5, 4.5, 6.5]

for center, stage in zip(centers, stage_order):

    ax.text(
        center,
        -0.02,

        stage,

        transform=ax.get_xaxis_transform(),

        ha="center",
        va="top",

        fontsize=12,
        fontweight="bold",

        color=BLACK
    )


# -----------------------------------------------------------------------------
# Separator lines
# -----------------------------------------------------------------------------

for x in [1.5, 3.5, 5.5]:

    ax.axvline(
        x=x,

        color=BLACK,

        linestyle="-",
        linewidth=0.55,

        alpha=1.0,
        zorder=0
    )


# =============================================================================
# Legend
# =============================================================================

legend_handles = [

    Patch(
        facecolor=palette["japonica"],
        edgecolor=BLACK,
        linewidth=0.8,
        label="Japonica"
    ),

    Patch(
        facecolor=palette["indica"],
        edgecolor=BLACK,
        linewidth=0.8,
        label="Indica"
    )
]

legend = ax.legend(
    handles=legend_handles,

    frameon=False,

    fontsize=12,
    title_fontsize=12,

    loc="lower right"
)

legend.get_title().set_ha("left")
legend._legend_box.align = "left"

for text in legend.get_texts():

    text.set_color(BLACK)

    text.set_fontstyle("italic")
    text.set_fontweight("bold")


# =============================================================================
# Axis labels (BOLD)
# =============================================================================

ax.set_xlabel(
    "",
    fontsize=12,
    fontweight="bold",
    color=BLACK
)

ax.set_ylabel(
    "Predicted grain yield per plant (BLUP) [g]",

    fontsize=11,
    fontweight="bold",

    labelpad=8,

    color=BLACK
)


# =============================================================================
# Tick style
# =============================================================================

ax.tick_params(
    axis="x",
    which="major",

    length=0,
    pad=2,

    colors=BLACK,
    labelcolor=BLACK
)

ax.tick_params(
    axis="y",
    which="major",

    direction="out",

    length=4,
    width=1.0,

    left=True,
    right=False,

    labelsize=12,

    colors=BLACK,
    labelcolor=BLACK
)

# =============================================================================
# Y-axis range + manual ticks
# =============================================================================

ax.set_ylim(20, 28)

ticks = [20, 22, 24, 26, 28]

ax.set_yticks(ticks)

ax.set_yticklabels(
    [str(t) for t in ticks],
    fontsize=12
)


# =============================================================================
# Final style
# =============================================================================

ax.grid(False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color(BLACK)
ax.spines["bottom"].set_color(BLACK)

ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)


# =============================================================================
# Layout
# =============================================================================

plt.subplots_adjust(
    bottom=0.20,
    left=0.11,
    right=0.96,
    top=0.97
)


# =============================================================================
# Save
# =============================================================================

plt.savefig(
    OUT_FIG,
    bbox_inches="tight"
)

plt.savefig(
    OUT_FIG.replace(".pdf", ".png"),
    dpi=600,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# Summary
# =============================================================================

print("Done!")
print("Output:", OUT_FIG)

print("\nCounts by Stage and Group:")

print(
    df.groupby(
        ["Stage", "Group"],
        observed=True
    ).size().unstack(fill_value=0)
)

print("\nSummary statistics:")

print(
    df.groupby(
        ["Stage", "Group"],
        observed=True
    )[TRAIT_COL].describe()
)