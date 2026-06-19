#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

META_FILE = "inbred_line_120_to_download.csv"
PRED_FILE = "120_inbred_line.predictions.tsv"
OUT_FIG   = "GYP_4env_heatmap_by_subpop_stage_sorted_horizontal_bwstage.pdf"

GROUP_COL = 4

PRED_COLS = [
    "GYP_BeiJ15_Pred",
    "GYP_WenJ15_Pred",
    "GYP_YangZ15_Pred",
    "GYP_LingS15_Pred"
]

COL_RENAME = {
    "GYP_BeiJ15_Pred": "BeiJing",
    "GYP_WenJ15_Pred": "SiChuan-WenJiang",
    "GYP_YangZ15_Pred": "JiangSu-YangZhou",
    "GYP_LingS15_Pred": "HaiNan-LingShui"
}

GROUP_ORDER = ["japonica", "indica"]
STAGE_ORDER = ["1976–1999", "2000–2010", "2011–2020", "2021-2024"]
STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}

STAGE_COLORS = {
    "1976–1999": "#D9D9D9",
    "2000–2010": "#AFAFAF",
    "2011–2020": "#707070",
    "2021-2024": "#202020"
}

GROUP_BAND_COLORS = {
    "japonica": "#6FA87C",
    "indica": "#D9798A"
}

BLACK = "#000000"
WHITE = "#FFFFFF"

available_fonts = {f.name for f in fm.fontManager.ttflist}
if "Arial" not in available_fonts:
    raise RuntimeError(
        "Arial font was not found by Matplotlib. "
        "Please install Arial or configure Matplotlib font path first."
    )

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["axes.unicode_minus"] = False

sns.set_style("white")

# 强制所有默认文字/轴/刻度为纯黑
mpl.rcParams.update({
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "legend.edgecolor": BLACK,
    "legend.labelcolor": BLACK,

    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

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
        return "2021-2024"
    else:
        return None

meta = pd.read_csv(META_FILE, header=None)
meta = meta.rename(columns={
    0: "Sample_ID",
    3: "Year",
    GROUP_COL: "Group"
})

meta["Sample_ID"] = meta["Sample_ID"].astype(str).str.strip()
meta["Year"] = pd.to_numeric(meta["Year"], errors="coerce")
meta["Group"] = meta["Group"].astype(str).str.strip().str.lower()

pred = pd.read_csv(PRED_FILE, sep="\t")
pred["Sample_ID"] = pred["Sample_ID"].astype(str).str.strip()

missing_cols = [c for c in PRED_COLS if c not in pred.columns]
if missing_cols:
    raise ValueError(f"Missing columns in prediction file: {missing_cols}")

meta = meta[meta["Group"].isin(GROUP_ORDER)].copy()
meta["Stage"] = meta["Year"].apply(assign_stage)
meta = meta.dropna(subset=["Stage"]).copy()

df = pd.merge(
    meta[["Sample_ID", "Group", "Year", "Stage"]],
    pred[["Sample_ID"] + PRED_COLS],
    on="Sample_ID",
    how="inner"
)

df = df.dropna(subset=PRED_COLS).copy()

if df.empty:
    raise ValueError("Merged dataframe is empty. Please check Sample_ID matching and GROUP_COL.")

df["Mean_GYP"] = df[PRED_COLS].mean(axis=1)
df["Stage_rank"] = df["Stage"].map(STAGE_RANK)

panel_data = {}
panel_stage_bounds = {}
panel_stage_spans = {}
max_n = 0

for grp in GROUP_ORDER:
    sub = df[df["Group"] == grp].copy()

    sub = sub.sort_values(
        by=["Stage_rank", "Year", "Mean_GYP"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    heat_df = (
        sub.set_index("Sample_ID")[PRED_COLS]
        .rename(columns=COL_RENAME)
        .T
    )

    panel_data[grp] = heat_df
    max_n = max(max_n, heat_df.shape[1])

    bounds = []
    spans = []
    start = 0

    for stage in STAGE_ORDER:
        n = (sub["Stage"] == stage).sum()
        if n > 0:
            end = start + n
            bounds.append((stage, end))
            spans.append((stage, start, end))
            start = end

    panel_stage_bounds[grp] = bounds
    panel_stage_spans[grp] = spans

fig_w = max(11.0, 0.085 * max_n)
fig_h = 4.5

fig = plt.figure(figsize=(fig_w, fig_h))
gs = GridSpec(
    nrows=3,
    ncols=2,
    width_ratios=[40, 2.0],
    height_ratios=[1, 1, 0.13],
    wspace=0.02,
    hspace=0.20
)

ax_top = fig.add_subplot(gs[0, 0])
ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_band_top = fig.add_subplot(gs[0, 1])
ax_band_bottom = fig.add_subplot(gs[1, 1])
cax = fig.add_subplot(gs[2, 0])

axes = [ax_top, ax_bottom]
band_axes = [ax_band_top, ax_band_bottom]

cmap = sns.blend_palette(
    ["#3B4DA1", "#6FA8DC", "#DCE6F2", "#F3E7C3", "#F3B36A", "#E85C3B", "#B40426"],
    as_cmap=True
)

vmin = df[PRED_COLS].min().min()
vmax = df[PRED_COLS].max().max()

for i, grp in enumerate(GROUP_ORDER):
    ax = axes[i]
    band_ax = band_axes[i]
    heat_df = panel_data[grp]

    sns.heatmap(
        heat_df,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=(i == 1),
        cbar_ax=cax if i == 1 else None,
        linewidths=0.20,
        linecolor=WHITE,
        xticklabels=False,
        yticklabels=True,
        cbar_kws={"orientation": "horizontal"} if i == 1 else None
    )

    ax.set_xlabel("", color=BLACK)
    ax.set_ylabel("", color=BLACK)

    ax.tick_params(
        axis="y",
        labelsize=13,
        rotation=0,
        pad=7,
        colors=BLACK,
        labelcolor=BLACK
    )
    ax.tick_params(
        axis="x",
        length=0,
        colors=BLACK,
        labelcolor=BLACK
    )

    ax.set_yticklabels(
        heat_df.index.tolist(),
        rotation=0,
        fontfamily="Arial",
        color=BLACK
    )

    bounds = panel_stage_bounds[grp]
    for stage, end_col in bounds[:-1]:
        ax.vlines(
            x=end_col,
            ymin=0,
            ymax=heat_df.shape[0],
            colors=BLACK,
            linewidth=1.2,
            linestyles="-",
            zorder=5
        )

    spans = panel_stage_spans[grp]
    if i == 0:
        trans = ax.get_xaxis_transform()

        band_y = 1.08
        band_h = 0.18
        text_y = band_y + band_h / 2

        for stage, start, end in spans:
            rect = Rectangle(
                (start, band_y),
                end - start,
                band_h,
                transform=trans,
                facecolor=STAGE_COLORS[stage],
                edgecolor="none",
                clip_on=False,
                zorder=6
            )
            ax.add_patch(rect)

            txt_color = WHITE if stage in ["2011–2020", "2021-2024"] else BLACK

            ax.text(
                (start + end) / 2,
                text_y,
                stage,
                transform=trans,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=txt_color,
                fontfamily="Arial",
                zorder=8
            )

        for stage, end_col in bounds[:-1]:
            ax.plot(
                [end_col, end_col],
                [band_y, band_y + band_h],
                transform=trans,
                color=BLACK,
                linewidth=1.0,
                clip_on=False,
                zorder=7
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    band_ax.set_xlim(0, 1)
    band_ax.set_ylim(0, 1)
    band_ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=GROUP_BAND_COLORS[grp],
            edgecolor="none"
        )
    )

    band_ax.text(
        0.5,
        0.5,
        grp.capitalize(),
        ha="center",
        va="center",
        rotation=90,
        fontsize=16,
        color=WHITE,
        fontweight="bold",
        fontstyle="italic",
        fontfamily="Arial"
    )

    band_ax.set_xticks([])
    band_ax.set_yticks([])

    for spine in band_ax.spines.values():
        spine.set_visible(False)

cbar = axes[1].collections[0].colorbar
cbar.ax.tick_params(
    labelsize=14,
    length=3,
    colors=BLACK,
    labelcolor=BLACK
)

for label in cbar.ax.get_xticklabels():
    label.set_color(BLACK)
    label.set_fontfamily("Arial")

cbar.set_label(
    "Predicted grain yield per plant [g]",
    fontsize=14,
    fontweight="bold",
    labelpad=8,
    fontfamily="Arial",
    color=BLACK
)
cbar.outline.set_visible(False)

plt.subplots_adjust(
    left=0.08,
    right=0.96,
    top=0.92,
    bottom=0.13
)

plt.savefig(OUT_FIG, bbox_inches="tight")
plt.savefig(OUT_FIG.replace(".pdf", ".png"), dpi=600, bbox_inches="tight")
plt.close()

print("Done!")
print("Output:", OUT_FIG)

print("\nCounts by group:")
print(df["Group"].value_counts())

print("\nCounts by group and stage:")
print(df.groupby(["Group", "Stage"]).size())

print("\nTop rows after sorting:")
for grp in GROUP_ORDER:
    sub = df[df["Group"] == grp].copy()
    sub = sub.sort_values(
        by=["Stage_rank", "Year", "Mean_GYP"],
        ascending=[True, True, False]
    )
    print(f"\n[{grp}]")
    print(sub[["Sample_ID", "Stage", "Year", "Mean_GYP"] + PRED_COLS].head(10))