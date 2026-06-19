#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch
from scipy.ndimage import gaussian_filter1d


PERIOD_LABELS = ["1976–1999", "2000–2010", "2011–2020", "2021–2024"]

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--importance", required=True)
    p.add_argument("--trend", required=True)
    p.add_argument("-o", "--output-prefix", default="fig_importance_indica_selected")

    p.add_argument("--smooth", type=float, default=5)
    p.add_argument("--top-k", type=int, default=500)

    p.add_argument("--fig-width", type=float, default=6)
    p.add_argument("--fig-height", type=float, default=3)

    p.add_argument("--legend-fontsize", type=float, default=12)

    p.add_argument(
        "--indica-loci",
        default="SNP-1-2141520,SNP-2-24461527,SNP-10-2130431"
    )

    return p.parse_args()


def make_palette(base_rgb, factors=(0.35, 0.55, 0.75, 1.0)):
    base = np.array(base_rgb) / 255.0
    white = np.array([1.0, 1.0, 1.0])
    return [white * (1 - f) + base * f for f in factors]


INDICA_COLORS = make_palette((220, 127, 144))

CHR_COLORS = [
    np.array((81, 75, 75)) / 255.0,
    np.array((204, 206, 206)) / 255.0,
]


def load_importance(path, top_k):
    df = pd.read_csv(path, sep="\t")
    imp_col = "importance_mean" if "importance_mean" in df.columns else "importance"

    parts = df["locus_id"].str.split("-", expand=True)
    df["chr"] = parts[1].astype(int)
    df["pos"] = parts[2].astype(int)
    df = df.sort_values(["chr", "pos"]).copy()

    if "rank" in df.columns:
        keep = df["rank"] <= top_k
    else:
        keep = df[imp_col].rank(ascending=False, method="first") <= top_k

    df["importance_for_smooth"] = df[imp_col].where(keep, 0)
    return df


def build_genome_layout(df):
    all_chrs = sorted(df["chr"].unique())
    chr_offsets = {}
    chr_centers = []

    offset = 0
    for c in all_chrs:
        sub = df[df["chr"] == c]
        max_pos = sub["pos"].max()
        min_pos = sub["pos"].min()

        length = max_pos - min_pos
        if length <= 0:
            length = max_pos if max_pos > 0 else 1

        chr_offsets[c] = offset
        chr_centers.append((c, offset + length / 2))
        offset += length

    return all_chrs, chr_offsets, chr_centers


def add_cum_pos(df, chr_offsets):
    df = df.copy()
    df["cum_pos"] = df.apply(
        lambda r: int(r["pos"]) + chr_offsets[int(r["chr"])],
        axis=1
    )
    return df


def load_trend(path):
    df = pd.read_csv(path)
    df["locus_id"] = df.apply(
        lambda r: f"SNP-{int(r['chr'])}-{int(r['pos'])}",
        axis=1
    )
    return df


def parse_loci_arg(loci_arg):
    return [x.strip() for x in loci_arg.split(",") if x.strip()]


def select_indica_loci(trend_df, indica_loci_arg):
    wanted = parse_loci_arg(indica_loci_arg)
    order = {l: i for i, l in enumerate(wanted)}

    sub = trend_df[trend_df["locus_id"].isin(wanted)].copy()
    missing = [l for l in wanted if l not in set(sub["locus_id"])]

    if missing:
        print(f"[WARN] Missing Indica loci: {', '.join(missing)}")

    if sub.empty:
        raise ValueError("No Indica loci selected.")

    sub["__order"] = sub["locus_id"].map(order)
    sub = sub.sort_values("__order").drop(columns="__order")

    return sub


def draw_locus_panel(ax, row):
    w = 0.18

    ind_cols = [
        "ind_1976_1999_AF",
        "ind_2000_2010_AF",
        "ind_2011_2020_AF",
        "ind_2021_2024_AF",
    ]

    x = np.arange(4) * w

    for i, col in enumerate(ind_cols):
        v = row[col]
        if pd.notna(v) and v != 0:
            ax.bar(
                x[i],
                float(v),
                width=w,
                color=INDICA_COLORS[i],
                edgecolor="white",
                linewidth=0.45,
                zorder=3
            )

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([])

    ax.set_title(
        f"SNP-{int(row['chr'])}-{int(row['pos'])}",
        fontsize=11,
        fontweight="bold",
        pad=6
    )

    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=3, width=0.8)
    ax.grid(False)


def plot_importance_by_chr(ax, imp_df, all_chrs, y_smooth):
    imp_df = imp_df.copy()
    imp_df["y_smooth"] = y_smooth

    for i, chrom in enumerate(all_chrs):
        sub = imp_df[imp_df["chr"] == chrom].sort_values("cum_pos")
        color = CHR_COLORS[i % 2]

        ax.plot(
            sub["cum_pos"].values,
            sub["y_smooth"].values,
            color=color,
            linewidth=1.25,
            zorder=2
        )

        ax.fill_between(
            sub["cum_pos"].values,
            sub["y_smooth"].values,
            0,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1
        )


def add_indica_legend(fig, fontsize=12):

    ax_leg = fig.add_axes([0.04, 0.89, 0.88, 0.09])
    ax_leg.axis("off")

    y = 0.50

    title_x = 0.00
    x0 = 0.14
    dx = 0.24

    box_w = 0.030
    box_h = 0.18
    text_dx = 0.040

    ax_leg.text(
        title_x,
        y,
        "Indica",
        ha="left",
        va="center",
        fontsize=fontsize + 0.5,
        fontstyle="italic",
        fontweight="bold",
        family="Arial"
    )

    for i, label in enumerate(PERIOD_LABELS):

        x = x0 + i * dx

        ax_leg.add_patch(
            plt.Rectangle(
                (x, y - box_h / 2),
                box_w,
                box_h,
                facecolor=INDICA_COLORS[i],
                edgecolor="white",
                linewidth=0.4,
                transform=ax_leg.transAxes,
                clip_on=False
            )
        )

        ax_leg.text(
            x + text_dx,
            y,
            label,
            ha="left",
            va="center",
            fontsize=fontsize,
            family="Arial"
        )


def main():
    args = parse_args()

    imp_df = load_importance(args.importance, args.top_k)
    all_chrs, chr_offsets, chr_centers = build_genome_layout(imp_df)
    imp_df = add_cum_pos(imp_df, chr_offsets).sort_values("cum_pos")

    trend_df = load_trend(args.trend)

    loci_df = select_indica_loci(
        trend_df,
        args.indica_loci
    )
    loci_df = add_cum_pos(loci_df, chr_offsets)

    x = imp_df["cum_pos"].values
    y = imp_df["importance_for_smooth"].values

    if args.smooth > 0:
        y_smooth = gaussian_filter1d(y, sigma=args.smooth)
    else:
        y_smooth = y.copy()

    n = len(loci_df)

    fig = plt.figure(figsize=(args.fig_width, args.fig_height))

    gs = fig.add_gridspec(
        2, n,
        height_ratios=[0.8, 1.0],
        hspace=0.40,
        wspace=0.35
    )

    add_indica_legend(
        fig,
        fontsize=args.legend_fontsize
    )

    top_axes = []

    for i, (_, row) in enumerate(loci_df.iterrows()):
        ax = fig.add_subplot(gs[0, i])
        draw_locus_panel(ax, row)

        if i == 0:
            ax.set_ylabel("Allele\nfrequency", fontweight="bold", fontsize=9)
        else:
            ax.set_yticklabels([])

        top_axes.append(ax)

    ax_imp = fig.add_subplot(gs[1, :])

    plot_importance_by_chr(ax_imp, imp_df, all_chrs, y_smooth)

    ax_imp.set_xlim(0, max(x))
    ax_imp.set_ylim(0, 0.4)

    yticks = np.linspace(0, 0.4, 5)
    ax_imp.set_yticks(yticks)
    ax_imp.set_yticklabels([f"{v:.1f}" for v in yticks], fontsize=9)

    ax_imp.set_ylabel(
        "Importance\n(Gaussian smoothed)",
        fontsize=9,
        fontweight="bold"
    )

    ax_imp.set_xlabel(
        "Chromosome (Trait: BLUP of grain yield per plant)",
        fontweight="bold",
        fontsize=12
    )

    ax_imp.set_xticks([center for _, center in chr_centers])
    ax_imp.set_xticklabels(
        [str(chrom) for chrom, _ in chr_centers],
        fontsize=11
    )

    ax_imp.tick_params(axis="x", length=0, pad=4)
    ax_imp.tick_params(axis="y", direction="out", length=3, width=0.8)
    ax_imp.grid(False)

    marker_x = []
    marker_y = []

    for ax, (_, row) in zip(top_axes, loci_df.iterrows()):
        locus_x = float(row["cum_pos"])
        locus_y = float(np.interp(locus_x, x, y_smooth))

        marker_x.append(locus_x)
        marker_y.append(locus_y)

        con = ConnectionPatch(
            xyA=(0.5, -0.02),
            coordsA=ax.transAxes,
            xyB=(locus_x, locus_y),
            coordsB=ax_imp.transData,
            color="#666666",
            linewidth=0.75,
            alpha=0.72,
            linestyle="-",
            zorder=1,
            clip_on=False
        )

        ax_imp.add_artist(con)

    ax_imp.scatter(
        marker_x,
        marker_y,
        s=38,
        color="#c23b4b",
        edgecolor="white",
        linewidth=0.8,
        zorder=30,
        clip_on=False
    )

    fig.subplots_adjust(
        top=0.80,
        bottom=0.10,
        left=0.08,
        right=0.99
    )

    fig.savefig(f"{args.output_prefix}.pdf")
    fig.savefig(f"{args.output_prefix}.png")
    plt.close()

    print(f"Saved: {args.output_prefix}.pdf / .png")
    print(f"Shown loci: {', '.join(loci_df['locus_id'].tolist())}")


if __name__ == "__main__":
    main()