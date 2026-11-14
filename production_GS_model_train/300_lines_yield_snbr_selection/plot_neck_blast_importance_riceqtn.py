import argparse
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Input / Output
# =============================================================================

IMPORTANCE_FILE = "/home/gulei/projects/Aquila-GS/case/gwas_interpretability/Rice300.leafblast.gwas_interpretability.ScaleOfLeafBlastResistance_BLUP.ensemble.position_importance/importance_ranking_ScaleOfLeafBlastResistance_BLUP.tsv"
RICENAVI_FILE = "/home/gulei/projects/Aquila-GS/evolve/ricenavi_319site_evaluation.csv"

OUT_PREFIX = "LeafBlastResistance_BLUP_RiceNavi_gene_label"

TOP_K = 500
SMOOTH_SIGMA = 5
TOLERANCE_BP = 100_000

FIG_WIDTH = 6
FIG_HEIGHT = 1.6

YMAX = 0.25
LABEL_Y_THRESHOLD = 0.02


# =============================================================================
# Style
# =============================================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


CHR_COLORS = [
    np.array((81, 75, 75)) / 255.0,
    np.array((204, 206, 206)) / 255.0,
]


# =============================================================================
# Functions
# =============================================================================

def parse_locus_id(x):
    """
    SNP-9-9544724-1 -> chr=9, pos=9544724
    SNP-9-9544724   -> chr=9, pos=9544724
    """
    m = re.match(r"^SNP-(\d+)-(\d+)(?:-\d+)?$", str(x))
    if not m:
        return pd.Series([np.nan, np.nan])

    return pd.Series([int(m.group(1)), int(m.group(2))])


def parse_chr(x):
    s = str(x).strip()
    s = s.replace("Chr", "").replace("chr", "")
    return int(s)


def load_importance(path, top_k):
    df = pd.read_csv(path, sep="\t")

    if "locus_id" not in df.columns:
        raise ValueError("Importance 文件中必须包含 locus_id 列")

    if "importance_mean" in df.columns:
        imp_col = "importance_mean"
    elif "importance" in df.columns:
        imp_col = "importance"
    else:
        raise ValueError("Importance 文件中必须包含 importance_mean 或 importance 列")

    df[["chr", "pos"]] = df["locus_id"].apply(parse_locus_id)
    df = df.dropna(subset=["chr", "pos"]).copy()

    df["chr"] = df["chr"].astype(int)
    df["pos"] = df["pos"].astype(int)

    df = df.sort_values(["chr", "pos"]).copy()

    if "rank" in df.columns:
        keep = df["rank"] <= top_k
    else:
        keep = df[imp_col].rank(ascending=False, method="first") <= top_k

    df["importance_value"] = df[imp_col]
    df["importance_for_smooth"] = df[imp_col].where(keep, 0)

    return df


def build_genome_layout(df):
    all_chrs = sorted(df["chr"].unique())

    chr_offsets = {}
    chr_centers = []

    offset = 0

    for chrom in all_chrs:
        sub = df[df["chr"] == chrom]

        min_pos = sub["pos"].min()
        max_pos = sub["pos"].max()
        length = max_pos - min_pos

        if length <= 0:
            length = max_pos if max_pos > 0 else 1

        chr_offsets[chrom] = offset
        chr_centers.append((chrom, offset + length / 2))

        offset += length

    return all_chrs, chr_offsets, chr_centers


def add_cum_pos(df, chr_offsets):
    df = df.copy()

    df["cum_pos"] = df.apply(
        lambda r: int(r["pos"]) + chr_offsets[int(r["chr"])],
        axis=1
    )

    return df


def load_ricenavi(path):
    df = pd.read_csv(path)

    required_cols = [
        "Chr",
        "Pos_7.0",
        "Method_Genotyping",
        "Alt_Allele_Func",
        "GeneName",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"RiceNavi 文件缺少必要列：{col}")

    sub = df[
        (df["Alt_Allele_Func"].astype(str).str.strip() == "increasing blast resistance") &
        (df["Method_Genotyping"].astype(str).str.strip().isin(["GATK3", "GATK4"]))
    ].copy()

    sub["chr"] = sub["Chr"].apply(parse_chr)
    sub["pos"] = sub["Pos_7.0"].astype(int)
    sub["GeneName"] = sub["GeneName"].astype(str).str.strip()

    sub = sub[
        sub["GeneName"].notna() &
        (sub["GeneName"] != "") &
        (sub["GeneName"] != "nan")
    ].copy()

    sub = sub.sort_values(["chr", "pos", "GeneName"])

    return sub


def match_ricenavi_to_importance(imp_df, rice_df, tolerance_bp):
    hits = []

    for _, r in rice_df.iterrows():
        chrom = int(r["chr"])
        rice_pos = int(r["pos"])

        cand = imp_df[
            (imp_df["chr"] == chrom) &
            (imp_df["pos"].between(rice_pos - tolerance_bp, rice_pos + tolerance_bp))
        ].copy()

        if cand.empty:
            continue

        cand["distance_to_ricenavi"] = (cand["pos"] - rice_pos).abs()

        cand = cand.sort_values(
            ["importance_value", "distance_to_ricenavi"],
            ascending=[False, True]
        )

        best = cand.iloc[0]

        hits.append({
            "locus_id": best["locus_id"],
            "chr": chrom,
            "pos": int(best["pos"]),
            "cum_pos": float(best["cum_pos"]),
            "importance_value": float(best["importance_value"]),
            "rice_pos": rice_pos,
            "GeneName": r["GeneName"],
            "Method_Genotyping": r["Method_Genotyping"],
            "Alt_Allele_Func": r["Alt_Allele_Func"],
            "distance_to_ricenavi": int(abs(int(best["pos"]) - rice_pos)),
        })

    if not hits:
        return pd.DataFrame()

    hit_df = pd.DataFrame(hits)

    merged = (
        hit_df
        .groupby(["locus_id", "chr", "pos", "cum_pos"], as_index=False)
        .agg({
            "importance_value": "max",
            "GeneName": lambda x: "/".join(sorted(set(map(str, x)))),
            "rice_pos": lambda x: ",".join(map(str, sorted(set(map(int, x))))),
            "distance_to_ricenavi": "min",
            "Method_Genotyping": lambda x: ",".join(sorted(set(map(str, x)))),
            "Alt_Allele_Func": "first",
        })
    )

    merged = merged.sort_values(["chr", "pos"]).copy()

    return merged


def plot_importance_by_chr(ax, imp_df, all_chrs):
    for i, chrom in enumerate(all_chrs):
        sub = imp_df[imp_df["chr"] == chrom].sort_values("cum_pos")
        color = CHR_COLORS[i % 2]

        ax.plot(
            sub["cum_pos"].values,
            sub["y_smooth"].values,
            color=color,
            linewidth=1.15,
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


def add_gene_labels(ax, hit_df, imp_df, show_marker=True, label_y_threshold=LABEL_Y_THRESHOLD):
    x_all = imp_df["cum_pos"].values
    y_all = imp_df["y_smooth"].values

    plot_rows = []
    marker_x = []
    marker_y = []

    for _, row in hit_df.iterrows():
        x = float(row["cum_pos"])
        y = float(np.interp(x, x_all, y_all))

        if y <= label_y_threshold:
            continue

        marker_x.append(x)
        marker_y.append(y)

        plot_rows.append({
            "x": x,
            "y": y,
            "label": row["GeneName"],
            "importance_value": row["importance_value"],
        })

    if len(plot_rows) == 0:
        return

    plot_rows = sorted(plot_rows, key=lambda z: z["x"])

    if show_marker:
        ax.scatter(
            marker_x,
            marker_y,
            s=32,
            color="#c23b4b",
            edgecolor="white",
            linewidth=1,
            zorder=3,
            clip_on=False
        )

    ymax = ax.get_ylim()[1]
    xmin, xmax = ax.get_xlim()
    xrange = xmax - xmin

    y_offsets = [
        0.035,
        0.060,
        0.085,
        0.110,
    ]

    x_offsets = [
        -0.014,
        0.014,
        -0.026,
        0.026,
    ]

    for i, row in enumerate(plot_rows):
        x = row["x"]
        y = row["y"]

        yoff = y_offsets[i % len(y_offsets)] * ymax
        xoff = x_offsets[i % len(x_offsets)] * xrange

        label_x = x + xoff
        label_y = y + yoff

        label_x = max(xmin + 0.01 * xrange, min(label_x, xmax - 0.01 * xrange))
        label_y = min(label_y, ymax * 0.96)

        arrowprops = None
        if show_marker:
            arrowprops = dict(
                arrowstyle="-",
                color="#666666",
                linewidth=0.6,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0"
            )

        ann = ax.annotate(
            row["label"],
            xy=(x, y),
            xytext=(label_x, label_y),
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontstyle="italic",
            color="#222222",
            arrowprops=arrowprops,
            zorder=6,
            clip_on=False
        )
        ann.set_zorder(6)
        if ann.arrow_patch is not None:
            ann.arrow_patch.set_zorder(6)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot genome-wide importance and label nearby RiceNavi blast QTNs."
    )
    parser.add_argument(
        "--importance",
        default=IMPORTANCE_FILE,
        help="Importance ranking TSV with locus_id and importance or importance_mean.",
    )
    parser.add_argument(
        "--ricenavi",
        default=RICENAVI_FILE,
        help="RiceNavi site table used to name nearby QTNs.",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        default=OUT_PREFIX,
        help="Prefix for the PDF, PNG, and matched-locus TSV.",
    )
    parser.add_argument(
        "--no-marker",
        action="store_true",
        help="Do not draw red QTN points or arrows; place gene names next to the loci.",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--tolerance-bp", type=int, default=TOLERANCE_BP)
    parser.add_argument("--smooth-sigma", type=float, default=SMOOTH_SIGMA)
    parser.add_argument("--label-y-threshold", type=float, default=LABEL_Y_THRESHOLD)
    parser.add_argument(
        "--ymax",
        type=float,
        default=YMAX,
        help="Upper limit of the importance axis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_prefix = args.output_prefix
    if args.no_marker and output_prefix == OUT_PREFIX:
        output_prefix = f"{OUT_PREFIX}.names_only"

    imp_df = load_importance(args.importance, args.top_k)

    all_chrs, chr_offsets, chr_centers = build_genome_layout(imp_df)

    imp_df = add_cum_pos(imp_df, chr_offsets)
    imp_df = imp_df.sort_values("cum_pos").copy()

    y = imp_df["importance_for_smooth"].values

    if args.smooth_sigma > 0:
        imp_df["y_smooth"] = gaussian_filter1d(y, sigma=args.smooth_sigma)
    else:
        imp_df["y_smooth"] = y.copy()

    rice_df = load_ricenavi(args.ricenavi)
    hit_df = match_ricenavi_to_importance(imp_df, rice_df, args.tolerance_bp)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    plot_importance_by_chr(ax, imp_df, all_chrs)

    ax.set_xlim(0, imp_df["cum_pos"].max())
    ax.set_ylim(0, args.ymax)

    yticks = np.arange(0, args.ymax + 1e-9, 0.05)
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        ["0" if abs(tick) < 1e-9 else f"{tick:.2f}".rstrip("0").rstrip(".") for tick in yticks]
    )

    ax.set_ylabel(
        "Importance\n(Gaussian smoothed)",
        fontsize=9.5,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Chromosome",
        fontsize=10.5,
        fontweight="bold"
    )

    ax.set_xticks([center for _, center in chr_centers])
    ax.set_xticklabels(
        [str(chrom) for chrom, _ in chr_centers],
        fontsize=9
    )

    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", direction="out", length=3, width=0.8)

    ax.grid(False)

    if not hit_df.empty:
        add_gene_labels(
            ax,
            hit_df,
            imp_df,
            show_marker=not args.no_marker,
            label_y_threshold=args.label_y_threshold,
        )

        # legend_handle = ax.scatter(
        #     [], [],
        #     s=32,
        #     color="#c23b4b",
        #     edgecolor="white",
        #     linewidth=0.7
        # )

        # ax.legend(
        #     [legend_handle],
        #     ["RiceNavi blast resistance QTGs"],
        #     loc="upper left",
        #     frameon=False,
        #     fontsize=8,
        #     handletextpad=0.5,
        #     borderpad=0.2
        # )
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.17,
        top=0.97
    )

    fig.savefig(f"{output_prefix}.pdf")
    fig.savefig(f"{output_prefix}.png")

    hit_df.to_csv(
        f"{output_prefix}.matched_ricenavi_loci.tsv",
        sep="\t",
        index=False
    )

    plt.close()

    print(f"Saved: {output_prefix}.pdf")
    print(f"Saved: {output_prefix}.png")
    print(f"Saved: {output_prefix}.matched_ricenavi_loci.tsv")
    print(f"Filtered RiceNavi loci: {len(rice_df)}")
    print(f"Matched importance loci: {len(hit_df)}")

    if not hit_df.empty:
        print()
        print(hit_df[[
            "locus_id",
            "GeneName",
            "chr",
            "pos",
            "rice_pos",
            "distance_to_ricenavi",
            "importance_value"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()