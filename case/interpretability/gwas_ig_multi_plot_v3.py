#!/usr/bin/env python3
"""
GWAS and Position Importance Multi-trial Comparison Plot (V3)

Three-panel layout:
- Row 1: GWAS Manhattan plot (-log10(p))
- Row 2: Raw top-K importance loci as scatter points
- Row 3: Gaussian-smoothed importance trend

Supports QTN gene annotation on the smoothed importance panel.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# Publication-style palette (violin-like palette)
# =============================================================================
_THEME = {
    "bg": "#ffffff",
    "panel_bg": "#ffffff",
    "text": "#222222",
    "muted_text": "#5f6b6d",
    "spine": "#000000",
    "boundary": "#d7dcdd",

    # GWAS alternating chromosome colors
    "gwas_odd": "#455781",
    "gwas_even": "#0B3564",

    # GWAS smoothed mode
    "gwas_smooth": "#2f5f5d",
    "gwas_fill": "#dbe7e5",

    # Raw importance scatter
    "ig_scatter": "#0B3564",
    "ig_scatter_edge": "#7a3b2e",

    # Smoothed importance panel
    "ig_line": "#0B3564",
    "ig_fill": "#FFFFFF",
    "ig_std": "#FFFFFF",

    # Threshold / annotations
    "threshold": "#7f8c8d",
    "annot_text": "#000000",
    "annot_arrow": "#000000",
}


def set_publication_style():
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": _THEME["spine"],
        "xtick.color": _THEME["text"],
        "ytick.color": _THEME["text"],
        "text.color": _THEME["text"],
        "axes.labelcolor": _THEME["text"],
        "axes.titlecolor": _THEME["text"],
        "savefig.facecolor": _THEME["bg"],
        "figure.facecolor": _THEME["bg"],
        "axes.facecolor": _THEME["panel_bg"],
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description="GWAS and Position Importance Multi-trial Comparison Plot (V3)"
    )

    parser.add_argument("--gwas", type=str, required=True, help="Path to GWAS result file (GEMMA .assoc.txt)")
    parser.add_argument("--importance", type=str, required=True, help="Path to importance ranking file (mean)")
    parser.add_argument("-o", "--output", type=str, default="gwas_ig_comparison_v3.png",
                        help="Output image path (.png or .pdf, suffix determines format)")
    parser.add_argument("--sig-threshold", type=float, default=4.259270e-06,
                        help="GWAS significance threshold (default: 4.259270e-06)")
    parser.add_argument("--show-std", action="store_true", help="Show standard deviation error band")
    parser.add_argument("--smooth", type=float, default=0, help="Gaussian smoothing sigma for importance")
    parser.add_argument("--gwas-smooth", type=float, default=0, help="Gaussian smoothing sigma for GWAS")
    parser.add_argument("--max-std", type=float, default=None, help="Mask loci with std > this value")
    parser.add_argument("--ig-top-k", type=int, default=500, help="Only plot top K ranked loci for IG")
    parser.add_argument("--scatter-size", type=float, default=8, help="Base scatter size for top-K raw importance points")
    parser.add_argument("--scatter-alpha-min", type=float, default=0.18, help="Minimum alpha for top-K scatter")
    parser.add_argument("--scatter-alpha-max", type=float, default=0.75, help="Maximum alpha for top-K scatter")
    parser.add_argument("--qtn-annot", type=str, default=None,
                        help="Path to QTN annotation Excel file")
    parser.add_argument("--annot-threshold", type=float, default=0.1,
                        help="Minimum normalized importance to consider for QTN annotation")
    parser.add_argument("--annot-tolerance", type=float, default=1e-5,
                        help="Maximum distance (bp) between a locus and QTN to trigger annotation")

    return parser.parse_args()


def load_gwas_data(gwas_file):
    print(f"Loading GWAS data from {gwas_file}...")
    chunks = []
    for chunk in pd.read_csv(gwas_file, sep="\t", chunksize=100000):
        snp_chunk = chunk[chunk["rs"].str.startswith("SNP-", na=False)]
        chunks.append(snp_chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df)} SNP rows")

    df["chr"] = df["chr"].astype(int)
    df["pos"] = df["ps"]
    df["neg_log_p"] = -np.log10(df["p_wald"])
    df = df.sort_values(["chr", "pos"])
    return df


def load_importance_data(importance_file):
    print(f"Loading importance data from {importance_file}...")
    df = pd.read_csv(importance_file, sep="\t")
    print(f"  Loaded {len(df)} loci")

    has_std = "importance_std" in df.columns
    df["snp_parts"] = df["locus_id"].str.split("-")
    df["chr"] = df["snp_parts"].str[1].astype(int)
    df["pos"] = df["snp_parts"].str[2].astype(int)
    df = df.sort_values(["chr", "pos"])
    return df, has_std


def load_qtn_annot(qtn_file):
    print(f"Loading QTN annotations from {qtn_file}...")
    df = pd.read_excel(qtn_file)

    df["chr_str"] = df["Chr"].astype(str).str.replace("Chr", "", regex=False).str.strip()
    df["chr_int"] = pd.to_numeric(df["chr_str"], errors="coerce")
    df = df.dropna(subset=["chr_int"])
    df["chr_int"] = df["chr_int"].astype(int)

    def parse_pos(val):
        s = str(val).strip()
        if not s or s.lower() in ("nan", "na", "none", "<mnp>"):
            return None
        try:
            if "-" in s:
                parts = s.split("-")
                return (float(parts[0]) + float(parts[1])) / 2
            return float(s)
        except (ValueError, IndexError):
            return None

    qtn_by_chr = {}
    skipped = 0
    for _, row in df.iterrows():
        gene = str(row["CommonlyUsedName"]).strip()
        pos = parse_pos(row["Pos_7.0"])
        if pos is None:
            skipped += 1
            continue
        c = int(row["chr_int"])
        qtn_by_chr.setdefault(c, []).append((pos, gene))

    total = sum(len(v) for v in qtn_by_chr.values())
    print(f"  Loaded {total} QTNs across {len(qtn_by_chr)} chromosomes"
          f"{f' (skipped {skipped} unparseable rows)' if skipped else ''}")
    return qtn_by_chr


def find_nearest_qtn(chr_int, pos, qtn_by_chr, tolerance):
    if chr_int not in qtn_by_chr:
        return None, None

    best_gene, best_dist = None, None
    for qtn_pos, gene in qtn_by_chr[chr_int]:
        dist = abs(pos - qtn_pos)
        if dist <= tolerance and (best_dist is None or dist < best_dist):
            best_dist = dist
            best_gene = gene
    return best_gene, best_dist


def _annotate_genes(
    ax,
    df,
    imp_col,
    qtn_by_chr,
    annot_threshold,
    annot_tolerance,
    smooth_sigma,
    y_ref=None,
):
    """Add gene labels (with arrows) for loci near known QTNs."""
    annotated = []
    for _, row in df.iterrows():
        imp_val = row.get(imp_col)
        if imp_val is None or imp_val < annot_threshold:
            continue

        gene, dist = find_nearest_qtn(
            int(row["chr"]),
            int(row["pos"]),
            qtn_by_chr,
            annot_tolerance,
        )
        if not gene or dist is None:
            continue

        cum_x = row["cum_pos"]
        y_val = float(imp_val)

        # vertical offset to sit just above the scatter point
        y_base = y_val
        y_top = y_val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04

        dup = sum(1 for (_, _, cx) in annotated if abs(cx - cum_x) < 150)
        row_idx = dup % 4
        dx_pts = 8 + row_idx * 10
        dy_pts = 8 + row_idx * 5

        ax.annotate(
            gene,
            xy=(cum_x, y_base),
            xytext=(dx_pts, dy_pts),
            textcoords="offset points",
            fontsize=4,
            color=_THEME["annot_text"],
            fontstyle="italic",
            fontweight="normal",
            ha="left",
            va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                color=_THEME["annot_arrow"],
                lw=0.6,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=10,
        )
        annotated.append((gene, dist, cum_x))

    print(f"  {len(annotated)} loci annotated on {ax.get_label()}")


def build_genome_layout(gwas_df, importance_df):
    all_chrs = sorted(set(gwas_df["chr"].unique()) | set(importance_df["chr"].unique()))

    chr_sizes = {}
    cumulative_offset = 0
    chr_boundaries = [0]
    chr_centers = []

    for chrom in all_chrs:
        chr_df_g = gwas_df[gwas_df["chr"] == chrom]
        chr_df_i = importance_df[importance_df["chr"] == chrom]

        max_pos = 0
        min_pos = 0

        if len(chr_df_g) > 0:
            max_pos = max(max_pos, chr_df_g["pos"].max())
            min_pos = min(min_pos, chr_df_g["pos"].min())
        if len(chr_df_i) > 0:
            max_pos = max(max_pos, chr_df_i["pos"].max())
            min_pos = min(min_pos, chr_df_i["pos"].min())

        chr_length = max_pos - min_pos
        if chr_length <= 0:
            chr_length = max_pos if max_pos > 0 else 1

        chr_sizes[chrom] = chr_length
        chr_centers.append((chrom, cumulative_offset + chr_length / 2))
        cumulative_offset += chr_length
        chr_boundaries.append(cumulative_offset)

    return all_chrs, chr_sizes, chr_boundaries, chr_centers


def get_cumulative_pos(df, chr_sizes, all_chrs):
    result = []
    offset = 0
    for chrom in all_chrs:
        chr_df = df[df["chr"] == chrom].copy()
        if len(chr_df) > 0:
            chr_df["cum_pos"] = chr_df["pos"] + offset
            result.append(chr_df)
        offset += chr_sizes.get(chrom, 0)

    if result:
        return pd.concat(result, ignore_index=True)
    return pd.DataFrame()


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_THEME["spine"])
    ax.spines["bottom"].set_color(_THEME["spine"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=_THEME["text"], length=3.2, width=0.8)
    ax.grid(False)
    ax.set_axisbelow(True)


def alpha_map(values, alpha_min=0.18, alpha_max=0.95):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.full_like(values, fill_value=alpha_max, dtype=float)
    scaled = (values - vmin) / (vmax - vmin)
    return alpha_min + scaled * (alpha_max - alpha_min)


def create_comparison_plot(
    gwas_df,
    importance_df,
    output_path,
    sig_threshold=4.259270e-06,
    show_std=False,
    smooth_sigma=0,
    gwas_smooth_sigma=0,
    max_std=None,
    ig_top_k=500,
    scatter_size=8,
    scatter_alpha_min=0.18,
    scatter_alpha_max=0.75,
    qtn_by_chr=None,
    annot_threshold=0.1,
    annot_tolerance=1e6
):
    print("Creating comparison plot...")

    ig_top_k = int(ig_top_k)
    importance_df = importance_df.copy()
    imp_col = "importance_mean" if "importance_mean" in importance_df.columns else "importance"

    importance_df["importance_raw_for_scatter"] = importance_df[imp_col].copy()

    if "rank" in importance_df.columns:
        topk_mask = importance_df["rank"] <= ig_top_k
        n_zeroed = (~topk_mask).sum()
        importance_df.loc[~topk_mask, imp_col] = 0
    else:
        ranked = importance_df[imp_col].rank(ascending=False, method="first")
        topk_mask = ranked <= ig_top_k
        n_zeroed = (~topk_mask).sum()
        importance_df.loc[~topk_mask, imp_col] = 0

    importance_df["is_topk"] = topk_mask.values
    print(f"  IG: {n_zeroed}/{len(importance_df)} loci zeroed (top_k={ig_top_k})")

    all_chrs, chr_sizes, chr_boundaries, chr_centers = build_genome_layout(gwas_df, importance_df)
    gwas_df = get_cumulative_pos(gwas_df, chr_sizes, all_chrs)
    importance_df = get_cumulative_pos(importance_df, chr_sizes, all_chrs)

    genome_span = float(max(
        gwas_df["cum_pos"].max() if len(gwas_df) else 0,
        importance_df["cum_pos"].max() if len(importance_df) else 0,
        1
    ))
    sig_line = -np.log10(sig_threshold)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(8, 4.2),
        facecolor=_THEME["bg"],
        gridspec_kw={
            "hspace": 0.10,
            "height_ratios": [0.6, 0.6, 0.8]
        }
    )

    # =========================================================================
    # Row 1: GWAS
    # =========================================================================
    ax1 = axes[0]
    ax1.set_facecolor(_THEME["panel_bg"])

    gwas_sorted = gwas_df.sort_values("cum_pos")
    gwas_x = gwas_sorted["cum_pos"].values
    gwas_y = gwas_sorted["neg_log_p"].values
    gwas_chr = gwas_sorted["chr"].astype(int).values
    gwas_colors = np.array([
        _THEME["gwas_odd"] if (int(c) % 2 == 1) else _THEME["gwas_even"]
        for c in gwas_chr
    ])

    if gwas_smooth_sigma > 0:
        gwas_y_smooth = gaussian_filter1d(gwas_y, sigma=gwas_smooth_sigma)
        print(f"  Applied GWAS Gaussian smoothing (sigma={gwas_smooth_sigma})")
        ax1.fill_between(gwas_x, gwas_y_smooth, 0, color=_THEME["gwas_fill"], alpha=0.40, linewidth=0, zorder=1)
        ax1.plot(gwas_x, gwas_y_smooth, color=_THEME["gwas_smooth"], linewidth=1.0, alpha=0.96, zorder=2)
        gwas_plot_max = float(np.nanmax(gwas_y_smooth)) if len(gwas_y_smooth) else 1.0
    else:
        ax1.scatter(gwas_x, gwas_y, c=gwas_colors, s=4, alpha=0.85, linewidths=0, rasterized=True, zorder=2)
        gwas_plot_max = float(np.nanmax(gwas_y)) if len(gwas_y) else 1.0

    ax1.axhline(y=sig_line, color=_THEME["threshold"], linestyle="--", linewidth=0.95, dashes=(4, 3), alpha=0.95, zorder=3)

    for boundary in chr_boundaries[1:-1]:
        ax1.axvline(x=boundary, color=_THEME["boundary"], linestyle="-", linewidth=0.8, alpha=1.0, zorder=0)

    y_max = max(gwas_plot_max * 1.08, sig_line * 1.16, 1.0)
    ax1.set_ylim(0, y_max)
    ax1.set_xlim(0, genome_span)

    chrom_label_y = y_max * 0.93
    for chrom, center in chr_centers:
        ax1.text(center, chrom_label_y, str(int(chrom)), ha="center", va="bottom", fontsize=10, color=_THEME["muted_text"])

    ax1.set_xticks([])
    ax1.tick_params(axis="x", which="both", length=0)
    ax1.set_ylabel(r"$-\log_{10}[P]$", fontsize=10)
    style_axis(ax1)

    # =========================================================================
    # Row 2: Raw top-K importance scatter
    # =========================================================================
    ax2 = axes[1]
    ax2.set_facecolor(_THEME["panel_bg"])

    importance_sorted = importance_df.sort_values("cum_pos")
    scatter_df = importance_sorted[importance_sorted["is_topk"]].copy()
    scatter_df = scatter_df[scatter_df["importance_raw_for_scatter"] > 0].copy()

    if not scatter_df.empty:
        scatter_x = scatter_df["cum_pos"].values
        scatter_y = scatter_df["importance_raw_for_scatter"].values
        scatter_alpha = alpha_map(scatter_y, alpha_min=scatter_alpha_min, alpha_max=scatter_alpha_max)

        base_rgba = np.array(plt.matplotlib.colors.to_rgba(_THEME["ig_scatter"]))
        colors = np.tile(base_rgba, (len(scatter_y), 1))
        colors[:, 3] = scatter_alpha

        ax2.scatter(
            scatter_x,
            scatter_y,
            s=scatter_size,
            c=colors,
            edgecolors="none",
            zorder=3,
            rasterized=True
        )

    for boundary in chr_boundaries[1:-1]:
        ax2.axvline(x=boundary, color=_THEME["boundary"], linestyle="-", linewidth=0.8, alpha=1.0, zorder=0)

    if not scatter_df.empty:
        y_scatter = scatter_df["importance_raw_for_scatter"].values.astype(float)
        y_min = np.nanmin(y_scatter)
        y_max = np.nanmax(y_scatter)

        if not np.isfinite(y_min) or not np.isfinite(y_max):
            y_min, y_max = 0.0, 1.0

        if y_max == y_min:
            pad = max(abs(y_max) * 0.05, 1e-6)
        else:
            pad = (y_max - y_min) * 0.06   # 6%留白，可自行调小调大

        lower = y_min - pad
        upper = y_max + pad

        # 如果你不希望出现负值坐标，可以加这一句
        lower = max(0, lower)

        ax2.set_ylim(lower, upper)
    else:
        ax2.set_ylim(0, 1)
    ax2.set_xlim(0, genome_span)
    ax2.set_xticks([])
    ax2.tick_params(axis="x", which="both", length=0)
    ax2.set_ylabel("Importance")
    style_axis(ax2)

    # =========================================================================
    # QTN annotation (on Row 2 scatter, using raw importance y-values)
    # =========================================================================
    if qtn_by_chr is not None:
        print(
            f"\nAnnotating scatter (Row 2) with importance > {annot_threshold} "
            f"within {annot_tolerance/1e6:.3f} Mb of a known QTN..."
        )
        _annotate_genes(
            ax2,
            scatter_df,
            imp_col="importance_raw_for_scatter",
            qtn_by_chr=qtn_by_chr,
            annot_threshold=annot_threshold,
            annot_tolerance=annot_tolerance,
            smooth_sigma=0,
            y_ref=scatter_df["importance_raw_for_scatter"].values if len(scatter_df) else None,
        )

    # =========================================================================
    # Row 3: Gaussian-smoothed importance trend
    # =========================================================================
    ax3 = axes[2]
    ax3.set_facecolor(_THEME["panel_bg"])

    x = importance_sorted["cum_pos"].values
    y_mean = importance_sorted[imp_col].values.copy()

    if max_std is not None and "importance_std" in importance_sorted.columns:
        y_std_raw = importance_sorted["importance_std"].values
        mask = y_std_raw > max_std
        n_masked = mask.sum()
        if n_masked > 0:
            y_mean[mask] = 0
            print(f"  Masked {n_masked} loci with std > {max_std}")

    if smooth_sigma > 0:
        y_mean_smooth = gaussian_filter1d(y_mean, sigma=smooth_sigma)
        print(f"  Applied IG Gaussian smoothing (sigma={smooth_sigma})")
    else:
        y_mean_smooth = y_mean.copy()

    ax3.fill_between(x, y_mean_smooth, 0, color=_THEME["ig_fill"], alpha=0.55, linewidth=0, zorder=1)
    ax3.plot(x, y_mean_smooth, color=_THEME["ig_line"], linewidth=1.2, alpha=0.98, zorder=2)

    upper = None
    if show_std and "importance_std" in importance_sorted.columns:
        std_for_band = importance_sorted["importance_std"].values.copy()
        if max_std is not None:
            std_for_band[importance_sorted["importance_std"].values > max_std] = 0

        if smooth_sigma > 0:
            std_smooth = gaussian_filter1d(std_for_band, sigma=smooth_sigma)
        else:
            std_smooth = std_for_band

        lower = np.clip(y_mean_smooth - std_smooth, 0, None)
        upper = y_mean_smooth + std_smooth

        ax3.fill_between(
            x, lower, upper,
            color=_THEME["ig_std"],
            alpha=0.7,
            linewidth=0,
            zorder=0.9,
            label="±1 s.d."
        )

    for boundary in chr_boundaries[1:-1]:
        ax3.axvline(x=boundary, color=_THEME["boundary"], linestyle="-", linewidth=0.8, alpha=1.0, zorder=0)

    ymax_curve = float(np.nanmax(y_mean_smooth)) if len(y_mean_smooth) else 0.0
    ymax_band = float(np.nanmax(upper)) if upper is not None and len(upper) else 0.0
    ymax3 = max(ymax_curve, ymax_band)

    if ymax3 <= 0:
        ymax3 = 1.0

    ax3.set_ylim(-ymax3 * 0.03, ymax3 * 1.03)
    ax3.set_xlim(0, genome_span)
    ax3.set_xticks([])
    ax3.tick_params(axis="x", which="both", length=0)
    ax3.set_xlabel("Chromosome", fontsize = 10, fontweight="bold")
    ax3.set_ylabel("Importance\n(Gaussian smoothed)")

    if show_std and "importance_std" in importance_sorted.columns:
        leg = ax3.legend(
            loc="upper right",
            frameon=True,
            framealpha=1,
            edgecolor="#d8d8d8",
            facecolor="#ffffff"
        )
        leg.get_frame().set_linewidth(0.8)

    style_axis(ax3)

    plt.subplots_adjust(left=0.09, right=0.995, top=0.985, bottom=0.08, hspace=0.12)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.close()


def main():
    args = parse_args()
    set_publication_style()

    print("=" * 80)
    print("GWAS and Position Importance Multi-trial Comparison Plot (V3)")
    print("=" * 80)

    gwas_df = load_gwas_data(args.gwas)
    importance_df, has_std = load_importance_data(args.importance)

    print(f"\nGWAS: {len(gwas_df)} SNPs across {gwas_df['chr'].nunique()} chromosomes")
    print(f"Importance: {len(importance_df)} loci across {importance_df['chr'].nunique()} chromosomes")
    print(f"Has std: {has_std}")
    print(f"Significance threshold: {args.sig_threshold} "
          f"(-log10(p) = {-np.log10(args.sig_threshold):.2f})")

    qtn_by_chr = None
    if args.qtn_annot:
        qtn_by_chr = load_qtn_annot(args.qtn_annot)

    create_comparison_plot(
        gwas_df=gwas_df,
        importance_df=importance_df,
        output_path=args.output,
        sig_threshold=args.sig_threshold,
        show_std=args.show_std,
        smooth_sigma=args.smooth,
        gwas_smooth_sigma=args.gwas_smooth,
        max_std=args.max_std,
        ig_top_k=args.ig_top_k,
        scatter_size=args.scatter_size,
        scatter_alpha_min=args.scatter_alpha_min,
        scatter_alpha_max=args.scatter_alpha_max,
        qtn_by_chr=qtn_by_chr,
        annot_threshold=args.annot_threshold,
        annot_tolerance=args.annot_tolerance,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()