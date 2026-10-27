#!/usr/bin/env python3
"""
GWAS and production-model position-importance comparison plot (V3)

Three-panel layout:
- Row 1: GWAS Manhattan plot (-log10(p))
- Row 2: Raw top-K importance loci as scatter points
- Row 3: Gaussian-smoothed importance trend

Optional RiceNavi QTN labels: each QTN is assigned the highest-importance
SNP within ±tolerance as a proxy, and only proxies inside the IG top-K are drawn.
"""

import argparse
from pathlib import Path
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
        description="Compare one production-model IG ranking with a GWAS Manhattan plot"
    )

    parser.add_argument("--gwas", type=str, required=True, help="Path to GWAS result file (GEMMA .assoc.txt)")
    parser.add_argument("--importance", type=str, required=True, help="Path to importance ranking file (mean)")
    parser.add_argument("-o", "--output", type=str, default="gwas_ig_comparison_v3.png",
                        help="Output image path (.png or .pdf, suffix determines format)")
    parser.add_argument(
        "--also-png",
        action="store_true",
        help="Also save a PNG copy alongside the primary output",
    )
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
    parser.add_argument(
        "--qtn-annot",
        type=str,
        default=None,
        help="Optional RiceNavi QTN Excel file. QTN labels are drawn only when this is set.",
    )
    parser.add_argument(
        "--annot-tolerance-kb",
        type=float,
        default=100,
        help=(
            "Half-window around each QTN in kb (default: 100, i.e. ±100 kb). "
            "Used only with --qtn-annot."
        ),
    )
    parser.add_argument(
        "--annot-threshold",
        type=float,
        default=0.1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--highlight",
        action="append",
        default=[],
        metavar="CHR:POS",
        help="Mark this position on every panel. Repeat for more than one locus.",
    )

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


def _nonempty_name(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "na"} else text


def _parse_qtn_position(value):
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "none", "<mnp>"}:
        return None
    try:
        if "-" in text:
            left, right = text.split("-", 1)
            return (float(left) + float(right)) / 2.0
        return float(text)
    except (TypeError, ValueError):
        return None


def load_qtn_table(qtn_file):
    path = Path(qtn_file)
    print(f"Loading QTN annotations from {path}...")
    workbook = pd.ExcelFile(path)
    sheet = (
        "all_347_genotyped"
        if "all_347_genotyped" in workbook.sheet_names
        else workbook.sheet_names[0]
    )
    frame = pd.read_excel(
        workbook,
        sheet_name=sheet,
        usecols=lambda column: column
        in {"MSU_ID", "Gene", "CommonlyUsedName", "Chr", "Pos_7.0"},
    )
    required = {"Chr", "Pos_7.0"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"QTN workbook is missing columns: {sorted(missing)}")
    for column in ("MSU_ID", "Gene", "CommonlyUsedName"):
        if column not in frame.columns:
            frame[column] = ""

    chr_text = (
        frame["Chr"].astype(str).str.replace("Chr", "", regex=False).str.strip()
    )
    frame["chromosome"] = pd.to_numeric(chr_text, errors="coerce")
    frame["position"] = frame["Pos_7.0"].map(_parse_qtn_position)
    frame = frame.dropna(subset=["chromosome", "position"]).copy()
    frame["chromosome"] = frame["chromosome"].astype(int)
    frame["position"] = frame["position"].round().astype(int)
    frame["qtn_name"] = [
        _nonempty_name(common) or _nonempty_name(gene) or _nonempty_name(msu)
        or f"{int(chrom)}:{int(pos)}"
        for common, gene, msu, chrom, pos in zip(
            frame["CommonlyUsedName"],
            frame["Gene"],
            frame["MSU_ID"],
            frame["chromosome"],
            frame["position"],
        )
    ]
    print(
        f"  Loaded {len(frame)} QTNs from sheet {sheet!r} "
        f"across {frame['chromosome'].nunique()} chromosomes"
    )
    return frame.reset_index(drop=True)


def assign_qtn_proxy_snps(
    qtns,
    importance_df,
    *,
    tolerance_bp,
    top_k,
    importance_col,
):
    """Assign each QTN the highest-importance SNP within ±tolerance_bp."""
    if tolerance_bp < 0:
        raise ValueError("annot-tolerance-kb must be nonnegative")
    markers = importance_df.copy()
    if "rank" not in markers.columns:
        markers["rank"] = (
            markers[importance_col]
            .rank(ascending=False, method="first")
            .astype(int)
        )
    by_chromosome = {
        int(chrom): group
        for chrom, group in markers.groupby(markers["chr"].astype(int), sort=False)
    }

    rows = []
    for qtn in qtns.itertuples(index=False):
        chrom = int(qtn.chromosome)
        position = int(qtn.position)
        group = by_chromosome.get(chrom)
        record = {
            "qtn_name": str(qtn.qtn_name),
            "qtn_chromosome": chrom,
            "qtn_position": position,
            "proxy_locus_id": "",
            "proxy_chr": np.nan,
            "proxy_pos": np.nan,
            "proxy_importance": np.nan,
            "proxy_rank": np.nan,
            "distance_bp": np.nan,
            "in_top_k": False,
        }
        if group is None or group.empty:
            rows.append(record)
            continue
        window = group[
            (group["pos"] >= position - tolerance_bp)
            & (group["pos"] <= position + tolerance_bp)
        ].copy()
        if window.empty:
            rows.append(record)
            continue
        window["_distance_bp"] = (window["pos"] - position).abs()
        window = window.sort_values(
            [importance_col, "_distance_bp", "pos"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        best = window.iloc[0]
        rank = int(best["rank"])
        record.update(
            {
                "proxy_locus_id": str(best["locus_id"]),
                "proxy_chr": int(best["chr"]),
                "proxy_pos": int(best["pos"]),
                "proxy_importance": float(best[importance_col]),
                "proxy_rank": rank,
                "distance_bp": int(best["_distance_bp"]),
                "in_top_k": rank <= int(top_k),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _annotate_qtn_proxies(ax, importance_df, proxies, importance_col):
    """Label QTN names at their top-K proxy SNPs."""
    if proxies.empty:
        print("  0 QTN proxies annotated")
        return

    lookup = importance_df.set_index("locus_id")
    annotated = []
    seen = set()
    for _, proxy in proxies.iterrows():
        locus_id = str(proxy["proxy_locus_id"])
        gene = str(proxy["qtn_name"])
        key = (gene, locus_id)
        if not locus_id or key in seen or locus_id not in lookup.index:
            continue
        seen.add(key)
        hit = lookup.loc[locus_id]
        if isinstance(hit, pd.DataFrame):
            hit = hit.iloc[0]
        cum_x = float(hit["cum_pos"])
        y_val = float(hit[importance_col])
        dup = sum(1 for (_, _, cx) in annotated if abs(cx - cum_x) < 150)
        row_idx = dup % 4
        ax.annotate(
            gene,
            xy=(cum_x, y_val),
            xytext=(8 + row_idx * 10, 8 + row_idx * 5),
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
        annotated.append((gene, float(proxy["distance_bp"]), cum_x))

    print(f"  {len(annotated)} QTN labels drawn on the importance scatter")


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


def _parse_highlight(spec):
    text = str(spec).strip().lower().replace("chr", "")
    chrom, pos = text.split(":")
    return int(chrom), int(pos)


def _mark_highlights(axes, gwas_df, importance_df, highlights):
    if not highlights:
        return

    ax1, ax2, ax3 = axes
    color = "#c0392b"
    for spec in highlights:
        chrom, pos = _parse_highlight(spec)
        label = f"chr{chrom}:{pos:,}"
        gwas_hit = gwas_df[(gwas_df["chr"] == chrom) & (gwas_df["pos"] == pos)]
        imp_hit = importance_df[(importance_df["chr"] == chrom) & (importance_df["pos"] == pos)]
        if gwas_hit.empty and imp_hit.empty:
            raise ValueError(f"Highlight {label} is not in the GWAS file or the importance ranking")

        if not imp_hit.empty:
            cum_x = float(imp_hit["cum_pos"].iloc[0])
        else:
            cum_x = float(gwas_hit["cum_pos"].iloc[0])

        for ax in axes:
            ax.axvline(cum_x, color=color, linestyle="--", linewidth=0.8, alpha=0.95, zorder=6)

        if not gwas_hit.empty:
            gwas_y = float(gwas_hit["neg_log_p"].iloc[0])
            ax1.scatter([cum_x], [gwas_y], s=28, c=color, marker="o", zorder=7, linewidths=0)
            span = ax1.get_xlim()[1] - ax1.get_xlim()[0]
            ax1.text(
                cum_x + span * 0.004,
                ax1.get_ylim()[1] * 0.58,
                label,
                color=color,
                fontsize=7,
                ha="left",
                va="bottom",
                rotation=90,
                zorder=8,
            )
            print(f"  Highlight {label}: GWAS -log10(p)={gwas_y:.3f} rs={gwas_hit['rs'].iloc[0]}")

        if not imp_hit.empty:
            imp_y = float(imp_hit["importance_raw_for_scatter"].iloc[0])
            lower, upper = ax2.get_ylim()
            if imp_y < lower:
                ax2.set_ylim(0, upper)
            elif imp_y > upper:
                ax2.set_ylim(lower, imp_y * 1.08)
            ax2.scatter([cum_x], [imp_y], s=28, c=color, marker="o", zorder=7, linewidths=0)
            rank = int(imp_hit["rank"].iloc[0]) if "rank" in imp_hit.columns else None
            rank_text = f", rank {rank}" if rank is not None else ""
            print(
                f"  Highlight {label}: importance={imp_y:.4f}{rank_text} "
                f"locus={imp_hit['locus_id'].iloc[0]}"
            )


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
    also_png=False,
    sig_threshold=4.259270e-06,
    show_std=False,
    smooth_sigma=0,
    gwas_smooth_sigma=0,
    max_std=None,
    ig_top_k=500,
    scatter_size=8,
    scatter_alpha_min=0.18,
    scatter_alpha_max=0.75,
    qtn_table=None,
    annot_tolerance_bp=100_000,
    highlights=None,
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
    if qtn_table is not None:
        print(
            f"\nAssigning each RiceNavi QTN the highest-importance SNP "
            f"within ±{annot_tolerance_bp / 1000:.0f} kb; "
            f"label proxies in IG top {ig_top_k}..."
        )
        proxies = assign_qtn_proxy_snps(
            qtn_table,
            importance_df,
            tolerance_bp=annot_tolerance_bp,
            top_k=ig_top_k,
            importance_col="importance_raw_for_scatter",
        )
        n_assigned = int(proxies["proxy_locus_id"].ne("").sum())
        n_labeled = int(proxies["in_top_k"].sum())
        print(
            f"  Proxies assigned for {n_assigned}/{len(proxies)} QTNs; "
            f"{n_labeled} have a top-{ig_top_k} proxy"
        )
        proxy_path = Path(output_path).with_suffix(".qtn_proxies.tsv")
        proxies.to_csv(proxy_path, sep="\t", index=False)
        print(f"  Wrote QTN proxy table: {proxy_path}")
        _annotate_qtn_proxies(
            ax2,
            importance_df,
            proxies[proxies["in_top_k"]].copy(),
            importance_col="importance_raw_for_scatter",
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

    _mark_highlights(axes, gwas_df, importance_df, highlights or [])

    plt.subplots_adjust(left=0.09, right=0.995, top=0.985, bottom=0.08, hspace=0.12)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if also_png:
        png_path = str(Path(output_path).with_suffix(".png"))
        if Path(png_path) != Path(output_path):
            plt.savefig(png_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {png_path}")

    plt.close()


def main():
    args = parse_args()
    set_publication_style()

    print("=" * 80)
    print("GWAS and production-model IG comparison")
    print("=" * 80)

    gwas_df = load_gwas_data(args.gwas)
    importance_df, has_std = load_importance_data(args.importance)

    print(f"\nGWAS: {len(gwas_df)} SNPs across {gwas_df['chr'].nunique()} chromosomes")
    print(f"Importance: {len(importance_df)} loci across {importance_df['chr'].nunique()} chromosomes")
    print(f"Has std: {has_std}")
    print(f"Significance threshold: {args.sig_threshold} "
          f"(-log10(p) = {-np.log10(args.sig_threshold):.2f})")

    qtn_table = None
    annot_tolerance_bp = None
    if args.qtn_annot:
        if args.annot_tolerance_kb < 0:
            raise ValueError("annot-tolerance-kb must be nonnegative")
        annot_tolerance_bp = int(round(float(args.annot_tolerance_kb) * 1000))
        qtn_table = load_qtn_table(args.qtn_annot)

    create_comparison_plot(
        gwas_df=gwas_df,
        importance_df=importance_df,
        output_path=args.output,
        also_png=args.also_png,
        sig_threshold=args.sig_threshold,
        show_std=args.show_std,
        smooth_sigma=args.smooth,
        gwas_smooth_sigma=args.gwas_smooth,
        max_std=args.max_std,
        ig_top_k=args.ig_top_k,
        scatter_size=args.scatter_size,
        scatter_alpha_min=args.scatter_alpha_min,
        scatter_alpha_max=args.scatter_alpha_max,
        qtn_table=qtn_table,
        annot_tolerance_bp=annot_tolerance_bp if annot_tolerance_bp is not None else 100_000,
        highlights=args.highlight,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()