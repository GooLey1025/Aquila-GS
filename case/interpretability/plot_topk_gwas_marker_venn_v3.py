#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Non-redundant UpSetPlot for Top-K SNP loci and source categories.

Outputs:
1. UpSet figure
2. Source percentage TSV:
   Trait, Category, TopK_count, Total_loci, Percentage_of_total_loci
"""

import argparse
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

try:
    from upsetplot import UpSet, from_contents
except ImportError:
    raise SystemExit(
        "Error: upsetplot is not installed.\n"
        "Please install it with:\n"
        "    pip install upsetplot"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot non-redundant UpSetPlot for Top-K SNP loci and source categories."
    )

    parser.add_argument("--importance", required=True)
    parser.add_argument("--gwas", required=True, nargs="+")
    parser.add_argument("--markers", required=True)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--p-threshold", type=float, default=4.259270e-06)
    parser.add_argument("--max-intersections", type=int, default=6)
    parser.add_argument("-o", "--output", default="topk_multisource_upset.png")

    parser.add_argument(
        "--summary-tsv",
        default=None,
        help="Output TSV recording source-wise percentage of total loci."
    )

    return parser.parse_args()


def make_chr_pos_key(chr_val, pos_val):
    return f"{int(chr_val)}:{int(pos_val)}"


def parse_locus_id_to_chr_pos(locus_id):
    parts = str(locus_id).split("-")

    if len(parts) < 4:
        raise ValueError(f"Unexpected locus_id format: {locus_id}")

    return int(parts[1]), int(parts[2])


def load_importance_topk(path, top_k):
    df = pd.read_csv(path, sep="\t")

    required = {"rank", "locus_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Importance file missing columns: {missing}")

    df_top = df[df["rank"] <= top_k].copy()

    chr_list, pos_list, key_list = [], [], []

    for locus in df_top["locus_id"]:
        c, p = parse_locus_id_to_chr_pos(locus)
        chr_list.append(c)
        pos_list.append(p)
        key_list.append(make_chr_pos_key(c, p))

    df_top["chr"] = chr_list
    df_top["pos"] = pos_list
    df_top["chr_pos"] = key_list

    return df, df_top


def load_gwas_significant(path, p_threshold):
    df = pd.read_csv(path, sep="\t")

    required = {"chr", "rs", "ps", "p_wald"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GWAS file missing columns: {missing}")

    df = df.copy()

    n_total = len(df)

    df = df[df["rs"].astype(str).str.startswith("SNP-", na=False)].copy()

    n_snp = len(df)
    n_removed = n_total - n_snp

    print(
        f"        SNP records kept: {n_snp:,}; "
        f"INDEL/SV/non-SNP records removed: {n_removed:,}"
    )

    df["chr"] = df["chr"].astype(int)
    df["pos"] = df["ps"].astype(int)

    df["chr_pos"] = df.apply(
        lambda r: make_chr_pos_key(r["chr"], r["pos"]),
        axis=1
    )

    sig_df = df[df["p_wald"] < p_threshold].copy()

    print(f"        Significant SNP records: {len(sig_df):,}")

    return df, sig_df


def load_multiple_gwas_significant(paths, p_threshold):
    all_df_list = []
    sig_df_list = []

    for path in paths:
        print(f"      - {path}")

        df, sig_df = load_gwas_significant(path, p_threshold)

        df["gwas_file"] = path
        sig_df["gwas_file"] = path

        all_df_list.append(df)
        sig_df_list.append(sig_df)

    if len(all_df_list) == 0:
        raise ValueError("No GWAS files were provided.")

    all_df = pd.concat(all_df_list, axis=0, ignore_index=True)
    sig_df = pd.concat(sig_df_list, axis=0, ignore_index=True)

    all_df = all_df.drop_duplicates(subset=["chr_pos"]).copy()
    sig_df = sig_df.drop_duplicates(subset=["chr_pos"]).copy()

    return all_df, sig_df


def load_markers(path):
    df = pd.read_csv(path, sep="\t")

    required = {"chr", "pos", "source", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Marker source file missing columns: {missing}")

    df = df.copy()

    n_total = len(df)

    df = df[df["type"].astype(str).str.lower() == "snp"].copy()

    n_snp = len(df)
    n_removed = n_total - n_snp

    print(
        f"        Marker SNP records kept: {n_snp:,}; "
        f"INDEL/SV/non-SNP marker records removed: {n_removed:,}"
    )

    df["chr"] = df["chr"].astype(int)
    df["pos"] = df["pos"].astype(int)

    df["chr_pos"] = df.apply(
        lambda r: make_chr_pos_key(r["chr"], r["pos"]),
        axis=1
    )

    return df


def build_marker_annotation(markers_df):
    ann = {}

    for _, row in markers_df.iterrows():
        key = row["chr_pos"]
        source = str(row["source"]).strip()

        if key not in ann:
            ann[key] = {
                "sources": set(),
                "type": set(),
                "population": set(),
            }

        ann[key]["sources"].add(source)

        if "type" in row and pd.notna(row["type"]):
            ann[key]["type"].add(str(row["type"]).strip())

        if "population" in row and pd.notna(row["population"]):
            ann[key]["population"].add(str(row["population"]).strip())

    for key, info in ann.items():
        sources_lower = {s.lower() for s in info["sources"]}

        if "ricenavi" in sources_lower:
            info["main_category"] = "RiceNavi"

        elif "gwas_ld" in sources_lower:
            info["main_category"] = "Other-trait GWAS"

        elif any("public_gwas" in s for s in sources_lower):
            info["main_category"] = "Public data GWAS"

        elif "wg_ld" in sources_lower:
            info["main_category"] = "WG_LD"

        else:
            info["main_category"] = "Other"

    return ann


def annotate_topk(topk_df, marker_ann, trait_sig_pos_set):
    out = topk_df.copy()

    categories = []
    in_trait_gwas = []

    for _, row in out.iterrows():
        key = row["chr_pos"]

        is_trait_gwas = key in trait_sig_pos_set
        in_trait_gwas.append(is_trait_gwas)

        if is_trait_gwas:
            final_cat = "Target-trait GWAS"
        else:
            if key in marker_ann:
                final_cat = marker_ann[key]["main_category"]
            else:
                final_cat = "Other"

        categories.append(final_cat)

    out["in_target_trait_gwas"] = in_trait_gwas
    out["source_category"] = categories

    return out


def build_sets(topk_annot_df, topk_set):
    return {
        "Top-K importance": set(topk_set),

        "Target-trait GWAS": set(
            topk_annot_df.loc[
                topk_annot_df["source_category"] == "Target-trait GWAS",
                "chr_pos"
            ]
        ),

        "RiceNavi": set(
            topk_annot_df.loc[
                topk_annot_df["source_category"] == "RiceNavi",
                "chr_pos"
            ]
        ),

        "Other-trait GWAS": set(
            topk_annot_df.loc[
                topk_annot_df["source_category"] == "Other-trait GWAS",
                "chr_pos"
            ]
        ),

        "Public data GWAS": set(
            topk_annot_df.loc[
                topk_annot_df["source_category"] == "Public data GWAS",
                "chr_pos"
            ]
        ),

        "WG_LD": set(
            topk_annot_df.loc[
                topk_annot_df["source_category"] == "WG_LD",
                "chr_pos"
            ]
        ),

        "Other": set(
            topk_annot_df.loc[
                topk_annot_df["source_category"] == "Other",
                "chr_pos"
            ]
        ),
    }


def build_global_category_sets(marker_ann, trait_gwas_set, topk_set):
    global_sets = {
        "Top-K importance": set(topk_set),
        "Target-trait GWAS": set(),
        "RiceNavi": set(),
        "Other-trait GWAS": set(),
        "Public data GWAS": set(),
        "WG_LD": set(),
        "Other": set(),
    }

    for key, info in marker_ann.items():
        if key in trait_gwas_set:
            global_sets["Target-trait GWAS"].add(key)
            continue

        cat = info.get("main_category", "Other")

        if cat in global_sets:
            global_sets[cat].add(key)
        else:
            global_sets["Other"].add(key)

    return global_sets


def build_global_set_sizes(marker_ann, trait_gwas_set, topk_set):
    global_sets = build_global_category_sets(
        marker_ann=marker_ann,
        trait_gwas_set=trait_gwas_set,
        topk_set=topk_set
    )

    return {k: len(v) for k, v in global_sets.items()}


def build_intersection_annotation_info(sets_dict, global_set_sizes):
    categories = [
        "Target-trait GWAS",
        "RiceNavi",
        "Other-trait GWAS",
        "Public data GWAS",
        "WG_LD",
        "Other",
    ]

    records = []

    topk_set = sets_dict.get("Top-K importance", set())

    for cat in categories:
        cat_set = sets_dict.get(cat, set())
        count = len(topk_set & cat_set)

        denominator = int(global_set_sizes.get(cat, 0))
        percent = 0.0 if denominator == 0 else count / denominator * 100

        records.append({
            "category": cat,
            "count": count,
            "denominator": denominator,
            "percent": percent,
        })

    records = sorted(
        records,
        key=lambda x: x["percent"],
        reverse=True
    )

    return records


def save_source_percentage_tsv(annotation_records, trait_name, output_tsv):
    rows = []

    for rec in annotation_records:
        rows.append({
            "Trait": trait_name,
            "Category": rec["category"],
            "TopK_count": int(rec["count"]),
            "Total_loci": int(rec["denominator"]),
            "Percentage_of_total_loci": float(rec["percent"])
        })

    out_df = pd.DataFrame(rows)

    out_df.to_csv(output_tsv, sep="\t", index=False)

    print(f"Saved source percentage summary: {output_tsv}")


def make_subset_key(category_names, active_categories):
    active_categories = set(active_categories)
    return tuple(cat in active_categories for cat in category_names)


def reorder_upset_intersections_by_percent(upset, annotation_records, max_intersections):
    if not hasattr(upset, "intersections"):
        return annotation_records

    category_names = list(upset.intersections.index.names)

    desired_keys = []
    kept_records = []
    percent_values = []

    for rec in annotation_records:
        if rec["count"] <= 0:
            continue

        cat = rec["category"]

        key = make_subset_key(
            category_names=category_names,
            active_categories={"Top-K importance", cat}
        )

        if key in upset.intersections.index:
            desired_keys.append(key)
            kept_records.append(rec)
            percent_values.append(float(rec["percent"]))

    if max_intersections is not None and max_intersections > 0:
        desired_keys = desired_keys[:max_intersections]
        kept_records = kept_records[:max_intersections]
        percent_values = percent_values[:max_intersections]

    if len(desired_keys) == 0:
        return annotation_records

    upset.intersections = upset.intersections.reindex(desired_keys)
    upset.intersections.loc[:] = percent_values

    return kept_records


def save_fig(fig, output_path):
    suffix = output_path.rsplit(".", 1)[-1].lower()

    if suffix == "pdf":
        fig.savefig(output_path, bbox_inches="tight")
    else:
        fig.savefig(output_path, dpi=400, bbox_inches="tight")

    print(f"Saved: {output_path}")


def annotate_percentage_bars(axes, annotation_records):
    if "intersections" not in axes:
        return

    ax = axes["intersections"]

    for text in list(ax.texts):
        text.remove()

    patches = list(ax.patches)

    patches = sorted(
        patches,
        key=lambda p: p.get_x() + p.get_width() / 2
    )

    max_height = max([p.get_height() for p in patches], default=1)

    n = min(len(patches), len(annotation_records))

    for patch, rec in zip(patches[:n], annotation_records[:n]):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()

        ax.text(
            x,
            y + max_height * 0.035,
            f"{rec['count']:,}",
            ha="center",
            va="bottom",
            fontsize=8.5
        )

    ax.set_ylim(0, max_height * 1.25)


def plot_upset(
    sets_dict,
    global_set_sizes,
    trait_name,
    output_path,
    max_intersections=6
):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    filtered_sets = {
        k: v for k, v in sets_dict.items()
        if len(v) > 0
    }

    if "Top-K importance" not in filtered_sets:
        raise ValueError(
            "Top-K importance set is empty. Please check the importance file and --top-k."
        )

    data = from_contents(filtered_sets)

    fig = plt.figure(figsize=(8.4, 7.3), facecolor="white")

    upset = UpSet(
        data,
        subset_size="count",
        show_counts=False,
        sort_by=None,
        sort_categories_by=None,
        facecolor="#4C78A8",
        element_size=28,
        intersection_plot_elements=max_intersections,
        totals_plot_elements=0,
    )

    annotation_records = build_intersection_annotation_info(
        sets_dict=sets_dict,
        global_set_sizes=global_set_sizes
    )

    annotation_records = reorder_upset_intersections_by_percent(
        upset=upset,
        annotation_records=annotation_records,
        max_intersections=max_intersections
    )

    axes = upset.plot(fig=fig)

    if "intersections" in axes:
        axes["intersections"].set_ylabel(
            f"Percentage of total loci (%)\n({trait_name})",
            fontsize=10,
            fontweight="bold"
        )

        axes["intersections"].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0f}%")
        )

        axes["intersections"].spines["top"].set_visible(False)
        axes["intersections"].spines["right"].set_visible(False)
        axes["intersections"].tick_params(
            axis="both",
            direction="out",
            length=4
        )

    if "matrix" in axes:
        axes["matrix"].tick_params(axis="both", length=0)

    # Uncomment if you want count labels on upper bars
    # annotate_percentage_bars(
    #     axes=axes,
    #     annotation_records=annotation_records
    # )

    save_fig(fig, output_path)
    plt.close(fig)


def main():
    args = parse_args()

    print("=" * 80)
    print("Non-redundant Top-K SNP source UpSet plot")
    print("=" * 80)

    print(f"[1/4] Loading importance: {args.importance}")
    _, imp_topk_df = load_importance_topk(
        path=args.importance,
        top_k=args.top_k
    )

    print(f"[2/4] Loading GWAS files:")
    _, gwas_sig_df = load_multiple_gwas_significant(
        paths=args.gwas,
        p_threshold=args.p_threshold
    )

    print(f"[3/4] Loading markers: {args.markers}")
    markers_df = load_markers(args.markers)

    topk_set = set(imp_topk_df["chr_pos"])

    raw_trait_gwas_set = set(gwas_sig_df["chr_pos"])
    marker_panel_set = set(markers_df["chr_pos"])

    trait_gwas_set = raw_trait_gwas_set & marker_panel_set

    print(f"[4/4] Annotating Top-K loci")
    marker_ann = build_marker_annotation(markers_df)

    topk_annot_df = annotate_topk(
        topk_df=imp_topk_df,
        marker_ann=marker_ann,
        trait_sig_pos_set=trait_gwas_set
    )

    source_counts = Counter(topk_annot_df["source_category"])

    print("\nNon-redundant source composition within Top-K:")
    for cat in [
        "Target-trait GWAS",
        "RiceNavi",
        "Other-trait GWAS",
        "Public data GWAS",
        "WG_LD",
        "Other",
    ]:
        print(f"  {cat:28s}: {int(source_counts.get(cat, 0))}")

    print("\nGWAS / marker filtering summary:")
    print(f"  Number of GWAS files                  : {len(args.gwas)}")
    print(f"  Raw merged significant GWAS SNP loci   : {len(raw_trait_gwas_set)}")
    print(f"  Marker-panel SNP loci                  : {len(marker_panel_set)}")
    print(f"  Target-trait GWAS in marker-panel SNPs : {len(trait_gwas_set)}")
    print(f"  Top-K loci                             : {len(topk_set)}")
    print(f"  Top-K ∩ Target-trait GWAS              : {len(topk_set & trait_gwas_set)}")

    sets_dict = build_sets(
        topk_annot_df=topk_annot_df,
        topk_set=topk_set
    )

    global_set_sizes = build_global_set_sizes(
        marker_ann=marker_ann,
        trait_gwas_set=trait_gwas_set,
        topk_set=topk_set
    )

    print("\nNon-redundant global SNP source set sizes:")
    for cat in [
        "Top-K importance",
        "Target-trait GWAS",
        "RiceNavi",
        "Other-trait GWAS",
        "Public data GWAS",
        "WG_LD",
        "Other",
    ]:
        print(f"  {cat:28s}: {int(global_set_sizes.get(cat, 0))}")

    print("\nIntersection percentage summary:")
    records = build_intersection_annotation_info(
        sets_dict=sets_dict,
        global_set_sizes=global_set_sizes
    )

    for rec in records:
        print(
            f"  {rec['category']:28s}: "
            f"{rec['count']:6,d} / {rec['denominator']:6,d} = {rec['percent']:.2f}%"
        )

    if args.summary_tsv is None:
        if "." in args.output:
            args.summary_tsv = args.output.rsplit(".", 1)[0] + ".source_percentage.tsv"
        else:
            args.summary_tsv = args.output + ".source_percentage.tsv"

    save_source_percentage_tsv(
        annotation_records=records,
        trait_name=args.trait,
        output_tsv=args.summary_tsv
    )

    plot_upset(
        sets_dict=sets_dict,
        global_set_sizes=global_set_sizes,
        trait_name=args.trait,
        output_path=args.output,
        max_intersections=args.max_intersections
    )

    print("\nDone.")


if __name__ == "__main__":
    main()