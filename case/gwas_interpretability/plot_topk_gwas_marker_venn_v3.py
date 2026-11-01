#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Non-redundant UpSetPlot for one production model's top-K SNP loci and source categories.

Outputs:
1. UpSet figure
2. Source percentage TSV, including a one-sided hypergeometric enrichment test
   of Top-K loci versus each source category inside the model SNP universe
"""

import argparse
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

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
        help="Output TSV with exclusive source percentages and hypergeometric enrichment.",
    )

    return parser.parse_args()


SOURCE_CATEGORIES = [
    "Target-trait GWAS",
    "RiceNavi",
    "Other-trait GWAS",
    "Public data GWAS",
    "WG_LD",
]

_SOURCE_PRIORITY = (
    ({"ricenavi", "ricenavi_qtn"}, "RiceNavi"),
    (
        {"current_cohort_gwas_ld", "other_cohort_gwas_ld", "gwas_ld"},
        "Other-trait GWAS",
    ),
    ({"gw_ld", "wg_ld"}, "WG_LD"),
)


def categories_for_sources(sources):
    """Return all matching source categories; a locus may belong to more than one."""
    lowered = {str(source).strip().lower() for source in sources if str(source).strip()}
    matched = []
    for names, category in _SOURCE_PRIORITY:
        if lowered & names:
            matched.append(category)
    if any(name.startswith("public_gwas") for name in lowered):
        matched.append("Public data GWAS")
    return matched


def category_for_sources(sources):
    categories = categories_for_sources(sources)
    return categories[0] if categories else None


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

    chr_list, pos_list, key_list = [], [], []
    for locus in df["locus_id"]:
        chrom, pos = parse_locus_id_to_chr_pos(locus)
        chr_list.append(chrom)
        pos_list.append(pos)
        key_list.append(make_chr_pos_key(chrom, pos))

    df = df.copy()
    df["chr"] = chr_list
    df["pos"] = pos_list
    df["chr_pos"] = key_list
    df_top = df[df["rank"] <= top_k].copy()
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
    rename = {}
    if "CHROM" in df.columns and "chr" not in df.columns:
        rename["CHROM"] = "chr"
    if "POS" in df.columns and "pos" not in df.columns:
        rename["POS"] = "pos"
    if "Variant_Type" in df.columns and "type" not in df.columns:
        rename["Variant_Type"] = "type"
    if "Selected_Source" in df.columns and "source" not in df.columns:
        rename["Selected_Source"] = "source"
    elif "Source" in df.columns and "source" not in df.columns:
        rename["Source"] = "source"
    df = df.rename(columns=rename)

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
        raw_sources = row["All_Sources"] if "All_Sources" in row and pd.notna(row["All_Sources"]) else row["source"]
        sources = [part.strip() for part in str(raw_sources).split(",") if part.strip()]

        if key not in ann:
            ann[key] = {
                "sources": set(),
                "type": set(),
                "population": set(),
            }

        ann[key]["sources"].update(sources)

        if "type" in row and pd.notna(row["type"]):
            ann[key]["type"].add(str(row["type"]).strip())

        if "population" in row and pd.notna(row["population"]):
            ann[key]["population"].add(str(row["population"]).strip())

    for key, info in ann.items():
        info["categories"] = categories_for_sources(info["sources"])
        info["main_category"] = (
            info["categories"][0] if info["categories"] else None
        )

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
                final_cat = None

        categories.append(final_cat)

    out["in_target_trait_gwas"] = in_trait_gwas
    out["source_category"] = categories

    return out


def build_sets(topk_annot_df, topk_set):
    sets = {"Top-K importance": set(topk_set)}
    for category in SOURCE_CATEGORIES:
        sets[category] = set(
            topk_annot_df.loc[
                topk_annot_df["source_category"] == category,
                "chr_pos",
            ]
        )
    return sets


def build_global_category_sets(marker_ann, trait_gwas_set, topk_set):
    global_sets = {"Top-K importance": set(topk_set)}
    global_sets.update({category: set() for category in SOURCE_CATEGORIES})

    for key, info in marker_ann.items():
        if key in trait_gwas_set:
            global_sets["Target-trait GWAS"].add(key)
            continue

        cat = info.get("main_category")

        if cat in global_sets:
            global_sets[cat].add(key)

    return global_sets


def build_global_set_sizes(marker_ann, trait_gwas_set, topk_set):
    global_sets = build_global_category_sets(
        marker_ann=marker_ann,
        trait_gwas_set=trait_gwas_set,
        topk_set=topk_set
    )

    return {k: len(v) for k, v in global_sets.items()}


def hypergeometric_one_sided(overlap, universe, n_category, n_draw):
    """One-sided P(X >= overlap) for drawing n_draw from universe with n_category red balls."""
    if universe <= 0 or n_category <= 0 or n_draw <= 0:
        return {
            "expected_overlap": float("nan"),
            "fold_enrichment": float("nan"),
            "hypergeometric_p": float("nan"),
        }
    expected = n_draw * n_category / universe
    fold = overlap / expected if expected > 0 else float("nan")
    p_value = float(hypergeom.sf(overlap - 1, universe, n_category, n_draw))
    return {
        "expected_overlap": expected,
        "fold_enrichment": fold,
        "hypergeometric_p": p_value,
    }


def build_universe_category_sets(universe, marker_ann, trait_gwas_set):
    """Non-exclusive source membership restricted to model SNPs."""
    category_sets = {category: set() for category in SOURCE_CATEGORIES}
    for key in universe:
        if key in trait_gwas_set:
            category_sets["Target-trait GWAS"].add(key)
        if key in marker_ann:
            for category in marker_ann[key].get(
                "categories", []
            ):
                if category in category_sets:
                    category_sets[category].add(key)
    return category_sets


def build_enrichment_records(universe, topk_set, category_sets):
    universe_size = len(universe)
    n_draw = len(topk_set)
    records = []
    for category in SOURCE_CATEGORIES:
        red = set(category_sets.get(category, set())) & universe
        overlap = topk_set & red
        stats = hypergeometric_one_sided(
            len(overlap), universe_size, len(red), n_draw
        )
        records.append(
            {
                "category": category,
                "universe": universe_size,
                "category_in_universe": len(red),
                "top_k": n_draw,
                "observed_overlap": len(overlap),
                **stats,
            }
        )
    n_tests = sum(1 for record in records if record["category_in_universe"] > 0)
    for record in records:
        p_value = record["hypergeometric_p"]
        if n_tests and p_value == p_value:
            record["bonferroni_p"] = min(1.0, float(p_value) * n_tests)
        else:
            record["bonferroni_p"] = float("nan")
    return records


def build_intersection_annotation_info(sets_dict, global_set_sizes):
    categories = list(SOURCE_CATEGORIES)

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


def save_source_percentage_tsv(
    annotation_records, enrichment_records, trait_name, output_tsv
):
    enrichment_by_category = {
        record["category"]: record for record in enrichment_records
    }
    rows = []
    for rec in annotation_records:
        enrichment = enrichment_by_category.get(rec["category"], {})
        rows.append(
            {
                "Trait": trait_name,
                "Category": rec["category"],
                "TopK_count": int(rec["count"]),
                "Total_loci": int(rec["denominator"]),
                "Percentage_of_total_loci": float(rec["percent"]),
                "Universe": enrichment.get("universe"),
                "Category_in_universe": enrichment.get("category_in_universe"),
                "Observed_overlap": enrichment.get("observed_overlap"),
                "Expected_overlap": enrichment.get("expected_overlap"),
                "Fold_enrichment": enrichment.get("fold_enrichment"),
                "Hypergeometric_p": enrichment.get("hypergeometric_p"),
                "Bonferroni_p": enrichment.get("bonferroni_p"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Saved source percentage and enrichment summary: {output_tsv}")


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

    upset.intersections = upset.intersections.reindex(desired_keys).astype(float)
    upset.intersections.loc[:] = percent_values

    return kept_records


def save_fig(fig, output_path):
    suffix = output_path.rsplit(".", 1)[-1].lower()

    if suffix == "pdf":
        fig.savefig(output_path, bbox_inches="tight")
    else:
        fig.savefig(output_path, dpi=400, bbox_inches="tight")

    print(f"Saved: {output_path}")


def significance_symbol(p_value):
    """Return a conventional significance symbol for an adjusted P value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def annotate_significance_bars(axes, annotation_records, enrichment_records):
    """Annotate bars with symbols based on Bonferroni-adjusted P values."""
    if "intersections" not in axes:
        return

    ax = axes["intersections"]
    patches = sorted(list(ax.patches), key=lambda patch: patch.get_x() + patch.get_width() / 2)
    if not patches:
        return

    enrichment_by_category = {
        record["category"]: record for record in enrichment_records
    }
    max_height = max(patch.get_height() for patch in patches)
    n_bars = min(len(patches), len(annotation_records))
    for patch, rec in zip(patches[:n_bars], annotation_records[:n_bars]):
        p_value = enrichment_by_category.get(rec["category"], {}).get("bonferroni_p")
        if p_value is None or p_value != p_value:
            continue
        label = significance_symbol(p_value)
        if not label:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + max_height * 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, max_height * 1.18)


def plot_upset(
    sets_dict,
    global_set_sizes,
    trait_name,
    output_path,
    max_intersections=6,
    enrichment_records=None,
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

    if enrichment_records:
        annotate_significance_bars(
            axes=axes,
            annotation_records=annotation_records,
            enrichment_records=enrichment_records,
        )

    save_fig(fig, output_path)
    plt.close(fig)


def main():
    args = parse_args()

    print("=" * 80)
    print("Non-redundant Top-K SNP source UpSet plot")
    print("=" * 80)

    print(f"[1/4] Loading importance: {args.importance}")
    imp_df, imp_topk_df = load_importance_topk(
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
    universe = set(imp_df["chr_pos"])

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
    for cat in SOURCE_CATEGORIES:
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
    for cat in ["Top-K importance", *SOURCE_CATEGORIES]:
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

    universe_category_sets = build_universe_category_sets(
        universe=universe,
        marker_ann=marker_ann,
        trait_gwas_set=trait_gwas_set & universe,
    )
    enrichment_records = build_enrichment_records(
        universe=universe,
        topk_set=topk_set,
        category_sets=universe_category_sets,
    )

    print("\nHypergeometric enrichment of Top-K inside the model SNP universe:")
    print(
        f"  Universe = {len(universe):,} model SNPs; "
        f"drawing Top-K = {len(topk_set):,}"
    )
    for rec in enrichment_records:
        p_value = rec["hypergeometric_p"]
        p_text = "NA" if p_value != p_value else f"{p_value:.6g}"
        fold = rec["fold_enrichment"]
        fold_text = "NA" if fold != fold else f"{fold:.2f}"
        expected = rec["expected_overlap"]
        expected_text = "NA" if expected != expected else f"{expected:.2f}"
        bonf = rec["bonferroni_p"]
        bonf_text = "NA" if bonf != bonf else f"{bonf:.6g}"
        print(
            f"  {rec['category']:28s}: "
            f"{rec['observed_overlap']:4d}/{rec['category_in_universe']:5d} "
            f"observed/red; expected={expected_text}; "
            f"fold={fold_text}; p={p_text}; Bonferroni p={bonf_text}"
        )

    if args.summary_tsv is None:
        if "." in args.output:
            args.summary_tsv = args.output.rsplit(".", 1)[0] + ".source_percentage.tsv"
        else:
            args.summary_tsv = args.output + ".source_percentage.tsv"

    save_source_percentage_tsv(
        annotation_records=records,
        enrichment_records=enrichment_records,
        trait_name=args.trait,
        output_tsv=args.summary_tsv
    )

    plot_upset(
        sets_dict=sets_dict,
        global_set_sizes=global_set_sizes,
        trait_name=args.trait,
        output_path=args.output,
        max_intersections=args.max_intersections,
        enrichment_records=enrichment_records,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()