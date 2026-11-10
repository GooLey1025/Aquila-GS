#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnnotationBbox, TextArea, HPacker


def format_mean_std(center, std):
    """Show enough decimals for a non-zero SD to stay visible."""
    for digits in range(3, 8):
        std_text = f"{std:.{digits}f}"
        if std == 0 or float(std_text) != 0:
            return f"{center:.{digits}f} ± {std_text}"
    return f"{center:.5f} ± {std:.5f}"


def label_fontsize(text, base=11.0, max_chars=26, floor=7.0):
    if len(text) <= max_chars:
        return base
    return max(floor, base * max_chars / len(text))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publication-style RiceNavi QTN SI gain panel using mean gain and evolved major GT across seeds."
    )

    parser.add_argument("--evolve-dir", required=True)
    parser.add_argument("--screening-name", default="screening_si_per_round.tsv")
    parser.add_argument(
        "--evolved-vcf-name",
        default="Teqing__SAMN04505840.1171rice.snp.impute.biallelic_evolve.vcf.gz",
        help="Evolved VCF file name inside each seed directory."
    )

    parser.add_argument("--ricenavi-file", required=True)
    parser.add_argument("--base-vcf", required=True, help="VCF before evolution, e.g. Teqing baseline VCF.")
    parser.add_argument(
        "--hhz-vcf",
        default=None,
        help="Huanghuazhan VCF. When set, the genotype panel shows Teqing, Huanghuazhan, and the evolved genotype."
    )

    parser.add_argument("--out-prefix", default="QTN_gain_contribution_seed_mean")
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Number of top QTNs to plot. 0 means plot all matched QTNs."
    )

    parser.add_argument("--gain-col", default="SI_evolved",
                        help="Column name for the per-round marginal SI. SI_evolved is the change from the genome at the start of that round.")
    parser.add_argument("--skiprows", type=int, default=1)

    parser.add_argument("--fig-width", type=float, default=11)
    parser.add_argument(
        "--row-height",
        type=float,
        default=0.5,
        help="Figure height per plotted site."
    )
    parser.add_argument("--min-fig-height", type=float, default=8)
    parser.add_argument("--max-fig-height", type=float, default=80.0)

    parser.add_argument("--base-label", default="Before\noptimalized")
    parser.add_argument("--evolved-label", default="After\noptimalized")
    parser.add_argument(
        "--all-sites",
        action="store_true",
        help="Dump gain statistics for ALL SNP sites (not just RiceNavi QTNs) "
             "and exit without plotting."
    )

    return parser.parse_args()


def run_cmd(cmd):
    res = subprocess.run(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}\n\nSTDERR:\n{res.stderr}")
    return res.stdout


def clean_text(x):
    return str(x).strip().replace("\r", "").replace("\ufeff", "")


def normalize_gt(gt):
    gt = clean_text(gt).split(":")[0]

    if gt in [".", "./.", ".|.", "nan", "None", ""]:
        return ".|."

    gt = gt.replace("/", "|")
    alleles = gt.split("|")

    if len(alleles) != 2 or "." in alleles:
        return ".|."

    return "|".join(alleles)


def gt_display(gt):
    gt = normalize_gt(gt)
    return gt if gt != ".|." else "NA"


def alt_dosage(gt):
    gt = normalize_gt(gt)
    if gt == ".|.":
        return None
    try:
        return sum(int(allele) for allele in gt.split("|"))
    except ValueError:
        return None


def classify_hhz_selection(teqing_gt, hhz_gt, evolved_major_gt):
    teqing_gt = normalize_gt(teqing_gt)
    hhz_gt = normalize_gt(hhz_gt)
    evolved_gt = normalize_gt(evolved_major_gt)

    if teqing_gt == ".|." or hhz_gt == ".|." or evolved_gt == ".|.":
        return "Unknown"
    if evolved_gt == teqing_gt:
        return "Unknown"
    if evolved_gt == hhz_gt and hhz_gt != teqing_gt:
        return "HHZ-selected"
    if hhz_gt == teqing_gt and evolved_gt != teqing_gt:
        return "Novel-selected"

    t_dosage = alt_dosage(teqing_gt)
    h_dosage = alt_dosage(hhz_gt)
    e_dosage = alt_dosage(evolved_gt)
    if t_dosage is None or h_dosage is None or e_dosage is None:
        return "Novel-selected"
    if t_dosage < h_dosage and t_dosage < e_dosage < h_dosage:
        return "Partial-HHZ"
    if t_dosage > h_dosage and h_dosage < e_dosage < t_dosage:
        return "Partial-HHZ"
    return "Novel-selected"


def annotate_source_qtns(gain_summary, ricenavi):
    """Attach RiceNavi gene and trait labels through source QTN IDs."""
    lookup = ricenavi.drop_duplicates("SNP_ID").set_index("SNP_ID")
    gene_labels = []
    trait_labels = []
    for source_text in gain_summary["source_qtn_ids"]:
        genes = []
        traits = []
        for source_id in str(source_text).split(","):
            source_id = source_id.strip()
            if source_id not in lookup.index:
                continue
            row = lookup.loc[source_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            genes.append(str(row["GeneName"]))
            traits.append(row["Trait"])
        gene_labels.append(";".join(dict.fromkeys(genes)) if genes else np.nan)
        trait_labels.append(traits[0] if traits else np.nan)
    annotated = gain_summary.copy()
    annotated["GeneName"] = gene_labels
    annotated["Trait"] = trait_labels
    return annotated


def annotate_by_coordinate(gain_summary, ricenavi):
    """Match evaluated VCF IDs to RiceNavi by chromosome and position."""
    grouped = {}
    for _, row in ricenavi.iterrows():
        key = (str(row["Chr_clean"]), int(row["Pos_clean"]))
        bucket = grouped.setdefault(key, {"genes": [], "traits": []})
        gene = str(row["GeneName"])
        trait = str(row["Trait"])
        if gene not in bucket["genes"]:
            bucket["genes"].append(gene)
        if trait not in bucket["traits"]:
            bucket["traits"].append(trait)
    genes = []
    traits = []
    for snp_id in gain_summary["evaluated_snp_id"]:
        parsed = parse_snp_id(snp_id)
        chrom, pos = parsed.iloc[0], parsed.iloc[1]
        bucket = None
        if pd.notna(chrom) and pd.notna(pos):
            chrom = str(chrom)
            if chrom.lower().startswith("chr"):
                chrom = chrom[3:]
            bucket = grouped.get((chrom, int(pos)))
        genes.append(";".join(bucket["genes"]) if bucket else np.nan)
        traits.append(bucket["traits"][0] if bucket else np.nan)
    annotated = gain_summary.copy()
    annotated["GeneName"] = genes
    annotated["Trait"] = traits
    return annotated


def trait_category_en(trait):
    mapping = {
        "产量组成相关": "Yield component",
        "其他": "Other",
        "口感品质": "Eating quality",
        "抽穗期": "Heading date",
        "植株形态": "Plant architecture",
        "次生代谢相关": "Secondary metabolism",
        "生物胁迫": "Biotic stress",
        "种子形态": "Seed morphology",
        "非生物胁迫": "Abiotic stress",
    }
    return mapping.get(str(trait).strip(), "Other")


def parse_snp_id(snp_id):
    try:
        parts = str(snp_id).strip().split("-")
        chrom = str(parts[1]).strip()
        pos = int(parts[2])
        return pd.Series([chrom, pos])
    except Exception:
        return pd.Series([pd.NA, pd.NA])


def seed_sort_key(path):
    name = os.path.basename(path.rstrip("/"))
    if name.startswith("seed_"):
        try:
            return int(name.replace("seed_", ""))
        except ValueError:
            return name
    return name


def find_seed_dirs(evolve_dir):
    seed_dirs = sorted(
        [
            d for d in glob.glob(os.path.join(evolve_dir, "seed_*"))
            if os.path.isdir(d)
        ],
        key=seed_sort_key
    )

    if seed_dirs:
        return seed_dirs
    if os.path.isfile(os.path.join(evolve_dir, "screening_si_per_round.tsv")):
        return [evolve_dir]

    raise FileNotFoundError(
        f"No seed_* directories or screening_si_per_round.tsv found under: {evolve_dir}"
    )


def load_screening_one_file(path, skiprows, gain_col):
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().lstrip("#").strip().split("\t")
    if "Evaluated_SNP_ID" in header:
        df = pd.read_csv(path, sep="\t", comment=None)
        df.columns = [str(column).lstrip("#") for column in df.columns]
        df = df.rename(columns={
            "Evaluated_SNP_ID": "evaluated_snp_id",
            "Source_QTN_IDs": "source_qtn_ids",
        })
    else:
        screening_cols = [
            "round",
            "evaluated_snp",
            "evaluated_snp_id",
            "SI_baseline",
            "SI_evolved",
            "SI_gain",
            "score_baseline",
            "score_evolved",
            "gain",
            "accepted",
        ]
        df = pd.read_csv(
            path,
            sep="\t",
            skiprows=skiprows,
            header=None,
            names=screening_cols,
        )

    df["evaluated_snp_id"] = df["evaluated_snp_id"].astype(str).str.strip()
    if "source_qtn_ids" in df.columns:
        df["source_qtn_ids"] = df["source_qtn_ids"].fillna("").astype(str).str.strip()

    if gain_col not in df.columns:
        raise ValueError(
            f"--gain-col '{gain_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["gain_plot"] = pd.to_numeric(df[gain_col], errors="coerce")
    columns = ["evaluated_snp_id", "gain_plot"]
    if "source_qtn_ids" in df.columns:
        columns.append("source_qtn_ids")
    return df[columns].copy()


def load_seed_mean_gain(seed_dirs, screening_name, skiprows, gain_col):
    all_records = []
    used_files = []

    for seed_dir in seed_dirs:
        screening_file = os.path.join(seed_dir, screening_name)

        if not os.path.exists(screening_file):
            print(f"[Warning] Missing screening file: {screening_file}")
            continue

        seed = os.path.basename(seed_dir).replace("seed_", "")

        df = load_screening_one_file(
            screening_file,
            skiprows=skiprows,
            gain_col=gain_col
        )

        df["seed"] = seed
        all_records.append(df)
        used_files.append(screening_file)

    if not all_records:
        raise FileNotFoundError(
            f"No valid screening files found: seed_*/{screening_name}"
        )

    all_gain = pd.concat(all_records, ignore_index=True)

    aggregations = {
        "gain_mean": ("gain_plot", "mean"),
        "gain_std": ("gain_plot", "std"),
        "gain_n_seed": ("gain_plot", "count"),
    }
    if "source_qtn_ids" in all_gain.columns:
        aggregations["source_qtn_ids"] = ("source_qtn_ids", "first")
    gain_summary = (
        all_gain
        .groupby("evaluated_snp_id", as_index=False)
        .agg(**aggregations)
    )

    gain_summary["gain_std"] = gain_summary["gain_std"].fillna(0)

    print(f"[INFO] Loaded screening files: {len(used_files)}")
    print(f"[INFO] Unique evaluated SNPs: {gain_summary['evaluated_snp_id'].nunique()}")

    return gain_summary, all_gain


def load_single_sample_gt(vcf, site_ids, label):
    site_set = set(site_ids)

    cmd = f"bcftools query -f '%ID\\t[%GT]\\n' {vcf}"
    out = run_cmd(cmd)

    gt_dict = {}

    for line in out.strip().splitlines():
        arr = line.rstrip("\n").split("\t")
        if len(arr) < 2:
            continue

        snp_id = clean_text(arr[0])
        if snp_id not in site_set:
            continue

        gt_dict[snp_id] = normalize_gt(arr[1])

    print(f"[INFO] {label} matched GT sites: {len(gt_dict)} / {len(site_ids)}")
    return gt_dict


def load_one_seed_evolved_gt(vcf, site_ids):
    site_set = set(site_ids)

    cmd = f"bcftools query -f '%ID\\t[%GT]\\n' {vcf}"
    out = run_cmd(cmd)

    gt_dict = {}

    for line in out.strip().splitlines():
        arr = line.rstrip("\n").split("\t")
        if len(arr) < 2:
            continue

        snp_id = clean_text(arr[0])
        if snp_id not in site_set:
            continue

        gt_dict[snp_id] = normalize_gt(arr[1])

    return gt_dict


def load_evolved_major_gt_across_seed_vcfs(seed_dirs, evolved_vcf_name, site_ids):
    per_site_gts = {sid: [] for sid in site_ids}
    used_vcfs = []

    for seed_dir in seed_dirs:
        vcf = os.path.join(seed_dir, evolved_vcf_name)

        if not os.path.exists(vcf):
            print(f"[Warning] Missing evolved VCF: {vcf}")
            continue

        gt_dict = load_one_seed_evolved_gt(vcf, site_ids)

        for sid, gt in gt_dict.items():
            per_site_gts[sid].append(gt)

        used_vcfs.append(vcf)

    if not used_vcfs:
        raise FileNotFoundError(
            f"No evolved VCF files found: seed_*/{evolved_vcf_name}"
        )

    major_dict = {}
    freq_dict = {}
    count_dict = {}

    gt_order = ["0|0", "0|1", "1|0", "1|1"]

    for sid in site_ids:
        gts = per_site_gts.get(sid, [])

        counts = Counter(gts)
        n_missing = counts.get(".|.", 0)
        n_valid = len(gts) - n_missing

        valid_counts = {gt: counts.get(gt, 0) for gt in gt_order}

        if n_valid > 0:
            major_gt = max(gt_order, key=lambda gt: valid_counts.get(gt, 0))
            major_n = valid_counts[major_gt]
            major_freq = major_n / n_valid
        else:
            major_gt = ".|."
            major_freq = np.nan

        major_dict[sid] = major_gt
        freq_dict[sid] = major_freq

        count_dict[sid] = {
            "N_valid": n_valid,
            "N_0|0": valid_counts["0|0"],
            "N_0|1": valid_counts["0|1"],
            "N_1|0": valid_counts["1|0"],
            "N_1|1": valid_counts["1|1"],
            "N_missing": n_missing,
        }

    print(f"[INFO] Loaded evolved seed VCFs: {len(used_vcfs)}")
    print(f"[INFO] Evolved GT sites summarized: {len(major_dict)}")

    return major_dict, freq_dict, count_dict


def draw_gt_box(ax, x, y, text, facecolor, width=0.78, height=0.58):
    rect = Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        linewidth=0,
        facecolor=facecolor,
        alpha=0.88,
        clip_on=False,
    )
    ax.add_patch(rect)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.5,
        color="black",
    )


def write_html_wrapper(svg_path, html_path, title):
    with open(svg_path, "r", encoding="utf-8") as f:
        svg = f.read()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    margin: 0;
    padding: 16px;
    font-family: Arial, sans-serif;
    background: #ffffff;
  }}
  .figure-container {{
    width: 100%;
    overflow-x: auto;
  }}
  svg {{
    max-width: 100%;
    height: auto;
    display: block;
  }}
</style>
</head>
<body>
<div class="figure-container">
{svg}
</div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    args = parse_args()

    seed_dirs = find_seed_dirs(args.evolve_dir)

    gain_summary, all_seed_gain = load_seed_mean_gain(
        seed_dirs=seed_dirs,
        screening_name=args.screening_name,
        skiprows=args.skiprows,
        gain_col=args.gain_col,
    )

    gain_summary[["Chr", "Pos"]] = gain_summary["evaluated_snp_id"].apply(parse_snp_id)

    # --- All-sites mode: dump gain stats and exit ---
    if args.all_sites:
        all_sites_out = f"{args.out_prefix}.all_sites_gain.tsv"
        gain_summary[[
            "evaluated_snp_id", "Chr", "Pos",
            "gain_mean", "gain_std", "gain_n_seed"
        ]].to_csv(all_sites_out, sep="\t", index=False)
        all_seed_gain.to_csv(f"{args.out_prefix}.all_seed_gain.tsv", sep="\t", index=False)
        print(f"[INFO] All-sites gain stats saved to: {all_sites_out}")
        print(f"[INFO] Per-seed gain data saved to: {args.out_prefix}.all_seed_gain.tsv")
        print(f"[INFO] Total sites: {len(gain_summary)}")
        return

    ricenavi = pd.read_csv(args.ricenavi_file)
    ricenavi.columns = (
        ricenavi.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    ricenavi = ricenavi.loc[:, ~ricenavi.columns.duplicated()]

    if "Method_Genotyping" in ricenavi.columns:
        ricenavi = ricenavi[
            ricenavi["Method_Genotyping"]
            .astype(str)
            .str.strip()
            .isin(["GATK3", "GATK4"])
        ].copy()

    if "Pos_7.0" not in ricenavi.columns:
        raise ValueError("RiceNavi file must contain column: Pos_7.0")
    if "Chr" not in ricenavi.columns:
        raise ValueError("RiceNavi file must contain column: Chr")

    ricenavi = ricenavi[
        ~ricenavi["Pos_7.0"].astype(str).str.contains("-", na=False)
    ].copy()

    ricenavi["Chr_clean"] = (
        ricenavi["Chr"]
        .astype(str)
        .str.strip()
        .str.replace(r"^chr", "", case=False, regex=True)
        .str.replace(r"^Chr", "", regex=True)
    )

    ricenavi["Pos_clean"] = pd.to_numeric(
        ricenavi["Pos_7.0"],
        errors="coerce"
    ).astype("Int64")

    ricenavi = ricenavi.dropna(subset=["Chr_clean", "Pos_clean"]).copy()

    ricenavi["SNP_ID"] = (
        "SNP-"
        + ricenavi["Chr_clean"].astype(str)
        + "-"
        + ricenavi["Pos_clean"].astype(str)
        + "-1"
    )

    has_sources = (
        "source_qtn_ids" in gain_summary.columns
        and gain_summary["source_qtn_ids"].astype(str).str.strip().replace("nan", "").ne("").any()
    )
    if has_sources:
        merged = annotate_source_qtns(gain_summary, ricenavi)
    else:
        merged = gain_summary.merge(
            ricenavi,
            left_on="evaluated_snp_id",
            right_on="SNP_ID",
            how="left",
            suffixes=("", "_ricenavi"),
        )
        missing = merged["GeneName"].isna()
        if missing.any():
            by_coordinate = annotate_by_coordinate(gain_summary, ricenavi)
            merged.loc[missing, "GeneName"] = by_coordinate.loc[missing, "GeneName"].to_numpy()
            merged.loc[missing, "Trait"] = by_coordinate.loc[missing, "Trait"].to_numpy()

    matched_all = merged[merged["GeneName"].notna()].copy()

    if len(matched_all) == 0:
        print("[DEBUG] First 5 screening IDs:")
        print(gain_summary["evaluated_snp_id"].head().to_string(index=False))
        print("[DEBUG] First 5 RiceNavi SNP_ID:")
        print(ricenavi["SNP_ID"].head().to_string(index=False))
        raise ValueError("No matched RiceNavi QTNs found.")

    site_ids = matched_all["evaluated_snp_id"].dropna().astype(str).unique().tolist()

    base_gt = load_single_sample_gt(args.base_vcf, site_ids, args.base_label)

    evolved_major_gt, evolved_major_freq, evolved_counts = load_evolved_major_gt_across_seed_vcfs(
        seed_dirs=seed_dirs,
        evolved_vcf_name=args.evolved_vcf_name,
        site_ids=site_ids,
    )

    merged["Base_GT"] = merged["evaluated_snp_id"].map(base_gt)
    merged["Evolved_Major_GT"] = merged["evaluated_snp_id"].map(evolved_major_gt)
    compare_hhz = bool(args.hhz_vcf)
    if compare_hhz:
        hhz_gt = load_single_sample_gt(args.hhz_vcf, site_ids, "Huanghuazhan")
        merged["HHZ_GT"] = merged["evaluated_snp_id"].map(hhz_gt)
        merged["Evolutionary_pattern"] = [
            classify_hhz_selection(base, hhz, evolved)
            for base, hhz, evolved in zip(
                merged["Base_GT"], merged["HHZ_GT"], merged["Evolved_Major_GT"]
            )
        ]
    merged["Evolved_Major_GT_Freq"] = merged["evaluated_snp_id"].map(evolved_major_freq)

    for col in ["N_valid", "N_0|0", "N_0|1", "N_1|0", "N_1|1", "N_missing"]:
        merged[col] = merged["evaluated_snp_id"].map(
            {
                sid: counts.get(col, np.nan)
                for sid, counts in evolved_counts.items()
            }
        )

    merged["Gene_label"] = (
        merged["GeneName"]
        .astype("string")
        .fillna(merged["evaluated_snp_id"].astype("string"))
    )

    merged["Trait_category"] = merged["Trait"].apply(trait_category_en)
    merged.loc[merged["Trait"].isna(), "Trait_category"] = "Other"

    merged_sorted = merged.sort_values("gain_mean", ascending=False).reset_index(drop=True)
    merged_sorted["global_rank"] = merged_sorted.index + 1

    matched_all = merged[merged["GeneName"].notna()].copy()

    matched_all = matched_all.merge(
        merged_sorted[["evaluated_snp_id", "global_rank"]],
        on="evaluated_snp_id",
        how="left",
    )

    matched_all = matched_all.sort_values(
        ["gain_mean", "evaluated_snp_id"],
        ascending=[False, True],
    )
    before_qtn = len(matched_all)
    matched_all = matched_all.drop_duplicates("GeneName", keep="first")
    matched_all = matched_all.reset_index(drop=True)
    matched_all["global_rank"] = matched_all.index + 1
    print(
        f"Unique QTGs: {len(matched_all)} "
        f"(collapsed {before_qtn - len(matched_all)} lower-SI matches)"
    )

    if args.top_n and args.top_n > 0:
        matched = matched_all.head(args.top_n).copy()
    else:
        matched = matched_all.copy()

    matched.to_csv(f"{args.out_prefix}.publication_panel.tsv", sep="\t", index=False)
    merged.to_csv(f"{args.out_prefix}.all_merged.tsv", sep="\t", index=False)
    all_seed_gain.to_csv(f"{args.out_prefix}.all_seed_gain.tsv", sep="\t", index=False)

    print(f"Total evaluated SNPs: {len(merged)}")
    print(f"Matched RiceNavi QTNs: {len(matched_all)}")
    print(f"Plotted QTNs: {len(matched)}")

    category_colors = {
        "Yield component": "#E64B35",
        "Eating quality": "#4DBBD5",
        "Heading date": "#00A087",
        "Plant architecture": "#3C5488",
        "Secondary metabolism": "#F39B7F",
        "Biotic stress": "#8491B4",
        "Seed morphology": "#91D1C2",
        "Abiotic stress": "#7E6148",
        "Other": "#B09C85",
    }

    genotype_colors = {
        "Base": "#DDEBF7",
        "Evolved": "#DDF1D8",
    }

    matched = matched.iloc[::-1].reset_index(drop=True)
    n = len(matched)
    y = np.arange(n)

    auto_height = max(
        args.min_fig_height,
        min(args.max_fig_height, n * args.row_height + 2.8)
    )

    fig_width = args.fig_width + (4.6 if compare_hhz else 0.0)
    legend_height = 1.25 if compare_hhz else 0.0
    fig = plt.figure(figsize=(fig_width, auto_height + legend_height))

    if compare_hhz:
        width_ratios = [0.75, 3.35, 5.0, 2.15, 3.7]
    else:
        width_ratios = [0.75, 3.35, 5.0, 2.65]
    gs = fig.add_gridspec(
        nrows=2 if compare_hhz else 1,
        ncols=len(width_ratios),
        height_ratios=[auto_height, legend_height] if compare_hhz else None,
        width_ratios=width_ratios,
        hspace=0.03,
        wspace=0.08,
    )

    ax_rank = fig.add_subplot(gs[0, 0])
    ax_gene = fig.add_subplot(gs[0, 1], sharey=ax_rank)
    ax_bar = fig.add_subplot(gs[0, 2], sharey=ax_rank)
    ax_pattern = fig.add_subplot(gs[0, 3], sharey=ax_rank) if compare_hhz else None
    ax_gt = fig.add_subplot(gs[0, 3 if not compare_hhz else 4], sharey=ax_rank)

    for ax in [ax_rank, ax_gene, ax_gt, ax_pattern]:
        if ax is not None:
            ax.axis("off")

    for ax in [ax_rank, ax_gene, ax_bar, ax_gt, ax_pattern]:
        if ax is not None:
            ax.set_ylim(-0.7, n - 0.3)

    ax_rank.text(0.5, n + 0.35, "Rank", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax_gene.text(0.0, n + 0.35, "Gene (SNP_ID)", ha="left", va="bottom", fontsize=12, fontweight="bold")

    if compare_hhz:
        pattern_colors = {
            "HHZ-selected": "#009E9A",
            "Partial-HHZ": "#E84A5F",
            "Novel-selected": "#7A3DD8",
            "Unknown": "#B8B8B8",
        }
        ax_pattern.text(0.5, n + 0.35, "Evolutionary\npattern", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax_gt.text(1.5, n + 0.7, "Genotype", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax_gt.text(0, n - 0.05, "Teqing", ha="center", va="bottom", fontsize=11)
        ax_gt.text(1.5, n - 0.05, "HHZ", ha="center", va="bottom", fontsize=11)
        ax_gt.text(3.0, n - 0.05, "Evolved", ha="center", va="bottom", fontsize=11)
    else:
        pattern_colors = {}
        ax_gt.text(0.75, n + 0.7, "Genotype", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax_gt.text(0, n - 0.05, args.base_label, ha="center", va="bottom", fontsize=10)
        ax_gt.text(1.5, n - 0.05, args.evolved_label, ha="center", va="bottom", fontsize=10)

    ax_rank.plot([0.05, 0.95], [n + 0.18, n + 0.18], color="0.35", lw=0.8, clip_on=False)
    ax_gene.plot([0.0, 1.0], [n + 0.18, n + 0.18], color="0.35", lw=0.8, clip_on=False)

    gain_mean = matched["gain_mean"].fillna(0).to_numpy(dtype=float)
    gain_std = matched["gain_std"].fillna(0).to_numpy(dtype=float)
    right_edge = gain_mean + gain_std
    left_edge = gain_mean - gain_std
    high_edge = float(np.nanmax(right_edge)) if len(right_edge) else 0.001
    low_edge = float(np.nanmin(left_edge)) if len(left_edge) else 0.0
    if not np.isfinite(high_edge) or high_edge <= 0:
        high_edge = 0.001
    x_min = 0.0 if low_edge >= -1e-12 else low_edge * 1.08
    x_max = high_edge * 1.7

    ax_bar.set_xlim(x_min, x_max)
    ax_bar.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_bar.grid(axis="x", linestyle="--", alpha=0.25)

    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.spines["top"].set_visible(True)
    ax_bar.spines["top"].set_linewidth(0.8)
    ax_bar.spines["top"].set_color("0.35")

    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.xaxis.set_ticks_position("top")
    ax_bar.xaxis.set_label_position("top")
    ax_bar.set_xlabel("Mean SI evolved (ΔSI, mean ± SD)", fontsize=12, fontweight="bold", labelpad=8)

    ax_bar.tick_params(
        axis="x",
        which="major",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        direction="out",
        length=3,
        pad=2,
        labelsize=9,
    )

    point_colors = [
        category_colors.get(c, category_colors["Other"])
        for c in matched["Trait_category"]
    ]
    ax_bar.barh(
        y,
        gain_mean,
        xerr=gain_std,
        color=point_colors,
        edgecolor="black",
        linewidth=0.45,
        height=0.58,
        error_kw={"ecolor": "black", "elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
    )

    for i, (_, r) in enumerate(matched.iterrows()):
        rank = int(r["global_rank"])

        gene = str(r["Gene_label"])
        snp_id = str(r["evaluated_snp_id"])

        parts = snp_id.split("-")
        if len(parts) >= 4:
            snp_id_short = "-".join(parts[:-1])
        else:
            snp_id_short = snp_id

        ax_rank.text(
            0.5,
            i,
            str(rank),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold"
        )
        label_size = label_fontsize(f"{gene} ({snp_id_short})")
        gene_box = TextArea(
            gene,
            textprops=dict(
                fontsize=label_size,
                fontweight="bold",
                fontstyle="italic",
                family="Arial"
            )
        )

        snp_box = TextArea(
            f" ({snp_id_short})",
            textprops=dict(
                fontsize=label_size,
                family="Arial"
            )
        )

        packed = HPacker(
            children=[gene_box, snp_box],
            align="center",
            pad=0,
            sep=0
        )

        ab = AnnotationBbox(
            packed,
            (0.0, i),
            xycoords="data",
            box_alignment=(0, 0.5),
            frameon=False
        )

        ax_gene.add_artist(ab)

        center = float(r["gain_mean"])
        std = float(r["gain_std"]) if pd.notna(r["gain_std"]) else 0.0
        gain_label = format_mean_std(center, std)

        ax_bar.text(
            center + std + high_edge * 0.03,
            i,
            gain_label,
            ha="left",
            va="center",
            fontsize=9.5,
        )

        if compare_hhz:
            pattern = str(r["Evolutionary_pattern"])
            rect = Rectangle(
                (0.04, i - 0.29),
                0.92,
                0.58,
                linewidth=0,
                facecolor=pattern_colors.get(pattern, "#B8B8B8"),
                alpha=0.9,
                clip_on=False,
            )
            ax_pattern.add_patch(rect)
            ax_pattern.text(0.5, i, pattern, ha="center", va="center", fontsize=8.2, color="white")
            draw_gt_box(ax_gt, 0, i, gt_display(r["Base_GT"]), "#DDEBF7")
            draw_gt_box(ax_gt, 1.5, i, gt_display(r["HHZ_GT"]), "#F9D9D3")
            draw_gt_box(ax_gt, 3.0, i, gt_display(r["Evolved_Major_GT"]), genotype_colors["Evolved"])
        else:
            draw_gt_box(ax_gt, 0, i, gt_display(r["Base_GT"]), genotype_colors["Base"])
            draw_gt_box(ax_gt, 1.5, i, gt_display(r["Evolved_Major_GT"]), genotype_colors["Evolved"])

    ax_gt.set_xlim(-0.65, 3.65 if compare_hhz else 2.15)
    if compare_hhz:
        ax_pattern.set_xlim(0, 1)
        ax_legend = fig.add_subplot(gs[1, :])
        ax_legend.axis("off")
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.add_patch(
            Rectangle(
                (0.01, 0.04),
                0.98,
                0.92,
                facecolor="white",
                edgecolor="0.75",
                linewidth=0.8,
                transform=ax_legend.transAxes,
            )
        )
        present_trait_categories = [
            name
            for name in category_colors
            if name in set(matched["Trait_category"].dropna().astype(str))
        ]
        ax_legend.text(
            0.025, 0.78, "Trait category",
            fontsize=11, fontweight="bold", transform=ax_legend.transAxes,
        )
        for idx, name in enumerate(present_trait_categories):
            col = idx % 3
            row = idx // 3
            xx = 0.025 + col * 0.165
            yy = 0.42 - row * 0.34
            ax_legend.add_patch(
                Rectangle(
                    (xx, yy), 0.016, 0.20,
                    facecolor=category_colors[name],
                    edgecolor="black",
                    linewidth=0.4,
                    transform=ax_legend.transAxes,
                )
            )
            ax_legend.text(
                xx + 0.022, yy + 0.10, name,
                fontsize=9.5, va="center", transform=ax_legend.transAxes,
            )
        ax_legend.plot(
            [0.54, 0.54], [0.12, 0.88],
            color="0.75", linestyle=":", transform=ax_legend.transAxes,
        )
        ax_legend.text(
            0.57, 0.78, "Evolutionary pattern",
            fontsize=11, fontweight="bold", transform=ax_legend.transAxes,
        )
        pattern_desc = {
            "HHZ-selected": "Evolved genotype identical to HHZ allele state",
            "Novel-selected": "Evolved genotype distinct from both parental allele states",
        }
        for idx, name in enumerate(pattern_desc):
            yy = 0.42 - idx * 0.34
            ax_legend.add_patch(
                Rectangle(
                    (0.57, yy), 0.115, 0.22,
                    facecolor=pattern_colors[name],
                    edgecolor="none",
                    transform=ax_legend.transAxes,
                )
            )
            ax_legend.text(
                0.627, yy + 0.11, name,
                ha="center", va="center", fontsize=7.6, color="white",
                fontweight="bold", transform=ax_legend.transAxes,
            )
            ax_legend.text(
                0.70, yy + 0.11, pattern_desc[name],
                ha="left", va="center", fontsize=7.2, transform=ax_legend.transAxes,
            )

    handles = []
    labels = []

    for cat, color in category_colors.items():
        if cat in set(matched["Trait_category"]):
            handles.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.4))
            labels.append(cat)

    legend_fig = plt.figure(figsize=(8.5, 1.0))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")

    legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=min(5, max(1, len(labels))),
        frameon=False,
        fontsize=9,
        title="Trait category",
        title_fontsize=10,
    )

    legend_pdf_path = f"{args.out_prefix}.trait_category_legend.pdf"
    legend_png_path = f"{args.out_prefix}.trait_category_legend.png"
    legend_svg_path = f"{args.out_prefix}.trait_category_legend.svg"

    legend_fig.savefig(legend_pdf_path, bbox_inches="tight")
    legend_fig.savefig(legend_png_path, dpi=300, bbox_inches="tight")
    legend_fig.savefig(legend_svg_path, bbox_inches="tight")
    plt.close(legend_fig)

    print(f"Saved: {legend_pdf_path}")
    print(f"Saved: {legend_png_path}")
    print(f"Saved: {legend_svg_path}")

    pdf_path = f"{args.out_prefix}.publication_panel.pdf"
    png_path = f"{args.out_prefix}.publication_panel.png"
    svg_path = f"{args.out_prefix}.publication_panel.svg"
    html_path = f"{args.out_prefix}.publication_panel.html"

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    write_html_wrapper(
        svg_path=svg_path,
        html_path=html_path,
        title="QTN SI evolved contribution panel"
    )

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {html_path}")
    print(f"Saved: {args.out_prefix}.publication_panel.tsv")
    print(f"Saved: {args.out_prefix}.all_merged.tsv")
    print(f"Saved: {args.out_prefix}.all_seed_gain.tsv")


if __name__ == "__main__":
    main()