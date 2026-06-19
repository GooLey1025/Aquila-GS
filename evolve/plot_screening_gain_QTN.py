#!/usr/bin/env python3
import argparse
import subprocess
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publication-style RiceNavi QTN SI gain panel with evolutionary pattern and genotype matrix."
    )

    parser.add_argument("--screening-file", required=True)
    parser.add_argument("--ricenavi-file", required=True)

    parser.add_argument("--teqing-vcf", required=True)
    parser.add_argument("--hhz-vcf", required=True)
    parser.add_argument("--evolved-vcf", required=True)

    parser.add_argument("--out-prefix", default="QTN_gain_contribution")
    parser.add_argument("--top-n", type=int, default=10)

    parser.add_argument("--gain-col", default="gain")
    parser.add_argument("--skiprows", type=int, default=1)

    parser.add_argument("--fig-width", type=float, default=14.5)
    parser.add_argument("--fig-height", type=float, default=10.5)

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


def gt_bases(gt, vcf_ref, vcf_alt):
    """
    Convert normalized GT (e.g. "0|1", "1|2") to actual nucleotide bases.

    Allele index mapping in VCF:
        0 -> REF
        1 -> ALT[0]
        2 -> ALT[1]
        3 -> ALT[2]  ...etc

    Examples:
        gt="0|1", ref="G", alt=["C","T"]  -> "G/C"
        gt="1|1", ref="G", alt=["C","T"]  -> "C/C"
        gt="0|2", ref="G", alt=["C","T"]  -> "G/T"
        gt="2|2", ref="G", alt=["C","T"]  -> "T/T"
        gt=".|.", ref="G", alt=["C"]      -> "NA"
    """
    gt = normalize_gt(gt)
    if gt == ".|.":
        return "NA"

    alleles = gt.split("|")
    if len(alleles) != 2:
        return "NA"

    result = []
    for idx_str in alleles:
        try:
            idx = int(idx_str)
        except ValueError:
            return "NA"

        if idx == 0:
            result.append(vcf_ref if vcf_ref else "?")
        else:
            alt_list = vcf_alt if vcf_alt else []
            if idx - 1 < len(alt_list):
                result.append(alt_list[idx - 1])
            else:
                result.append("?")

    return "/".join(result)


def alt_dosage(gt):
    gt = normalize_gt(gt)
    if gt == ".|.":
        return None
    try:
        return sum(int(a) for a in gt.split("|"))
    except ValueError:
        return None


def parse_alleles(alleles_str):
    """
    Split a comma-delimited allele string into individual alleles.

    Examples:
        "C,T"     -> ["C", "T"]
        "A"       -> ["A"]
        "C,T,G"   -> ["C", "T", "G"]
        "" / "."  -> []
    """
    s = str(alleles_str).strip()
    if s.lower() in ("nan", "", "."):
        return []
    return [a.strip().upper() for a in s.split(",") if a.strip()]


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


def load_single_sample_gt(vcf, site_ids, label):
    site_set = set(site_ids)
    # Query returns: ID, REF, ALT, [GT] for each sample
    cmd = f"bcftools query -f '%ID\\t%REF\\t%ALT\\t[%GT]\\n' {vcf}"
    out = run_cmd(cmd)

    gt_dict = {}
    alleles_dict = {}

    for line in out.strip().splitlines():
        arr = line.rstrip("\n").split("\t")
        if len(arr) < 4:
            continue

        snp_id = clean_text(arr[0])
        if snp_id not in site_set:
            continue

        gt_dict[snp_id] = normalize_gt(arr[3])
        alleles_dict[snp_id] = {
            "ref": clean_text(arr[1]).upper(),
            "alt": parse_alleles(arr[2]),
        }

    print(f"[INFO] {label} matched GT sites: {len(gt_dict)} / {len(site_ids)}")
    return gt_dict, alleles_dict


def load_evolved_major_gt(vcf, site_ids):
    site_set = set(site_ids)

    sample_out = run_cmd(f"bcftools query -l {vcf}")
    samples = [x for x in sample_out.strip().splitlines() if x.strip()]
    print(f"[INFO] Evolved VCF samples: {len(samples)}")

    cmd = f"bcftools query -f '%ID\\t%REF\\t%ALT[\\t%GT]\\n' {vcf}"
    out = run_cmd(cmd)

    major_dict = {}
    freq_dict = {}
    count_dict = {}
    alleles_dict = {}

    for line in out.strip().splitlines():
        arr = line.rstrip("\n").split("\t")
        if len(arr) < 4:
            continue

        snp_id = clean_text(arr[0])
        if snp_id not in site_set:
            continue

        alleles_dict[snp_id] = {
            "ref": clean_text(arr[1]).upper(),
            "alt": parse_alleles(arr[2]),
        }

        gts = [normalize_gt(x) for x in arr[3:]]
        counts = Counter(gts)

        n_missing = counts.get(".|.", 0)
        n_valid = len(gts) - n_missing

        valid_counts = {
            "0|0": counts.get("0|0", 0),
            "0|1": counts.get("0|1", 0),
            "1|0": counts.get("1|0", 0),
            "1|1": counts.get("1|1", 0),
        }

        if n_valid > 0:
            major_gt = max(valid_counts, key=valid_counts.get)
            major_n = valid_counts[major_gt]
            major_freq = major_n / n_valid
        else:
            major_gt = ".|."
            major_freq = np.nan

        major_dict[snp_id] = major_gt
        freq_dict[snp_id] = major_freq
        count_dict[snp_id] = {
            "N_valid": n_valid,
            "N_0|0": valid_counts["0|0"],
            "N_0|1": valid_counts["0|1"],
            "N_1|0": valid_counts["1|0"],
            "N_1|1": valid_counts["1|1"],
            "N_missing": n_missing,
        }

    print(f"[INFO] Evolved matched GT sites: {len(major_dict)} / {len(site_ids)}")
    return major_dict, freq_dict, count_dict, alleles_dict


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


def draw_rect_label(ax, x, y, text, color, width=0.95, height=0.50, fontsize=8.2):
    rect = Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        linewidth=0,
        facecolor=color,
        alpha=0.95,
        clip_on=False,
    )
    ax.add_patch(rect)

    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=fontsize,
        fontweight="bold",
        clip_on=False,
    )


def draw_gt_box(ax, x, y, text, facecolor, width=0.72, height=0.58):
    rect = Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        linewidth=0,
        facecolor=facecolor,
        alpha=0.85,
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


def main():
    args = parse_args()

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

    screen = pd.read_csv(
        args.screening_file,
        sep="\t",
        skiprows=args.skiprows,
        header=None,
        names=screening_cols,
    )

    screen["evaluated_snp_id"] = screen["evaluated_snp_id"].astype(str).str.strip()
    screen[["Chr", "Pos"]] = screen["evaluated_snp_id"].apply(parse_snp_id)

    if args.gain_col not in screen.columns:
        raise ValueError(
            f"--gain-col '{args.gain_col}' not found. "
            f"Available columns: {screen.columns.tolist()}"
        )

    screen["gain_plot"] = pd.to_numeric(
        screen[args.gain_col],
        errors="coerce"
    ).fillna(0)

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

    # Extract and normalize REF/ALT alleles from RiceNavi for validation
    has_ref_alt = "Ref_geno" in ricenavi.columns and "Alt_geno" in ricenavi.columns
    if has_ref_alt:
        ricenavi["RN_REF"] = (
            ricenavi["Ref_geno"].astype(str).str.strip().str.upper()
        )
        ricenavi["RN_ALT"] = ricenavi["Alt_geno"].apply(parse_alleles)
    else:
        print("[WARN] Ref_geno/Alt_geno columns not found in RiceNavi file — skipping allele validation")
        ricenavi["RN_REF"] = ""
        ricenavi["RN_ALT"] = [[] for _ in range(len(ricenavi))]

    # Load genotype + allele data from VCFs
    teqing_gt, teqing_alleles = load_single_sample_gt(
        args.teqing_vcf, site_ids := ricenavi["SNP_ID"].unique().tolist(), "Teqing"
    )
    hhz_gt, hhz_alleles = load_single_sample_gt(
        args.hhz_vcf, site_ids, "Huanghuazhan"
    )
    evolved_major_gt, evolved_major_freq, evolved_counts, evolved_alleles = (
        load_evolved_major_gt(args.evolved_vcf, site_ids)
    )

    if has_ref_alt:
        skipped = 0
        for idx, row in ricenavi.iterrows():
            sid = row["SNP_ID"]
            rn_ref = row["RN_REF"]
            rn_alt = row["RN_ALT"]

            vcf_alleles = teqing_alleles.get(sid)
            if vcf_alleles is None:
                vcf_alleles = hhz_alleles.get(sid)

            if vcf_alleles is None:
                skipped += 1
                ricenavi.at[idx, "SNP_ID"] = pd.NA
                continue

            vcf_ref = vcf_alleles["ref"]
            vcf_alt = vcf_alleles["alt"]

            if vcf_ref != rn_ref:
                print(
                    f"[WARN] {sid}: VCF REF '{vcf_ref}' != RiceNavi REF '{rn_ref}' — skipped"
                )
                ricenavi.at[idx, "SNP_ID"] = pd.NA
                skipped += 1
                continue

            rn_alt_set = set(rn_alt)
            vcf_alt_set = set(vcf_alt)

            if rn_alt_set and not rn_alt_set.issubset(vcf_alt_set):
                print(
                    f"[WARN] {sid}: RiceNavi ALT {rn_alt} not fully found in VCF ALT {vcf_alt} — skipped"
                )
                ricenavi.at[idx, "SNP_ID"] = pd.NA
                skipped += 1
                continue

        if skipped:
            print(f"[INFO] Skipped {skipped} RiceNavi records due to allele mismatch")

    ricenavi = ricenavi[ricenavi["SNP_ID"].notna()].copy()

    merged = screen.merge(
        ricenavi,
        left_on="evaluated_snp_id",
        right_on="SNP_ID",
        how="left",
        suffixes=("", "_ricenavi"),
    )

    matched_all = merged[merged["GeneName"].notna()].copy()

    if len(matched_all) == 0:
        print("[DEBUG] First 5 screening IDs:")
        print(screen["evaluated_snp_id"].head().to_string(index=False))
        print("[DEBUG] First 5 RiceNavi SNP_ID:")
        print(ricenavi["SNP_ID"].head().to_string(index=False))
        raise ValueError("No matched RiceNavi QTNs found.")

    site_ids = matched_all["evaluated_snp_id"].dropna().astype(str).unique().tolist()

    merged["Teqing_GT"] = merged["evaluated_snp_id"].map(teqing_gt)
    merged["HHZ_GT"] = merged["evaluated_snp_id"].map(hhz_gt)
    merged["Evolved_Major_GT"] = merged["evaluated_snp_id"].map(evolved_major_gt)
    merged["Evolved_Major_GT_Freq"] = merged["evaluated_snp_id"].map(evolved_major_freq)

    # Store VCF allele info alongside merged data
    merged["VCF_REF"] = merged["evaluated_snp_id"].map(
        {sid: a["ref"] for sid, a in teqing_alleles.items()}
    ).fillna(
        merged["evaluated_snp_id"].map({sid: a["ref"] for sid, a in hhz_alleles.items()})
    )
    merged["VCF_ALT"] = merged["evaluated_snp_id"].map(
        {sid: a["alt"] for sid, a in teqing_alleles.items()}
    ).fillna(
        merged["evaluated_snp_id"].map({sid: a["alt"] for sid, a in hhz_alleles.items()})
    )

    merged["Evolutionary_pattern"] = merged.apply(
        lambda r: classify_hhz_selection(
            r.get("Teqing_GT", ".|."),
            r.get("HHZ_GT", ".|."),
            r.get("Evolved_Major_GT", ".|."),
        ),
        axis=1,
    )

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

    merged_sorted = merged.sort_values("gain_plot", ascending=False).reset_index(drop=True)
    merged_sorted["global_rank"] = merged_sorted.index + 1

    matched_all = merged[merged["GeneName"].notna()].copy()

    matched_all = matched_all.merge(
        merged_sorted[["evaluated_snp_id", "global_rank"]],
        on="evaluated_snp_id",
        how="left",
    )

    matched = (
        matched_all
        .sort_values("gain_plot", ascending=False)
        .head(args.top_n)
        .reset_index(drop=True)
    )

    matched.to_csv(f"{args.out_prefix}.publication_panel.tsv", sep="\t", index=False)
    merged.to_csv(f"{args.out_prefix}.all_merged.tsv", sep="\t", index=False)

    print(f"Total evaluated SNPs: {len(merged)}")
    print(f"Matched RiceNavi QTNs: {len(matched_all)}")
    print(f"Plotted top QTNs: {len(matched)}")
    print("[INFO] Trait categories in plotted top-N:")
    print(matched["Trait_category"].value_counts(dropna=False).to_string())
    print("[INFO] Evolutionary pattern:")
    print(matched["Evolutionary_pattern"].value_counts(dropna=False).to_string())

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

    pattern_colors = {
        "HHZ-selected": "#009E9A",
        "Partial-HHZ": "#E84A5F",
        "Novel-selected": "#7A3DD8",
        "Unknown": "#B8B8B8",
    }

    genotype_colors = {
        "Teqing": "#DDEBF7",
        "HHZ": "#F9D9D3",
        "Evolved": "#DDF1D8",
    }

    matched = matched.iloc[::-1].reset_index(drop=True)
    n = len(matched)
    y = np.arange(n)

    fig = plt.figure(figsize=(args.fig_width, args.fig_height))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=5,
        height_ratios=[18, 2.4],
        width_ratios=[0.85, 3.15, 4.35, 2.05, 3.45],
        hspace=0.005,
        wspace=0.08,
    )

    ax_rank = fig.add_subplot(gs[0, 0])
    ax_gene = fig.add_subplot(gs[0, 1], sharey=ax_rank)
    ax_bar = fig.add_subplot(gs[0, 2], sharey=ax_rank)
    ax_pattern = fig.add_subplot(gs[0, 3], sharey=ax_rank)
    ax_gt = fig.add_subplot(gs[0, 4], sharey=ax_rank)
    ax_legend = fig.add_subplot(gs[1, 0:5])

    for ax in [ax_rank, ax_gene, ax_pattern, ax_gt, ax_legend]:
        ax.axis("off")

    for ax in [ax_rank, ax_gene, ax_bar, ax_pattern, ax_gt]:
        ax.set_ylim(-0.7, n - 0.3)

    ax_rank.text(
        0.5, n + 0.35, "Rank",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold"
    )
    ax_gene.text(
        0.0, n + 0.35, "Gene (SNP_ID)",
        ha="left", va="bottom",
        fontsize=11, fontweight="bold"
    )
    ax_pattern.text(
        0.5, n + 0.35, "Evolutionary\npattern",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold"
    )

    ax_gt.text(
        1.5, n + 0.7, "Genotype",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold"
    )
    ax_gt.text(0, n + 0.15, "Teqing", ha="center", va="bottom", fontsize=10)
    ax_gt.text(1.5, n + 0.15, "HHZ", ha="center", va="bottom", fontsize=10)
    ax_gt.text(3.0, n + 0.15, "Evolved", ha="center", va="bottom", fontsize=10)

    ax_rank.plot([0.05, 0.95], [n + 0.18, n + 0.18], color="0.35", lw=0.8, clip_on=False)
    ax_gene.plot([0.0, 1.0], [n + 0.18, n + 0.18], color="0.35", lw=0.8, clip_on=False)
    ax_gt.plot([-0.55, 3.55], [n + 0.18, n + 0.18], color="0.35", lw=0.8, clip_on=False)

    max_gain = matched["gain_plot"].max()
    if pd.isna(max_gain) or max_gain <= 0:
        max_gain = 0.001

    x_max = max_gain * 1.22

    ax_bar.set_xlim(0, x_max)
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
    ax_bar.set_xlabel("SI gain (ΔSI)", fontsize=11, fontweight="bold", labelpad=8)

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

    ax_bar.barh(
        y,
        matched["gain_plot"],
        color=[
            category_colors.get(c, category_colors["Other"])
            for c in matched["Trait_category"]
        ],
        edgecolor="black",
        linewidth=0.45,
        height=0.58,
    )

    for i, (_, r) in enumerate(matched.iterrows()):
        rank = int(r["global_rank"])

        gene = str(r["Gene_label"])
        snp_id = str(r["evaluated_snp_id"])

        parts = snp_id.split("-")
        if len(parts) >= 4:
            snp_id = "-".join(parts[:-1])

        pattern = str(r["Evolutionary_pattern"])

        ax_rank.text(
            0.5,
            i,
            str(rank),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

        ax_gene.text(
            0.0,
            i,
            rf"$\it{{{gene}}}$ ({snp_id})",
            ha="left",
            va="center",
            fontsize=8.6,
            fontweight="bold"
        )

        gain = float(r["gain_plot"])

        ax_bar.text(
            gain + max_gain * 0.025,
            i,
            f"{gain:.3f}",
            ha="left",
            va="center",
            fontsize=9,
        )

        draw_rect_label(
            ax_pattern,
            0.5,
            i,
            pattern,
            pattern_colors.get(pattern, "#B8B8B8"),
            width=0.95,
            height=0.50,
            fontsize=7.0 if len(pattern) > 9 else 8.2,
        )

        draw_gt_box(
            ax_gt,
            0,
            i,
            gt_bases(r["Teqing_GT"], r.get("VCF_REF"), r.get("VCF_ALT")),
            genotype_colors["Teqing"]
        )

        draw_gt_box(
            ax_gt,
            1.5,
            i,
            gt_bases(r["HHZ_GT"], r.get("VCF_REF"), r.get("VCF_ALT")),
            genotype_colors["HHZ"]
        )

        draw_gt_box(
            ax_gt,
            3.0,
            i,
            gt_bases(r["Evolved_Major_GT"], r.get("VCF_REF"), r.get("VCF_ALT")),
            genotype_colors["Evolved"]
        )

    ax_pattern.set_xlim(0, 1)
    ax_gt.set_xlim(-0.65, 3.65)

    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    legend_box = Rectangle(
        (0.02, 0.08),
        0.96,
        0.82,
        facecolor="white",
        edgecolor="0.75",
        linewidth=0.8,
        transform=ax_legend.transAxes,
    )
    ax_legend.add_patch(legend_box)

    present_trait_categories = [
        c for c in category_colors.keys()
        if c in set(matched["Trait_category"].dropna().astype(str))
    ]

    trait_items = [
        (name, category_colors.get(name, category_colors["Other"]))
        for name in present_trait_categories
    ]

    ax_legend.text(
        0.04,
        0.75,
        "Trait category",
        fontsize=10,
        fontweight="bold",
        transform=ax_legend.transAxes,
    )

    x0 = 0.04
    y0 = 0.47
    dx = 0.13

    for idx, (name, color) in enumerate(trait_items):
        col = idx % 4
        row = idx // 4
        xx = x0 + col * dx
        yy = y0 - row * 0.25

        ax_legend.add_patch(
            Rectangle(
                (xx, yy),
                0.018,
                0.08,
                facecolor=color,
                edgecolor="black",
                linewidth=0.4,
                transform=ax_legend.transAxes,
            )
        )
        ax_legend.text(
            xx + 0.025,
            yy + 0.04,
            name,
            fontsize=8.3,
            va="center",
            transform=ax_legend.transAxes,
        )

    ax_legend.plot(
        [0.57, 0.57],
        [0.18, 0.82],
        color="0.75",
        linestyle=":",
        transform=ax_legend.transAxes,
    )

    ax_legend.text(
        0.60,
        0.75,
        "Evolutionary pattern",
        fontsize=10,
        fontweight="bold",
        transform=ax_legend.transAxes,
    )

    present_patterns = [
        p for p in ["HHZ-selected", "Partial-HHZ", "Novel-selected", "Unknown"]
        if p in set(matched["Evolutionary_pattern"].dropna().astype(str))
    ]

    pattern_desc = {
        "HHZ-selected": "Evolved genotype identical to HHZ allele state",
        "Partial-HHZ": "Evolved genotype shifted toward the HHZ allele dosage",
        "Novel-selected": "Evolved genotype distinct from both parental allele states",
        "Unknown": "Missing or unchanged diagnostic genotype state",
    }

    pattern_items = [
        (p, pattern_desc[p], pattern_colors[p])
        for p in present_patterns
    ]

    for idx, (name, desc, color) in enumerate(pattern_items):
        yy = 0.58 - idx * 0.18

        ax_legend.add_patch(
            Rectangle(
                (0.60, yy),
                0.105,
                0.105,
                facecolor=color,
                edgecolor="none",
                transform=ax_legend.transAxes,
            )
        )
        ax_legend.text(
            0.6525,
            yy + 0.042,
            name,
            ha="center",
            va="center",
            fontsize=6.0,
            color="white",
            fontweight="bold",
            transform=ax_legend.transAxes,
        )
        ax_legend.text(
            0.72,
            yy + 0.042,
            desc,
            ha="left",
            va="center",
            fontsize=7.0,
            transform=ax_legend.transAxes,
        )

    plt.savefig(f"{args.out_prefix}.publication_panel.pdf", bbox_inches="tight")
    plt.savefig(f"{args.out_prefix}.publication_panel.png", dpi=300, bbox_inches="tight")

    print(f"Saved: {args.out_prefix}.publication_panel.pdf")
    print(f"Saved: {args.out_prefix}.publication_panel.png")
    print(f"Saved: {args.out_prefix}.publication_panel.tsv")
    print(f"Saved: {args.out_prefix}.all_merged.tsv")


if __name__ == "__main__":
    main()