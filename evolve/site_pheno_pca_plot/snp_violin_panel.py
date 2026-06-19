#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

try:
    import pysam
except ImportError:
    raise ImportError("Please install pysam first: conda install -c bioconda pysam")


TRAITS = [
    ("GYP_LingS15", "GYP in LingShui"),
    ("GYP_YangZ15", "GYP in YangZhou"),
    ("Delta_GYP", "ΔGYP (YangZhou - LingShui)"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snps", required=True, help="TSV file with columns: snp_id, gene_name, optimized_GT")
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--pheno", required=True)
    parser.add_argument("--outdir", default="snp_violin_panels")
    parser.add_argument("--id-col", default="LINE")
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--width-per-col", type=float, default=3.5)
    parser.add_argument("--height-per-row", type=float, default=3.0)

    parser.add_argument("--color-ref", default="#e2aeb4")
    parser.add_argument("--color-alt", default="#aec7ad")
    parser.add_argument("--optimized-label-color", default="#A66A00")

    parser.add_argument("--gene-title-size", type=float, default=12)
    parser.add_argument("--trait-title-size", type=float, default=12)

    return parser.parse_args()


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def fmt_pvalue(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-300:
        return "<1e-300"
    return f"{p:.2e}"


def format_gene_mathtext(gene_name):
    gene_name = str(gene_name).strip().replace("_", r"\_")
    return rf"$\bf{{\it{{{gene_name}}}}}$"


def parse_snp_id(snp_id):
    parts = snp_id.strip().split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse SNP ID: {snp_id}")
    return parts[1], int(parts[2])


def normalize_optimized_gt(gt):
    return str(gt).strip().replace("/", "|")


def optimized_gt_to_label(optimized_gt, ref, alt):
    optimized_gt = normalize_optimized_gt(optimized_gt)

    if optimized_gt == "0|0":
        return f"{ref}|{ref}"
    if optimized_gt == "1|1":
        return f"{alt}|{alt}"
    if optimized_gt in ["0|1", "1|0"]:
        return f"{ref}|{alt}"

    return optimized_gt


def read_pheno(path, id_col):
    pheno = pd.read_csv(path, sep=r"\s+", engine="python")

    if id_col not in pheno.columns:
        raise ValueError(f"Cannot find ID column {id_col}. Columns: {pheno.columns.tolist()}")

    for c in ["GYP_LingS15", "GYP_YangZ15"]:
        if c not in pheno.columns:
            raise ValueError(f"Cannot find phenotype column {c}. Columns: {pheno.columns.tolist()}")

    pheno = pheno.rename(columns={id_col: "ID"})
    pheno["ID"] = pheno["ID"].astype(str)

    pheno["GYP_LingS15"] = pd.to_numeric(pheno["GYP_LingS15"], errors="coerce")
    pheno["GYP_YangZ15"] = pd.to_numeric(pheno["GYP_YangZ15"], errors="coerce")
    pheno["Delta_GYP"] = pheno["GYP_YangZ15"] - pheno["GYP_LingS15"]

    return pheno[["ID", "GYP_LingS15", "GYP_YangZ15", "Delta_GYP"]]


def extract_genotype(vcf_path, snp_id):
    chrom, pos = parse_snp_id(snp_id)
    vcf = pysam.VariantFile(vcf_path)

    record = None
    for rec in vcf.fetch(chrom, pos - 1, pos):
        if rec.pos == pos:
            record = rec
            if rec.id == snp_id:
                break

    if record is None:
        raise ValueError(f"Cannot find {snp_id} at {chrom}:{pos}")

    ref = record.ref
    alt = record.alts[0] if record.alts else "N"

    ref_gt = f"{ref}|{ref}"
    alt_gt = f"{alt}|{alt}"
    het_gt = f"{ref}|{alt}"

    rows = []
    for sample in record.samples:
        gt = record.samples[sample].get("GT")

        if gt is None or None in gt:
            label = "NA"
            dosage = np.nan
        else:
            dosage = sum(1 for a in gt if a == 1)

            if dosage == 0:
                label = ref_gt
            elif dosage == 1:
                label = het_gt
            elif dosage == 2:
                label = alt_gt
            else:
                label = "NA"

        rows.append({
            "ID": str(sample),
            "GT": label,
            "Dosage_ALT": dosage,
        })

    return pd.DataFrame(rows), ref, alt, record.contig, record.pos


def calc_pvalue(x, y):
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        _, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        return p
    except Exception:
        return np.nan


def add_gene_title(ax, gene_name, snp_id, ref, alt, optimized_label, fontsize):
    gene_text = format_gene_mathtext(gene_name)
    snp_label = f"{snp_id.rsplit('-', 1)[0]} ({ref}>{alt})"

    ax.text(
        0.5,
        1.18,
        rf"{gene_text}  {snp_label}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#f4f4f4",
            edgecolor="#7f7f7f",
            linewidth=0.8
        )
    )

    ax.text(
        0.5,
        1.05,
        f"* Optimized genotype: {optimized_label}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize * 0.72,
        color="black"
    )


def add_trait_title(ax, label, fontsize):
    ax.text(
        -0.20,
        0.5,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize,
        fontweight="bold"
    )


def plot_violin(ax, df, trait, ref_gt, alt_gt, optimized_label, args):
    ref_values = df.loc[df["GT"] == ref_gt, trait].dropna().values
    alt_values = df.loc[df["GT"] == alt_gt, trait].dropna().values

    data = []
    positions = []
    plot_colors = []

    if len(ref_values) > 0:
        data.append(ref_values)
        positions.append(1)
        plot_colors.append(args.color_ref)

    if len(alt_values) > 0:
        data.append(alt_values)
        positions.append(2)
        plot_colors.append(args.color_alt)

    if len(data) > 0:
        violin_parts = ax.violinplot(
            data,
            positions=positions,
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body, color in zip(violin_parts["bodies"], plot_colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_linewidth(0.6)
            body.set_alpha(1.0)

        ax.boxplot(
            data,
            positions=positions,
            widths=0.25,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=0.8),
            whiskerprops=dict(color="black", linewidth=0.6),
            capprops=dict(color="black", linewidth=0.6),
            boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.6),
        )
    else:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)

    n_ref = len(ref_values)
    n_alt = len(alt_values)
    total_n = n_ref + n_alt

    pval = calc_pvalue(ref_values, alt_values)

    ax.set_xlim(0.45, 2.55)
    ax.set_xticks([1, 2])

    ref_label = f"{ref_gt}*" if ref_gt == optimized_label else ref_gt
    alt_label = f"{alt_gt}*" if alt_gt == optimized_label else alt_gt

    ax.set_xticklabels([ref_label, alt_label], fontsize=9)

    for tick in ax.get_xticklabels():
        if tick.get_text().endswith("*"):
            tick.set_fontweight("bold")
            tick.set_color(args.optimized_label_color)
        else:
            tick.set_fontweight("normal")
            tick.set_color("black")

    ax.set_xlabel("Genotype", fontsize=10, fontweight="bold")
    ax.set_ylabel("")

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - yr * 0.15, ymax + yr * 0.22)

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1

    label_y = ymin + yr * 0.035

    ref_pct = n_ref / total_n * 100 if total_n > 0 else 0
    alt_pct = n_alt / total_n * 100 if total_n > 0 else 0

    ax.text(
        1,
        label_y,
        f"n = {n_ref} ({ref_pct:.0f}%)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=args.optimized_label_color if ref_gt == optimized_label else "black",
        fontweight="bold" if ref_gt == optimized_label else "normal"
    )

    ax.text(
        2,
        label_y,
        f"n = {n_alt} ({alt_pct:.0f}%)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=args.optimized_label_color if alt_gt == optimized_label else "black",
        fontweight="bold" if alt_gt == optimized_label else "normal"
    )

    if pd.notna(pval):
        x1, x2 = 1, 2
        y = ymax - yr * 0.11
        h = yr * 0.035

        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.7)

        x_mid = (x1 + x2) / 2
        y_text = y + h + yr * 0.012

        ax.text(x_mid, y_text, "p = ", ha="right", va="bottom",
                fontsize=8.5, fontweight="bold", fontstyle="italic")
        ax.text(x_mid, y_text, fmt_pvalue(pval), ha="left", va="bottom",
                fontsize=8.5, fontweight="bold")

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.0)

    ax.tick_params(axis="both", width=1.0, length=3.2, labelsize=8.5)
    ax.set_title("")


def plot_one_gene_panel(record, outdir, args):
    snp_id = record["snp_id"]
    gene_name = record["gene_name"]
    optimized_gt = record["optimized_GT"]
    df = record["df"]
    ref = record["ref"]
    alt = record["alt"]

    ref_gt = f"{ref}|{ref}"
    alt_gt = f"{alt}|{alt}"
    optimized_label = optimized_gt_to_label(optimized_gt, ref, alt)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(args.width_per_col, args.height_per_row * 3),
        dpi=args.dpi,
    )

    add_gene_title(axes[0], gene_name, snp_id, ref, alt, optimized_label, args.gene_title_size)

    for r, (trait, label) in enumerate(TRAITS):
        add_trait_title(axes[r], label, args.trait_title_size)
        plot_violin(axes[r], df, trait, ref_gt, alt_gt, optimized_label, args)

    fig.tight_layout()

    prefix = safe_name(snp_id)
    png = os.path.join(outdir, f"{prefix}_transposed_violin.png")
    pdf = os.path.join(outdir, f"{prefix}_transposed_violin.pdf")

    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


def plot_big_panel(records, outdir, args):
    n_genes = len(records)
    n_traits = len(TRAITS)

    if n_genes == 0:
        return

    fig, axes = plt.subplots(
        n_traits,
        n_genes,
        figsize=(args.width_per_col * n_genes, args.height_per_row * n_traits),
        dpi=args.dpi,
        squeeze=False
    )

    for c, rec in enumerate(records):
        snp_id = rec["snp_id"]
        gene_name = rec["gene_name"]
        optimized_gt = rec["optimized_GT"]
        df = rec["df"]
        ref = rec["ref"]
        alt = rec["alt"]

        ref_gt = f"{ref}|{ref}"
        alt_gt = f"{alt}|{alt}"
        optimized_label = optimized_gt_to_label(optimized_gt, ref, alt)

        add_gene_title(axes[0, c], gene_name, snp_id, ref, alt, optimized_label, args.gene_title_size)

        for r, (trait, label) in enumerate(TRAITS):
            if c == 0:
                add_trait_title(axes[r, c], label, args.trait_title_size)

            plot_violin(axes[r, c], df, trait, ref_gt, alt_gt, optimized_label, args)

    fig.tight_layout()

    png = os.path.join(outdir, "all_snps_transposed_violin_panel.png")
    pdf = os.path.join(outdir, "all_snps_transposed_violin_panel.pdf")

    fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    snp_table = pd.read_csv(args.snps, sep="\t")

    required = {"snp_id", "gene_name", "optimized_GT"}
    missing = required - set(snp_table.columns)
    if missing:
        raise ValueError(f"Input file must have columns: snp_id, gene_name, optimized_GT. Missing: {missing}")

    pheno = read_pheno(args.pheno, args.id_col)

    records = []
    summary_rows = []

    for _, row in snp_table.iterrows():
        snp_id = str(row["snp_id"]).strip()
        gene_name = str(row["gene_name"]).strip()
        optimized_gt = normalize_optimized_gt(row["optimized_GT"])

        print(f"\n===== Processing {snp_id} ({gene_name}) =====")

        try:
            gt, ref, alt, chrom, pos = extract_genotype(args.vcf, snp_id)
        except Exception as e:
            print(f"[WARN] Skip {snp_id}: {e}")
            continue

        optimized_label = optimized_gt_to_label(optimized_gt, ref, alt)

        print(f"  VCF site: {chrom}:{pos} {ref}>{alt}")
        print(f"  Optimized genotype: {optimized_gt} -> {optimized_label}")

        df = gt.merge(pheno, on="ID", how="inner")
        df = df.dropna(subset=["GT"]).copy()

        df.to_csv(os.path.join(args.outdir, f"{safe_name(snp_id)}_merged.tsv"), sep="\t", index=False)

        record = {
            "snp_id": snp_id,
            "gene_name": gene_name,
            "optimized_GT": optimized_gt,
            "df": df,
            "ref": ref,
            "alt": alt,
        }

        records.append(record)

        png, pdf = plot_one_gene_panel(record, args.outdir, args)
        print(f"  Saved: {png}")
        print(f"  Saved: {pdf}")

        for gt_name, sub in df.groupby("GT"):
            if gt_name == "NA":
                continue

            summary_rows.append({
                "SNP": snp_id,
                "Gene": gene_name,
                "CHROM": chrom,
                "POS": pos,
                "REF": ref,
                "ALT": alt,
                "GT": gt_name,
                "Optimized_GT_code": optimized_gt,
                "Optimized_GT_label": optimized_label,
                "is_optimized": gt_name == optimized_label,
                "n": len(sub),
                "GYP_LingS15_mean": sub["GYP_LingS15"].mean(),
                "GYP_YangZ15_mean": sub["GYP_YangZ15"].mean(),
                "Delta_GYP_mean": sub["Delta_GYP"].mean(),
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(args.outdir, "summary_by_snp_genotype.tsv"),
            sep="\t",
            index=False
        )

    plot_big_panel(records, args.outdir, args)

    print(f"\nAll done. Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()