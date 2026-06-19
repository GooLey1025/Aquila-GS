#!/usr/bin/env python3
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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


def wrap_trait_label(trait_en):
    words = str(trait_en).split()
    if len(words) <= 1:
        return trait_en
    return "\n".join(words)


def is_hhz_like(row):
    same_change = str(row.get("Same_Change_As_HHZ", "")).strip()
    major_same = str(row.get("Major_GT_Same_As_HHZ", "")).strip()

    if same_change == "Yes":
        return True

    if major_same == "Yes":
        return True

    return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--out-prefix", default="Teqing_1000seed_evolved_sites")
    parser.add_argument(
        "--metric",
        default="Selection_Strength",
        choices=[
            "CEF",
            "Major_GT_Freq",
            "Evolved_AF",
            "Delta_AF",
            "Selection_Strength",
            "Freq_Same_As_HHZ"
        ]
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-per-row", type=int, default=10)
    parser.add_argument("--title", default=None)
    parser.add_argument("--bar-width", type=float, default=0.72)
    parser.add_argument("--show-hhz-text", action="store_true")

    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t").copy()

    df.columns = (
        df.columns
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    numeric_cols = [
        "CEF",
        "Major_GT_Freq",
        "Evolved_AF",
        "Delta_AF",
        "Freq_Same_As_HHZ",
        "Selection_Strength"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({
                    "": np.nan,
                    "NA": np.nan,
                    "NaN": np.nan,
                    "nan": np.nan,
                    "None": np.nan,
                })
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Selection_Strength" not in df.columns or df["Selection_Strength"].isna().all():
        if "Delta_AF" in df.columns and "Major_GT_Freq" in df.columns:
            df["Selection_Strength"] = df["Delta_AF"].abs() * df["Major_GT_Freq"]

    df["Trait_EN"] = df["Trait"].apply(trait_category_en)
    df["Trait_Label"] = df["Trait_EN"].apply(wrap_trait_label)
    df["GeneName"] = df["GeneName"].fillna("NA").astype(str)

    df["HHZ_like"] = df.apply(is_hhz_like, axis=1)

    df = df.dropna(subset=[args.metric]).copy()

    if df.empty:
        available = [
            c for c in [
                "CEF",
                "Delta_AF",
                "Major_GT_Freq",
                "Evolved_AF",
                "Selection_Strength",
                "Freq_Same_As_HHZ"
            ]
            if c in df.columns and df[c].notna().any()
        ]

        raw_df = pd.read_csv(args.input, sep="\t")

        raise SystemExit(
            f"No non-null values for metric '{args.metric}' in input file.\n"
            f"  Total rows: {len(raw_df)}\n"
            f"  Available non-null metrics: {available or 'none'}"
        )

    df = df.sort_values(args.metric, ascending=False).head(args.top_n).reset_index(drop=True)

    df["Rank"] = df[args.metric].rank(method="min", ascending=False).astype(int)

    color_map = {
        "Yield component": "#ef3b2c",
        "Eating quality": "#41b6c4",
        "Plant architecture": "#253494",
        "Biotic stress": "#8c96c6",
        "Abiotic stress": "#7fc97f",
        "Heading date": "#fdb462",
        "Secondary metabolism": "#b3de69",
        "Seed morphology": "#fccde5",
        "Other": "#b8a58f",
    }

    n = len(df)
    max_per_row = args.max_per_row
    n_rows = math.ceil(n / max_per_row)

    fig_width = max(12, max_per_row * 1.15)
    fig_height = 4.6 * n_rows + 1.4

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    axes = axes.flatten()

    global_ymin = df[args.metric].min()
    global_ymax = df[args.metric].max()
    global_range = global_ymax - global_ymin

    if args.metric == "Delta_AF":
        y_lower = global_ymin - (global_range * 0.25 if global_range > 0 else 0.05)
        y_upper = global_ymax + (global_range * 0.35 if global_range > 0 else 0.05)
    else:
        if global_range == 0:
            y_lower = max(0, global_ymin - 0.05)
            y_upper = global_ymax + 0.10
        else:
            y_lower = max(0, global_ymin - global_range * 0.35)
            y_upper = global_ymax + global_range * 0.65

    for row_idx in range(n_rows):
        ax = axes[row_idx]

        start = row_idx * max_per_row
        end = min(start + max_per_row, n)
        sub = df.iloc[start:end].copy()

        x = np.arange(len(sub))
        y = sub[args.metric].astype(float)

        colors = [color_map.get(x, "#b8a58f") for x in sub["Trait_EN"]]

        labels = []
        for _, r in sub.iterrows():
            gene = str(r.GeneName)
            if bool(r.HHZ_like):
                gene = gene + "*"
            labels.append(f"{gene}\n({r.Trait_Label})")

        bars = ax.bar(
            x,
            y,
            width=args.bar_width,
            color=colors,
            edgecolor=[
                "black" if not bool(v) else "#111111"
                for v in sub["HHZ_like"]
            ],
            linewidth=[
                0.6 if not bool(v) else 2.2
                for v in sub["HHZ_like"]
            ]
        )

        for bar, (_, r) in zip(bars, sub.iterrows()):
            val = float(r[args.metric])

            offset = (
                global_range * 0.035
                if global_range > 0
                else max(abs(val) * 0.035, 0.01)
            )

            rank_text = f"#{int(r.Rank)}\n{val:.4f}"

            if args.show_hhz_text and bool(r.HHZ_like):
                rank_text += "\nHHZ-like"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                rank_text,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if bool(r.HHZ_like) else "normal"
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)

        ax.set_ylabel(args.metric.replace("_", " "))
        ax.set_ylim(y_lower, y_upper)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        if args.metric == "Delta_AF":
            ax.axhline(0, color="black", linewidth=0.8)

    if args.title:
        title = args.title
    else:
        title = f"Top {args.top_n} RiceNavi QTNs by {args.metric.replace('_', ' ')}"

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)

    used_traits = set(df["Trait_EN"])

    legend_items = [
        Patch(facecolor=color, edgecolor="black", label=trait)
        for trait, color in color_map.items()
        if trait in used_traits
    ]

    if df["HHZ_like"].any():
        legend_items.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=2.2,
                label="HHZ-like change (*)"
            )
        )

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=min(5, len(legend_items)),
        frameon=False,
        fontsize=8
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])

    out_png = f"{args.out_prefix}.{args.metric}.top{args.top_n}.HHZ_trait_bar.png"
    out_pdf = f"{args.out_prefix}.{args.metric}.top{args.top_n}.HHZ_trait_bar.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Saved: {out_png}")
    print(f"[DONE] Saved: {out_pdf}")

    print("[INFO] HHZ-like sites in plotted data:")
    print(df[df["HHZ_like"]][[
        "SNP_ID",
        "GeneName",
        "Trait",
        "Parent_GT",
        "HHZ_GT",
        "Major_GT",
        "Same_Change_As_HHZ",
        "Freq_Same_As_HHZ",
        args.metric
    ]].to_string(index=False))


if __name__ == "__main__":
    main()