#!/usr/bin/env python3
import os
import glob
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot round-wise phenotype predictions across all seed directories."
    )

    parser.add_argument(
        "--evolve-dir",
        default="1171rice_SI_screening_evolve",
        help="Directory containing seed_*/round_predictions.tsv"
    )

    parser.add_argument(
        "--out",
        default="all_traits_round_predictions",
        help="Output prefix"
    )

    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument(
        "--max-per-col",
        type=int,
        default=3,
        help="Maximum traits per column"
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=7
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.08
    )

    parser.add_argument(
        "--fig-width-per-col",
        type=float,
        default=4.0
    )

    parser.add_argument(
        "--fig-height-per-row",
        type=float,
        default=3.0
    )

    return parser.parse_args()


def seed_sort_key(path):
    seed_name = os.path.basename(os.path.dirname(path))
    try:
        return int(seed_name.replace("seed_", ""))
    except ValueError:
        return seed_name


def short_trait_name(trait):
    mapping = {
        "Heading_date": "Heading date",
        "Plant_height": "Plant height",
        "Leaf_length": "Leaf length",
        "Leaf_width": "Leaf width",
        "Tiller_angle": "Tiller angle",
        "Full_grain_number_per_plant": "Full grain number per plant",
        "Valid_panicle_number": "Valid panicle number",
        "Full_grain_number_per_panicle": "Full grain number per panicle",
        "Kilo-grain_weight": "1000-grain weight",
        "Seed_setting_rate": "Seed setting rate",
        "Yield_per_plant": "Yield per plant",
        "Amylose_content": "Amylose content",
        "Gel_consistency": "Gel consistency",
        "Chalkiness": "Chalkiness",
        "Chalky_grain_percentage": "Chalky grain percentage",
        "Grain_shape(Length_to_width_ratio_of_polished_grain)": "Length/width ratio",
        "Grain_translucency_level": "Grain translucency level",
    }
    return mapping.get(trait, trait.replace("_", " "))


def main():
    args = parse_args()

    files = sorted(
        glob.glob(os.path.join(args.evolve_dir, "seed_*", "round_predictions.tsv")),
        key=seed_sort_key
    )

    if not files:
        raise FileNotFoundError(
            f"No round_predictions.tsv found under: {args.evolve_dir}/seed_*"
        )

    all_df = []

    for f in files:
        seed = os.path.basename(os.path.dirname(f)).replace("seed_", "")

        df = pd.read_csv(f, sep=r"\s+", engine="python")
        df.columns = df.columns.str.strip()

        if "Round" not in df.columns:
            raise ValueError(f"Column 'Round' not found in {f}")

        df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
        df = df.dropna(subset=["Round"]).copy()
        df["Round"] = df["Round"].astype(int)

        df["Seed"] = seed
        all_df.append(df)

    data = pd.concat(all_df, ignore_index=True)

    pred_cols = [
        c for c in data.columns
        if c.endswith("_pred")
    ]

    if not pred_cols:
        raise ValueError(
            "No *_pred columns found. Please check file delimiter or column names."
        )

    preferred_order = [
        "Heading_date_pred",
        "Plant_height_pred",
        "Amylose_content_pred",
        "Gel_consistency_pred",
        "Chalkiness_pred",
        "Grain_shape(Length_to_width_ratio_of_polished_grain)_pred",
        "Grain_translucency_level_pred",
        "Leaf_length_pred",
        "Leaf_width_pred",
        "Tiller_angle_pred",
        "Full_grain_number_per_plant_pred",
        "Valid_panicle_number_pred",
        "Full_grain_number_per_panicle_pred",
        "Kilo-grain_weight_pred",
        "Seed_setting_rate_pred",
        "Yield_per_plant_pred",
        "Chalky_grain_percentage_pred",
    ]

    pred_cols = [c for c in preferred_order if c in pred_cols] + [
        c for c in pred_cols if c not in preferred_order
    ]

    n_traits = len(pred_cols)
    n_rows = args.max_per_col
    n_cols = math.ceil(n_traits / n_rows)

    fig_width = args.fig_width_per_col * n_cols
    fig_height = args.fig_height_per_row * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    round_min = int(data["Round"].min())
    round_max = int(data["Round"].max())

    for i, pred_col in enumerate(pred_cols):
        row = i % n_rows
        col = i // n_rows
        ax = axes[row, col]

        trait = pred_col[:-5]
        trait_label = short_trait_name(trait)

        plot_df = data[["Round", pred_col]].copy()
        plot_df[pred_col] = pd.to_numeric(plot_df[pred_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[pred_col])

        ax.scatter(
            plot_df["Round"],
            plot_df[pred_col],
            s=args.point_size,
            alpha=args.alpha,
            color="black",
            linewidths=0
        )

        first_mean = plot_df.loc[
            plot_df["Round"] == round_min,
            pred_col
        ].mean()

        last_mean = plot_df.loc[
            plot_df["Round"] == round_max,
            pred_col
        ].mean()

        if pd.notna(first_mean):
            ax.scatter(
                round_min,
                first_mean,
                s=45,
                color="gold",
                edgecolor="black",
                linewidth=0.3,
                zorder=5
            )

        if pd.notna(last_mean):
            ax.scatter(
                round_max,
                last_mean,
                s=45,
                color="red",
                edgecolor="black",
                linewidth=0.3,
                zorder=5
            )

        ax.set_title(trait_label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Predicted value")
        ax.grid(True, alpha=0.3)

    total_axes = n_rows * n_cols
    for j in range(n_traits, total_axes):
        row = j % n_rows
        col = j // n_rows
        axes[row, col].axis("off")

    plt.tight_layout()

    png = f"{args.out}.png"
    pdf = f"{args.out}.pdf"

    plt.savefig(png, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    print(f"Loaded seeds: {len(files)}")
    print(f"Traits plotted: {n_traits}")
    print("Detected traits:")
    for c in pred_cols:
        print("  -", c[:-5])

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()