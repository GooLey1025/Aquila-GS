#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evolve-dir", required=True)
    parser.add_argument("--direction", default=None)
    parser.add_argument("--task-mapping", default=None)
    parser.add_argument("--out", default="trait_change_summary")
    parser.add_argument("--fig-width-per-col", type=float, default=0.3,
                        help="Width scaling factor per character-width unit (default: 0.3)")
    parser.add_argument("--fig-height-per-row", type=float, default=0.5,
                        help="Height per trait row in inches (default: 0.4)")
    return parser.parse_args()


def direction_symbol(direction):
    if direction == "maximize":
        return "↑"
    if direction == "minimize":
        return "↓"
    if direction == "maintain":
        return "→"   # "hold current value"
    if direction == "neutral":
        return "—"
    return ""


def direction_color(direction):
    if direction == "maximize":
        return "crimson"
    if direction == "minimize":
        return "#1f5fbf"
    if direction == "maintain":
        return "#2e8b57"   # sea green — stability intent
    return "gray"


def maintain_color(change_percent, threshold_pct=5.0):
    """Color for the actual-change column of maintain traits.

    Green = barely changed (good, stayed near baseline).
    Red   = drifted far from baseline (penalized by SI).
    """
    if change_percent is None or np.isnan(change_percent):
        return "gray"
    abs_change = abs(change_percent)
    # Interpolate: 0% → green, threshold_pct% → yellow, beyond → red
    t = min(abs_change / threshold_pct, 1.0)
    # green → yellow → red
    if t < 0.5:
        r = int(255 * (2 * t))
        g = 200
        b = 0
    else:
        r = 255
        g = int(200 * (1 - (t - 0.5) * 2))
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


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

    # Load task mapping to get all regression traits
    if args.task_mapping:
        task_map = pd.read_csv(args.task_mapping, sep="\t")
        task_map.columns = task_map.columns.str.strip()
        all_traits = task_map[task_map["task_type"] == "regression"]["task_name"].tolist()
    else:
        # Fallback: collect all traits from direction file
        all_traits = None

    # Load direction overrides
    if args.direction:
        dir_df = pd.read_csv(args.direction, sep=r"\s+", engine="python")
        dir_df.columns = dir_df.columns.str.strip()
        # Support both 'phenotype' and 'task_name' column names
        if "phenotype" in dir_df.columns:
            dir_df = dir_df.rename(columns={"phenotype": "task_name"})
        dir_df["task_name"] = dir_df["task_name"].astype(str).str.strip()
        dir_df["direction"] = dir_df["direction"].astype(str).str.strip().str.lower()
        direction = dict(zip(dir_df["task_name"], dir_df["direction"]))
        if all_traits is None:
            all_traits = list(direction.keys())
    else:
        direction = {}
        if all_traits is None:
            raise ValueError("Must provide either --direction or --task-mapping")

    # Build full direction table with neutral defaults
    records = []
    for trait in all_traits:
        direc = direction.get(trait, "neutral")
        records.append({"task_name": trait, "direction": direc})
    direction_df = pd.DataFrame(records)

    files = sorted(
        glob.glob(os.path.join(args.evolve_dir, "seed_*", "round_predictions.tsv")),
        key=lambda x: int(os.path.basename(os.path.dirname(x)).replace("seed_", ""))
    )

    if not files:
        raise FileNotFoundError(f"No round_predictions.tsv found in {args.evolve_dir}/seed_*")

    records = []

    for _, row in direction_df.iterrows():
        trait = row["task_name"]
        direc = row["direction"]

        pred_col = f"{trait}_pred"
        baseline_col = f"{trait}_baseline"

        ratios = []
        baselines = []
        finals = []

        for f in files:
            df = pd.read_csv(f, sep=r"\s+", engine="python")
            df.columns = df.columns.str.strip()

            if pred_col not in df.columns or baseline_col not in df.columns:
                continue

            df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
            df = df.dropna(subset=["Round"]).sort_values("Round")

            if df.empty:
                continue

            last = df.iloc[-1]

            baseline = pd.to_numeric(last[baseline_col], errors="coerce")
            final = pd.to_numeric(last[pred_col], errors="coerce")

            if pd.isna(baseline) or pd.isna(final):
                continue

            if abs(baseline) < 1e-12:
                ratio = np.nan
            else:
                ratio = (final - baseline) / abs(baseline) * 100

            ratios.append(ratio)
            baselines.append(baseline)
            finals.append(final)

        ratios = np.asarray(ratios, dtype=float)

        records.append({
            "trait": trait,
            "trait_label": short_trait_name(trait),
            "direction": direc,
            "mean_change_percent": np.nanmean(ratios),
            "sd_change_percent": np.nanstd(ratios),
            "n_seed": np.sum(~np.isnan(ratios)),
            "mean_baseline": np.nanmean(baselines),
            "mean_final": np.nanmean(finals),
        })

    result = pd.DataFrame(records)

    optimize = result[result["direction"].isin(["maximize", "minimize"])]
    maintain = result[result["direction"].isin(["maintain"])]
    neutral = result[result["direction"].isin(["neutral"])]

    result = pd.concat([optimize, maintain, neutral], ignore_index=True)

    result.to_csv(f"{args.out}.tsv", sep="\t", index=False)

    n_traits = len(result)

    fig_width = args.fig_width_per_col * 30  # ~30 "units" of character width budget
    fig_height = args.fig_height_per_row * (n_traits + 2)  # +2 for header rows

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(n_traits + 1, -1)
    ax.axis("off")

    x_trait = 0.02
    x_direction = 0.58
    x_change = 0.88

    ax.text(
        x_trait,
        -0.2,
        f"Trait ({n_traits})",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    ax.text(
        x_direction,
        -0.2,
        "Direction\n(target)",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="bottom"
    )

    ax.text(
        x_change,
        -0.2,
        "Mean change ratio\n(relative to Teqing)",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="bottom"
    )

    for i, row in result.iterrows():
        y = i + 1

        trait = row["trait_label"]
        direc = row["direction"]
        change = row["mean_change_percent"]

        ax.text(
            x_trait,
            y,
            f"({i + 1}) {trait}",
            fontsize=13,
            ha="left",
            va="center"
        )

        ax.text(
            x_direction,
            y,
            direction_symbol(direc),
            fontsize=16,
            fontweight="bold",
            color=direction_color(direc),
            ha="center",
            va="center"
        )

        if np.isnan(change):
            change_text = "NA"
            change_color = "gray"
        else:
            change_text = f"{change:+.1f}%"
            if direc == "maintain":
                change_color = maintain_color(change)
            elif direc == "neutral":
                change_color = "gray"
            else:
                change_color = direction_color(direc)

        ax.text(
            x_change,
            y,
            change_text,
            fontsize=16,
            fontweight="bold",
            color=change_color,
            ha="center",
            va="center"
        )

    plt.savefig(f"{args.out}.pdf", bbox_inches="tight")
    plt.savefig(f"{args.out}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loaded seeds: {len(files)}")
    print(f"Saved: {args.out}.tsv")
    print(f"Saved: {args.out}.pdf")
    print(f"Saved: {args.out}.png")


if __name__ == "__main__":
    main()