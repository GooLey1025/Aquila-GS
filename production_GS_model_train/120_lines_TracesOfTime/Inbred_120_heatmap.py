#!/usr/bin/env python3
"""Heatmap of four-environment GYP predictions by period and subspecies."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from openpyxl.styles import Alignment, Font


PRED_COLS = [
    "GYP_BeiJ15_Pred",
    "GYP_WenJ15_Pred",
    "GYP_YangZ15_Pred",
    "GYP_LingS15_Pred",
]
COL_RENAME = {
    "GYP_BeiJ15_Pred": "BeiJing",
    "GYP_WenJ15_Pred": "SiChuan-WenJiang",
    "GYP_YangZ15_Pred": "JiangSu-YangZhou",
    "GYP_LingS15_Pred": "HaiNan-LingShui",
}
GROUP_ORDER = ["japonica", "indica"]
STAGE_ORDER = ["1976–2000", "2001–2010", "2011–2020", "2021–2024"]
STAGE_RANK = {stage: rank for rank, stage in enumerate(STAGE_ORDER)}
STAGE_COLORS = {
    "1976–2000": "#D9D9D9",
    "2001–2010": "#AFAFAF",
    "2011–2020": "#707070",
    "2021–2024": "#202020",
}
GROUP_COLORS = {"japonica": "#6FA87C", "indica": "#D9798A"}
BLACK = "#000000"
WHITE = "#FFFFFF"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot four-environment GYP predictions by breeding period and subspecies."
    )
    parser.add_argument("--meta", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument(
        "--panel",
        required=True,
        help="Workbook listing the 300 lines and the figure each line belongs to.",
    )
    parser.add_argument(
        "--figure",
        default="Fig. 7C",
        help="Value in the workbook Figure column used for this heatmap (default: Fig. 7C).",
    )
    parser.add_argument(
        "--output",
        default="GYP_4env_heatmap_by_subpop_stage_sorted_horizontal_bwstage.pdf",
    )
    return parser.parse_args()


def assign_stage(year):
    if pd.isna(year):
        return None
    if 1976 <= year <= 2000:
        return STAGE_ORDER[0]
    if 2001 <= year <= 2010:
        return STAGE_ORDER[1]
    if 2011 <= year <= 2020:
        return STAGE_ORDER[2]
    if 2021 <= year <= 2024:
        return STAGE_ORDER[3]
    return None


def _figure_key(value):
    text = str(value).strip().lower().replace(" ", "").replace(".", "")
    if text.startswith("figure"):
        text = "fig" + text.removeprefix("figure")
    return text


def load_panel_ids(path, figure):
    """Return Sample_IDs whose Figure column matches the requested panel."""
    raw = pd.read_excel(path, header=None)
    header_rows = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Sample_ID")]
    if len(header_rows) != 1:
        raise ValueError(f"{path} does not have a single Sample_ID header row")
    panel = pd.read_excel(path, header=int(header_rows[0]))
    required = {"Sample_ID", "Figure"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    panel["Sample_ID"] = panel["Sample_ID"].astype(str).str.strip()
    panel = panel.loc[panel["Sample_ID"].ne("") & panel["Sample_ID"].ne("nan")].copy()
    selected = panel.loc[
        panel["Figure"].map(_figure_key) == _figure_key(figure), "Sample_ID"
    ]
    ids = selected.tolist()
    if not ids:
        available = sorted(panel["Figure"].dropna().astype(str).unique())
        raise ValueError(
            f"No samples with Figure {_figure_key(figure)!r} in {path}. "
            f"Available values: {available}"
        )
    duplicated = pd.Series(ids).duplicated()
    if duplicated.any():
        raise ValueError(f"Duplicate Sample_ID values for {figure}: {ids[:10]}")
    return ids


def load_data(meta_path, pred_path, panel_ids):
    meta = pd.read_csv(meta_path, sep="\t")
    required_meta = {"Sample_ID", "Year", "Subspecies"}
    missing_meta = required_meta - set(meta.columns)
    if missing_meta:
        raise ValueError(f"Metadata is missing columns: {sorted(missing_meta)}")

    meta["Sample_ID"] = meta["Sample_ID"].astype(str).str.strip()
    missing_ids = sorted(set(panel_ids) - set(meta["Sample_ID"]))
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} panel samples are absent from metadata: {missing_ids[:10]}"
        )
    meta = meta.loc[meta["Sample_ID"].isin(panel_ids)].copy()
    meta["Year"] = pd.to_numeric(meta["Year"], errors="coerce")
    meta["Group"] = meta["Subspecies"].astype(str).str.strip().str.lower()
    meta = meta.loc[meta["Group"].isin(GROUP_ORDER)].copy()
    meta["Stage"] = meta["Year"].apply(assign_stage)
    meta = meta.dropna(subset=["Stage"])

    pred = pd.read_csv(pred_path, sep="\t")
    required_pred = {"Sample_ID", *PRED_COLS}
    missing_pred = required_pred - set(pred.columns)
    if missing_pred:
        raise ValueError(f"Predictions are missing columns: {sorted(missing_pred)}")
    pred["Sample_ID"] = pred["Sample_ID"].astype(str).str.strip()
    pred[PRED_COLS] = pred[PRED_COLS].apply(pd.to_numeric, errors="coerce")

    df = meta[["Sample_ID", "Group", "Year", "Stage"]].merge(
        pred[["Sample_ID", *PRED_COLS]],
        on="Sample_ID",
        how="inner",
        validate="one_to_one",
    )
    df = df.dropna(subset=PRED_COLS).copy()
    if df.empty:
        raise ValueError("No complete samples matched between metadata and predictions")
    df["Mean_GYP"] = df[PRED_COLS].mean(axis=1)
    df["Stage_rank"] = df["Stage"].map(STAGE_RANK)
    return df


def configure_style():
    sns.set_style("white")
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "axes.edgecolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
    })


def prepare_panels(df):
    panels = {}
    spans = {}
    max_samples = 0
    for group in GROUP_ORDER:
        sub = df.loc[df["Group"] == group].sort_values(
            ["Stage_rank", "Year", "Mean_GYP", "Sample_ID"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        panels[group] = sub.set_index("Sample_ID")[PRED_COLS].rename(
            columns=COL_RENAME
        ).T
        max_samples = max(max_samples, len(sub))

        group_spans = []
        start = 0
        for stage in STAGE_ORDER:
            count = int((sub["Stage"] == stage).sum())
            if count:
                group_spans.append((stage, start, start + count))
                start += count
        spans[group] = group_spans
    return panels, spans, max_samples


def plot_heatmap(df, output):
    configure_style()
    panels, spans, max_samples = prepare_panels(df)
    fig = plt.figure(figsize=(max(11.0, 0.17 * max_samples), 4.5))
    grid = GridSpec(
        3,
        2,
        width_ratios=[40, 2],
        height_ratios=[1, 1, 0.13],
        wspace=0.02,
        hspace=0.20,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
    ]
    band_axes = [
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
    ]
    colorbar_axis = fig.add_subplot(grid[2, 0])
    cmap = sns.blend_palette(
        ["#3B4DA1", "#6FA8DC", "#DCE6F2", "#F3E7C3", "#F3B36A", "#E85C3B", "#B40426"],
        as_cmap=True,
    )
    vmin = df[PRED_COLS].min().min()
    vmax = df[PRED_COLS].max().max()

    for index, group in enumerate(GROUP_ORDER):
        axis = axes[index]
        heat_df = panels[group]
        sns.heatmap(
            heat_df,
            ax=axis,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=index == 1,
            cbar_ax=colorbar_axis if index == 1 else None,
            cbar_kws={"orientation": "horizontal"} if index == 1 else None,
            linewidths=0.20,
            linecolor=WHITE,
            xticklabels=False,
            yticklabels=True,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="y", labelsize=12, rotation=0, pad=7)
        axis.tick_params(axis="x", length=0)
        axis.set_yticklabels(heat_df.index, rotation=0)

        for _, _, end in spans[group][:-1]:
            axis.vlines(end, 0, heat_df.shape[0], colors=BLACK, linewidth=1.2)

        if index == 0:
            transform = axis.get_xaxis_transform()
            for stage, start, end in spans[group]:
                axis.add_patch(
                    Rectangle(
                        (start, 1.08),
                        end - start,
                        0.18,
                        transform=transform,
                        facecolor=STAGE_COLORS[stage],
                        edgecolor="none",
                        clip_on=False,
                    )
                )
                text_color = WHITE if stage in STAGE_ORDER[2:] else BLACK
                axis.text(
                    (start + end) / 2,
                    1.17,
                    stage,
                    transform=transform,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=text_color,
                )

        band_axis = band_axes[index]
        band_axis.add_patch(
            Rectangle((0, 0), 1, 1, facecolor=GROUP_COLORS[group], edgecolor="none")
        )
        band_axis.text(
            0.5,
            0.5,
            group.capitalize(),
            ha="center",
            va="center",
            rotation=90,
            fontsize=15,
            color=WHITE,
            fontweight="bold",
            fontstyle="italic",
        )
        band_axis.set_xlim(0, 1)
        band_axis.set_ylim(0, 1)
        band_axis.axis("off")

    colorbar = axes[1].collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12, length=3)
    colorbar.set_label(
        "Predicted grain yield per plant [g]",
        fontsize=13,
        fontweight="bold",
        labelpad=8,
    )
    colorbar.outline.set_visible(False)
    plt.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.13)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


SOURCE_COLUMNS = {
    "GYP_BeiJ15_Pred": "Predicted_GYP_BeiJing_g",
    "GYP_WenJ15_Pred": "Predicted_GYP_SiChuan_WenJiang_g",
    "GYP_YangZ15_Pred": "Predicted_GYP_JiangSu_YangZhou_g",
    "GYP_LingS15_Pred": "Predicted_GYP_HaiNan_LingShui_g",
}


def _ordered_samples(df):
    frames = []
    for group in GROUP_ORDER:
        sub = df.loc[df["Group"] == group].sort_values(
            ["Stage_rank", "Year", "Mean_GYP", "Sample_ID"],
            ascending=[True, True, False, True],
            kind="mergesort",
        ).copy()
        sub["Panel_column"] = range(1, len(sub) + 1)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def write_source_data(df, output):
    """Write the heatmap cell values for a Nature source-data file."""
    table = _ordered_samples(df)
    individuals = pd.DataFrame({
        "Sample_ID": table["Sample_ID"].to_numpy(),
        "Subspecies": table["Group"].str.capitalize().to_numpy(),
        "Release_year": table["Year"].to_numpy(),
        "Breeding_period": table["Stage"].to_numpy(),
        "Panel_column": table["Panel_column"].to_numpy(),
    })
    for source, label in SOURCE_COLUMNS.items():
        individuals[label] = table[source].to_numpy()
    individuals["Mean_of_four_environments_g"] = table["Mean_GYP"].to_numpy()

    n_samples = int(individuals["Sample_ID"].nunique())
    notes = pd.DataFrame({
        "Item": [
            "Figure",
            "Samples",
            "Predicted_GYP_columns",
            "Panel_column",
            "Mean_of_four_environments_g",
            "Breeding_period",
            "Color_scale",
        ],
        "Description": [
            "Heatmap of predicted grain yield per plant in four environments, split into japonica and indica panels.",
            f"{n_samples} rice lines labelled Fig. 7C, restricted to japonica or indica and release years 1976–2024. Each subspecies panel contains 15 lines in each breeding period.",
            "Aquila-SNP 10-fold-mean predictions, in grams per plant. Columns follow the heatmap row order: Beijing, Sichuan-Wenjiang, Jiangsu-Yangzhou, and Hainan-Lingshui.",
            "Left-to-right position within a subspecies panel. Lines are ordered by breeding period, then release year, then the four-environment mean from high to low.",
            "Arithmetic mean of the four predicted environment values for that line. It determines the within-period order and is not a separate heatmap row.",
            "1976–2000, 2001–2010, 2011–2020, and 2021–2024 are assigned from the cultivar release year. Vertical lines in the heatmap separate these periods.",
            f"One color scale covers both panels, from the minimum to the maximum predicted value among these {n_samples} lines.",
        ],
    })

    path = Path(output).with_name(Path(output).stem + "_source_data.xlsx")
    sheet_name = "Source_data"
    header_row = 2
    notes_title_row = header_row + len(individuals) + 2
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        individuals.to_excel(
            writer, sheet_name=sheet_name, index=False, startrow=header_row - 1
        )
        notes.to_excel(
            writer, sheet_name=sheet_name, index=False, startrow=notes_title_row
        )
        worksheet = writer.sheets[sheet_name]
        regular = Font(name="Times New Roman", size=11)
        bold = Font(name="Times New Roman", size=11, bold=True)
        for row_index, title in (
            (1, "Heatmap values"),
            (notes_title_row, "Notes"),
        ):
            worksheet.cell(row_index, 1, title).font = bold
        for row_index in (header_row, notes_title_row + 1):
            for cell in worksheet[row_index]:
                if cell.value is not None:
                    cell.font = bold
        last_column = individuals.shape[1]
        for row_index in range(notes_title_row + 2, notes_title_row + 2 + len(notes)):
            worksheet.merge_cells(
                start_row=row_index,
                start_column=2,
                end_row=row_index,
                end_column=last_column,
            )
            worksheet.cell(row_index, 2).alignment = Alignment(
                wrap_text=True, vertical="center"
            )
            worksheet.row_dimensions[row_index].height = 32
        worksheet.freeze_panes = "A3"
        worksheet.auto_filter.ref = (
            f"A{header_row}:{worksheet.cell(header_row, last_column).column_letter}"
            f"{header_row + len(individuals)}"
        )
        for column_index, column in enumerate(individuals.columns, start=1):
            width = max(len(str(column)), 14) + 2
            worksheet.column_dimensions[
                worksheet.cell(header_row, column_index).column_letter
            ].width = min(width, 46)
        for row in worksheet.iter_rows():
            for cell in row:
                cell.font = bold if cell.font and cell.font.bold else regular
    return path


def main():
    args = parse_args()
    panel_ids = load_panel_ids(args.panel, args.figure)
    print(f"[DATA] {args.figure} samples in panel table: {len(panel_ids)}")
    df = load_data(args.meta, args.pred, panel_ids)
    print(f"[DATA] Complete matched samples: {len(df)}")
    print("\nCounts by group and stage:")
    print(
        df.groupby(["Stage", "Group"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=STAGE_ORDER, columns=GROUP_ORDER, fill_value=0)
    )
    plot_heatmap(df, args.output)
    source_data = write_source_data(df, args.output)
    print(f"\n[DONE] {args.output}")
    print(f"[DONE] {Path(args.output).with_suffix('.png')}")
    print(f"[DONE] {source_data}")


if __name__ == "__main__":
    main()
