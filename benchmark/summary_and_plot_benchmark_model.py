#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Collect cohort benchmark results and create publication-ready figures."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, ttest_rel


MODEL_SPECS = (
    ("CropAR-Net", "croparnet"),
    ("Cropformer", "cropformer"),
    ("XGBoost", "xgboost"),
    ("BayesCpi", "bayescpi"),
    ("rrBLUP", "rrBLUP"),
    ("Lasso", "Lasso"),
    ("ElasticNet", "ElasticNet"),
    ("CLCNet", "CLCNet"),
    ("MENET", "MENET"),
    ("DNAwhisper", "Whisperer_of_DNA"),
    ("BNNs", "BNNs"),
    ("Aquila-SNP", "aquila-snp"),
)

MODEL_COLORS = {
    "CropAR-Net": "#56B4A9",
    "Cropformer": "#4C91C6",
    "XGBoost": "#9A89B8",
    "BayesCpi": "#78A6D1",
    "rrBLUP": "#D7B64C",
    "Lasso": "#A8CFAE",
    "ElasticNet": "#E9B985",
    "CLCNet": "#D2779E",
    "MENET": "#9B79C6",
    "DNAwhisper": "#6D8F72",
    "BNNs": "#C28D62",
    "Aquila-SNP": "#DF6878",
}

SCALE_NAMES = ("normalized", "processed", "standardized")
LINE_WIDTH = 1.05


@dataclass(frozen=True)
class ModelResult:
    model: str
    source_file: Path | None
    values: Mapping[str, float]
    status: str
    message: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect final nested-CV Pearson correlations for one cohort and "
            "draw model benchmark figures."
        )
    )
    parser.add_argument(
        "cohort",
        help="Cohort name used below each model's results directory, e.g. Maize1404.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Benchmark root directory. Default: directory containing this script.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <benchmark-dir>/benchmark_summary/<cohort>.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of leading models counted for each trait. Default: 3.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=500,
        help="PNG resolution. Default: 500.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures after saving them.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("mean", "outer_fold_mean", "test_pearsonr_mean"):
            number = finite_float(value.get(key))
            if number is not None:
                return number
        values = value.get("values")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            numbers = [number for item in values if (number := finite_float(item)) is not None]
            return float(np.mean(numbers)) if numbers else None
    return finite_float(value)


def pearson_from_scale(scale: Any, trait: str | None = None) -> float | None:
    if not isinstance(scale, Mapping):
        return None

    if trait is not None:
        per_trait = scale.get("per_trait")
        if isinstance(per_trait, Mapping) and trait in per_trait:
            trait_metrics = per_trait[trait]
            if isinstance(trait_metrics, Mapping):
                number = mean_value(trait_metrics.get("pearson"))
                if number is not None:
                    return number

    for container_name in ("aggregate",):
        container = scale.get(container_name)
        if isinstance(container, Mapping):
            number = mean_value(container.get("pearson"))
            if number is not None:
                return number

    for metric_name in ("pearson", "avg_pearson"):
        number = mean_value(scale.get(metric_name))
        if number is not None:
            return number
    return None


def pearson_from_metric_bundle(bundle: Any, trait: str | None = None) -> float | None:
    if not isinstance(bundle, Mapping):
        return None
    for scale_name in SCALE_NAMES:
        number = pearson_from_scale(bundle.get(scale_name), trait)
        if number is not None:
            return number
    return pearson_from_scale(bundle, trait)


def extract_aggregate(payload: Mapping[str, Any]) -> dict[str, float]:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return {}
    values: dict[str, float] = {}
    for trait, metrics in aggregate.items():
        number = pearson_from_metric_bundle(metrics, str(trait))
        if number is not None:
            values[str(trait)] = number
    return values


def extract_trait_results(payload: Mapping[str, Any]) -> dict[str, float]:
    results = payload.get("results")
    if not isinstance(results, Mapping):
        return {}
    values: dict[str, float] = {}
    for trait, result in results.items():
        if not isinstance(result, Mapping):
            continue
        number = pearson_from_metric_bundle(result.get("outer_fold_summary"), str(trait))
        if number is not None:
            values[str(trait)] = number
    return values


def extract_outer_fold_summary(payload: Mapping[str, Any]) -> dict[str, float]:
    summary = payload.get("outer_fold_summary")
    if not isinstance(summary, Mapping):
        return {}

    values: dict[str, float] = {}
    for scale_name in SCALE_NAMES:
        scale = summary.get(scale_name)
        if not isinstance(scale, Mapping):
            continue
        per_trait = scale.get("per_trait")
        if isinstance(per_trait, Mapping):
            for trait, metrics in per_trait.items():
                if isinstance(metrics, Mapping):
                    number = mean_value(metrics.get("pearson"))
                    if number is not None:
                        values[str(trait)] = number
        if values:
            return values

    for trait, metrics in summary.items():
        if not isinstance(metrics, Mapping):
            continue
        number = pearson_from_metric_bundle(metrics, str(trait))
        if number is not None:
            values[str(trait)] = number
    return values


def extract_result_rows(payload: Mapping[str, Any]) -> dict[str, float]:
    results = payload.get("results")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        return {}
    grouped: dict[str, list[float]] = {}
    for result in results:
        if not isinstance(result, Mapping):
            continue
        trait = result.get("trait")
        if trait is None:
            continue
        number = finite_float(result.get("test_pearson"))
        if number is None:
            number = pearson_from_metric_bundle(result.get("metrics"), str(trait))
        if number is not None:
            grouped.setdefault(str(trait), []).append(number)
    return {
        trait: float(np.mean(numbers))
        for trait, numbers in grouped.items()
        if numbers
    }


def extract_summary_values(payload: Mapping[str, Any]) -> dict[str, float]:
    for extractor in (
        extract_aggregate,
        extract_trait_results,
        extract_outer_fold_summary,
        extract_result_rows,
    ):
        values = extractor(payload)
        if values:
            return values
    return {}


def extract_trait_summary_files(directory: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not directory.is_dir():
        return values
    for summary_file in sorted(directory.glob("*/summary.json")):
        payload = read_json(summary_file)
        trait = payload.get("trait") or summary_file.parent.name
        number = pearson_from_metric_bundle(payload.get("metrics"), str(trait))
        if number is not None:
            values[str(trait)] = number
    return values


def extract_aquila_fold_metrics(directory: Path) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for metrics_file in sorted(directory.glob("fold_*/metrics.json")):
        payload = read_json(metrics_file)
        scale = payload.get("normalized")
        if not isinstance(scale, Mapping):
            continue
        per_trait = scale.get("per_trait")
        if not isinstance(per_trait, Mapping):
            continue
        for trait, metrics in per_trait.items():
            if not isinstance(metrics, Mapping):
                continue
            number = finite_float(metrics.get("pearson"))
            if number is not None:
                grouped.setdefault(str(trait), []).append(number)
    return {
        trait: float(np.mean(numbers))
        for trait, numbers in grouped.items()
        if numbers
    }


def cohort_matches(payload: Mapping[str, Any], cohort: str) -> bool:
    data_dir = payload.get("data_dir")
    if data_dir is None:
        return False
    normalized = str(data_dir).replace("\\", "/").rstrip("/")
    return (
        Path(normalized).name in {cohort, f"{cohort}.cv.data"}
        or f"/{cohort}/" in f"{normalized}/"
    )


def model_matches(payload: Mapping[str, Any], display_name: str) -> bool:
    model = payload.get("model_name", payload.get("model"))
    if model is None:
        return True
    canonical = str(model).casefold().replace("-", "").replace("_", "")
    expected = display_name.casefold().replace("-", "").replace("_", "")
    return canonical == expected


def summary_candidates(benchmark_dir: Path, model_dir: str, cohort: str) -> list[Path]:
    root = (
        benchmark_dir / model_dir
        if "/results/" in model_dir
        else benchmark_dir / model_dir / "results"
    )
    candidates = [root / cohort / "summary.json"]
    if model_dir == "Lasso":
        candidates.append(root / "lasso" / "summary.json")
    elif model_dir == "ElasticNet":
        candidates.append(root / "elasticnet" / "summary.json")
    return candidates


def load_model_result(
    benchmark_dir: Path,
    cohort: str,
    display_name: str,
    model_dir: str,
) -> ModelResult:
    candidates = summary_candidates(benchmark_dir, model_dir, cohort)
    summary_file = next((path for path in candidates if path.is_file()), None)
    if summary_file is None:
        return ModelResult(display_name, None, {}, "missing", "Final summary not found")

    try:
        payload = read_json(summary_file)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return ModelResult(display_name, summary_file, {}, "invalid", str(error))

    has_data_dir = payload.get("data_dir") is not None
    is_cohort_directory = summary_file.parent.name == cohort
    if (has_data_dir and not cohort_matches(payload, cohort)) or (
        not has_data_dir and not is_cohort_directory
    ):
        return ModelResult(
            display_name,
            summary_file,
            {},
            "mismatch",
            f"Summary does not identify cohort {cohort}",
        )
    if not model_matches(payload, display_name):
        return ModelResult(
            display_name,
            summary_file,
            {},
            "mismatch",
            f"Summary model identifier does not match {display_name}",
        )

    values = extract_summary_values(payload)
    if not values and model_dir == "Whisperer_of_DNA":
        values = extract_trait_summary_files(summary_file.parent)
    if not values and model_dir == "aquila-snp":
        values = extract_aquila_fold_metrics(summary_file.parent)

    if not values:
        return ModelResult(
            display_name,
            summary_file,
            {},
            "invalid",
            "No final per-trait Pearson correlations found",
        )
    return ModelResult(display_name, summary_file, values, "available", "OK")


def load_expected_traits(benchmark_dir: Path, cohort: str) -> list[str]:
    metadata_file = benchmark_dir / f"{cohort}.cv.data" / "metadata.json"
    if not metadata_file.is_file():
        return []
    try:
        payload = read_json(metadata_file)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    traits = payload.get("trait_names")
    if not isinstance(traits, Sequence) or isinstance(traits, (str, bytes)):
        return []
    return [str(trait) for trait in traits]


def collect_results(
    benchmark_dir: Path,
    cohort: str,
) -> tuple[list[ModelResult], list[str]]:
    results = [
        load_model_result(benchmark_dir, cohort, display_name, model_dir)
        for display_name, model_dir in MODEL_SPECS
    ]
    expected_traits = load_expected_traits(benchmark_dir, cohort)
    discovered_traits = sorted(
        {trait for result in results for trait in result.values},
        key=str.casefold,
    )
    traits = expected_traits or discovered_traits
    for trait in discovered_traits:
        if trait not in traits:
            traits.append(trait)
    return results, traits


def make_long_table(results: Sequence[ModelResult], traits: Sequence[str]) -> pd.DataFrame:
    records = []
    for result in results:
        for trait in traits:
            records.append(
                {
                    "model": result.model,
                    "phenotype": trait,
                    "pearson_r": result.values.get(trait, np.nan),
                    "status": result.status,
                    "source_file": (
                        str(result.source_file) if result.source_file is not None else ""
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def make_status_table(
    results: Sequence[ModelResult],
    traits: Sequence[str],
) -> pd.DataFrame:
    expected = len(traits)
    records = []
    for result in results:
        values = np.asarray(list(result.values.values()), dtype=float)
        finite = values[np.isfinite(values)]
        records.append(
            {
                "model": result.model,
                "status": result.status,
                "traits_found": int(finite.size),
                "traits_expected": expected,
                "mean_pearson_r": float(finite.mean()) if finite.size else np.nan,
                "source_file": (
                    str(result.source_file) if result.source_file is not None else ""
                ),
                "message": result.message,
            }
        )
    return pd.DataFrame.from_records(records)


def model_order(status_df: pd.DataFrame) -> list[str]:
    available = status_df[status_df["traits_found"] > 0].sort_values(
        ["mean_pearson_r", "model"],
        ascending=[False, True],
        na_position="last",
    )
    available_models = available["model"].tolist()
    configured = [display_name for display_name, _ in MODEL_SPECS]
    return available_models + [model for model in configured if model not in available_models]


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(list(pvalues), dtype=float)
    count = values.size
    if count == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, values[index] * (count - rank))
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def pairwise_paired_t_tests(long_df: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Paired t-tests of per-trait Pearson r between model means."""
    wide = long_df.pivot(index="phenotype", columns="model", values="pearson_r")
    records: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []

    available = [model for model in order if model in wide.columns]
    for left_index, left in enumerate(available):
        for right in available[left_index + 1 :]:
            paired = wide[[left, right]].dropna()
            left_values = paired[left].to_numpy(dtype=float)
            right_values = paired[right].to_numpy(dtype=float)
            if left_values.size < 2:
                continue
            try:
                statistic, pvalue = ttest_rel(
                    left_values,
                    right_values,
                    alternative="two-sided",
                )
            except ValueError:
                statistic, pvalue = float("nan"), 1.0
            pvalue = 1.0 if not math.isfinite(float(pvalue)) else float(pvalue)
            raw_pvalues.append(pvalue)
            differences = left_values - right_values
            records.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "n_paired_traits": int(left_values.size),
                    "mean_difference": float(np.mean(differences)),
                    "t_statistic": (
                        float(statistic) if math.isfinite(float(statistic)) else np.nan
                    ),
                    "p_raw": pvalue,
                }
            )

    adjusted = holm_adjust(raw_pvalues)
    for record, p_holm in zip(records, adjusted):
        record["p_holm"] = p_holm
        record["significant_0.05"] = bool(p_holm < 0.05)
    return pd.DataFrame.from_records(records)


def friedman_pvalue(long_df: pd.DataFrame, models: Sequence[str]) -> float | None:
    wide = long_df.pivot(index="phenotype", columns="model", values="pearson_r")
    present = [model for model in models if model in wide.columns]
    if len(present) < 3:
        return None
    complete = wide[present].dropna(axis=0, how="any")
    if complete.shape[0] < 2:
        return None
    try:
        _statistic, pvalue = friedmanchisquare(
            *[complete[model].to_numpy(dtype=float) for model in present]
        )
    except ValueError:
        return None
    number = finite_float(pvalue)
    return number


def _maximal_cliques(adjacency: np.ndarray) -> list[set[int]]:
    count = int(adjacency.shape[0])
    cliques: list[set[int]] = []

    def bron_kerbosch(current: set[int], candidates: set[int], excluded: set[int]) -> None:
        if not candidates and not excluded:
            cliques.append(set(current))
            return
        pivot = next(iter(candidates | excluded))
        pivot_neighbors = {index for index in range(count) if adjacency[pivot, index]}
        for vertex in list(candidates - pivot_neighbors):
            neighbors = {index for index in range(count) if adjacency[vertex, index]}
            bron_kerbosch(current | {vertex}, candidates & neighbors, excluded & neighbors)
            candidates.remove(vertex)
            excluded.add(vertex)

    bron_kerbosch(set(), set(range(count)), set())
    return cliques


def compact_letter_display(
    models: Sequence[str],
    pairwise_df: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, str]:
    """Letter groups for models; shared letters mean no significant difference."""
    names = list(models)
    if not names:
        return {}
    count = len(names)
    index = {name: position for position, name in enumerate(names)}
    different = np.zeros((count, count), dtype=bool)
    for row in pairwise_df.itertuples(index=False):
        if row.model_a not in index or row.model_b not in index:
            continue
        if float(row.p_holm) < alpha:
            left = index[str(row.model_a)]
            right = index[str(row.model_b)]
            different[left, right] = True
            different[right, left] = True

    adjacent = ~different
    np.fill_diagonal(adjacent, False)
    cliques = _maximal_cliques(adjacent)
    if not cliques:
        cliques = [{position} for position in range(count)]
    cliques.sort(key=lambda clique: (min(clique), -len(clique)))

    letters = [""] * count
    for clique_index, clique in enumerate(cliques):
        glyph = chr(ord("a") + clique_index)
        for position in sorted(clique):
            letters[position] += glyph
    return {name: letters[position] for position, name in enumerate(names)}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.linewidth": LINE_WIDTH,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.major.width": LINE_WIDTH,
            "ytick.major.width": LINE_WIDTH,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    sns.set_theme(style="white", context="paper")


def beautify_axes(axis: mpl.axes.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(LINE_WIDTH)
    axis.spines["bottom"].set_linewidth(LINE_WIDTH)
    axis.grid(axis="y", color="#D8DCE2", linewidth=0.65, alpha=0.75, zorder=0)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="both", length=3.5, width=LINE_WIDTH)


def save_figure(
    figure: mpl.figure.Figure,
    output_stem: Path,
    dpi: int,
) -> None:
    figure.savefig(Path(f"{output_stem}.pdf"), bbox_inches="tight")
    figure.savefig(Path(f"{output_stem}.png"), dpi=dpi, bbox_inches="tight")


def plot_performance_distribution(
    long_df: pd.DataFrame,
    status_df: pd.DataFrame,
    order: Sequence[str],
    cohort: str,
    output_dir: Path,
    dpi: int,
    letters: Mapping[str, str] | None = None,
    friedman_p: float | None = None,
) -> mpl.figure.Figure:
    plot_df = long_df[np.isfinite(long_df["pearson_r"])].copy()
    width = max(9.5, 0.78 * len(order))
    figure, axis = plt.subplots(figsize=(width, 5.55))
    palette = {model: MODEL_COLORS[model] for model in order}

    if not plot_df.empty:
        sns.boxplot(
            data=plot_df,
            x="model",
            y="pearson_r",
            order=order,
            hue="model",
            hue_order=order,
            palette=palette,
            dodge=False,
            legend=False,
            width=0.58,
            fliersize=0,
            linewidth=0.9,
            boxprops={"edgecolor": "#222222"},
            whiskerprops={"linewidth": 0.9, "color": "#222222"},
            capprops={"linewidth": 0.9, "color": "#222222"},
            medianprops={"color": "#111111", "linewidth": 1.25},
            ax=axis,
        )
        sns.stripplot(
            data=plot_df,
            x="model",
            y="pearson_r",
            order=order,
            color="#202020",
            size=2.6,
            alpha=0.30,
            jitter=0.17,
            ax=axis,
        )
        means = plot_df.groupby("model")["pearson_r"].mean()
        for index, model in enumerate(order):
            if model in means:
                axis.scatter(
                    index,
                    means[model],
                    marker="D",
                    s=25,
                    color="#111111",
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=5,
                )

    finite_values = plot_df["pearson_r"].to_numpy(dtype=float)
    lower = min(-0.05, float(np.nanmin(finite_values)) - 0.08) if finite_values.size else -0.05
    letter_pad = 0.07 if letters else 0.08
    upper = (
        max(0.85, float(np.nanmax(finite_values)) + letter_pad)
        if finite_values.size
        else 1.0
    )
    lower = max(-1.0, lower)
    upper = min(1.08, upper)
    axis.set_ylim(lower, upper)

    if letters and not plot_df.empty:
        tops = plot_df.groupby("model")["pearson_r"].max()
        for index, model in enumerate(order):
            letter = letters.get(model)
            if not letter or model not in tops:
                continue
            axis.text(
                index,
                float(tops[model]) + 0.018 * (upper - lower),
                letter,
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#111111",
                clip_on=False,
                zorder=6,
            )

    missing = set(status_df.loc[status_df["traits_found"] == 0, "model"])
    for index, model in enumerate(order):
        if model in missing:
            axis.text(
                index,
                lower + 0.025 * (upper - lower),
                "No result",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#9A9A9A",
            )

    axis.axhline(0, color="#777777", linewidth=0.75, linestyle=(0, (3, 3)), zorder=1)
    axis.set_title(f"{cohort}: predictive performance across traits", pad=12)
    axis.set_xlabel("")
    axis.set_ylabel("Outer-test Pearson's r")
    axis.set_xlim(-0.5, len(order) - 0.5)
    axis.set_xticks(np.arange(len(order)))
    axis.set_xticklabels(order)
    axis.tick_params(axis="x", labelrotation=28, labelsize=9.5)
    for label in axis.get_xticklabels():
        label.set_ha("right")
        label.set_fontweight("bold")
    beautify_axes(axis)
    note = (
        "Letters: paired t-tests on per-trait Pearson's r "
        "(Holm-adjusted α = 0.05). Models that share a letter are not significantly different."
    )
    if friedman_p is not None:
        note = f"Friedman p = {friedman_p:.2e}. " + note
    axis.text(
        0.0,
        -0.28,
        note,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
        wrap=True,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    save_figure(figure, output_dir / f"{cohort}.model_performance", dpi)
    return figure


def calculate_top_k(
    long_df: pd.DataFrame,
    order: Sequence[str],
    traits: Sequence[str],
    top_k: int,
) -> pd.DataFrame:
    ranked = long_df.copy()
    ranked["rank"] = ranked.groupby("phenotype")["pearson_r"].rank(
        ascending=False,
        method="min",
        na_option="keep",
    )
    counts = ranked.loc[ranked["rank"] <= top_k, "model"].value_counts()
    means = ranked.groupby("model")["pearson_r"].mean()
    denominator = len(traits)
    records = []
    for model in order:
        count = int(counts.get(model, 0))
        has_result = bool(ranked.loc[ranked["model"] == model, "pearson_r"].notna().any())
        records.append(
            {
                "model": model,
                "top_k_count": count if has_result else np.nan,
                "frequency": count / denominator if has_result and denominator else np.nan,
                "mean_pearson_r": means.get(model, np.nan),
            }
        )
    return pd.DataFrame.from_records(records)


def plot_top_k(
    top_df: pd.DataFrame,
    cohort: str,
    top_k: int,
    trait_count: int,
    output_dir: Path,
    dpi: int,
) -> mpl.figure.Figure:
    width = max(9.5, 0.78 * len(top_df))
    figure, axis = plt.subplots(figsize=(width, 5.0))
    x_positions = np.arange(len(top_df))

    for index, row in top_df.iterrows():
        frequency = finite_float(row["frequency"])
        if frequency is None:
            continue
        axis.bar(
            index,
            frequency,
            width=0.62,
            color=MODEL_COLORS[str(row["model"])],
            edgecolor="#222222",
            linewidth=0.9,
            zorder=3,
        )
        axis.text(
            index,
            frequency + 0.018,
            f"{int(row['top_k_count'])}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    for index, row in top_df.iterrows():
        if not math.isfinite(float(row["frequency"])):
            axis.text(
                index,
                0.018,
                "No result",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#9A9A9A",
            )

    max_frequency = pd.to_numeric(top_df["frequency"], errors="coerce").max()
    upper = max(0.5, min(1.08, float(max_frequency) + 0.13)) if pd.notna(max_frequency) else 1.0
    axis.set_ylim(0, upper)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        top_df["model"],
        rotation=28,
        ha="right",
        fontsize=9.5,
        fontweight="bold",
    )
    axis.set_title(f"{cohort}: frequency among the top {top_k} models", pad=12)
    axis.set_xlabel("")
    axis.set_ylabel(f"Proportion of traits in top {top_k}")
    axis.text(
        0.995,
        0.98,
        f"Numbers above bars are counts; {trait_count} traits total",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#666666",
    )
    beautify_axes(axis)
    figure.tight_layout()
    save_figure(figure, output_dir / f"{cohort}.model_top{top_k}", dpi)
    return figure


def plot_performance_heatmap(
    long_df: pd.DataFrame,
    traits: Sequence[str],
    cohort: str,
    output_dir: Path,
    dpi: int,
    letters: Mapping[str, str] | None = None,
    friedman_p: float | None = None,
) -> mpl.figure.Figure:
    """Plot a relative trait-by-model heatmap with aligned model boxplots."""
    raw_df = long_df.pivot(
        index="phenotype",
        columns="model",
        values="pearson_r",
    ).reindex(index=list(traits))
    model_means = raw_df.mean(axis=0, skipna=True)
    model_order = (
        pd.DataFrame(
            {
                "model": raw_df.columns,
                "mean_pearson_r": model_means.to_numpy(),
            }
        )
        .sort_values(
            ["mean_pearson_r", "model"],
            ascending=[False, True],
            na_position="last",
        )["model"]
        .tolist()
    )
    model_order = [model for model in model_order if pd.notna(model_means[model])]
    raw_df = raw_df.reindex(columns=model_order)
    model_means = model_means.reindex(model_order)

    # Colors represent within-trait relative performance (row-wise Min-Max);
    # annotations retain the original Pearson correlations.
    display_raw_df = pd.concat(
        [
            raw_df,
            pd.DataFrame(
                [model_means.to_numpy()],
                index=["Mean"],
                columns=model_order,
            ),
        ]
    )
    display_row_mins = display_raw_df.min(axis=1, skipna=True)
    display_row_ranges = (
        display_raw_df.max(axis=1, skipna=True) - display_row_mins
    ).replace(0.0, np.nan)
    display_relative_df = display_raw_df.sub(display_row_mins, axis=0).div(
        display_row_ranges,
        axis=0,
    )
    display_relative_df = display_relative_df.fillna(
        display_raw_df.notna().astype(float).where(display_raw_df.notna(), np.nan) * 0.5
    )
    annotations = display_raw_df.map(
        lambda value: f"{value:.3g}" if pd.notna(value) else ""
    )

    width = max(10.0, 0.82 * len(model_order) + 2.5)
    height = max(7.5, 0.46 * len(display_raw_df) + 4.8)
    figure = plt.figure(figsize=(width, height))
    grid = figure.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=(2.3, max(2.7, 0.46 * len(display_raw_df)), 0.18),
        hspace=0.56,
    )
    box_axis = figure.add_subplot(grid[0])
    heatmap_axis = figure.add_subplot(grid[1])
    colorbar_axis = figure.add_subplot(grid[2])

    boxplot_df = long_df[
        long_df["model"].isin(model_order) & np.isfinite(long_df["pearson_r"])
    ].copy()
    palette = {model: MODEL_COLORS[model] for model in model_order}
    sns.boxplot(
        data=boxplot_df,
        x="model",
        y="pearson_r",
        order=model_order,
        hue="model",
        hue_order=model_order,
        palette=palette,
        dodge=False,
        legend=False,
        width=0.58,
        fliersize=0,
        linewidth=0.9,
        boxprops={"edgecolor": "#222222"},
        whiskerprops={"linewidth": 0.9, "color": "#222222"},
        capprops={"linewidth": 0.9, "color": "#222222"},
        medianprops={"color": "#111111", "linewidth": 1.25},
        ax=box_axis,
    )
    sns.stripplot(
        data=boxplot_df,
        x="model",
        y="pearson_r",
        order=model_order,
        color="#202020",
        size=2.5,
        alpha=0.28,
        jitter=0.16,
        ax=box_axis,
    )

    finite_values = boxplot_df["pearson_r"].to_numpy(dtype=float)
    lower = (
        max(-1.0, min(-0.05, float(np.nanmin(finite_values)) - 0.08))
        if finite_values.size
        else -0.05
    )
    upper = (
        min(1.12, max(0.85, float(np.nanmax(finite_values)) + 0.20))
        if finite_values.size
        else 1.0
    )
    box_axis.set_ylim(lower, upper)
    tops = boxplot_df.groupby("model")["pearson_r"].max()
    for column_index, model in enumerate(model_order):
        letter = letters.get(model) if letters else None
        if letter and model in tops:
            box_axis.text(
                column_index,
                float(tops[model]) + 0.025 * (upper - lower),
                letter,
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                clip_on=False,
                zorder=6,
            )
    box_axis.axhline(
        0,
        color="#777777",
        linewidth=0.75,
        linestyle=(0, (3, 3)),
        zorder=1,
    )
    box_axis.set_title(
        f"{cohort}: model performance across phenotypes",
        pad=12,
    )
    box_axis.set_xlabel("")
    box_axis.set_ylabel("Outer-test Pearson's r")
    box_axis.set_xlim(-0.5, len(model_order) - 0.5)
    box_axis.set_xticks(np.arange(len(model_order)))
    box_axis.set_xticklabels(
        model_order,
        rotation=35,
        ha="right",
        fontsize=9,
        fontweight="bold",
    )
    box_axis.tick_params(axis="x", pad=3)
    beautify_axes(box_axis)
    test_note = (
        "Letters: paired t-tests of phenotype-level model means, "
        "Holm-adjusted α = 0.05; "
        "shared letters indicate no significant difference."
    )
    if friedman_p is not None:
        test_note = f"Friedman p = {friedman_p:.2e}. " + test_note
    box_axis.text(
        0.0,
        0.99,
        test_note,
        transform=box_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )

    sns.heatmap(
        display_relative_df,
        mask=display_raw_df.isna(),
        cmap="vlag",
        vmin=0.0,
        vmax=1.0,
        center=0.5,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 7.5},
        linewidths=0.45,
        linecolor="white",
        cbar_kws={
            "label": "Relative performance within phenotype (Min-Max)",
            "orientation": "horizontal",
        },
        cbar_ax=colorbar_axis,
        ax=heatmap_axis,
    )
    heatmap_axis.set_facecolor("#E6E6E6")
    heatmap_axis.set_title("")
    heatmap_axis.set_ylabel("Phenotype")
    heatmap_axis.set_xlabel("")
    heatmap_axis.tick_params(
        axis="x",
        top=False,
        bottom=False,
        labeltop=False,
        labelbottom=False,
    )
    heatmap_axis.tick_params(axis="y", labelrotation=0, labelsize=9)
    heatmap_axis.axhline(
        len(raw_df),
        color="#202020",
        linewidth=2.2,
    )
    heatmap_axis.get_yticklabels()[-1].set_fontweight("bold")
    colorbar_axis.xaxis.set_ticks_position("bottom")
    colorbar_axis.xaxis.set_label_position("bottom")
    colorbar_axis.set_xticks([0.0, 1.0])
    colorbar_axis.set_xticklabels(["Low", "High"])

    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.94,
        bottom=0.08,
        hspace=0.56,
    )
    save_figure(figure, output_dir / f"{cohort}.model_trait_heatmap", dpi)
    return figure


def write_tables(
    long_df: pd.DataFrame,
    status_df: pd.DataFrame,
    top_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    letters: Mapping[str, str],
    traits: Sequence[str],
    output_dir: Path,
    cohort: str,
) -> None:
    long_df.to_csv(
        output_dir / f"{cohort}.pearson_by_model_and_trait.long.tsv",
        sep="\t",
        index=False,
        na_rep="",
    )
    pivot = long_df.pivot(index="model", columns="phenotype", values="pearson_r")
    pivot = pivot.reindex([display_name for display_name, _ in MODEL_SPECS])
    pivot = pivot.reindex(columns=list(traits))
    pivot["mean_pearson_r"] = pivot.mean(axis=1, skipna=True)
    pivot.to_csv(
        output_dir / f"{cohort}.pearson_by_model_and_trait.pivot.tsv",
        sep="\t",
        na_rep="",
    )
    status_df.to_csv(
        output_dir / f"{cohort}.model_status.tsv",
        sep="\t",
        index=False,
        na_rep="",
    )
    top_df.to_csv(
        output_dir / f"{cohort}.top_model_frequency.tsv",
        sep="\t",
        index=False,
        na_rep="",
    )
    if not pairwise_df.empty:
        pairwise_df.to_csv(
            output_dir / f"{cohort}.model_pairwise_paired_ttest.tsv",
            sep="\t",
            index=False,
            na_rep="",
        )
    if letters:
        pd.DataFrame(
            [{"model": model, "cld_letter": letter} for model, letter in letters.items()]
        ).to_csv(
            output_dir / f"{cohort}.model_significance_letters.tsv",
            sep="\t",
            index=False,
        )


def print_summary(status_df: pd.DataFrame, output_dir: Path) -> None:
    print("Model result discovery:")
    for row in status_df.itertuples(index=False):
        source = row.source_file or "-"
        mean_text = (
            f"{row.mean_pearson_r:.4f}"
            if finite_float(row.mean_pearson_r) is not None
            else "-"
        )
        print(
            f"  {row.model:<14} {row.status:<10} "
            f"traits={row.traits_found}/{row.traits_expected} "
            f"mean_r={mean_text} source={source}"
        )
    print(f"\nOutputs written to: {output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    benchmark_dir = args.benchmark_dir.resolve()
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else benchmark_dir / "benchmark_summary" / args.cohort
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results, traits = collect_results(benchmark_dir, args.cohort)
    if not traits:
        raise RuntimeError(
            f"No trait metadata or completed model results found for cohort {args.cohort}"
        )

    long_df = make_long_table(results, traits)
    status_df = make_status_table(results, traits)
    order = model_order(status_df)
    tested_models = [
        model
        for model in order
        if int(status_df.loc[status_df["model"] == model, "traits_found"].iloc[0]) > 0
    ]
    pairwise_df = pairwise_paired_t_tests(long_df, tested_models)
    letters = compact_letter_display(tested_models, pairwise_df)
    friedman_p = friedman_pvalue(long_df, tested_models)
    top_df = calculate_top_k(long_df, order, traits, args.top_k)
    write_tables(
        long_df,
        status_df,
        top_df,
        pairwise_df,
        letters,
        traits,
        output_dir,
        args.cohort,
    )

    configure_style()
    figures = [
        plot_performance_distribution(
            long_df,
            status_df,
            order,
            args.cohort,
            output_dir,
            args.dpi,
            letters=letters,
            friedman_p=friedman_p,
        ),
        plot_top_k(
            top_df,
            args.cohort,
            args.top_k,
            len(traits),
            output_dir,
            args.dpi,
        ),
        plot_performance_heatmap(
            long_df,
            traits,
            args.cohort,
            output_dir,
            args.dpi,
            letters=letters,
            friedman_p=friedman_p,
        ),
    ]
    print_summary(status_df, output_dir)

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*set_ticklabels.*")
        main()
