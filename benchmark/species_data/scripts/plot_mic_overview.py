#!/usr/bin/env python3
"""Chromosome distribution and PCA of the shared 10k MIC marker panels."""

import gzip
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/data4/gulei/projects/Aquila-GS/benchmark/species_data")
OUT = BASE / "MIC_plots"

PANELS = [
    {
        "species": "Maize",
        "vcf": BASE / "Maize1404/Maize1404.MIC.rename.vcf.gz",
        "positions": Path("/tmp/mic_pos/maize.tsv"),
        "eigenvec": BASE / "Maize1404/Maize1404.MIC.rename.eigenvec",
        "eigenval": BASE / "Maize1404/Maize1404.MIC.rename.eigenval",
    },
    {
        "species": "Soybean",
        "vcf": BASE / "Soybean975/Soybean2795.MIC.vcf.gz",
        "positions": Path("/tmp/mic_pos/soy.tsv"),
        "eigenvec": BASE / "Soybean975/Soybean2795.MIC.eigenvec",
        "eigenval": BASE / "Soybean975/Soybean2795.MIC.eigenval",
    },
    {
        "species": "Wheat",
        "vcf": BASE / "wheat994/wheat994.MIC.vcf.gz",
        "positions": Path("/tmp/mic_pos/wheat.tsv"),
        "eigenvec": BASE / "wheat994/wheat994.MIC.eigenvec",
        "eigenval": BASE / "wheat994/wheat994.MIC.eigenval",
    },
]


def chrom_key(name: str):
    text = str(name)
    if text.startswith("chr"):
        text = text[3:]
    match = re.match(r"(\d+)(.*)", text)
    if match:
        return (0, int(match.group(1)), match.group(2))
    return (1, 0, text)


def contig_lengths(vcf_path: Path) -> dict[str, int]:
    lengths = {}
    with gzip.open(vcf_path, "rt") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            if not line.startswith("##contig"):
                continue
            match = re.search(r"ID=([^,>]+).*length=(\d+)", line)
            if match:
                lengths[match.group(1)] = int(match.group(2))
    return lengths


def load_positions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", header=None, names=["chrom", "pos"])
    frame["chrom"] = frame["chrom"].astype(str)
    return frame


def plot_counts(ax, species, positions, lengths):
    order = sorted(lengths, key=chrom_key)
    counts = positions.groupby("chrom").size()
    values = [int(counts.get(chrom, 0)) for chrom in order]
    labels = [chrom[3:] if chrom.startswith("chr") else chrom for chrom in order]
    colors = ["#c44e52" if value == max(values) else "#4c72b0" for value in values]
    ax.bar(range(len(order)), values, color=colors, width=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Selected SNPs")
    ax.set_title(f"{species}  n={len(positions):,}")
    top = order[int(np.argmax(values))]
    top_label = top[3:] if top.startswith("chr") else top
    ax.text(
        0.98,
        0.95,
        f"{top_label}: {max(values):,} ({100 * max(values) / len(positions):.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )


def plot_density(ax, species, positions, lengths):
    order = sorted(lengths, key=chrom_key)
    offset = 0.0
    ticks = []
    labels = []
    boundaries = [0.0]
    for chrom in order:
        length_mb = lengths[chrom] / 1e6
        subset = positions.loc[positions["chrom"] == chrom, "pos"] / 1e6
        if len(subset):
            bins = max(int(length_mb / 20), 1)
            hist, edges = np.histogram(subset, bins=bins, range=(0, length_mb))
            centers = offset + (edges[:-1] + edges[1:]) / 2
            ax.bar(centers, hist, width=(edges[1] - edges[0]) * 0.9, color="#4c72b0", linewidth=0)
        ticks.append(offset + length_mb / 2)
        labels.append(chrom[3:] if chrom.startswith("chr") else chrom)
        offset += length_mb
        boundaries.append(offset)
        ax.axvline(offset, color="0.85", linewidth=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_xlim(0, offset)
    ax.set_ylabel("SNPs / 20 Mb")
    ax.set_title(species)
    ax.set_xlabel("Chromosome")


def plot_pca(ax, species, eigenvec, eigenval):
    frame = pd.read_csv(eigenvec, sep=r"\s+")
    values = pd.read_csv(eigenval, header=None)[0].to_numpy()
    explained = values / values.sum() * 100
    ax.scatter(frame["PC1"], frame["PC2"], s=12, alpha=0.75, c="#4c72b0", linewidths=0)
    ax.axhline(0, color="0.7", linewidth=0.5)
    ax.axvline(0, color="0.7", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title(f"{species}  n={len(frame)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = []
    for panel in PANELS:
        loaded.append(
            {
                **panel,
                "positions": load_positions(panel["positions"]),
                "lengths": contig_lengths(panel["vcf"]),
            }
        )

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for ax, panel in zip(axes, loaded):
        plot_counts(ax, panel["species"], panel["positions"], panel["lengths"])
    fig.savefig(OUT / "MIC_snp_per_chromosome.png", dpi=200)
    fig.savefig(OUT / "MIC_snp_per_chromosome.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)
    for ax, panel in zip(axes, loaded):
        plot_density(ax, panel["species"], panel["positions"], panel["lengths"])
    fig.savefig(OUT / "MIC_snp_along_genome.png", dpi=200)
    fig.savefig(OUT / "MIC_snp_along_genome.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, panel in zip(axes, loaded):
        plot_pca(ax, panel["species"], panel["eigenvec"], panel["eigenval"])
    fig.savefig(OUT / "MIC_pca.png", dpi=200)
    fig.savefig(OUT / "MIC_pca.pdf")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
