#!/usr/bin/env python3
"""Predict with and average every fold model in one production output directory.

Each ``fold_<k>/best_model.pt`` is scored with ``aquila_predict.py`` on the
original phenotype scale, then the fold outputs are averaged into one table.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

from aquila.scripts.aquila_ig_ensemble import discover_fold_checkpoints
from aquila.scripts.aquila_predict import main as predict_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict a VCF with every fold model under --model-dir and "
            "write the mean prediction"
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="One directory containing fold_<k>/best_model.pt",
    )
    parser.add_argument("--vcf", required=True, help="Genotype VCF to score")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="TSV of Sample_ID and mean predictions",
    )
    parser.add_argument(
        "--encoding-type",
        default=None,
        help="Passed through to aquila_predict.py when the checkpoint has none",
    )
    parser.add_argument(
        "--id-prefix",
        default=None,
        help="Keep VCF IDs with this prefix, for example SNP-",
    )
    parser.add_argument(
        "--normalize-heterozygous-order",
        action="store_true",
        help=(
            "Treat unphased 1/0 and 1|0 calls as 0/1 while reading the VCF; "
            "passed through to aquila_predict.py"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device passed through to aquila_predict.py",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prediction batch size passed through to aquila_predict.py",
    )
    parser.add_argument(
        "--keep-fold-predictions",
        default=None,
        help="If set, also write each fold TSV into this directory",
    )
    return parser.parse_args()


def mean_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average numeric prediction columns across fold tables."""
    if not frames:
        raise ValueError("No fold predictions to average")
    indexed = []
    for frame in frames:
        if "Sample_ID" not in frame.columns:
            raise ValueError("Prediction table is missing Sample_ID")
        current = frame.copy()
        current["Sample_ID"] = current["Sample_ID"].astype(str).str.strip()
        if current["Sample_ID"].duplicated().any():
            duplicates = current.loc[
                current["Sample_ID"].duplicated(), "Sample_ID"
            ].unique()
            raise ValueError(
                "Prediction table contains duplicate Sample_ID values: "
                + ", ".join(duplicates[:5])
            )
        current = current.set_index("Sample_ID").sort_index()
        indexed.append(current)
    index = indexed[0].index
    columns = list(indexed[0].columns)
    for frame in indexed[1:]:
        if not frame.index.equals(index):
            raise ValueError("Fold predictions do not contain the same samples")
        if list(frame.columns) != columns:
            raise ValueError("Fold predictions do not contain the same columns")
    numeric = [frame.apply(pd.to_numeric, errors="raise") for frame in indexed]
    mean = sum(numeric) / len(numeric)
    return mean.reset_index()


def _predict_fold(
    checkpoint: Path,
    vcf: str,
    output: Path,
    args: argparse.Namespace,
) -> None:
    config = checkpoint.parent / "config.yaml"
    argv = [
        "aquila_predict.py",
        "--checkpoint",
        str(checkpoint),
        "--vcf",
        vcf,
        "-o",
        str(output),
    ]
    if config.is_file():
        argv.extend(["--config", str(config)])
    if args.encoding_type:
        argv.extend(["--encoding-type", args.encoding_type])
    if args.id_prefix:
        argv.extend(["--id-prefix", args.id_prefix])
    if args.normalize_heterozygous_order:
        argv.append("--normalize-heterozygous-order")
    if args.device:
        argv.extend(["--device", args.device])
    if args.batch_size is not None:
        argv.extend(["--batch-size", str(args.batch_size)])
    previous = sys.argv
    try:
        sys.argv = argv
        predict_main()
    finally:
        sys.argv = previous


def predict_model_dir(
    model_dir: Path,
    vcf: str,
    args: argparse.Namespace,
    fold_output_dir: Path | None,
) -> pd.DataFrame:
    """Score every fold checkpoint and return the mean prediction table."""
    checkpoints = discover_fold_checkpoints(model_dir)
    print(f"[ensemble] {model_dir}: {len(checkpoints)} folds")
    frames = []
    with tempfile.TemporaryDirectory(prefix="aquila_predict_ensemble_") as temporary:
        temporary_dir = Path(temporary)
        for checkpoint in checkpoints:
            fold_name = checkpoint.parent.name
            output = temporary_dir / f"{fold_name}.tsv"
            print(f"[ensemble] {fold_name}: {checkpoint}")
            _predict_fold(checkpoint, vcf, output, args)
            frame = pd.read_csv(output, sep="\t")
            frames.append(frame)
            if fold_output_dir is not None:
                destination = fold_output_dir / f"{model_dir.name}.{fold_name}.tsv"
                destination.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(destination, sep="\t", index=False, float_format="%.6f")
    return mean_prediction_frames(frames)


def main() -> None:
    args = parse_args()
    fold_output_dir = (
        Path(args.keep_fold_predictions) if args.keep_fold_predictions else None
    )
    predictions = predict_model_dir(
        Path(args.model_dir), args.vcf, args, fold_output_dir
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, sep="\t", index=False, float_format="%.6f")
    print(
        f"[ensemble] wrote {len(predictions)} samples and "
        f"{len(predictions.columns) - 1} prediction columns to {output}"
    )


if __name__ == "__main__":
    main()
