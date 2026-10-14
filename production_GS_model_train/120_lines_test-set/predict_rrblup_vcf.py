#!/usr/bin/env python3
"""Predict phenotypes from a VCF with a fitted production rrBLUP model.

The script is trait-general. It reads ``dosage_coding.json`` and each
``<trait>/full/model.rds`` under ``--model-dir``, encodes the VCF with the
VCF ALT-allele dosage used in training, mean-imputes missing dosages from that
trait's training matrix, and returns predictions on the original scale.

Example:
    python predict_rrblup_vcf.py \\
        --vcf new_samples.vcf.gz \\
        --model-dir results/655rice.r2_0.045.rrblup \\
        --traits GYP_BLUP HD_BLUP \\
        -o predictions.tsv
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from aquila.encoding import parse_genotype_file


def _load_effects(model_path, rscript):
    """Read the intercept and marker effects stored by rrblup_worker.R."""
    with tempfile.TemporaryDirectory() as tmp:
        effects_path = Path(tmp) / "effects.txt"
        script = (
            "fit <- readRDS(commandArgs(trailingOnly=TRUE)[[1]])\n"
            "cat(sprintf('%.17g\\n', as.numeric(fit$beta)[[1]]))\n"
            "write.table(as.numeric(fit$marker_effects), "
            "commandArgs(trailingOnly=TRUE)[[2]], "
            "row.names=FALSE, col.names=FALSE)\n"
        )
        completed = subprocess.run(
            [rscript, "-e", script, str(model_path), str(effects_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to read {model_path}\n{completed.stderr.strip()}"
            )
        beta = float(completed.stdout.strip().splitlines()[-1])
        effects = np.loadtxt(effects_path, dtype=np.float64)
    return beta, np.atleast_1d(effects)


def _dosages_from_vcf(vcf_path, coding, id_prefix):
    """Encode VCF ALT dosages, aligned to the training marker order."""
    encoded = parse_genotype_file(
        vcf_path,
        encoding_type="diploid_onehot",
        variant_type="snp",
        id_prefix=id_prefix,
    )
    if isinstance(encoded, dict):
        matrix = encoded["matrix"]
        sample_ids = [str(sample) for sample in encoded["sample_ids"]]
        marker_ids = [str(marker) for marker in encoded["variant_ids"]]
    else:
        matrix, sample_ids, marker_ids = encoded
        sample_ids = [str(sample) for sample in sample_ids]
        marker_ids = [str(marker) for marker in marker_ids]

    expected = [str(marker) for marker in coding["variant_ids"]]
    index = {marker: position for position, marker in enumerate(marker_ids)}
    missing = [marker for marker in expected if marker not in index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} training markers are missing from {vcf_path} "
            f"(examples: {preview})"
        )
    order = [index[marker] for marker in expected]
    values = np.asarray(matrix[:, order, :], dtype=np.float64)
    alt = np.asarray(coding["alt_allele_index"], dtype=np.int64)
    if values.shape[1] != len(alt) or values.shape[2] != 8:
        raise ValueError(
            f"Encoded genotype shape {values.shape} does not match "
            f"{len(alt)} training markers"
        )
    markers = np.arange(values.shape[1])
    dosage = values[:, markers, alt] + values[:, markers, alt + 4]
    dosage[values.sum(axis=2) <= 0] = np.nan
    return sample_ids, dosage


def _inverse_transform(values, preprocessing, trait):
    traits = preprocessing["traits"]
    matches = [item for item in traits if item["name"] == trait]
    if len(matches) != 1:
        raise ValueError(f"{trait} is missing from preprocessing.json")
    spec = matches[0]
    restored = np.asarray(values, dtype=np.float64) * float(spec["std"])
    restored += float(spec["mean"])
    if spec["use_log1p"]:
        restored = np.expm1(restored) - float(spec["log_shift"])
    return restored


def _training_means(train_x_path, n_markers):
    train_x = np.loadtxt(train_x_path, dtype=np.float64)
    if train_x.ndim != 2 or train_x.shape[1] != n_markers:
        raise ValueError(
            f"{train_x_path} has shape {getattr(train_x, 'shape', None)}; "
            f"expected (*, {n_markers})"
        )
    return train_x.mean(axis=0)


def predict(vcf_path, model_dir, traits, rscript, id_prefix):
    model_dir = Path(model_dir)
    coding_path = model_dir / "dosage_coding.json"
    coding = json.loads(coding_path.read_text(encoding="utf-8"))
    if id_prefix is None:
        variant_ids = [str(marker) for marker in coding["variant_ids"]]
        if variant_ids and all(marker.startswith("SNP-") for marker in variant_ids):
            id_prefix = "SNP-"
    print(f"[VCF] {vcf_path}")
    sample_ids, dosage = _dosages_from_vcf(vcf_path, coding, id_prefix)
    print(f"[VCF] {len(sample_ids)} samples, {dosage.shape[1]} markers")

    if not traits:
        traits = sorted(
            path.name
            for path in model_dir.iterdir()
            if (path / "full" / "model.rds").is_file()
        )
    columns = {"Sample_ID": sample_ids}
    for trait in traits:
        full_dir = model_dir / trait / "full"
        model_path = full_dir / "model.rds"
        if not model_path.is_file():
            raise FileNotFoundError(f"No rrBLUP model for {trait}: {model_path}")
        beta, effects = _load_effects(model_path, rscript)
        if len(effects) != dosage.shape[1]:
            raise ValueError(
                f"{trait} has {len(effects)} marker effects for "
                f"{dosage.shape[1]} dosages"
            )
        means = _training_means(full_dir / "train_x.tsv", dosage.shape[1])
        filled = np.where(np.isnan(dosage), means, dosage)
        normalized = beta + filled @ effects
        preprocessing = json.loads(
            (full_dir / "preprocessing.json").read_text(encoding="utf-8")
        )
        columns[trait] = _inverse_transform(normalized, preprocessing, trait)
        print(f"[PRED] {trait}: mean={columns[trait].mean():.4g}")
    return pd.DataFrame(columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True, help="Genotype VCF to score")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Production rrBLUP directory containing dosage_coding.json",
    )
    parser.add_argument(
        "--traits",
        nargs="*",
        default=None,
        help="Trait names to predict. Default: every trait with full/model.rds",
    )
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument(
        "--id-prefix",
        default=None,
        help="VCF ID prefix to keep. Default: SNP- when the model used SNP IDs.",
    )
    args = parser.parse_args()
    predictions = predict(
        args.vcf, args.model_dir, args.traits, args.rscript, args.id_prefix
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, sep="\t", index=False)
    print(f"[WRITE] {output}")


if __name__ == "__main__":
    main()
