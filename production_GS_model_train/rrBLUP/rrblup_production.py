#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/cran/rrBLUP

"""Fit production rrBLUP models on all observed reference samples.

The prepared directory must come from aquila_data_cv_production.py. The
configuration must define exactly one parameter combination. No validation or
test split is used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aquila.encoding import parse_genotype_file  # noqa: E402
from benchmark.r_models_common.nested_cv import (  # noqa: E402
    ModelSpec,
    RWorker,
    SplitData,
    derive_seed,
    expand_grid,
    impute_from_training,
    write_json,
)

NUCLEOTIDES = ("A", "C", "G", "T")
WORKER = (
    PROJECT_ROOT
    / "benchmark"
    / "rrBLUP"
    / "src_benchmark"
    / "rrblup_worker.R"
)
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "production.yaml"


def dosages_from_diploid_onehot(
    features: np.ndarray | torch.Tensor,
    alt_allele_index: Sequence[int],
) -> np.ndarray:
    """Return VCF ALT-allele dosages, matching the benchmark rrBLUP encoding.

    Channel order matches aquila diploid one-hot: A, C, G, T for each allele.
    Missing calls (all-zero one-hot) stay NaN.
    """
    if isinstance(features, torch.Tensor):
        array = features.detach().cpu().numpy()
    else:
        array = np.asarray(features)
    if array.ndim != 3 or array.shape[2] != 8:
        raise ValueError("Expected diploid one-hot features [samples, markers, 8]")
    values = np.asarray(array, dtype=np.float64)
    alt = np.asarray(alt_allele_index, dtype=np.int64)
    if alt.shape != (values.shape[1],):
        raise ValueError("ALT allele indices must match the marker dimension")
    if np.any((alt < 0) | (alt >= len(NUCLEOTIDES))):
        raise ValueError("ALT allele indices must be nucleotide channels 0..3")
    markers = np.arange(values.shape[1])
    dosage = values[:, markers, alt] + values[:, markers, alt + 4]
    dosage = np.array(dosage, dtype=np.float64, copy=True)
    dosage[values.sum(axis=2) <= 0] = np.nan
    return dosage


def _alt_alleles(
    genotype_file: str,
    expected_variant_ids: Sequence[str],
    id_prefix: str | None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Align usable nucleotide SNPs to the prepared marker order.

    Star and other non-nucleotide alleles cannot be represented by Aquila's
    diploid one-hot channels; exclude their markers from rrBLUP entirely.
    """
    parsed = parse_genotype_file(
        genotype_file,
        encoding_type="diploid_onehot",
        variant_type="snp",
        id_prefix=id_prefix,
    )
    if not isinstance(parsed, dict):
        raise ValueError("Expected structured SNP parsing output")
    marker_ids = [str(marker) for marker in parsed["variant_ids"]]
    refs = [str(allele).upper() for allele in parsed["refs"]]
    alts = [str(allele).upper() for allele in parsed["alts"]]
    by_marker = dict(zip(marker_ids, zip(refs, alts)))
    missing = [marker for marker in expected_variant_ids if marker not in by_marker]
    if missing:
        raise ValueError(
            f"{len(missing)} prepared markers are absent from the source VCF"
        )
    aligned = [by_marker[marker] for marker in expected_variant_ids]
    keep = np.asarray(
        [ref in NUCLEOTIDES and alt in NUCLEOTIDES and ref != alt
         for ref, alt in aligned],
        dtype=bool,
    )
    if not keep.any():
        raise ValueError("No biallelic A/C/G/T SNP markers remain for rrBLUP")
    if not keep.all():
        excluded = sorted({(ref, alt) for (ref, alt), valid in zip(aligned, keep) if not valid})
        print(
            f"[INFO] Excluding {np.count_nonzero(~keep)} non-nucleotide or "
            f"non-biallelic markers from rrBLUP (REF/ALT: {excluded[:5]})",
            flush=True,
        )
    retained_alts = [alt for (_, alt), valid in zip(aligned, keep) if valid]
    indices = np.asarray(
        [NUCLEOTIDES.index(allele) for allele in retained_alts], dtype=np.int64
    )
    return retained_alts, indices, keep


def _marker_ids(metadata: Mapping[str, Any]) -> list[str]:
    """Return the single diploid-onehot marker list stored in metadata."""
    variant_ids = metadata.get("variant_ids")
    if isinstance(variant_ids, list):
        return [str(marker_id) for marker_id in variant_ids]
    if isinstance(variant_ids, dict):
        if set(variant_ids) != {"main"}:
            raise ValueError(
                "rrBLUP production training expects one diploid_onehot marker "
                f"branch, found {sorted(variant_ids)}"
            )
        return [str(marker_id) for marker_id in variant_ids["main"]]
    raise ValueError("Prepared metadata has no variant_ids")


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected a tensor at {path}")
    return tensor


def _observed_split(
    dosage: np.ndarray,
    sample_ids: Sequence[str],
    indices: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    trait_index: int,
    variants: tuple[Any, ...],
) -> SplitData:
    if len(indices) != len(processed):
        raise ValueError("Processed targets do not align with fold indices")
    positions = []
    discarded = []
    for position, index in enumerate(indices):
        sample_index = int(index)
        if bool(mask[sample_index, trait_index]):
            positions.append(position)
        else:
            discarded.append(str(sample_ids[sample_index]))
    if len(positions) < 3:
        raise ValueError("Trait has fewer than three observed samples in a split")
    targets = np.asarray(processed[positions, trait_index], dtype=np.float64)
    if not np.isfinite(targets).all() or np.any(targets == -999):
        raise ValueError("Missing phenotype sentinel entered rrBLUP targets")
    observed_indices = indices[np.asarray(positions, dtype=np.int64)]
    return SplitData(
        dosage[observed_indices],
        targets,
        tuple(str(sample_ids[int(index)]) for index in observed_indices),
        tuple(discarded),
        variants,
    )


def _fit_full(
    worker: RWorker,
    work_dir: Path,
    train: SplitData,
    parameters: Mapping[str, Any],
    seed: int,
) -> None:
    train_x, predict_x, _ = impute_from_training(train, train)
    worker.run(
        work_dir,
        train_x,
        train.targets,
        predict_x,
        parameters,
        seed,
    )


def run_trait(
    *,
    worker: RWorker,
    dosage: np.ndarray,
    sample_ids: Sequence[str],
    mask: np.ndarray,
    data_dir: Path,
    output_dir: Path,
    trait: str,
    trait_index: int,
    parameters: Mapping[str, Any],
    base_seed: int,
    variants: tuple[Any, ...],
) -> dict[str, Any]:
    started = time.time()
    trait_dir = output_dir / trait
    trait_dir.mkdir(parents=True, exist_ok=True)
    all_indices = np.arange(len(sample_ids), dtype=np.int64)
    full_path = data_dir / "full"
    full_targets = _load_tensor(full_path / "Y_processed.pt").numpy()
    train = _observed_split(
        dosage,
        sample_ids,
        all_indices,
        full_targets,
        mask,
        trait_index,
        variants,
    )
    full_dir = trait_dir / "full"
    # The worker always scores predict_x. Reusing the training matrix only
    # satisfies that interface; those rows stay in-sample and are not metrics.
    _fit_full(
        worker,
        full_dir,
        train,
        parameters,
        derive_seed(base_seed, trait, 0, 0, 999),
    )
    insample_predictions = full_dir / "predictions.tsv"
    if insample_predictions.exists():
        insample_predictions.rename(full_dir / "insample_predictions.tsv")
    write_json(
        full_dir / "fit.json",
        {
            "evaluation_set": "none",
            "n_train_observed": len(train.sample_ids),
            "parameters": dict(parameters),
        },
    )
    shutil.copy2(full_path / "preprocessing.json", full_dir / "preprocessing.json")
    write_json(trait_dir / "selected_parameters.json", dict(parameters))
    return {
        "trait": trait,
        "parameters": dict(parameters),
        "n_train_observed": len(train.sample_ids),
        "model": str((full_dir / "model.rds").resolve()),
        "elapsed_seconds": time.time() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit one rrBLUP model per trait on the full prepared dataset."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--traits", nargs="+", default=None)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Prepared data directory not found: {data_dir}")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("data_mode") != "deployment":
        raise ValueError(
            "rrBLUP production training requires data prepared by "
            "aquila_data_cv_production.py"
        )
    if metadata.get("encoding_type") != "diploid_onehot":
        raise ValueError("rrBLUP production training expects diploid_onehot features")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    sample_ids = [str(sample_id) for sample_id in metadata["sample_ids"]]
    trait_names = list(metadata["trait_names"])
    regression_traits = list(metadata.get("regression_tasks") or trait_names)
    traits = args.traits or regression_traits
    unknown = [trait for trait in traits if trait not in trait_names]
    if unknown:
        raise ValueError(f"Unknown traits: {unknown}")
    non_regression = [trait for trait in traits if trait not in regression_traits]
    if non_regression:
        raise ValueError(f"rrBLUP only supports regression traits: {non_regression}")

    features = _load_tensor(data_dir / "X.pt")
    variants = tuple(_marker_ids(metadata))
    if tuple(features.shape) != (len(sample_ids), len(variants), 8):
        raise ValueError(
            "X.pt shape "
            f"{tuple(features.shape)} does not match "
            f"{len(sample_ids)} samples and {len(variants)} markers"
        )
    alt_alleles, alt_indices, keep = _alt_alleles(
        str(metadata["genotype_file"]),
        variants,
        metadata.get("id_prefix"),
    )
    variants = tuple(marker for marker, valid in zip(variants, keep) if valid)
    dosage = dosages_from_diploid_onehot(features[:, keep, :], alt_indices)
    mask = _load_tensor(data_dir / "Y_mask.pt").bool().numpy()
    if mask.shape != (len(sample_ids), len(trait_names)):
        raise ValueError("Y_mask.pt does not match prepared samples and traits")
    spec = ModelSpec(
        "rrBLUP",
        "https://github.com/cran/rrBLUP",
        WORKER,
        ("jsonlite", "rrBLUP"),
        ("method",),
    )
    candidates = expand_grid(spec, config)
    if len(candidates) != 1:
        raise ValueError(
            "rrBLUP production training requires exactly one fixed parameter "
            f"combination, found {len(candidates)}"
        )
    parameters = candidates[0]
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    worker = RWorker(spec, args.rscript)
    worker.check_environment()
    write_json(
        output_dir / "dosage_coding.json",
        {
            "coding": "vcf_alt_dosage_0_1_2",
            "nucleotides": list(NUCLEOTIDES),
            "variant_ids": list(variants),
            "alt_allele_index": alt_indices.tolist(),
            "alt_allele": alt_alleles,
        },
    )
    shutil.copy2(Path(args.config).resolve(), output_dir / Path(args.config).name)
    results = []
    for trait in traits:
        trait_index = trait_names.index(trait)
        print(
            f"[INFO] rrBLUP trait={trait} mode=full-data",
            flush=True,
        )
        results.append(
            run_trait(
                worker=worker,
                dosage=dosage,
                sample_ids=sample_ids,
                mask=mask,
                data_dir=data_dir,
                output_dir=output_dir,
                trait=trait,
                trait_index=trait_index,
                parameters=parameters,
                base_seed=seed,
                variants=variants,
            )
        )
    write_json(
        output_dir / "summary.json",
        {
            "status": "completed",
            "model": "rrBLUP",
            "model_type": "production",
            "evaluation_set": "none",
            "data_dir": str(data_dir),
            "config": str(Path(args.config).resolve()),
            "sample_count": len(sample_ids),
            "seed": seed,
            "traits": results,
            "dosage_coding": str((output_dir / "dosage_coding.json").resolve()),
        },
    )
    print(
        f"Completed rrBLUP full-data fit for {len(results)} trait(s); "
        f"summary: {output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
