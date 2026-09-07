#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Export DEM RF selections as split-local Aquila prepared data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parents[1]
for import_path in (
    PROJECT_ROOT / "src",
    SCRIPT_DIRECTORY / "src" / "biodem" / "utils",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from aquila.data.preprocessing import PerTraitPreprocessor
from data_ncv_benchmark import MODALITY_ORDER, Variant, load_vcf_branches


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DEM RF-selected markers for Aquila nested CV."
    )
    parser.add_argument("--data-dir", required=True, help="Source Aquila data.")
    parser.add_argument(
        "--dem-output-dir",
        required=True,
        help="DEM output containing TRAIT/fold_N/rf_selection.",
    )
    parser.add_argument("--trait", required=True)
    parser.add_argument("--outer-fold", required=True, type=int)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _safe_trait(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Trait name cannot form an output path: {value!r}")
    return safe


def _selection_path(
    dem_output: Path,
    trait: str,
    outer_fold: int,
    inner_fold: int | None,
) -> Path:
    root = dem_output / _safe_trait(trait) / f"fold_{outer_fold}" / "rf_selection"
    return root / (
        "final.json" if inner_fold is None else f"inner_fold_{inner_fold}.json"
    )


def _load_selection(path: Path, trait: str, outer_fold: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"DEM RF selection artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "dem_rf_marker_selection":
        raise ValueError(f"Unsupported DEM RF selection schema: {path}")
    if payload.get("rf_enabled") is not True:
        raise ValueError(f"DEM RF selection is not enabled: {path}")
    if payload.get("trait") != trait:
        raise ValueError(f"DEM RF selection trait mismatch: {path}")
    if int(payload.get("outer_fold", -1)) != outer_fold:
        raise ValueError(f"DEM RF selection outer-fold mismatch: {path}")
    return payload


def _variant_tuple(row: Mapping[str, Any]) -> Variant:
    return (
        str(row["chrom"]),
        str(row["position"]),
        str(row["id"]),
        str(row["ref"]),
        str(row["alt"]),
    )


def _source_schemas(
    metadata: Mapping[str, Any],
    features: torch.Tensor | Mapping[str, torch.Tensor],
) -> tuple[dict[str, tuple[Variant, ...]], dict[str, str]]:
    genotype_path = Path(str(metadata["genotype_file"]))
    if not genotype_path.is_absolute():
        source_directory = metadata.get("_source_data_directory")
        if source_directory is not None:
            source_directory = Path(str(source_directory))
            candidates = (
                source_directory / genotype_path,
                source_directory.parent / genotype_path,
                Path.cwd() / genotype_path,
            )
            genotype_path = next(
                (candidate for candidate in candidates if candidate.is_file()),
                candidates[0],
            )
    if not genotype_path.is_file():
        raise FileNotFoundError(f"Source genotype VCF not found: {genotype_path}")
    if isinstance(features, torch.Tensor):
        modalities = ("snp",)
        feature_keys = {"snp": "main"}
    else:
        normalized = {str(name).lower(): str(name) for name in features}
        modalities = tuple(name for name in MODALITY_ORDER if name in normalized)
        feature_keys = {name: normalized[name] for name in modalities}
    branches = load_vcf_branches(genotype_path, modalities)
    variant_ids = metadata.get("variant_ids", {})
    for modality in modalities:
        feature_key = feature_keys[modality]
        expected_ids = [variant[2] for variant in branches[modality].variants]
        recorded_ids = [
            str(value) for value in variant_ids.get(feature_key, [])
        ]
        if recorded_ids and recorded_ids != expected_ids:
            raise ValueError(
                f"Aquila X.pt marker order does not match source VCF for {modality}"
            )
        tensor = features if isinstance(features, torch.Tensor) else features[feature_key]
        if tensor.shape[1] != len(expected_ids):
            raise ValueError(
                f"Aquila feature length does not match source VCF for {modality}"
            )
    return (
        {name: branches[name].variants for name in modalities},
        feature_keys,
    )


def _map_selection(
    selection: Mapping[str, Any],
    schemas: Mapping[str, tuple[Variant, ...]],
    feature_keys: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected_indices: dict[str, np.ndarray] = {}
    branch_metadata: dict[str, Any] = {}
    branches = selection.get("branches")
    if not isinstance(branches, Mapping):
        raise ValueError("DEM RF selection branches must be an object")
    if set(branches) != set(schemas):
        raise ValueError(
            "DEM RF selection modalities do not match Aquila feature branches"
        )
    for modality, schema in schemas.items():
        positions = {variant: index for index, variant in enumerate(schema)}
        rows = branches[modality].get("variants", [])
        variants = [_variant_tuple(row) for row in rows]
        missing = [variant for variant in variants if variant not in positions]
        if missing:
            raise ValueError(
                f"DEM selected {modality} variant is absent from Aquila X.pt: "
                f"{missing[0]}"
            )
        indices = np.asarray([positions[variant] for variant in variants], dtype=np.int64)
        if len(indices) == 0 or len(np.unique(indices)) != len(indices):
            raise ValueError(f"DEM selected {modality} indices are empty or duplicated")
        selected_indices[feature_keys[modality]] = indices
        branch_metadata[feature_keys[modality]] = {
            "modality": modality,
            "selected_count": int(len(indices)),
            "source_count": int(len(schema)),
            "variants": rows,
        }
    return selected_indices, branch_metadata


def _slice_preprocessor(source: Path, destination: Path, trait: str) -> None:
    processor = PerTraitPreprocessor.load_json(source)
    selected = [item for item in processor.traits if item.name == trait]
    if len(selected) != 1:
        raise ValueError(f"Preprocessing cache does not contain trait {trait!r}")
    processor.traits = selected
    processor.save_json(destination)


def _prepare_single_trait_targets(
    source: Path,
    output: Path,
    metadata: dict[str, Any],
    trait_index: int,
) -> None:
    raw = _torch_load(source / "Y_raw.pt")
    mask = _torch_load(source / "Y_mask.pt")
    torch.save(raw[:, [trait_index]].contiguous(), output / "Y_raw.pt")
    torch.save(mask[:, [trait_index]].contiguous(), output / "Y_mask.pt")
    outer_count = int(metadata["outer_folds"])
    inner_count = int(metadata["inner_folds"])
    for outer_fold in range(outer_count):
        source_outer = source / "cv" / f"outer_fold_{outer_fold}"
        output_outer = output / "cv" / f"outer_fold_{outer_fold}"
        output_outer.mkdir(parents=True, exist_ok=True)
        for name in ("train_idx.npy", "test_idx.npy"):
            shutil.copy2(source_outer / name, output_outer / name)
        source_final = source_outer / "final"
        output_final = output_outer / "final"
        output_final.mkdir(parents=True, exist_ok=True)
        for role in ("train", "test"):
            values = _torch_load(source_final / f"Y_{role}_processed.pt")
            torch.save(
                values[:, [trait_index]].contiguous(),
                output_final / f"Y_{role}_processed.pt",
            )
        _slice_preprocessor(
            source_final / "preprocessing.json",
            output_final / "preprocessing.json",
            metadata["trait_names"][trait_index],
        )
        for inner_fold in range(inner_count):
            source_inner = source_outer / f"inner_fold_{inner_fold}"
            output_inner = output_outer / f"inner_fold_{inner_fold}"
            output_inner.mkdir(parents=True, exist_ok=True)
            for name in ("train_idx.npy", "valid_idx.npy"):
                shutil.copy2(source_inner / name, output_inner / name)
            for role in ("train", "valid"):
                values = _torch_load(source_inner / f"Y_{role}_processed.pt")
                torch.save(
                    values[:, [trait_index]].contiguous(),
                    output_inner / f"Y_{role}_processed.pt",
                )
            _slice_preprocessor(
                source_inner / "preprocessing.json",
                output_inner / "preprocessing.json",
                metadata["trait_names"][trait_index],
            )


def export_data(
    source: Path,
    dem_output: Path,
    trait: str,
    outer_fold: int,
    output: Path,
    overwrite: bool = False,
) -> Path:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}")
        shutil.rmtree(output)
    with (source / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["_source_data_directory"] = str(source.resolve())
    trait_names = [str(value) for value in metadata["trait_names"]]
    if trait not in trait_names:
        raise ValueError(f"Unknown trait {trait!r}; available: {trait_names}")
    if outer_fold < 0 or outer_fold >= int(metadata["outer_folds"]):
        raise ValueError(f"Outer fold is outside prepared data: {outer_fold}")
    trait_index = trait_names.index(trait)
    features = _torch_load(source / "X.pt")
    schemas, feature_keys = _source_schemas(metadata, features)

    output.mkdir(parents=True)
    _link_or_copy(source / "X.pt", output / "X.pt")
    shutil.copy2(source / "sample_fold_mapping.txt", output / "sample_fold_mapping.txt")
    _prepare_single_trait_targets(source, output, metadata, trait_index)

    inner_count = int(metadata["inner_folds"])
    selection_summary: dict[str, Any] = {"inner": {}, "final": None}
    for inner_fold in range(inner_count):
        selection = _load_selection(
            _selection_path(dem_output, trait, outer_fold, inner_fold),
            trait,
            outer_fold,
        )
        indices, branches = _map_selection(selection, schemas, feature_keys)
        marker_path = (
            output
            / "cv"
            / f"outer_fold_{outer_fold}"
            / f"inner_fold_{inner_fold}"
            / "marker_indices.npz"
        )
        np.savez(marker_path, **indices)
        artifact = {
            "schema": "aquila_split_local_marker_indices",
            "schema_version": 1,
            "trait": trait,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "selection_scope": "inner_train",
            "branches": branches,
            "dem_selection_file": str(
                _selection_path(dem_output, trait, outer_fold, inner_fold).resolve()
            ),
            "fit_sample_ids": selection["fit_sample_ids"],
            "discarded_missing_target_sample_ids": selection[
                "discarded_missing_target_sample_ids"
            ],
            "rf_parameters": selection["rf_parameters"],
        }
        _write_json(marker_path.with_suffix(".json"), artifact)
        selection_summary["inner"][str(inner_fold)] = artifact

    selection = _load_selection(
        _selection_path(dem_output, trait, outer_fold, None),
        trait,
        outer_fold,
    )
    indices, branches = _map_selection(selection, schemas, feature_keys)
    marker_path = (
        output
        / "cv"
        / f"outer_fold_{outer_fold}"
        / "final"
        / "marker_indices.npz"
    )
    np.savez(marker_path, **indices)
    final_artifact = {
        "schema": "aquila_split_local_marker_indices",
        "schema_version": 1,
        "trait": trait,
        "outer_fold": outer_fold,
        "selection_scope": "outer_train",
        "branches": branches,
        "dem_selection_file": str(
            _selection_path(dem_output, trait, outer_fold, None).resolve()
        ),
        "fit_sample_ids": selection["fit_sample_ids"],
        "discarded_missing_target_sample_ids": selection[
            "discarded_missing_target_sample_ids"
        ],
        "rf_parameters": selection["rf_parameters"],
    }
    _write_json(marker_path.with_suffix(".json"), final_artifact)
    selection_summary["final"] = final_artifact

    derived_metadata = dict(metadata)
    derived_metadata.pop("_source_data_directory", None)
    derived_metadata.update(
        {
            "schema_version": max(int(metadata.get("schema_version", 1)), 2),
            "trait_names": [trait],
            "regression_tasks": [trait],
            "classification_tasks": [],
            "trait_tasks": ["regression"],
            "n_traits": 1,
            "n_regression_tasks": 1,
            "n_classification_tasks": 0,
            "feature_storage": "split_local_marker_indices",
            "split_local_features": {
                "schema_version": 1,
                "selection_method": "DEM_random_forest",
                "trait": trait,
                "outer_fold": outer_fold,
                "source_data_dir": str(source.resolve()),
                "dem_output_dir": str(dem_output.resolve()),
            },
            "fold_specific_features": {
                "enabled": True,
                "outer_fold": outer_fold,
                "trait": trait,
                "selection_method": "DEM_random_forest",
            },
        }
    )
    _write_json(output / "metadata.json", derived_metadata)
    _write_json(output / "split_local_selection.json", selection_summary)
    return output


def main() -> None:
    args = parse_args()
    output = export_data(
        Path(args.data_dir),
        Path(args.dem_output_dir),
        args.trait,
        args.outer_fold,
        Path(args.output_dir),
        args.overwrite,
    )
    print(f"Exported DEM RF-selected Aquila data to {output}")


if __name__ == "__main__":
    main()
