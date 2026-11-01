#!/usr/bin/env python3
"""Integrated gradients of a 10-fold ensemble prediction.

The attributed function is the mean of the fold models after each fold's
own preprocessing is inverted. Gradients are taken of that mean, so the
saved HDF5 matches ``aquila_ig.py`` and can be ranked by
``aquila_ig_interpretation.py``.

Expected layout::

    model_dir/fold_0/best_model.pt
    model_dir/fold_1/best_model.pt
    ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import h5py
import torch
import yaml

from aquila.scripts.aquila_ig import (
    IntegratedGradients,
    compute_ig_in_memory,
    compute_ig_streaming,
    load_model_and_data,
    save_results_h5,
)
from aquila.utils import load_config
from aquila.varnn import create_model_from_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute integrated gradients for the mean of inverse-transformed "
            "fold predictions"
        )
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing fold_<k>/best_model.pt checkpoints",
    )
    parser.add_argument(
        "--vcf",
        type=str,
        required=True,
        help="Path to the VCF aligned to the training markers",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="ig_ensemble_output",
        help="Output directory for integrated gradients results",
    )
    parser.add_argument(
        "--encoding-type",
        type=str,
        default="diploid_onehot",
        choices=[
            "token",
            "diploid_onehot",
            "onehot",
            "snp_vcf",
            "indel_vcf",
            "sv_vcf",
            "snp_indel_vcf",
            "snp_indel_sv_vcf",
        ],
        help="Encoding type (must match training)",
    )
    parser.add_argument(
        "--variant-type",
        type=str,
        default=None,
        choices=["snp", "indel", "sv"],
        help="Encode every kept VCF record as this type",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default=None,
        help="Keep VCF IDs with this prefix, for example 'SNP-'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of interpolation steps (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Kept for compatibility with aquila_ig.py; IG uses one sample",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name to attribute. Default: every regression task",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        choices=["regression", "classification", None],
        help="Task type. Fold ensembles are regression models",
    )
    parser.add_argument(
        "--samples-set",
        type=str,
        default=None,
        help="Text file of sample IDs to attribute. Default: all VCF samples",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Write each sample to HDF5 immediately",
    )
    return parser.parse_args()


def discover_fold_checkpoints(model_dir: Path | str) -> List[Path]:
    """Return ``fold_<k>/best_model.pt`` paths in fold order."""
    root = Path(model_dir)
    found = []
    if not root.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {root}")
    for path in root.glob("fold_*/best_model.pt"):
        suffix = path.parent.name.split("_", 1)[1]
        if suffix.isdigit():
            found.append((int(suffix), path))
    found.sort()
    if not found:
        raise FileNotFoundError(
            f"No fold checkpoints in {root}. Expected fold_<k>/best_model.pt. "
            "A cv-refit directory only keeps the full-data refit, so train "
            "the locked folds first."
        )
    fold_ids = [fold_id for fold_id, _ in found]
    expected = list(range(fold_ids[0], fold_ids[0] + len(fold_ids)))
    if fold_ids != expected:
        raise FileNotFoundError(
            f"Fold checkpoints in {root} are not contiguous: {fold_ids}"
        )
    if fold_ids[0] != 0:
        raise FileNotFoundError(
            f"Fold checkpoints in {root} must start at fold_0, found {fold_ids}"
        )
    return [path for _, path in found]


def regression_inverse_specs(
    preprocessing: Mapping | None,
    task_names: Sequence[str],
) -> List[Mapping[str, float | bool | str]]:
    """Return one inverse-transform spec per regression output, in head order."""
    traits = []
    if isinstance(preprocessing, Mapping):
        raw_traits = preprocessing.get("traits") or []
        traits = [
            trait
            for trait in raw_traits
            if str(trait.get("task", "regression")) == "regression"
        ]
    by_name = {str(trait["name"]): trait for trait in traits}
    specs = []
    missing = [name for name in task_names if name not in by_name]
    if missing and traits:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"Preprocessor is missing {len(missing)} regression tasks "
            f"(for example: {preview})"
        )
    for name in task_names:
        trait = by_name.get(str(name))
        if trait is None:
            specs.append(
                {
                    "name": str(name),
                    "std": 1.0,
                    "mean": 0.0,
                    "use_log1p": False,
                    "log_shift": 0.0,
                }
            )
            continue
        specs.append(
            {
                "name": str(name),
                "std": float(trait["std"]),
                "mean": float(trait["mean"]),
                "use_log1p": bool(trait.get("use_log1p", False)),
                "log_shift": float(trait.get("log_shift") or 0.0),
            }
        )
    return specs


def inverse_regression(
    normalized: torch.Tensor,
    specs: Sequence[Mapping[str, float | bool | str]],
) -> torch.Tensor:
    """Map one fold's normalized head output back to the phenotype scale."""
    if normalized.shape[-1] != len(specs):
        raise ValueError(
            f"Regression outputs ({normalized.shape[-1]}) do not match "
            f"inverse specs ({len(specs)})"
        )
    columns = []
    for index, spec in enumerate(specs):
        restored = normalized[:, index] * float(spec["std"]) + float(spec["mean"])
        if spec["use_log1p"]:
            restored = torch.expm1(restored) - float(spec["log_shift"])
        columns.append(restored)
    return torch.stack(columns, dim=1)


class FoldEnsemble(torch.nn.Module):
    """Mean of fold predictions on the original phenotype scale."""

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        specs: Sequence[Sequence[Mapping]],
    ) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError("An ensemble needs at least two fold models")
        if len(models) != len(specs):
            raise ValueError("Each fold model needs its own inverse-transform spec")
        self.models = torch.nn.ModuleList(models)
        self.specs = list(specs)

    def forward(self, inputs):
        restored = []
        for model, specs in zip(self.models, self.specs):
            outputs = model(inputs)
            restored.append(inverse_regression(outputs["regression"], specs))
        stacked = torch.stack(restored, dim=0)
        return {"regression": stacked.mean(dim=0)}


def _state_dict(checkpoint: Mapping) -> Dict[str, torch.Tensor]:
    raw = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    cleaned = {}
    for key, value in raw.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def _ensure_fold_config(checkpoint_path: Path, checkpoint: Mapping) -> Path:
    config_path = checkpoint_path.parent / "config.yaml"
    if config_path.is_file():
        return config_path
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise FileNotFoundError(
            f"{checkpoint_path.parent} has no config.yaml and the checkpoint "
            "does not embed one"
        )
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    print(f"Wrote {config_path} from the checkpoint config")
    return config_path


def _load_additional_model(
    checkpoint_path: Path,
    encoding_type: str,
    seq_length: Mapping[str, int],
    regression_tasks: Sequence[str],
    classification_tasks: Sequence[str],
    reference_variants,
    device: str,
) -> tuple[torch.nn.Module, List[Mapping]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata") or {}
    variant_ids = metadata.get("variant_ids")
    if reference_variants is not None and variant_ids != reference_variants:
        raise ValueError(
            f"{checkpoint_path} variant order does not match the first fold"
        )
    checkpoint_tasks = list((checkpoint.get("config") or {}).get("data", {}).get("regression_tasks") or [])
    if checkpoint_tasks and list(checkpoint_tasks) != list(regression_tasks):
        raise ValueError(f"{checkpoint_path} regression tasks do not match the first fold")
    # Per-fold HPO can pick a different kernel size or width for each fold.
    config_path = _ensure_fold_config(checkpoint_path, checkpoint)
    config = load_config(str(config_path))
    config.setdefault("data", {})
    config["data"]["encoding_type"] = encoding_type
    model = create_model_from_config(
        config=config,
        seq_length=dict(seq_length),
        regression_tasks=list(regression_tasks),
        classification_tasks=list(classification_tasks),
    )
    model.load_state_dict(_state_dict(checkpoint))
    model = model.to(device)
    model.eval()
    specs = regression_inverse_specs(checkpoint.get("preprocessing"), regression_tasks)
    return model, specs


def load_ensemble(args):
    """Load every fold and the shared genotype matrix."""
    checkpoints = discover_fold_checkpoints(args.model_dir)
    print(f"Ensemble folds: {len(checkpoints)}")
    for path in checkpoints:
        print(f"  {path}")

    first_checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    _ensure_fold_config(checkpoints[0], first_checkpoint)
    reference_variants = (first_checkpoint.get("metadata") or {}).get("variant_ids")
    first_dir = str(checkpoints[0].parent)
    args.model_dir = first_dir
    (
        first_model,
        data_dict,
        sample_ids,
        snp_ids,
        task_info,
        is_multi_branch,
    ) = load_model_and_data(args)
    if task_info["task_type"] != "regression":
        raise ValueError("Fold ensemble IG currently supports regression tasks")

    seq_length = {name: int(values.shape[1]) for name, values in data_dict.items()}
    regression_tasks = list(task_info["regression_tasks"])
    classification_tasks = list(task_info["classification_tasks"])
    specs = [
        regression_inverse_specs(
            first_checkpoint.get("preprocessing"),
            regression_tasks,
        )
    ]
    models = [first_model]
    for checkpoint_path in checkpoints[1:]:
        model, model_specs = _load_additional_model(
            checkpoint_path,
            args.encoding_type,
            seq_length,
            regression_tasks,
            classification_tasks,
            reference_variants,
            args.device,
        )
        models.append(model)
        specs.append(model_specs)

    ensemble = FoldEnsemble(models, specs)
    ensemble = ensemble.to(args.device)
    ensemble.eval()
    task_info = dict(task_info)
    task_info["ensemble_checkpoints"] = [str(path) for path in checkpoints]
    task_info["n_models"] = len(models)
    return ensemble, data_dict, sample_ids, snp_ids, task_info, is_multi_branch


def _write_ensemble_metadata(output_dir: Path, task_info: Mapping, summary: Mapping) -> None:
    output_dir = Path(output_dir)
    summary_path = output_dir / "ig_summary.json"
    payload = dict(summary)
    payload["ensemble"] = "fold_mean_inverse"
    payload["n_models"] = int(task_info["n_models"])
    payload["model_checkpoints"] = list(task_info["ensemble_checkpoints"])
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    h5_path = output_dir / "ig_results.h5"
    with h5py.File(h5_path, "a") as handle:
        handle.attrs["ensemble"] = "fold_mean_inverse"
        handle.attrs["n_models"] = int(task_info["n_models"])
    print(f"  Saved ensemble metadata: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.task_type == "classification":
        raise ValueError("Fold ensemble IG currently supports regression tasks")
    (
        ensemble,
        data_dict,
        sample_ids,
        snp_ids,
        task_info,
        is_multi_branch,
    ) = load_ensemble(args)
    print(
        f"\nAttributing the mean of {task_info['n_models']} "
        "inverse-transformed fold predictions"
    )

    if args.streaming:
        compute_ig_streaming(
            ensemble,
            data_dict,
            sample_ids,
            snp_ids,
            task_info,
            is_multi_branch,
            args,
        )
        computed_tasks = [task_info["task_names"][index] for index in task_info["task_indices"]]
        summary = {
            "num_samples": len(sample_ids),
            "sample_ids": sample_ids,
            "num_tasks": len(computed_tasks),
            "task_names": computed_tasks,
            "task_type": task_info["task_type"],
            "variant_types": list(snp_ids.keys()),
            "num_variants": {name: len(ids) for name, ids in snp_ids.items()},
            "num_steps": args.num_steps,
        }
    else:
        results = compute_ig_in_memory(
            ensemble,
            data_dict,
            sample_ids,
            snp_ids,
            task_info,
            is_multi_branch,
            args,
        )
        save_results_h5(
            results,
            sample_ids,
            snp_ids,
            task_info,
            args,
            args.output,
        )
        computed_tasks = [task_info["task_names"][index] for index in task_info["task_indices"]]
        summary = {
            "num_samples": len(sample_ids),
            "sample_ids": sample_ids,
            "num_tasks": len(computed_tasks),
            "task_names": computed_tasks,
            "task_type": task_info["task_type"],
            "variant_types": list(snp_ids.keys()),
            "num_variants": {name: len(ids) for name, ids in snp_ids.items()},
            "num_steps": args.num_steps,
        }
    _write_ensemble_metadata(Path(args.output), task_info, summary)
    print("\n" + "=" * 80)
    print("Ensemble Integrated Gradients Computation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
