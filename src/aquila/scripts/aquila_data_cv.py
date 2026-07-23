#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Prepare aligned Aquila tensors and deterministic nested CV folds."""

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from aquila.data.cv import (
    generate_nested_folds,
    generate_nested_folds_from_assignments,
    save_nested_folds,
    validate_outer_fold_observations,
)
from aquila.data.preprocessing import PerTraitPreprocessor
from aquila.encoding import parse_genotype_file


@dataclass(frozen=True)
class PreparationConfig:
    """Configuration for preparing one nested-CV dataset."""

    genotype_file: Path
    phenotype_file: Path
    output_directory: Path
    encoding_type: str = "diploid_onehot"
    variant_type: str | None = None
    sample_id_column: str | None = None
    traits: Sequence[str] | None = None
    classification_tasks: Sequence[str] | None = None
    missing_sentinel: float = -999.0
    outer_folds: int = 5
    inner_folds: int = 4
    seed: int = 42
    fold_mapping: Path | None = None
    min_observed_per_fold: int = 10
    skew_threshold: float = 2.0
    preprocessing_epsilon: float = 1e-8
    overwrite: bool = False


@dataclass(frozen=True)
class EncodedGenotypes:
    """Normalized result from the existing genotype parser."""

    features: torch.Tensor | Dict[str, torch.Tensor]
    sample_ids: List[str]
    feature_shapes: Dict[str, List[int]]
    variant_ids: Dict[str, List[str]]


class NestedCVDataPreparer:
    """Align source data and persist raw targets and nested fold indices."""

    _ARTIFACTS = (
        "X.pt",
        "Y_raw.pt",
        "Y_mask.pt",
        "metadata.json",
        "sample_fold_mapping.txt",
        "cv",
    )

    def __init__(self, config: PreparationConfig) -> None:
        self.config = config

    def prepare(self) -> Dict[str, Any]:
        self._validate_inputs()
        self._prepare_output_directory()

        parsed = parse_genotype_file(
            str(self.config.genotype_file),
            encoding_type=self.config.encoding_type,
            variant_type=self.config.variant_type,
        )
        genotypes = self._normalize_genotypes(parsed)
        phenotype = self._read_phenotypes()
        aligned = self._align(genotypes, phenotype)

        if self.config.fold_mapping is None:
            folds = generate_nested_folds(
                n_samples=len(aligned["sample_ids"]),
                outer_folds=self.config.outer_folds,
                inner_folds=self.config.inner_folds,
                seed=self.config.seed,
            )
        else:
            assignments = self._load_outer_fold_mapping(aligned["sample_ids"])
            folds = generate_nested_folds_from_assignments(
                assignments,
                outer_folds=self.config.outer_folds,
                inner_folds=self.config.inner_folds,
                seed=self.config.seed,
            )
        observed_counts = validate_outer_fold_observations(
            aligned["target_mask"].numpy(),
            folds,
            phenotype["trait_names"],
            min_observed=self.config.min_observed_per_fold,
        )
        metadata = self._build_metadata(genotypes, phenotype, aligned)
        metadata["fold_mapping_file"] = (
            str(self.config.fold_mapping.resolve())
            if self.config.fold_mapping is not None
            else None
        )
        metadata["min_observed_per_outer_fold"] = (
            self.config.min_observed_per_fold
        )
        metadata["outer_fold_observed_counts"] = observed_counts
        metadata["preprocessing"] = {
            "skew_threshold": self.config.skew_threshold,
            "epsilon": self.config.preprocessing_epsilon,
            "fit_scope": "inner_train_and_outer_train_only",
        }
        self._save_artifacts(aligned, metadata, folds)
        return metadata

    def _validate_inputs(self) -> None:
        if not self.config.genotype_file.is_file():
            raise FileNotFoundError(
                f"Genotype file not found: {self.config.genotype_file}"
            )
        if not self.config.phenotype_file.is_file():
            raise FileNotFoundError(
                f"Phenotype file not found: {self.config.phenotype_file}"
            )
        if not math.isfinite(self.config.missing_sentinel):
            raise ValueError("missing_sentinel must be finite")
        if self.config.skew_threshold < 0:
            raise ValueError("skew_threshold must be nonnegative")
        if self.config.preprocessing_epsilon <= 0:
            raise ValueError("preprocessing_epsilon must be positive")
        if self.config.min_observed_per_fold < 10:
            raise ValueError("min_observed_per_fold must be at least 10")
        if (
            self.config.fold_mapping is not None
            and not self.config.fold_mapping.is_file()
        ):
            raise FileNotFoundError(
                f"Fold mapping file not found: {self.config.fold_mapping}"
            )

    def _prepare_output_directory(self) -> None:
        output = self.config.output_directory
        existing = [name for name in self._ARTIFACTS if (output / name).exists()]
        if existing and not self.config.overwrite:
            raise FileExistsError(
                "Output artifacts already exist; use --overwrite to replace them: "
                + ", ".join(existing)
            )
        output.mkdir(parents=True, exist_ok=True)
        if self.config.overwrite:
            for name in self._ARTIFACTS:
                path = output / name
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()

    def _normalize_genotypes(self, parsed: Any) -> EncodedGenotypes:
        if isinstance(parsed, tuple) and len(parsed) >= 2:
            matrix, sample_ids = parsed[:2]
            parsed = {
                "matrix": matrix,
                "sample_ids": sample_ids,
                "variant_ids": parsed[2] if len(parsed) > 2 else [],
            }
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("Genotype parser returned an unsupported result")

        if "matrix" in parsed:
            sample_ids = self._validate_sample_ids(parsed.get("sample_ids"))
            tensor = torch.as_tensor(np.asarray(parsed["matrix"]))
            if tensor.shape[0] != len(sample_ids):
                raise ValueError("Genotype matrix and sample IDs have different lengths")
            return EncodedGenotypes(
                features=tensor,
                sample_ids=sample_ids,
                feature_shapes={"main": list(tensor.shape)},
                variant_ids={
                    "main": [str(value) for value in parsed.get("variant_ids", [])]
                },
            )

        features: Dict[str, torch.Tensor] = {}
        feature_shapes: Dict[str, List[int]] = {}
        variant_ids: Dict[str, List[str]] = {}
        shared_sample_ids: List[str] | None = None
        for raw_name, branch in parsed.items():
            if branch is None:
                continue
            if not isinstance(branch, dict) or "matrix" not in branch:
                raise ValueError(f"Invalid genotype branch: {raw_name!r}")
            name = str(raw_name).lower()
            if name in features:
                raise ValueError(f"Duplicate normalized genotype branch: {name!r}")
            branch_sample_ids = self._validate_sample_ids(branch.get("sample_ids"))
            if shared_sample_ids is None:
                shared_sample_ids = branch_sample_ids
            elif branch_sample_ids != shared_sample_ids:
                raise ValueError("Genotype branches have different sample orders")
            tensor = torch.as_tensor(np.asarray(branch["matrix"]))
            if tensor.shape[0] != len(branch_sample_ids):
                raise ValueError(
                    f"Genotype branch {name!r} and sample IDs have different lengths"
                )
            features[name] = tensor
            feature_shapes[name] = list(tensor.shape)
            variant_ids[name] = [
                str(value) for value in branch.get("variant_ids", [])
            ]

        if not features or shared_sample_ids is None:
            raise ValueError("Genotype parser returned no populated feature branches")
        return EncodedGenotypes(
            features=features,
            sample_ids=shared_sample_ids,
            feature_shapes=feature_shapes,
            variant_ids=variant_ids,
        )

    @staticmethod
    def _validate_sample_ids(values: Any) -> List[str]:
        if values is None:
            raise ValueError("Genotype parser did not return sample IDs")
        sample_ids = [str(value) for value in values]
        if not sample_ids:
            raise ValueError("Genotype data contain no samples")
        if any(not sample_id for sample_id in sample_ids):
            raise ValueError("Genotype sample IDs must be nonempty")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Genotype sample IDs must be unique")
        return sample_ids

    def _read_phenotypes(self) -> Dict[str, Any]:
        frame = pd.read_csv(
            self.config.phenotype_file,
            sep=None,
            engine="python",
            dtype=str,
            keep_default_na=False,
        )
        if frame.empty:
            raise ValueError("Phenotype file contains no data rows")
        if len(frame.columns) < 2:
            raise ValueError(
                "Phenotype file must contain a sample ID and at least one trait"
            )

        id_column = self.config.sample_id_column or str(frame.columns[0])
        if id_column not in frame.columns:
            raise ValueError(f"Sample ID column not found: {id_column!r}")
        traits = list(self.config.traits or [
            str(column) for column in frame.columns if column != id_column
        ])
        if not traits:
            raise ValueError("At least one phenotype trait is required")
        if len(set(traits)) != len(traits):
            raise ValueError("Phenotype trait names must be unique")
        missing_traits = [trait for trait in traits if trait not in frame.columns]
        if missing_traits:
            raise ValueError(
                f"Phenotype traits not found: {', '.join(missing_traits)}"
            )

        classification_tasks = list(self.config.classification_tasks or [])
        if len(set(classification_tasks)) != len(classification_tasks):
            raise ValueError("Classification task names must be unique")
        missing_classification = [
            task for task in classification_tasks if task not in traits
        ]
        if missing_classification:
            raise ValueError(
                "Classification tasks not found among selected traits: "
                f"{', '.join(missing_classification)}"
            )
        classification_set = set(classification_tasks)
        # Keep regression columns first, then classification columns.
        regression_tasks = [
            trait for trait in traits if trait not in classification_set
        ]
        ordered_traits = regression_tasks + classification_tasks
        trait_tasks = (
            ["regression"] * len(regression_tasks)
            + ["classification"] * len(classification_tasks)
        )

        sample_ids = [str(value).strip() for value in frame[id_column]]
        if any(not sample_id for sample_id in sample_ids):
            raise ValueError("Phenotype sample IDs must be nonempty")
        if len(set(sample_ids)) != len(sample_ids):
            duplicates = sorted(
                sample_id for sample_id in set(sample_ids)
                if sample_ids.count(sample_id) > 1
            )
            raise ValueError(
                f"Phenotype sample IDs must be unique; duplicates: {duplicates}"
            )

        values = np.empty((len(frame), len(ordered_traits)), dtype=np.float32)
        mask = np.empty_like(values, dtype=bool)
        for trait_index, trait in enumerate(ordered_traits):
            for row_index, raw_value in enumerate(frame[trait]):
                value = self._parse_phenotype_value(
                    raw_value,
                    sample_ids[row_index],
                    trait,
                )
                values[row_index, trait_index] = value
                mask[row_index, trait_index] = (
                    not np.isnan(value)
                    and value != self.config.missing_sentinel
                )

        return {
            "sample_ids": sample_ids,
            "id_column": id_column,
            "trait_names": ordered_traits,
            "regression_tasks": regression_tasks,
            "classification_tasks": classification_tasks,
            "trait_tasks": trait_tasks,
            "values": values,
            "mask": mask,
        }

    @staticmethod
    def _missing_text(value: str) -> bool:
        return value.strip().lower() in {"", "nan", "na", "n/a", "null"}

    def _parse_phenotype_value(
        self,
        raw_value: Any,
        sample_id: str,
        trait: str,
    ) -> float:
        text = str(raw_value).strip()
        if self._missing_text(text):
            return float("nan")
        try:
            value = float(text)
        except ValueError as error:
            raise ValueError(
                f"Invalid phenotype value {text!r} for sample {sample_id!r}, "
                f"trait {trait!r}"
            ) from error
        if math.isnan(value):
            return value
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite phenotype value for sample {sample_id!r}, "
                f"trait {trait!r}"
            )
        return value

    def _align(
        self,
        genotypes: EncodedGenotypes,
        phenotype: Dict[str, Any],
    ) -> Dict[str, Any]:
        phenotype_positions = {
            sample_id: index
            for index, sample_id in enumerate(phenotype["sample_ids"])
        }
        genotype_indices = [
            index
            for index, sample_id in enumerate(genotypes.sample_ids)
            if sample_id in phenotype_positions
        ]
        if not genotype_indices:
            raise ValueError("Genotype and phenotype files have no shared samples")

        sample_ids = [genotypes.sample_ids[index] for index in genotype_indices]
        phenotype_indices = [
            phenotype_positions[sample_id] for sample_id in sample_ids
        ]
        if isinstance(genotypes.features, dict):
            features: torch.Tensor | Dict[str, torch.Tensor] = {
                name: tensor[genotype_indices].contiguous()
                for name, tensor in genotypes.features.items()
            }
        else:
            features = genotypes.features[genotype_indices].contiguous()

        targets = torch.from_numpy(
            phenotype["values"][phenotype_indices].copy()
        ).to(torch.float32)
        target_mask = torch.from_numpy(
            phenotype["mask"][phenotype_indices].copy()
        ).to(torch.bool)
        return {
            "features": features,
            "targets": targets,
            "target_mask": target_mask,
            "sample_ids": sample_ids,
            "trait_names": phenotype["trait_names"],
            "regression_tasks": phenotype["regression_tasks"],
            "classification_tasks": phenotype["classification_tasks"],
            "trait_tasks": phenotype["trait_tasks"],
        }

    def _load_outer_fold_mapping(
        self,
        aligned_sample_ids: Sequence[str],
    ) -> np.ndarray:
        mapping_path = self.config.fold_mapping
        if mapping_path is None:
            raise RuntimeError("Fold mapping path is not configured")
        frame = pd.read_csv(mapping_path, sep=None, engine="python", dtype=str)
        required = {"SampleID", "OuterFold"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                "Fold mapping must contain columns SampleID and OuterFold"
            )
        if "Role" in frame.columns:
            frame = frame[frame["Role"].str.lower() == "test"].copy()
        if frame.empty:
            raise ValueError("Fold mapping contains no outer test assignments")

        assignments: Dict[str, int] = {}
        for row in frame.itertuples(index=False):
            sample_id = str(getattr(row, "SampleID")).strip()
            try:
                fold_id = int(getattr(row, "OuterFold"))
            except ValueError as error:
                raise ValueError(
                    f"Invalid OuterFold for sample {sample_id!r}"
                ) from error
            if sample_id in assignments:
                raise ValueError(
                    f"Sample {sample_id!r} has multiple outer test assignments"
                )
            assignments[sample_id] = fold_id

        missing_samples = [
            sample_id
            for sample_id in aligned_sample_ids
            if sample_id not in assignments
        ]
        if missing_samples:
            preview = ", ".join(missing_samples[:10])
            raise ValueError(
                "Aligned samples missing from fold mapping: "
                f"{preview}"
            )
        selected = np.asarray(
            [assignments[sample_id] for sample_id in aligned_sample_ids],
            dtype=np.int64,
        )
        expected = set(range(self.config.outer_folds))
        observed = set(selected.tolist())
        if observed != expected:
            raise ValueError(
                f"Fold mapping must contain folds {sorted(expected)}, "
                f"got {sorted(observed)} after genotype alignment"
            )
        return selected

    def _build_metadata(
        self,
        genotypes: EncodedGenotypes,
        phenotype: Dict[str, Any],
        aligned: Dict[str, Any],
    ) -> Dict[str, Any]:
        aligned_ids = set(aligned["sample_ids"])
        phenotype_ids = set(phenotype["sample_ids"])
        return {
            "schema_version": 1,
            "genotype_file": str(self.config.genotype_file.resolve()),
            "phenotype_file": str(self.config.phenotype_file.resolve()),
            "encoding_type": self.config.encoding_type,
            "variant_type": self.config.variant_type,
            "feature_kind": (
                "branches" if isinstance(aligned["features"], dict) else "tensor"
            ),
            "feature_shapes": (
                {
                    name: list(tensor.shape)
                    for name, tensor in aligned["features"].items()
                }
                if isinstance(aligned["features"], dict)
                else {"main": list(aligned["features"].shape)}
            ),
            "variant_ids": genotypes.variant_ids,
            "sample_id_column": phenotype["id_column"],
            "sample_ids": aligned["sample_ids"],
            "trait_names": phenotype["trait_names"],
            "regression_tasks": phenotype["regression_tasks"],
            "classification_tasks": phenotype["classification_tasks"],
            "trait_tasks": phenotype["trait_tasks"],
            "missing_sentinel": self.config.missing_sentinel,
            "n_samples": len(aligned["sample_ids"]),
            "n_traits": len(phenotype["trait_names"]),
            "n_regression_tasks": len(phenotype["regression_tasks"]),
            "n_classification_tasks": len(phenotype["classification_tasks"]),
            "outer_folds": self.config.outer_folds,
            "inner_folds": self.config.inner_folds,
            "seed": self.config.seed,
            "source_checksums": {
                "genotype_sha256": self._sha256(self.config.genotype_file),
                "phenotype_sha256": self._sha256(self.config.phenotype_file),
            },
            "software_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "git_commit": self._git_commit(),
            "excluded_genotype_sample_ids": [
                sample_id for sample_id in genotypes.sample_ids
                if sample_id not in phenotype_ids
            ],
            "excluded_phenotype_sample_ids": [
                sample_id for sample_id in phenotype["sample_ids"]
                if sample_id not in aligned_ids
            ],
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _git_commit() -> str | None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    def _save_artifacts(
        self,
        aligned: Dict[str, Any],
        metadata: Dict[str, Any],
        folds: Sequence[Dict[str, object]],
    ) -> None:
        output = self.config.output_directory
        torch.save(aligned["features"], output / "X.pt")
        torch.save(aligned["targets"], output / "Y_raw.pt")
        torch.save(aligned["target_mask"], output / "Y_mask.pt")
        save_nested_folds(folds, output / "cv")
        self._save_processed_targets(aligned, folds, output / "cv")
        self._save_fold_mapping(
            output / "sample_fold_mapping.txt",
            aligned["sample_ids"],
            folds,
        )
        with (output / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")

    def _save_processed_targets(
        self,
        aligned: Dict[str, Any],
        folds: Sequence[Dict[str, object]],
        cv_directory: Path,
    ) -> None:
        """Cache split-local targets using train-only preprocessing fits."""
        targets = aligned["targets"]
        target_mask = aligned["target_mask"]
        trait_names = aligned["trait_names"]
        trait_tasks = aligned.get("trait_tasks")
        for outer in folds:
            outer_id = int(outer["fold"])
            outer_path = cv_directory / f"outer_fold_{outer_id}"
            processor = self._fit_preprocessor(
                targets,
                target_mask,
                outer["train"],
                trait_names,
                trait_tasks,
            )
            final_targets = processor.apply(targets, target_mask)
            final_path = outer_path / "final"
            final_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                final_targets[outer["train"]].contiguous(),
                final_path / "Y_train_processed.pt",
            )
            torch.save(
                final_targets[outer["test"]].contiguous(),
                final_path / "Y_test_processed.pt",
            )
            processor.save_json(final_path / "preprocessing.json")

            for inner in outer["inner"]:
                inner_id = int(inner["fold"])
                inner_path = outer_path / f"inner_fold_{inner_id}"
                processor = self._fit_preprocessor(
                    targets,
                    target_mask,
                    inner["train"],
                    trait_names,
                    trait_tasks,
                )
                processed = processor.apply(targets, target_mask)
                inner_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    processed[inner["train"]].contiguous(),
                    inner_path / "Y_train_processed.pt",
                )
                torch.save(
                    processed[inner["valid"]].contiguous(),
                    inner_path / "Y_valid_processed.pt",
                )
                processor.save_json(inner_path / "preprocessing.json")

    def _fit_preprocessor(
        self,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        train_indices: Sequence[int],
        trait_names: Sequence[str],
        trait_tasks: Sequence[str] | None = None,
    ) -> PerTraitPreprocessor:
        return PerTraitPreprocessor(
            skew_threshold=self.config.skew_threshold,
            epsilon=self.config.preprocessing_epsilon,
        ).fit(
            targets,
            target_mask,
            train_indices,
            trait_names,
            trait_tasks=trait_tasks,
        )

    @staticmethod
    def _save_fold_mapping(
        path: Path,
        sample_ids: Sequence[str],
        folds: Sequence[Dict[str, object]],
    ) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("SampleIndex\tSampleID\tOuterFold\tRole\n")
            for outer in folds:
                outer_number = int(outer["fold"])
                test_indices = set(
                    np.asarray(outer["test"], dtype=np.int64).tolist()
                )
                for index, sample_id in enumerate(sample_ids):
                    role = "test" if index in test_indices else "train"
                    handle.write(
                        f"{index}\t{sample_id}\t{outer_number}\t{role}\n"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare aligned Aquila data and deterministic nested CV folds."
    )
    parser.add_argument(
        "--genotype",
        "--geno",
        "--vcf",
        dest="genotype",
        required=True,
        help="Input genotype or VCF file.",
    )
    parser.add_argument(
        "--phenotype",
        "--pheno",
        dest="phenotype",
        required=True,
        help="Input phenotype table.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--output-directory",
        dest="output_directory",
        required=True,
        help="Directory for prepared artifacts.",
    )
    parser.add_argument(
        "--encoding",
        "--encoding-type",
        choices=("token", "diploid_onehot", "onehot"),
        default="diploid_onehot",
        help="Genotype encoding passed to aquila.encoding.parse_genotype_file.",
    )
    parser.add_argument(
        "--variant-type",
        choices=("snp", "snp_indel", "snp_indel_sv"),
        default=None,
        help="Optional variant selection passed to the genotype parser.",
    )
    parser.add_argument(
        "--sample-id-column",
        default=None,
        help="Phenotype sample ID column; defaults to the first column.",
    )
    parser.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Phenotype columns to retain; defaults to all non-ID columns.",
    )
    parser.add_argument(
        "--classification-tasks",
        nargs="+",
        default=None,
        help=(
            "Phenotype columns treated as binary classification tasks. "
            "All other selected traits remain regression. Defaults to none "
            "(all traits are regression)."
        ),
    )
    parser.add_argument(
        "--missing-sentinel",
        type=float,
        default=-999.0,
        help="Numeric phenotype value treated as missing.",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
        help=(
            "Number of outer folds. Without --fold-mapping, folds are generated "
            "here; with --fold-mapping, this must match the mapping's fold count "
            "and does not repartition samples."
        ),
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=4,
        help=(
            "Number of inner folds. Inner folds are always generated by this "
            "command from each outer-training partition."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic fold generation.",
    )
    parser.add_argument(
        "--fold-mapping",
        default=None,
        help=(
            "Predefined outer-fold mapping created by aquila_cv.py. When "
            "provided, outer sample assignments come exclusively from this "
            "file; --outer-folds only validates its fold count, while "
            "--inner-folds controls newly generated inner folds."
        ),
    )
    parser.add_argument(
        "--min-observed-per-fold",
        type=int,
        default=10,
        help="Minimum observed values per trait in every outer test fold.",
    )
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=2.0,
        help="Absolute skewness threshold for per-trait log1p transforms.",
    )
    parser.add_argument(
        "--preprocessing-epsilon",
        type=float,
        default=1e-8,
        help="Numerical epsilon used by phenotype preprocessing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing prepared artifacts in the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreparationConfig(
        genotype_file=Path(args.genotype),
        phenotype_file=Path(args.phenotype),
        output_directory=Path(args.output_directory),
        encoding_type=args.encoding,
        variant_type=args.variant_type,
        sample_id_column=args.sample_id_column,
        traits=args.traits,
        classification_tasks=args.classification_tasks,
        missing_sentinel=args.missing_sentinel,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        fold_mapping=Path(args.fold_mapping) if args.fold_mapping else None,
        min_observed_per_fold=args.min_observed_per_fold,
        skew_threshold=args.skew_threshold,
        preprocessing_epsilon=args.preprocessing_epsilon,
        overwrite=args.overwrite,
    )
    metadata = NestedCVDataPreparer(config).prepare()
    print(
        f"Prepared {metadata['n_samples']} samples and "
        f"{metadata['n_traits']} traits in {config.output_directory}"
    )
    print(
        f"  Regression tasks ({metadata['n_regression_tasks']}): "
        f"{metadata['regression_tasks']}"
    )
    print(
        f"  Classification tasks ({metadata['n_classification_tasks']}): "
        f"{metadata['classification_tasks']}"
    )


if __name__ == "__main__":
    main()
