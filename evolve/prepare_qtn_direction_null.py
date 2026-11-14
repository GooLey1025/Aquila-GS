#!/usr/bin/env python3
"""Draw random two-trait directions that exclude the reported target traits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing", required=True, help="fold_0 preprocessing.json")
    parser.add_argument("--direction-file", required=True, help="Reported direction file")
    parser.add_argument("--n-directions", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def regression_traits(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text())
    names = []
    for trait in payload["traits"]:
        if trait.get("task") == "regression":
            names.append(str(trait["name"]))
    if len(names) < 4:
        raise ValueError(f"{path} does not list enough regression traits")
    return names


def target_traits(path: str) -> set[str]:
    excluded = set()
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mode = parts[2].lower()
        if mode in {"maximize", "minimize"}:
            excluded.add(parts[0])
    if len(excluded) < 1:
        raise ValueError(f"No maximize/minimize traits in {path}")
    return excluded


def main() -> None:
    args = parse_args()
    if args.n_directions < 1:
        raise ValueError("--n-directions must be positive")
    excluded = target_traits(args.direction_file)
    pool = [name for name in regression_traits(args.preprocessing) if name not in excluded]
    if len(pool) < 2:
        raise ValueError("Need at least two traits outside the reported direction")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    manifest = ["perm\tmaximize\tminimize"]
    for perm in range(1, args.n_directions + 1):
        pair = list(rng.choice(pool, size=2, replace=False))
        rng.shuffle(pair)
        maximize, minimize = pair
        path = out_dir / f"perm_{perm:02d}.directions.tsv"
        path.write_text(
            f"{maximize}\tregression\tmaximize\n{minimize}\tregression\tminimize\n"
        )
        manifest.append(f"{perm}\t{maximize}\t{minimize}")
    (out_dir / "directions.tsv").write_text("\n".join(manifest) + "\n")
    print(f"Wrote {args.n_directions} directions to {out_dir}")


if __name__ == "__main__":
    main()
