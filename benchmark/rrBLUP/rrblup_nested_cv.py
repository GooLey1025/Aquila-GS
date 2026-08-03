#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/cran/rrBLUP

"""Nested-CV entry point for rrBLUP."""

from pathlib import Path
import sys

MODEL_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = MODEL_ROOT.parent
BENCHMARK_SOURCE = MODEL_ROOT / "src_benchmark"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from r_models_common.nested_cv import launch


if __name__ == "__main__":
    config_path = MODEL_ROOT / "configs" / "nested_cv.yaml"
    launch(
        "rrBLUP",
        "https://github.com/cran/rrBLUP",
        BENCHMARK_SOURCE / "rrblup_worker.R",
        ("jsonlite", "rrBLUP"),
        ("method",),
        config_path,
    )
