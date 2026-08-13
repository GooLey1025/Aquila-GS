#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/cran/glmnet

from pathlib import Path

import yaml


def test_lasso_has_explicit_lambda_grid() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root.parent / "configs" / "nested_cv.yaml").read_text(encoding="utf-8")
    )
    assert len(config["grid"]["lambda"]) > 1
    assert "alpha = 1" in (root / "lasso_worker.R").read_text(encoding="utf-8")
