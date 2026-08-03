#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/cran/rrBLUP

from pathlib import Path

import yaml


def test_default_is_reml_singleton_and_worker_uses_beta() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root.parent / "configs" / "nested_cv.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    worker = (root / "rrblup_worker.R").read_text(encoding="utf-8")
    assert config["grid"] == {"method": ["REML"]}
    assert "beta[[1]] +" in worker
