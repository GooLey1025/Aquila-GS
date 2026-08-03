# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Contract tests for the BayesCpi benchmark adapter."""

from pathlib import Path

import yaml


def test_bayescpi_uses_fixed_mcmc_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root.parent / "configs" / "nested_cv.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    worker = (root / "bayescpi_worker.R").read_text(encoding="utf-8")

    assert config["parameters"]["niter"] > config["parameters"]["nburn"]
    assert config["parameters"]["thin"] > 0
    assert 'method = "BayesCpi"' in worker
