# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com
# Migrated from: https://github.com/ganlab/MENET

"""Benchmark-specific MENET model implementations."""

from .menet_benchmark import MeNetBenchmark
from .trait_encoder_benchmark import TraitSpecificEncoderBenchmark

__all__ = ["MeNetBenchmark", "TraitSpecificEncoderBenchmark"]
