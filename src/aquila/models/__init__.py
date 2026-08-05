# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Model construction APIs."""

from .registry import MODEL_REGISTRY, ModelRegistry, create_model, register_model

__all__ = [
    "MODEL_REGISTRY",
    "ModelRegistry",
    "create_model",
    "register_model",
]
