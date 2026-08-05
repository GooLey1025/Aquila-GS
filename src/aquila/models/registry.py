# -*- coding: utf-8 -*-
# Author: Lei Gu
# Contact: goley04@foxmail.com

"""Model factory registry for Aquila training workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

ModelFactory = Callable[..., Any]


class ModelRegistry:
    """Store named model factories behind a small, stable API."""

    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(
        self,
        name: str,
        factory: ModelFactory | None = None,
        *,
        replace: bool = False,
    ) -> ModelFactory | Callable[[ModelFactory], ModelFactory]:
        """Register a factory directly or as a decorator."""
        normalized_name = self._normalize_name(name)

        def decorator(candidate: ModelFactory) -> ModelFactory:
            if normalized_name in self._factories and not replace:
                raise KeyError(f"Model factory '{normalized_name}' is already registered")
            self._factories[normalized_name] = candidate
            return candidate

        return decorator(factory) if factory is not None else decorator

    def create(self, name: str = "aquila", **kwargs: Any) -> Any:
        """Build a model using a registered factory."""
        normalized_name = self._normalize_name(name)
        try:
            factory = self._factories[normalized_name]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "<none>"
            raise KeyError(
                f"Unknown model '{normalized_name}'. Available models: {available}"
            ) from error
        return factory(**kwargs)

    def get(self, name: str) -> ModelFactory:
        """Return a registered factory."""
        normalized_name = self._normalize_name(name)
        if normalized_name not in self._factories:
            raise KeyError(f"Unknown model '{normalized_name}'")
        return self._factories[normalized_name]

    def names(self) -> tuple[str, ...]:
        """Return registered model names in deterministic order."""
        return tuple(sorted(self._factories))

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized_name = str(name).strip().lower()
        if not normalized_name:
            raise ValueError("Model name must not be empty")
        return normalized_name


def _create_aquila_model(
    *,
    config: Mapping[str, Any],
    seq_length: int | Mapping[str, int],
    regression_tasks: list[str] | None = None,
    classification_tasks: list[str] | None = None,
    **_: Any,
) -> Any:
    from aquila.varnn import create_model_from_config

    return create_model_from_config(
        config=dict(config),
        seq_length=seq_length,
        regression_tasks=regression_tasks,
        classification_tasks=classification_tasks,
    )


MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register("aquila", _create_aquila_model)


def register_model(
    name: str,
    factory: ModelFactory | None = None,
    *,
    replace: bool = False,
) -> ModelFactory | Callable[[ModelFactory], ModelFactory]:
    """Register a model factory in the process-wide registry."""
    return MODEL_REGISTRY.register(name, factory, replace=replace)


def create_model(name: str = "aquila", **kwargs: Any) -> Any:
    """Create a model from the process-wide registry."""
    return MODEL_REGISTRY.create(name, **kwargs)
