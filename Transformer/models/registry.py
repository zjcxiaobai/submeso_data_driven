from __future__ import annotations

from typing import Callable

import torch.nn as nn


MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    def decorator(factory: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = factory
        return factory

    return decorator


def build_model(config: dict) -> nn.Module:
    name = config.get("name")
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")

    kwargs = {key: value for key, value in config.items() if key != "name"}
    return MODEL_REGISTRY[name](**kwargs)
