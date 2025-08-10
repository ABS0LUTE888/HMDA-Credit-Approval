from __future__ import annotations

from typing import Callable, Dict

from omegaconf import DictConfig

from .base import BaseModel

# Registry
MODELS: Dict[str, Callable[[DictConfig], BaseModel]] = {}


def register(name: str | None = None):
    """
    Decorator to register a model class in the MODELS registry.
    Optionally takes a custom name; defaults to the class name.
    """

    def decorator(target):
        key = name or target.__name__
        MODELS[key] = (lambda cfg, t=target: t(cfg))
        return target

    return decorator


def create(name: str, cfg: DictConfig) -> BaseModel:
    """
    Make an instance of a registered model by name.
    Raises KeyError if the model is not found.
    """
    try:
        return MODELS[name](cfg)
    except KeyError as e:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODELS)}") from e


def from_cfg(cfg: DictConfig) -> BaseModel:
    """Create a model using cfg.model.name"""
    return create(cfg.model.name, cfg)
