from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict, List

from omegaconf import DictConfig

from .base import BaseModel

# Registry
MODELS: Dict[str, Callable[[DictConfig], BaseModel]] = {}

_DISCOVERED = False


def _discover_models() -> None:
    """Import all model modules in this package so @register decorators run"""
    global _DISCOVERED
    if _DISCOVERED:
        return

    pkg_name = __package__
    if not pkg_name:
        return

    pkg = importlib.import_module(pkg_name)

    for _finder, modname, ispkg in pkgutil.walk_packages(
            pkg.__path__, prefix=pkg.__name__ + "."
    ):
        if ispkg:
            continue
        leaf = modname.rsplit(".", 1)[-1]
        if leaf in {"base", "registry", "__init__"} or leaf.startswith("_"):
            continue
        importlib.import_module(modname)

    _DISCOVERED = True


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
    _discover_models()

    try:
        return MODELS[name](cfg)
    except KeyError as e:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(MODELS)}") from e


def from_cfg(cfg: DictConfig) -> BaseModel:
    """Create a model using cfg.model.name"""
    return create(cfg.model.name, cfg)


def get_models() -> List[str]:
    """Get list of registered models"""
    _discover_models()

    return list(MODELS.keys())
