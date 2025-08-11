"""
Decision tree model wrapper
"""

from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig
from sklearn.tree import DecisionTreeClassifier

from .base import SciKitModel
from .registry import register


@register("decision_tree")
class DecisionTreeModel(SciKitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build(self) -> Any:
        return DecisionTreeClassifier(random_state=self.seed)

    def param_grid(self) -> Dict[str, Any]:
        return {
            "max_depth": list(self.cfg.model.max_depth),
            "min_samples_leaf": list(self.cfg.model.min_samples_leaf),
        }
