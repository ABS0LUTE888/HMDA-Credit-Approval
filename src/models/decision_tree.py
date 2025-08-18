"""
Decision tree model wrapper
"""

from __future__ import annotations

from typing import Any

from sklearn.tree import DecisionTreeClassifier

from .base import SciKitModel
from .registry import register


@register("decision_tree")
class DecisionTreeModel(SciKitModel):
    def build(self) -> Any:
        return DecisionTreeClassifier(**self.params)
