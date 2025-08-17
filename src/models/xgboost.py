"""
XGBoost model wrapper
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from xgboost import XGBClassifier

from .base import SciKitModel
from .registry import register


@register("xgboost")
class XGBoostModel(SciKitModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.params: Dict[str, Any] = dict(cfg.model.params)

    def build(self) -> Any:
        params = self.params
        data = self.data

        pos = np.sum(data.y_tr == 1)
        neg = np.sum(data.y_tr == 0)

        params["scale_pos_weight"] = neg / pos
        return XGBClassifier(**params)

    def param_grid(self) -> Dict[str, Any]:
        return {
            "n_estimators": list(self.cfg.model.n_estimators),
            "max_depth": list(self.cfg.model.max_depth),
            "learning_rate": list(self.cfg.model.learning_rate),
        }
