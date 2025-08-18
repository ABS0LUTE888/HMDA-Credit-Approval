"""
XGBoost model wrapper
"""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from .base import SciKitModel
from .registry import register


@register("xgboost")
class XGBoostModel(SciKitModel):
    def build(self) -> Any:
        params = self.params
        data = self.data

        pos = np.sum(data.y_tr == 1)
        neg = np.sum(data.y_tr == 0)

        params["scale_pos_weight"] = neg / pos
        return XGBClassifier(**params)
