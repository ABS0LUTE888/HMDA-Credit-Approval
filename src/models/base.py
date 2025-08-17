from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


@dataclass
class DataBundle:
    """Container for train/validation splits passed into models"""
    X_tr: Any
    y_tr: Any
    X_val: Optional[Any] = None
    y_val: Optional[Any] = None


class BaseModel(ABC):
    """Abstract interface all model implementations must follow"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.seed: int = cfg.seed
        self.est_: Any = None
        self.best_params_: Dict[str, Any] = {}

    @abstractmethod
    def train(self, data: DataBundle) -> Tuple[Any, Dict[str, Any]]:
        """Fit the model and return the fitted estimator and best params"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """Return predictions (labels)"""
        raise NotImplementedError

    def predict_proba(self, X):
        """
        Return class probabilities from the trained model
        """
        if self.est_ is None:
            raise RuntimeError("Model is not trained. Call train() first")
        if hasattr(self.est_, "predict_proba"):
            return self.est_.predict_proba(X)
        raise AttributeError(f"{self.__class__.__name__} has no predict_proba()")

    def evaluate(self, data: DataBundle) -> Dict[str, Any]:
        """Evaluate model performance on validation data"""
        raise NotImplementedError

    def export_pipeline(self, preprocessor_path: Path, out_path: Path) -> Path:
        """
        Combine preprocessing and trained model into one pipeline, then save it
        """
        raise NotImplementedError


class SciKitModel(BaseModel, ABC):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._n_jobs: int = cfg.model.n_jobs
        self._cv: int = cfg.model.cv
        self._scoring: str = cfg.model.scoring
        self._verbose: int = cfg.model.verbose
        self.data: Optional[DataBundle] = None

    @abstractmethod
    def build(self) -> Any:
        """Return a model instance"""
        raise NotImplementedError

    def param_grid(self) -> Optional[Dict[str, Any]]:
        """Return parameter grid for hyperparameter tuning"""
        return None

    def train(self, data: DataBundle) -> Tuple[Any, Dict[str, Any]]:
        self.data = data
        est = self.build()
        grid = self.param_grid()

        if grid:
            cv = StratifiedKFold(n_splits=self._cv, shuffle=True, random_state=self.seed)
            gs = GridSearchCV(
                estimator=est,
                param_grid=grid,
                scoring=self._scoring,
                cv=cv,
                n_jobs=self._n_jobs,
                pre_dispatch=self._n_jobs,
                refit=True,
                verbose=self._verbose,
            )
            gs.fit(data.X_tr, data.y_tr)
            self.est_ = gs.best_estimator_
            self.best_params_ = dict(gs.best_params_)
        else:
            est.fit(data.X_tr, data.y_tr)
            self.est_ = est
            self.best_params_ = {}

        return self.est_, self.best_params_

    def predict(self, X):
        if self.est_ is None:
            raise RuntimeError("Model is not trained. Call train() first")
        return self.est_.predict(X)

    def evaluate(self, data: DataBundle) -> Dict[str, Any]:
        if data.X_val is None or data.y_val is None:
            raise ValueError("Validation data (X_val, y_val) is required for evaluation")

        proba = self.predict_proba(data.X_val)[:, 1]
        preds = self.predict(data.X_val)
        auroc = float(roc_auc_score(data.y_val, proba))
        report = classification_report(data.y_val, preds, digits=3)
        print(f"AUROC : {auroc:.3f}")
        print(report)
        return {"auroc": auroc, "report": report}

    def export_pipeline(self, preprocessor_path: Path, out_path: Path) -> Path:
        preproc = joblib.load(preprocessor_path)
        pipe = Pipeline([("preprocess", preproc), ("model", self.est_)])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_path)
        print(f"Saved pipeline to {out_path}")
        return out_path
