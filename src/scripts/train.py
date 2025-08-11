from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
import scipy.sparse as sp
from omegaconf import DictConfig

from src.models.base import DataBundle
from src.models.registry import from_cfg


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.dataset_dir)

    # Load training and validation data
    X_tr = sp.load_npz(data_dir / "X_train.npz")
    X_val = sp.load_npz(data_dir / "X_val.npz")

    # Load labels
    y_tr = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0].values
    y_val = pd.read_csv(data_dir / "y_val.csv").iloc[:, 0].values

    # Package data into a bundle for model
    data = DataBundle(X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val)

    # Initialize model from configuration
    model = from_cfg(cfg)

    # Train and evaluate model
    model.train(data)
    model.evaluate(data)

    # Export trained pipeline with preprocessor
    out_dir = Path(cfg.model_dir)
    model.export_pipeline(
        preprocessor_path=data_dir / "preprocessor.pkl",
        out_path=out_dir / f"{cfg.model.name}_pipeline.pkl",
    )


if __name__ == "__main__":
    main()
