from __future__ import annotations

import json
from pathlib import Path

import hydra
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
from omegaconf import DictConfig
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_dir = Path(cfg.dataset_dir)
    model_dir = Path(cfg.model_dir)

    # Load validation data
    X_val = sp.load_npz(dataset_dir / "X_val.npz")
    y_val = pd.read_csv(dataset_dir / "y_val.csv").iloc[:, 0].values

    # Load model
    pipeline_path = model_dir / f"{cfg.model.name}_pipeline.pkl"
    pipeline = joblib.load(pipeline_path)
    model = pipeline.steps[-1][1]

    n_expected = model.n_features_in_
    if X_val.shape[1] != n_expected:
        raise ValueError(
            f"X_val has {X_val.shape[1]} features but model expects {n_expected}"
        )

    # Predictions
    proba = model.predict_proba(X_val)[:, 1]
    threshold = float(cfg.threshold)
    preds = (proba >= threshold).astype(int)

    # Metrics
    metrics = {
        "auroc": roc_auc_score(y_val, proba),
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1": f1_score(y_val, preds, zero_division=0),
    }

    print("\n" + " Evaluation metrics ".center(40, "="))
    for k, v in metrics.items():
        print(f"{k:>9}: {v:.4f}")

    print("\n" + " Classification report ".center(40, "="))
    print(classification_report(y_val, preds, digits=3))

    # Artifacts
    plot_dir = model_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ROC Curve
    RocCurveDisplay.from_predictions(y_val, proba)
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_curve.png", dpi=150)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_val, preds, normalize="true")
    ConfusionMatrixDisplay(cm).plot(colorbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    with (model_dir / "metrics.json").open("w") as fp:
        json.dump(metrics, fp, indent=2)

    print("\nSaved metrics to", model_dir / "metrics.json")
    print("Saved plots to", plot_dir)


if __name__ == "__main__":
    main()
