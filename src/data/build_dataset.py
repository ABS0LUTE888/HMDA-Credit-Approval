from pathlib import Path

import hydra
import joblib
import pyarrow.dataset as ds
import scipy.sparse as sp
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed = cfg.seed

    processed_dir = Path(cfg.processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError("Run the cleaning script first")

    table = ds.dataset(processed_dir, format="parquet").to_table().combine_chunks()
    df = table.to_pandas()

    num_cols = [c for c in cfg.features.numeric if c in df.columns]
    cat_cols = [c for c in cfg.features.categorical if c in df.columns]

    X = df[num_cols + cat_cols]
    y = df["target_approved"].astype("int8")

    assert "action_taken", "denial_reason_1" not in X.columns

    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preproc = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    preproc.fit(X_tr)

    dataset_dir = Path(cfg.dataset_dir)
    dataset_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(preproc, dataset_dir / "preprocessor.pkl")
    sp.save_npz(dataset_dir / "X_train.npz", preproc.transform(X_tr))
    sp.save_npz(dataset_dir / "X_val.npz", preproc.transform(X_val))
    y_tr.to_csv(dataset_dir / "y_train.csv", index=False)
    y_val.to_csv(dataset_dir / "y_val.csv", index=False)

    print(f"Dataset saved to {dataset_dir}")


if __name__ == "__main__":
    main()
