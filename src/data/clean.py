"""
Clean raw CSV in chunks, convert data types, create target column, and save as Parquet dataset
"""

import shutil
from pathlib import Path

import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

CHUNK_ROWS = 250_000


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_csv = Path(cfg.csv_path)
    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)

    out_dir = Path(cfg.processed_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Labels and feature definitions from config
    pos = set(cfg.label.approved)
    neg = set(cfg.label.denied)
    target_col = cfg.label.field
    num_cols = set(cfg.features.numeric)

    # Read CSV in chunks
    reader = pd.read_csv(raw_csv, dtype=str, chunksize=CHUNK_ROWS, low_memory=False)

    pd.set_option("future.no_silent_downcasting", True)

    # Process each chunk
    for chunk in tqdm(reader, desc="Processing chunks"):
        # Replace empty/placeholder values with NA
        chunk = chunk.replace({"": pd.NA, "NA": pd.NA, "Exempt": pd.NA})

        # Convert numeric columns to numbers
        for col in num_cols & set(chunk.columns) & set(cfg.allowed_cols):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Create binary target column
        chunk = chunk.loc[chunk[target_col].astype(str).isin(map(str, pos | neg))].copy()
        chunk["target_approved"] = chunk[target_col].astype(str).isin(map(str, pos)).astype("int8")

        # Drop any disallowed columns
        cols_to_drop = list(((set(chunk.columns) - set(cfg.allowed_cols)) | {target_col}) - {"target_approved"})
        if cols_to_drop:
            chunk = chunk.drop(columns=cols_to_drop)

        assert target_col not in chunk.columns

        # Save processed chunk as Parquet
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=out_dir,
            partition_cols=None,
            compression="zstd",
        )

    print(f"Cleaned data saved to {out_dir}")


if __name__ == "__main__":
    main()
