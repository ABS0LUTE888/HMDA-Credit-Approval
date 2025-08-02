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

    pos = set(cfg.label.approved)
    neg = set(cfg.label.denied)
    target_col = cfg.label.field
    drop_cols = set(cfg.drop)
    num_cols = set(cfg.features.numeric)
    partitions = ["state_code"] if "state_code" in cfg.features.categorical or "state_code" in num_cols else []

    reader = pd.read_csv(raw_csv, dtype=str, chunksize=CHUNK_ROWS, low_memory=False)

    pd.set_option('future.no_silent_downcasting', True)

    for chunk in tqdm(reader, desc="Processing chunks"):
        chunk = chunk.replace({"": pd.NA, "NA": pd.NA, "Exempt": pd.NA})

        for col in num_cols & set(chunk.columns):
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk.loc[chunk[target_col].astype(str).isin(map(str, pos | neg))].copy()

        chunk["target_approved"] = chunk[target_col].astype(str).isin(map(str, pos)).astype("int8")

        cols_to_drop = [c for c in drop_cols | {target_col} if c in chunk.columns]
        if cols_to_drop:
            chunk = chunk.drop(columns=cols_to_drop)

        assert "target_approved" in chunk.columns
        assert target_col not in chunk.columns
        assert "denial_reason_1" not in chunk.columns
        assert not chunk.empty

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=out_dir,
            partition_cols=partitions,
            compression="zstd",
        )

    print(f"Parquet saved to {out_dir}")


if __name__ == "__main__":
    main()
