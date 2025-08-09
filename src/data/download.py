"""
Download HMDA LAR raw dataset
"""

import shutil
import zipfile
from pathlib import Path

import hydra
import requests
from omegaconf import DictConfig
from tqdm import tqdm

CHUNK = 1024 * 1024  # Chunk size in bytes


def stream_download(url, dest):
    # Stream file from URL and save to dest with progress bar
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(CHUNK):
            f.write(chunk)
            bar.update(len(chunk))


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    raw_dir = Path(cfg.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Path to downloaded ZIP file
    zip_path = raw_dir / Path(cfg.data.file_template).name

    # Download ZIP
    print(f"Downloading: {cfg.data.file_template}")
    stream_download(cfg.data.file_template, zip_path)

    # Extract CSV if it doesn't already exist
    csv_path = Path(cfg.csv_path)
    if not csv_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            member = next(m for m in zf.namelist() if m.endswith(".csv"))
            zf.extract(member, path=raw_dir)
            shutil.move(raw_dir / member, csv_path)
    print(f"Extracted to {csv_path}")


if __name__ == "__main__":
    main()
