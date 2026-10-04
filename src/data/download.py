"""Dataset download utilities for the ATR Human Parsing dataset.

Supports direct download from the Hugging Face mirror of Deep Human Parsing ATR
(mattmdjaga/human_parsing_dataset) or Kaggle API if configured.
"""

from pathlib import Path
import urllib.request
import shutil
from tqdm import tqdm

HF_BASE_URL = "https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset/resolve/main"
HF_FILES = [
    ("id2label.json", "id2label.json"),
    ("data/train-00000-of-00002-f3a663f7140ee7fd.parquet", "train-00000.parquet"),
    ("data/train-00001-of-00002-74610e243c32d5d6.parquet", "train-00001.parquet"),
]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file_with_progress(url: str, output_path: Path) -> Path:
    """Download a file with progress bar and atomic rename."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".download")
    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(url, filename=temp_path, reporthook=t.update_to)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return output_path


def download_atr_dataset(raw_dir: str | Path = "data/raw") -> list[Path]:
    """Ensure ATR dataset files exist in ``raw_dir``, downloading if necessary.

    Returns:
        List of paths to parquet files.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    parquet_paths: list[Path] = []

    for remote_rel, local_name in HF_FILES:
        target = raw_path / local_name
        if not target.exists() or target.stat().st_size == 0:
            url = f"{HF_BASE_URL}/{remote_rel}"
            download_file_with_progress(url, target)
        if target.suffix == ".parquet":
            parquet_paths.append(target)

    return parquet_paths
