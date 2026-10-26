"""Load the challenge files. The data itself is not part of this repo (see data/README.md)."""
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    "x_train": "X_train_9xQjqvZ.csv",
    "y_train": "y_train_Ppwhaz8.csv",
    "x_test": "X_test_1zTtEnD.csv",
}


def _find(data_dir: Path, name: str) -> Path:
    exact = data_dir / FILES[name]
    if exact.exists():
        return exact
    prefix = FILES[name].split("_")[0] + "_" + FILES[name].split("_")[1]  # e.g. X_train
    hits = sorted(data_dir.glob(prefix + "*.csv"))
    if not hits:
        raise FileNotFoundError(f"{FILES[name]} not found in {data_dir} (see data/README.md)")
    return hits[0]


def load(data_dir):
    """Return x_train, y_train (0/1), raw next-day return, x_test."""
    data_dir = Path(data_dir)
    x_train = pd.read_csv(_find(data_dir, "x_train"))
    y_raw = pd.read_csv(_find(data_dir, "y_train"))
    x_test = pd.read_csv(_find(data_dir, "x_test"))
    target_col = [c for c in y_raw.columns if c != "ROW_ID"][0]
    df = x_train.merge(y_raw, on="ROW_ID", how="inner")
    ret_next = df[target_col].to_numpy(dtype=float)
    y = pd.Series((ret_next > 0).astype(int), name="y")
    x_train = df.drop(columns=[target_col]).reset_index(drop=True)
    return x_train, y, ret_next, x_test.reset_index(drop=True)
