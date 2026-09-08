from __future__ import annotations

import argparse
from pathlib import Path

from src.data_validation import load_csv_rows


def existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"missing path: {value}")
    return path


def load_training(path: Path):
    return load_csv_rows(path, schema="training")
