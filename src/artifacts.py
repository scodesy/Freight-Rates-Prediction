from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    paths = list((root / "src").glob("*.py")) + list((root / "scripts").glob("*.py")) + [root / "requirements.txt"]
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(paths) if path.exists()}


def data_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in paths.items()}


def runtime_info(start: float, workers: int) -> dict[str, Any]:
    finished = time.time()
    return {
        "started_at_utc": start,
        "finished_at_utc": finished,
        "wall_seconds": finished - start,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
        "workers": workers,
        "packages": _package_versions(("catboost", "numpy", "pandas", "scikit-learn", "scipy", "xgboost")),
    }


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def atomic_run_dir(root: Path, run_id: str, files: dict[str, str | bytes]) -> Path:
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    final = runs / run_id
    if final.exists():
        raise FileExistsError(f"run already exists: {run_id}")
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=runs))
    try:
        for name, content in files.items():
            mode = "wb" if isinstance(content, bytes) else "w"
            with (tmp / name).open(mode, encoding=None if isinstance(content, bytes) else "utf-8", newline="" if name.endswith(".csv") else None) as fh:
                fh.write(content)
        tmp.rename(final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def append_ledger(root: Path, row: Mapping[str, Any]) -> None:
    path = root / "runs.csv"
    exists = path.exists()
    fields = ["run_id", "status", "model", "target", "feature_view", "mean_mae", "mean_rmse", "mean_wmape"]
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
