from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import append_ledger, atomic_run_dir, config_hash, data_hashes, runtime_info, source_hashes
from src.metrics import regression_metrics, slice_metrics, validate_predictions
from src.models import make_estimator
from src.preprocessing import FreightPreprocessor
from src.splits import FoldSpec, development_folds, holdout_fold, horizon_fold, split_rows
from src.targets import target_adapter


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: str
    run_dir: Path
    aggregate: dict[str, Any]


def fold_by_name(name: str) -> FoldSpec:
    folds = {fold.name: fold for fold in (*development_folds(), horizon_fold(), holdout_fold())}
    if name not in folds:
        raise ValueError(f"unknown fold: {name}")
    return folds[name]


def run_experiment(rows: list[Mapping[str, object]], config: Mapping[str, Any], artifact_root: Path, *, data_paths: Mapping[str, Path] | None = None, allow_holdout: bool = False) -> RunResult:
    start = time.time()
    root = Path.cwd()
    full_config = {"schema_version": 1, **dict(config)}
    chash = config_hash(full_config)
    run_id = f"{time.time_ns()}-{chash}"
    predictions: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    status = {"status": "completed", "reason_code": None, "error_type": None, "message": ""}
    try:
        if str(config.get("model")) in {"xgboost", "catboost"}:
            from src.models import available_optional_models

            if available_optional_models().get(str(config["model"])) != "available":
                raise ImportError(f"{config['model']} not available")
        fold_names = list(config.get("folds", ["july_1m", "august_1m", "september_1m"]))
        for fold_name in fold_names:
            fold = fold_by_name(fold_name)
            train, valid = split_rows(rows, fold, allow_holdout=allow_holdout)
            view = str(config["feature_view"])
            pre = FreightPreprocessor(feature_view="full" if view.startswith("full") else view).fit(list(train))
            x_train = pre.transform_frame(list(train))
            x_valid = pre.transform_frame(list(valid))
            if view == "full_no_route" and "route" in x_train.columns:
                x_train = x_train.drop(columns=["route"])
                x_valid = x_valid.drop(columns=["route"])
            y_train = np.asarray([float(row["posted_rate"]) for row in train], dtype=float)
            y_valid = np.asarray([float(row["posted_rate"]) for row in valid], dtype=float)
            adapter = target_adapter(str(config["target"]))
            estimator = make_estimator(str(config["model"]), dict(config.get("parameters", {})))
            estimator.fit(x_train, adapter.transform(y_train, x_train))
            pred = adapter.inverse_transform(estimator.predict(x_valid), x_valid)
            pred = np.asarray(validate_predictions(pred), dtype=float)
            metrics = regression_metrics(y_valid, pred)
            aggregates.append({"fold": fold.name, "role": fold.role, **metrics.__dict__})
            seen = {(str(row["pickup"]), str(row["delivery"])) for row in train}
            for item in slice_metrics(list(valid), y_valid, pred, seen):
                slices.append({"fold": fold.name, **item})
            for row, actual, value in zip(valid, y_valid, pred, strict=True):
                predictions.append({"run_id": run_id, "config_hash": chash, "fold": fold.name, "load_id": row.get("load_id", ""), "date": row["date"], "actual_rate": actual, "predicted_rate": value, "absolute_error": abs(actual - value)})
    except Exception as exc:
        status = {"status": "failed", "reason_code": "exception", "error_type": type(exc).__name__, "message": str(exc)[:500]}
    selection = [row for row in aggregates if row["role"] == "selection"]
    aggregate = {
        "selection_folds": [row["fold"] for row in selection],
        "mean_mae": float(np.mean([row["mae"] for row in selection])) if selection else None,
        "mean_rmse": float(np.mean([row["rmse"] for row in selection])) if selection else None,
        "mean_wmape": float(np.mean([row["wmape"] for row in selection])) if selection else None,
        "folds": aggregates,
    }
    metrics_doc = {
        "schema_version": 1,
        "run_id": run_id,
        "selection_folds": [r for r in aggregates if r["role"] == "selection"],
        "diagnostic_folds": [r for r in aggregates if r["role"] != "selection"],
        "aggregate": aggregate,
        "slices": slices,
    }
    pred_csv = pd.DataFrame(predictions, columns=["run_id", "config_hash", "fold", "load_id", "date", "actual_rate", "predicted_rate", "absolute_error"]).to_csv(index=False)
    config_doc = {
        "schema_version": 1,
        "run_id": run_id,
        "config": full_config,
        "config_hash": chash,
        "folds": [fold_by_name(name).__dict__ for name in config.get("folds", [])],
        "eligibility": "eligible" if config.get("feature_view") == "december_compatible" else "diagnostic",
        "data_hashes": data_hashes(data_paths or {}) if data_paths else {},
        "source_hashes": source_hashes(root),
    }
    workers = int(config.get("workers", dict(config.get("parameters", {})).get("thread_count", 1)))
    files = {
        "config.json": json.dumps(config_doc, indent=2, default=str),
        "predictions.csv": pred_csv,
        "metrics.json": json.dumps(metrics_doc, indent=2, default=str),
        "runtime.json": json.dumps(runtime_info(start, workers), indent=2, default=str),
        "status.json": json.dumps(status, indent=2),
    }
    run_dir = atomic_run_dir(artifact_root, run_id, files)
    append_ledger(artifact_root, {"run_id": run_id, "status": status["status"], "model": config.get("model"), "target": config.get("target"), "feature_view": config.get("feature_view"), **aggregate})
    return RunResult(run_id, status["status"], run_dir, aggregate)


def record_skipped_run(config: Mapping[str, Any], artifact_root: Path, reason_code: str, message: str) -> RunResult:
    start = time.time()
    full_config = {"schema_version": 1, **dict(config)}
    chash = config_hash(full_config)
    run_id = f"{time.time_ns()}-{chash}"
    metrics_doc = {"schema_version": 1, "run_id": run_id, "selection_folds": [], "diagnostic_folds": [], "aggregate": {"selection_folds": [], "mean_mae": None, "mean_rmse": None, "mean_wmape": None, "folds": []}, "slices": []}
    config_doc = {
        "schema_version": 1,
        "run_id": run_id,
        "config": full_config,
        "config_hash": chash,
        "folds": [],
        "eligibility": "eligible" if config.get("feature_view") == "december_compatible" else "diagnostic",
        "data_hashes": {},
        "source_hashes": source_hashes(Path.cwd()),
    }
    status = {"status": "skipped", "reason_code": reason_code, "error_type": None, "message": message[:500]}
    files = {
        "config.json": json.dumps(config_doc, indent=2, default=str),
        "predictions.csv": pd.DataFrame(columns=["run_id", "config_hash", "fold", "load_id", "date", "actual_rate", "predicted_rate", "absolute_error"]).to_csv(index=False),
        "metrics.json": json.dumps(metrics_doc, indent=2, default=str),
        "runtime.json": json.dumps(runtime_info(start, int(config.get("workers", 1))), indent=2, default=str),
        "status.json": json.dumps(status, indent=2),
    }
    run_dir = atomic_run_dir(artifact_root, run_id, files)
    append_ledger(artifact_root, {"run_id": run_id, "status": "skipped", "model": config.get("model"), "target": config.get("target"), "feature_view": config.get("feature_view")})
    return RunResult(run_id, "skipped", run_dir, metrics_doc["aggregate"])


def load_completed_runs(artifact_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((artifact_root / "runs").glob("*/metrics.json")):
        status = json.loads((path.parent / "status.json").read_text())
        config = json.loads((path.parent / "config.json").read_text())
        metrics = json.loads(path.read_text())
        if status["status"] == "completed":
            rows.append({"run_id": config["run_id"], "config": config["config"], **metrics["aggregate"]})
    return rows


def rank_runs(artifact_root: Path) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in load_completed_runs(artifact_root)
            if row["config"].get("feature_view") == "december_compatible" and row.get("mean_mae") is not None
        ],
        key=lambda r: (
            r["mean_mae"],
            max(f["mae"] for f in r["folds"] if f["role"] == "selection"),
            r["mean_rmse"],
            r["mean_wmape"],
        ),
    )
