from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.artifacts import config_hash, source_hashes, write_json
from src.models import make_estimator
from src.preprocessing import FreightPreprocessor
from src.targets import target_adapter


@dataclass
class FittedRateBundle:
    bundle_version: str
    preprocessor: FreightPreprocessor
    estimator: Any
    target_name: str
    feature_columns: tuple[str, ...]
    feature_dtypes: dict[str, str]
    locked_config: dict[str, Any]
    training_window: dict[str, str]
    hashes: dict[str, Any]


def fit_bundle(rows: list[Mapping[str, object]], config: Mapping[str, Any], hashes: Mapping[str, Any] | None = None) -> FittedRateBundle:
    pre = FreightPreprocessor(feature_view=str(config["feature_view"])).fit(list(rows))
    x = pre.transform_frame(list(rows))
    y = np.asarray([float(row["posted_rate"]) for row in rows], dtype=float)
    adapter = target_adapter(str(config["target"]))
    estimator = make_estimator(str(config["model"]), dict(config.get("parameters", {})))
    estimator.fit(x, adapter.transform(y, x))
    return FittedRateBundle(
        "1",
        pre,
        estimator,
        adapter.name,
        tuple(x.columns),
        {c: str(t) for c, t in x.dtypes.items()},
        dict(config),
        {"start": str(min(row["date"] for row in rows)), "end": str(max(row["date"] for row in rows))},
        dict(hashes or {}),
    )


def predict_rates(bundle: FittedRateBundle, rows: list[Mapping[str, object]]) -> pd.DataFrame:
    x = bundle.preprocessor.transform_frame(list(rows))
    if tuple(x.columns) != bundle.feature_columns:
        raise ValueError("feature column mismatch")
    adapter = target_adapter(bundle.target_name)
    pred = adapter.inverse_transform(bundle.estimator.predict(x), x)
    if (pred <= 0).any() or not np.isfinite(pred).all():
        raise ValueError("predictions must be positive finite")
    data: dict[str, Any] = {"predicted_rate": pred}
    if rows and "load_id" in rows[0]:
        data["load_id"] = [row["load_id"] for row in rows]
    if rows and "date" in rows[0]:
        data["date"] = [row["date"] for row in rows]
    return pd.DataFrame(data)


def save_bundle(bundle: FittedRateBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> FittedRateBundle:
    return joblib.load(path)


def write_submission_outputs(bundle: FittedRateBundle, validation_rows: list[Mapping[str, object]], template_path: Path, december_rows: list[Mapping[str, object]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "validation_predictions.csv"
    december_path = output_dir / "December_Chart_Predictions.csv"
    if validation_path.exists() or december_path.exists():
        raise FileExistsError("submission output collision")
    template = pd.read_csv(template_path)
    template = template.assign(_order=np.arange(len(template)))
    preds = predict_rates(bundle, validation_rows)[["load_id", "predicted_rate"]]
    if preds["load_id"].duplicated().any() or template["load_id"].duplicated().any():
        raise ValueError("duplicate load_id in join")
    template_ids = set(template["load_id"].astype(str))
    pred_ids = set(preds["load_id"].astype(str))
    missing = template_ids - pred_ids
    extra = pred_ids - template_ids
    if missing or extra:
        raise ValueError(f"ID mismatch: {len(missing)} missing, {len(extra)} extra")
    merged = template.merge(preds, on="load_id", how="left", validate="one_to_one", suffixes=("", "_model"))
    rate_col = "predicted_rate_model" if "predicted_rate_model" in merged.columns else "predicted_rate"
    if merged[rate_col].isna().any():
        raise ValueError("missing validation predictions")
    merged.sort_values("_order")[["load_id", rate_col]].rename(columns={rate_col: "predicted_rate"}).to_csv(validation_path, index=False)
    dec = pd.DataFrame(december_rows).copy()
    dec_preds = predict_rates(bundle, december_rows)[["date", "predicted_rate"]]
    if dec["date"].duplicated().any() or dec_preds["date"].duplicated().any():
        raise ValueError("duplicate December date")
    dec = dec.drop(columns=["predicted_rate"], errors="ignore").merge(dec_preds, on="date", how="left", validate="one_to_one")
    dec[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]].to_csv(december_path, index=False)
    return validation_path, december_path


def fresh_process_reload_check(bundle_path: Path, validation_rows: list[Mapping[str, object]], *, expected_fingerprint: tuple[int, float] | None = None) -> None:
    sample_path = bundle_path.parent / "reload_sample.json"
    sample_path.write_text(json.dumps(list(validation_rows[:50])), encoding="utf-8")
    code = (
        "import json,joblib; from pathlib import Path; "
        "from src.submission import predict_rates; "
        f"b=joblib.load(Path(r'{bundle_path}')); "
        f"rows=json.loads(Path(r'{sample_path}').read_text()); "
        "p=predict_rates(b, rows); print(len(p), round(float(p.predicted_rate.sum()), 6))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=Path.cwd(), text=True, capture_output=True, check=True)
    parts = result.stdout.strip().split()
    if len(parts) != 2:
        raise RuntimeError(f"fresh process reload check failed: unexpected output format: {result.stdout.strip()!r}")
    try:
        row_count = int(parts[0])
        pred_sum = float(parts[1])
    except ValueError as exc:
        raise RuntimeError(f"fresh process reload check failed: could not parse output: {result.stdout.strip()!r}") from exc
    if expected_fingerprint is not None:
        expected_count, expected_sum = expected_fingerprint
        if row_count != expected_count or abs(pred_sum - expected_sum) > 0.001:
            raise RuntimeError(f"fresh process reload check failed: expected ({expected_count}, {expected_sum}), got ({row_count}, {pred_sum})")
    elif row_count != 50:
        raise RuntimeError(f"fresh process reload check failed: expected 50 rows, got {row_count}")


def run_scorer(predictions: Path, december_predictions: Path, output_dir: Path) -> dict[str, Any]:
    result = subprocess.run([sys.executable, "scorer.py", "--predictions", str(predictions), "--december-predictions", str(december_predictions), "--output-dir", str(output_dir)], text=True, capture_output=True, cwd=Path.cwd(), check=False)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "chart": str(output_dir / "candidate_december.png")}


def freeze_selection(artifact_root: Path, winner_run_id: str, comparator_run_id: str) -> Path:
    out = artifact_root / "selection" / "final_locked.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    runs = {}
    for run_id in (winner_run_id, comparator_run_id):
        run_dir = artifact_root / "runs" / run_id
        runs[run_id] = {"config": json.loads((run_dir / "config.json").read_text()), "metrics": json.loads((run_dir / "metrics.json").read_text())}
    for run_id, data in runs.items():
        status_path = artifact_root / "runs" / run_id / "status.json"
        status = json.loads(status_path.read_text())
        if status["status"] != "completed":
            raise ValueError(f"cannot freeze run {run_id}: status is {status['status']!r}")
        if data["config"].get("config", {}).get("feature_view") != "december_compatible":
            raise ValueError(f"cannot freeze run {run_id}: feature_view is {data['config'].get('config', {}).get('feature_view')!r}")
        folds = data["metrics"].get("selection_folds", [])
        if not folds:
            raise ValueError(f"cannot freeze run {run_id}: no selection folds completed")
        if data["metrics"].get("aggregate", {}).get("mean_mae") is None:
            raise ValueError(f"cannot freeze run {run_id}: no valid mean_mae")
    winner_mae = runs[winner_run_id]["metrics"]["aggregate"]["mean_mae"]
    comparator_mae = runs[comparator_run_id]["metrics"]["aggregate"]["mean_mae"]
    if winner_mae is None or comparator_mae is None:
        raise ValueError("cannot freeze: one or both runs have no valid mean_mae")
    if winner_mae >= comparator_mae:
        raise ValueError(f"cannot freeze: winner MAE ({winner_mae}) is not better than comparator MAE ({comparator_mae})")
    lock = {
        "schema_version": 1,
        "winner_run_id": winner_run_id,
        "comparator_run_id": comparator_run_id,
        "runs": runs,
        "source_hashes": source_hashes(Path.cwd()),
        "lock_hash_input": config_hash({"winner": winner_run_id, "comparator": comparator_run_id}),
    }
    lock["lock_hash"] = config_hash(lock)
    write_json(out, lock)
    return out
