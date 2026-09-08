from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import existing_path, load_training
from src.artifacts import file_sha256, write_json
from src.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=existing_path, required=True)
    parser.add_argument("--lock", type=existing_path, required=True)
    parser.add_argument("--access-marker", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    if args.access_marker.exists():
        raise SystemExit("holdout access marker already exists")
    args.access_marker.parent.mkdir(parents=True, exist_ok=True)
    lock = json.loads(args.lock.read_text())
    rows = load_training(args.data)
    results = {}
    for label in ("winner_run_id", "comparator_run_id"):
        config = lock["runs"][lock[label]]["config"]["config"]
        holdout_config = {**config, "folds": ["october_holdout"]}
        result = run_experiment(rows, holdout_config, args.artifact_root, data_paths={"training": args.data}, allow_holdout=True)
        print(label, result)
        results[label] = result
    all_completed = all(r.status == "completed" for r in results.values())
    if not all_completed:
        failed = [label for label, r in results.items() if r.status != "completed"]
        raise SystemExit(f"holdout evaluation failed for: {', '.join(failed)}")
    write_json(args.access_marker, {
        "schema_version": 1,
        "lock_hash": lock["lock_hash"],
        "created_at": time.time(),
        "purpose": "one-time October confirmation",
        "data_hash": file_sha256(args.data),
        "status": "completed",
        "results": {label: {"run_id": r.run_id, "status": r.status} for label, r in results.items()},
    })


if __name__ == "__main__":
    main()
