from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import existing_path, load_training
from src.data_validation import load_csv_rows
from src.experiments import rank_runs, run_experiment
from src.submission import fit_bundle, fresh_process_reload_check, run_scorer, save_bundle, write_submission_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=existing_path, required=True)
    parser.add_argument("--validation", type=existing_path, required=True)
    parser.add_argument("--template", type=existing_path, required=True)
    parser.add_argument("--december", type=existing_path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("experiments"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    rows = load_training(args.data)
    common = {"feature_view": "december_compatible", "target": "direct", "seed": 42, "workers": args.workers, "folds": ["july_1m", "august_1m", "september_1m", "september_2m"]}
    configs = [
        {**common, "model": "median", "parameters": {}},
        {**common, "model": "global_rpm", "parameters": {}},
        {**common, "model": "equipment_rpm", "parameters": {}},
        {**common, "model": "ridge", "parameters": {"alpha": 10.0}},
    ]
    paths = {"training": args.data, "validation": args.validation, "template": args.template, "december": args.december}
    for config in configs:
        result = run_experiment(rows, config, args.artifact_root, data_paths=paths)
        print(result.run_id, result.status, result.aggregate)
    best = rank_runs(args.artifact_root)[0]
    pre_oct = [row for row in rows if str(row["date"]) < "2025-10-01"]
    bundle = fit_bundle(pre_oct, best["config"], {"data_hashes": {k: str(v) for k, v in paths.items()}})
    proof = args.artifact_root / "baseline_proof" / best["run_id"]
    save_bundle(bundle, proof / "model_bundle.joblib")
    validation = load_csv_rows(args.validation, schema="full_validation")
    december = load_csv_rows(args.december, schema="december")
    validation_csv, december_csv = write_submission_outputs(bundle, validation, args.template, december, proof)
    fresh_process_reload_check(proof / "model_bundle.joblib", validation)
    scorer = run_scorer(validation_csv, december_csv, proof / "scorer_results")
    print("BASELINE_PROOF", best["run_id"], scorer)


if __name__ == "__main__":
    main()
