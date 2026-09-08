from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import existing_path, load_training
from src.experiments import rank_runs, record_skipped_run, run_experiment
from src.models import available_optional_models, predefined_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=existing_path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("experiments"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-wall-minutes", type=float, default=45.0)
    parser.add_argument("--model", choices=["all", "extra_trees", "xgboost", "catboost"], default="all")
    args = parser.parse_args()
    rows = load_training(args.data)
    optional = available_optional_models()
    print("OPTIONAL_MODELS", optional)
    if args.model in {"all", "xgboost"} and optional.get("xgboost") != "available":
        skipped = record_skipped_run(
            {"model": "xgboost", "target": "direct", "feature_view": "december_compatible", "parameters": {}, "seed": args.seed, "workers": args.workers},
            args.artifact_root,
            "import_error",
            optional.get("xgboost", "not_installed"),
        )
        print("SKIPPED", skipped.run_id, "xgboost", optional.get("xgboost"))
    deadline = time.time() + args.max_wall_minutes * 60
    configs = [config for config in predefined_configs(args.seed, args.workers)[4:] if args.model == "all" or config["model"] == args.model]
    if not configs:
        print("NO_CONFIGS", args.model)
    for config in configs:
        if time.time() > deadline:
            print("SKIP budget_exhausted", config["model"], config["target"])
            break
        result = run_experiment(rows, config, args.artifact_root, data_paths={"training": args.data})
        print(result.run_id, result.status, result.aggregate)
    print("RANKING")
    for row in rank_runs(args.artifact_root)[:10]:
        print(row["run_id"], row["config"]["model"], row["config"]["target"], row["config"]["feature_view"], row["mean_mae"], row["mean_rmse"], row["mean_wmape"])


if __name__ == "__main__":
    main()
