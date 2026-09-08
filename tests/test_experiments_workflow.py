from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.experiments import rank_runs, run_experiment
from src.submission import freeze_selection


def row(load_id: str, date: str, rate: float) -> dict[str, str]:
    return {
        "load_id": load_id,
        "pickup": "Atlanta",
        "delivery": "Chicago",
        "pickup_lat": "33.7490",
        "pickup_lon": "-84.3880",
        "delivery_lat": "41.8781",
        "delivery_lon": "-87.6298",
        "distance": "500",
        "equipment": "Van",
        "weight": "42000",
        "date": date,
        "market_index": "100",
        "quote_signal": "0.2",
        "posted_rate": str(rate),
    }


class ExperimentWorkflowTests(unittest.TestCase):
    def test_run_experiment_writes_contract_artifacts_and_excludes_holdout_by_default(self) -> None:
        rows = [row("L1", "2025-01-01", 1000), row("L2", "2025-07-01", 1100), row("L3", "2025-10-01", 900)]
        config = {"model": "median", "target": "direct", "feature_view": "december_compatible", "parameters": {}, "folds": ["july_1m"]}
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_experiment(rows, config, Path(tmpdir))
            self.assertEqual(result.status, "completed")
            self.assertTrue((result.run_dir / "config.json").exists())
            self.assertTrue((result.run_dir / "predictions.csv").read_text().startswith("run_id,config_hash,fold"))
            denied = run_experiment(rows, {**config, "folds": ["october_holdout"]}, Path(tmpdir))
            self.assertEqual(denied.status, "failed")
            self.assertIn("PermissionError", (denied.run_dir / "status.json").read_text())

    def test_rank_and_freeze_are_development_artifact_only(self) -> None:
        rows = [row("L1", "2025-01-01", 1000), row("L2", "2025-07-01", 1100), row("L3", "2025-08-01", 1200)]
        config = {"model": "median", "target": "direct", "feature_view": "december_compatible", "parameters": {}, "folds": ["july_1m"]}
        noisy_rows = [row("L1", "2025-01-01", 5000), row("L2", "2025-07-01", 6000), row("L3", "2025-08-01", 7000)]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            good = run_experiment(rows, config, root)
            bad = run_experiment(noisy_rows, config, root)
            self.assertGreaterEqual(len(rank_runs(root)), 2)
            lock = freeze_selection(root, good.run_id, bad.run_id)
            data = json.loads(lock.read_text())
            self.assertEqual(data["winner_run_id"], good.run_id)
            with self.assertRaises(FileExistsError):
                freeze_selection(root, good.run_id, bad.run_id)


if __name__ == "__main__":
    unittest.main()
