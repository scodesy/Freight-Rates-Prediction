from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_validation import validate_raw_frame
from src.models import make_estimator
from src.preprocessing import FreightPreprocessor
from src.submission import fit_bundle, predict_rates, save_bundle, write_submission_outputs
from src.targets import target_adapter


def row(load_id: str, date: str = "2025-01-01", rate: str = "1000", equipment: str = "Van") -> dict[str, str]:
    return {
        "load_id": load_id,
        "pickup": "Atlanta",
        "delivery": "Chicago",
        "pickup_lat": "33.7490",
        "pickup_lon": "-84.3880",
        "delivery_lat": "41.8781",
        "delivery_lon": "-87.6298",
        "distance": "500",
        "equipment": equipment,
        "weight": "42000",
        "date": date,
        "market_index": "100",
        "quote_signal": "0.2",
        "posted_rate": rate,
    }


class TargetModelSubmissionTests(unittest.TestCase):
    def test_targets_are_reversible_and_validate_distance(self) -> None:
        x = pd.DataFrame({"distance": [100.0, 200.0]})
        y = np.asarray([1000.0, 2500.0])
        for name in ("direct", "log", "rpm"):
            adapter = target_adapter(name)
            np.testing.assert_allclose(adapter.inverse_transform(adapter.transform(y, x), x), y)
        with self.assertRaisesRegex(ValueError, "positive finite distance"):
            target_adapter("rpm").transform(y, pd.DataFrame({"distance": [0.0, 1.0]}))

    def test_baseline_models_preserve_rows_unknown_equipment_and_serialize(self) -> None:
        rows = validate_raw_frame([row("L1", rate="1000", equipment="Van"), row("L2", rate="3000", equipment="Reefer")], schema="training")
        processor = FreightPreprocessor(feature_view="december_compatible").fit(rows)
        x = processor.transform_frame(rows)
        y = np.asarray([1000.0, 3000.0])
        for name in ("median", "global_rpm", "equipment_rpm", "ridge"):
            estimator = make_estimator(name, {"alpha": 1.0} if name == "ridge" else {}).fit(x, y)
            loaded = pickle.loads(pickle.dumps(estimator))
            pred = loaded.predict(x)
            self.assertEqual(len(pred), 2)
            self.assertTrue(np.isfinite(pred).all())

    def test_catboost_uses_native_categoricals_unknowns_and_serializes(self) -> None:
        try:
            import catboost  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"catboost unavailable: {type(exc).__name__}")

        train_rows = validate_raw_frame(
            [
                row("L1", rate="1000", equipment="Van"),
                row("L2", rate="3000", equipment="Reefer"),
                row("L3", rate="1800", equipment="Flatbed"),
            ],
            schema="training",
        )
        valid_rows = validate_raw_frame([row("L4", rate="2000", equipment="Stepdeck")], schema="training")
        processor = FreightPreprocessor(feature_view="december_compatible").fit(train_rows)
        x_train = processor.transform_frame(train_rows)
        x_valid = processor.transform_frame(valid_rows)
        estimator = make_estimator(
            "catboost",
            {"iterations": 2, "depth": 2, "learning_rate": 0.1, "random_seed": 42, "thread_count": 1, "loss_function": "RMSE", "eval_metric": "MAE", "verbose": False},
        ).fit(x_train, np.asarray([1000.0, 3000.0, 1800.0]))

        loaded = pickle.loads(pickle.dumps(estimator))
        pred = loaded.predict(x_valid)

        self.assertEqual(getattr(loaded, "cat_feature_names_", None), ("pickup", "delivery", "equipment"))
        self.assertEqual(len(pred), 1)
        self.assertTrue(np.isfinite(pred).all())
        self.assertGreater(float(pred[0]), 0.0)

    def test_submission_outputs_join_by_keys_not_row_position(self) -> None:
        train = validate_raw_frame([row("L1", rate="1000"), row("L2", date="2025-02-01", rate="1200")], schema="training")
        validation_raw = [row("TE-000002"), row("TE-000001")]
        for item in validation_raw:
            item.pop("posted_rate")
        validation = validate_raw_frame(validation_raw, schema="full_validation")
        december = validate_raw_frame(
            [
                {"pickup": "Atlanta", "delivery": "Chicago", "distance": "500", "equipment": "Van", "weight": "42000", "date": "2025-12-02", "predicted_rate": ""},
                {"pickup": "Atlanta", "delivery": "Chicago", "distance": "500", "equipment": "Van", "weight": "42000", "date": "2025-12-01", "predicted_rate": ""},
            ],
            schema="december",
        )
        bundle = fit_bundle(train, {"model": "median", "target": "direct", "feature_view": "december_compatible", "parameters": {}})
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template = root / "template.csv"
            template.write_text("load_id,predicted_rate\nTE-000001,\nTE-000002,\n", encoding="utf-8")
            save_bundle(bundle, root / "bundle.joblib")
            out_validation, out_december = write_submission_outputs(bundle, validation, template, december, root / "out")
            self.assertEqual(pd.read_csv(out_validation)["load_id"].tolist(), ["TE-000001", "TE-000002"])
            self.assertEqual(pd.read_csv(out_december)["date"].tolist(), ["2025-12-02", "2025-12-01"])
            self.assertEqual(len(predict_rates(bundle, validation)), 2)


if __name__ == "__main__":
    unittest.main()
