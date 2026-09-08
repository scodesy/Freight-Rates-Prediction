import math
import unittest
from dataclasses import FrozenInstanceError

from src.metrics import regression_metrics, slice_metrics, validate_predictions


def freight_row(**overrides):
    row = {
        "pickup": "Atlanta",
        "delivery": "Chicago",
        "equipment": "Van",
        "distance": "600",
        "weight": "42000",
        "market_index": "100",
    }
    row.update(overrides)
    return row


class RegressionMetricsTests(unittest.TestCase):
    def test_regression_metrics_compute_count_mae_rmse_and_wmape(self):
        metrics = regression_metrics([100.0, 200.0, 300.0], [110.0, 190.0, 330.0])

        self.assertEqual(metrics.count, 3)
        self.assertAlmostEqual(metrics.mae, 50.0 / 3.0)
        self.assertAlmostEqual(metrics.rmse, math.sqrt((100.0 + 100.0 + 900.0) / 3.0))
        self.assertAlmostEqual(metrics.wmape, 50.0 / 600.0)
        with self.assertRaises(FrozenInstanceError):
            metrics.count = 10

    def test_regression_metrics_reject_invalid_sequences(self):
        cases = [
            ([], [], "empty"),
            ([1.0], [1.0, 2.0], "equal length"),
            ([[1.0]], [[1.0]], "1D"),
            ([1.0, float("nan")], [1.0, 2.0], "finite"),
            ([0.0], [1.0], "zero WMAPE"),
        ]

        for y_true, y_pred, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    regression_metrics(y_true, y_pred)

    def test_validate_predictions_requires_finite_strictly_positive_values(self):
        self.assertEqual(validate_predictions([1, 2.5, "3"]), (1.0, 2.5, 3.0))
        for values in ([0.0], [-1.0], [float("inf")], ["not numeric"]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_predictions(values)

    def test_slice_metrics_returns_report_only_slices_with_suppression(self):
        rows = []
        y_true = []
        y_pred = []
        for index in range(30):
            rows.append(
                freight_row(
                    pickup="Atlanta",
                    delivery="Chicago",
                    equipment="Van",
                    distance="100" if index < 10 else "600",
                    weight="" if index < 5 else ("-42000" if index < 10 else "42000"),
                    market_index="" if index < 8 else "100",
                )
            )
            y_true.append(1000.0 + index)
            y_pred.append(1005.0 + index)
        for index in range(70):
            rows.append(
                freight_row(
                    pickup="Dallas",
                    delivery="Miami",
                    equipment="Reefer",
                    distance="1500" if index < 35 else "3000",
                )
            )
            y_true.append(2000.0 + index)
            y_pred.append(2010.0 + index)

        slices = slice_metrics(rows, y_true, y_pred, {("Atlanta", "Chicago")}, minimum_count=25)
        by_key = {(item["family"], item["value"]): item for item in slices}

        self.assertEqual(by_key[("equipment", "Van")]["count"], 30)
        self.assertAlmostEqual(by_key[("equipment", "Van")]["mae"], 5.0)
        self.assertEqual(by_key[("route", "seen")]["count"], 30)
        self.assertEqual(by_key[("route", "unseen")]["count"], 70)
        self.assertEqual(by_key[("weight_missing", "missing")]["count"], 5)
        self.assertIsNone(by_key[("weight_missing", "missing")]["mae"])
        self.assertEqual(by_key[("weight_negative", "negative")]["count"], 5)
        self.assertEqual(by_key[("market_index", "missing")]["count"], 8)
        self.assertEqual(by_key[("high_value", "top10")]["count"], 10)
        self.assertEqual(by_key[("high_value", "rest")]["count"], 90)
        self.assertIn(("distance", "1500+") , by_key)

    def test_slice_metrics_rejects_length_mismatch_and_invalid_minimum_count(self):
        rows = [freight_row()]
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            slice_metrics(rows, [1.0], [1.0, 2.0], set())
        with self.assertRaisesRegex(ValueError, "minimum_count"):
            slice_metrics(rows, [1.0], [1.0], set(), minimum_count=0)


if __name__ == "__main__":
    unittest.main()
