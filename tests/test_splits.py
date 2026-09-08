import unittest
from dataclasses import FrozenInstanceError
from datetime import date

from src.splits import (
    FoldSpec,
    development_folds,
    development_rows,
    holdout_fold,
    horizon_fold,
    split_rows,
)


def row(load_id, value_date, posted_rate="2000"):
    return {
        "load_id": load_id,
        "date": value_date,
        "posted_rate": posted_rate,
        "pickup": "Atlanta",
        "delivery": "Chicago",
    }


class FoldSpecTests(unittest.TestCase):
    def test_fold_specs_are_exact_and_immutable(self):
        folds = development_folds()

        self.assertEqual(
            folds,
            (
                FoldSpec("july_1m", date(2025, 1, 1), date(2025, 6, 30), date(2025, 7, 1), date(2025, 7, 31), "selection"),
                FoldSpec("august_1m", date(2025, 1, 1), date(2025, 7, 31), date(2025, 8, 1), date(2025, 8, 31), "selection"),
                FoldSpec("september_1m", date(2025, 1, 1), date(2025, 8, 31), date(2025, 9, 1), date(2025, 9, 30), "selection"),
            ),
        )
        self.assertEqual(
            horizon_fold(),
            FoldSpec("september_2m", date(2025, 1, 1), date(2025, 7, 31), date(2025, 9, 1), date(2025, 9, 30), "diagnostic"),
        )
        self.assertEqual(holdout_fold().role, "holdout")
        with self.assertRaises(FrozenInstanceError):
            folds[0].name = "changed"

    def test_split_rows_preserves_order_and_does_not_mutate_or_inspect_targets(self):
        rows = [
            row("L1", "2025-01-01", posted_rate=object()),
            row("L2", "2025-06-30", posted_rate=object()),
            row("L3", "2025-07-01", posted_rate=object()),
            row("L4", "2025-07-31", posted_rate=object()),
            row("L5", "2025-08-01", posted_rate=object()),
        ]

        train, validation = split_rows(rows, development_folds()[0])

        self.assertEqual([item["load_id"] for item in train], ["L1", "L2"])
        self.assertEqual([item["load_id"] for item in validation], ["L3", "L4"])
        self.assertIs(train[0], rows[0])
        self.assertIs(validation[0], rows[2])

    def test_split_rows_rejects_invalid_inputs_and_bad_chronology(self):
        valid_fold = development_folds()[0]
        invalid_fold = FoldSpec("bad", date(2025, 7, 1), date(2025, 7, 31), date(2025, 7, 1), date(2025, 7, 31), "selection")

        cases = [
            ([], valid_fold, "empty"),
            ([row("L1", "2025-01-01"), row("L1", "2025-07-01")], valid_fold, "Duplicate load_id"),
            ([row("L1", "not-a-date"), row("L2", "2025-07-01")], valid_fold, "Unparseable date"),
            ([row("L1", "2025-07-01")], valid_fold, "empty train"),
            ([row("L1", "2025-01-01")], valid_fold, "empty validation"),
            ([row("L1", "2025-01-01"), row("L2", "2025-07-01")], invalid_fold, "before validation"),
        ]

        for rows, fold, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(ValueError, expected_error):
                    split_rows(rows, fold)

    def test_october_holdout_requires_explicit_access(self):
        rows = [row("L1", "2025-01-01"), row("L2", "2025-10-01")]

        with self.assertRaisesRegex(PermissionError, "October"):
            split_rows(rows, holdout_fold())

        train, validation = split_rows(rows, holdout_fold(), allow_holdout=True)
        self.assertEqual([item["load_id"] for item in train], ["L1"])
        self.assertEqual([item["load_id"] for item in validation], ["L2"])

    def test_development_rows_drops_october_without_reading_targets_and_rejects_domain_drift(self):
        rows = [
            row("L1", "2025-09-30", posted_rate=object()),
            row("L2", "2025-10-01", posted_rate=object()),
        ]

        result = development_rows(rows)

        self.assertEqual([item["load_id"] for item in result], ["L1"])
        with self.assertRaisesRegex(ValueError, "outside supported 2025 Jan-Oct domain"):
            development_rows([row("L1", "2024-12-31")])
        with self.assertRaisesRegex(ValueError, "outside supported 2025 Jan-Oct domain"):
            development_rows([row("L1", "2025-11-01")])


if __name__ == "__main__":
    unittest.main()
