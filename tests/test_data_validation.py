import copy
import tempfile
import unittest
from pathlib import Path

from src.data_validation import SchemaValidationError, load_csv_rows, validate_raw_frame


def training_row(**overrides):
    row = {
        "load_id  ": "L1",
        " pickup        ": " Atlanta ",
        " delivery      ": "Chicago",
        " pickup_lat": "33.7490",
        " pickup_lon": "-84.3880",
        " delivery_lat": "41.8781",
        " delivery_lon": "-87.6298",
        " distance": "716",
        " equipment": "Van",
        " weight  ": "42000",
        " date      ": "2025-10-01",
        " market_index": "101.5",
        " quote_signal": "0.25",
        " posted_rate": "2300",
    }
    row.update(overrides)
    return row


class DataValidationTests(unittest.TestCase):
    def test_training_schema_accepts_padded_headers_without_mutating_rows(self):
        rows = [training_row()]
        original = copy.deepcopy(rows)

        validated = validate_raw_frame(rows, schema="training")

        self.assertEqual(rows, original)
        self.assertEqual(validated[0]["load_id"], "L1")
        self.assertEqual(validated[0]["pickup"], "Atlanta")

    def test_validation_fails_for_missing_unexpected_and_duplicate_ids(self):
        with self.assertRaisesRegex(SchemaValidationError, "unexpected columns"):
            validate_raw_frame([training_row(extra="x")], schema="training")

        with self.assertRaisesRegex(SchemaValidationError, "missing columns"):
            bad = training_row()
            bad.pop(" posted_rate")
            validate_raw_frame([bad], schema="training")

        with self.assertRaisesRegex(SchemaValidationError, "Duplicate load_id"):
            validate_raw_frame([training_row(), training_row()], schema="training")

    def test_validation_fails_for_normalized_header_collisions(self):
        bad = training_row()
        bad["pickup"] = "Dallas"

        with self.assertRaisesRegex(SchemaValidationError, "normalize to the same column"):
            validate_raw_frame([bad], schema="training")

    def test_validation_fails_for_invalid_load_ids(self):
        for load_id in (None, "", 123):
            with self.subTest(load_id=load_id):
                with self.assertRaisesRegex(SchemaValidationError, "Invalid load_id"):
                    validate_raw_frame([training_row(**{"load_id  ": load_id})], schema="training")

    def test_validation_fails_for_invalid_values(self):
        invalid_cases = [
            ({" date      ": "10/01/2025"}, "Unparseable date"),
            ({" pickup_lat": "95"}, "Invalid coordinate"),
            ({" distance": "0"}, "distance must be positive"),
            ({" equipment": ""}, "Missing categorical"),
            ({" posted_rate": "0"}, "posted_rate must be positive"),
            ({" quote_signal": "nan"}, "Non-finite numeric"),
        ]

        for overrides, expected_error in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(SchemaValidationError, expected_error):
                    validate_raw_frame([training_row(**overrides)], schema="training")

    def test_full_validation_and_december_schemas_validate(self):
        full_validation = training_row()
        full_validation.pop(" posted_rate")
        december = {
            "pickup": "Atlanta",
            "delivery": "Chicago",
            "distance": "716",
            "equipment": "Van",
            "weight": "42000",
            "date": "2025-12-01",
            "predicted_rate": "",
        }

        self.assertEqual(
            validate_raw_frame([full_validation], schema="full_validation")[0]["load_id"],
            "L1",
        )
        self.assertEqual(
            validate_raw_frame([december], schema="december")[0]["date"],
            "2025-12-01",
        )

    def test_canonical_raw_data_files_load_from_data_directory(self):
        raw_data_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
        expected = [
            ("Train_Test.csv", "training", 48000),
            ("Validation.csv", "full_validation", 12000),
            ("December_Chart_Inputs.csv", "december", 31),
        ]

        for filename, schema, expected_rows in expected:
            with self.subTest(filename=filename):
                rows = load_csv_rows(raw_data_dir / filename, schema=schema)

                self.assertEqual(len(rows), expected_rows)

    def test_raw_assessment_files_are_not_kept_in_project_root(self):
        allowed_outputs = {
            "validation_predictions.csv",
            "December_Chart_Predictions.csv",
        }
        root_files = {
            path.name
            for path in Path(".").glob("*.csv")
            if path.name not in allowed_outputs
        }

        self.assertFalse(root_files)

    def test_csv_loader_rejects_duplicate_headers_before_dict_reader_can_collapse_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("load_id,load_id\nL1,L2\n", encoding="utf-8")

            with self.assertRaisesRegex(SchemaValidationError, "Duplicate CSV header"):
                load_csv_rows(path, schema="training")

    def test_csv_loader_rejects_header_only_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text(
                "load_id,pickup,delivery,pickup_lat,pickup_lon,delivery_lat,"
                "delivery_lon,distance,equipment,weight,date,market_index,"
                "quote_signal,posted_rate\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SchemaValidationError, "contains no data rows"):
                load_csv_rows(path, schema="training")


if __name__ == "__main__":
    unittest.main()
