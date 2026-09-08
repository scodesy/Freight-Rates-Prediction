import copy
import math
import pickle
import unittest

from src.data_validation import validate_raw_frame
from src.preprocessing import FreightPreprocessor


def row(
    load_id,
    pickup="Atlanta",
    delivery="Chicago",
    equipment="Van",
    weight="42000",
    market_index="100",
    date="2025-10-01",
    distance="716",
):
    return {
        "load_id": load_id,
        "pickup": pickup,
        "delivery": delivery,
        "pickup_lat": "33.7490",
        "pickup_lon": "-84.3880",
        "delivery_lat": "41.8781",
        "delivery_lon": "-87.6298",
        "distance": distance,
        "equipment": equipment,
        "weight": weight,
        "date": date,
        "market_index": market_index,
        "quote_signal": "0.25",
        "posted_rate": "2300",
    }


class PreprocessingTests(unittest.TestCase):
    def test_weight_cleaning_indicators_and_fold_local_imputation(self):
        train = validate_raw_frame(
            [
                row("L1", equipment="Van", weight="40000", market_index="100"),
                row("L2", equipment="Flatbed", weight="50000", market_index="200"),
            ],
            schema="training",
        )
        validation = validate_raw_frame(
            [row("L3", equipment="Van", weight="-41000", market_index="999")],
            schema="training",
        )
        missing = validate_raw_frame(
            [row("L4", equipment="Van", weight="", market_index="")],
            schema="training",
        )

        processor = FreightPreprocessor(feature_view="full").fit(train)
        transformed_negative = processor.transform(validation)[0]
        transformed_missing = processor.transform(missing)[0]

        self.assertEqual(transformed_negative["weight_clean"], 41000.0)
        self.assertEqual(transformed_negative["weight_was_negative"], 1)
        self.assertEqual(transformed_missing["weight_was_missing"], 1)
        self.assertEqual(transformed_missing["weight_clean"], 40000.0)
        self.assertEqual(transformed_missing["market_index_clean"], 150.0)

    def test_transform_does_not_mutate_rows_and_preserves_order_and_count(self):
        train = validate_raw_frame([row("L1"), row("L2", pickup="Dallas")], schema="training")
        original = copy.deepcopy(train)

        transformed = FreightPreprocessor(feature_view="full").fit(train).transform(train)

        self.assertEqual(train, original)
        self.assertEqual(len(transformed), 2)
        self.assertEqual([item["pickup"] for item in transformed], ["Atlanta", "Dallas"])

    def test_unknown_cities_routes_and_equipment_map_to_unknown_without_failure(self):
        train = validate_raw_frame([row("L1")], schema="training")
        inference = validate_raw_frame(
            [row("L2", pickup="New City", delivery="Other City", equipment="Reefer")],
            schema="training",
        )

        transformed = FreightPreprocessor(feature_view="full").fit(train).transform(inference)[0]

        self.assertEqual(transformed["pickup"], "__unknown__")
        self.assertEqual(transformed["delivery"], "__unknown__")
        self.assertEqual(transformed["route"], "__unknown__")
        self.assertEqual(transformed["equipment"], "__unknown__")

    def test_refit_replaces_learned_categories_to_prevent_cross_fold_leakage(self):
        first_fold = validate_raw_frame([row("L1", pickup="Atlanta")], schema="training")
        second_fold = validate_raw_frame([row("L2", pickup="Dallas")], schema="training")

        processor = FreightPreprocessor(feature_view="full").fit(first_fold)
        processor.fit(second_fold)
        transformed = processor.transform(first_fold)[0]

        self.assertEqual(transformed["pickup"], "__unknown__")

    def test_date_and_numeric_features_are_deterministic_and_finite(self):
        train = validate_raw_frame([row("L1", date="2025-01-01")], schema="training")

        transformed = FreightPreprocessor(feature_view="full").fit(train).transform(train)[0]

        self.assertEqual(transformed["days_since_2025_01_01"], 0)
        self.assertEqual(transformed["day_of_week"], 2)
        self.assertEqual(transformed["is_weekend"], 0)
        self.assertNotIn("load_id", transformed)
        self.assertNotIn("posted_rate", transformed)
        for value in transformed.values():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value))

    def test_december_schema_transforms_with_shared_saved_transformer(self):
        train = validate_raw_frame([row("L1")], schema="training")
        december = validate_raw_frame(
            [
                {
                    "pickup": "Atlanta",
                    "delivery": "Chicago",
                    "distance": "716",
                    "equipment": "Van",
                    "weight": "",
                    "date": "2025-12-01",
                    "predicted_rate": "",
                }
            ],
            schema="december",
        )
        processor = FreightPreprocessor(feature_view="december_compatible").fit(train)

        before = processor.transform(december)
        after = pickle.loads(pickle.dumps(processor)).transform(december)

        self.assertEqual(before, after)
        self.assertNotIn("market_index_clean", before[0])
        self.assertNotIn("quote_signal", before[0])


if __name__ == "__main__":
    unittest.main()
