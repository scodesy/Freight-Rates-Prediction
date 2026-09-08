from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from datetime import date
from statistics import median

import pandas as pd

REFERENCE_DATE = date(2025, 1, 1)
UNKNOWN_CATEGORY = "__unknown__"


class FreightPreprocessor:
    def __init__(self, *, feature_view: str = "full") -> None:
        if feature_view not in {"full", "december_compatible"}:
            raise ValueError("feature_view must be 'full' or 'december_compatible'")
        self.feature_view = feature_view
        self.weight_medians_by_equipment: dict[str, float] = {}
        self.overall_weight_median: float = 0.0
        self.market_index_median: float = 0.0
        self.categories: dict[str, set[str]] = {
            "pickup": set(),
            "delivery": set(),
            "equipment": set(),
            "route": set(),
        }
        self.is_fitted = False
        self.schema_: tuple[tuple[str, str], ...] = ()

    def fit(self, rows: list[Mapping[str, object]]) -> FreightPreprocessor:
        weight_values_by_equipment: dict[str, list[float]] = defaultdict(list)
        all_weight_values: list[float] = []
        market_values: list[float] = []
        categories: dict[str, set[str]] = {
            "pickup": set(),
            "delivery": set(),
            "equipment": set(),
            "route": set(),
        }

        for row in rows:
            equipment = _string_value(row["equipment"])
            pickup = _string_value(row["pickup"])
            delivery = _string_value(row["delivery"])
            route = _route(pickup, delivery)

            categories["equipment"].add(equipment)
            categories["pickup"].add(pickup)
            categories["delivery"].add(delivery)
            categories["route"].add(route)

            weight = _optional_float(row.get("weight"))
            if weight is not None:
                clean_weight = abs(weight)
                weight_values_by_equipment[equipment].append(clean_weight)
                all_weight_values.append(clean_weight)

            market_index = _optional_float(row.get("market_index"))
            if market_index is not None:
                market_values.append(market_index)

        self.overall_weight_median = median(all_weight_values) if all_weight_values else 0.0
        self.weight_medians_by_equipment = {
            equipment: median(values)
            for equipment, values in weight_values_by_equipment.items()
        }
        self.market_index_median = median(market_values) if market_values else 0.0
        self.categories = categories
        self.is_fitted = True
        frame = pd.DataFrame(self.transform(rows))
        self.schema_ = tuple((column, str(dtype)) for column, dtype in frame.dtypes.items())
        return self

    def transform(self, rows: list[Mapping[str, object]]) -> list[dict[str, object]]:
        if not self.is_fitted:
            raise RuntimeError("FreightPreprocessor must be fitted before transform")
        return [self._transform_row(row) for row in rows]

    def transform_frame(self, rows: list[Mapping[str, object]]) -> pd.DataFrame:
        frame = pd.DataFrame(self.transform(rows))
        expected_columns = [column for column, _dtype in self.schema_]
        frame = frame.loc[:, expected_columns]
        actual = tuple((column, str(dtype)) for column, dtype in frame.dtypes.items())
        if actual != self.schema_:
            raise ValueError(f"Feature schema mismatch: expected {self.schema_}, got {actual}")
        return frame

    @property
    def schema_fingerprint(self) -> str:
        import hashlib
        import json

        return hashlib.sha256(json.dumps(self.schema_).encode("utf-8")).hexdigest()

    def _transform_row(self, row: Mapping[str, object]) -> dict[str, object]:
        pickup = _string_value(row["pickup"])
        delivery = _string_value(row["delivery"])
        equipment = _string_value(row["equipment"])
        route = _route(pickup, delivery)
        distance = _required_float(row["distance"])
        parsed_date = date.fromisoformat(_string_value(row["date"]))
        weight_clean, weight_was_missing, weight_was_negative = self._clean_weight(row)

        features: dict[str, object] = {
            "pickup": self._known_or_unknown("pickup", pickup),
            "delivery": self._known_or_unknown("delivery", delivery),
            "equipment": self._known_or_unknown("equipment", equipment),
            "distance": distance,
            "distance_log1p": math.log1p(distance),
            "weight_clean": weight_clean,
            "weight_was_missing": int(weight_was_missing),
            "weight_was_negative": int(weight_was_negative),
            **_date_features(parsed_date),
        }

        if self.feature_view == "full":
            features.update(self._full_view_features(row, pickup, delivery, route, distance))
        return features

    def _full_view_features(
        self,
        row: Mapping[str, object],
        pickup: str,
        delivery: str,
        route: str,
        distance: float,
    ) -> dict[str, object]:
        pickup_lat = _required_float(row["pickup_lat"])
        pickup_lon = _required_float(row["pickup_lon"])
        delivery_lat = _required_float(row["delivery_lat"])
        delivery_lon = _required_float(row["delivery_lon"])
        straight_line_distance = _haversine_miles(
            pickup_lat, pickup_lon, delivery_lat, delivery_lon
        )
        market_index = _optional_float(row.get("market_index"))
        market_index_was_missing = market_index is None

        return {
            "route": self._known_or_unknown("route", route),
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "delivery_lat": delivery_lat,
            "delivery_lon": delivery_lon,
            "lat_delta": delivery_lat - pickup_lat,
            "lon_delta": delivery_lon - pickup_lon,
            "straight_line_distance": straight_line_distance,
            "distance_minus_straight_line": distance - straight_line_distance,
            "distance_to_straight_line_ratio": distance / straight_line_distance
            if straight_line_distance > 0
            else 0.0,
            "market_index_was_missing": int(market_index_was_missing),
            "market_index_clean": self.market_index_median
            if market_index_was_missing
            else market_index,
            "quote_signal": _required_float(row["quote_signal"]),
        }

    def _clean_weight(self, row: Mapping[str, object]) -> tuple[float, bool, bool]:
        weight = _optional_float(row.get("weight"))
        weight_was_missing = weight is None
        weight_was_negative = weight is not None and weight < 0
        if weight is not None:
            return abs(weight), weight_was_missing, weight_was_negative

        equipment = _string_value(row["equipment"])
        return (
            self.weight_medians_by_equipment.get(equipment, self.overall_weight_median),
            weight_was_missing,
            weight_was_negative,
        )

    def _known_or_unknown(self, column: str, value: str) -> str:
        return value if value in self.categories[column] else UNKNOWN_CATEGORY


def _date_features(value: date) -> dict[str, float | int]:
    day_of_year = value.timetuple().tm_yday
    day_of_week = value.weekday()
    return {
        "days_since_2025_01_01": (value - REFERENCE_DATE).days,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        "month": value.month,
        "day_of_month": value.day,
        "day_of_year": day_of_year,
        "annual_sin": math.sin(2 * math.pi * day_of_year / 365.0),
        "annual_cos": math.cos(2 * math.pi * day_of_year / 365.0),
        "weekly_sin": math.sin(2 * math.pi * day_of_week / 7.0),
        "weekly_cos": math.cos(2 * math.pi * day_of_week / 7.0),
    }


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        delta_lambda / 2
    ) ** 2
    return radius_miles * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _route(pickup: str, delivery: str) -> str:
    return f"{pickup} → {delivery}"


def _required_float(value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Expected finite numeric value, got {value!r}")
    return parsed


def _optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    return _required_float(value)


def _string_value(value: object) -> str:
    return str(value).strip()
