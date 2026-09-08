from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    count: int
    mae: float
    rmse: float
    wmape: float


def _array(values: Iterable[object], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if len(array) == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def regression_metrics(y_true: Iterable[object], y_pred: Iterable[object]) -> RegressionMetrics:
    actual = _array(y_true, "actual")
    predicted = _array(y_pred, "predicted")
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal length")
    denominator = float(np.abs(actual).sum())
    if denominator == 0:
        raise ValueError("zero WMAPE denominator")
    error = np.abs(actual - predicted)
    return RegressionMetrics(len(actual), float(error.mean()), float(np.sqrt(np.mean((actual - predicted) ** 2))), float(error.sum() / denominator))


def validate_predictions(values: Iterable[object]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed or any((not math.isfinite(value)) or value <= 0 for value in parsed):
        raise ValueError("predictions must be finite and strictly positive")
    return parsed


def _metric_or_none(rows: list[int], actual: np.ndarray, predicted: np.ndarray, minimum_count: int) -> dict[str, object]:
    if len(rows) < minimum_count:
        return {"count": len(rows), "mae": None, "rmse": None, "wmape": None}
    return asdict(regression_metrics(actual[rows], predicted[rows]))


def slice_metrics(rows: list[Mapping[str, object]], y_true: Iterable[object], y_pred: Iterable[object], seen_routes: set[tuple[str, str]], *, minimum_count: int = 25) -> list[dict[str, object]]:
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    actual = _array(y_true, "actual")
    predicted = _array(y_pred, "predicted")
    if len(rows) != len(actual) or len(rows) != len(predicted):
        raise ValueError("length mismatch")
    buckets: dict[tuple[str, str], list[int]] = {}
    q90 = float(np.quantile(actual, 0.9))
    for index, row in enumerate(rows):
        route = (str(row.get("pickup")), str(row.get("delivery")))
        distance = float(row.get("distance", 0) or 0)
        weight = row.get("weight")
        market = row.get("market_index")
        pairs = [
            ("equipment", str(row.get("equipment"))),
            ("route", "seen" if route in seen_routes else "unseen"),
            ("distance", "<500" if distance < 500 else ("500-1499" if distance < 1500 else "1500+")),
            ("weight_missing", "missing" if weight in (None, "") else "present"),
            ("weight_negative", "negative" if str(weight).strip().startswith("-") else "nonnegative"),
            ("market_index", "missing" if market in (None, "") else "present"),
            ("high_value", "top10" if actual[index] >= q90 else "rest"),
        ]
        for key in pairs:
            buckets.setdefault(key, []).append(index)
    return [{"family": family, "value": value, **_metric_or_none(indices, actual, predicted, minimum_count)} for (family, value), indices in sorted(buckets.items())]
