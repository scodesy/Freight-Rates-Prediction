from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from datetime import date
from pathlib import Path


class SchemaValidationError(ValueError):
    """Raised when raw freight input data violates the preprocessing contract."""


SCHEMAS: dict[str, tuple[str, ...]] = {
    "training": (
        "load_id",
        "pickup",
        "delivery",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
        "quote_signal",
        "posted_rate",
    ),
    "full_validation": (
        "load_id",
        "pickup",
        "delivery",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
        "quote_signal",
    ),
    "december": (
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ),
}

COORDINATE_RANGES = {
    "pickup_lat": (-90.0, 90.0),
    "delivery_lat": (-90.0, 90.0),
    "pickup_lon": (-180.0, 180.0),
    "delivery_lon": (-180.0, 180.0),
}


def validate_raw_frame(
    rows: list[Mapping[str, object]], *, schema: str
) -> list[dict[str, object]]:
    if schema not in SCHEMAS:
        raise SchemaValidationError(f"Unknown schema: {schema}")

    expected_columns = set(SCHEMAS[schema])
    validated = [_normalize_row(row) for row in rows]
    _validate_columns(validated, expected_columns)
    _validate_ids(validated, schema)

    for index, row in enumerate(validated, start=1):
        _validate_categoricals(row, index)
        _validate_date(row, index)
        _validate_required_numeric(row, schema, index)
        _validate_optional_numeric(row, index)

    return validated


def load_csv_rows(path: str | Path, *, schema: str) -> list[dict[str, object]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SchemaValidationError(f"CSV file has no header: {csv_path}") from exc
        _validate_csv_header(header)
        rows = [_row_from_csv(header, values, index) for index, values in enumerate(reader, start=2)]

    if not rows:
        raise SchemaValidationError(f"CSV file contains no data rows: {csv_path}")
    return validate_raw_frame(rows, schema=schema)


def _validate_csv_header(header: list[str]) -> None:
    seen: dict[str, str] = {}
    for column in header:
        normalized = column.strip()
        if normalized in seen:
            raise SchemaValidationError(
                f"Duplicate CSV header after normalization: {seen[normalized]!r}, {column!r}"
            )
        seen[normalized] = column


def _row_from_csv(header: list[str], values: list[str], index: int) -> dict[str, object]:
    if len(values) != len(header):
        raise SchemaValidationError(f"Malformed CSV row {index}")
    return dict(zip(header, values, strict=True))


def _normalize_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    raw_keys_by_normalized_key: dict[str, str] = {}
    for key, value in row.items():
        normalized_key = str(key).strip()
        if normalized_key in raw_keys_by_normalized_key:
            raise SchemaValidationError(
                "Raw headers normalize to the same column: "
                f"{raw_keys_by_normalized_key[normalized_key]!r}, {key!r}"
            )
        raw_keys_by_normalized_key[normalized_key] = str(key)
        normalized[normalized_key] = value.strip() if isinstance(value, str) else value
    return normalized


def _validate_columns(rows: list[dict[str, object]], expected_columns: set[str]) -> None:
    for index, row in enumerate(rows, start=1):
        actual_columns = set(row)
        missing = sorted(expected_columns - actual_columns)
        unexpected = sorted(actual_columns - expected_columns)
        if missing:
            raise SchemaValidationError(f"Row {index} missing columns: {missing}")
        if unexpected:
            raise SchemaValidationError(f"Row {index} unexpected columns: {unexpected}")


def _validate_ids(rows: list[dict[str, object]], schema: str) -> None:
    if schema == "december":
        return

    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        raw_load_id = row.get("load_id")
        if not isinstance(raw_load_id, str) or not raw_load_id.strip():
            raise SchemaValidationError(f"Row {index} Invalid load_id")
        load_id = raw_load_id.strip()
        if load_id in seen:
            raise SchemaValidationError(f"Duplicate load_id: {load_id}")
        seen.add(load_id)


def _validate_categoricals(row: dict[str, object], index: int) -> None:
    for column in ("pickup", "delivery", "equipment"):
        if _is_missing(row.get(column)):
            raise SchemaValidationError(f"Row {index} Missing categorical: {column}")


def _validate_date(row: dict[str, object], index: int) -> None:
    try:
        date.fromisoformat(str(row["date"]))
    except (KeyError, ValueError) as exc:
        raise SchemaValidationError(f"Row {index} Unparseable date: {row.get('date')}") from exc


def _validate_required_numeric(row: dict[str, object], schema: str, index: int) -> None:
    required = ["distance"]
    if schema != "december":
        required.extend(["quote_signal", *COORDINATE_RANGES.keys()])
    if schema == "training":
        required.append("posted_rate")

    for column in required:
        value = _parse_finite_float(row.get(column), column, index)
        if column == "distance" and value <= 0:
            raise SchemaValidationError(f"Row {index} distance must be positive")
        if column == "posted_rate" and value <= 0:
            raise SchemaValidationError(f"Row {index} posted_rate must be positive")
        if column in COORDINATE_RANGES:
            lower, upper = COORDINATE_RANGES[column]
            if value < lower or value > upper:
                raise SchemaValidationError(f"Row {index} Invalid coordinate: {column}")


def _validate_optional_numeric(row: dict[str, object], index: int) -> None:
    for column in ("weight", "market_index"):
        if column not in row or _is_missing(row[column]):
            continue
        _parse_finite_float(row[column], column, index)

    if "predicted_rate" in row and not _is_missing(row["predicted_rate"]):
        _parse_finite_float(row["predicted_rate"], "predicted_rate", index)


def _parse_finite_float(value: object, column: str, index: int) -> float:
    if _is_missing(value):
        raise SchemaValidationError(f"Row {index} Missing numeric: {column}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"Row {index} Non-finite numeric: {column}") from exc
    if not math.isfinite(parsed):
        raise SchemaValidationError(f"Row {index} Non-finite numeric: {column}")
    return parsed


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")
