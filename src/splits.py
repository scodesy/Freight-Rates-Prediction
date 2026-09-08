from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    role: Literal["selection", "diagnostic", "holdout"]


def development_folds() -> tuple[FoldSpec, ...]:
    return (
        FoldSpec("july_1m", date(2025, 1, 1), date(2025, 6, 30), date(2025, 7, 1), date(2025, 7, 31), "selection"),
        FoldSpec("august_1m", date(2025, 1, 1), date(2025, 7, 31), date(2025, 8, 1), date(2025, 8, 31), "selection"),
        FoldSpec("september_1m", date(2025, 1, 1), date(2025, 8, 31), date(2025, 9, 1), date(2025, 9, 30), "selection"),
    )


def horizon_fold() -> FoldSpec:
    return FoldSpec("september_2m", date(2025, 1, 1), date(2025, 7, 31), date(2025, 9, 1), date(2025, 9, 30), "diagnostic")


def holdout_fold() -> FoldSpec:
    return FoldSpec("october_holdout", date(2025, 1, 1), date(2025, 9, 30), date(2025, 10, 1), date(2025, 10, 31), "holdout")


def parse_date(row: Mapping[str, object]) -> date:
    try:
        return date.fromisoformat(str(row["date"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unparseable date: {row.get('date')}") from exc


def split_rows(rows: list[Mapping[str, object]], fold: FoldSpec, *, allow_holdout: bool = False) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    if not rows:
        raise ValueError("empty rows")
    if fold.train_end >= fold.validation_start:
        raise ValueError("training must end before validation")
    if fold.role == "holdout" and not allow_holdout:
        raise PermissionError("October holdout requires explicit access")
    ids = [str(row.get("load_id")) for row in rows if "load_id" in row]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate load_id")
    train = [row for row in rows if fold.train_start <= parse_date(row) <= fold.train_end]
    validation = [row for row in rows if fold.validation_start <= parse_date(row) <= fold.validation_end]
    if not train:
        raise ValueError("empty train split")
    if not validation:
        raise ValueError("empty validation split")
    return train, validation


def development_rows(rows: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
    result = []
    for row in rows:
        value = parse_date(row)
        if value < date(2025, 1, 1) or value > date(2025, 10, 31):
            raise ValueError("date outside supported 2025 Jan-Oct domain")
        if value < date(2025, 10, 1):
            result.append(row)
    return result
