from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetAdapter:
    name: str

    def transform(self, y: np.ndarray, x: pd.DataFrame) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        if self.name == "direct":
            return values
        if self.name == "log":
            if (values <= 0).any():
                raise ValueError("log target requires positive values")
            return np.log1p(values)
        if self.name == "rpm":
            distance = _distance(x)
            return values / distance
        raise ValueError(f"unknown target adapter: {self.name}")

    def inverse_transform(self, prediction: np.ndarray, x: pd.DataFrame) -> np.ndarray:
        values = np.asarray(prediction, dtype=float)
        if self.name == "direct":
            result = values
        elif self.name == "log":
            result = np.expm1(values)
        elif self.name == "rpm":
            result = values * _distance(x)
        else:
            raise ValueError(f"unknown target adapter: {self.name}")
        if (result <= 0).any() or not np.isfinite(result).all():
            raise ValueError("inverse target produced non-positive or non-finite rates")
        return result


def target_adapter(name: str) -> TargetAdapter:
    if name not in {"direct", "log", "rpm"}:
        raise ValueError(f"unknown target adapter: {name}")
    return TargetAdapter(name)


def _distance(x: pd.DataFrame) -> np.ndarray:
    if "distance" not in x.columns:
        raise ValueError("rpm target requires distance feature")
    distance = x["distance"].to_numpy(dtype=float)
    if (distance <= 0).any() or not np.isfinite(distance).all():
        raise ValueError("rpm target requires positive finite distance")
    return distance
