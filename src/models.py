from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RateEstimator:
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> RateEstimator:
        raise NotImplementedError

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class GlobalMedianEstimator(RateEstimator):
    median_: float | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> GlobalMedianEstimator:
        del x
        self.median_ = float(np.median(y))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("estimator is not fitted")
        return np.full(len(x), self.median_, dtype=float)


@dataclass
class GlobalRpmEstimator(RateEstimator):
    rpm_: float | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> GlobalRpmEstimator:
        self.rpm_ = float(np.median(y / _distance(x)))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.rpm_ is None:
            raise RuntimeError("estimator is not fitted")
        return self.rpm_ * _distance(x)


@dataclass
class EquipmentRpmEstimator(RateEstimator):
    global_rpm_: float | None = None
    equipment_rpm_: dict[str, float] | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> EquipmentRpmEstimator:
        rpm = y / _distance(x)
        self.global_rpm_ = float(np.median(rpm))
        values: dict[str, list[float]] = {}
        for equipment, value in zip(x["equipment"].astype(str), rpm, strict=True):
            values.setdefault(equipment, []).append(float(value))
        self.equipment_rpm_ = {key: float(np.median(items)) for key, items in values.items()}
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.global_rpm_ is None or self.equipment_rpm_ is None:
            raise RuntimeError("estimator is not fitted")
        rpms = [self.equipment_rpm_.get(str(eq), self.global_rpm_) for eq in x["equipment"]]
        return np.asarray(rpms, dtype=float) * _distance(x)


@dataclass
class SklearnEstimator(RateEstimator):
    model_name: str
    parameters: dict[str, Any]
    pipeline_: Pipeline | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> SklearnEstimator:
        categorical = [column for column in x.columns if str(x[column].dtype) == "object"]
        numeric = [column for column in x.columns if column not in categorical]
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        model = _regressor(self.model_name, self.parameters)
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))]), numeric),
                ("cat", encoder, categorical),
            ],
            sparse_threshold=0.3,
        )
        self.pipeline_ = Pipeline([("pre", pre), ("model", model)])
        self.pipeline_.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("estimator is not fitted")
        return np.maximum(np.asarray(self.pipeline_.predict(x), dtype=float), 1e-6)


@dataclass
class CatBoostEstimator(RateEstimator):
    parameters: dict[str, Any]
    model_: Any | None = None
    feature_names_: tuple[str, ...] = ()
    cat_feature_names_: tuple[str, ...] = ()

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> CatBoostEstimator:
        from catboost import CatBoostRegressor

        self.feature_names_ = tuple(x.columns)
        self.cat_feature_names_ = tuple(column for column in x.columns if str(x[column].dtype) == "object")
        params = {"loss_function": "RMSE", "eval_metric": "MAE", "allow_writing_files": False, **self.parameters}
        self.model_ = CatBoostRegressor(**params)
        self.model_.fit(_catboost_frame(x, self.feature_names_, self.cat_feature_names_), y, cat_features=list(self.cat_feature_names_))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("estimator is not fitted")
        from catboost import Pool

        frame = _catboost_frame(x, self.feature_names_, self.cat_feature_names_)
        pool = Pool(frame, cat_features=list(self.cat_feature_names_))
        return np.maximum(np.asarray(self.model_.predict(pool), dtype=float), 1e-6)


def make_estimator(name: str, parameters: dict[str, Any] | None = None) -> RateEstimator:
    params = dict(parameters or {})
    if name == "median":
        return GlobalMedianEstimator()
    if name == "global_rpm":
        return GlobalRpmEstimator()
    if name == "equipment_rpm":
        return EquipmentRpmEstimator()
    if name == "catboost":
        return CatBoostEstimator(params)
    if name in {"ridge", "extra_trees", "xgboost"}:
        return SklearnEstimator(name, params)
    raise ValueError(f"unknown model: {name}")


def available_optional_models() -> dict[str, str]:
    result = {}
    for module, model in (("xgboost", "xgboost"), ("catboost", "catboost")):
        try:
            __import__(module)
            result[model] = "available"
        except Exception as exc:  # pragma: no cover - environment dependent
            result[model] = f"import_error:{type(exc).__name__}"
    return result


def predefined_configs(seed: int, workers: int) -> list[dict[str, Any]]:
    base = {"feature_view": "december_compatible", "seed": seed, "folds": ["july_1m", "august_1m", "september_1m", "september_2m"]}
    configs = [
        {**base, "model": "median", "target": "direct", "parameters": {}},
        {**base, "model": "global_rpm", "target": "direct", "parameters": {}},
        {**base, "model": "equipment_rpm", "target": "direct", "parameters": {}},
        {**base, "model": "ridge", "target": "direct", "parameters": {"alpha": 10.0}},
        {**base, "model": "extra_trees", "target": "direct", "parameters": {"n_estimators": 140, "min_samples_leaf": 8, "n_jobs": workers, "random_state": seed}},
        {**base, "model": "extra_trees", "target": "log", "parameters": {"n_estimators": 160, "min_samples_leaf": 6, "n_jobs": workers, "random_state": seed}},
        {**base, "model": "extra_trees", "target": "rpm", "parameters": {"n_estimators": 160, "min_samples_leaf": 6, "n_jobs": workers, "random_state": seed}},
    ]
    optional = available_optional_models()
    if optional.get("xgboost") == "available":
        configs.append({**base, "model": "xgboost", "target": "direct", "parameters": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "n_jobs": workers, "random_state": seed}})
    if optional.get("catboost") == "available":
        configs.append(
            {
                **base,
                "model": "catboost",
                "target": "direct",
                "parameters": {
                    "iterations": 400,
                    "depth": 6,
                    "learning_rate": 0.05,
                    "random_seed": seed,
                    "thread_count": workers,
                    "loss_function": "RMSE",
                    "eval_metric": "MAE",
                    "allow_writing_files": False,
                    "verbose": False,
                },
            }
        )
    return configs


def _regressor(name: str, parameters: dict[str, Any]) -> Any:
    if name == "ridge":
        return Ridge(**parameters)
    if name == "extra_trees":
        return ExtraTreesRegressor(**parameters)
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**parameters)
    raise ValueError(f"unsupported sklearn model: {name}")


def _distance(x: pd.DataFrame) -> np.ndarray:
    distance = x["distance"].to_numpy(dtype=float)
    if (distance <= 0).any() or not np.isfinite(distance).all():
        raise ValueError("distance must be positive finite")
    return distance


def _catboost_frame(x: pd.DataFrame, feature_names: tuple[str, ...], cat_feature_names: tuple[str, ...]) -> pd.DataFrame:
    missing = [column for column in feature_names if column not in x.columns]
    if missing:
        raise ValueError(f"missing CatBoost feature columns: {missing}")
    frame = x.loc[:, list(feature_names)].copy()
    for column in cat_feature_names:
        frame[column] = frame[column].astype(str)
    return frame
