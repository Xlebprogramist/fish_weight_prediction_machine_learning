"""Единая цепочка «предобработка → модель», собираемая из конфигурации.

Вся предобработка (производный признак, заполнение пропусков, логарифм,
масштабирование, one-hot) находится внутри одного объекта Pipeline, поэтому при
перекрёстной проверке она настраивается заново на каждой обучающей части.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

VOLUME_PROXY = "V_proxy"

REGRESSORS = {
    "dummy": DummyRegressor,
    "linear": LinearRegression,
    "ridge": Ridge,
    "elasticnet": ElasticNet,
    "random_forest": RandomForestRegressor,
    "hist_gb": HistGradientBoostingRegressor,
}
RANDOMIZED = {"elasticnet", "random_forest", "hist_gb"}

TARGET_TRANSFORMS = {
    "log": (np.log, np.exp),
    "log1p": (np.log1p, np.expm1),
}
FEATURE_TRANSFORMS = {"log": np.log, "log1p": np.log1p}


def add_volume_proxy(X: pd.DataFrame) -> pd.DataFrame:
    """V_proxy = Length3 · Height · Width (см³) — грубая оценка объёма тела.

    Масса не используется, поэтому утечки целевой переменной нет.
    """
    X = X.copy()
    X[VOLUME_PROXY] = X["Length3"] * X["Height"] * X["Width"]
    return X


def _volume_names_out(_transformer, names_in):
    return np.append(np.asarray(names_in, dtype=object), VOLUME_PROXY)


def numeric_columns(features: dict) -> list[str]:
    cols = list(features["numeric"])
    if features.get("volume_proxy") and VOLUME_PROXY not in cols:
        cols.append(VOLUME_PROXY)
    return cols


def build_preprocess(features: dict) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    transform = features.get("numeric_transform", "none")
    if transform != "none":
        num_steps.append(
            (
                transform,
                FunctionTransformer(FEATURE_TRANSFORMS[transform], feature_names_out="one-to-one"),
            )
        )
    if features.get("scale", True):
        num_steps.append(("scale", StandardScaler()))

    blocks = [("num", Pipeline(num_steps), numeric_columns(features))]
    if features.get("categorical"):
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        blocks.append(("cat", categorical, list(features["categorical"])))
    return ColumnTransformer(blocks, remainder="drop")


def build_regressor(model: dict, target: dict, seed: int):
    name = model["name"]
    params = dict(model.get("params", {}))
    if name in RANDOMIZED:
        params.setdefault("random_state", seed)
    regressor = REGRESSORS[name](**params)
    transform = target.get("transform", "none")
    if transform != "none":
        func, inverse = TARGET_TRANSFORMS[transform]
        regressor = TransformedTargetRegressor(
            regressor=regressor, func=func, inverse_func=inverse, check_inverse=False
        )
    return regressor


def build_pipeline(cfg: dict) -> Pipeline:
    steps = []
    if cfg["features"].get("volume_proxy"):
        steps.append(
            (
                "volume",
                FunctionTransformer(add_volume_proxy, feature_names_out=_volume_names_out),
            )
        )
    steps.append(("preprocess", build_preprocess(cfg["features"])))
    steps.append(("regressor", build_regressor(cfg["model"], cfg["target"], cfg["seed"])))
    return Pipeline(steps)


def grid_prefix(cfg: dict) -> str:
    """Префикс гиперпараметров регрессора в Pipeline (учитывает обёртку целевой переменной)."""
    return (
        "regressor__regressor__"
        if cfg["target"].get("transform", "none") != "none"
        else "regressor__"
    )


def final_estimator(pipe: Pipeline):
    reg = pipe.named_steps["regressor"]
    return getattr(reg, "regressor_", reg)


def feature_names(pipe: Pipeline) -> list[str]:
    return [str(n) for n in pipe.named_steps["preprocess"].get_feature_names_out()]


def coefficients(pipe: Pipeline) -> pd.Series | None:
    """Коэффициенты линейной модели в пространстве преобразованных признаков."""
    est = final_estimator(pipe)
    if not hasattr(est, "coef_"):
        return None
    return pd.Series(np.ravel(est.coef_), index=feature_names(pipe))
