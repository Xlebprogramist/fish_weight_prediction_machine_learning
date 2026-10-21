"""Быстрые проверки дополнительной части: bootstrap и интервалы прогноза."""

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import (
    interval_quality,
    metric_cis,
    metrics_on_resamples,
    prediction_intervals,
    resample_indices,
)
from src.metrics import regression_metrics


def toy(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pred = rng.uniform(5, 1500, n)
    weight = pred * np.exp(rng.normal(0, 0.1, n))  # мультипликативный шум, как у рыб
    return pd.DataFrame({"Weight": weight, "prediction": pred, "Species": "Perch"})


def test_resampled_metrics_match_regression_metrics_on_identity():
    df = toy()
    y, p = df["Weight"].to_numpy(), df["prediction"].to_numpy()
    boot = metrics_on_resamples(y, p, np.arange(len(df))[None, :])
    ref = regression_metrics(y, p)
    for m in ("mse", "mae", "r2"):
        assert boot[m][0] == pytest.approx(ref[m])


def test_bootstrap_is_deterministic_and_ci_contains_estimate():
    df = toy()
    a = metric_cis(df, resample_indices(len(df), 2000, 42), 0.95)
    b = metric_cis(df, resample_indices(len(df), 2000, 42), 0.95)
    assert a == b
    for m in ("mse", "mae", "r2"):
        assert a[f"{m}_ci_low"] <= a[m] <= a[f"{m}_ci_high"]


def test_multiplicative_intervals_are_positive_and_calibrated():
    calib, test = toy(2000, seed=1), toy(2000, seed=2)
    iv, params = prediction_intervals(calib, test, 0.8)
    assert params["mul_factor_low"] < 1 < params["mul_factor_high"]
    quality = interval_quality(iv, "mul")
    assert quality["n_negative_lower"] == 0
    assert quality["coverage"] == pytest.approx(0.8, abs=0.03)
