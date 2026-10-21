"""Быстрые проверки цепочки обработки, метрик и сохранения модели."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import CATEGORICAL, NUMERIC, TARGET, train_test_frames
from src.metrics import regression_metrics, vif
from src.pipeline import VOLUME_PROXY, add_volume_proxy, build_pipeline

INPUT = CATEGORICAL + NUMERIC
CONFIGS = [
    "configs/experiments/e1_esharma3.yaml",
    "configs/experiments/e1_ols.yaml",
    "configs/experiments/e2_ridge_loglog.yaml",
    "configs/experiments/e4_ols_vproxy.yaml",
]


@pytest.fixture(scope="module")
def frames():
    return train_test_frames(load_config("configs/base.yaml"))


def fitted(path: str, train: pd.DataFrame):
    cfg = load_config(path)
    return build_pipeline(cfg).fit(train[INPUT], train[TARGET])


@pytest.mark.parametrize("path", CONFIGS)
def test_pipeline_accepts_unknown_species(path, frames):
    train, _ = frames
    model = fitted(path, train)
    row = train[INPUT].head(1).assign(Species="Tuna")
    pred = model.predict(row)
    assert pred.shape == (1,) and np.isfinite(pred).all()


@pytest.mark.parametrize("path", CONFIGS)
def test_prediction_count_matches_input_and_is_finite(path, frames):
    train, test = frames
    pred = fitted(path, train).predict(test[INPUT])
    assert len(pred) == len(test)
    assert np.isfinite(pred).all()


def test_metrics_match_manual_calculation():
    y, y_hat = [1.0, 2.0, 3.0], [1.0, 3.0, 5.0]
    # MSE = (0 + 1 + 4) / 3; MAE = (0 + 1 + 2) / 3; R² = 1 − 5 / 2 (Σ(y − ȳ)² = 2)
    m = regression_metrics(y, y_hat)
    assert m["mse"] == pytest.approx(5 / 3)
    assert m["mae"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(-1.5)


def test_saved_pipeline_reproduces_predictions(frames, tmp_path):
    train, test = frames
    model = fitted("configs/experiments/e2_ridge_loglog.yaml", train)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(
        restored.predict(test[INPUT]), model.predict(test[INPUT]), rtol=1e-12
    )


def test_random_forest_is_deterministic_with_seed(frames):
    train, test = frames
    path = "configs/experiments/e3_random_forest.yaml"
    a = fitted(path, train).predict(test[INPUT])
    b = fitted(path, train).predict(test[INPUT])
    np.testing.assert_array_equal(a, b)


def test_volume_proxy_uses_only_geometry():
    X = pd.DataFrame({"Length3": [10.0], "Height": [2.0], "Width": [3.0], TARGET: [999.0]})
    out = add_volume_proxy(X.drop(columns=TARGET))
    assert out[VOLUME_PROXY].item() == pytest.approx(60.0)
    assert TARGET not in out.columns


def test_vif_detects_exact_collinearity():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    v = vif(pd.DataFrame({"a": a, "b": b, "c": a + 1e-6 * rng.normal(size=50)}))
    assert v["a"] > 1e6 and v["b"] < 2
