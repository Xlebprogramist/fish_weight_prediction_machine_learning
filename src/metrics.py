"""Метрики регрессии и диагностика линейной модели.

MSE  = 1/n · Σ (y − ŷ)²
MAE  = 1/n · Σ |y − ŷ|
R²   = 1 − Σ (y − ŷ)² / Σ (y − ȳ)²
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def vif(X: pd.DataFrame) -> pd.Series:
    """Коэффициент инфляции дисперсии: VIF_j = 1 / (1 − R²_j), R²_j — регрессия X_j на остальные."""
    values = {}
    A = X.to_numpy(dtype=float)
    for j, col in enumerate(X.columns):
        y = A[:, j]
        others = np.delete(A, j, axis=1)
        design = np.column_stack([np.ones(len(y)), others])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
        values[col] = float(np.inf if r2 >= 1 else 1 / (1 - r2))
    return pd.Series(values, name="VIF")


def breusch_pagan(residuals, X) -> dict[str, float]:
    """Тест Бройша — Пагана на неоднородность дисперсии остатков.

    LM = n · R² регрессии квадратов остатков на признаки; LM ~ χ²(p).
    Малое p-value — дисперсия остатков зависит от признаков.
    """
    e2 = np.asarray(residuals, dtype=float) ** 2
    A = np.asarray(X, dtype=float)
    design = np.column_stack([np.ones(len(e2)), A])
    beta, *_ = np.linalg.lstsq(design, e2, rcond=None)
    fitted = design @ beta
    r2 = 1 - ((e2 - fitted) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
    dof = np.linalg.matrix_rank(design) - 1
    lm = len(e2) * r2
    return {"bp_lm": float(lm), "bp_df": int(dof), "bp_pvalue": float(stats.chi2.sf(lm, dof))}
