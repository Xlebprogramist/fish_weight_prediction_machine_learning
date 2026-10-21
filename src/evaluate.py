"""Оценивание сохранённой модели и анализ ошибок.

    python -m src.evaluate                       # итоговая модель из reports/lab02-results.csv
    python -m src.evaluate --run e1_ols          # любой запуск из runs/

Что делает:
1. загружает сохранённую цепочку (`runs/<run>/model.joblib`), заново считает
   метрики на замороженном тесте и сверяет их с записанными при обучении;
2. строит таблицы анализа ошибок по прогнозам вне выборки (CV, 88 строк) и
   по тесту (23 строки): систематические зоны остатков, крупнейшие ошибки,
   MAE по видам и диапазонам массы, отрицательные прогнозы;
3. диагностика линейной модели: VIF признаков, тест Бройша — Пагана,
   устойчивость коэффициентов по разбиениям CV;
4. пишет таблицы в reports/tables/ и графики в reports/figures/.
"""

from __future__ import annotations

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import CATEGORICAL, NUMERIC, PROJECT_ROOT, TARGET, train_test_frames  # noqa: E402
from src.metrics import breusch_pagan, regression_metrics, vif  # noqa: E402
from src.pipeline import add_volume_proxy  # noqa: E402

RUNS = PROJECT_ROOT / "runs"
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"
RESULTS = PROJECT_ROOT / "reports" / "lab02-results.csv"
WEIGHT_BINS = [0, 100, 300, 700, np.inf]
WEIGHT_LABELS = ["<100 г", "100–300 г", "300–700 г", "≥700 г"]
SHOW = ["Species", TARGET, *NUMERIC, "prediction", "residual"]


def by_group(pred: pd.DataFrame, key: str | pd.Series) -> pd.DataFrame:
    g = pred.assign(abs_error=pred["residual"].abs()).groupby(key, observed=True)
    out = g.agg(
        n=("residual", "size"),
        mae=("abs_error", "mean"),
        mean_residual=("residual", "mean"),
        mean_weight=(TARGET, "mean"),
    )
    out["relative_mae_%"] = 100 * out["mae"] / out["mean_weight"]
    return out.round(2)


def residual_zones(pred: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Средний остаток по квантильным корзинам прогноза: систематический сдвиг ≠ 0."""
    bins = pd.qcut(pred["prediction"], n_bins, duplicates="drop")
    out = pred.groupby(bins, observed=True)["residual"].agg(["size", "mean", "std"])
    out.index = out.index.astype(str)
    out.index.name = "prediction_bin"
    return out.round(2)


def residual_plot_oof(oof: pd.DataFrame, run: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for sp, part in oof.groupby("Species"):
        axes[0].scatter(part["prediction"], part["residual"], s=22, label=sp)
        axes[1].scatter(part[TARGET], part["residual"] / part[TARGET] * 100, s=22, label=sp)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Прогноз, г")
    axes[0].set_ylabel("Остаток, г")
    axes[0].set_title("Остатки вне выборки (CV, 88 строк)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Истинная масса, г (лог. шкала)")
    axes[1].set_ylabel("Относительный остаток, %")
    axes[1].set_title("Относительная ошибка по размеру")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle(run)
    fig.tight_layout()
    fig.savefig(FIGURES / f"lab02-{run}-residuals-oof.png", dpi=120)
    plt.close(fig)


def linear_diagnostics(train: pd.DataFrame, model, run: str) -> dict:
    """VIF исходных и производных признаков, тест Бройша — Пагана на остатках модели."""
    feats = add_volume_proxy(train[NUMERIC])
    vif_table = pd.DataFrame(
        {
            "VIF_raw": vif(feats[NUMERIC]),
            "VIF_raw_plus_Vproxy": vif(feats),
            "VIF_log1p": vif(np.log1p(feats[NUMERIC])),
            "VIF_log1p_plus_Vproxy": vif(np.log1p(feats)),
        }
    ).round(1)
    vif_table.to_csv(TABLES / "lab02-vif.csv")
    corr = feats.assign(Weight=train[TARGET]).corr().round(4)
    corr.to_csv(TABLES / "lab02-correlations.csv")

    X_design = model[:-1].transform(train[CATEGORICAL + NUMERIC])
    resid = train[TARGET] - model.predict(train[CATEGORICAL + NUMERIC])
    bp = breusch_pagan(resid, X_design)
    print(f"\nVIF:\n{vif_table}\n\nБройш — Паган ({run}, остатки на train): {bp}")
    return bp


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", help="имя запуска (по умолчанию — выбранный по CV)")
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(RESULTS)
    run = args.run or results.loc[results["selected"], "run_name"].item()
    run_dir = RUNS / run
    cfg = load_config(run_dir / "config.yaml")
    train, test = train_test_frames(cfg)

    # 1. воспроизведение метрик сохранённой цепочкой
    model = joblib.load(run_dir / "model.joblib")
    recomputed = regression_metrics(test[TARGET], model.predict(test[CATEGORICAL + NUMERIC]))
    stored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    for m in ("mse", "mae", "r2"):
        assert np.isclose(recomputed[m], stored[f"test_{m}"], rtol=1e-9), m
    print(f"{run}: метрики теста воспроизведены сохранённой моделью: {recomputed}")

    # 2. анализ ошибок
    oof = pd.read_csv(run_dir / "predictions_oof.csv", index_col="row")
    tst = pd.read_csv(run_dir / "predictions_test.csv", index_col="row")
    tables = {
        "zones_oof": residual_zones(oof),
        "top_errors_oof": oof.assign(abs_error=oof["residual"].abs())
        .nlargest(5, "abs_error")[SHOW + ["abs_error"]]
        .round(2),
        "top_errors_test": tst.nlargest(5, "abs_error")[SHOW + ["abs_error"]].round(2),
        "mae_by_species_oof": by_group(oof, "Species"),
        "mae_by_species_test": by_group(tst, "Species"),
        "mae_by_weight_oof": by_group(
            oof, pd.cut(oof[TARGET], WEIGHT_BINS, labels=WEIGHT_LABELS, right=False)
        ),
        "mae_by_weight_test": by_group(
            tst, pd.cut(tst[TARGET], WEIGHT_BINS, labels=WEIGHT_LABELS, right=False)
        ),
    }
    for key, table in tables.items():
        table.to_csv(TABLES / f"lab02-{run}-{key}.csv")
        print(f"\n== {key}\n{table}")
    residual_plot_oof(oof, run)

    # 3. отрицательные и физически невозможные прогнозы во всех запусках
    rows = []
    for name in results["run_name"]:
        for part, fname in (("oof", "predictions_oof.csv"), ("test", "predictions_test.csv")):
            p = pd.read_csv(RUNS / name / fname)
            rows.append(
                {
                    "run": name,
                    "part": part,
                    "n": len(p),
                    "n_negative": int((p["prediction"] < 0).sum()),
                    "min_prediction": round(float(p["prediction"].min()), 2),
                    "n_nonfinite": int((~np.isfinite(p["prediction"])).sum()),
                }
            )
    negatives = pd.DataFrame(rows)
    negatives.to_csv(TABLES / "lab02-negative-predictions.csv", index=False)
    print(f"\n== отрицательные прогнозы\n{negatives[negatives['n_negative'] > 0]}")

    # 4. диагностика линейной модели
    if (run_dir / "coefficients.csv").exists():
        bp = linear_diagnostics(train, model, run)
        (TABLES / f"lab02-{run}-breusch-pagan.json").write_text(json.dumps(bp, indent=2))


if __name__ == "__main__":
    main()
