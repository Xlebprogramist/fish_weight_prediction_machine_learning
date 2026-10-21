"""Дополнительная часть ЛР2: доверительные интервалы и интервалы прогноза.

    python -m src.bootstrap        # после python -m src.train --all

1. Доверительные интервалы метрик методом повторной выборки с возвращением
   (bootstrap, перцентильный метод): для каждого запуска по тесту (23 строки)
   и по прогнозам вне выборки на CV (88 строк).
2. Парный bootstrap: разность MSE каждой модели и итоговой на одних и тех же
   повторных выборках — значимо ли итоговая модель лучше.
3. Интервалы прогноза итоговой модели, откалиброванные на остатках вне выборки
   (CV) и проверенные на тесте. Сравниваются два вида:
   - мультипликативный: ŷ · exp(q), q — квантили log(y / ŷ);
   - аддитивный:        ŷ + q,      q — квантили y − ŷ (в граммах).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data import PROJECT_ROOT, TARGET  # noqa: E402

RUNS = PROJECT_ROOT / "runs"
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"
RESULTS = PROJECT_ROOT / "reports" / "lab02-results.csv"
SIZE_BINS = [0, 300, np.inf]
SIZE_LABELS = ["<300 г", "≥300 г"]


def resample_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """Матрица индексов (n_resamples × n): выборки с возвращением."""
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def metrics_on_resamples(y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> dict:
    """MSE, MAE и R² на каждой повторной выборке (векторно)."""
    yt, yp = y[idx], pred[idx]
    err = yt - yp
    sse = (err**2).sum(axis=1)
    sst = ((yt - yt.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(sst > 0, 1 - sse / sst, np.nan)
    return {"mse": sse / y.size, "mae": np.abs(err).mean(axis=1), "r2": r2}


def percentile_ci(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1 - confidence) / 2
    lo, hi = np.nanquantile(values, [alpha, 1 - alpha])
    return float(lo), float(hi)


def metric_cis(pred: pd.DataFrame, idx: np.ndarray, confidence: float) -> dict:
    y, p = pred[TARGET].to_numpy(float), pred["prediction"].to_numpy(float)
    boot = metrics_on_resamples(y, p, idx)
    point = metrics_on_resamples(y, p, np.arange(y.size)[None, :])
    row = {}
    for m in ("mse", "mae", "r2"):
        lo, hi = percentile_ci(boot[m], confidence)
        row.update({m: float(point[m][0]), f"{m}_ci_low": lo, f"{m}_ci_high": hi})
    return row


def prediction_intervals(
    calib: pd.DataFrame, target: pd.DataFrame, level: float
) -> tuple[pd.DataFrame, dict]:
    """Интервалы прогноза по квантилям остатков калибровочной выборки."""
    alpha = (1 - level) / 2
    log_ratio = np.log(calib[TARGET] / calib["prediction"])
    q_mul = np.quantile(log_ratio, [alpha, 1 - alpha])
    q_add = np.quantile(calib[TARGET] - calib["prediction"], [alpha, 1 - alpha])
    out = target[[TARGET, "Species", "prediction"]].copy()
    out["mul_low"] = out["prediction"] * np.exp(q_mul[0])
    out["mul_high"] = out["prediction"] * np.exp(q_mul[1])
    out["add_low"] = out["prediction"] + q_add[0]
    out["add_high"] = out["prediction"] + q_add[1]
    params = {
        "level": level,
        "mul_factor_low": float(np.exp(q_mul[0])),
        "mul_factor_high": float(np.exp(q_mul[1])),
        "add_low_g": float(q_add[0]),
        "add_high_g": float(q_add[1]),
    }
    return out, params


def interval_quality(iv: pd.DataFrame, kind: str) -> dict:
    lo, hi, y = iv[f"{kind}_low"], iv[f"{kind}_high"], iv[TARGET]
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "coverage": float(inside.mean()),
        "mean_width_g": float(width.mean()),
        "median_relative_width_%": float((100 * width / iv["prediction"]).median()),
        "n_negative_lower": int((lo < 0).sum()),
    }


def interval_report(intervals: pd.DataFrame, params: dict, part: str) -> list[dict]:
    rows = []
    sizes = pd.cut(intervals[TARGET], SIZE_BINS, labels=SIZE_LABELS, right=False)
    for kind, name in (("mul", "мультипликативный"), ("add", "аддитивный")):
        rows.append(
            {
                "part": part,
                "level": params["level"],
                "interval": name,
                "group": "все",
                "n": len(intervals),
                **interval_quality(intervals, kind),
            }
        )
        for label, chunk in intervals.groupby(sizes, observed=True):
            rows.append(
                {
                    "part": part,
                    "level": params["level"],
                    "interval": name,
                    "group": str(label),
                    "n": len(chunk),
                    **interval_quality(chunk, kind),
                }
            )
    return rows


def plot_intervals(iv: pd.DataFrame, run: str, level: float) -> None:
    iv = iv.sort_values(TARGET).reset_index(drop=True)
    x = np.arange(len(iv))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, kind, title in (
        (axes[0], "mul", "мультипликативный: ŷ·exp(q)"),
        (axes[1], "add", "аддитивный: ŷ + q"),
    ):
        lo, hi = iv[f"{kind}_low"], iv[f"{kind}_high"]
        inside = (iv[TARGET] >= lo) & (iv[TARGET] <= hi)
        ax.vlines(
            x,
            lo.clip(lower=1),
            hi,
            color="tab:blue",
            alpha=0.45,
            lw=6,
            label=f"интервал {int(level * 100)} %",
        )
        ax.scatter(
            x[inside], iv.loc[inside, TARGET], color="black", s=18, zorder=3, label="истина внутри"
        )
        ax.scatter(
            x[~inside],
            iv.loc[~inside, TARGET],
            color="tab:red",
            s=30,
            zorder=3,
            marker="x",
            label="истина вне интервала",
        )
        neg = lo < 0
        if neg.any():
            ax.scatter(
                x[neg],
                np.ones(neg.sum()),
                marker="v",
                color="tab:orange",
                s=30,
                zorder=4,
                label="нижняя граница < 0",
            )
        ax.set_yscale("log")
        ax.set_xlabel("рыбы теста, по возрастанию массы")
        ax.set_title(f"{title}; покрытие {inside.mean():.0%}")
    axes[0].set_ylabel("масса, г (лог. шкала)")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{run}: интервалы прогноза на тесте (калибровка на остатках CV)")
    fig.tight_layout()
    fig.savefig(FIGURES / f"lab02-{run}-prediction-intervals.png", dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config("configs/base.yaml")["bootstrap"]
    B, seed, conf = cfg["n_resamples"], cfg["seed"], cfg["confidence"]
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(RESULTS)
    selected = results.loc[results["selected"], "run_name"].item()
    preds = {
        part: {
            r: pd.read_csv(RUNS / r / f"predictions_{part}.csv", index_col="row")
            for r in results["run_name"]
        }
        for part in ("test", "oof")
    }

    # 1. доверительные интервалы метрик
    rows = []
    for part, by_run in preds.items():
        n = len(next(iter(by_run.values())))
        idx = resample_indices(n, B, seed)  # одни и те же выборки для всех моделей
        for run, pred in by_run.items():
            rows.append({"part": part, "run": run, "n": n, **metric_cis(pred, idx, conf)})
    cis = pd.DataFrame(rows)
    cis.to_csv(TABLES / "lab02-bootstrap-ci.csv", index=False, float_format="%.4f")

    # 2. парный bootstrap: MSE(модель) − MSE(итоговая)
    rows = []
    for part, by_run in preds.items():
        base = by_run[selected]
        n = len(base)
        idx = resample_indices(n, B, seed)
        base_mse = metrics_on_resamples(
            base[TARGET].to_numpy(float), base["prediction"].to_numpy(float), idx
        )["mse"]
        for run, pred in by_run.items():
            if run == selected:
                continue
            assert (pred.index == base.index).all(), "строки прогнозов должны совпадать"
            mse = metrics_on_resamples(
                pred[TARGET].to_numpy(float), pred["prediction"].to_numpy(float), idx
            )["mse"]
            delta = mse - base_mse
            lo, hi = percentile_ci(delta, conf)
            rows.append(
                {
                    "part": part,
                    "run": run,
                    "vs": selected,
                    "delta_mse": float(delta.mean()),
                    "delta_ci_low": lo,
                    "delta_ci_high": hi,
                    "share_selected_better": float((delta > 0).mean()),
                    "significant": bool(lo > 0 or hi < 0),
                }
            )
    paired = pd.DataFrame(rows)
    paired.to_csv(TABLES / "lab02-bootstrap-paired.csv", index=False, float_format="%.4f")

    # 3. интервалы прогноза: калибровка на CV вне выборки, проверка на тесте
    rows, params_all = [], []
    calib, test = preds["oof"][selected], preds["test"][selected]
    for level in cfg["interval_levels"]:
        iv_test, params = prediction_intervals(calib, test, level)
        iv_calib, _ = prediction_intervals(calib, calib, level)
        rows += interval_report(iv_calib, params, "cv (калибровка)")
        rows += interval_report(iv_test, params, "test")
        params_all.append(params)
        if level == min(cfg["interval_levels"]):  # на 80 % различие видов нагляднее
            iv_test.to_csv(
                TABLES / f"lab02-{selected}-prediction-intervals-test.csv", float_format="%.2f"
            )
            plot_intervals(iv_test, selected, level)
    quality = pd.DataFrame(rows)
    quality.to_csv(TABLES / "lab02-prediction-intervals.csv", index=False, float_format="%.4f")
    (TABLES / "lab02-prediction-interval-params.json").write_text(
        json.dumps(params_all, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pd.set_option("display.width", 200)
    show = [
        "part",
        "run",
        "mse",
        "mse_ci_low",
        "mse_ci_high",
        "mae",
        "mae_ci_low",
        "mae_ci_high",
        "r2",
        "r2_ci_low",
        "r2_ci_high",
    ]
    print(f"Повторных выборок: {B}, seed {seed}, уровень {conf:.0%}\n")
    print("== доверительные интервалы метрик\n", cis[show].round(3).to_string(index=False))
    print(f"\n== парный bootstrap против {selected}\n", paired.round(3).to_string(index=False))
    print("\n== интервалы прогноза\n", pd.DataFrame(params_all).round(3).to_string(index=False))
    print(quality.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
