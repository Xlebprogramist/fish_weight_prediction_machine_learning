"""Обучение и оценивание экспериментов ЛР2.

    python -m src.train --all                                   # все конфигурации E0–E4
    python -m src.train --config configs/experiments/e2_ridge_loglog.yaml
    python -m src.train --all --no-mlflow                       # без записи в MLflow

Для каждой конфигурации:
1. гиперпараметры (если задана сетка) выбираются GridSearchCV на RepeatedKFold
   внутри обучающей выборки, метрика — MSE в граммах;
2. метрики CV (среднее и std по 15 разбиениям) считаются для выбранных
   гиперпараметров тем же набором разбиений;
3. модель обучается на всей обучающей выборке и один раз оценивается на
   замороженном тесте;
4. результаты пишутся в runs/<имя>/ и в MLflow.
Итоговая модель выбирается по правилу из configs/base.yaml (только по CV).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RepeatedKFold

from src.config import load_config
from src.data import CATEGORICAL, NUMERIC, PROJECT_ROOT, TARGET, file_sha256, train_test_frames
from src.metrics import regression_metrics
from src.pipeline import build_pipeline, coefficients, grid_prefix, numeric_columns
from src.plots import save_run_plots
from src.tracking import attach_to_run, env_info, git_info, log_to_mlflow, set_seed

EXPERIMENTS_DIR = PROJECT_ROOT / "configs" / "experiments"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_CSV = PROJECT_ROOT / "reports" / "lab02-results.csv"
BEST_MODEL = PROJECT_ROOT / "models" / "model.joblib"
MODEL_CARD = PROJECT_ROOT / "models" / "model-card.md"
INPUT_COLUMNS = CATEGORICAL + NUMERIC


def drop_weight_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Правило IQR исходного проекта, но только по обучающей части и без жёсткого индекса."""
    q1, q3 = train[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    keep = train[TARGET].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return train[keep]


def cv_splitter(cfg: dict) -> RepeatedKFold:
    cv = cfg["cv"]
    return RepeatedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"], random_state=cv["seed"])


def tune(cfg: dict, pipe, X: pd.DataFrame, y: pd.Series, splits) -> tuple[dict, dict]:
    """Подбор по заранее заданной сетке; возвращает лучшие параметры и сведения о бюджете."""
    grid = cfg["model"].get("grid")
    if not grid:
        return {}, {"grid_size": 1, "fits": 0}
    prefix = grid_prefix(cfg)
    param_grid = {prefix + k: v for k, v in grid.items()}
    search = GridSearchCV(
        pipe, param_grid, cv=splits, scoring=cfg["cv"]["scoring"], refit=False, n_jobs=1
    )
    search.fit(X, y)
    best = {k.removeprefix(prefix): v for k, v in search.best_params_.items()}
    n_candidates = len(search.cv_results_["params"])
    return best, {"grid_size": n_candidates, "fits": n_candidates * len(splits)}


def cross_validate(cfg: dict, pipe, train: pd.DataFrame, splits) -> tuple[pd.DataFrame, ...]:
    """Метрики по каждому разбиению, прогнозы вне выборки (первый повтор) и коэффициенты."""
    rows, oof, coefs = [], [], []
    n_splits = cfg["cv"]["n_splits"]
    for k, (fit_idx, val_idx) in enumerate(splits):
        fit_part, val_part = train.iloc[fit_idx], train.iloc[val_idx]
        if cfg["train"].get("drop_weight_outliers_iqr"):
            fit_part = drop_weight_outliers(fit_part)
        model = clone(pipe).fit(fit_part[INPUT_COLUMNS], fit_part[TARGET])
        pred = model.predict(val_part[INPUT_COLUMNS])
        rows.append({"fold": k, **regression_metrics(val_part[TARGET], pred)})
        if k < n_splits:  # первый повтор покрывает каждую обучающую строку ровно один раз
            oof.append(
                val_part.assign(prediction=pred, residual=val_part[TARGET] - pred, fold=k)
            )
        coef = coefficients(model)
        if coef is not None:
            coefs.append(coef.rename(k))
    coef_df = pd.DataFrame(coefs) if coefs else pd.DataFrame()
    return pd.DataFrame(rows), pd.concat(oof).sort_index(), coef_df


def flat_params(cfg: dict, best: dict, data_sha: str, n_train_fit: int) -> dict:
    model_params = {**cfg["model"].get("params", {}), **best}
    params = {
        "experiment_id": cfg["id"],
        "question": cfg.get("question", ""),
        "seed": cfg["seed"],
        "data.path": cfg["data"]["path"],
        "data.sha256": data_sha,
        "split.path": cfg["data"]["split_path"],
        "split.rule": "train_test_split(test_size=0.2, random_state=42, stratify=Species)",
        "cv": f"RepeatedKFold(n_splits={cfg['cv']['n_splits']}, n_repeats="
        f"{cfg['cv']['n_repeats']}, random_state={cfg['cv']['seed']})",
        "features.numeric": ",".join(numeric_columns(cfg["features"])),
        "features.categorical": ",".join(cfg["features"].get("categorical", [])),
        "features.numeric_transform": cfg["features"].get("numeric_transform", "none"),
        "features.scale": cfg["features"].get("scale", True),
        "target.transform": cfg["target"].get("transform", "none"),
        "train.drop_weight_outliers_iqr": cfg["train"].get("drop_weight_outliers_iqr", False),
        "train.n_rows_fit": n_train_fit,
        "model.class": cfg["model"]["name"],
        "model.grid": json.dumps(cfg["model"].get("grid", {})),
    }
    params.update({f"model.{k}": v for k, v in model_params.items()})
    return params


def run_experiment(cfg: dict, use_mlflow: bool) -> dict:
    set_seed(cfg["seed"])
    name = cfg["run_name"]
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train, test = train_test_frames(cfg)
    splits = list(cv_splitter(cfg).split(train))
    pipe = build_pipeline(cfg)

    best, budget = tune(cfg, pipe, train[INPUT_COLUMNS], train[TARGET], splits)
    if best:
        pipe.set_params(**{grid_prefix(cfg) + k: v for k, v in best.items()})

    folds, oof, coefs = cross_validate(cfg, pipe, train, splits)

    fit_part = drop_weight_outliers(train) if cfg["train"].get("drop_weight_outliers_iqr") else train
    model = clone(pipe).fit(fit_part[INPUT_COLUMNS], fit_part[TARGET])
    test_pred = model.predict(test[INPUT_COLUMNS])
    test_metrics = regression_metrics(test[TARGET], test_pred)
    seconds = time.perf_counter() - started

    metrics = {}
    for m in ("mse", "mae", "r2"):
        metrics[f"cv_{m}_mean"] = float(folds[m].mean())
        metrics[f"cv_{m}_std"] = float(folds[m].std(ddof=1))
        metrics[f"test_{m}"] = test_metrics[m]
    metrics["test_n_negative_predictions"] = int((test_pred < 0).sum())
    metrics["cv_n_negative_predictions"] = int((oof["prediction"] < 0).sum())
    metrics["train_seconds"] = seconds
    metrics["grid_fits"] = budget["fits"]

    predictions = test.assign(
        prediction=test_pred,
        residual=test[TARGET] - test_pred,
        abs_error=np.abs(test[TARGET] - test_pred),
    )
    predictions.index.name = "row"
    oof.index.name = "row"
    predictions.to_csv(run_dir / "predictions_test.csv")
    oof.to_csv(run_dir / "predictions_oof.csv")
    folds.to_csv(run_dir / "cv_folds.csv", index=False)
    if not coefs.empty:
        summary = pd.DataFrame(
            {
                "coef_final": coefficients(model),
                "coef_cv_mean": coefs.mean(),
                "coef_cv_std": coefs.std(ddof=1),
            }
        )
        summary["coef_cv_cv"] = summary["coef_cv_std"] / summary["coef_cv_mean"].abs()
        summary.to_csv(run_dir / "coefficients.csv")
    save_run_plots(predictions, run_dir, title=name)

    data_sha = file_sha256(cfg["data"]["path"])
    params = flat_params(cfg, best, data_sha, len(fit_part))
    tags = {**git_info(), **env_info(), "lab": "02", "experiment_group": cfg["id"].split("_")[0]}
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, fh,
                       allow_unicode=True, sort_keys=False)
    for fname, payload in (("params.json", params), ("metrics.json", metrics), ("tags.json", tags)):
        with open(run_dir / fname, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    joblib.dump(model, run_dir / "model.joblib")

    run_id = ""
    if use_mlflow:
        # уникальное имя запуска: конфигурация + короткий хеш коммита + время
        unique = f"{name}-{tags['git.commit'][:7]}-{time.strftime('%Y%m%d-%H%M%S')}"
        run_id = log_to_mlflow(
            cfg, unique, params, metrics, tags, run_dir, model, test[INPUT_COLUMNS].head(3)
        )
    print(
        f"{name:28s} CV MSE {metrics['cv_mse_mean']:9.1f} ± {metrics['cv_mse_std']:8.1f} | "
        f"test MSE {metrics['test_mse']:9.1f} MAE {metrics['test_mae']:6.1f} "
        f"R2 {metrics['test_r2']:.4f} | {seconds:5.1f} s"
    )
    return {
        "id": cfg["id"],
        "run_name": name,
        "mlflow_run_id": run_id,
        "model": cfg["model"]["name"],
        "features": params["features.numeric"]
        + ("+" + params["features.categorical"] if params["features.categorical"] else ""),
        "numeric_transform": params["features.numeric_transform"],
        "target_transform": params["target.transform"],
        "best_params": json.dumps(best),
        **metrics,
        "git_commit": tags["git.commit"],
    }


def select_best(results: pd.DataFrame) -> pd.Series:
    """Минимальная средняя MSE на CV; среди моделей в пределах одного std — первая по сложности."""
    order = results.sort_values("cv_mse_mean")
    leader = order.iloc[0]
    close = order[order["cv_mse_mean"] <= leader["cv_mse_mean"] + leader["cv_mse_std"]]
    complexity = {"dummy": 0, "linear": 1, "ridge": 2, "elasticnet": 2, "random_forest": 3,
                  "hist_gb": 3}
    return close.assign(c=close["model"].map(complexity)).sort_values(["c", "cv_mse_mean"]).iloc[0]


def experiment_configs(args) -> list[dict]:
    paths = sorted(EXPERIMENTS_DIR.glob("*.yaml")) if args.all else [Path(args.config)]
    return [load_config(p) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="все файлы configs/experiments/*.yaml")
    group.add_argument("--config", help="одна конфигурация эксперимента")
    parser.add_argument("--no-mlflow", action="store_true", help="не писать запуски в MLflow")
    args = parser.parse_args()

    results = [run_experiment(cfg, not args.no_mlflow) for cfg in experiment_configs(args)]
    if not args.all:
        return

    table = pd.DataFrame(results)
    best = select_best(table)
    table["selected"] = table["run_name"] == best["run_name"]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False, float_format="%.4f")

    BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(joblib.load(RUNS_DIR / best["run_name"] / "model.joblib"), BEST_MODEL)
    print(f"\nВыбрана по CV: {best['run_name']} -> {BEST_MODEL.relative_to(PROJECT_ROOT)}")
    if not args.no_mlflow and best["mlflow_run_id"]:
        attach_to_run(
            load_config(EXPERIMENTS_DIR / f"{best['run_name']}.yaml"),
            best["mlflow_run_id"],
            [MODEL_CARD] if MODEL_CARD.exists() else [],
            {"selected": "true"},
        )


if __name__ == "__main__":
    main()
