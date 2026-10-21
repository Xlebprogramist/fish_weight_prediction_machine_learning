"""Воспроизводимость и запись запусков в MLflow."""

from __future__ import annotations

import os
import platform
import random
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np

from src.data import PROJECT_ROOT


def set_seed(seed: int) -> None:
    """Начальное значение генератора для Python и NumPy.

    scikit-learn получает seed через параметр random_state в конфигурации.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — глобальный генератор для кода, не принимающего rng


def git_info() -> dict[str, str]:
    def run(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "git.commit": run("rev-parse", "HEAD"),
        "git.branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git.dirty": str(bool(status) and status != "unknown").lower(),
    }


def env_info() -> dict[str, str]:
    info = {
        "env.python": platform.python_version(),
        "env.platform": platform.platform(),
        "env.machine": platform.machine(),
    }
    for pkg in ("scikit-learn", "numpy", "pandas", "scipy", "mlflow"):
        try:
            info[f"env.{pkg}"] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info[f"env.{pkg}"] = "not-installed"
    return info


def log_to_mlflow(
    cfg: dict,
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict,
    run_dir: Path,
    model,
    input_example,
) -> str:
    """Записывает один запуск в MLflow и возвращает его run_id."""
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_artifacts(str(run_dir), artifact_path="run")
        try:  # MLflow 3.x
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
        except TypeError:  # MLflow 2.x
            mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)
        return run.info.run_id


def attach_to_run(cfg: dict, run_id: str, files: list[Path], tags: dict | None = None) -> None:
    """Добавляет файлы (например, паспорт модели) и теги к уже записанному запуску."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    for path in files:
        client.log_artifact(run_id, str(path))
    for key, value in (tags or {}).items():
        client.set_tag(run_id, key, value)
