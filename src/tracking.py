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
        log_sklearn_model(model, input_example)
        return run.info.run_id


def _own_model_types(model) -> list[str]:
    """Типы внутри модели, которые skops не считает доверенными по умолчанию.

    Модель собрана этим же скриптом из классов scikit-learn/NumPy и функций
    src.pipeline, поэтому её типы можно явно разрешить при сохранении.
    """
    import skops.io as sio

    return sio.get_untrusted_types(data=sio.dumps(model))


def log_sklearn_model(model, input_example) -> None:
    """Сохраняет цепочку обработки в MLflow с учётом различий версий MLflow.

    MLflow ≥ 3.x по умолчанию сериализует модели через skops и требует явный
    список доверенных типов (иначе — MlflowException: untrusted types). Если
    skops недоступен или сохранение через него не удалось, используется cloudpickle.
    MLflow 2.x не знает параметр name — тогда artifact_path.
    """
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException

    attempts = []
    try:
        attempts.append({"skops_trusted_types": _own_model_types(model)})
    except Exception:  # skops не установлен или не умеет сохранить объект
        pass
    attempts.append({"serialization_format": "cloudpickle"})

    last_error: Exception | None = None
    for extra in attempts:
        for target in ({"name": "model"}, {"artifact_path": "model"}):
            try:
                mlflow.sklearn.log_model(model, input_example=input_example, **target, **extra)
                return
            except TypeError as err:  # параметр не поддерживается этой версией MLflow
                last_error = err
            except MlflowException as err:  # отказ сериализации
                last_error = err
                break
    raise RuntimeError("Не удалось сохранить модель в MLflow") from last_error


def attach_to_run(cfg: dict, run_id: str, files: list[Path], tags: dict | None = None) -> None:
    """Добавляет файлы (например, паспорт модели) и теги к уже записанному запуску."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    for path in files:
        client.log_artifact(run_id, str(path))
    for key, value in (tags or {}).items():
        client.set_tag(run_id, key, value)
