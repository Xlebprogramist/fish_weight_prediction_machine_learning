"""Экспорт метаданных MLflow в репозиторий (сам журнал mlruns/ не коммитится).

    python -m src.export_mlflow

Создаёт:
- reports/mlflow/runs.csv        — все запуски эксперимента: параметры, метрики, теги;
- reports/mlflow/structure.txt   — снимок структуры запусков и их артефактов.
"""

from __future__ import annotations

from src.config import load_config
from src.data import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports" / "mlflow"


def main() -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    cfg = load_config("configs/base.yaml")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg["mlflow"]["experiment"])
    if experiment is None:
        raise SystemExit("Эксперимент не найден: сначала python -m src.train --all")

    OUT.mkdir(parents=True, exist_ok=True)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time ASC"]
    )
    runs.to_csv(OUT / "runs.csv", index=False)

    def walk(run_id: str, path: str = "", depth: int = 1) -> list[str]:
        lines = []
        for item in client.list_artifacts(run_id, path or None):
            name = item.path.rsplit("/", 1)[-1]
            lines.append("    " * depth + name + ("/" if item.is_dir else ""))
            if item.is_dir:
                lines.extend(walk(run_id, item.path, depth + 1))
        return lines

    lines = [f"experiment: {experiment.name} (id {experiment.experiment_id})", ""]
    for _, row in runs.iterrows():
        lines.append(
            f"{row['tags.mlflow.runName']}  run_id={row['run_id']}  "
            f"commit={str(row.get('tags.git.commit', ''))[:7]}  "
            f"cv_mse={row['metrics.cv_mse_mean']:.1f}  test_mse={row['metrics.test_mse']:.1f}"
        )
        lines.extend(walk(row["run_id"]))
        lines.append("")
    # MLflow 3 хранит сохранённые модели отдельно от артефактов запуска
    try:
        models = mlflow.search_logged_models(
            experiment_ids=[experiment.experiment_id], output_format="list"
        )
    except AttributeError:  # MLflow 2.x: модель лежит в артефактах запуска
        models = []
    if models:
        names = dict(zip(runs["run_id"], runs["tags.mlflow.runName"], strict=False))
        lines.append("logged models (mlflow.sklearn.log_model):")
        for m in models:
            lines.append(
                f"    {m.name}  model_id={m.model_id}  "
                f"run={names.get(m.source_run_id, m.source_run_id)}"
            )
    (OUT / "structure.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(runs)} запусков -> {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
