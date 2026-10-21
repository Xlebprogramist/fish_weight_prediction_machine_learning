"""Графики запуска: остатки и прогноз против истинного значения."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import TARGET  # noqa: E402


def save_run_plots(pred: pd.DataFrame, run_dir: Path, title: str) -> None:
    species = sorted(pred["Species"].unique())
    colors = dict(zip(species, plt.cm.tab10.colors, strict=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    for sp, part in pred.groupby("Species"):
        ax.scatter(part["prediction"], part["residual"], s=28, color=colors[sp], label=sp)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Прогноз массы, г")
    ax.set_ylabel("Остаток y − ŷ, г")
    ax.set_title(f"{title}: остатки (тест)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(run_dir / "residuals.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for sp, part in pred.groupby("Species"):
        ax.scatter(part[TARGET], part["prediction"], s=28, color=colors[sp], label=sp)
    hi = max(pred[TARGET].max(), pred["prediction"].max()) * 1.05
    lo = min(0, pred["prediction"].min())
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
    ax.set_xlabel("Истинная масса, г")
    ax.set_ylabel("Прогноз, г")
    ax.set_title(f"{title}: прогноз против истины (тест)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(run_dir / "pred_vs_true.png", dpi=120)
    plt.close(fig)
