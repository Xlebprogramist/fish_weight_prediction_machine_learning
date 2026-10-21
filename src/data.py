"""Загрузка данных, контрольная сумма и замороженное разбиение train/test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET = "Weight"
CATEGORICAL = ["Species"]
NUMERIC = ["Length1", "Length2", "Length3", "Height", "Width"]
COLUMNS = ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"]


def resolve(path: str | Path) -> Path:
    """Путь относительно корня проекта (если он не абсолютный)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: str | Path) -> str:
    """SHA-256 файла — версия данных для журнала экспериментов."""
    digest = hashlib.sha256()
    with open(resolve(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(path: str | Path) -> pd.DataFrame:
    """Читает CSV и проверяет схему. Строки не меняются и не переупорядочиваются."""
    df = pd.read_csv(resolve(path))
    if list(df.columns) != COLUMNS:
        raise ValueError(f"Неожиданная схема: {list(df.columns)}, ожидалось {COLUMNS}")
    if df[COLUMNS].isna().any().any():
        raise ValueError("В данных есть пропуски")
    return df


def make_split(df: pd.DataFrame, test_size: float, seed: int, stratify: str | None) -> dict:
    """Детерминированное разбиение по позициям строк исходного файла."""
    positions = np.arange(len(df))
    labels = df[stratify].to_numpy() if stratify else None
    train_idx, test_idx = train_test_split(
        positions, test_size=test_size, random_state=seed, stratify=labels
    )
    return {
        "train_idx": sorted(int(i) for i in train_idx),
        "test_idx": sorted(int(i) for i in test_idx),
    }


def save_split(split: dict, path: str | Path) -> None:
    with open(resolve(path), "w", encoding="utf-8") as fh:
        json.dump(split, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def load_split(path: str | Path, data_path: str | Path | None = None) -> dict:
    """Читает замороженное разбиение; при указании data_path сверяет контрольную сумму."""
    with open(resolve(path), encoding="utf-8") as fh:
        split = json.load(fh)
    if data_path is not None and split.get("data_sha256") != file_sha256(data_path):
        raise ValueError(
            "Контрольная сумма данных не совпадает с той, для которой заморожено разбиение"
        )
    if set(split["train_idx"]) & set(split["test_idx"]):
        raise ValueError("Части выборки пересекаются")
    return split


def train_test_frames(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Обучающая и тестовая таблицы по замороженному разбиению из конфигурации."""
    df = load_data(cfg["data"]["path"])
    split = load_split(cfg["data"]["split_path"], cfg["data"]["path"])
    return df.iloc[split["train_idx"]].copy(), df.iloc[split["test_idx"]].copy()
