"""Чтение YAML-конфигураций: базовый файл + файл эксперимента."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from src.data import resolve


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    """Читает конфигурацию; ключ `extends` подключает базовый файл (путь от корня проекта)."""
    with open(resolve(path), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load_config(parent), cfg)
    cfg.setdefault("_source", str(path))
    return cfg
