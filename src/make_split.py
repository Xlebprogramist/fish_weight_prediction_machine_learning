"""Однократное замораживание тестовой выборки (выполняется до любых экспериментов).

    python -m src.make_split --config configs/base.yaml

Повторный запуск с теми же данными и seed даёт те же индексы; если файл
разбиения уже существует, он не перезаписывается без флага --force.
"""

from __future__ import annotations

import argparse
from collections import Counter

from src.config import load_config
from src.data import file_sha256, load_data, make_split, resolve, save_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg, split_cfg = cfg["data"], cfg["split"]
    out = resolve(data_cfg["split_path"])
    if out.exists() and not args.force:
        raise SystemExit(f"{out} уже существует: тестовая выборка заморожена")

    df = load_data(data_cfg["path"])
    split = make_split(df, split_cfg["test_size"], split_cfg["seed"], split_cfg["stratify"])
    record = {
        "data_path": data_cfg["path"],
        "data_sha256": file_sha256(data_cfg["path"]),
        "rule": (
            f"sklearn.model_selection.train_test_split(test_size={split_cfg['test_size']}, "
            f"random_state={split_cfg['seed']}, stratify={split_cfg['stratify']}) "
            "по позициям строк исходного файла"
        ),
        "seed": split_cfg["seed"],
        "n_train": len(split["train_idx"]),
        "n_test": len(split["test_idx"]),
        **split,
    }
    save_split(record, out)
    species = df[split_cfg["stratify"]]
    print(f"train={record['n_train']} test={record['n_test']} -> {out}")
    print("test по видам:", dict(Counter(species.iloc[split["test_idx"]])))


if __name__ == "__main__":
    main()
