"""Быстрые проверки данных и замороженного разбиения (без обучения моделей)."""

import pandas as pd
import pytest

from src.config import load_config
from src.data import COLUMNS, NUMERIC, file_sha256, load_data, load_split, resolve

CFG = load_config("configs/base.yaml")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data(CFG["data"]["path"])


@pytest.fixture(scope="module")
def split() -> dict:
    return load_split(CFG["data"]["split_path"], CFG["data"]["path"])


def test_schema_and_physical_ranges(df):
    assert list(df.columns) == COLUMNS
    assert (df[["Weight", *NUMERIC]] > 0).all().all()
    assert (df["Length1"] <= df["Length2"]).all() and (df["Length2"] <= df["Length3"]).all()
    assert (df["Species"].str.strip() == df["Species"]).all()


def test_train_test_indices_do_not_overlap(df, split):
    train, test = set(split["train_idx"]), set(split["test_idx"])
    assert not train & test
    assert train | test == set(range(len(df)))


def test_split_is_bound_to_data_version(split):
    assert split["data_sha256"] == file_sha256(CFG["data"]["path"])


def test_demo_holdout_is_contained_in_training_file(df):
    """Документирует утечку исходного проекта: demo-holdout целиком внутри обучающего файла."""
    demo = pd.read_csv(resolve("data/fish_holdout_demo.csv"))
    assert len(demo.merge(df, how="inner")) == len(demo)
