import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn.downbeat_baselines import tunable_split_ids  # noqa: E402

SPLITS = {"train": ["a", "b"], "val": ["c", "d"], "test": ["e"], "seed": 1337}


def test_the_test_split_is_refused_by_name_before_anything_is_read():
    with pytest.raises(RuntimeError, match="read once, by the verdict"):
        tunable_split_ids(SPLITS, "test")


def test_the_test_split_is_refused_even_when_the_splits_document_is_empty():
    with pytest.raises(RuntimeError, match="read once, by the verdict"):
        tunable_split_ids({}, "test")


def test_val_passes_and_returns_its_ids():
    assert tunable_split_ids(SPLITS, "val") == ["c", "d"]
    assert tunable_split_ids(SPLITS, "train") == ["a", "b"]


def test_an_unknown_split_is_named_rather_than_returning_nothing():
    with pytest.raises(RuntimeError, match="no 'holdout' split"):
        tunable_split_ids(SPLITS, "holdout")
