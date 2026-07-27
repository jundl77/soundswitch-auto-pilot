"""Split hygiene for the downbeat nulls (``training/nn/downbeat_baselines.py``).

The module computes the baselines a downbeat F1 has to be read against, and one
of them -- the phase histogram of predicted peaks -- reads the annotated bar
phase to say *which decoder knob is worth tuning*.  That makes it a tuning
instrument by the branch's own rule, and tuning instruments do not touch the test
split.  The rule is guarded on split **membership** rather than on the flag that
usually selects it, because an explicit ``--split`` walks straight past a
help-text warning.

Deliberately tested against a plain dict rather than the corpus: the refusal
happens before anything is loaded, and a guard that needed 1,400 tracks to prove
itself would not be run.
"""
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


def test_the_refusal_needs_no_corpus_at_all():
    # Nothing about the refusal may depend on the splits document, or a caller
    # with a hand-built one walks past it.
    with pytest.raises(RuntimeError, match="read once, by the verdict"):
        tunable_split_ids({}, "test")


def test_val_passes_and_returns_its_ids():
    assert tunable_split_ids(SPLITS, "val") == ["c", "d"]
    assert tunable_split_ids(SPLITS, "train") == ["a", "b"]


def test_an_unknown_split_is_named_rather_than_returning_nothing():
    with pytest.raises(RuntimeError, match="no 'holdout' split"):
        tunable_split_ids(SPLITS, "holdout")
