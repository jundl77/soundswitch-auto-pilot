"""#345 transition axis: the buildup->drop entry bonus, default-neutral."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn.decoder import (  # noqa: E402
    DecodeParams,
    FixedLagViterbi,
    load_decoder_config,
)
from nn.priors import Priors, transition_allowed  # noqa: E402
from nn.sweep import enumerate_configs  # noqa: E402

CLASSES = ("intro", "buildup", "breakdown", "drop", "outro")
INTRO, BUILDUP, BREAKDOWN, DROP, OUTRO = range(5)


def toy_priors(floor=4, hazard=0.25):
    n = len(CLASSES)
    transition = np.zeros((n, n), dtype=np.float64)
    for i, src in enumerate(CLASSES):
        for j, dst in enumerate(CLASSES):
            if transition_allowed(src, dst):
                transition[i, j] = 1.0
    rows = transition.sum(axis=1, keepdims=True)
    transition = np.divide(transition, rows, out=np.zeros_like(transition),
                           where=rows > 0)
    return Priors(
        classes=CLASSES,
        initial=np.full(n, 1.0 / n),
        transition=transition,
        floor_bars=np.full(n, int(floor), dtype=np.int64),
        hazard=np.full(n, float(hazard), dtype=np.float64),
        class_prior=np.full(n, 1.0 / n),
        corpus={},
    )


def one_hot(index, strength=0.97, n=5):
    row = np.full(n, (1.0 - strength) / (n - 1))
    row[index] = strength
    return row


def labels_of(decisions):
    return [d.label for d in decisions]


def climb_with_ambiguous_tail(ambiguity=0.02):
    """Intro, a clear climb, then bars where breakdown shades drop.

    The tail's per-bar evidence slightly favours breakdown, so a neutral
    decoder reads breakdown and only a preference on the buildup->drop edge
    can flip the entry -- the #345 case in miniature.
    """
    rows = [one_hot(INTRO)] * 4 + [one_hot(BUILDUP)] * 6
    tail = np.full(5, 0.02)
    tail[BREAKDOWN] = 0.47 + ambiguity / 2
    tail[DROP] = 0.47 - ambiguity / 2
    tail /= tail.sum()
    rows += [tail] * 8
    return np.asarray(rows)


def test_default_params_are_neutral_and_old_configs_still_load(tmp_path):
    assert DecodeParams().buildup_drop_bonus == 0.0

    old = {"chosen": {"lag_bars": 2, "prior_strength": 0.25},
           "class_space": ["intro", "buildup", "breakdown", "drop", "outro"]}
    path = tmp_path / "old_config.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    params = load_decoder_config(path)
    assert params.buildup_drop_bonus == 0.0

    new = {"chosen": {"lag_bars": 2, "buildup_drop_bonus": 0.7}}
    path2 = tmp_path / "new_config.json"
    path2.write_text(json.dumps(new), encoding="utf-8")
    assert load_decoder_config(path2).buildup_drop_bonus == 0.7

    # The config writer records the knob: asdict carries the field.
    assert "buildup_drop_bonus" in dataclasses.asdict(DecodeParams())


def test_neutral_bonus_is_byte_identically_the_knobless_decoder(monkeypatch):
    posteriors = climb_with_ambiguous_tail()
    reference = FixedLagViterbi(toy_priors(floor=1), lag_bars=2)

    monkeypatch.setattr(FixedLagViterbi, "_apply_buildup_drop_bonus",
                        lambda self, transition: None)
    knobless = FixedLagViterbi(toy_priors(floor=1), lag_bars=2)
    monkeypatch.undo()

    assert reference._transition.tobytes() == knobless._transition.tobytes()
    assert (labels_of(reference.decode(posteriors))
            == labels_of(knobless.decode(posteriors)))


def test_the_bonus_touches_exactly_the_buildup_to_drop_edge():
    neutral = FixedLagViterbi(toy_priors(floor=2), lag_bars=2)
    boosted = FixedLagViterbi(toy_priors(floor=2), lag_bars=2,
                              buildup_drop_bonus=0.7)
    with np.errstate(invalid="ignore"):
        delta = boosted._transition - neutral._transition
    delta = np.where(np.isnan(delta), 0.0, delta)  # -inf minus -inf
    source = int(neutral._final_state[BUILDUP])
    target = int(neutral._entry_state[DROP])
    assert delta[source, target] == pytest.approx(0.7)
    delta[source, target] = 0.0
    assert not delta.any()


def test_a_strong_bonus_resolves_an_ambiguous_pre_drop_bar_toward_drop():
    posteriors = climb_with_ambiguous_tail()
    neutral = FixedLagViterbi(toy_priors(floor=1), lag_bars=2)
    boosted = FixedLagViterbi(toy_priors(floor=1), lag_bars=2,
                              buildup_drop_bonus=6.0)

    neutral_labels = labels_of(neutral.decode(posteriors))
    boosted_labels = labels_of(boosted.decode(posteriors))
    assert neutral_labels != boosted_labels
    assert "drop" not in neutral_labels[10:]
    assert "drop" in boosted_labels[10:]


def test_a_non_finite_bonus_is_refused():
    with pytest.raises(ValueError, match="buildup_drop_bonus"):
        FixedLagViterbi(toy_priors(), buildup_drop_bonus=float("nan"))
    with pytest.raises(ValueError, match="buildup_drop_bonus"):
        FixedLagViterbi(toy_priors(), buildup_drop_bonus=float("inf"))


def test_the_sweep_can_enumerate_the_new_axis():
    configs = enumerate_configs(DecodeParams(),
                                {"buildup_drop_bonus": (0.0, 0.4, 0.7)})
    assert [c.buildup_drop_bonus for c in configs] == [0.0, 0.4, 0.7]
