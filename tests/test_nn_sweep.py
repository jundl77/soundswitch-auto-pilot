import dataclasses
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn import sweep  # noqa: E402
from nn.decoder import DecodeParams  # noqa: E402
from nn.sweep import (  # noqa: E402
    ABLATION_AXES,
    ablate,
    refinement_axes,
    run_configs,
    run_sweep,
    select_config,
    summarise_multi,
)


class FakeScore:
    def __init__(self, macro: float, flicker_count: int, crisp: float):
        self._macro = macro
        self._crisp = crisp
        self.counts = {"drop": (10.0, 2.0, 3.0)}
        self.no_intent_sec = 0.0
        self.exposure_sec = 600.0
        self.accuracy = macro
        self.boundary = {
            "class": {2.0: {"overall": {"n_truth": 9, "n_pred": 7, "matched": 5}}}
        }
        self.labels = ("drop",)
        self.flicker_count = flicker_count

    def boundary_prf(self, stream, tolerance, kind="overall", label=None):
        if tolerance == 0.5:
            return (self._crisp, self._crisp, self._crisp)
        return (0.7, 0.7, 0.7)

    def f1(self, label):
        return self._macro


def fake_result(params, macro, flicker, crisp=0.5):
    return {
        "params": params,
        "macro_f1": macro,
        "flicker_per_min": flicker,
        "score": FakeScore(macro, int(flicker * 10), crisp),
        "per_track": [],
        "confusion": None,
    }


class StubCache:
    def __init__(self, tag):
        self.tag = tag

    def for_params(self, params):
        return self.tag


def patched_evaluate(monkeypatch, per_seed):
    def fake_evaluate_config(inputs, priors, params, *, space, claims=None,
                             with_confusion=False):
        macro, flicker, crisp = per_seed[inputs](params)
        return fake_result(params, macro, flicker, crisp)

    monkeypatch.setattr(sweep, "evaluate_config", fake_evaluate_config)


def test_summarise_multi_selects_on_mean_macro_and_max_flicker():
    params = DecodeParams()
    row = summarise_multi(
        [fake_result(params, 0.6, 0.30, crisp=0.4),
         fake_result(params, 0.4, 0.40, crisp=0.6)], "raw9")
    assert row["macro_f1"] == pytest.approx(0.5)
    assert row["flicker_per_min"] == pytest.approx(0.40)
    assert row["flicker_per_min_mean"] == pytest.approx(0.35)
    assert row["crispness_05"] == pytest.approx(0.5)
    assert [seed["macro_f1"] for seed in row["seeds"]] == [0.6, 0.4]
    assert [seed["crispness_05"] for seed in row["seeds"]] == [0.4, 0.6]


def test_run_configs_single_cache_row_shape_is_unchanged(monkeypatch):
    patched_evaluate(monkeypatch, {"a": lambda p: (0.5, 0.2, 0.5)})
    rows = run_configs(StubCache("a"), None, [DecodeParams()])
    assert len(rows) == 1
    assert "seeds" not in rows[0]
    assert rows[0]["macro_f1"] == 0.5


def test_run_configs_multi_cache_scores_every_seed(monkeypatch):
    patched_evaluate(monkeypatch, {"a": lambda p: (0.6, 0.2, 0.5),
                                   "b": lambda p: (0.4, 0.3, 0.5)})
    rows = run_configs([StubCache("a"), StubCache("b")], None, [DecodeParams()])
    assert rows[0]["macro_f1"] == pytest.approx(0.5)
    assert rows[0]["flicker_per_min"] == pytest.approx(0.3)
    assert len(rows[0]["seeds"]) == 2


def test_select_config_honours_a_tighter_budget():
    def row(lag, macro):
        return {"params": dataclasses.asdict(DecodeParams(lag_bars=lag)),
                "macro_f1": macro, "flicker_per_min": 0.1}

    rows = [row(3, 0.9), row(2, 0.5), row(0, 0.4)]
    assert select_config(rows, 1.0)["macro_f1"] == 0.9
    assert select_config(rows, 1.0, budget_bars=2)["macro_f1"] == 0.5


def test_run_sweep_quick_starts_from_the_given_base(monkeypatch):
    captured = []

    def scorer(params):
        captured.append(params)
        return (0.5, 0.2, 0.5)

    patched_evaluate(monkeypatch, {"a": scorer})
    base = DecodeParams(min_coverage=1, lag_bars=2)
    result = run_sweep(StubCache("a"), None, flicker_ceiling=1.0, quick=True,
                       base=base, budget_bars=2)
    assert captured
    assert all(p.min_coverage == 1 for p in captured)
    quick_rows = [row for row in result["rows"] if row["stage"] == "quick"]
    assert quick_rows
    assert all(row["params"]["lag_bars"] == 2 for row in quick_rows)
    assert result["budget_bars"] == 2
    assert result["chosen"]["params"]["min_coverage"] == 1


def test_refinement_axes_takes_extra_axes():
    best = DecodeParams(outro_escape=0.01)
    axes = refinement_axes(best, {"outro_escape": (0.0, 0.01, 0.02, 0.04)})
    assert axes["outro_escape"] == (0.0, 0.01, 0.02)
    assert set(axes) == {"prior_strength", "drop_miss_cost", "boundary_weight",
                         "boundary_ref", "floor_scale", "outro_escape"}


def test_ablate_walks_the_axes_it_is_given(monkeypatch):
    patched_evaluate(monkeypatch, {"a": lambda p: (0.5, 0.2, 0.5)})
    curves, fresh = ablate(StubCache("a"), None, DecodeParams(), {},
                           axes={"outro_escape": (0.0, 0.02)})
    assert set(curves) == {"outro_escape"}
    assert len(curves["outro_escape"]) == 2
    assert len(fresh) == 2


def test_run_sweep_full_stages_include_extra_axis_and_respect_budget(monkeypatch):
    def scorer_a(params):
        macro = 0.5 + 0.05 * params.lag_bars + 0.1 * params.outro_escape
        return (macro, 0.2, 0.5)

    def scorer_b(params):
        macro = 0.45 + 0.05 * params.lag_bars + 0.1 * params.outro_escape
        return (macro, 0.25, 0.5)

    patched_evaluate(monkeypatch, {"a": scorer_a, "b": scorer_b})
    base = DecodeParams(min_coverage=1, lag_bars=2)
    result = run_sweep([StubCache("a"), StubCache("b")], None,
                       flicker_ceiling=1.0, base=base, budget_bars=2,
                       extra_stage_axes={"outro_escape": (0.0, 0.02)})
    stage_names = [stage["name"] for stage in result["stages"]]
    assert "outro_escape" in stage_names
    assert result["chosen"]["params"]["lag_bars"] <= 2
    assert any(row["params"]["lag_bars"] > 2 for row in result["lag_curve"])
    assert {axis["axis"] for axis in result["sensitivity"]} >= {"outro_escape",
                                                                "lag_bars"}


def test_ablation_axes_are_untouched_by_the_extra_axis_path():
    assert "outro_escape" not in ABLATION_AXES
