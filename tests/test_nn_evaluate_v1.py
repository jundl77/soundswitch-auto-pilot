"""Tests for the offline NN evaluation adapter and the decoder sweep.

The adapter is the join between two things that count time differently: the
decoder commits per BAR, the evaluator scores per BEAT.  Everything that can go
quietly wrong lives in that join, and each family below pins one way it could.

**Undecoded is not wrong.**  ``decode_track`` deliberately says nothing about
``[0, first_downbeat)`` -- there is no bar grid there -- and nothing past the
final bar line.  Those beats carry no prediction, so scoring them as errors
would charge the network for the annotation grid's origin and would land the
charge almost entirely on ``intro`` and ``outro``, the two classes that own
those regions.  The evaluator already has the concept (``NO_INTENT``: beats
before the engine's first commit are excluded, not counted wrong), so the
adapter reuses the sentinel rather than inventing a second one.

**The claim map is the whole fairness argument.**  The rule classifier's
ATMOSPHERIC is credited against ``intro`` OR ``outro``, because an intent cannot
know where in the arrangement it sits.  A network predicting ``label_v1`` CAN,
so it is scored with identity claims -- ``intro`` predicted over ``outro`` is a
miss.  That is a genuinely higher bar, so the same run is also scored with
rule-equivalent claims (intro/outro collapsed back into one ambiguous class),
which is the comparison that answers "is the NN only winning because the two
sides were scored differently?".  Both mappings are exercised here, and the
stream tests pin the second-order consequence: an ambiguous claim also hides an
intro -> outro switch from the class stream, exactly as it does for the rule.

**The sweep's fast path must be the same decoder.**  A sweep that ran a
different code path from ``decode_track`` would be tuning something the runtime
never executes, so the equivalence is asserted directly -- including the part
that makes it fast, reusing one decoder instance across tracks.
"""
import dataclasses
import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from build_training_table import NO_INTENT  # noqa: E402
from evaluate_against_labels import (  # noqa: E402
    PRIMARY_TOLERANCE_SEC,
    SPACES,
    TOLERANCES_SEC,
    TrackBeats,
    aggregate,
)
from nn.decoder import (  # noqa: E402
    DecodeParams,
    bar_grid,
    bar_observations,
    decode_track,
)
from nn.evaluate_v1 import (  # noqa: E402
    UNDECODED,
    TrackInputs,
    _head_to_head,
    _per_track_deltas,
    beat_classes,
    build_decoder,
    decode_bars,
    decode_beats,
    identity_claims,
    load_inputs,
    rule_equivalent_claims,
    score_predicted,
)
from nn.sweep import (  # noqa: E402
    enumerate_configs,
    refinement_axes,
    select_config,
    sensitivity,
)
from tests.test_nn_decoder import (  # noqa: E402
    synthetic_npz,
    toy_priors,
    write_beat_csv,
)

BREAKDOWN, DROP = 2, 3


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def steady(count, step=0.5, t0=0.0):
    return tuple(t0 + index * step for index in range(count))


def track(times, labels, intents=None, track_id="t"):
    """A ``TrackBeats`` whose v1 labels are also valid canonical labels."""
    labels = tuple(labels)
    return TrackBeats(
        track_id=track_id,
        times=tuple(times),
        intents=tuple(intents if intents is not None else [NO_INTENT] * len(labels)),
        labels={space: labels for space in SPACES},
    )


# --------------------------------------------------------------------------- #
# The bar -> beat adapter
# --------------------------------------------------------------------------- #


def test_beat_classes_reads_each_beat_off_the_bar_that_contains_it():
    got = beat_classes(steady(8), (0.0, 2.0, 4.0), ["intro", "drop"])
    assert got == ("intro",) * 4 + ("drop",) * 4


def test_beat_classes_leaves_the_pre_downbeat_head_undecoded():
    # The bar grid starts at the first downbeat, so the decoder emits nothing
    # for the beats before it.  Undecoded, never misdecoded -- that head is
    # almost always `intro`, and charging it would fake an intro failure.
    got = beat_classes((0.1, 0.4, 0.6, 1.2), (0.5, 2.5), ["drop"])
    assert got == (UNDECODED, UNDECODED, "drop", "drop")


def test_beat_classes_leaves_the_sub_bar_tail_undecoded():
    got = beat_classes((0.0, 1.9, 2.0, 2.5), (0.0, 2.0), ["drop"])
    assert got == ("drop", "drop", UNDECODED, UNDECODED)


def test_beat_classes_puts_a_beat_exactly_on_a_bar_line_in_the_new_bar():
    assert beat_classes((2.0,), (0.0, 2.0, 4.0), ["intro", "drop"]) == ("drop",)


def test_beat_classes_refuses_more_labels_than_the_grid_has_bars():
    with pytest.raises(ValueError, match="bar"):
        beat_classes((0.0,), (0.0, 2.0), ["intro", "drop"])


# --------------------------------------------------------------------------- #
# Undecoded regions are excluded, not wrong
# --------------------------------------------------------------------------- #


def test_undecoded_beats_are_excluded_from_every_class_count():
    scored = score_predicted(
        track(steady(6), ["intro"] * 2 + ["drop"] * 4),
        "v1", (UNDECODED, UNDECODED, "drop", "drop", "drop", "drop"))
    assert scored.counts["intro"] == [0.0, 0.0, 0.0], \
        "an undecoded head must not read as an intro miss"
    assert scored.no_intent_sec == pytest.approx(1.0)
    assert scored.macro_f1 == 1.0, "the decoder was right about everything it said"


def test_undecoded_beats_are_located_relative_to_the_decoded_ones():
    scored = score_predicted(
        track(steady(6), ["intro"] * 6),
        "v1", (UNDECODED, "intro", "intro", "intro", UNDECODED, UNDECODED))
    assert (scored.no_intent_leading, scored.no_intent_interior,
            scored.no_intent_trailing) == (1, 0, 2)


def test_undecoded_time_still_counts_as_show_time():
    # Flicker is per audience-minute: the room is still there during the head.
    scored = score_predicted(track(steady(4), ["drop"] * 4),
                             "v1", (UNDECODED, UNDECODED, "drop", "drop"))
    assert scored.exposure_sec == pytest.approx(2.0)
    assert scored.scored_sec == pytest.approx(1.0)


def test_an_undecoded_gap_is_not_reported_as_a_state_change():
    scored = score_predicted(track(steady(6), ["drop"] * 6),
                             "v1", ("drop", "drop", UNDECODED, UNDECODED,
                                    "drop", "drop"))
    assert scored.boundary["class"][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"] == 0
    assert scored.no_intent_interior == 2


# --------------------------------------------------------------------------- #
# The claim map
# --------------------------------------------------------------------------- #


def test_identity_claims_map_every_class_to_exactly_itself():
    claims = identity_claims("v1")
    assert claims == {label: (label,) for label in SPACES["v1"].labels}


def test_identity_claims_do_not_forgive_intro_predicted_over_outro():
    scored = score_predicted(track(steady(4), ["outro"] * 4), "v1", ("intro",) * 4)
    assert scored.counts["outro"][2] > 0, "a network can know where it is in a track"
    assert scored.counts["intro"][1] > 0
    assert scored.f1("outro") == 0.0


def test_rule_equivalent_claims_forgive_it_the_way_atmospheric_is_forgiven():
    scored = score_predicted(track(steady(4), ["outro"] * 4), "v1", ("intro",) * 4,
                             claims=rule_equivalent_claims("v1"))
    assert scored.counts["outro"][0] > 0
    assert scored.f1("outro") == 1.0


def test_an_ambiguous_claims_false_positive_is_split_not_duplicated():
    scored = score_predicted(track(steady(4), ["drop"] * 4), "v1", ("intro",) * 4,
                             claims=rule_equivalent_claims("v1"))
    assert scored.counts["intro"][1] == pytest.approx(scored.counts["outro"][1])
    assert (scored.counts["intro"][1] + scored.counts["outro"][1]
            == pytest.approx(scored.scored_sec))


# --------------------------------------------------------------------------- #
# The two change streams
# --------------------------------------------------------------------------- #


def test_identity_claims_make_the_intent_and_class_streams_identical():
    # A model predicting in the label space emits a class stream by
    # construction; under identity claims there is nothing for the intent
    # stream to add, and the report says so rather than quoting two numbers.
    scored = score_predicted(track(steady(8), ["intro"] * 4 + ["outro"] * 4),
                             "v1", ("intro",) * 4 + ("outro",) * 4)
    for tolerance in TOLERANCES_SEC:
        assert (scored.boundary["intent"][tolerance]["overall"]
                == scored.boundary["class"][tolerance]["overall"])
        assert (scored.flicker["intent"][tolerance]
                == scored.flicker["class"][tolerance])


def test_rule_equivalent_claims_hide_an_intro_outro_switch_from_the_class_stream():
    scored = score_predicted(track(steady(8), ["intro"] * 4 + ["outro"] * 4),
                             "v1", ("intro",) * 4 + ("outro",) * 4,
                             claims=rule_equivalent_claims("v1"))
    assert scored.boundary["intent"][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"] == 1
    assert scored.boundary["class"][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"] == 0


def test_a_change_far_from_any_boundary_counts_as_flicker():
    labels = ["drop"] * 12
    predicted = ["drop"] * 5 + ["breakdown"] + ["drop"] * 6
    scored = score_predicted(track(steady(12, step=1.0), labels), "v1", predicted)
    assert scored.flicker["class"][PRIMARY_TOLERANCE_SEC] == 2, \
        "one spurious bar is a change out and a change back"


# --------------------------------------------------------------------------- #
# Compatibility with the rule baseline's own accumulators
# --------------------------------------------------------------------------- #


def test_scores_aggregate_with_the_rule_baselines_aggregator():
    first = score_predicted(track(steady(4), ["drop"] * 4, track_id="a"),
                            "v1", ("drop",) * 4)
    second = score_predicted(track(steady(4), ["drop"] * 4, track_id="b"),
                             "v1", ("intro",) * 4)
    total = aggregate([first, second])
    assert total.tracks == 2
    assert total.counts["drop"][0] == pytest.approx(first.counts["drop"][0])
    assert total.counts["drop"][2] == pytest.approx(second.counts["drop"][2])


def test_a_perfect_prediction_scores_one_and_a_constant_one_does_not():
    labels = ["intro"] * 4 + ["drop"] * 4
    perfect = score_predicted(track(steady(8), labels), "v1", tuple(labels))
    lazy = score_predicted(track(steady(8), labels), "v1", ("drop",) * 8)
    assert perfect.macro_f1 == pytest.approx(1.0)
    assert lazy.macro_f1 < perfect.macro_f1


# --------------------------------------------------------------------------- #
# The sweep's fast path is the decoder the runtime runs
# --------------------------------------------------------------------------- #


def inputs_from(npz, beats, params, labels=("drop",), intents=(NO_INTENT,),
                times=(0.0,)):
    edges = bar_grid(beats)
    posteriors, boundary = bar_observations(
        npz, edges, min_coverage=params.min_coverage,
        boundary_tolerance_sec=params.boundary_tolerance_sec)
    return TrackInputs(
        track_id="t", youtube_id="t", edges=edges, posteriors=posteriors,
        boundary=boundary, times=tuple(times),
        labels={space: tuple(labels) for space in SPACES}, intents=tuple(intents))


def test_the_cached_fast_path_decodes_exactly_like_decode_track(tmp_path):
    npz, beats = tmp_path / "t.npz", tmp_path / "t.beat.csv"
    synthetic_npz(npz, [BREAKDOWN] * 100 + [DROP] * 100, thin_frames=4)
    write_beat_csv(beats, bars=10, bar_sec=2.0, t0=0.0)

    priors = toy_priors(floor=3)
    params = DecodeParams(lag_bars=2, boundary_weight=0.0)
    reference = [label for _, label in decode_track(npz, beats, params, priors=priors)]

    inputs = inputs_from(npz, beats, params)
    assert list(decode_bars(inputs, build_decoder(priors, params))) == reference


def test_one_decoder_instance_carries_nothing_between_tracks(tmp_path):
    # Reusing the instance is what makes a sweep cost seconds; if any state
    # leaked, every config's numbers would depend on track order.
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    beats = tmp_path / "t.beat.csv"
    synthetic_npz(first, [DROP] * 200, thin_frames=4)
    synthetic_npz(second, [BREAKDOWN] * 100 + [DROP] * 100, thin_frames=4)
    write_beat_csv(beats, bars=10, bar_sec=2.0, t0=0.0)

    priors = toy_priors(floor=3)
    params = DecodeParams(lag_bars=2, boundary_weight=0.0)
    a_inputs = inputs_from(first, beats, params)
    b_inputs = inputs_from(second, beats, params)

    shared = build_decoder(priors, params)
    decode_bars(a_inputs, shared)
    after = decode_bars(b_inputs, shared)
    assert after == decode_bars(b_inputs, build_decoder(priors, params))


def test_decode_beats_places_the_bar_decisions_on_the_beat_grid(tmp_path):
    npz, beats = tmp_path / "t.npz", tmp_path / "t.beat.csv"
    synthetic_npz(npz, [DROP] * 200, thin_frames=4)
    write_beat_csv(beats, bars=10, bar_sec=2.0, t0=2.0)

    priors = toy_priors(floor=3)
    params = DecodeParams(lag_bars=2, boundary_weight=0.0)
    # Beats at 0.0 and 1.0 sit before the first downbeat at 2.0.
    inputs = inputs_from(npz, beats, params, labels=("drop",) * 4,
                         intents=(NO_INTENT,) * 4, times=(0.0, 1.0, 3.0, 5.0))
    got = decode_beats(inputs, build_decoder(priors, params))
    assert got[:2] == (UNDECODED, UNDECODED)
    assert set(got[2:]) == {"drop"}


def test_a_missing_track_fails_loud_instead_of_shrinking_the_split(tmp_path):
    # A missing sidecar drops out of BOTH columns at once, so they stay
    # comparable to each other while quietly ceasing to cover the split the
    # header names.  That is the failure mode worth a raise.
    table = tmp_path / "empty.csv.gz"
    with gzip.open(table, "wt", encoding="utf-8", newline="") as handle:
        handle.write("track_id,youtube_id,t_song,intent_at_beat,"
                     "label_canonical,label_v1\n")
    with pytest.raises(RuntimeError, match="missing inputs"):
        load_inputs(tmp_path, ["nosuchtrack"], table_path=table)
    kept, skipped = load_inputs(tmp_path, ["nosuchtrack"], table_path=table,
                                allow_missing=True)
    assert kept == [] and len(skipped) == 1, "the override still records the drop"


def test_per_track_head_to_head_separates_the_two_readings():
    # The full reading hands the rule a structural zero on classes the
    # beat-indexed table stops it claiming, so only the restricted reading can
    # go negative -- the artifact has to carry both or a universality claim
    # gets read off the wrong one.  This fixture is that exact shape: the NN
    # wins overall because it can name `intro`, and loses on `drop` itself.
    labels = ["intro"] * 2 + ["drop"] * 6
    nn = [score_predicted(track(steady(8), labels, track_id="a"), "v1",
                          ("intro",) * 2 + ("drop",) * 3 + ("breakdown",) * 3)]
    rule = [score_predicted(track(steady(8), labels, track_id="a"), "v1",
                            ("drop",) * 8)]
    rows = _per_track_deltas(nn, rule, ["drop"])
    assert rows[0]["delta"] > 0, "over all classes the NN wins -- it names intro"
    assert rows[0]["restricted_delta"] < 0, \
        "on `drop` alone the always-drop stream has the better F1"
    assert _head_to_head(rows, "delta")["nn_better"] == 1
    assert _head_to_head(rows, "restricted_delta")["rule_better"] == 1


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def test_enumerate_configs_is_a_deterministic_joint_grid():
    axes = {"prior_strength": (-1.0, 0.0), "drop_miss_cost": (1.0, 2.0, 4.0)}
    configs = enumerate_configs(DecodeParams(), axes)
    assert [(c.prior_strength, c.drop_miss_cost) for c in configs] == [
        (-1.0, 1.0), (-1.0, 2.0), (-1.0, 4.0),
        (0.0, 1.0), (0.0, 2.0), (0.0, 4.0),
    ], "a joint grid in declared axis order, not two line searches"
    assert configs == enumerate_configs(DecodeParams(), axes)


def test_enumerate_configs_keeps_the_untouched_knobs_at_the_base():
    base = DecodeParams(lag_bars=5, floor_scale=2.0)
    configs = enumerate_configs(base, {"prior_strength": (-0.5,)})
    assert configs[0] == DecodeParams(lag_bars=5, floor_scale=2.0, prior_strength=-0.5)


def test_enumerate_configs_rejects_an_unknown_knob():
    with pytest.raises(TypeError):
        enumerate_configs(DecodeParams(), {"stickiness": (1.0,)})


def result(macro_f1, flicker, **params):
    return {"params": DecodeParams(**params), "macro_f1": macro_f1,
            "flicker_per_min": flicker}


def test_selection_takes_the_best_macro_f1_that_clears_the_flicker_ceiling():
    chosen = select_config(
        [result(0.90, 5.0, lag_bars=1),
         result(0.80, 2.0, lag_bars=2),
         result(0.70, 1.0, lag_bars=3)],
        flicker_ceiling=3.0)
    assert chosen["macro_f1"] == 0.80, \
        "the higher-F1 config changes the lights more often than the baseline"


def test_selection_breaks_ties_on_flicker_then_enumeration_order():
    chosen = select_config([result(0.8, 2.0, lag_bars=1),
                            result(0.8, 1.0, lag_bars=2),
                            result(0.8, 1.0, lag_bars=3)],
                           flicker_ceiling=3.0)
    assert chosen["params"].lag_bars == 2


def test_selection_refuses_rather_than_returning_a_config_that_flickers():
    with pytest.raises(RuntimeError, match="flicker"):
        select_config([result(0.9, 5.0)], flicker_ceiling=3.0)


def test_selection_will_not_choose_a_config_outside_the_latency_budget():
    # The lag curve is a deliverable, so long lags are measured; a config that
    # needs more future audio than the show has still cannot be shipped.
    chosen = select_config([result(0.99, 1.0, lag_bars=8),
                            result(0.50, 1.0, lag_bars=2)],
                           flicker_ceiling=3.0, budget_bars=3)
    assert chosen["params"].lag_bars == 2


def test_sensitivity_reports_the_best_per_value_curve_and_its_spread():
    rows = [result(0.60, 1.0, prior_strength=0.0),
            result(0.55, 1.0, prior_strength=0.0),      # worse at the same value
            result(0.70, 1.0, prior_strength=-0.5)]
    for row in rows:
        row["params"] = dataclasses.asdict(row["params"])
        row.update(drop_recall=0.0, accuracy=0.0)
    report = sensitivity(rows, "prior_strength")
    assert report["values"]["0.0"]["macro_f1"] == 0.60, "best at that value, not last"
    assert report["best_value"] == -0.5
    assert report["spread"] == pytest.approx(0.10)


def test_refinement_reopens_the_trellis_axes_but_not_the_latency_policy():
    # A staged search can only find a coordinate-wise optimum; the refinement
    # has to move the axes together to prove the winner is not one.  lag_bars
    # is excluded on purpose -- it is bounded by the look-ahead budget, not by
    # macro-F1, so it gets its own stage and curve.
    axes = refinement_axes(DecodeParams(prior_strength=-0.5, drop_miss_cost=1.0,
                                        boundary_weight=1.0, boundary_ref=0.3,
                                        floor_scale=1.0))
    assert set(axes) == {"prior_strength", "drop_miss_cost", "boundary_weight",
                         "boundary_ref", "floor_scale"}
    assert "lag_bars" not in axes
    assert -0.5 in axes["prior_strength"] and len(axes["prior_strength"]) > 1
    assert 1.0 in axes["drop_miss_cost"], "an endpoint winner keeps its own value"
    grid = enumerate_configs(DecodeParams(), axes)
    expected = 1
    for values in axes.values():
        expected *= len(values)
    assert len(grid) == expected, "a joint product, not a union of line searches"
