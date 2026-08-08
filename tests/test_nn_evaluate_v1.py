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
    INTENT_TO_LABELS,
    LABEL_COLUMN,
    PRIMARY_SPACE,
    PRIMARY_TOLERANCE_SEC,
    RAW9,
    SPACES,
    TOLERANCES_SEC,
    TrackBeats,
    aggregate,
)
from nn.decoder import (  # noqa: E402
    SHIPPING_DECODER_CONFIG,
    DecodeParams,
    bar_grid,
    bar_observations,
    decode_track,
    load_decoder_config,
)
from nn.evaluate_v1 import (  # noqa: E402
    DEFAULT_SPACE,
    EVAL_FILE,
    UNDECODED,
    TrackInputs,
    _head_to_head,
    _per_track_deltas,
    beat_classes,
    build_decoder,
    decode_bars,
    decode_beats,
    default_output_name,
    identity_claims,
    load_inputs,
    read_ids_file,
    render,
    rule_equivalent_claims,
    score_predicted,
    sidecar_model_sha,
)
from nn.sweep import (  # noqa: E402
    enumerate_configs,
    refinement_axes,
    select_config,
    sensitivity,
)
from tests.test_nn_decoder import (  # noqa: E402
    graded_bars,
    rows_npz,
    synthetic_npz,
    toy_priors,
    write_beat_csv,
)

BREAKDOWN, DROP = 2, 3

SPACE = DEFAULT_SPACE


def steady(count, step=0.5, t0=0.0):
    return tuple(t0 + index * step for index in range(count))


def track(times, labels, intents=None, track_id="t"):
    labels = tuple(labels)
    return TrackBeats(
        track_id=track_id,
        times=tuple(times),
        intents=tuple(intents if intents is not None else [NO_INTENT] * len(labels)),
        labels={name: tuple(spec.view(label) for label in labels)
                for name, spec in SPACES.items()},
    )


def test_the_default_space_is_the_primary_raw_vocabulary():
    assert DEFAULT_SPACE == PRIMARY_SPACE == RAW9


def test_beat_classes_reads_each_beat_off_the_bar_that_contains_it():
    got = beat_classes(steady(8), (0.0, 2.0, 4.0), ["intro", "drop"])
    assert got == ("intro",) * 4 + ("drop",) * 4


def test_beat_classes_leaves_the_pre_downbeat_head_undecoded():
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


def test_undecoded_beats_are_excluded_from_every_class_count():
    scored = score_predicted(
        track(steady(6), ["intro"] * 2 + ["drop"] * 4),
        SPACE, (UNDECODED, UNDECODED, "drop", "drop", "drop", "drop"))
    assert scored.counts["intro"] == [0.0, 0.0, 0.0], \
        "an undecoded head must not read as an intro miss"
    assert scored.no_intent_sec == pytest.approx(1.0)
    assert scored.macro_f1 == 1.0, "the decoder was right about everything it said"


def test_undecoded_beats_are_located_relative_to_the_decoded_ones():
    scored = score_predicted(
        track(steady(6), ["intro"] * 6),
        SPACE, (UNDECODED, "intro", "intro", "intro", UNDECODED, UNDECODED))
    assert (scored.no_intent_leading, scored.no_intent_interior,
            scored.no_intent_trailing) == (1, 0, 2)


def test_undecoded_time_still_counts_as_show_time():
    scored = score_predicted(track(steady(4), ["drop"] * 4),
                             SPACE, (UNDECODED, UNDECODED, "drop", "drop"))
    assert scored.exposure_sec == pytest.approx(2.0)
    assert scored.scored_sec == pytest.approx(1.0)


def test_an_undecoded_gap_is_not_reported_as_a_state_change():
    scored = score_predicted(track(steady(6), ["drop"] * 6),
                             SPACE, ("drop", "drop", UNDECODED, UNDECODED,
                                    "drop", "drop"))
    assert scored.boundary["class"][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"] == 0
    assert scored.no_intent_interior == 2


def test_identity_claims_map_every_class_to_exactly_itself():
    claims = identity_claims(SPACE)
    assert claims == {label: (label,) for label in SPACES[SPACE].labels}


def test_identity_claims_do_not_forgive_intro_predicted_over_outro():
    scored = score_predicted(track(steady(4), ["outro"] * 4), SPACE, ("intro",) * 4)
    assert scored.counts["outro"][2] > 0, "a network can know where it is in a track"
    assert scored.counts["intro"][1] > 0
    assert scored.f1("outro") == 0.0


def test_rule_equivalent_claims_forgive_it_the_way_atmospheric_is_forgiven():
    scored = score_predicted(track(steady(4), ["outro"] * 4), SPACE, ("intro",) * 4,
                             claims=rule_equivalent_claims(SPACE))
    assert scored.counts["outro"][0] > 0
    assert scored.f1("outro") == 1.0


def test_an_ambiguous_claims_false_positive_is_split_not_duplicated():
    scored = score_predicted(track(steady(4), ["drop"] * 4), SPACE, ("intro",) * 4,
                             claims=rule_equivalent_claims(SPACE))
    quiet = INTENT_TO_LABELS["atmospheric"][SPACE]
    for label in quiet:
        assert scored.counts[label][1] == pytest.approx(
            scored.counts[quiet[0]][1]), label
    assert (sum(scored.counts[label][1] for label in quiet)
            == pytest.approx(scored.scored_sec))


def test_identity_claims_make_the_intent_and_class_streams_identical():
    scored = score_predicted(track(steady(8), ["intro"] * 4 + ["outro"] * 4),
                             SPACE, ("intro",) * 4 + ("outro",) * 4)
    for tolerance in TOLERANCES_SEC:
        assert (scored.boundary["intent"][tolerance]["overall"]
                == scored.boundary["class"][tolerance]["overall"])
        assert (scored.flicker["intent"][tolerance]
                == scored.flicker["class"][tolerance])


def test_rule_equivalent_claims_hide_an_intro_outro_switch_from_the_class_stream():
    scored = score_predicted(track(steady(8), ["intro"] * 4 + ["outro"] * 4),
                             SPACE, ("intro",) * 4 + ("outro",) * 4,
                             claims=rule_equivalent_claims(SPACE))
    assert scored.boundary["intent"][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"] == 1
    assert scored.boundary["class"][PRIMARY_TOLERANCE_SEC]["overall"]["n_pred"] == 0


def test_a_change_far_from_any_boundary_counts_as_flicker():
    labels = ["drop"] * 12
    predicted = ["drop"] * 5 + ["breakdown"] + ["drop"] * 6
    scored = score_predicted(track(steady(12, step=1.0), labels), SPACE, predicted)
    assert scored.flicker["class"][PRIMARY_TOLERANCE_SEC] == 2, \
        "one spurious bar is a change out and a change back"


def test_scores_aggregate_with_the_rule_baselines_aggregator():
    first = score_predicted(track(steady(4), ["drop"] * 4, track_id="a"),
                            SPACE, ("drop",) * 4)
    second = score_predicted(track(steady(4), ["drop"] * 4, track_id="b"),
                             SPACE, ("intro",) * 4)
    total = aggregate([first, second])
    assert total.tracks == 2
    assert total.counts["drop"][0] == pytest.approx(first.counts["drop"][0])
    assert total.counts["drop"][2] == pytest.approx(second.counts["drop"][2])


def test_a_perfect_prediction_scores_one_and_a_constant_one_does_not():
    labels = ["intro"] * 4 + ["drop"] * 4
    perfect = score_predicted(track(steady(8), labels), SPACE, tuple(labels))
    lazy = score_predicted(track(steady(8), labels), SPACE, ("drop",) * 8)
    assert perfect.macro_f1 == pytest.approx(1.0)
    assert lazy.macro_f1 < perfect.macro_f1


def inputs_from(npz, beats, params, labels=("drop",), intents=(NO_INTENT,),
                times=(0.0,)):
    edges = bar_grid(beats)
    posteriors, boundary = bar_observations(
        npz, edges, min_coverage=params.min_coverage,
        boundary_tolerance_sec=params.boundary_tolerance_sec,
        temperature=params.temperature)
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


def test_the_cached_fast_path_decodes_like_decode_track_at_the_shipped_knobs(tmp_path):
    npz, beats = tmp_path / "t.npz", tmp_path / "t.beat.csv"
    rows_npz(npz, graded_bars(24), frame_sec=0.25, label_pool=2, label_t0=0.5)
    write_beat_csv(beats, bars=24, bar_sec=2.0, t0=0.5)

    priors = toy_priors(floor=3)
    shipped = load_decoder_config(SHIPPING_DECODER_CONFIG)
    for params in (shipped, dataclasses.replace(shipped, temperature=8.0)):
        reference = [label for _, label
                     in decode_track(npz, beats, params, priors=priors)]
        inputs = inputs_from(npz, beats, params)
        assert list(decode_bars(inputs, build_decoder(priors, params))) == reference


def test_the_two_paths_disagree_when_one_of_them_drops_a_knob(tmp_path):
    npz, beats = tmp_path / "t.npz", tmp_path / "t.beat.csv"
    rows_npz(npz, graded_bars(24), frame_sec=0.25, label_pool=2, label_t0=0.5)
    write_beat_csv(beats, bars=24, bar_sec=2.0, t0=0.5)

    priors = toy_priors(floor=3)
    # Neutralised around the shipped config: the l9 drop bonus saturates this
    # toy decode to one class under any temperature, which would hide a
    # dropped knob -- the very thing this test exists to catch.
    base = dataclasses.replace(load_decoder_config(SHIPPING_DECODER_CONFIG),
                               drop_miss_cost=1.0, prior_strength=0.0)
    hot = dataclasses.replace(base, temperature=8.0)
    assert (inputs_from(npz, beats, hot).posteriors.tolist()
            != inputs_from(npz, beats, base).posteriors.tolist())
    assert ([label for _, label in decode_track(npz, beats, hot, priors=priors)]
            != [label for _, label in decode_track(npz, beats, base, priors=priors)])


def test_one_decoder_instance_carries_nothing_between_tracks(tmp_path):
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
    inputs = inputs_from(npz, beats, params, labels=("drop",) * 4,
                         intents=(NO_INTENT,) * 4, times=(0.0, 1.0, 3.0, 5.0))
    got = decode_beats(inputs, build_decoder(priors, params))
    assert got[:2] == (UNDECODED, UNDECODED)
    assert set(got[2:]) == {"drop"}


def test_a_missing_track_fails_loud_instead_of_shrinking_the_split(tmp_path):
    table = tmp_path / "empty.csv.gz"
    with gzip.open(table, "wt", encoding="utf-8", newline="") as handle:
        handle.write(f"track_id,youtube_id,t_song,intent_at_beat,{LABEL_COLUMN}\n")
    with pytest.raises(RuntimeError, match="missing inputs"):
        load_inputs(tmp_path, ["nosuchtrack"], table_path=table)
    kept, skipped = load_inputs(tmp_path, ["nosuchtrack"], table_path=table,
                                allow_missing=True)
    assert kept == [] and len(skipped) == 1, "the override still records the drop"


def loadable_track(data_dir, youtube_id="abc", model_sha=None):
    track_id = f"0001.{youtube_id}"
    table = data_dir / "table.csv.gz"
    with gzip.open(table, "wt", encoding="utf-8", newline="") as handle:
        handle.write(f"track_id,youtube_id,t_song,intent_at_beat,{LABEL_COLUMN}\n")
        for index in range(8):
            handle.write(f"{track_id},{youtube_id},{index * 2.0},drop,drop\n")

    beats = data_dir / "annotations" / "beats"
    beats.mkdir(parents=True, exist_ok=True)
    write_beat_csv(beats / f"{track_id}.beat.csv", bars=10, bar_sec=2.0, t0=0.0)

    posteriors = data_dir / "posteriors"
    posteriors.mkdir(parents=True, exist_ok=True)
    sidecar = posteriors / f"{youtube_id}.npz"
    synthetic_npz(sidecar, [DROP] * 200, thin_frames=4)
    if model_sha is not None:
        with np.load(sidecar) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["model_sha"] = np.str_(model_sha)
        np.savez(sidecar, **arrays)
    return table, sidecar


def test_load_inputs_refuses_sidecars_written_by_a_different_model(tmp_path):
    table, _sidecar = loadable_track(tmp_path, model_sha="a" * 64)

    kept, _skipped = load_inputs(tmp_path, ["abc"], table_path=table,
                                 model_sha="a" * 64)
    assert len(kept) == 1, "the matching sha must load"

    with pytest.raises(RuntimeError, match="different model"):
        load_inputs(tmp_path, ["abc"], table_path=table, model_sha="b" * 64)


def test_a_wrong_model_sidecar_is_never_downgraded_to_a_skip(tmp_path):
    table, _sidecar = loadable_track(tmp_path, model_sha="a" * 64)
    with pytest.raises(RuntimeError, match="different model"):
        load_inputs(tmp_path, ["abc"], table_path=table, model_sha="b" * 64,
                    allow_missing=True)


def test_an_unstamped_sidecar_reads_as_unknown_rather_than_raising(tmp_path):
    table, sidecar = loadable_track(tmp_path)
    assert sidecar_model_sha(sidecar) is None
    with pytest.raises(RuntimeError, match="unstamped"):
        load_inputs(tmp_path, ["abc"], table_path=table, model_sha="a" * 64)
    kept, _ = load_inputs(tmp_path, ["abc"], table_path=table)
    assert len(kept) == 1, "no expected sha means no identity claim to check"


def test_read_ids_file_accepts_both_shapes_and_refuses_ambiguity(tmp_path):
    array = tmp_path / "a.json"
    array.write_text('["one", "two"]', encoding="utf-8")
    assert read_ids_file(array) == ["one", "two"]

    wrapped = tmp_path / "w.json"
    wrapped.write_text('{"ids": ["one", "two"]}', encoding="utf-8")
    assert read_ids_file(wrapped) == ["one", "two"]

    lines = tmp_path / "l.txt"
    lines.write_text("# why this list exists\none\n\ntwo  # trailing note\n",
                     encoding="utf-8")
    assert read_ids_file(lines) == ["one", "two"]

    dupes = tmp_path / "d.json"
    dupes.write_text('["one", "two", "one"]', encoding="utf-8")
    with pytest.raises(RuntimeError, match="repeats"):
        read_ids_file(dupes)

    empty = tmp_path / "e.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no ids"):
        read_ids_file(empty)


def test_a_subset_run_cannot_overwrite_the_splits_published_verdict(tmp_path):
    published = default_output_name("test")

    assert published == EVAL_FILE.format(split="test")
    assert default_output_name("test", tmp_path / "v1subset101.txt") != published


def test_two_different_id_lists_do_not_land_on_each_other(tmp_path):
    assert (default_output_name("test", tmp_path / "subset_a.txt")
            != default_output_name("test", tmp_path / "subset_b.txt"))


def test_per_track_head_to_head_separates_the_two_readings():
    labels = ["intro"] * 2 + ["drop"] * 6
    nn = [score_predicted(track(steady(8), labels, track_id="a"), SPACE,
                          ("intro",) * 2 + ("drop",) * 3 + ("breakdown",) * 3)]
    rule = [score_predicted(track(steady(8), labels, track_id="a"), SPACE,
                            ("drop",) * 8)]
    rows = _per_track_deltas(nn, rule, ["drop"])
    assert rows[0]["delta"] > 0, "over all classes the NN wins -- it names intro"
    assert rows[0]["restricted_delta"] < 0, \
        "on `drop` alone the always-drop stream has the better F1"
    assert _head_to_head(rows, "delta")["nn_better"] == 1
    assert _head_to_head(rows, "restricted_delta")["rule_better"] == 1


def _side_block(space=SPACE) -> dict:
    tolerances = {f"{tolerance}": {"f1": 0.5, "to_drop": {"f1": 0.5}}
                  for tolerance in TOLERANCES_SEC}
    return {
        "macro_f1": 0.5, "accuracy": 0.5, "undecoded_share": 0.1, "changes": 7,
        "per_class_f1": {label: 0.5 for label in SPACES[space].labels},
        "drop": {"recall": 0.5, "precision": 0.5},
        "boundary": tolerances,
        "flicker_per_audience_minute": {f"{t}": 0.5 for t in TOLERANCES_SEC},
    }


def _head(count=1) -> dict:
    return {"tracks": count, "nn_better": 1, "rule_better": 0, "tied": 0,
            "min_delta": 0.0, "median_delta": 0.0, "max_delta": 0.0}


def minimal_report(space=SPACE) -> dict:
    side = {"nn": _side_block(space), "rule": _side_block(space)}
    return {
        "split": "val", "tracks": 1, "space": space,
        "primary": side,
        "rule_equivalent_claims": side,
        "rule_intent_stream": _side_block(space),
        "nn_streams_identical": True,
        "expressible_comparison": {
            "classes": ["drop"], "unreachable_for_rule": ["intro"],
            "nn_macro_f1": 0.5, "rule_macro_f1": 0.4, "delta": 0.1},
        "rule_structural": {"macro_f1_best_achievable": 0.4,
                            "macro_f1_upper_bound": 0.6},
        "head_to_head": {"full": _head(), "restricted": _head()},
    }


def test_the_report_names_the_class_count_from_the_vocabulary_not_a_literal():
    """'all 5 classes' was hardcoded; the vocabulary is nine classes now."""
    text = render(minimal_report())
    assert f"all {len(SPACES[SPACE].labels)} classes" in text
    assert "all 5 classes" not in text


def test_the_report_names_the_space_it_scored_rather_than_a_baked_in_one():
    assert f"macro-F1 ({SPACE})" in render(minimal_report())


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
    chosen = select_config([result(0.99, 1.0, lag_bars=8),
                            result(0.50, 1.0, lag_bars=2)],
                           flicker_ceiling=3.0, budget_bars=3)
    assert chosen["params"].lag_bars == 2


def test_sensitivity_reports_the_best_per_value_curve_and_its_spread():
    rows = [result(0.60, 1.0, prior_strength=0.0),
            result(0.55, 1.0, prior_strength=0.0),
            result(0.70, 1.0, prior_strength=-0.5)]
    for row in rows:
        row["params"] = dataclasses.asdict(row["params"])
        row.update(drop_recall=0.0, accuracy=0.0)
    report = sensitivity(rows, "prior_strength")
    assert report["values"]["0.0"]["macro_f1"] == 0.60, "best at that value, not last"
    assert report["best_value"] == -0.5
    assert report["spread"] == pytest.approx(0.10)


def test_refinement_reopens_the_trellis_axes_but_not_the_latency_policy():
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
