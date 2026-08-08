"""Tests for the label-aligned evaluator (training/evaluate_against_labels.py).

The evaluator produces the numbers the owner will use to decide whether to
replace the rule classifier with a model.  Every metric here is a claim about
the current system's musical competence, so the arithmetic is pinned on
synthetic timelines where the right answer is obvious by inspection:

* a perfect timeline must score 1.0 everywhere,
* a degenerate constant-intent timeline must score exactly the degenerate
  values (not something merely "low"),
* the tolerance edges of boundary matching must be exact, not approximately
  right, because the whole boundary-F1 table is read at three tolerances.

No corpus, no audio, no simulation.
"""
import gzip
import math
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import build_training_table  # noqa: E402
from build_training_table import TABLE_HEADER  # noqa: E402
from evaluate_against_labels import (  # noqa: E402
    INTENT_ORDER,
    INTENT_TO_LABELS,
    LABEL_COLUMN,
    LEGACY_V1,
    MAX_GAP_FACTOR,
    PRIMARY_SPACE,
    RAW9,
    SPACES,
    STREAM_ORDER,
    TOLERANCES_SEC,
    TrackBeats,
    aggregate,
    beat_weights,
    best_achievable_macro_f1,
    class_changes,
    evaluate,
    flicker_instants,
    fold_to_legacy_v1,
    intent_changes,
    label_boundaries,
    load_tracks,
    match_events,
    prf,
    render_report,
    score_track,
    typed_predictions,
)
from lib.engine.effect_definitions import (  # noqa: E402
    SECTION_CLASS_INTENTS,
    LightIntent,
)
from lib.label_space import SECTION_LABELS  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def track(times, intents, labels, track_id="t"):
    """A synthetic track carrying RAW labels; each space's view derives its own."""
    labels = tuple(labels)
    return TrackBeats(
        track_id=track_id,
        times=tuple(float(t) for t in times),
        intents=tuple(intents),
        labels={name: tuple(spec.view(label) for label in labels)
                for name, spec in SPACES.items()},
    )


def steady(n, start=0.0, step=0.5):
    return [start + i * step for i in range(n)]


def f1_of(score, label):
    return prf(*score.counts[label])[2]


def empty_buckets(space=RAW9):
    return {label: [] for label in SPACES[space].labels}


# --------------------------------------------------------------------------- #
# the label spaces -- one column, two views
# --------------------------------------------------------------------------- #


def test_exactly_two_spaces_exist_and_the_raw_nine_is_primary():
    assert set(SPACES) == {RAW9, LEGACY_V1}
    assert PRIMARY_SPACE == RAW9
    assert SPACES[RAW9].labels == SECTION_LABELS


def test_the_legacy_space_is_the_retired_five_class_vocabulary():
    assert SPACES[LEGACY_V1].labels == ("intro", "buildup", "breakdown",
                                        "drop", "outro")


def test_a_space_is_a_view_over_the_one_label_column_not_a_column_of_its_own():
    """There is one label column now, so a space cannot be keyed on a column."""
    assert LABEL_COLUMN in TABLE_HEADER
    assert [column for column in TABLE_HEADER
            if column.startswith("label")] == [LABEL_COLUMN]
    for spec in SPACES.values():
        assert not hasattr(spec, "column")
        assert callable(spec.view)


def test_the_raw_space_view_is_the_identity_on_every_label():
    for label in SECTION_LABELS:
        assert SPACES[RAW9].view(label) == label


def test_the_legacy_view_maps_every_one_of_the_nine_labels_into_its_own_space():
    expected = {
        "intro": "intro",
        "altintro": "intro",
        "buildup": "buildup",
        "breakdown": "breakdown",
        "bridge": "breakdown",
        "drop": "drop",
        "cooldown": "breakdown",
        "outro": "outro",
        "altoutro": "outro",
    }
    assert {label: fold_to_legacy_v1(label) for label in SECTION_LABELS} == expected
    assert set(expected.values()) == set(SPACES[LEGACY_V1].labels)


def test_the_legacy_view_reproduces_the_retired_canonical_then_v1_chain_exactly():
    """Score-neutrality of the migration: the banked record stays comparable.

    The retired pipeline folded twice -- ``end`` dropped, then
    ``altintro``/``bridge`` into the canonical-7, then ``cooldown``/``altoutro``
    into label_v1-5.  Composing those two steps by hand here is the whole
    evidence that the surviving one-step view is the same function.
    """
    retired_canonical_fold = {"altintro": "intro", "bridge": "breakdown"}
    retired_v1_fold = {"cooldown": "breakdown", "altoutro": "outro"}

    for label in SECTION_LABELS:
        canonical = retired_canonical_fold.get(label, label)
        assert fold_to_legacy_v1(label) == retired_v1_fold.get(canonical, canonical)


def test_the_legacy_fold_is_a_pure_function_nothing_downstream_stores():
    """The fold survives ONLY here, applied at evaluation time.

    A stage that persisted its output would put a second label space back into
    the data, which is exactly what ruling #264 retired.
    """
    assert fold_to_legacy_v1("cooldown") == fold_to_legacy_v1("cooldown")
    for name in ("label_v1", "V1_ORDER", "V1_MAP", "CANONICAL_ORDER",
                 "canonical_coverage"):
        assert not hasattr(build_training_table, name), name


def test_an_unknown_label_passes_through_the_legacy_view_rather_than_being_invented():
    assert fold_to_legacy_v1("chorus") == "chorus"


# --------------------------------------------------------------------------- #
# the mapping dict
# --------------------------------------------------------------------------- #


def test_every_intent_a_section_class_can_produce_has_a_mapping_in_every_space():
    for intent in SECTION_CLASS_INTENTS.values():
        assert intent.value in INTENT_TO_LABELS, intent
        for space in SPACES:
            assert INTENT_TO_LABELS[intent.value][space], (intent.value, space)


def test_mapping_targets_are_real_labels_of_their_space():
    for intent, per_space in INTENT_TO_LABELS.items():
        for space, targets in per_space.items():
            for target in targets:
                assert target in SPACES[space].labels, (intent, space, target)


def test_the_raw_claim_map_is_the_inverse_of_the_shows_own_class_to_intent_map():
    """The evaluator's notion of 'correct' must BE the mapping the show uses.

    This fails loudly the moment the owner revises the provisional class ->
    intent assignments, which is the point: a claim map that drifted from the
    engine would score the show against lights it never plays.
    """
    inverse: dict = {}
    for label in SECTION_LABELS:
        inverse.setdefault(SECTION_CLASS_INTENTS[label].value, []).append(label)
    assert {intent: per_space[RAW9]
            for intent, per_space in INTENT_TO_LABELS.items()} == \
        {intent: tuple(labels) for intent, labels in inverse.items()}


def test_the_live_intent_alphabet_carries_no_retired_intent():
    assert "groove" not in INTENT_TO_LABELS and "peak" not in INTENT_TO_LABELS
    assert set(INTENT_ORDER) == set(INTENT_TO_LABELS)


def test_the_scored_alphabet_is_exactly_the_shows_own_alphabet():
    """The intent layer is a pure mapped image of the classes -- no engine-derived
    state like PEAK survives, so an intent the evaluator scores that the enum
    cannot commit (or the reverse) is a bug in one of the two."""
    assert set(INTENT_TO_LABELS) == {intent.value for intent in LightIntent}
    assert set(INTENT_TO_LABELS) == {intent.value
                                     for intent in SECTION_CLASS_INTENTS.values()}


def test_atmospheric_is_position_blind_and_matches_any_quiet_class():
    """An intent cannot know track position, so quiet == intro OR outro."""
    assert INTENT_TO_LABELS["atmospheric"][RAW9] == (
        "intro", "altintro", "outro", "altoutro")
    assert INTENT_TO_LABELS["atmospheric"][LEGACY_V1] == ("intro", "outro")


def test_breakdown_claims_the_three_stripped_classes_in_the_raw_space():
    assert INTENT_TO_LABELS["breakdown"][RAW9] == ("breakdown", "bridge", "cooldown")
    assert INTENT_TO_LABELS["breakdown"][LEGACY_V1] == ("breakdown",)


def test_the_legacy_claims_are_the_historical_ones():
    assert {intent: per_space[LEGACY_V1]
            for intent, per_space in INTENT_TO_LABELS.items()} == {
        "atmospheric": ("intro", "outro"),
        "breakdown": ("breakdown",),
        "buildup": ("buildup",),
        "drop": ("drop",),
    }


# --------------------------------------------------------------------------- #
# time weighting
# --------------------------------------------------------------------------- #


def test_each_beat_is_weighted_by_the_time_until_the_next_beat():
    weights, clamped = beat_weights([0.0, 0.5, 2.0])
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(1.5)
    assert clamped == 0.0


def test_the_last_beat_is_weighted_by_the_median_beat_interval():
    weights, _ = beat_weights([0.0, 0.5, 1.0, 1.5])
    assert weights[-1] == pytest.approx(0.5)


def test_a_long_beat_dropout_is_clamped_and_the_excess_reported():
    """Beats lost to a dropout must not smear one label over the whole gap."""
    times = [0.0, 0.5, 1.0, 1.5, 100.0]
    weights, clamped = beat_weights(times)
    cap = MAX_GAP_FACTOR * 0.5
    assert weights[3] == pytest.approx(cap)
    assert clamped == pytest.approx(98.5 - cap)


def test_a_single_beat_track_weighs_nothing_instead_of_crashing():
    weights, clamped = beat_weights([4.0])
    assert weights == [0.0]
    assert clamped == 0.0
    assert beat_weights([]) == ([], 0.0)


# --------------------------------------------------------------------------- #
# confusion / per-class F1
# --------------------------------------------------------------------------- #


def test_a_perfect_timeline_scores_one_everywhere():
    times = steady(8)
    labels = ["intro"] * 2 + ["buildup"] * 2 + ["drop"] * 2 + ["breakdown"] * 2
    intents = ["atmospheric"] * 2 + ["buildup"] * 2 + ["drop"] * 2 + ["breakdown"] * 2
    score = score_track(track(times, intents, labels), RAW9)
    assert score.accuracy == pytest.approx(1.0)
    assert score.macro_f1 == pytest.approx(1.0)
    for label in ("intro", "buildup", "drop", "breakdown"):
        assert f1_of(score, label) == pytest.approx(1.0)


def test_the_new_classes_are_scored_correct_in_the_raw_space():
    """altintro/bridge/cooldown/altoutro are real classes now, not folds."""
    times = steady(8)
    labels = ["altintro"] * 2 + ["bridge"] * 2 + ["cooldown"] * 2 + ["altoutro"] * 2
    intents = ["atmospheric"] * 2 + ["breakdown"] * 4 + ["atmospheric"] * 2
    score = score_track(track(times, intents, labels), RAW9)
    assert score.accuracy == pytest.approx(1.0)
    for label in ("altintro", "bridge", "cooldown", "altoutro"):
        assert f1_of(score, label) == pytest.approx(1.0)


def test_the_legacy_view_collapses_the_new_classes_onto_their_retired_homes():
    times = steady(8)
    labels = ["altintro"] * 2 + ["bridge"] * 2 + ["cooldown"] * 2 + ["altoutro"] * 2
    intents = ["atmospheric"] * 2 + ["breakdown"] * 4 + ["atmospheric"] * 2
    score = score_track(track(times, intents, labels), LEGACY_V1)
    assert set(score.macro_classes) == {"intro", "breakdown", "outro"}
    assert score.accuracy == pytest.approx(1.0)


def test_a_constant_drop_timeline_scores_the_exact_degenerate_values():
    """All-DROP lights: perfect drop recall, precision == the drop time share."""
    times = steady(4)                       # 0.5 s each, 2.0 s total
    labels = ["drop", "drop", "breakdown", "intro"]
    score = score_track(track(times, ["drop"] * 4, labels), RAW9)
    precision, recall, f1 = prf(*score.counts["drop"])
    assert recall == pytest.approx(1.0)
    assert precision == pytest.approx(0.5)
    assert f1 == pytest.approx(2 / 3)
    assert prf(*score.counts["breakdown"])[1] == pytest.approx(0.0)
    assert prf(*score.counts["intro"])[1] == pytest.approx(0.0)
    assert score.accuracy == pytest.approx(0.5)


def test_the_confusion_matrix_is_in_seconds_not_beats():
    """Uneven beats: a slow stretch must outweigh a fast one, beat counts equal."""
    times = [0.0, 0.25, 0.5, 1.5]           # 0.25, 0.25, 1.0 s
    labels = ["drop", "drop", "breakdown", "breakdown"]
    score = score_track(track(times, ["drop"] * 4, labels), RAW9)
    assert score.confusion["drop"]["drop"] == pytest.approx(0.5)
    assert score.confusion["drop"]["breakdown"] == pytest.approx(1.0 + 0.25)


def test_beats_with_no_committed_intent_are_counted_but_never_scored():
    times = steady(4)
    labels = ["drop"] * 4
    score = score_track(track(times, ["", "", "drop", "drop"], labels), RAW9)
    assert score.no_intent_sec == pytest.approx(1.0)
    assert score.no_intent_rows == 2
    assert sum(sum(row.values()) for row in score.confusion.values()) == pytest.approx(1.0)
    assert score.accuracy == pytest.approx(1.0)     # judged on predictions only


def test_unpredicted_beats_are_located_relative_to_the_committed_ones():
    """An INTERIOR gap silently bridges the change stream, so it must be visible."""
    times = steady(6)
    labels = ["drop"] * 6
    score = score_track(
        track(times, ["", "drop", "", "drop", "breakdown", ""], labels), RAW9)
    assert (score.no_intent_leading, score.no_intent_interior,
            score.no_intent_trailing) == (1, 1, 1)
    assert score.no_intent_rows == 3


def test_a_track_that_never_commits_counts_every_beat_as_leading():
    score = score_track(track(steady(3), ["", "", ""], ["drop"] * 3), RAW9)
    assert (score.no_intent_leading, score.no_intent_interior,
            score.no_intent_trailing) == (3, 0, 0)


def test_atmospheric_is_credited_against_intro_and_against_outro():
    intro = score_track(track(steady(2), ["atmospheric"] * 2, ["intro"] * 2), RAW9)
    outro = score_track(track(steady(2), ["atmospheric"] * 2, ["altoutro"] * 2), RAW9)
    assert intro.counts["intro"][0] > 0 and intro.counts["intro"][2] == 0
    assert outro.counts["altoutro"][0] > 0 and outro.counts["altoutro"][2] == 0
    assert intro.accuracy == pytest.approx(1.0)
    assert outro.accuracy == pytest.approx(1.0)


def test_a_wrong_atmospheric_splits_its_false_positive_across_the_quiet_classes():
    """No single class may absorb the blame for an ambiguous prediction."""
    score = score_track(track(steady(2), ["atmospheric"] * 2, ["drop"] * 2), RAW9)
    claimed = INTENT_TO_LABELS["atmospheric"][RAW9]
    assert score.counts["drop"][2] == pytest.approx(1.0)        # fn
    for label in claimed:
        assert score.counts[label][1] == pytest.approx(1.0 / len(claimed))
    assert score.accuracy == pytest.approx(0.0)


def test_macro_f1_covers_classes_that_are_present_or_predicted_only():
    """A class nothing can predict still drags the macro down -- that is the point."""
    times = steady(4)
    labels = ["intro", "intro", "drop", "drop"]
    score = score_track(track(times, ["drop"] * 4, labels), RAW9)
    assert set(score.macro_classes) == {"intro", "drop"}
    assert score.macro_f1 == pytest.approx((0.0 + 2 / 3) / 2)


def test_expressible_macro_excludes_classes_no_observed_intent_can_say():
    times = steady(4)
    labels = ["intro", "intro", "drop", "drop"]
    score = score_track(track(times, ["drop"] * 4, labels), RAW9)
    assert set(score.expressible_classes) == {"drop"}
    assert score.macro_f1_expressible == pytest.approx(2 / 3)


def test_the_achievable_macro_f1_is_below_the_naive_upper_bound():
    """The unreachable mass has to be predicted as SOMETHING, so it lands as
    false positives on the classes that do exist.  A bound that ignores that is
    not a target anything can reach."""
    support = {"a": 1.0, "b": 1.0, "c": 2.0}            # c unreachable, U = 2
    # optimum concentrates the damage: one class keeps F1 1.0, the other takes
    # all 2.0 of it -> 2*1/(2*1+2) = 0.5.  Mean over all three classes.
    assert best_achievable_macro_f1(support, ["c"]) == pytest.approx(1.5 / 3)
    assert 0.5 < 2 / 3                                  # the naive bound


def test_with_nothing_unreachable_the_two_bounds_agree():
    support = {"a": 3.0, "b": 1.0}
    assert best_achievable_macro_f1(support, []) == pytest.approx(1.0)


def test_the_achievable_allocation_beats_every_other_allocation():
    """Property check: no hand allocation of the unreachable mass may do better.

    This is the test that caught the seductive wrong answer -- equalising the
    marginal loss across classes finds the MINIMUM of a convex objective, and
    scored 0.415 here against 0.533 for simply dumping everything on the
    largest class."""
    support = {"a": 1.0, "b": 3.0, "c": 4.0}
    best = best_achievable_macro_f1(support, ["c"])
    extra = 4.0
    for step in range(0, 101):
        x_a = extra * step / 100.0
        x_b = extra - x_a
        rival = (2 * 1.0 / (2 * 1.0 + x_a) + 2 * 3.0 / (2 * 3.0 + x_b)) / 3
        assert best >= rival - 1e-12
    assert best == pytest.approx((1.0 + 6 / 10) / 3)    # all 4.0 onto b


def test_an_unknown_label_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="unknown label"):
        score_track(track(steady(2), ["drop"] * 2, ["chorus"] * 2), RAW9)


def test_an_unknown_intent_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="unknown intent"):
        score_track(track(steady(2), ["strobe"] * 2, ["drop"] * 2), RAW9)


def test_a_retired_intent_in_a_stale_table_is_refused_rather_than_scored():
    """A table built before the intent alphabet shrank must fail, not read as new."""
    for retired in ("groove", "peak"):
        with pytest.raises(ValueError, match="unknown intent"):
            score_track(track(steady(2), [retired] * 2, ["drop"] * 2), RAW9)


# --------------------------------------------------------------------------- #
# event extraction
# --------------------------------------------------------------------------- #


def test_intent_changes_are_stamped_at_the_first_beat_carrying_the_new_intent():
    times = [0.0, 1.0, 2.0, 3.0]
    changes = intent_changes(times, ["breakdown", "breakdown", "drop", "drop"])
    assert changes == [(2.0, "drop")]


def test_the_first_committed_intent_is_not_a_change():
    """Rising from 'no intent yet' is the engine starting, not a musical move."""
    times = [0.0, 1.0, 2.0]
    assert intent_changes(times, ["", "breakdown", "breakdown"]) == []
    assert intent_changes(times, ["", "breakdown", "drop"]) == [(2.0, "drop")]


def test_label_boundaries_use_the_same_first_beat_convention_as_predictions():
    """Both sides quantise to beats, so a perfectly timed change lands at 0.0 s."""
    times = [0.0, 1.0, 2.0, 3.0]
    labels = ["intro", "intro", "drop", "drop"]
    assert label_boundaries(times, labels) == [(2.0, "drop")]
    assert intent_changes(times,
                          ["atmospheric", "atmospheric", "drop", "drop"])[0][0] == 2.0


def test_the_intent_stream_and_the_class_stream_are_now_the_same_stream():
    """With GROOVE and PEAK retired the intent -> claim map became injective:
    no two live intents claim the same class, so every lighting change is also
    a class change.  Asserted rather than assumed -- the two streams are still
    computed independently, and printing one number twice would look like
    corroboration."""
    times = steady(8)
    intents = (["atmospheric"] * 2 + ["buildup"] * 2 + ["drop"] * 2
               + ["breakdown"] * 2)
    labels = ["intro"] * 8
    for space in SPACES:
        claims = [INTENT_TO_LABELS[intent][space] for intent in INTENT_TO_LABELS]
        assert len(set(claims)) == len(claims), space
    score = score_track(track(times, intents, labels), RAW9)
    assert len(intent_changes(times, intents)) == 3
    assert [t for t, _ in class_changes(times, intents, RAW9)] == [1.0, 2.0, 3.0]
    for tol in TOLERANCES_SEC:
        assert (score.boundary["intent"][tol]["overall"]
                == score.boundary["class"][tol]["overall"])
        assert score.flicker["intent"][tol] == score.flicker["class"][tol]


def test_the_class_stream_can_never_flicker_more_than_the_intent_stream():
    times = steady(20)
    intents = ["drop" if i % 2 else "breakdown" for i in range(16)] + ["buildup"] * 4
    score = score_track(track(times, intents, ["drop"] * 20), RAW9)
    for tol in TOLERANCES_SEC:
        assert score.flicker["class"][tol] <= score.flicker["intent"][tol]


def test_the_class_stream_names_the_classes_the_new_intent_claims():
    times = steady(4)
    changes = class_changes(times, ["drop", "drop", "breakdown", "breakdown"], RAW9)
    assert [t for t, _ in changes] == [1.0]
    assert changes[0][1] == ("breakdown", "bridge", "cooldown")


def test_both_streams_are_reported_in_json_and_in_the_report():
    times = steady(8)
    labels = ["breakdown"] * 4 + ["drop"] * 4
    intents = ["atmospheric"] * 2 + ["breakdown"] * 2 + ["buildup"] * 2 + ["drop"] * 2
    result = evaluate([track(times, intents, labels, track_id="x")])
    space = result["spaces"][RAW9]
    assert set(space["streams"]) == set(STREAM_ORDER)
    assert set(space["per_song"][0]["changes"]) == set(STREAM_ORDER)
    assert set(space["per_song"][0]["boundary_f1"]) == set(STREAM_ORDER)
    assert space["streams"]["intent"]["changes_total"] == 3
    assert space["streams"]["class"]["changes_total"] == 3
    text = render_report(result)
    assert "intent stream" in text.lower() and "class stream" in text.lower()


def test_each_space_block_names_itself():
    result = evaluate([track(steady(2), ["drop"] * 2, ["drop"] * 2)])
    for name, space in result["spaces"].items():
        assert space["space"] == name


# --------------------------------------------------------------------------- #
# boundary matching
# --------------------------------------------------------------------------- #


def test_a_boundary_exactly_at_the_tolerance_matches_and_just_past_it_does_not():
    assert match_events([10.0], [12.0], 2.0) == 1
    assert match_events([10.0], [8.0], 2.0) == 1
    assert match_events([10.0], [12.0 + 1e-9], 2.0) == 0


def test_two_predictions_near_one_boundary_match_only_once():
    assert match_events([10.0], [9.5, 10.5], 2.0) == 1


def test_two_boundaries_near_one_prediction_match_only_once():
    assert match_events([10.0, 11.0], [10.5], 2.0) == 1


def test_matching_does_not_waste_a_prediction_on_the_earlier_boundary():
    """Greedy in time order must still find the maximum matching."""
    assert match_events([0.0, 10.0], [9.5], 2.0) == 1
    assert match_events([0.0, 10.0], [1.0, 9.5], 2.0) == 2


def test_matching_is_symmetric_in_cardinality_and_order_independent():
    truth = [3.0, 1.0, 8.0]
    pred = [8.4, 0.9, 100.0]
    assert match_events(truth, pred, 0.5) == 2
    assert match_events(sorted(truth), sorted(pred), 0.5) == 2


def test_a_perfect_timeline_has_boundary_f1_one_at_every_tolerance():
    times = steady(8)
    labels = ["intro"] * 4 + ["drop"] * 4
    intents = ["atmospheric"] * 4 + ["drop"] * 4
    score = score_track(track(times, intents, labels), RAW9)
    for stream in STREAM_ORDER:
        for tol in TOLERANCES_SEC:
            overall = score.boundary[stream][tol]["overall"]
            assert prf(overall["matched"],
                       overall["n_pred"] - overall["matched"],
                       overall["n_truth"] - overall["matched"])[2] == pytest.approx(1.0)


def test_events_never_match_across_track_boundaries():
    """Two tracks each starting at t=0 must not shadow each other's boundaries."""
    a = track(steady(4), ["breakdown"] * 2 + ["drop"] * 2,
              ["breakdown"] * 2 + ["drop"] * 2, track_id="a")
    b = track(steady(4), ["drop"] * 4, ["drop"] * 4, track_id="b")
    corpus = aggregate([score_track(a, RAW9), score_track(b, RAW9)])
    assert corpus.boundary["intent"][2.0]["overall"]["n_truth"] == 1
    assert corpus.boundary["intent"][2.0]["overall"]["n_pred"] == 1
    assert corpus.boundary["intent"][2.0]["overall"]["matched"] == 1


# --------------------------------------------------------------------------- #
# typed (per-boundary-type) breakdown
# --------------------------------------------------------------------------- #


def test_a_drop_boundary_is_only_credited_to_a_change_into_drop():
    times = steady(8)
    labels = ["breakdown"] * 4 + ["drop"] * 4
    hit = score_track(track(times, ["breakdown"] * 4 + ["drop"] * 4, labels), RAW9)
    assert hit.boundary["intent"][2.0]["by_type"]["drop"]["matched"] == 1
    miss = score_track(track(times, ["breakdown"] * 4 + ["buildup"] * 4, labels), RAW9)
    assert miss.boundary["intent"][2.0]["by_type"]["drop"]["matched"] == 0
    assert miss.boundary["intent"][2.0]["by_type"]["drop"]["n_truth"] == 1
    assert miss.boundary["intent"][2.0]["by_type"]["drop"]["n_pred"] == 0


def test_typed_counts_partition_the_overall_counts():
    """Typed predictions must PARTITION the stream, or a precision denominator
    silently counts one change twice.  ATMOSPHERIC is in the timeline precisely
    because it claims four classes."""
    times = steady(15)
    labels = (["intro"] * 3 + ["buildup"] * 3 + ["drop"] * 3
              + ["cooldown"] * 3 + ["altoutro"] * 3)
    intents = (["atmospheric"] * 3 + ["buildup"] * 3 + ["drop"] * 3
               + ["breakdown"] * 3 + ["atmospheric"] * 3)
    score = score_track(track(times, intents, labels), RAW9)
    for stream in STREAM_ORDER:
        for tol in TOLERANCES_SEC:
            overall = score.boundary[stream][tol]["overall"]
            by_type = score.boundary[stream][tol]["by_type"]
            assert sum(v["n_truth"] for v in by_type.values()) == overall["n_truth"]
            assert sum(v["n_pred"] for v in by_type.values()) == overall["n_pred"]
            assert sum(v["matched"] for v in by_type.values()) <= overall["matched"]


def test_an_ambiguous_change_is_one_prediction_credited_to_one_class():
    """ATMOSPHERIC claims four quiet classes; counting it in each inflates all."""
    claimed = INTENT_TO_LABELS["atmospheric"][RAW9]
    changes = [(10.0, claimed)]
    near_intro = typed_predictions(
        changes, {**empty_buckets(), "intro": [10.5], "altoutro": [40.0]})
    assert near_intro["intro"] == [10.0]
    assert near_intro["altoutro"] == []
    near_outro = typed_predictions(
        changes, {**empty_buckets(), "intro": [40.0], "altoutro": [10.5]})
    assert near_outro["altoutro"] == [10.0]
    assert near_outro["intro"] == []
    for buckets in (near_intro, near_outro):
        assert sum(len(bucket) for bucket in buckets.values()) == len(changes)


def test_an_ambiguous_change_with_nothing_to_match_still_lands_somewhere_once():
    changes = [(10.0, INTENT_TO_LABELS["atmospheric"][RAW9])]
    buckets = typed_predictions(changes, empty_buckets())
    assert buckets["intro"] == [10.0]           # deterministic: first claimed
    assert sum(len(bucket) for bucket in buckets.values()) == 1


# --------------------------------------------------------------------------- #
# flicker
# --------------------------------------------------------------------------- #


def test_flicker_counts_changes_far_from_every_boundary():
    truth = [10.0, 50.0]
    pred = [10.2, 30.0, 49.0]
    assert flicker_instants(pred, truth, 2.0) == [30.0]


def test_flicker_forgives_a_double_change_at_one_boundary_that_precision_punishes():
    """Proximity, not matching: the product question is 'did the lights twitch?'."""
    truth = [10.0]
    pred = [9.5, 10.5]
    assert flicker_instants(pred, truth, 2.0) == []
    assert match_events(truth, pred, 2.0) == 1      # precision still 0.5


def test_flicker_rate_is_per_minute_of_evaluated_time():
    times = steady(240, step=0.5)                   # 120 s of beats
    labels = ["drop"] * 240
    intents = ["drop" if i % 2 else "breakdown" for i in range(240)]
    score = score_track(track(times, intents, labels), RAW9)
    corpus = aggregate([score])
    assert corpus.exposure_sec == pytest.approx(120.0)
    rate = corpus.flicker["intent"][2.0] / (corpus.exposure_sec / 60.0)
    assert rate == pytest.approx(corpus.flicker_per_minute["intent"][2.0])
    assert corpus.flicker_per_minute["intent"][2.0] > 100   # 239 changes in 2 min


def test_the_flicker_denominator_is_show_time_not_predicted_time():
    """Using scored time would make the rate creep up as coverage falls."""
    times = steady(4)
    score = score_track(
        track(times, ["", "drop", "breakdown", "drop"], ["drop"] * 4), RAW9)
    assert score.exposure_sec > score.scored_sec > 0
    loose = score.flicker["intent"][2.0]
    assert loose > 0
    assert score.flicker_per_minute["intent"][2.0] == pytest.approx(
        loose / (score.exposure_sec / 60.0))
    assert score.flicker_per_minute["intent"][2.0] != pytest.approx(
        loose / (score.scored_sec / 60.0))


def test_a_stable_correct_timeline_never_flickers():
    times = steady(8)
    labels = ["breakdown"] * 4 + ["drop"] * 4
    score = score_track(track(times, ["breakdown"] * 4 + ["drop"] * 4, labels), RAW9)
    for stream in STREAM_ORDER:
        for tol in TOLERANCES_SEC:
            assert score.flicker[stream][tol] == 0


# --------------------------------------------------------------------------- #
# aggregation and the worst-song list
# --------------------------------------------------------------------------- #


def test_corpus_counts_are_the_sum_of_the_song_counts():
    a = score_track(track(steady(4), ["drop"] * 4, ["drop"] * 4, track_id="a"), RAW9)
    b = score_track(track(steady(4), ["breakdown"] * 4, ["drop"] * 4, track_id="b"),
                    RAW9)
    corpus = aggregate([a, b])
    for index in range(3):
        assert corpus.counts["drop"][index] == pytest.approx(
            a.counts["drop"][index] + b.counts["drop"][index])
    assert corpus.exposure_sec == pytest.approx(a.exposure_sec + b.exposure_sec)


def test_worst_songs_are_ordered_by_score_then_id():
    good = track(steady(4), ["drop"] * 4, ["drop"] * 4, track_id="good")
    bad_a = track(steady(4), ["breakdown"] * 4, ["drop"] * 4, track_id="a-bad")
    bad_b = track(steady(4), ["breakdown"] * 4, ["drop"] * 4, track_id="b-bad")
    result = evaluate([bad_b, good, bad_a], worst=2)
    worst = [row["track_id"] for row in result["spaces"][RAW9]["worst_songs"]]
    assert worst == ["a-bad", "b-bad"]


def test_prf_returns_zero_rather_than_dividing_by_zero():
    assert prf(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)
    assert prf(0.0, 1.0, 0.0) == (0.0, 0.0, 0.0)
    assert prf(0.0, 0.0, 1.0) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# end to end: loader, result shape, renderer
# --------------------------------------------------------------------------- #


def _write_table(path, rows):
    header = ",".join(TABLE_HEADER)
    lines = [header]
    for row in rows:
        full = {column: "0" for column in TABLE_HEADER}
        full.update(row)
        lines.append(",".join(str(full[column]) for column in TABLE_HEADER))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")


def test_load_tracks_groups_by_track_and_keeps_beat_order(tmp_path):
    path = tmp_path / "training_table.csv.gz"
    _write_table(path, [
        {"track_id": "a", "t_song": "0.0", "intent_at_beat": "drop",
         LABEL_COLUMN: "drop"},
        {"track_id": "a", "t_song": "0.5", "intent_at_beat": "",
         LABEL_COLUMN: "drop"},
        {"track_id": "b", "t_song": "1.0", "intent_at_beat": "breakdown",
         LABEL_COLUMN: "cooldown"},
    ])
    tracks = load_tracks(path)
    assert [t.track_id for t in tracks] == ["a", "b"]
    assert tracks[0].times == (0.0, 0.5)
    assert tracks[0].intents == ("drop", "")
    assert tracks[1].labels[RAW9] == ("cooldown",)
    assert tracks[1].labels[LEGACY_V1] == ("breakdown",)


def test_load_tracks_derives_every_space_from_the_one_column(tmp_path):
    path = tmp_path / "training_table.csv.gz"
    _write_table(path, [
        {"track_id": "a", "t_song": f"{index}.0", "intent_at_beat": "atmospheric",
         LABEL_COLUMN: label}
        for index, label in enumerate(SECTION_LABELS)
    ])
    loaded = load_tracks(path)[0]
    assert loaded.labels[RAW9] == SECTION_LABELS
    assert loaded.labels[LEGACY_V1] == tuple(
        fold_to_legacy_v1(label) for label in SECTION_LABELS)


def test_evaluate_reports_both_spaces_and_a_renderable_report():
    times = steady(12)
    labels = ["intro"] * 3 + ["buildup"] * 3 + ["drop"] * 3 + ["cooldown"] * 3
    intents = ["atmospheric"] * 3 + ["buildup"] * 3 + ["drop"] * 3 + ["breakdown"] * 3
    result = evaluate([track(times, intents, labels, track_id="x")])
    assert set(result["spaces"]) == {RAW9, LEGACY_V1}
    assert result["spaces"][RAW9]["macro_f1"] <= 1.0
    assert result["coverage"]["rows"] == 12
    text = render_report(result)
    for heading in ("CONFUSION", "PER-CLASS", "BOUNDARY", "FLICKER",
                    "ATMOSPHERIC", "WORST"):
        assert heading in text.upper()
    assert text.isascii()                   # pasteable into any console/report


def test_the_report_names_the_raw_space_first_because_it_is_primary():
    result = evaluate([track(steady(4), ["drop"] * 4, ["drop"] * 4)])
    text = render_report(result)
    assert text.index(RAW9.upper()) < text.index(LEGACY_V1.upper())


def test_evaluate_survives_a_corpus_with_nothing_to_score():
    result = evaluate([track([0.0], [""], ["drop"], track_id="empty")])
    assert result["spaces"][RAW9]["macro_f1"] == 0.0
    assert math.isfinite(
        result["spaces"][RAW9]["streams"]["intent"]["flicker_per_audience_minute"]["2.0"])
    assert render_report(result)


def test_atmospheric_absence_is_reported_as_a_structural_finding():
    """A whole label family no observed intent can express is not a mistake."""
    times = steady(9)
    labels = ["intro"] * 3 + ["drop"] * 3 + ["buildup"] * 3
    intents = ["breakdown"] * 3 + ["drop"] * 3 + ["buildup"] * 3
    result = evaluate([track(times, intents, labels)])
    structural = result["spaces"][RAW9]["structural"]
    assert structural["observed_intents"] == ["breakdown", "buildup", "drop"]
    assert structural["unreachable_classes"] == ["intro", "altintro", "outro",
                                                 "altoutro"]
    assert structural["unreachable_label_share"] == pytest.approx(1 / 3)
    assert structural["macro_f1_upper_bound"] == pytest.approx(5 / 9)
    assert structural["macro_f1_best_achievable"] < structural["macro_f1_upper_bound"]
