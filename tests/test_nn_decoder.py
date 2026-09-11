"""The fitted section priors and the fixed-lag Viterbi decoder."""
import ast
import dataclasses
import inspect
import json
import subprocess
import sys
from pathlib import Path

import math
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from lib.label_space import SECTION_LABELS  # noqa: E402
from nn import priors as nn_priors  # noqa: E402
from nn.decoder import (  # noqa: E402
    DEFAULT_BOUNDARY_REF,
    SHIPPING_DECODER_CONFIG,
    DecodeParams,
    FixedLagViterbi,
    bar_grid,
    bar_observations,
    build_decoder,
    decode_track,
    load_decoder_config,
    observation_knobs,
    segments,
    temper,
    trellis_knobs,
)
from nn.priors import (  # noqa: E402
    PRIORS_FILE,
    SKIP_CAUSES,
    SKIP_DEGENERATE_GRID,
    SKIP_NO_ANNOTATION,
    SKIP_NO_BAR_RUNS,
    SKIP_NO_BEAT_GRID,
    Priors,
    bar_runs,
    corpus_bar_runs,
    corpus_bar_runs_by_cause,
    fit,
    fit_runs,
    label_runs,
    section_classes,
    INTRO_FAMILY,
    OUTRO_FAMILY,
    transition_allowed,
)

# The shipping chain's five classes.  A subset of the vocabulary, in vocabulary
# order, which ruling #264 requires to keep fitting, loading and decoding -- so
# the trellis tests below are written against it rather than the full nine.
SHIPPING_CLASSES = ("intro", "buildup", "breakdown", "drop", "outro")

INTRO, BUILDUP, BREAKDOWN, DROP, OUTRO = range(5)
INDEX = {label: i for i, label in enumerate(SHIPPING_CLASSES)}


def fit_shipping(sequences, **kwargs) -> Priors:
    return fit_runs(sequences, classes=SHIPPING_CLASSES, **kwargs)


def corpus_runs():
    return [
        [("intro", 16), ("buildup", 16), ("drop", 32), ("breakdown", 16), ("outro", 8)],
        [("intro", 16), ("buildup", 24), ("breakdown", 8), ("drop", 32), ("outro", 16)],
        [("intro", 24), ("buildup", 8), ("breakdown", 16), ("drop", 48), ("outro", 8)],
        [("intro", 8), ("buildup", 32), ("breakdown", 24), ("drop", 16), ("outro", 24)],
        [("intro", 32), ("buildup", 16), ("breakdown", 32), ("drop", 32), ("outro", 16)],
    ]


def toy_priors(floor=4, hazard=0.25, class_prior=None, initial=None):
    classes = SHIPPING_CLASSES
    n = len(classes)
    transition = np.zeros((n, n), dtype=np.float64)
    for i, src in enumerate(classes):
        for j, dst in enumerate(classes):
            if transition_allowed(src, dst):
                transition[i, j] = 1.0
    rows = transition.sum(axis=1, keepdims=True)
    transition = np.divide(transition, rows, out=np.zeros_like(transition),
                           where=rows > 0)
    if initial is None:
        initial = np.full(n, 1.0 / n)
    if class_prior is None:
        class_prior = np.full(n, 1.0 / n)
    return Priors(
        classes=tuple(classes),
        initial=np.asarray(initial, dtype=np.float64),
        transition=transition,
        floor_bars=np.full(n, int(floor), dtype=np.int64),
        hazard=np.full(n, float(hazard), dtype=np.float64),
        class_prior=np.asarray(class_prior, dtype=np.float64),
        corpus={},
    )


def one_hot(index, strength=0.97, n=5):
    row = np.full(n, (1.0 - strength) / (n - 1))
    row[index] = strength
    return row


def labels_of(decisions):
    return [d.label for d in decisions]


def test_the_transition_rule_bars_intro_re_entry_outro_exit_and_self_loops():
    for src in SHIPPING_CLASSES:
        assert not transition_allowed(src, "intro")
        assert not transition_allowed("outro", src)
        assert not transition_allowed(src, src)
    assert transition_allowed("buildup", "drop")
    assert transition_allowed("drop", "breakdown")
    assert transition_allowed("breakdown", "outro")


def test_the_structural_rule_is_about_families_not_single_classes():
    """The measured fact -- nothing re-enters the introduction, nothing follows
    the outro -- was measured while the folds collapsed each pair into one
    class, so it was only ever true at family granularity."""
    for src in SECTION_LABELS:
        if src not in INTRO_FAMILY:
            assert not transition_allowed(src, "intro")
            assert not transition_allowed(src, "altintro")
        if src not in OUTRO_FAMILY:
            assert not transition_allowed("outro", src)
            assert not transition_allowed("altoutro", src)


def test_a_within_family_move_is_legal_because_that_is_the_owners_beat_in():
    assert transition_allowed("altintro", "intro")
    assert transition_allowed("intro", "altintro")
    assert transition_allowed("outro", "altoutro")
    assert transition_allowed("altoutro", "outro")


def test_a_family_is_still_sealed_against_everything_outside_it():
    assert not transition_allowed("drop", "altintro")
    assert not transition_allowed("altoutro", "drop")
    assert not transition_allowed("altintro", "altintro")


def test_fitted_transition_rows_are_stochastic_and_illegal_entries_are_zero():
    priors = fit_shipping(corpus_runs())
    for i, src in enumerate(SHIPPING_CLASSES):
        row = priors.transition[i]
        for j, dst in enumerate(SHIPPING_CLASSES):
            if not transition_allowed(src, dst):
                assert row[j] == 0.0, f"{src}->{dst} must be structurally impossible"
        if src == "outro":
            assert row.sum() == 0.0, "outro has no legal successor at all"
        else:
            assert row.sum() == pytest.approx(1.0)


def test_log_transition_is_minus_inf_exactly_where_the_probability_is_zero():
    priors = fit_shipping(corpus_runs())
    zero = priors.transition == 0.0
    assert np.all(np.isneginf(priors.log_transition[zero]))
    assert np.all(np.isfinite(priors.log_transition[~zero]))


def test_buildup_fork_is_forced_near_uniform_despite_a_lopsided_corpus():
    runs = [[("intro", 16), ("buildup", 16), ("breakdown", 16), ("outro", 16)]] * 8
    runs += [[("intro", 16), ("buildup", 16), ("drop", 16), ("outro", 16)]] * 2
    priors = fit_shipping(runs)
    row = priors.transition[INDEX["buildup"]]
    assert row[INDEX["breakdown"]] == pytest.approx(row[INDEX["drop"]])
    combined = row[INDEX["breakdown"]] + row[INDEX["drop"]]
    assert combined > 0.9, "the fork should still hold nearly all of buildup's mass"


def test_a_legal_but_unobserved_transition_keeps_a_little_mass():
    priors = fit_shipping(corpus_runs())
    assert priors.transition[INDEX["intro"], INDEX["outro"]] > 0.0


def test_fitting_refuses_a_corpus_that_contradicts_the_structural_graph():
    bad = corpus_runs() + [[("intro", 16), ("outro", 16), ("drop", 16)]]
    with pytest.raises(RuntimeError, match="outro->drop"):
        fit_shipping(bad)
    relaxed = fit_shipping(bad, strict=False)
    assert relaxed.corpus["illegal_observed"]["outro->drop"] == 1
    assert relaxed.transition[INDEX["outro"], INDEX["drop"]] == 0.0


def test_initial_distribution_is_fitted_not_assumed():
    priors = fit_shipping(corpus_runs() * 8)
    assert priors.initial.sum() == pytest.approx(1.0)
    assert priors.initial.argmax() == INDEX["intro"]
    assert priors.initial[INDEX["intro"]] > 0.9
    assert np.all(priors.initial > 0.0), "no opening is impossible, only unlikely"


def test_duration_floor_is_the_corpus_fifth_percentile_in_bars():
    priors = fit_shipping(corpus_runs())
    drop_bars = sorted([32, 32, 48, 16, 32])
    expected = max(1, int(round(float(np.percentile(drop_bars, 5.0)))))
    assert priors.floor_bars[INDEX["drop"]] == expected


def test_duration_tail_is_the_geometric_that_halves_at_the_corpus_median():
    priors = fit_shipping(corpus_runs())
    index = INDEX["drop"]
    floor = int(priors.floor_bars[index])
    median = float(np.median([32, 32, 48, 16, 32]))
    residual = max(1.0, median - floor)
    assert priors.hazard[index] == pytest.approx(1.0 - 0.5 ** (1.0 / residual))
    assert (1.0 - priors.hazard[index]) ** residual == pytest.approx(0.5)


def test_floor_is_at_least_one_bar_even_for_a_degenerate_class():
    priors = fit_shipping([[("intro", 1), ("drop", 1), ("outro", 1)]])
    assert np.all(priors.floor_bars >= 1)
    assert np.all(priors.hazard > 0.0)
    assert np.all(priors.hazard <= 1.0)


def test_class_prior_is_bar_occupancy_not_run_count():
    runs = [[("intro", 8), ("breakdown", 8), ("drop", 64), ("outro", 8)]] * 4
    priors = fit_shipping(runs)
    assert priors.class_prior.sum() == pytest.approx(1.0)
    assert priors.class_prior[INDEX["drop"]] == pytest.approx(64 / 88)
    assert priors.class_prior[INDEX["breakdown"]] == pytest.approx(8 / 88)


def test_priors_json_round_trips_and_the_same_content_gives_the_same_bytes(tmp_path):
    priors = fit_shipping(corpus_runs())
    path = tmp_path / PRIORS_FILE
    priors.save(path)
    again = Priors.load(path)
    assert again.classes == priors.classes
    for field in ("initial", "transition", "floor_bars", "hazard", "class_prior"):
        np.testing.assert_array_equal(getattr(again, field), getattr(priors, field))
    second = tmp_path / "again.json"
    again.save(second)
    assert path.read_bytes() == second.read_bytes()


def test_priors_file_is_plain_json_with_no_infinities():
    priors = fit_shipping(corpus_runs())
    text = json.dumps(priors.to_dict())
    assert "Infinity" not in text and "NaN" not in text


def test_label_runs_merge_across_a_dropped_sentinel_without_folding():
    sections = [
        (0.0, 10.0, "intro"),
        (10.0, 20.0, "drop"),
        (20.0, 30.0, "end"),
        (30.0, 40.0, "drop"),
        (40.0, 50.0, "cooldown"),
        (50.0, 60.0, "altoutro"),
    ]
    assert label_runs(sections) == [
        (0.0, 10.0, "intro"), (10.0, 40.0, "drop"),
        (40.0, 50.0, "cooldown"), (50.0, 60.0, "altoutro")]


def test_the_default_class_space_to_fit_in_is_the_whole_vocabulary():
    assert section_classes() == SECTION_LABELS
    assert fit_runs(corpus_runs()).classes == SECTION_LABELS


def test_the_owners_beat_in_now_fits_instead_of_raising():
    """``altintro`` -> ``intro`` was invisible while the folds collapsed the
    pair.  Under family semantics it is an ordinary within-family move, so a
    corpus carrying it fits rather than failing the strict gate."""
    fitted = fit_runs([[("altintro", 8), ("intro", 8), ("drop", 8), ("outro", 8)]])
    assert fitted.classes == SECTION_LABELS


def test_a_cross_family_violation_still_fails_the_strict_gate_loudly():
    with pytest.raises(RuntimeError, match="drop->intro"):
        fit_runs([[("altintro", 8), ("drop", 8), ("intro", 8), ("outro", 8)]])


def test_a_priors_file_naming_the_shipping_five_classes_still_loads(tmp_path):
    path = tmp_path / PRIORS_FILE
    fit_shipping(corpus_runs()).save(path)
    assert Priors.load(path).classes == SHIPPING_CLASSES


@pytest.mark.parametrize("classes, complaint", [
    (("intro", "verse", "breakdown", "drop", "outro"), "vocabulary does not know"),
    (("buildup", "intro", "breakdown", "drop", "outro"), "wrong order"),
    (("intro", "intro", "breakdown", "drop", "outro"), "duplicate"),
])
def test_a_priors_file_whose_class_space_is_not_the_vocabulary_is_refused(
        tmp_path, classes, complaint):
    """Every table in the file is indexed by this list, so a wrong one loads
    cleanly and decodes shifted."""
    path = tmp_path / PRIORS_FILE
    document = fit_shipping(corpus_runs()).to_dict()
    document["classes"] = list(classes)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=complaint) as error:
        Priors.load(path)
    assert str(path) in str(error.value)


def test_a_decoder_built_on_a_class_space_outside_the_vocabulary_is_refused():
    """The floors, the class prior and the transition matrix are all positional."""
    broken = toy_priors()._replace(
        classes=("intro", "verse", "breakdown", "drop", "outro"))
    with pytest.raises(ValueError, match="vocabulary does not know"):
        FixedLagViterbi(broken, lag_bars=2)


@pytest.mark.parametrize("name", ["v1_classes", "v1_runs"])
def test_the_folded_class_space_is_gone_from_the_priors_module(name):
    """Retired, not merely unused: a surviving alias is a second vocabulary."""
    assert not hasattr(nn_priors, name)


def test_bar_runs_counts_downbeats_and_never_reattributes_dropped_time():
    bar_sec = 2.0
    downbeats = np.arange(0.0, 40.0, bar_sec)
    sections = [
        (0.0, 10.0, "intro"),
        (10.0, 20.0, "drop"),
        (20.0, 30.0, "end"),
        (30.0, 40.0, "drop"),
    ]
    assert bar_runs(sections, downbeats) == [("intro", 5), ("drop", 10)]


# --------------------------------------------------------------------------- #
# Skip accounting: four causes, and the arithmetic nobody was checking
# --------------------------------------------------------------------------- #

SKIP_SECTIONS = [
    {"name": "intro", "start": 0.0, "end": 8.0},
    {"name": "drop", "start": 8.0, "end": 24.0},
]


def skip_corpus(tmp_path, *, records, grids, split=None):
    """A minimal corpus on disk: ``segments.json``, beat grids, a frozen split.

    ``records`` are the ids ``segments.json`` names; ``grids`` maps an id to the
    ``(bars, t0)`` of the grid written for it.  An id in ``records`` with no
    grid has no file on disk; an id in neither is the incident's own cause --
    requested, but absent from the annotation source being consulted.
    """
    data_dir = tmp_path / "raveform"
    beats = data_dir / "annotations" / "beats"
    beats.mkdir(parents=True)
    with open(data_dir / "annotations" / "segments.json", "w",
              encoding="utf-8") as handle:
        json.dump([{"key": name, "id": name, "title": f"An Artist - {name}",
                    "duration": 24.0, "sections": SKIP_SECTIONS}
                   for name in records], handle)
    for name, (bars, t0) in grids.items():
        write_beat_csv(beats / f"{name}.beat.csv", bars=bars, bar_sec=2.0, t0=t0)
    if split is not None:
        with open(data_dir / "splits.json", "w", encoding="utf-8") as handle:
            json.dump({"train": list(split), "val": [], "test": []}, handle)
    return data_dir


def four_cause_corpus(tmp_path, *, split=True):
    """One id per cause plus two that fit -- six requested, two fitted."""
    ids = ["fits_a", "fits_b", "no_grid", "one_downbeat", "off_grid", "ghost"]
    data_dir = skip_corpus(
        tmp_path,
        records=[name for name in ids if name != "ghost"],
        grids={"fits_a": (8, 0.0), "fits_b": (8, 0.0), "one_downbeat": (1, 0.0),
               "off_grid": (8, 100.0)},
        split=ids if split else None,
    )
    return data_dir, ids


def test_a_track_absent_from_the_annotation_source_is_not_a_missing_beat_grid(tmp_path):
    """The incident, exactly: five tracks whose grids were present and valid on
    disk were reported under a key that names the beat grid, and everyone went
    looking at the wrong artifact."""
    data_dir = skip_corpus(tmp_path, records=["present"],
                           grids={"present": (8, 0.0)})

    sequences, skipped = corpus_bar_runs_by_cause(data_dir, ["present", "ghost"])

    assert len(sequences) == 1
    assert skipped[SKIP_NO_ANNOTATION] == ["ghost"]
    assert skipped[SKIP_NO_BEAT_GRID] == []


def test_a_track_whose_grid_file_is_missing_is_counted_under_the_missing_grid_cause(tmp_path):
    data_dir = skip_corpus(tmp_path, records=["present", "no_grid"],
                           grids={"present": (8, 0.0)})

    sequences, skipped = corpus_bar_runs_by_cause(data_dir, ["present", "no_grid"])

    assert len(sequences) == 1
    assert skipped[SKIP_NO_BEAT_GRID] == ["no_grid"]
    assert skipped[SKIP_NO_ANNOTATION] == []


def test_a_degenerate_grid_and_a_track_with_no_bar_runs_land_in_their_own_causes(tmp_path):
    data_dir = skip_corpus(
        tmp_path, records=["one_downbeat", "off_grid"],
        grids={"one_downbeat": (1, 0.0), "off_grid": (8, 100.0)})

    sequences, skipped = corpus_bar_runs_by_cause(
        data_dir, ["one_downbeat", "off_grid"])

    assert sequences == []
    assert skipped[SKIP_DEGENERATE_GRID] == ["one_downbeat"]
    assert skipped[SKIP_NO_BAR_RUNS] == ["off_grid"]
    assert skipped[SKIP_NO_BEAT_GRID] == []


def test_the_per_cause_counts_and_the_fitted_count_sum_to_the_ids_requested(tmp_path):
    """The arithmetic nothing was checking: a fit over 955 of 960 tracks looked
    exactly like a fit over 960."""
    data_dir, ids = four_cause_corpus(tmp_path, split=False)

    sequences, skipped = corpus_bar_runs_by_cause(data_dir, ids)

    assert set(skipped) == set(SKIP_CAUSES)
    assert len(sequences) + sum(len(v) for v in skipped.values()) == len(ids)
    assert len(sequences) == 2


def test_the_flat_skip_list_still_carries_every_cause_for_existing_readers(tmp_path):
    data_dir, ids = four_cause_corpus(tmp_path, split=False)

    _sequences, skipped = corpus_bar_runs(data_dir, ids)

    assert skipped == ["ghost", "no_grid", "off_grid", "one_downbeat"]


def test_the_provenance_names_only_genuinely_missing_grids_under_the_legacy_key(tmp_path):
    data_dir, ids = four_cause_corpus(tmp_path)

    corpus = fit(data_dir, classes=SHIPPING_CLASSES).corpus

    assert corpus["skipped_no_beat_grid"] == ["no_grid"]
    assert corpus["skipped_no_annotation_record"] == ["ghost"]
    assert corpus["skipped_degenerate_beat_grid"] == ["one_downbeat"]
    assert corpus["skipped_no_bar_runs"] == ["off_grid"]
    assert corpus["split_size"] == len(ids)
    assert corpus["fitted_tracks"] + corpus["skipped_tracks"] == corpus["split_size"]


def test_the_opt_in_strictness_flag_refuses_a_fit_that_dropped_an_id(tmp_path):
    data_dir, _ids = four_cause_corpus(tmp_path)

    with pytest.raises(RuntimeError, match="skipped_no_annotation_record"):
        fit(data_dir, classes=SHIPPING_CLASSES, require_all_ids=True)


def test_a_dropped_id_warns_by_default_rather_than_raising(tmp_path, caplog):
    data_dir, ids = four_cause_corpus(tmp_path)

    with caplog.at_level("WARNING"):
        priors = fit(data_dir, classes=SHIPPING_CLASSES)

    assert priors.corpus["fitted_tracks"] == 2
    warning = "\n".join(record.getMessage() for record in caplog.records)
    assert "4 of 6" in warning
    assert "skipped_no_annotation_record=1" in warning
    assert "ghost" in warning


def test_a_complete_fit_neither_warns_nor_refuses_under_the_strict_flag(tmp_path, caplog):
    data_dir = skip_corpus(tmp_path, records=["a", "b"],
                           grids={"a": (8, 0.0), "b": (8, 0.0)},
                           split=["a", "b"])

    with caplog.at_level("WARNING"):
        priors = fit(data_dir, classes=SHIPPING_CLASSES, require_all_ids=True)

    assert priors.corpus["skipped_tracks"] == 0
    assert caplog.records == []


def test_the_warning_samples_the_dropped_ids_rather_than_listing_all_of_them(tmp_path, caplog):
    ghosts = [f"ghost{index:03d}" for index in range(40)]
    data_dir = skip_corpus(tmp_path, records=["a"], grids={"a": (8, 0.0)},
                           split=["a"] + ghosts)

    with caplog.at_level("WARNING"):
        fit(data_dir, classes=SHIPPING_CLASSES)

    warning = "\n".join(record.getMessage() for record in caplog.records)
    assert "40 of 41" in warning
    assert warning.count("ghost") < 40, "a 900-id dump is not a warning anyone reads"
    assert "more" in warning


def test_isolated_flicker_bars_are_outvoted_by_the_duration_prior():
    priors = toy_priors(floor=4)
    posteriors = np.array([one_hot(DROP) for _ in range(40)])
    single_bar_spikes_of_breakdown = (7, 13, 22, 31)
    for bar in single_bar_spikes_of_breakdown:
        posteriors[bar] = one_hot(BREAKDOWN)
    decoder = FixedLagViterbi(priors, lag_bars=3)
    assert set(labels_of(decoder.decode(posteriors))) == {"drop"}


def test_a_genuine_switch_is_followed_once_and_cleanly():
    priors = toy_priors(floor=4)
    posteriors = np.array([one_hot(BREAKDOWN)] * 20 + [one_hot(DROP)] * 20)
    decoder = FixedLagViterbi(priors, lag_bars=3)
    spans = segments(decoder.decode(posteriors))
    assert [span[2] for span in spans] == ["breakdown", "drop"]
    assert spans[1][0] == 20, "the switch belongs on the bar the evidence changes"


def test_minimum_duration_is_honoured_under_bar_by_bar_alternation():
    priors = toy_priors(floor=5)
    posteriors = np.array([one_hot(DROP if bar % 2 else BREAKDOWN)
                           for bar in range(60)])
    spans = segments(FixedLagViterbi(priors, lag_bars=3).decode(posteriors))
    for start, end, label in spans[:-1]:
        assert end - start >= int(priors.floor_bars[INDEX[label]])


def test_floor_scale_widens_or_relaxes_the_minimum_dwell():
    priors = toy_priors(floor=8)
    posteriors = np.array([one_hot(DROP if (bar // 4) % 2 else BREAKDOWN)
                           for bar in range(64)])
    tight = segments(FixedLagViterbi(priors, lag_bars=2, floor_scale=0.25)
                     .decode(posteriors))
    loose = segments(FixedLagViterbi(priors, lag_bars=2, floor_scale=1.0)
                     .decode(posteriors))
    assert len(tight) > len(loose), "a smaller floor must permit more switches"


def test_no_illegal_transition_is_emitted_under_adversarial_evidence():
    priors = toy_priors(floor=2)
    posteriors = np.array(
        [one_hot(INTRO)] * 6 + [one_hot(OUTRO)] * 6
        + [one_hot(DROP)] * 6 + [one_hot(INTRO)] * 6)
    spans = segments(FixedLagViterbi(priors, lag_bars=2).decode(posteriors))
    for before, after in zip(spans, spans[1:]):
        assert transition_allowed(before[2], after[2]), f"{before[2]}->{after[2]}"


def test_a_class_with_no_legal_successor_absorbs_the_rest_of_the_track():
    priors = toy_priors(floor=2)
    posteriors = np.array([one_hot(OUTRO)] * 10 + [one_hot(DROP)] * 10)
    spans = segments(FixedLagViterbi(priors, lag_bars=2).decode(posteriors))
    assert spans[-1][2] == "outro"
    assert "outro" not in [span[2] for span in spans[:-1]]


def test_intro_is_never_re_entered_after_leaving_it():
    priors = toy_priors(floor=2)
    posteriors = np.array([one_hot(INTRO)] * 8 + [one_hot(DROP)] * 8
                          + [one_hot(INTRO)] * 8)
    spans = segments(FixedLagViterbi(priors, lag_bars=2).decode(posteriors))
    assert [span[2] for span in spans].count("intro") == 1


@pytest.mark.parametrize("lag", [0, 1, 3, 6])
def test_a_decision_is_emitted_exactly_lag_bars_after_its_own(lag):
    priors = toy_priors(floor=2)
    posteriors = np.array([one_hot(BREAKDOWN)] * 12 + [one_hot(DROP)] * 12)
    decoder = FixedLagViterbi(priors, lag_bars=lag)
    for bar, row in enumerate(posteriors):
        emitted = decoder.push(row)
        expected = [bar - lag] if bar >= lag else []
        assert [d.bar for d in emitted] == expected
    assert [d.bar for d in decoder.flush()] == list(range(len(posteriors) - lag,
                                                          len(posteriors)))


def test_an_emitted_decision_never_changes_when_more_audio_arrives():
    priors = toy_priors(floor=4)
    rng = np.random.default_rng(20260726)
    posteriors = rng.dirichlet(np.full(5, 0.6), size=64)
    full = {d.bar: d.label for d in FixedLagViterbi(priors, lag_bars=3).decode(posteriors)}

    for k in range(1, len(posteriors) + 1):
        decoder = FixedLagViterbi(priors, lag_bars=3)
        emitted = {}
        for row in posteriors[:k]:
            for decision in decoder.push(row):
                emitted[decision.bar] = decision.label
        for bar, label in emitted.items():
            assert label == full[bar], (
                f"bar {bar} was emitted as {label} after {k} bars but the full "
                f"decode says {full[bar]}")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("alpha", [0.25, 1.0])
@pytest.mark.parametrize("floor,lag", [(6, 2), (4, 3), (8, 3)])
def test_the_emitted_stream_is_itself_one_legal_path(seed, alpha, floor, lag):
    priors = toy_priors(floor=floor)
    rng = np.random.default_rng(seed)
    posteriors = rng.dirichlet(np.full(5, alpha), size=40)
    spans = segments(FixedLagViterbi(priors, lag_bars=lag).decode(posteriors))

    for before, after in zip(spans, spans[1:]):
        assert transition_allowed(before[2], after[2]), (
            f"emitted {before[2]}->{after[2]}, which is -inf in the matrix")
    for start, end, label in spans[:-1]:
        assert end - start >= int(priors.floor_bars[INDEX[label]]), (
            f"emitted a {end - start}-bar {label} run under a {floor}-bar floor")


def test_every_bar_is_decided_exactly_once_and_in_order():
    priors = toy_priors(floor=3)
    rng = np.random.default_rng(7)
    posteriors = rng.dirichlet(np.full(5, 0.5), size=37)
    decoder = FixedLagViterbi(priors, lag_bars=4)
    decisions = []
    for row in posteriors:
        decisions.extend(decoder.push(row))
    decisions.extend(decoder.flush())
    assert [d.bar for d in decisions] == list(range(len(posteriors)))


@pytest.mark.parametrize("lag", [0, 1, 3, 5])
def test_the_backtrace_ring_holds_exactly_lag_plus_one_rows(lag):
    priors = toy_priors(floor=3)
    rng = np.random.default_rng(11)
    posteriors = rng.dirichlet(np.full(5, 0.5), size=400)
    decoder = FixedLagViterbi(priors, lag_bars=lag)
    decoder.decode(posteriors)
    assert decoder.backtrace_rows == lag + 1


def test_the_ring_decodes_exactly_as_an_unbounded_backtrace_did():
    priors = toy_priors(floor=3)
    rng = np.random.default_rng(13)
    posteriors = rng.dirichlet(np.full(5, 0.4), size=120)
    boundary = rng.random(120)

    bounded = FixedLagViterbi(priors, lag_bars=2)
    decisions = bounded.decode(posteriors, boundary)

    unbounded = _UnboundedBacktrace(priors, lag_bars=2)
    reference = unbounded.decode(posteriors, boundary)

    assert [d.bar for d in decisions] == list(range(120))
    assert len(set(labels_of(decisions))) > 1
    assert decisions == reference
    assert bounded.backtrace_rows == 3
    assert len(unbounded._kept) == 120, 'the reference arm bounded itself'


class _UnboundedBacktrace(FixedLagViterbi):
    def reset(self):
        super().reset()
        self._kept: dict = {}

    def _remember(self, bar, back):
        self._kept[int(bar)] = back

    def _recall(self, bar):
        return self._kept[int(bar)]


def test_flush_is_idempotent_and_a_decoder_can_be_reset_and_reused():
    priors = toy_priors(floor=3)
    posteriors = np.array([one_hot(DROP)] * 12)
    decoder = FixedLagViterbi(priors, lag_bars=2)
    first = decoder.decode(posteriors)
    assert decoder.flush() == []
    decoder.reset()
    assert decoder.decode(posteriors) == first


def imbalanced_case():
    prior = np.array([0.12, 0.09, 0.28, 0.41, 0.10])
    row = np.zeros(5)
    row[DROP], row[BUILDUP], row[BREAKDOWN] = 0.50, 0.42, 0.08
    return toy_priors(floor=2, class_prior=prior), np.array([row] * 12)


def test_class_prior_division_recovers_a_class_the_imbalance_buries():
    priors, posteriors = imbalanced_case()
    plain = FixedLagViterbi(priors, lag_bars=2, class_prior_division=False)
    divided = FixedLagViterbi(priors, lag_bars=2, prior_strength=1.0)
    assert set(labels_of(plain.decode(posteriors))) == {"drop"}
    assert set(labels_of(divided.decode(posteriors))) == {"buildup"}


def test_prior_division_strength_scales_the_correction_in_both_directions():
    priors, posteriors = imbalanced_case()
    weak = FixedLagViterbi(priors, lag_bars=2, prior_strength=0.1)
    reversed_ = FixedLagViterbi(priors, lag_bars=2, prior_strength=-1.0)
    assert set(labels_of(weak.decode(posteriors))) == {"drop"}
    assert set(labels_of(reversed_.decode(posteriors))) == {"drop"}
    assert (reversed_._emission_bonus[DROP] - reversed_._emission_bonus[BUILDUP]
            > weak._emission_bonus[DROP] - weak._emission_bonus[BUILDUP])


def test_the_default_prior_strength_is_neutral_because_the_net_is_pre_balanced():
    """Applying the correction twice measured 71.5 % -> 39.3 % per-bar on val."""
    priors, posteriors = imbalanced_case()
    default = FixedLagViterbi(priors, lag_bars=2)
    off = FixedLagViterbi(priors, lag_bars=2, class_prior_division=False)
    assert default.decode(posteriors) == off.decode(posteriors)
    assert np.allclose(default._emission_bonus, 0.0)


def test_drop_miss_cost_buys_drop_recall_at_the_price_of_precision():
    priors = toy_priors(floor=2)
    row = np.zeros(5)
    row[BREAKDOWN], row[DROP], row[BUILDUP] = 0.53, 0.42, 0.05
    posteriors = np.array([row] * 4)
    neutral = FixedLagViterbi(priors, lag_bars=1, drop_miss_cost=1.0)
    eager = FixedLagViterbi(priors, lag_bars=1, drop_miss_cost=3.0)
    assert set(labels_of(neutral.decode(posteriors))) == {"breakdown"}
    assert set(labels_of(eager.decode(posteriors))) == {"drop"}


def test_drop_miss_cost_is_charged_once_at_the_entry_edge_not_once_per_bar():
    """Per bar instead, the cost compounds x657 over drop's 16-bar floor at 1.5."""
    priors = toy_priors(floor=4)
    neutral = FixedLagViterbi(priors, lag_bars=2, drop_miss_cost=1.0)
    eager = FixedLagViterbi(priors, lag_bars=2, drop_miss_cost=3.0)

    np.testing.assert_allclose(eager._emission_bonus, neutral._emission_bonus)

    entry = int(eager._entry_state[DROP])
    finite = np.isfinite(neutral._transition[:, entry])
    assert finite.any(), "some edge must enter drop"
    np.testing.assert_allclose(
        eager._transition[finite, entry] - neutral._transition[finite, entry],
        np.log(3.0))
    np.testing.assert_allclose(eager._log_initial[entry] - neutral._log_initial[entry],
                               np.log(3.0))

    saturated = int(eager._final_state[DROP])
    assert eager._transition[saturated, saturated] == \
        neutral._transition[saturated, saturated]


@pytest.mark.parametrize("cost", [1.0, 3.0, 20.0, 200.0, 1000.0])
def test_raising_the_cost_never_lengthens_a_drop_run(cost):
    """Per bar, cost 1000 swallowed the whole track: 41 % -> 64 % drop occupancy."""
    priors = toy_priors(floor=4)
    posteriors = np.array([one_hot(BREAKDOWN)] * 12 + [one_hot(DROP)] * 12
                          + [one_hot(BREAKDOWN)] * 12)
    spans = segments(FixedLagViterbi(priors, lag_bars=2, drop_miss_cost=cost)
                     .decode(posteriors))
    drops = [span for span in spans if span[2] == "drop"]
    assert drops == [(12, 24, "drop")], (
        f"at cost {cost} the drop covered {drops} instead of exactly the bars "
        f"the evidence supports -- is the bonus per bar?")


@pytest.mark.parametrize("spike", [10, 12, 14])
def test_boundary_hazard_sharpens_where_an_ambiguous_switch_lands(spike):
    priors = toy_priors(floor=4)
    ambiguous = np.zeros(5)
    ambiguous[BREAKDOWN], ambiguous[DROP] = 0.5, 0.5
    posteriors = np.array(
        [one_hot(BREAKDOWN)] * 9 + [ambiguous] * 8 + [one_hot(DROP)] * 9)
    boundary = np.full(len(posteriors), 0.02)
    boundary[spike] = 0.99

    decoder = FixedLagViterbi(priors, lag_bars=3, boundary_weight=6.0)
    spans = segments(decoder.decode(posteriors, boundary))
    assert [span[2] for span in spans] == ["breakdown", "drop"]
    assert spans[1][0] == spike


def test_boundary_weight_zero_ignores_the_boundary_head_entirely():
    priors = toy_priors(floor=4)
    posteriors = np.array([one_hot(BREAKDOWN)] * 12 + [one_hot(DROP)] * 12)
    boundary = np.full(len(posteriors), 0.02)
    boundary[5] = 1.0
    off = FixedLagViterbi(priors, lag_bars=3, boundary_weight=0.0)
    assert off.decode(posteriors, boundary) == off.decode(posteriors)


def test_boundary_reference_is_the_neutral_point_of_the_hazard():
    priors = toy_priors(floor=4)
    posteriors = np.array([one_hot(BREAKDOWN)] * 12 + [one_hot(DROP)] * 12)
    flat = np.full(len(posteriors), DEFAULT_BOUNDARY_REF)
    decoder = FixedLagViterbi(priors, lag_bars=3, boundary_weight=8.0)
    assert decoder.decode(posteriors, flat) == decoder.decode(posteriors)


def test_bars_with_no_usable_evidence_hold_the_last_state():
    priors = toy_priors(floor=3)
    posteriors = np.array([one_hot(BREAKDOWN)] * 10 + [np.full(5, np.nan)] * 6)
    labels = labels_of(FixedLagViterbi(priors, lag_bars=2).decode(posteriors))
    assert labels == ["breakdown"] * 16


def test_decoding_is_deterministic_and_free_of_instance_state():
    priors = toy_priors(floor=4)
    rng = np.random.default_rng(11)
    posteriors = rng.dirichlet(np.full(5, 0.4), size=50)
    boundary = rng.random(50)
    first = FixedLagViterbi(priors, lag_bars=3).decode(posteriors, boundary)
    second = FixedLagViterbi(priors, lag_bars=3).decode(posteriors, boundary)
    assert first == second


def test_an_empty_track_decodes_to_nothing():
    decoder = FixedLagViterbi(toy_priors(), lag_bars=3)
    assert decoder.decode(np.zeros((0, 5))) == []


def test_a_hazard_outside_zero_to_one_is_refused():
    priors = toy_priors()
    for bad in (0.0, 1.0, 1.5):
        broken = priors._replace(hazard=np.full(len(priors.classes), bad))
        with pytest.raises(ValueError, match="hazard"):
            FixedLagViterbi(broken, lag_bars=2)


def test_importing_the_decoder_does_not_drag_torch_onto_the_decode_path():
    """nn.dataset alone pulls torch: 1.9 s and 1,127 modules."""
    probe = (
        "import sys, training.nn.decoder;"
        "leaked = sorted(m for m in ('torch', 'training.nn.dataset')"
        "                if m in sys.modules);"
        "print(leaked, len(sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO_ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    leaked, modules = result.stdout.strip().rsplit(" ", 1)
    assert leaked == "[]", f"decode path imported {leaked}"
    assert int(modules) < 500, (
        f"a bare decoder import loaded {modules} modules -- something heavy "
        f"crept back onto the decode path")


def write_beat_csv(path, bars, beats_per_bar=4, bar_sec=2.0, t0=0.5):
    lines = ["time,downbeat,section"]
    step = bar_sec / beats_per_bar
    for bar in range(bars):
        for beat in range(beats_per_bar):
            lines.append(f"{t0 + bar * bar_sec + beat * step:.4f},{beat + 1},x")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_bar_grid_reads_downbeats_and_closes_the_final_bar(tmp_path):
    path = tmp_path / "g.beat.csv"
    write_beat_csv(path, bars=5, bar_sec=2.0, t0=0.5)
    edges = bar_grid(path)
    assert len(edges) == 6, "five bars need six edges"
    np.testing.assert_allclose(edges[:5], [0.5, 2.5, 4.5, 6.5, 8.5])
    assert edges[5] == pytest.approx(10.5), "the last bar gets a median-length span"


def test_bar_grid_refuses_a_grid_with_no_downbeats(tmp_path):
    path = tmp_path / "g.beat.csv"
    path.write_text("time,downbeat,section\n0.1,2,x\n0.6,3,x\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="downbeat"):
        bar_grid(path)


def synthetic_npz(path, classes, *, frame_sec=0.05, label_pool=2, thin_frames=4):
    pooled = len(classes)
    frames = pooled * label_pool
    label_post = np.array([one_hot(c, 0.99) for c in classes], dtype=np.float32)
    coverage = np.full(frames, 32, dtype=np.uint16)
    coverage[:thin_frames] = 1
    coverage[-thin_frames:] = 1
    np.savez(
        path,
        label_post=label_post,
        boundary=np.full(frames, 0.05, dtype=np.float32),
        coverage=coverage,
        frame_sec=np.float64(frame_sec),
        t0=np.float64(frame_sec),
        label_frame_sec=np.float64(frame_sec * label_pool),
        label_t0=np.float64(frame_sec * label_pool),
        label_pool=np.int32(label_pool),
    )


def test_bar_observations_average_the_bar_and_drop_edge_only_frames(tmp_path):
    npz = tmp_path / "t.npz"
    synthetic_npz(npz, [BREAKDOWN] * 100 + [DROP] * 100, thin_frames=4)
    edges = np.arange(0.0, 20.1, 2.0)
    posteriors, boundary = bar_observations(npz, edges, min_coverage=2)
    assert posteriors.shape == (10, 5)
    assert not np.isnan(posteriors).any(), "no bar is edge-only in this fixture"
    assert posteriors[:10, BREAKDOWN].argmax() == 0
    assert posteriors[5:].argmax(axis=1).tolist() == [DROP] * 5
    assert boundary.shape == (10,)


def test_bar_observations_flag_a_bar_that_only_edge_frames_reach(tmp_path):
    npz = tmp_path / "t.npz"
    synthetic_npz(npz, [DROP] * 100, thin_frames=44)
    edges = np.arange(0.0, 20.1, 2.0)
    posteriors, _boundary = bar_observations(npz, edges, min_coverage=2)
    assert np.isnan(posteriors[0]).all(), "first bar is inside the unread edge"
    assert np.isnan(posteriors[-1]).all()
    assert not np.isnan(posteriors[3]).any()


def test_decode_track_end_to_end_returns_bar_stamped_labels(tmp_path):
    npz = tmp_path / "t.npz"
    beats = tmp_path / "t.beat.csv"
    synthetic_npz(npz, [BREAKDOWN] * 100 + [DROP] * 100, thin_frames=4)
    write_beat_csv(beats, bars=10, bar_sec=2.0, t0=0.0)

    priors = toy_priors(floor=3)
    params = DecodeParams(lag_bars=2, boundary_weight=0.0)
    timeline = decode_track(npz, beats, params, priors=priors)

    assert [t for t, _ in timeline] == pytest.approx(list(np.arange(0.0, 20.0, 2.0)))
    assert [label for _, label in timeline] == ["breakdown"] * 5 + ["drop"] * 5
    assert decode_track(npz, beats, params, priors=priors) == timeline


def test_segments_run_length_encodes_a_decision_stream():
    priors = toy_priors(floor=2)
    posteriors = np.array([one_hot(BREAKDOWN)] * 6 + [one_hot(DROP)] * 6)
    spans = segments(FixedLagViterbi(priors, lag_bars=1).decode(posteriors))
    assert spans == [(0, 6, "breakdown"), (6, 12, "drop")]


def test_a_config_naming_a_knob_the_decoder_lacks_is_refused(tmp_path):
    path = tmp_path / "decoder_config.json"
    path.write_text(json.dumps({"chosen": {"lag_bars": 2, "tempurature": 0.5}}))
    with pytest.raises(ValueError, match="tempurature"):
        load_decoder_config(path)


def test_a_config_of_known_knobs_round_trips(tmp_path):
    path = tmp_path / "decoder_config.json"
    path.write_text(json.dumps({"chosen": {"lag_bars": 2, "temperature": 0.5,
                                           "floor_bars": [1, 2, 3, 4, 5]}}))
    params = load_decoder_config(path)
    assert params.lag_bars == 2
    assert params.temperature == 0.5
    assert params.floor_bars == (1, 2, 3, 4, 5)


def test_the_entry_bonus_in_the_config_equals_the_same_bonus_premultiplied():
    """The two expressions of the entry preference must agree, because BOTH
    exist on disk and applying both would double it.

    The l9c campaign measured every candidate with the bonus premultiplied into
    the priors' ->buildup column (`priors_DAE1.json`, whose transition rows
    therefore do NOT sum to 1).  The generation ships the FITTED priors and
    states the bonus as a config knob instead.  That is only legitimate if the
    two produce the same trellis -- otherwise the shipped chain is not the one
    the campaign measured -- and it is only safe while exactly one of them is
    in force at a time.
    """
    bonus = 1.0
    plain = toy_priors()
    index = plain.classes.index("buildup")
    premultiplied = np.array(plain.transition, dtype=np.float64, copy=True)
    premultiplied[:, index] *= math.exp(bonus)
    premultiplied = plain._replace(transition=premultiplied)

    knob = FixedLagViterbi(plain, lag_bars=2, buildup_entry_bonus=bonus)
    baked = FixedLagViterbi(premultiplied, lag_bars=2, buildup_entry_bonus=0.0)

    for name in ("_transition", "_cold_initial"):
        ours, theirs = getattr(knob, name), getattr(baked, name)
        # A finite bonus may not lift a structurally forbidden edge, so the
        # -inf pattern is part of the claim rather than a detail of it.
        assert (np.isinf(ours) == np.isinf(theirs)).all(), name
        finite = np.isfinite(ours) & np.isfinite(theirs)
        assert np.abs(ours[finite] - theirs[finite]).max() < 1e-12, name


def test_the_shipped_priors_carry_no_premultiplied_entry_bonus(nn_artifacts):
    """The doubling guard, read off the artifact the show actually loads.

    A priors file with the bonus already in it has a ->buildup column scaled by
    exp(bonus), so its transition rows stop summing to 1.  The config knob is
    non-zero, so normalised rows are what says the preference is applied once.
    """
    from lib.section_chain import artifacts

    priors = Priors.load(artifacts().priors)
    rows = np.asarray(priors.transition, dtype=np.float64).sum(axis=1)
    assert np.abs(rows[rows > 0] - 1.0).max() < 1e-9


def test_the_shipping_config_loads_and_is_the_l9d_point():
    """The l9d point, per the file's own provenance block, and it is NOT a
    sweep result.  The base is arm N's sweep pick; two knobs are moved on top
    of it.  buildup_entry_bonus 1.0 is a decision bias of the same family as
    drop_miss_cost -- chosen on 12 probe arcs, never validated on val, and the
    config says so rather than implying a sweep.  The buildup floor sits at 3
    bars against the sweep pick's 2: the campaign's own candidate was a
    hand-set 4, which the axis was later measured against and found exactly
    neutral -- floor 3 matches it on climbs, drop coverage and deficit and reads
    slightly better on crispness, so it is taken because it costs nothing and
    retires a hand-set value for the swept-adjacent one.  floor_bars is an explicit
    vector on purpose -- it overrides floor_scale, so a priors refit does not
    move these floors."""
    params = load_decoder_config(SHIPPING_DECODER_CONFIG)
    document = json.loads(SHIPPING_DECODER_CONFIG.read_text())
    assert document["name"] == "l9d_buildup_entry_1nat_floor3"
    assert dataclasses.asdict(params) == {
        "lag_bars": 2,
        "class_prior_division": True,
        "prior_strength": 0.0,
        "drop_miss_cost": 10.0,
        "boundary_weight": 4.0,
        "boundary_ref": 0.1,
        "boundary_tolerance_sec": 0.5,
        "min_coverage": 1,
        "floor_scale": 1.0,
        "floor_bars": (4, 4, 3, 4, 4, 8, 4, 4, 2),
        "outro_escape": 0.04,
        "temperature": 1.0,
        "buildup_drop_bonus": 0.0,
        "buildup_entry_bonus": 1.0,
    }


def test_the_shipping_config_names_every_knob_the_decoder_has():
    chosen = json.loads(SHIPPING_DECODER_CONFIG.read_text())["chosen"]
    assert set(chosen) == {f.name for f in dataclasses.fields(DecodeParams)}


def test_a_floor_vector_off_disk_is_a_tuple_so_the_record_stays_hashable():
    params = DecodeParams(floor_bars=[8, 8, 6, 9, 8])
    assert params.floor_bars == (8, 8, 6, 9, 8)
    assert hash(params)
    assert DecodeParams(floor_bars=[8, 8, 6, 9, 8]) == params


def test_a_floor_vector_overrides_the_scalar_class_by_class():
    priors = toy_priors(floor=4)
    scaled = FixedLagViterbi(priors, floor_scale=2.0)
    vector = FixedLagViterbi(priors, floor_scale=2.0, floor_bars=(1, 2, 3, 4, 5))
    assert scaled._floors.tolist() == [8] * 5
    assert vector._floors.tolist() == [1, 2, 3, 4, 5]


def test_a_floor_vector_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="floor_bars has 4 entries"):
        FixedLagViterbi(toy_priors(), floor_bars=(1, 2, 3, 4))


def test_the_floor_vector_sets_the_minimum_run_length_class_by_class():
    priors = toy_priors(floor=8, hazard=0.3)
    posteriors = np.array([one_hot(BREAKDOWN)] * 8 + [one_hot(DROP)] * 2
                          + [one_hot(BREAKDOWN)] * 10)

    long_floors = FixedLagViterbi(priors, lag_bars=0, boundary_weight=0.0)
    short_floors = FixedLagViterbi(priors, lag_bars=0, boundary_weight=0.0,
                                   floor_bars=(2, 2, 2, 2, 2))
    assert labels_of(long_floors.decode(posteriors)).count("drop") == 8
    assert labels_of(short_floors.decode(posteriors)).count("drop") == 2


def test_outro_is_terminal_until_an_escape_is_opened():
    priors = toy_priors(floor=2, hazard=0.3)
    posteriors = np.array([one_hot(OUTRO)] * 4 + [one_hot(DROP)] * 12)

    terminal = FixedLagViterbi(priors, lag_bars=0, boundary_weight=0.0)
    escaping = FixedLagViterbi(priors, lag_bars=0, boundary_weight=0.0,
                               outro_escape=0.2)
    assert set(labels_of(terminal.decode(posteriors))) == {"outro"}
    assert "drop" in labels_of(escaping.decode(posteriors))


def test_a_zero_escape_reproduces_the_terminal_decoder_exactly():
    priors = toy_priors(floor=2, hazard=0.3)
    posteriors = np.array([one_hot(OUTRO)] * 4 + [one_hot(DROP)] * 12)
    assert (labels_of(FixedLagViterbi(priors, lag_bars=0, outro_escape=0.0)
                      .decode(posteriors))
            == labels_of(FixedLagViterbi(priors, lag_bars=0).decode(posteriors)))


@pytest.mark.parametrize("escape", [0.5, 0.6, -0.01])
def test_an_escape_that_leaves_no_probability_to_stay_is_refused(escape):
    with pytest.raises(ValueError, match="outro_escape must lie"):
        FixedLagViterbi(toy_priors(), outro_escape=escape)


def test_the_escape_only_opens_the_two_classes_a_track_can_resume_into():
    decoder = FixedLagViterbi(toy_priors(floor=2), outro_escape=0.2)
    source = int(decoder._final_state[decoder.classes.index("outro")])
    reachable = {decoder.classes[int(decoder._state_class[target])]
                 for target in np.flatnonzero(np.isfinite(decoder._transition[source]))}
    assert reachable == {"outro", "breakdown", "drop"}


def intro_then_bars_that_shade_breakdown(ambiguity=0.02):
    """Intro, then bars where breakdown edges out buildup by a hair.

    A neutral decoder leaves intro for breakdown; only a preference on the
    edges *into* buildup can send it the other way, which is the knob's own
    case in miniature.
    """
    tail = np.full(5, 0.02)
    tail[BREAKDOWN] = 0.47 + ambiguity / 2
    tail[BUILDUP] = 0.47 - ambiguity / 2
    tail /= tail.sum()
    return np.asarray([one_hot(INTRO)] * 4 + [tail] * 8)


def entry_bonus_pair(bonus, floor=2):
    """A neutral and a boosted trellis, and what the bonus added between them.

    ``-inf`` minus ``-inf`` is nan rather than zero, so a forbidden edge that
    stayed forbidden reads as no change -- which is what it is.
    """
    neutral = FixedLagViterbi(toy_priors(floor=floor), lag_bars=2)
    boosted = FixedLagViterbi(toy_priors(floor=floor), lag_bars=2,
                              buildup_entry_bonus=bonus)
    with np.errstate(invalid="ignore"):
        delta = boosted._transition - neutral._transition
    return neutral, boosted, np.where(np.isnan(delta), 0.0, delta)


def test_the_default_entry_bonus_is_bit_identically_the_knobless_decoder(monkeypatch):
    """The neutral point has to be a genuine no-op, not an approximate one.

    ``0.0`` is what every config cut before the knob existed decodes with, so
    a float that merely rounds back to the same trellis would silently re-cut
    the whole committed baseline.
    """
    posteriors = intro_then_bars_that_shade_breakdown()
    reference = FixedLagViterbi(toy_priors(floor=1), lag_bars=2)

    monkeypatch.setattr(FixedLagViterbi, "_apply_buildup_entry_bonus",
                        lambda self, transition, switch: None)
    knobless = FixedLagViterbi(toy_priors(floor=1), lag_bars=2)
    monkeypatch.undo()

    assert DecodeParams().buildup_entry_bonus == 0.0
    assert reference._transition.tobytes() == knobless._transition.tobytes()
    assert reference._cold_initial.tobytes() == knobless._cold_initial.tobytes()
    assert (labels_of(reference.decode(posteriors))
            == labels_of(knobless.decode(posteriors)))


def test_the_entry_bonus_lifts_every_legal_edge_into_buildup_by_exactly_itself():
    """From ANY source class, not only from the one that motivated it.

    The knob is a preference for the class, not for one particular approach to
    it, so intro -> buildup and drop -> buildup have to move exactly as far as
    breakdown -> buildup does.
    """
    neutral, _, delta = entry_bonus_pair(0.7)
    target = int(neutral._entry_state[BUILDUP])

    moved = np.flatnonzero(delta.any(axis=1))
    assert delta[moved, target] == pytest.approx(0.7)
    assert {neutral.classes[int(neutral._state_class[state])] for state in moved} == {
        "intro", "breakdown", "drop"}

    delta[moved, target] = 0.0
    assert not delta.any()


def test_an_edge_the_graph_forbids_into_buildup_stays_forbidden():
    """A finite bonus cannot lift a -inf, and must not be allowed to try.

    Nothing leaves the outro family, so outro -> buildup is structural, and a
    knob that could open it would be re-writing the graph rather than
    weighting it.
    """
    neutral, boosted, _ = entry_bonus_pair(6.0)
    source = int(neutral._final_state[OUTRO])
    target = int(neutral._entry_state[BUILDUP])
    assert neutral._transition[source, target] == -np.inf
    assert boosted._transition[source, target] == -np.inf


def test_the_entry_bonus_leaves_the_cold_start_alone():
    """The subtle one, and the reason the knob sits on the switch edges.

    ``_cold_initial`` is ``log_initial + _entry_bonus``, and a track is allowed
    to open in buildup -- that entry is finite.  Routing the preference through
    ``_entry_bonus`` would therefore also move where a track may BEGIN, which
    is a different decoder from the one this value was measured on.
    """
    neutral, boosted, _ = entry_bonus_pair(6.0)
    entry = int(neutral._entry_state[BUILDUP])
    assert np.isfinite(neutral._cold_initial[entry])
    assert boosted._cold_initial.tobytes() == neutral._cold_initial.tobytes()
    assert boosted._entry_bonus.tobytes() == neutral._entry_bonus.tobytes()

    neutral.reset()
    boosted.reset()
    assert boosted._log_initial.tobytes() == neutral._log_initial.tobytes()


def test_the_entry_bonus_resolves_an_ambiguous_exit_toward_buildup():
    posteriors = intro_then_bars_that_shade_breakdown()
    neutral = FixedLagViterbi(toy_priors(floor=1), lag_bars=2)
    boosted = FixedLagViterbi(toy_priors(floor=1), lag_bars=2,
                              buildup_entry_bonus=1.0)

    neutral_labels = labels_of(neutral.decode(posteriors))
    boosted_labels = labels_of(boosted.decode(posteriors))
    assert "buildup" not in neutral_labels and "breakdown" in neutral_labels
    assert "buildup" in boosted_labels
    # The cold start is untouched, so the track still OPENS where the evidence says.
    assert boosted_labels[0] == "intro"


def test_a_non_finite_entry_bonus_is_refused():
    with pytest.raises(ValueError, match="buildup_entry_bonus"):
        FixedLagViterbi(toy_priors(), buildup_entry_bonus=float("nan"))
    with pytest.raises(ValueError, match="buildup_entry_bonus"):
        FixedLagViterbi(toy_priors(), buildup_entry_bonus=float("inf"))


def test_the_entry_bonus_round_trips_through_a_config_and_a_typo_still_raises(tmp_path):
    path = tmp_path / "decoder_config.json"
    path.write_text(json.dumps({"chosen": {"lag_bars": 2,
                                           "buildup_entry_bonus": 1.0}}))
    assert load_decoder_config(path).buildup_entry_bonus == 1.0

    # A config cut before the knob existed still loads, at the neutral point.
    old = tmp_path / "old_config.json"
    old.write_text(json.dumps({"chosen": {"lag_bars": 2}}))
    assert load_decoder_config(old).buildup_entry_bonus == 0.0

    typo = tmp_path / "typo_config.json"
    typo.write_text(json.dumps({"chosen": {"buildup_entrance_bonus": 1.0}}))
    with pytest.raises(ValueError, match="buildup_entrance_bonus"):
        load_decoder_config(typo)


def test_temper_is_the_identity_at_one_and_returns_the_same_array():
    post = np.array([[0.7, 0.1, 0.1, 0.05, 0.05]])
    assert temper(post, 1.0) is post


def test_a_cold_temperature_sharpens_and_a_hot_one_flattens():
    post = np.array([[0.6, 0.1, 0.1, 0.1, 0.1]])
    assert temper(post, 0.5)[0, 0] > post[0, 0]
    assert temper(post, 2.0)[0, 0] < post[0, 0]
    for temperature in (0.5, 2.0):
        assert temper(post, temperature).sum() == pytest.approx(1.0)


def test_tempering_never_moves_the_argmax():
    rng = np.random.default_rng(7)
    post = rng.dirichlet(np.ones(5), size=64)
    for temperature in (0.25, 0.5, 2.0, 4.0):
        assert (temper(post, temperature).argmax(axis=1) == post.argmax(axis=1)).all()


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_a_non_positive_temperature_is_refused(temperature):
    with pytest.raises(ValueError, match="temperature must be > 0"):
        temper(np.array([[0.2] * 5]), temperature)


def test_temperature_reaches_the_bar_average_through_bar_observations(tmp_path):
    npz = tmp_path / "t.npz"
    synthetic_npz(npz, [BREAKDOWN] * 100 + [DROP] * 100, thin_frames=4)
    edges = np.arange(0.0, 20.1, 2.0)
    neutral, _ = bar_observations(npz, edges, min_coverage=2)
    hot, _ = bar_observations(npz, edges, min_coverage=2, temperature=8.0)
    assert hot[0, BREAKDOWN] < neutral[0, BREAKDOWN]


def test_a_sidecar_where_no_frame_clears_the_coverage_threshold_raises(tmp_path):
    npz = tmp_path / "t.npz"
    synthetic_npz(npz, [DROP] * 20, thin_frames=40)
    edges = np.arange(0.0, 4.1, 2.0)
    with pytest.raises(RuntimeError, match="no frame has coverage"):
        bar_observations(npz, edges, min_coverage=2)
    assert not np.isnan(bar_observations(npz, edges, min_coverage=1)[0]).all()


def test_decode_track_carries_every_knob_the_config_names(tmp_path):
    npz = tmp_path / "t.npz"
    beats = tmp_path / "t.beat.csv"
    synthetic_npz(npz, [BREAKDOWN] * 80 + [DROP] * 40 + [BREAKDOWN] * 80,
                  thin_frames=4)
    write_beat_csv(beats, bars=10, bar_sec=2.0, t0=0.0)
    priors = toy_priors(floor=8, hazard=0.3)

    long_floors = DecodeParams(lag_bars=0, boundary_weight=0.0)
    short_floors = DecodeParams(lag_bars=0, boundary_weight=0.0,
                                floor_bars=(2, 2, 2, 2, 2))
    decoded_long = [label for _, label in
                    decode_track(npz, beats, long_floors, priors=priors)]
    decoded_short = [label for _, label in
                     decode_track(npz, beats, short_floors, priors=priors)]
    assert decoded_long.count("drop") != decoded_short.count("drop")
    assert decoded_short.count("drop") == 2


def rows_npz(path, rows, *, frame_sec, label_pool, label_t0, thin_frames=0):
    label_post = np.asarray(rows, dtype=np.float32)
    frames = len(rows) * label_pool
    coverage = np.full(frames, 32, dtype=np.uint16)
    if thin_frames:
        coverage[:thin_frames] = 1
        coverage[-thin_frames:] = 1
    np.savez(
        path,
        label_post=label_post,
        boundary=np.full(frames, 0.05, dtype=np.float32),
        coverage=coverage,
        frame_sec=np.float64(frame_sec),
        t0=np.float64(frame_sec),
        label_frame_sec=np.float64(frame_sec * label_pool),
        label_t0=np.float64(label_t0),
        label_pool=np.int32(label_pool),
    )


def graded_bars(bars):
    loud = np.full(5, 0.01)
    loud[DROP] = 0.96
    quiet = np.full(5, (1.0 - 0.02 - 0.25) / 3.0)
    quiet[DROP], quiet[BREAKDOWN] = 0.02, 0.25
    return [row for _ in range(bars) for row in (loud, quiet, quiet, quiet)]


def test_decode_track_forwards_the_temperature_to_the_bar_average(tmp_path):
    npz, beats = tmp_path / "t.npz", tmp_path / "t.beat.csv"
    rows_npz(npz, graded_bars(16), frame_sec=0.25,
             label_pool=2, label_t0=0.5)
    write_beat_csv(beats, bars=16, bar_sec=2.0, t0=0.5)
    priors = toy_priors(floor=2, hazard=0.3)

    neutral = DecodeParams(lag_bars=0, boundary_weight=0.0, min_coverage=1)
    hot = dataclasses.replace(neutral, temperature=8.0)
    assert (set(label for _, label in decode_track(npz, beats, neutral, priors=priors))
            == {"drop"})
    assert (set(label for _, label in decode_track(npz, beats, hot, priors=priors))
            == {"breakdown"})


def test_decode_track_forwards_the_outro_escape_to_the_trellis(tmp_path):
    npz, beats = tmp_path / "t.npz", tmp_path / "t.beat.csv"
    synthetic_npz(npz, [OUTRO] * 80 + [DROP] * 240, thin_frames=4)
    write_beat_csv(beats, bars=16, bar_sec=2.0, t0=0.0)
    priors = toy_priors(floor=2, hazard=0.3)

    terminal = DecodeParams(lag_bars=0, boundary_weight=0.0, min_coverage=1)
    escaping = dataclasses.replace(terminal, outro_escape=0.2)
    assert (set(label for _, label in decode_track(npz, beats, terminal, priors=priors))
            == {"outro"})
    assert "drop" in [label for _, label in
                      decode_track(npz, beats, escaping, priors=priors)]


NON_DEFAULT_TRELLIS_KNOBS = {
    "lag_bars": 1,
    "class_prior_division": False,
    "prior_strength": 0.4,
    "drop_miss_cost": 3.0,
    "boundary_weight": 5.0,
    "boundary_ref": 0.2,
    "floor_scale": 2.0,
    "floor_bars": (1, 2, 3, 4, 5),
    "outro_escape": 0.05,
    "buildup_drop_bonus": 0.75,
}


def test_every_decode_knob_is_claimed_by_exactly_one_stage():
    params = DecodeParams()
    trellis, observation = set(trellis_knobs(params)), set(observation_knobs(params))
    assert not trellis & observation
    assert trellis | observation == {f.name for f in dataclasses.fields(DecodeParams)}
    # Both halves must be placeable, or the partition merely moves the drop:
    # a trellis knob lands as an unexpected keyword, an observation knob here.
    assert observation <= set(inspect.signature(bar_observations).parameters)


def test_build_decoder_carries_every_trellis_knob_onto_the_decoder():
    params = DecodeParams(**NON_DEFAULT_TRELLIS_KNOBS)
    assert set(NON_DEFAULT_TRELLIS_KNOBS) == set(trellis_knobs(params))
    decoder = build_decoder(toy_priors(), params)
    for name, value in trellis_knobs(params).items():
        assert getattr(decoder, name) == value, name


def _repo_sources(root):
    """The repository's own python, which is not everything under these names.

    `training/data` is the gitignored corpus, and it holds ops copies of
    campaign scripts and whole shadow trees of a vendored decoder.  Those are
    data this machine happens to have, not source this repository ships, so a
    rule about the source must not read them -- and reading them made the gate
    below a statement about whether the corpus was downloaded.
    """
    corpus = root / "training" / "data"
    for directory in ("lib", "simulate", "training"):
        for path in (root / directory).rglob("*.py"):
            if corpus in path.parents:
                continue
            yield path


def test_only_one_place_builds_the_trellis():
    """Four hand-written call sites once; buildup_drop_bonus reached three.

    A second construction site is a second list of knobs to keep in step, which
    is how a decoder came to ignore part of the config it was handed.  Tests are
    exempt: they exercise the constructor's own arguments deliberately.
    """
    root = Path(__file__).resolve().parents[1]
    shared = root / "training" / "nn" / "decoder.py"
    offenders = []
    for path in _repo_sources(root):
        if path == shared:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders += [f"{path.relative_to(root)}:{node.lineno}"
                      for node in ast.walk(tree)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name)
                      and node.func.id == "FixedLagViterbi"]
    assert offenders == [], (
        f"build_decoder is the one way to build one; {offenders} hand-list the "
        f"knobs and can fall behind DecodeParams")


def test_the_source_walk_skips_the_gitignored_corpus(tmp_path):
    """440 corpus files were being parsed, and two of them do not parse at all.

    On a machine that has the corpus the gate above died in `ast.parse` on an
    ops copy carrying a BOM -- so a rule about this repository's source failed
    for a reason that is a fact about the download.  Every offender the walk
    reported was a shadow tree's vendored decoder, none of it ours.
    """
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "ours.py").write_text("x = 1\n", encoding="utf-8")
    corpus = tmp_path / "training" / "data" / "raveform" / "models" / "campaign"
    corpus.mkdir(parents=True)
    (corpus / "ops_copy.py").write_text(
        '﻿"""an ops copy with a BOM"""\nFixedLagViterbi(1)\n', encoding="utf-8")

    assert [p.name for p in _repo_sources(tmp_path)] == ["ours.py"]
