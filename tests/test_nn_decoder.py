"""Tests for the section priors (``training/nn/priors.py``) and the fixed-lag
Viterbi decoder (``training/nn/decoder.py``).

The decoder is the *committer*: it is the only component allowed to say what
the lights are doing, and once it has said it, the statement is final.  Three
families of property live here, and each one is a promise the rest of the
system is built on.

**Structure is not advice.**  ``anything -> intro`` and ``outro -> anything``
are not preferences the evidence may outvote; they are -inf.  The old engine
implemented the same graph as a veto that *held the current state* when a jump
was illegal.  The decoder instead picks the best **legal** path, which is a
different and better answer -- but only if the illegal edges genuinely never
appear in the output, under any posterior, including adversarial ones.

**Immutability is the product.**  A light show cannot un-fire a strobe.  A
fixed-lag decoder that re-reads its own backtrace every bar would silently
revise decisions already sent to the rig, so the decision for bar B is pruned
into the trellis the moment it is emitted: every surviving path must agree with
it.  The test for that is not "the output looks stable", it is "feeding more
future audio never changes an already-emitted bar" -- checked against prefixes.

**Stickiness comes from the duration model, not from a smoother.**  A single
anomalous bar must not flip the show, and a genuine 32-bar drop must not be
chopped into four 8-bar ones.  Both are the same min-floor + geometric-tail
prior, so both are tested against the same synthetic posteriors: flicker in,
one contiguous run out; a real switch in, one clean switch out.

Everything here runs on synthetic posteriors and hand-built run sequences.  The
corpus lives in a gitignored data directory that CI does not have, so nothing
in this file reads it -- the corpus-fitting entry point is a thin I/O wrapper
around ``fit_runs``, which is pure and tested directly.
"""
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from build_training_table import V1_ORDER  # noqa: E402
from nn.decoder import (  # noqa: E402
    DEFAULT_BOUNDARY_REF,
    SHIPPING_DECODER_CONFIG,
    DecodeParams,
    FixedLagViterbi,
    bar_grid,
    bar_observations,
    decode_track,
    load_decoder_config,
    segments,
    temper,
)
from nn.priors import (  # noqa: E402
    PRIORS_FILE,
    Priors,
    bar_runs,
    fit_runs,
    transition_allowed,
    v1_runs,
)

INTRO, BUILDUP, BREAKDOWN, DROP, OUTRO = range(5)
INDEX = {label: i for i, label in enumerate(V1_ORDER)}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def corpus_runs():
    """A miniature corpus with the real structural shape.

    Intro only ever first, outro only ever last, and a lopsided buildup fork
    (4:1 towards breakdown) so the near-uniform override has something to
    override.  Bar counts are chosen so the percentiles are exact integers and
    can be asserted by hand.
    """
    return [
        [("intro", 16), ("buildup", 16), ("drop", 32), ("breakdown", 16), ("outro", 8)],
        [("intro", 16), ("buildup", 24), ("breakdown", 8), ("drop", 32), ("outro", 16)],
        [("intro", 24), ("buildup", 8), ("breakdown", 16), ("drop", 48), ("outro", 8)],
        [("intro", 8), ("buildup", 32), ("breakdown", 24), ("drop", 16), ("outro", 24)],
        [("intro", 32), ("buildup", 16), ("breakdown", 32), ("drop", 32), ("outro", 16)],
    ]


def toy_priors(floor=4, hazard=0.25, class_prior=None, initial=None):
    """Uniform-ish priors with a small floor, for short synthetic sequences.

    Real floors are 8-16 bars; a 60-bar test would then have room for three
    runs.  The floor is a parameter of the decoder's behaviour, not of its
    correctness, so the properties are exercised at a size that fits in a test.
    """
    classes = V1_ORDER
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
    """A confident posterior row: ``strength`` on ``index``, rest uniform."""
    row = np.full(n, (1.0 - strength) / (n - 1))
    row[index] = strength
    return row


def labels_of(decisions):
    return [d.label for d in decisions]


# --------------------------------------------------------------------------- #
# Priors: structure
# --------------------------------------------------------------------------- #


def test_transition_rule_matches_the_corpus_structural_facts():
    """The -inf graph is exactly: no re-entry to intro, no exit from outro, no
    self-loop (persistence is the duration model's job, not the matrix's)."""
    for src in V1_ORDER:
        assert not transition_allowed(src, "intro")
        assert not transition_allowed("outro", src)
        assert not transition_allowed(src, src)
    assert transition_allowed("buildup", "drop")
    assert transition_allowed("drop", "breakdown")
    assert transition_allowed("breakdown", "outro")


def test_fitted_transition_rows_are_stochastic_and_illegal_entries_are_zero():
    priors = fit_runs(corpus_runs())
    for i, src in enumerate(V1_ORDER):
        row = priors.transition[i]
        for j, dst in enumerate(V1_ORDER):
            if not transition_allowed(src, dst):
                assert row[j] == 0.0, f"{src}->{dst} must be structurally impossible"
        if src == "outro":
            assert row.sum() == 0.0, "outro has no legal successor at all"
        else:
            assert row.sum() == pytest.approx(1.0)


def test_log_transition_is_minus_inf_exactly_where_the_probability_is_zero():
    priors = fit_runs(corpus_runs())
    zero = priors.transition == 0.0
    assert np.all(np.isneginf(priors.log_transition[zero]))
    assert np.all(np.isfinite(priors.log_transition[~zero]))


def test_buildup_fork_is_forced_near_uniform_despite_a_lopsided_corpus():
    """The corpus fork is ~0.15 nats of information; the look-ahead evidence
    decides breakdown vs drop, not the prior.  A 4:1 sample must still come out
    even, with the *combined* fork mass preserved."""
    runs = [[("intro", 16), ("buildup", 16), ("breakdown", 16), ("outro", 16)]] * 8
    runs += [[("intro", 16), ("buildup", 16), ("drop", 16), ("outro", 16)]] * 2
    priors = fit_runs(runs)
    row = priors.transition[INDEX["buildup"]]
    assert row[INDEX["breakdown"]] == pytest.approx(row[INDEX["drop"]])
    combined = row[INDEX["breakdown"]] + row[INDEX["drop"]]
    assert combined > 0.9, "the fork should still hold nearly all of buildup's mass"


def test_a_legal_but_unobserved_transition_keeps_a_little_mass():
    """intro->outro never happens in this corpus and is still not impossible: a
    decoder that could not represent it would hold intro forever on a track
    that opens straight into its outro."""
    priors = fit_runs(corpus_runs())
    assert priors.transition[INDEX["intro"], INDEX["outro"]] > 0.0


def test_fitting_refuses_a_corpus_that_contradicts_the_structural_graph():
    """The -inf entries are a claim about the data.  If a later corpus revision
    contains an outro that is not terminal, the claim is wrong and the fit must
    say so rather than quietly discard the evidence."""
    bad = corpus_runs() + [[("intro", 16), ("outro", 16), ("drop", 16)]]
    with pytest.raises(RuntimeError, match="outro->drop"):
        fit_runs(bad)
    relaxed = fit_runs(bad, strict=False)
    assert relaxed.corpus["illegal_observed"]["outro->drop"] == 1
    assert relaxed.transition[INDEX["outro"], INDEX["drop"]] == 0.0


def test_initial_distribution_is_fitted_not_assumed():
    """Intro is pure-*initial*, but the first run is not always intro (one train
    track opens on a drop), so the initial vector is fitted and smoothed."""
    priors = fit_runs(corpus_runs() * 8)
    assert priors.initial.sum() == pytest.approx(1.0)
    assert priors.initial.argmax() == INDEX["intro"]
    assert priors.initial[INDEX["intro"]] > 0.9
    assert np.all(priors.initial > 0.0), "no opening is impossible, only unlikely"


# --------------------------------------------------------------------------- #
# Priors: duration and occupancy
# --------------------------------------------------------------------------- #


def test_duration_floor_is_the_corpus_fifth_percentile_in_bars():
    priors = fit_runs(corpus_runs())
    drop_bars = sorted([32, 32, 48, 16, 32])
    expected = max(1, int(round(float(np.percentile(drop_bars, 5.0)))))
    assert priors.floor_bars[INDEX["drop"]] == expected


def test_duration_tail_is_the_geometric_that_halves_at_the_corpus_median():
    """"Widened per spec" means the tail is memoryless above the floor -- a
    constant per-bar hazard, no peak the evidence has to fight -- pinned only by
    the corpus median residual length."""
    priors = fit_runs(corpus_runs())
    index = INDEX["drop"]
    floor = int(priors.floor_bars[index])
    median = float(np.median([32, 32, 48, 16, 32]))
    residual = max(1.0, median - floor)
    assert priors.hazard[index] == pytest.approx(1.0 - 0.5 ** (1.0 / residual))
    # survival at the median residual is exactly one half
    assert (1.0 - priors.hazard[index]) ** residual == pytest.approx(0.5)


def test_floor_is_at_least_one_bar_even_for_a_degenerate_class():
    priors = fit_runs([[("intro", 1), ("drop", 1), ("outro", 1)]])
    assert np.all(priors.floor_bars >= 1)
    assert np.all(priors.hazard > 0.0)
    assert np.all(priors.hazard <= 1.0)


def test_class_prior_is_bar_occupancy_not_run_count():
    """Class-prior division corrects the imbalance the decoder actually sees,
    and the decoder sees bars.  Breakdown occurs as often as drop by run count
    here but occupies far fewer bars."""
    runs = [[("intro", 8), ("breakdown", 8), ("drop", 64), ("outro", 8)]] * 4
    priors = fit_runs(runs)
    assert priors.class_prior.sum() == pytest.approx(1.0)
    assert priors.class_prior[INDEX["drop"]] == pytest.approx(64 / 88)
    assert priors.class_prior[INDEX["breakdown"]] == pytest.approx(8 / 88)


def test_priors_json_round_trips_exactly(tmp_path):
    priors = fit_runs(corpus_runs())
    path = tmp_path / PRIORS_FILE
    priors.save(path)
    again = Priors.load(path)
    assert again.classes == priors.classes
    for field in ("initial", "transition", "floor_bars", "hazard", "class_prior"):
        np.testing.assert_array_equal(getattr(again, field), getattr(priors, field))
    # Same content -> same bytes: the priors file is an input to every decode.
    second = tmp_path / "again.json"
    again.save(second)
    assert path.read_bytes() == second.read_bytes()


def test_priors_file_is_plain_json_with_no_infinities():
    """-inf is carried as a probability of exactly zero, not as a JSON literal:
    ``Infinity`` is a Python extension that no other reader has to accept."""
    priors = fit_runs(corpus_runs())
    text = json.dumps(priors.to_dict())
    assert "Infinity" not in text and "NaN" not in text


# --------------------------------------------------------------------------- #
# Priors: corpus adapters
# --------------------------------------------------------------------------- #


def test_v1_runs_folds_and_merges_across_a_dropped_sentinel():
    sections = [
        (0.0, 10.0, "intro"),
        (10.0, 20.0, "drop"),
        (20.0, 30.0, "breakdown"),
        (30.0, 40.0, "cooldown"),      # folds to breakdown -> merges with above
        (40.0, 50.0, "altoutro"),      # folds to outro
        (50.0, 55.0, "end"),           # sentinel, dropped
    ]
    assert [run[2] for run in v1_runs(sections)] == [
        "intro", "drop", "breakdown", "outro"]


def test_bar_runs_counts_downbeats_and_never_reattributes_dropped_time():
    downbeats = np.arange(0.0, 40.0, 2.0)     # a bar every 2 s
    sections = [
        (0.0, 10.0, "intro"),
        (10.0, 20.0, "drop"),
        (20.0, 30.0, "end"),           # dropped: its 5 bars belong to nobody
        (30.0, 40.0, "drop"),          # merges with the earlier drop run
    ]
    assert bar_runs(sections, downbeats) == [("intro", 5), ("drop", 10)]


# --------------------------------------------------------------------------- #
# Decoder: stickiness and switching
# --------------------------------------------------------------------------- #


def test_isolated_flicker_bars_are_outvoted_by_the_duration_prior():
    priors = toy_priors(floor=4)
    posteriors = np.array([one_hot(DROP) for _ in range(40)])
    for bar in (7, 13, 22, 31):                 # single-bar spikes of breakdown
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
    """Alternating posteriors are the worst case for a smoother.  Every run the
    decoder commits -- except a final one the track truncates -- must be at
    least the class floor."""
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


# --------------------------------------------------------------------------- #
# Decoder: the structural graph
# --------------------------------------------------------------------------- #


def test_no_structurally_illegal_transition_is_ever_emitted():
    """Adversarial posteriors: the evidence demands outro, then drop, then
    intro again.  The decoder must find the best *legal* path instead."""
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


# --------------------------------------------------------------------------- #
# Decoder: lag semantics
# --------------------------------------------------------------------------- #


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
    """Decoding the first k bars and decoding all of them agree on every bar the
    k-bar run had already emitted, for every k.

    This is the weak half of the freeze rule -- it catches a decoder that
    re-emits or revises a bar at flush time.  The half that actually needs the
    trellis pruning is the next test.
    """
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
    """The strong half of the freeze rule, and the reason the trellis is pruned.

    A fixed-lag decoder reads bar B off the best path at bar B+lag and bar B+1
    off the best path at bar B+1+lag -- and those are two *different* paths.
    Stitching their answers together produces a stream that no single path ever
    proposed: on high-entropy posteriors an unpruned decoder emits ``outro ->
    drop``, re-enters intro, and commits runs a third of the min-duration floor.
    Every guarantee in this file would then hold only per-bar and none of them
    end to end.

    Pruning the disagreeing states to -inf at commit time makes the invariant
    global: every surviving path already agrees with everything emitted, so the
    emitted stream *is* a legal HSMM path.  That is what this asserts, on
    exactly the adversarial posteriors that break the naive version.
    """
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
def test_the_backtrace_is_a_ring_of_the_only_bars_it_can_read(lag):
    """A set is hours long and the decoder is never allowed to stall.

    ``_ancestors`` walks back at most from the frontier to the last commit,
    which the fixed lag pins at ``lag_bars`` rows; keeping the whole history was
    an offline convenience (bar-absolute indexing, a few hundred kB per track)
    and is an unbounded live allocation.  Bounding it is only safe because the
    ring RAISES on an evicted bar rather than returning a stale row, so a commit
    rule that ever reached further would fail loudly instead of decoding from
    whatever the modulo landed on.
    """
    priors = toy_priors(floor=3)
    rng = np.random.default_rng(11)
    posteriors = rng.dirichlet(np.full(5, 0.5), size=400)
    decoder = FixedLagViterbi(priors, lag_bars=lag)
    decoder.decode(posteriors)
    assert decoder.backtrace_rows == lag + 1


def test_the_ring_decodes_exactly_as_an_unbounded_backtrace_did():
    """Bounding the ring is a memory change, not a decoding change."""
    priors = toy_priors(floor=3)
    rng = np.random.default_rng(13)
    posteriors = rng.dirichlet(np.full(5, 0.4), size=120)
    boundary = rng.random(120)
    decoder = FixedLagViterbi(priors, lag_bars=2)
    decisions = decoder.decode(posteriors, boundary)
    assert [d.bar for d in decisions] == list(range(120))
    assert len(set(labels_of(decisions))) > 1


def test_flush_is_idempotent_and_a_decoder_can_be_reset_and_reused():
    priors = toy_priors(floor=3)
    posteriors = np.array([one_hot(DROP)] * 12)
    decoder = FixedLagViterbi(priors, lag_bars=2)
    first = decoder.decode(posteriors)
    assert decoder.flush() == []
    decoder.reset()
    assert decoder.decode(posteriors) == first


# --------------------------------------------------------------------------- #
# Decoder: the tunable knobs
# --------------------------------------------------------------------------- #


def imbalanced_case():
    """A bar where drop narrowly beats buildup, under the real corpus occupancy."""
    prior = np.array([0.12, 0.09, 0.28, 0.41, 0.10])
    row = np.zeros(5)
    row[DROP], row[BUILDUP], row[BREAKDOWN] = 0.50, 0.42, 0.08
    return toy_priors(floor=2, class_prior=prior), np.array([row] * 12)


def test_class_prior_division_recovers_a_class_the_imbalance_buries():
    """The corpus has 4.5x more drop bars than buildup ones.  Dividing by that
    occupancy turns a posterior back into a likelihood -- a runtime scalar, not
    a retrain."""
    priors, posteriors = imbalanced_case()
    plain = FixedLagViterbi(priors, lag_bars=2, class_prior_division=False)
    divided = FixedLagViterbi(priors, lag_bars=2, prior_strength=1.0)
    assert set(labels_of(plain.decode(posteriors))) == {"drop"}
    assert set(labels_of(divided.decode(posteriors))) == {"buildup"}


def test_prior_division_strength_scales_the_correction_in_both_directions():
    """The scalar is signed.  Positive divides the corpus prior out; negative
    puts it back, which is what a head already trained with inverse-frequency
    class weights actually needs."""
    priors, posteriors = imbalanced_case()
    weak = FixedLagViterbi(priors, lag_bars=2, prior_strength=0.1)
    reversed_ = FixedLagViterbi(priors, lag_bars=2, prior_strength=-1.0)
    assert set(labels_of(weak.decode(posteriors))) == {"drop"}
    assert set(labels_of(reversed_.decode(posteriors))) == {"drop"}
    # ...and at -1 the drop lead is wider than at +0.1, not merely preserved.
    assert (reversed_._emission_bonus[DROP] - reversed_._emission_bonus[BUILDUP]
            > weak._emission_bonus[DROP] - weak._emission_bonus[BUILDUP])


def test_the_default_prior_strength_is_neutral_because_the_net_is_pre_balanced():
    """``train.class_weights`` is inverse-frequency, so the label head already
    speaks under a uniform prior.  Shipping strength 1.0 would apply that same
    correction a second time (measured: 71.5 % -> 39.3 % per-bar on val), so the
    default divides by nothing and Task 5 owns the calibration."""
    priors, posteriors = imbalanced_case()
    default = FixedLagViterbi(priors, lag_bars=2)
    off = FixedLagViterbi(priors, lag_bars=2, class_prior_division=False)
    assert default.decode(posteriors) == off.decode(posteriors)
    assert np.allclose(default._emission_bonus, 0.0)


def test_drop_miss_cost_buys_drop_recall_at_the_price_of_precision():
    """Missing a drop is worse than a spurious one, and how much worse is a
    venue decision -- so it is an additive log-cost at the commit step, neutral
    at 1.0."""
    priors = toy_priors(floor=2)
    row = np.zeros(5)
    row[BREAKDOWN], row[DROP], row[BUILDUP] = 0.53, 0.42, 0.05
    posteriors = np.array([row] * 4)
    neutral = FixedLagViterbi(priors, lag_bars=1, drop_miss_cost=1.0)
    eager = FixedLagViterbi(priors, lag_bars=1, drop_miss_cost=3.0)
    assert set(labels_of(neutral.decode(posteriors))) == {"breakdown"}
    assert set(labels_of(eager.decode(posteriors))) == {"drop"}


def test_drop_miss_cost_is_charged_once_at_the_entry_edge_not_once_per_bar():
    """The spec puts asymmetric costs at the COMMIT step, and a commit is a run.

    Charging ``log(cost)`` on the emission instead compounds it with run length
    -- x657 over drop's 16-bar floor at cost 1.5 -- which turns a nominal [1, 3]
    sweep into "neutral ... everything is a drop".  This pins the arithmetic
    directly: the per-bar score is untouched, and every arc that *enters* drop is
    dearer by exactly log(cost).
    """
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

    # Continuing an existing drop is free: the run already paid at its edge.
    saturated = int(eager._final_state[DROP])
    assert eager._transition[saturated, saturated] == \
        neutral._transition[saturated, saturated]


@pytest.mark.parametrize("cost", [1.0, 3.0, 20.0, 200.0, 1000.0])
def test_raising_the_cost_never_lengthens_a_drop_run(cost):
    """The behavioural half of the same property, and the one that fails loudly
    under the per-bar implementation.

    A run pays once, so ``drop_miss_cost`` decides *whether* a drop is committed
    and the evidence alone decides how far it extends.  Charged per bar, every
    additional drop bar earns another ``log(cost)``, so the run grows without
    bound as the knob is turned: at 1000 the per-bar version swallows the entire
    track (``[(0, 36, 'drop')]``) -- the corpus-scale version of the reviewer's
    41 % -> 64 % drop occupancy while accuracy fell.

    (Far above this range the knob does eventually buy a one-bar drop as the
    track's truncated final run -- correct behaviour at 1:1,000,000 odds, not the
    compounding failure this pins.)
    """
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
    """The label head says "somewhere in here it becomes a drop"; the boundary
    head says "there".  With the hazard off the switch lands where the label
    evidence happens to tip; with it on, it lands on the boundary bar."""
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
    """The boundary head is a raw ranking score, not a probability, so it enters
    as a *relative* bounded bonus around a reference -- never as log P."""
    priors = toy_priors(floor=4)
    posteriors = np.array([one_hot(BREAKDOWN)] * 12 + [one_hot(DROP)] * 12)
    flat = np.full(len(posteriors), DEFAULT_BOUNDARY_REF)
    decoder = FixedLagViterbi(priors, lag_bars=3, boundary_weight=8.0)
    assert decoder.decode(posteriors, flat) == decoder.decode(posteriors)


# --------------------------------------------------------------------------- #
# Decoder: thin evidence and determinism
# --------------------------------------------------------------------------- #


def test_bars_with_no_usable_evidence_hold_the_last_state():
    """The last ~1 s of every track is covered by one window's deliberately
    unread edge.  A bar with nothing but that behind it gets a flat emission, so
    the duration prior -- which prefers staying -- carries it."""
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
    """A hazard of 0 makes a class unleavable and 1 caps every run at its floor.
    The fitter cannot produce either, so a prior that carries one has been
    hand-edited or is from another version -- refuse rather than decode oddly."""
    priors = toy_priors()
    for bad in (0.0, 1.0, 1.5):
        broken = priors._replace(hazard=np.full(len(priors.classes), bad))
        with pytest.raises(ValueError, match="hazard"):
            FixedLagViterbi(broken, lag_bars=2)


def test_importing_the_decoder_does_not_drag_torch_onto_the_decode_path():
    """The decode path must stay numpy-only: this exact object runs live.

    ``priors`` reaches the corpus for its *fitting* half (``nn.dataset`` alone
    pulls torch -- 1.9 s and 1,127 modules), so those imports live inside the
    functions that need them.  A module-level import added back would cost a
    show a two-second torch load and would not fail any other test here, since
    the dev venv has torch installed -- hence the subprocess.
    """
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


# --------------------------------------------------------------------------- #
# Track adapters: bar grid, aggregation, end to end
# --------------------------------------------------------------------------- #


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
    """A posterior sidecar with one-hot label posteriors on a known schedule.

    ``classes`` is one entry per *pooled* frame.  The first and last
    ``thin_frames`` frames get coverage 1 -- the sidecar's own marker for
    evidence that only one window's unread edge ever saw.
    """
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
    # 200 pooled frames of 0.1 s = 20 s; bars are 2 s, so 10 frames per bar.
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
    synthetic_npz(npz, [DROP] * 100, thin_frames=44)   # 2.2 s of edge at each end
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


# --------------------------------------------------------------------------- #
# The decoder generation the shipping config was measured on
# --------------------------------------------------------------------------- #


def test_a_config_naming_a_knob_the_decoder_lacks_is_refused(tmp_path):
    """The defect this generation exists to close.

    The loader used to filter unknown keys out against ``dataclasses.fields``.
    A config carrying a knob the running decoder does not have then loaded
    cleanly, decoded, and reported -- as a decoder nobody chose.
    """
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


def test_the_shipping_config_loads_and_is_the_frontier_pick():
    """Every knob, not the memorable ones.

    A half-pinned config is a config a merge resolution can quietly move: the
    five that used to be checked here left prior_strength, drop_miss_cost, the
    two boundary knobs and temperature free to become any other sweep row's
    values while the suite stayed green.  These are the lag-2 row of
    task1a_lag_sweep, which is what the file's own provenance block claims.
    """
    params = load_decoder_config(SHIPPING_DECODER_CONFIG)
    document = json.loads(SHIPPING_DECODER_CONFIG.read_text())
    assert document["name"] == "reduced_plus_floors_x0.75"
    assert dataclasses.asdict(params) == {
        "lag_bars": 2,
        "class_prior_division": True,
        "prior_strength": -0.25,
        "drop_miss_cost": 0.4642,
        "boundary_weight": 4.0,
        "boundary_ref": 0.2,
        "boundary_tolerance_sec": 0.5,
        "min_coverage": 1,
        "floor_scale": 1.25,
        "floor_bars": (8, 8, 6, 9, 8),
        "outro_escape": 0.02,
        "temperature": 1.0,
    }


def test_the_shipping_config_names_every_knob_the_decoder_has():
    """The other half: a knob added to the record and forgotten in the file."""
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
    """What shortening the floors buys, as behaviour rather than as a number.

    Both decoders enter the drop on the same evidence.  The floor decides how
    long they are then committed to it, and a run that outlives its evidence by
    six bars is exactly the crispness the shipped config bought back.
    """
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


def test_a_sidecar_no_frame_of_which_clears_the_threshold_raises(tmp_path):
    """A config that discards every frame decodes the priors and looks fine."""
    npz = tmp_path / "t.npz"
    synthetic_npz(npz, [DROP] * 20, thin_frames=40)
    edges = np.arange(0.0, 4.1, 2.0)
    with pytest.raises(RuntimeError, match="no frame has coverage"):
        bar_observations(npz, edges, min_coverage=2)
    assert not np.isnan(bar_observations(npz, edges, min_coverage=1)[0]).all()


def test_decode_track_carries_every_knob_the_config_names(tmp_path):
    """A param the end-to-end path drops is the loader defect one layer down."""
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
    """A sidecar from explicit posterior rows, for the knobs that read them.

    ``synthetic_npz`` writes one-hot rows, and tempering cannot change what a
    one-hot bar averages to -- which is exactly why a temperature the end-to-end
    path drops is invisible to every test built on it.
    """
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
    """Bars of one loud DROP frame against three quiet BREAKDOWN ones.

    The arithmetic mean of the bar reads DROP; the geometric mean reads
    BREAKDOWN, because DROP is nearly absent in three frames out of four.
    Temperature is what moves between those two readings, so this is a sidecar
    whose decoded labels depend on it.
    """
    loud = np.full(5, 0.01)
    loud[DROP] = 0.96
    quiet = np.full(5, (1.0 - 0.02 - 0.25) / 3.0)
    quiet[DROP], quiet[BREAKDOWN] = 0.02, 0.25
    return [row for _ in range(bars) for row in (loud, quiet, quiet, quiet)]


def test_decode_track_forwards_the_temperature_to_the_bar_average(tmp_path):
    """Reverting decode_track's ``temperature=`` argument turns this red.

    Nothing else pinned it: the knob only bites where a bar's frames disagree,
    and every other end-to-end fixture is one-hot.
    """
    npz, beats = tmp_path / "t.npz", tmp_path / "t.beat.csv"
    rows_npz(npz, graded_bars(16), frame_sec=0.25, label_pool=2, label_t0=0.5)
    write_beat_csv(beats, bars=16, bar_sec=2.0, t0=0.5)
    priors = toy_priors(floor=2, hazard=0.3)

    neutral = DecodeParams(lag_bars=0, boundary_weight=0.0, min_coverage=1)
    hot = dataclasses.replace(neutral, temperature=8.0)
    assert (set(label for _, label in decode_track(npz, beats, neutral, priors=priors))
            == {"drop"})
    assert (set(label for _, label in decode_track(npz, beats, hot, priors=priors))
            == {"breakdown"})


def test_decode_track_forwards_the_outro_escape_to_the_trellis(tmp_path):
    """Reverting decode_track's ``outro_escape=`` argument turns this red."""
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
