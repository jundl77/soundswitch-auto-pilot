"""What the committer believes when it is born, and how old it is.

A beat gap re-anchors the bar grid and restarts the committer.  Until this
package that restart was a cold start -- the corpus's START-OF-TRACK prior -- so
a decoder reborn 34 minutes into a set believed it was at bar zero of an
imaginary track: it could commit ``intro`` (which no fitted transition can
enter) and had to walk a whole duration floor before it was allowed to leave,
charging a minimum duration for time it was never alive to witness.

The semantics pinned here are the ones the rebirth gate measured -- the
reference is ``training/decoder_rebirth/rebirth.py`` and the verdict is
``models/phase_b/decoder_rebirth/rebirth_gate.json``.  R1 carries the committed
class into the reborn decoder, R2 puts the birth's mass on the FINAL duration
states, R3 pushes one virtual predecessor bar.  Ruling #199 ships all three or
none, because R3 is R1's safety valve: with no predecessor bar a carried belief
cannot be refused on the bar it is born on, and the gate's forced births were
accepted 85 of 85 under R1 alone against 37 of 85 under all three.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "training", REPO_ROOT / "training" / "decoder_rebirth"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import rebirth as RB  # noqa: E402

from nn.decoder import DecodeParams, FixedLagViterbi, segments  # noqa: E402

from lib.engine.section_decoder import (  # noqa: E402
    BEATS_PER_BAR,
    SectionDecoder,
    _BEAT_GAP_SEC,
    _FIRST_BEAT_BAR_POSITION,
)

from tests.test_nn_decoder import one_hot, toy_priors  # noqa: E402

INTRO, BUILDUP, BREAKDOWN, DROP, OUTRO = range(5)

STARTS_AT_INTRO = (0.996, 0.001, 0.001, 0.001, 0.001)

CONFIDENT = 0.97
FLAT = 1.0 / 5


def priors(*, floor=4, initial=STARTS_AT_INTRO):
    return toy_priors(floor=floor, initial=list(initial))


def viterbi(*, floor=4, lag_bars=1, initial=STARTS_AT_INTRO):
    return FixedLagViterbi(priors(floor=floor, initial=initial), lag_bars)


class Stage(SectionDecoder):
    """A ``SectionDecoder`` driven by a beat stream with one cell per bar.

    The cell rides just behind the line that opened its bar, so a bar's
    posterior is exactly the row asked for and its boundary is exactly the score
    asked for -- no neighbouring bar is inside the boundary tolerance at these
    tempos.  Every beat, row and score is recorded so the same take can be
    replayed through the gate's own reference decoder.
    """

    CELL_OFFSET = 0.1

    def __init__(self, *, floor=4, lag_bars=2, initial=STARTS_AT_INTRO,
                 period=0.5, start=0.0, **params):
        self.births: list = []
        super().__init__(priors(floor=floor, initial=initial),
                         DecodeParams(lag_bars=lag_bars, min_coverage=1,
                                      **params))
        self.period = float(period)
        self.now = float(start)
        self.decisions: list = []
        self.beats: list = []
        self.rows: list = []
        self.scores: list = []

    def _restart_committer_at(self, bar: int) -> None:
        self.births.append((int(bar), self._committed_class))
        super()._restart_committer_at(bar)

    def play(self, bars: int, label: int, *, boundary: float = 0.0,
             strength: float = CONFIDENT) -> "Stage":
        opened = 0
        while opened < bars:
            self.beats.append(self.now)
            self.decisions.extend(self.push_beat(self.now))
            if self.bar_edges and self.bar_edges[-1] == self.now:
                opened += 1
                row = one_hot(label, strength)
                self.rows.append(row)
                self.scores.append(float(boundary))
                self.decisions.extend(self.push_posterior(
                    self.now + self.CELL_OFFSET, row, boundary))
            self.now += self.period
        return self

    def beat_only(self, beats: int) -> "Stage":
        for _ in range(beats):
            self.beats.append(self.now)
            self.decisions.extend(self.push_beat(self.now))
            if self.bar_edges and self.bar_edges[-1] == self.now:
                self.rows.append(np.full(5, FLAT))
                self.scores.append(0.0)
            self.now += self.period
        return self

    def gap(self, seconds: float) -> "Stage":
        self.now += float(seconds)
        return self

    def since(self, at_sec: float) -> list:
        return [d for d in self.decisions if d.start_sec >= at_sec - 1e-9]


def labels_of(decisions) -> list:
    return [d.label for d in decisions]


def push_all(decoder, rows, scores=None) -> list:
    decisions = []
    for index, row in enumerate(rows):
        decisions.extend(decoder.push(
            row, None if scores is None else scores[index]))
    decisions.extend(decoder.flush())
    return decisions


def test_a_restart_is_born_old_so_its_first_run_may_end_before_the_floor():
    rows = [one_hot(INTRO)] + [one_hot(DROP)] * 8

    cold = viterbi(floor=4)
    assert segments(cold.decode(rows))[0] == (0, 4, "intro"), \
        "a track that starts at bar zero owes intro its whole floor"

    reborn = viterbi(floor=4)
    reborn.restart()
    assert segments(push_all(reborn, rows))[0] == (0, 1, "intro"), \
        "the birth charged a floor for time the grid never witnessed"


def test_a_restart_does_not_follow_the_decoder_into_the_next_offline_decode():
    decoder = viterbi(floor=4)
    rows = [one_hot(INTRO)] + [one_hot(DROP)] * 8
    expected = segments(decoder.decode(rows))

    decoder.restart(DROP)
    assert segments(decoder.decode(rows)) == expected, \
        "a rebirth outlived the committer it was for; the offline sweeps and " \
        "the eval chain share these instances and must keep decoding a track " \
        "from bar zero of that track"


def test_a_restart_carrying_a_class_believes_nothing_else():
    decoder = viterbi(floor=4)
    decoder.restart(BREAKDOWN)
    decisions = push_all(decoder, [one_hot(DROP)] * 6)

    assert decisions[0].label == "breakdown", \
        "the carry is certain, so the bar a committer is born on has no " \
        "second candidate -- which is the whole reason R3 exists"


def test_a_re_anchor_carries_the_class_the_show_was_in():
    stage = Stage().play(10, DROP)
    assert labels_of(stage.decisions)[-1] == "drop"

    stage.gap(_BEAT_GAP_SEC + 1.0)
    born_at = stage.now
    stage.play(6, INTRO, strength=FLAT)

    after = stage.since(born_at)
    assert after, "the re-anchored grid committed nothing"
    assert labels_of(after) == ["drop"] * len(after), \
        "a five second hole teleported the decoder back to bar zero of an " \
        "imaginary track"


def test_a_birth_pushes_the_virtual_predecessor_bar_before_any_real_one():
    stage = Stage()
    assert stage.backtrace_rows == 1, "a cold birth has no bar to switch from"
    assert stage.bars_pushed == 0, "the virtual bar was counted as a real one"

    stage.play(10, DROP).gap(_BEAT_GAP_SEC + 1.0)
    stage.play(1, INTRO, strength=FLAT)
    assert stage.births[-1][0] == stage.bars_pushed, \
        "the re-anchor's virtual bar was drawn from the grid"
    assert stage.backtrace_rows == 1


def test_a_carried_belief_can_be_refused_on_the_bar_it_is_born_on():
    stage = Stage().play(10, DROP)
    assert labels_of(stage.decisions)[-1] == "drop"

    stage.gap(_BEAT_GAP_SEC + 1.0)
    born_at = stage.now
    stage.play(8, BREAKDOWN, boundary=1.0)

    assert stage.births[-1][1] == DROP, "there was no belief to refuse"
    after = stage.since(born_at)
    assert after and after[0].label == "breakdown", \
        f"the carried drop could not be refused on the bar it was born on: " \
        f"{labels_of(after)[:4]}"


def test_a_boundary_on_the_grids_own_first_bar_has_a_transition_to_land_on():
    stage = Stage(boundary_weight=4.0).play(8, BUILDUP, boundary=1.0,
                                            strength=0.5)
    assert stage.decisions and stage.decisions[0].label == "buildup", \
        "the grid's own first bar had no transition to apply the boundary " \
        "bonus to, so the start-of-track prior kept it"


def test_the_shed_grid_restart_is_a_rebirth_too(caplog):
    stage = Stage().play(10, DROP)
    assert labels_of(stage.decisions)[-1] == "drop"
    cursor = stage.bars_pushed

    with caplog.at_level(logging.WARNING):
        stage.beat_only(80 * BEATS_PER_BAR)
    assert any("past the oldest line" in record.message
               for record in caplog.records), \
        "the grid never outran the committer, so this is not the shed path"

    assert stage.births[-1][0] > cursor
    assert stage.births[-1][1] == DROP, \
        "the third birth path threw the belief away"
    assert stage.backtrace_rows == 1, \
        "the third birth path was born with nothing to switch from"

    born_at = stage.now
    stage.play(6, INTRO, strength=FLAT)
    after = stage.since(born_at)
    assert after and labels_of(after) == ["drop"] * len(after)


def test_a_warm_restart_after_a_feature_gap_carries_the_belief_too():
    stage = Stage().play(10, DROP)
    stage.reset(cold_start=False)
    assert stage.births[-1][1] == DROP

    born_at = stage.now
    stage.play(8, INTRO, strength=FLAT)
    after = stage.since(born_at)
    assert after and labels_of(after) == ["drop"] * len(after), \
        "a hole in the feature stage is not a new track either"


def test_a_cold_reset_carries_nothing():
    stage = Stage().play(10, DROP)
    stage.reset()
    assert stage.births[-1][1] is None


def test_the_runtime_decides_what_the_gates_reference_decides():
    stage = Stage(lag_bars=2)
    stage.play(9, DROP).play(5, BREAKDOWN, boundary=0.9)
    stage.gap(_BEAT_GAP_SEC + 1.5)
    stage.play(6, DROP, strength=FLAT).play(7, OUTRO, boundary=0.8)

    grid = RB.live_grid(stage.beats)
    reference, births = RB.decode_live(
        grid, np.asarray(stage.rows, dtype=np.float64),
        np.asarray(stage.scores, dtype=np.float64),
        FixedLagViterbi(priors(), stage.params.lag_bars), RB.FULL)

    assert len(births) == 2 and births[1]["carried"] is not None, \
        "the take did not re-anchor, so it exercises no birth at all"
    assert stage.decisions, "the runtime committed nothing to compare"
    assert [(d.bar, d.label) for d in stage.decisions] == \
           [(d.bar, reference[d.bar]) for d in stage.decisions]


CASE_DIR = Path("models/phase_b/decoder_rebirth/case_inputs")
GATE_PRIORS = Path("models/phase_b_student_kd_t2_w05_s1234/priors.json")
GATE_FRONTIER = Path("models/phase_b/student_kd_t2_w05_s1234/frontier.json")
GATE_LAG_BARS = 2

WRONG_STATE_CASE = "case_cXBIZOiSaxA.json"
WRONG_STATE_RE_ANCHOR_SEC = 351.782
WRONG_STATE_HELD_SEC = 28.131

CLIFF_CASE = "case_cHwb8tOEHM.json"
CLIFF_BAR_SEC = 1.7291
CLIFF_ROWS_FULL_ARM = ((40.00, 69.395), (55.00, 68.833), (60.00, 68.646),
                       (64.00, 69.187), (66.50, 69.958), (68.93, 68.930))
CLIFF_UNTIL_SEC = 130.0


def gate_case(report_name: str, posteriors: str) -> tuple:
    import dataclasses

    from corpus_root import corpus_dir
    from nn.priors import Priors

    data_dir = corpus_dir()
    case_dir = data_dir / CASE_DIR
    report = case_dir / report_name
    frontier, priors_file = data_dir / GATE_FRONTIER, data_dir / GATE_PRIORS
    for path in (report, case_dir / posteriors, frontier, priors_file):
        if not path.exists():
            pytest.skip(f"the rebirth gate's inputs are absent: {path} lives "
                        f"in the gitignored corpus")

    document = json.loads(frontier.read_text(encoding="utf-8"))
    row = next(r for r in document["frontier"] if r["name"] == document["pick"])
    params = dataclasses.replace(DecodeParams(**row["params"]),
                                 lag_bars=GATE_LAG_BARS)

    beats = np.asarray([b["t"] for b in json.loads(
        report.read_text(encoding="utf-8"))["beats"]], dtype=np.float64)
    with np.load(case_dir / posteriors) as archive:
        trace = (np.asarray(archive["t"], dtype=np.float64),
                 np.asarray(archive["posterior"], dtype=np.float64),
                 np.asarray(archive["boundary"], dtype=np.float64))
    return Priors.load(priors_file), params, beats, trace


def replay(section: SectionDecoder, beats, trace, *, until=None) -> list:
    """The two streams merged in time order, beats first on a tie."""
    times, posterior, boundary = trace
    decisions, cell = [], 0
    logging.disable(logging.WARNING)
    try:
        for beat in beats:
            if until is not None and beat > until:
                break
            while cell < len(times) and times[cell] <= beat:
                decisions.extend(section.push_posterior(
                    times[cell], posterior[cell], boundary[cell]))
                cell += 1
            decisions.extend(section.push_beat(float(beat)))
        while cell < len(times) and (until is None or times[cell] <= until):
            decisions.extend(section.push_posterior(
                times[cell], posterior[cell], boundary[cell]))
            cell += 1
    finally:
        logging.disable(logging.NOTSET)
    return decisions


def first_change(decisions):
    """The case study's metric: when the show stops saying ``intro``."""
    for decision in decisions:
        if decision.label != "intro":
            return decision
    return None


@pytest.mark.integration
def test_the_runtime_commits_the_gates_class_at_the_re_anchor():
    gate_priors, params, beats, trace = gate_case(
        WRONG_STATE_CASE, "runtime_posteriors_cXBIZ.npz")
    section = SectionDecoder(gate_priors, params)
    decisions = replay(section, beats, trace)

    at_birth = [d for d in decisions
                if d.start_sec >= WRONG_STATE_RE_ANCHOR_SEC - 1e-3]
    assert at_birth, "the runtime never reached the re-anchor"
    assert at_birth[0].start_sec == pytest.approx(WRONG_STATE_RE_ANCHOR_SEC,
                                                  abs=1e-3)
    assert at_birth[0].label == "breakdown", \
        "the reborn committer did not carry the breakdown the model was " \
        "0.978 sure of; this is the 53.4s of near-dark stage the gate named"

    changed = next(d for d in at_birth if d.label != "breakdown")
    assert changed.label == "buildup"
    assert changed.start_sec - at_birth[0].start_sec == \
        pytest.approx(WRONG_STATE_HELD_SEC, abs=5e-3)


@pytest.mark.integration
@pytest.mark.parametrize("opens_at,expected", CLIFF_ROWS_FULL_ARM)
def test_the_runtime_reproduces_the_gates_cliff_table(opens_at, expected):
    gate_priors, params, _beats, trace = gate_case(
        CLIFF_CASE, "runtime_posteriors.npz")
    beat_sec = CLIFF_BAR_SEC / BEATS_PER_BAR
    first = opens_at - (BEATS_PER_BAR - _FIRST_BEAT_BAR_POSITION) * beat_sec
    beats = np.arange(first, CLIFF_UNTIL_SEC, beat_sec)

    section = SectionDecoder(gate_priors, params)
    decisions = replay(section, beats, trace, until=CLIFF_UNTIL_SEC)
    assert section.bar_edges[0] == pytest.approx(opens_at) or \
        min(abs(edge - opens_at) for edge in section.bar_edges) < 1e-6, \
        "the synthetic grid the runtime drew is not the gate's grid"

    change = first_change(decisions)
    assert change is not None, "the runtime never left intro"
    assert change.start_sec == pytest.approx(expected, abs=5e-3)
