"""The runtime grid against the grid the phase-tracking gate actually measured.

Candidate D was priced offline, on cached madmom streams, by a driver that is
not the runtime.  Nothing in that measurement transfers unless the two build the
same bar lines out of the same beats, so this feeds the recorded streams through
``SectionDecoder`` and demands equality against ``phase_tracking``'s own edge
builder at the anchor the committed gate artifact records.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "training", REPO_ROOT / "training" / "phase_tracking"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lib.engine.section_decoder import (  # noqa: E402
    BEATS_PER_BAR, DecodeParams, SectionDecoder, _BEAT_GAP_SEC)

GATE_ARTIFACT = Path("models/phase_b/phase_tracking/phase_tracking_gate.json")
CANDIDATE = "D_anchor"
PROBE = "fallback|anchor1"
MIN_TRACKS = 20


class RecordingDecoder(SectionDecoder):
    """``SectionDecoder`` with an unbounded record of every line it ever drew."""

    def reset(self, *, cold_start: bool = True):
        self.drawn: list = []
        super().reset(cold_start=cold_start)

    def _append_edge(self, at_sec: float) -> None:
        self.drawn.append(float(at_sec))
        super()._append_edge(at_sec)


def gate() -> tuple:
    from corpus_root import corpus_dir

    data_dir = corpus_dir()
    artifact = data_dir / GATE_ARTIFACT
    beats_dir = data_dir / "madmom_beats"
    if not artifact.exists() or not beats_dir.is_dir():
        pytest.skip(f"the phase-tracking gate's artifacts are absent: "
                    f"{artifact} and {beats_dir} live in the gitignored corpus")
    return json.loads(artifact.read_text(encoding="utf-8")), beats_dir


def gate_lines(pt, live: np.ndarray, anchor: int) -> np.ndarray:
    positions = (np.arange(len(live), dtype=np.int64) + anchor) % BEATS_PER_BAR
    advances = np.ones(len(live), dtype=np.int64)
    advances[0] = 0
    return pt.edges_from_positions(live, positions, advances, False)


def runtime_lines(live: np.ndarray) -> list:
    section = RecordingDecoder(_priors(), DecodeParams(lag_bars=2, min_coverage=1))
    for beat in live:
        section.push_beat(beat)
    return section.drawn


def _priors():
    from tests.test_nn_decoder import toy_priors

    return toy_priors()


@pytest.mark.integration
def test_the_runtime_draws_the_gates_candidate_d_lines_beat_for_beat():
    import phase_tracking as pt

    report, beats_dir = gate()
    spec = next(c for c in report["candidates"] if c["name"] == CANDIDATE)
    anchor = int(spec["anchor"])
    ids = [row["youtube_id"] for row in report["per_track"]]
    assert len(ids) >= MIN_TRACKS

    logging.disable(logging.WARNING)
    try:
        exact, gapped, total_gate_lines = [], [], 0
        for youtube_id in ids:
            live = np.load(beats_dir / f"{youtube_id}.npy")
            expected = gate_lines(pt, live, anchor)
            total_gate_lines += expected.size
            drawn = runtime_lines(live)

            gaps = np.flatnonzero(np.diff(live) > _BEAT_GAP_SEC)
            if gaps.size == 0:
                assert drawn == list(expected), \
                    f"{youtube_id}: the runtime drew a different grid"
                exact.append(youtube_id)
                continue
            until = float(live[gaps[0]])
            shared = [line for line in expected if line <= until]
            assert drawn[:len(shared)] == shared, \
                f"{youtube_id}: the two grids differ before the first beat gap"
            gapped.append(youtube_id)
    finally:
        logging.disable(logging.NOTSET)

    assert len(exact) >= MIN_TRACKS, \
        f"only {len(exact)} tracks are free of the re-anchor the gate never modelled"
    assert total_gate_lines == report["grid_geometry"][CANDIDATE]["bar_lines"], \
        "the reconstructed candidate-D grid is not the one the gate scored"
    assert len(exact) + len(gapped) == len(ids)


@pytest.mark.integration
def test_a_beat_gap_leaves_no_bar_position_worth_anchoring_on():
    import phase_tracking as pt
    from raveform_fetch_annotations import BEATS_DIR, annotations_dir

    from corpus_root import corpus_dir

    report, beats_dir = gate()
    annotations = annotations_dir(corpus_dir()) / BEATS_DIR
    if not annotations.is_dir():
        pytest.skip(f"the corpus beat grids are absent: {annotations}")
    by_id = {path.name.split(".")[1]: path for path in annotations.iterdir()}

    seen = [0] * BEATS_PER_BAR
    for row in report["per_track"]:
        youtube_id = row["youtube_id"]
        live = np.load(beats_dir / f"{youtube_id}.npy")
        gaps = np.flatnonzero(np.diff(live) > _BEAT_GAP_SEC)
        if gaps.size == 0 or youtube_id not in by_id:
            continue
        times, positions = pt.annotation_beats(by_id[youtube_id])
        phase = pt.expert_phase(times, positions)
        if phase is None:
            continue
        js, isx = pt.match(live, times)
        truth, known = pt.truth_phase_track(
            len(live), js, pt.required_phase(js, isx, phase))
        for gap in gaps:
            after = int(gap) + 1
            if known[after]:
                seen[int((after - truth[after]) % BEATS_PER_BAR)] += 1

    assert sum(seen) >= BEATS_PER_BAR * 4, f"too few gaps to read: {seen}"
    assert max(seen) * 2 < sum(seen), (
        f"the beat that ends a gap prefers a bar position ({seen}) -- the "
        f"re-anchor is not the coin toss it is written as")


@pytest.mark.integration
def test_the_gates_anchor_is_the_one_the_runtime_ships():
    from lib.engine.section_decoder import _FIRST_BEAT_BAR_POSITION

    report, _ = gate()
    spec = next(c for c in report["candidates"] if c["name"] == CANDIDATE)
    assert spec["probe"] == PROBE
    assert int(spec["anchor"]) == _FIRST_BEAT_BAR_POSITION


def test_the_stand_in_takes_the_reset_the_runtime_would_call_on_it():
    section = RecordingDecoder(_priors(), DecodeParams(lag_bars=2, min_coverage=1))
    for beat in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        section.push_beat(beat)

    section.reset(cold_start=False)
    for beat in (3.0, 3.5):
        section.push_beat(beat)

    assert section.drawn == [3.5]
