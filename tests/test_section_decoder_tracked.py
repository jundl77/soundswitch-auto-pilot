"""The decoder's bar grid under a phase source, and the counting degradation."""
import numpy as np

from lib.engine.section_decoder import DecodeParams, SectionDecoder
from tests.test_nn_decoder import toy_priors


class _ScriptedPhase:
    """A phase source that returns a scripted position stream."""

    def __init__(self, positions):
        self.positions = list(positions)
        self.cursor = 0
        self.resets = 0
        self.reanchors = 0
        self.chunks = []

    def push_beat(self, at_sec):
        position = self.positions[self.cursor % len(self.positions)]
        self.cursor += 1
        return position

    def reset(self):
        self.resets += 1
        self.cursor = 0

    def reanchor(self, at_sec):
        self.reanchors += 1
        self.cursor = 0
        return 0

    def push_chunk(self, chunk):
        self.chunks.append(chunk)


class RecordingDecoder(SectionDecoder):
    def reset(self, *, cold_start: bool = True):
        if not hasattr(self, "drawn"):
            self.drawn = []
        super().reset(cold_start=cold_start)

    def _append_edge(self, at_sec: float) -> None:
        self.drawn.append(float(at_sec))
        super()._append_edge(at_sec)


def _decoder(phase):
    return RecordingDecoder(toy_priors(), DecodeParams(lag_bars=2,
                                                       min_coverage=1),
                            phase=phase)


def test_edges_land_where_the_committed_count_wraps_to_zero():
    phase = _ScriptedPhase([0, 1, 2, 3, 0, 1, 2, 3])
    decoder = _decoder(phase)
    beats = 0.5 * np.arange(1, 9)
    for beat in beats:
        decoder.push_beat(beat)
    assert decoder.drawn == [0.5, 2.5]


def test_a_wrap_that_skips_zero_draws_no_line():
    # a forward-only correction can step 3 -> 1 without landing on 0
    phase = _ScriptedPhase([1, 2, 3, 1, 2, 3, 0, 1])
    decoder = _decoder(phase)
    for beat in 0.5 * np.arange(1, 9):
        decoder.push_beat(beat)
    assert decoder.drawn == [3.5]


def test_construction_resets_the_phase_and_a_cold_reset_does_again():
    phase = _ScriptedPhase([0, 1, 2, 3])
    decoder = _decoder(phase)
    assert phase.resets == 1
    decoder.push_beat(0.5)
    decoder.reset()
    assert phase.resets == 2


def test_a_feature_gap_reset_keeps_the_phase():
    phase = _ScriptedPhase([0, 1, 2, 3])
    decoder = _decoder(phase)
    for beat in (0.5, 1.0, 1.5):
        decoder.push_beat(beat)
    decoder.reset(cold_start=False)
    assert phase.resets == 1
    decoder.push_beat(2.0)
    assert phase.cursor == 4


def test_a_beat_gap_reanchors_the_phase_and_draws_one_edge():
    phase = _ScriptedPhase([1, 2, 3, 0, 1, 2])
    decoder = _decoder(phase)
    for beat in (0.5, 1.0, 1.5):
        decoder.push_beat(beat)
    decoder.push_beat(9.0)
    assert phase.reanchors == 1
    assert decoder.drawn[-1] == 9.0
    # the next wrap after the re-anchor still draws
    scripted_after = [1, 2, 3, 0]
    phase.positions = scripted_after
    phase.cursor = 0
    for k, beat in enumerate((9.5, 10.0, 10.5, 11.0)):
        decoder.push_beat(beat)
    assert decoder.drawn[-1] == 11.0


def test_evidence_reaches_the_phase():
    phase = _ScriptedPhase([0, 1, 2, 3])
    decoder = _decoder(phase)
    decoder.push_evidence("chunk")
    assert phase.chunks == ["chunk"]


def test_without_a_phase_evidence_is_dropped_and_counting_stands():
    decoder = RecordingDecoder(toy_priors(),
                               DecodeParams(lag_bars=2, min_coverage=1))
    decoder.push_evidence("chunk")
    for beat in 0.5 * np.arange(1, 9):
        decoder.push_beat(beat)
    # the warm-up anchor: first beat is position 1, so edges land on beats 4, 8
    assert decoder.drawn == [2.0, 4.0]
