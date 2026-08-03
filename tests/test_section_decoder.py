import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from nn.decoder import DecodeParams, FixedLagViterbi, temper  # noqa: E402

from lib.engine.section_decoder import (  # noqa: E402
    BEATS_PER_BAR,
    NOMINAL_BAR_SEC,
    BarDecision,
    SectionDecoder,
)

from tests.test_nn_decoder import one_hot, toy_priors  # noqa: E402

TOLERANCE = DecodeParams().boundary_tolerance_sec


def beats(n: int, *, period: float = 0.5, start: float = 0.0) -> list:
    return [start + i * period for i in range(n)]


def decoder(*, lag_bars=2, floor=4, feature_latency_sec=0.0, **params):
    return SectionDecoder(toy_priors(floor=floor),
                          DecodeParams(lag_bars=lag_bars, min_coverage=1,
                                       **params),
                          feature_latency_sec=feature_latency_sec)


class Driver:
    def __init__(self, section: SectionDecoder) -> None:
        self.section = section
        self.decisions: list = []
        self.observations: list = []

    def beat(self, when: float) -> None:
        self._absorb(self.section.push_beat(when))

    def cell(self, when: float, posterior, boundary: float = 0.0) -> None:
        self._absorb(self.section.push_posterior(when, posterior, boundary))

    def _absorb(self, decisions) -> None:
        fresh = self.section.bars_pushed - len(self.observations)
        if fresh:
            recent = list(self.section.recent_observations)
            assert fresh <= len(recent), 'the in-flight window overflowed'
            self.observations.extend(recent[-fresh:])
        self.decisions.extend(decisions)


def feed(section, *, bars, labels, period=0.5, cells_per_bar=8,
         boundary=0.0, start=0.0) -> Driver:
    driver = Driver(section)
    bar_sec = period * BEATS_PER_BAR
    for beat in beats((bars + 1) * BEATS_PER_BAR, period=period, start=start):
        driver.beat(beat)
    step = bar_sec / cells_per_bar
    for index in range(bars * cells_per_bar + cells_per_bar):
        which = min(index // cells_per_bar, len(labels) - 1)
        driver.cell(start + (index + 1) * step, one_hot(labels[which]),
                    boundary)
    return driver


def test_a_bar_is_four_beats_anchored_at_the_first_detected_beat():
    section = decoder()
    for beat in beats(9, period=0.5):
        section.push_beat(beat)
    assert section.bar_edges == [0.0, 2.0, 4.0]


def test_no_bar_is_decodable_until_a_second_edge_closes_it():
    section = decoder()
    for beat in beats(BEATS_PER_BAR, period=0.5):
        section.push_beat(beat)
    for index in range(40):
        assert section.push_posterior((index + 1) * 0.1, one_hot(2), 0.0) == []
    assert section.bars_pushed == 0


def test_a_bar_waits_for_the_cells_that_close_it():
    section = decoder()
    for beat in beats(3 * BEATS_PER_BAR, period=0.5):
        section.push_beat(beat)
    section.push_posterior(1.0, one_hot(2), 0.0)
    assert section.bars_pushed == 0
    section.push_posterior(1.9, one_hot(2), 0.0)
    assert section.bars_pushed == 0
    section.push_posterior(2.0, one_hot(2), 0.0)
    assert section.bars_pushed == 1


def test_a_short_bar_waits_for_the_boundary_window_too():
    fast = 0.05
    section = decoder()
    for beat in beats(3 * BEATS_PER_BAR, period=fast):
        section.push_beat(beat)
    section.push_posterior(fast * BEATS_PER_BAR, one_hot(2), 0.0)
    assert section.bars_pushed == 0
    section.push_posterior(TOLERANCE, one_hot(2), 0.0)
    assert section.bars_pushed == 1


def test_cells_before_the_first_beat_are_dropped_rather_than_decoded():
    section = decoder()
    for index in range(20):
        assert section.push_posterior(index * 0.1, one_hot(3), 0.9) == []
    for beat in beats(3 * BEATS_PER_BAR, period=0.5, start=5.0):
        section.push_beat(beat)
    assert section.pending_cells == 0


def test_beats_arriving_after_their_cells_still_form_the_bar():
    section = decoder()
    for index in range(60):
        section.push_posterior(index * 0.1, one_hot(1), 0.0)
    for beat in beats(3 * BEATS_PER_BAR, period=0.5):
        section.push_beat(beat)
    assert section.bars_pushed == 2


def test_a_bars_posterior_is_the_mean_of_the_cells_inside_it():
    section = decoder()
    for beat in beats(3 * BEATS_PER_BAR, period=0.5):
        section.push_beat(beat)
    inside = [one_hot(0), one_hot(3)]
    section.push_posterior(0.5, inside[0], 0.0)
    section.push_posterior(1.5, inside[1], 0.0)
    section.push_posterior(2.5, one_hot(4), 0.0)
    assert section.bars_pushed == 1
    np.testing.assert_allclose(section.recent_observations[-1].posterior,
                               np.mean(inside, axis=0))


def test_the_closing_edge_belongs_to_the_next_bar():
    section = decoder()
    for beat in beats(3 * BEATS_PER_BAR, period=0.5):
        section.push_beat(beat)
    section.push_posterior(1.0, one_hot(0), 0.0)
    section.push_posterior(2.0, one_hot(3), 0.0)
    section.push_posterior(2.5, one_hot(3), 0.0)
    np.testing.assert_allclose(section.recent_observations[-1].posterior,
                               one_hot(0))


def test_the_boundary_is_the_max_within_tolerance_of_the_bar_line_or_nan_with_no_cell():
    section = decoder()
    driver = Driver(section)
    for beat in beats(5 * BEATS_PER_BAR, period=0.5):
        driver.beat(beat)
    driver.cell(1.0, one_hot(0), 0.11)
    driver.cell(1.7, one_hot(0), 0.93)
    driver.cell(2.0, one_hot(3), 0.02)
    driver.cell(3.9, one_hot(3), 0.40)
    driver.cell(4.0, one_hot(3), 0.05)
    driver.cell(4.4, one_hot(3), 0.77)
    driver.cell(6.0, one_hot(3), 0.01)
    assert np.isnan(driver.observations[0].boundary)
    assert driver.observations[1].boundary == pytest.approx(0.93)
    assert driver.observations[2].boundary == pytest.approx(0.77)


def test_a_bar_with_no_cells_is_no_evidence_rather_than_a_flat_guess():
    section = decoder()
    driver = Driver(section)
    for beat in beats(4 * BEATS_PER_BAR, period=0.5):
        driver.beat(beat)
    driver.cell(0.5, one_hot(2), 0.0)
    driver.cell(4.5, one_hot(2), 0.0)
    assert section.bars_pushed == 2
    assert driver.observations[1].posterior is None


def test_temperature_is_applied_per_cell_before_the_bar_average():
    rows = [one_hot(0, strength=0.6), one_hot(3, strength=0.6)]
    section = decoder(temperature=0.5)
    for beat in beats(3 * BEATS_PER_BAR, period=0.5):
        section.push_beat(beat)
    section.push_posterior(0.5, rows[0], 0.0)
    section.push_posterior(1.5, rows[1], 0.0)
    section.push_posterior(2.0, one_hot(2), 0.0)
    np.testing.assert_allclose(
        section.recent_observations[-1].posterior,
        temper(np.asarray(rows, dtype=np.float64), 0.5).mean(axis=0))


def test_a_config_asking_for_overlap_coverage_is_refused():
    with pytest.raises(ValueError, match="coverage"):
        SectionDecoder(toy_priors(), DecodeParams(min_coverage=2))


def test_the_streaming_push_equals_the_offline_decode_up_to_the_bars_the_lag_still_holds():
    labels = [0] * 6 + [1] * 6 + [3] * 8 + [4] * 6
    driver = feed(decoder(lag_bars=2), bars=len(labels), labels=labels,
                  boundary=0.3)

    offline = FixedLagViterbi(toy_priors(), 2)
    expected = offline.decode([obs.posterior for obs in driver.observations],
                              [obs.boundary for obs in driver.observations])

    assert [(d.bar, d.label) for d in driver.decisions] == \
           [(d.bar, d.label) for d in expected[:len(driver.decisions)]]
    assert len(expected) - len(driver.decisions) == 2


def test_a_decision_carries_the_bar_line_it_starts_on():
    labels = [0] * 8 + [3] * 8
    driver = feed(decoder(lag_bars=2), bars=len(labels), labels=labels,
                  period=0.5)
    assert driver.decisions
    for decision in driver.decisions:
        assert isinstance(decision, BarDecision)
        assert decision.start_sec == pytest.approx(decision.bar * 2.0)


def test_the_backtrace_ring_does_not_grow_with_the_set():
    section = decoder(lag_bars=2)
    feed(section, bars=40, labels=[0] * 20 + [3] * 20)
    assert section.backtrace_rows == 3


def test_pending_cells_do_not_grow_with_the_set():
    section = decoder(lag_bars=2)
    feed(section, bars=40, labels=[0] * 20 + [3] * 20, cells_per_bar=16)
    assert section.pending_cells <= 32


def test_the_bar_grid_does_not_grow_with_the_set():
    section = decoder(lag_bars=2)
    bar_sec = 0.5 * BEATS_PER_BAR
    for bar in range(400):
        for beat in beats(BEATS_PER_BAR, period=0.5, start=bar * bar_sec):
            section.push_beat(beat)
        for cell in range(8):
            section.push_posterior(bar * bar_sec + (cell + 1) * bar_sec / 8,
                                   one_hot(3 if bar % 20 else 0), 0.0)
    assert len(section.bar_edges) <= 64
    assert section.bars_pushed > 350, 'the grid stopped advancing'


def test_the_tempo_follows_a_change_instead_of_being_outvoted_by_history():
    section = decoder(lag_bars=2, feature_latency_sec=0.0)
    for beat in beats(60 * BEATS_PER_BAR, period=0.5):
        section.push_beat(beat)
    assert section.bar_sec == pytest.approx(2.0)

    start = 60 * BEATS_PER_BAR * 0.5
    for beat in beats(60 * BEATS_PER_BAR, period=0.345, start=start):
        section.push_beat(beat)
    assert section.bar_sec == pytest.approx(1.38, abs=0.02)


def test_a_beat_gap_re_anchors_the_grid_instead_of_closing_a_bar_across_it():
    section = decoder(lag_bars=2)
    for beat in beats(8, period=0.5):
        section.push_beat(beat)
    for index in range(40):
        section.push_posterior(4.0 + index * 0.25, one_hot(3), 0.0)
    stalled = section.pending_cells
    assert stalled > 0

    section.push_beat(60.0)
    assert section.pending_cells == 0, 'the stall was carried into the new bar'
    assert section.bar_edges[-1] == pytest.approx(60.0)
    section.push_beat(60.5)
    assert section.bar_edges[-1] == pytest.approx(60.0), \
        'the beat after the gap did not open the bar'


def test_a_re_anchor_leaves_the_bar_it_had_already_closed_alone():
    section = decoder(lag_bars=2)
    for beat in beats(8, period=0.5):
        section.push_beat(beat)
    assert section.bar_edges == [pytest.approx(0.0), pytest.approx(2.0)]

    section.push_beat(60.0)
    assert section.bar_edges[1] == pytest.approx(2.0), \
        'the closed bar lost the line that closed it'
    assert 60.0 - 2.0 not in [pytest.approx(hi - lo) for lo, hi
                              in zip(section.bar_edges, section.bar_edges[1:])
                              if lo == pytest.approx(0.0)], \
        'bar 0 was stretched across the gap'


def test_the_bar_the_gap_falls_in_is_never_decoded():
    section = decoder(lag_bars=2)
    seen: list = []

    def absorb():
        for observation in section.recent_observations:
            if observation not in seen:
                seen.append(observation)

    for beat in beats(8, period=0.5):
        section.push_beat(beat)
    section.push_beat(60.0)
    for beat in beats(12, period=0.5, start=60.0):
        section.push_beat(beat)
    absorb()

    for when in [2.2, 2.5, 2.8]:
        section.push_posterior(when, one_hot(3), 0.9)
        absorb()
    for index in range(24):
        section.push_posterior(60.5 + index * 0.25, one_hot(0), 0.0)
        absorb()

    stretched = [obs for obs in seen
                 if obs.end_sec - obs.start_sec > 4.0 and obs.posterior is not None]
    assert stretched == [], \
        f'a bar spanning the gap was decoded from stale cells: {stretched}'


def test_a_commit_can_never_name_a_bar_line_the_grid_has_dropped():
    section = decoder(lag_bars=2)
    period, bar = 0.5, 0.5 * BEATS_PER_BAR
    beat = 0.0
    while beat < 200 * bar:
        section.push_beat(beat)
        beat += period
    base = section.bar_edges[0]

    section.push_posterior(base + bar, one_hot(3), 0.0)
    for _ in range(BEATS_PER_BAR):
        section.push_beat(beat)
        beat += period
    for step in range(2, 8):
        section.push_posterior(base + step * bar, one_hot(3), 0.0)


def test_pending_cells_are_capped_even_while_the_beat_stream_is_gone():
    section = decoder(lag_bars=2)
    for beat in beats(8, period=0.5):
        section.push_beat(beat)
    for index in range(4000):
        section.push_posterior(4.0 + index * 0.09, one_hot(3), 0.0)
    assert section.pending_cells <= 256


def test_reset_makes_the_decoder_indistinguishable_from_a_fresh_one():
    labels = [0] * 6 + [3] * 10
    used = decoder(lag_bars=2)
    feed(used, bars=len(labels), labels=labels, boundary=0.8)
    used.reset()
    assert used.bar_edges == []
    assert used.pending_cells == 0
    assert used.bars_pushed == 0

    again = feed(used, bars=len(labels), labels=labels, boundary=0.8)
    fresh = feed(decoder(lag_bars=2), bars=len(labels), labels=labels,
                 boundary=0.8)
    assert [tuple(d) for d in again.decisions] == \
           [tuple(d) for d in fresh.decisions]


def test_chain_latency_is_the_features_plus_the_lag_the_bars_actually_took():
    section = decoder(lag_bars=2, feature_latency_sec=7.9938)
    assert section.bar_sec == pytest.approx(NOMINAL_BAR_SEC)
    assert section.chain_latency_sec == pytest.approx(
        7.9938 + 3 * NOMINAL_BAR_SEC)

    for beat in beats(5 * BEATS_PER_BAR, period=0.5):
        section.push_beat(beat)
    assert section.bar_sec == pytest.approx(2.0)
    assert section.chain_latency_sec == pytest.approx(7.9938 + 6.0)


def test_the_nominal_bar_is_the_corpus_median_until_bars_are_measured():
    # 1.8898 s: the median over 47,278 bars of the 215 val tracks (Task 1a).
    assert NOMINAL_BAR_SEC == pytest.approx(1.8898)
    assert decoder().bar_sec == pytest.approx(NOMINAL_BAR_SEC)


def test_a_reused_chain_is_handed_out_reset():
    from simulate import runner

    used = decoder(lag_bars=2)
    feed(used, bars=8, labels=[0] * 8)
    chain = type('Chain', (), {})()
    chain.stream = type('Stream', (), {'reset': lambda self: None})()
    chain.decoder = used
    runner._SECTION_CHAIN = chain
    try:
        handed = runner.load_section_chain()
        assert handed.decoder.bar_edges == []
        assert handed.decoder.bars_pushed == 0
    finally:
        runner._SECTION_CHAIN = runner._UNBUILT
