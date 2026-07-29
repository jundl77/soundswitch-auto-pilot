import logging

import pytest

from lib.analyser.drift_watchdog import DriftWatchdog, ShedLevel


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


BUF = 256 / 44100


def _feed(dog, clock, buffers, wall_per_buffer):
    for _ in range(buffers):
        clock.advance(wall_per_buffer)
        dog.observe()
    return dog.level


def test_keeping_up_exactly_never_sheds():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    assert _feed(dog, clock, 3000, BUF) is ShedLevel.NONE
    assert dog.drift_sec == 0.0


def test_running_faster_than_real_time_never_sheds():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    assert _feed(dog, clock, 3000, BUF * 0.2) is ShedLevel.NONE
    assert dog.drift_sec <= 0.0


def test_a_sustained_shortfall_sheds_section_detection_first():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    assert _feed(dog, clock, 1200, BUF * 1.1) is ShedLevel.SECTION_DETECTION


def test_a_worse_shortfall_sheds_onset_detection_but_never_beats():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    level = _feed(dog, clock, 800, BUF * 3.0)
    assert level is ShedLevel.ONSET_DETECTION
    assert level is max(ShedLevel), 'beat tracking must never be shed'


def test_a_single_stall_does_not_latch_the_watchdog_forever():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 200, BUF)
    clock.advance(2.0)
    dog.observe()
    assert dog.level is not ShedLevel.NONE, 'a 2 s stall must be noticed'
    _feed(dog, clock, 8000, BUF * 0.3)
    assert dog.level is ShedLevel.NONE, 'and must be recovered from'


def test_running_at_exactly_real_time_counts_as_recovered():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 1200, BUF * 1.1)
    assert dog.level is ShedLevel.SECTION_DETECTION
    _feed(dog, clock, 2000, BUF)
    assert dog.level is ShedLevel.NONE


def test_recovery_waits_out_a_full_window_before_restoring_work():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 1200, BUF * 1.1)
    assert dog.level is ShedLevel.SECTION_DETECTION
    _feed(dog, clock, 300, BUF)
    assert dog.level is ShedLevel.SECTION_DETECTION


def test_a_dip_below_the_exit_threshold_does_not_bank_progress():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 1200, BUF * 1.1)
    for _ in range(8):
        _feed(dog, clock, 200, BUF)
        _feed(dog, clock, 200, BUF * 1.3)
    assert dog.level is not ShedLevel.NONE


def test_every_level_change_is_logged_at_warning_or_above(caplog):
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    with caplog.at_level(logging.WARNING):
        _feed(dog, clock, 400, BUF * 1.5)
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert any('drift' in r.message.lower() for r in caplog.records)


def test_peak_and_total_drift_are_reported_for_the_soak_run():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 200, BUF)
    clock.advance(1.5)
    dog.observe()
    assert dog.peak_drift_sec >= 1.4
    assert dog.total_drift_sec == pytest.approx(1.5, abs=0.01), \
        'total is cumulative and never forgets'


def test_total_drift_is_a_difference_of_totals_not_a_sum_of_excesses():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    # A hardware-paced input alternates: one read returns instantly because a
    # buffer was already queued, the next blocks for two buffer periods.
    for _ in range(2000):
        clock.advance(0.0)
        dog.observe()
        clock.advance(BUF * 2)
        dog.observe()
    assert abs(dog.total_drift_sec) < 0.05, \
        f'jitter must cancel, got {dog.total_drift_sec:.3f}s'


def test_reset_returns_it_to_the_constructed_state():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 400, BUF * 3.0)
    assert dog.level is not ShedLevel.NONE
    dog.reset()
    assert dog.level is ShedLevel.NONE
    assert dog.drift_sec == 0.0
    assert dog.peak_drift_sec == 0.0


def test_the_first_observation_cannot_report_drift():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    clock.advance(10.0)
    assert dog.observe() is ShedLevel.NONE
    assert dog.drift_sec == 0.0
