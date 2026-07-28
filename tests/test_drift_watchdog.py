"""Backpressure: the analyser must notice when it stops keeping up, and say so.

Audio arrives at exactly 1x and never waits. The look-ahead lead is a fixed
budget, spent once at start-up; every second the analyser runs slower than
real time is a second of lead permanently gone unless it is drained back. So
"fast enough on average" is not the property that matters — not falling behind
*and staying behind* is.
"""

import logging

from lib.analyser.drift_watchdog import DriftWatchdog, ShedLevel


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


BUF = 256 / 44100  # 5.805 ms — the live buffer period


def _feed(dog, clock, buffers, wall_per_buffer):
    """Process `buffers` buffers, each taking `wall_per_buffer` of wall time."""
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
    """The normal case: ~20 % of a core per buffer means the loop drains."""
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    assert _feed(dog, clock, 3000, BUF * 0.2) is ShedLevel.NONE
    assert dog.drift_sec <= 0.0


def test_a_sustained_shortfall_sheds_section_detection_first():
    """Beat-level function is the thing that must survive, so the first thing
    shed is the one the show can lose: YAMNet section detection."""
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    # 1.1x too slow: a 9 % shortfall, past level 1's 3 % and short of level 2's 15 %.
    assert _feed(dog, clock, 1200, BUF * 1.1) is ShedLevel.SECTION_DETECTION


def test_a_worse_shortfall_sheds_onset_detection_but_never_beats():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    level = _feed(dog, clock, 800, BUF * 3.0)
    assert level is ShedLevel.ONSET_DETECTION
    assert level is max(ShedLevel), 'beat tracking must never be shed'


def test_a_single_stall_does_not_latch_the_watchdog_forever():
    """PortAudio drops rather than queues without bound, so a stall's cost is
    permanent — cumulative lag would never fall again and the analyser would
    stay degraded for the rest of the show. The control signal is therefore the
    drift accumulated in a rolling window, which a one-off stall leaves."""
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 200, BUF)
    clock.advance(2.0)          # one 2-second stall
    dog.observe()
    assert dog.level is not ShedLevel.NONE, 'a 2 s stall must be noticed'
    _feed(dog, clock, 3000, BUF * 0.3)   # then catch up
    assert dog.level is ShedLevel.NONE, 'and must be recovered from'


def test_recovery_is_hysteretic_so_the_level_cannot_flap():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 1200, BUF * 1.1)
    assert dog.level is ShedLevel.SECTION_DETECTION
    # Exactly real-time is not recovery: it holds the lost lead, it does not
    # win it back. The level must persist.
    _feed(dog, clock, 1200, BUF)
    assert dog.level is ShedLevel.SECTION_DETECTION


def test_every_level_change_is_logged_at_warning_or_above(caplog):
    """Degrading silently is the failure mode this exists to prevent."""
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
    _feed(dog, clock, 3000, BUF * 0.3)
    assert dog.peak_drift_sec >= 1.4
    assert dog.total_drift_sec >= 1.4, 'total is cumulative and never forgets'


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
    """One sample is not a span; a watchdog that read drift from it would fire
    on start-up latency."""
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    clock.advance(10.0)
    assert dog.observe() is ShedLevel.NONE
    assert dog.drift_sec == 0.0
