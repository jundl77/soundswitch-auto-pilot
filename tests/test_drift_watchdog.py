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


def test_a_sustained_shortfall_sheds_the_nn():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    assert _feed(dog, clock, 1200, BUF * 1.1) is ShedLevel.NN_SHED


def test_the_worst_shortfall_there_is_still_never_sheds_beats():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    level = _feed(dog, clock, 800, BUF * 3.0)
    assert level is ShedLevel.NN_SHED
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
    assert dog.level is ShedLevel.NN_SHED
    _feed(dog, clock, 2000, BUF)
    assert dog.level is ShedLevel.NONE


def test_recovery_waits_out_a_full_window_before_restoring_work():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 1200, BUF * 1.1)
    assert dog.level is ShedLevel.NN_SHED
    _feed(dog, clock, 300, BUF)
    assert dog.level is ShedLevel.NN_SHED


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
    # A hardware-paced input alternates: one read returns a queued buffer
    # instantly, the next blocks for two buffer periods.
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


def test_a_deliberate_stall_sheds_unless_forgiven():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 2000, BUF)
    assert dog.level is ShedLevel.NONE

    clock.advance(0.2)
    dog.observe()
    assert dog.level is ShedLevel.NN_SHED


def test_an_unforgiven_stall_costs_about_ten_seconds_of_shed():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 2000, BUF)
    clock.advance(0.2)
    dog.observe()

    shed_sec = 0.0
    for _ in range(int(30.0 / BUF)):
        clock.advance(BUF)
        dog.observe()
        if dog.level is not ShedLevel.NONE:
            shed_sec += BUF
    assert 9.0 < shed_sec < 11.0


def test_forgiving_a_deliberate_stall_prevents_the_shed():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 2000, BUF)

    clock.advance(0.2)
    dog.forgive(0.2)
    _feed(dog, clock, 200, BUF)
    assert dog.level is ShedLevel.NONE
    assert abs(dog.total_drift_sec) < 0.05


def test_a_stage_fault_sheds_while_the_loop_keeps_perfect_time():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 2000, BUF)
    assert dog.report_fault('cuda_error') is ShedLevel.NN_SHED
    assert _feed(dog, clock, 2000, BUF) is ShedLevel.NN_SHED
    assert dog.drift_sec == pytest.approx(0.0, abs=1e-9)


def test_a_fault_holds_the_shed_until_the_stage_says_it_is_back():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    dog.report_fault('hung_pass')
    _feed(dog, clock, 5000, BUF * 0.2)
    assert dog.level is ShedLevel.NN_SHED
    assert dog.report_healthy() is ShedLevel.NONE


def test_the_two_inputs_are_two_locks_on_one_door():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    _feed(dog, clock, 1200, BUF * 1.1)
    dog.report_fault('ring_overrun')
    assert dog.report_healthy() is ShedLevel.NN_SHED, 'drift still over'
    _feed(dog, clock, 2000, BUF)
    assert dog.level is ShedLevel.NONE

    _feed(dog, clock, 1200, BUF * 1.1)
    dog.report_fault('ring_overrun')
    _feed(dog, clock, 2000, BUF)
    assert dog.level is ShedLevel.NN_SHED, 'fault still latched'


def test_the_fault_is_named_in_the_transition_log(caplog):
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    with caplog.at_level(logging.WARNING):
        dog.report_fault('cuda_error')
        dog.report_healthy()
    messages = ' | '.join(r.message for r in caplog.records)
    assert 'cuda_error' in messages
    assert 'NN_SHED' in messages
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) >= 1


def test_a_fault_reported_over_and_over_logs_one_transition(caplog):
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    with caplog.at_level(logging.WARNING):
        for _ in range(500):
            dog.report_fault('cuda_error')
    assert len(caplog.records) == 1


def test_a_different_fault_while_already_shed_is_still_recorded():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    dog.report_fault('cuda_error')
    dog.report_fault('hung_pass')
    assert dog.fault == 'hung_pass'
    assert dog.level is ShedLevel.NN_SHED


def test_reset_clears_a_latched_fault():
    clock = FakeClock()
    dog = DriftWatchdog(BUF, clock=clock)
    dog.report_fault('vram')
    dog.reset()
    assert dog.level is ShedLevel.NONE
    assert dog.fault is None
