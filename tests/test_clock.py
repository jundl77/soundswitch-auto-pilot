"""Tests for the injectable Clock abstraction (lib/clock.py)."""
import datetime
import time

from lib.clock import SystemClock, VirtualClock, SYSTEM_CLOCK


def test_system_clock_tracks_real_time():
    clock = SystemClock()
    assert abs((clock.now() - datetime.datetime.now()).total_seconds()) < 0.5
    assert abs(clock.monotonic() - time.monotonic()) < 0.5


def test_system_clock_singleton_is_system_clock():
    assert isinstance(SYSTEM_CLOCK, SystemClock)


def test_virtual_clock_starts_at_zero_and_fixed_epoch():
    clock = VirtualClock()
    assert clock.monotonic() == 0.0
    assert clock.now() == datetime.datetime(2000, 1, 1)


def test_virtual_clock_advance_moves_both_readings():
    clock = VirtualClock()
    clock.advance(2.5)
    clock.advance(0.5)
    assert clock.monotonic() == 3.0
    assert clock.now() == datetime.datetime(2000, 1, 1, 0, 0, 3)


def test_virtual_clock_is_deterministic_across_instances():
    a, b = VirtualClock(), VirtualClock()
    for _ in range(1000):
        a.advance(256 / 44100)
        b.advance(256 / 44100)
    assert a.monotonic() == b.monotonic()
    assert a.now() == b.now()
