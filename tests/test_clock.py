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


def test_the_system_clock_can_resolve_a_single_audio_buffer():
    """Everything in the pipeline is timed against a 5.805 ms buffer, and the
    drift watchdog measures spans of a few of them, so the clock underneath must
    resolve far finer than one.

    The assertion is on the OBSERVED granularity of `SYSTEM_CLOCK.monotonic`
    itself, not on which library function it happens to call. That is what makes
    it portable *and* discriminating: `time.monotonic` reports 15.625 ms on
    Windows/CPython 3.11 — coarser than the quantity being measured — and this
    test fails on it there, while on platforms where `time.monotonic` is already
    nanosecond-grade it correctly has nothing to complain about.
    """
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from lib.clock import SYSTEM_CLOCK

    buffer_sec = BUFFER_SIZE / SAMPLE_RATE
    samples = sorted({SYSTEM_CLOCK.monotonic() for _ in range(20000)})
    # A coarse clock shows up here first — 20000 reads of a 15.6 ms clock yield
    # a handful of distinct values — so this branch reports the real diagnosis
    # rather than letting a bare "did not advance" stand in for it.
    if len(samples) < 3:
        raise AssertionError(
            f'system clock produced only {len(samples)} distinct value(s) in '
            f'20000 reads — it is quantised far coarser than the '
            f'{buffer_sec * 1000:.3f} ms audio buffer it has to time')
    granularity = min(b - a for a, b in zip(samples, samples[1:]))
    assert granularity < buffer_sec / 10, (
        f'system clock granularity {granularity * 1000:.3f} ms cannot resolve a '
        f'{buffer_sec * 1000:.3f} ms audio buffer')
