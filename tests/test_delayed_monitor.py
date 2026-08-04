import pytest

from lib.clock import VirtualClock
from lib.delayed_monitor import DelayedMonitor


def _monitor(delay_sec=14.0):
    clock = VirtualClock()
    played = []
    return DelayedMonitor(delay_sec, played.append, clock=clock), clock, played


def test_nothing_reaches_the_headphones_until_the_delay_has_elapsed():
    monitor, clock, played = _monitor()
    for index in range(20):
        clock.advance(0.5)
        monitor.feed(index)
    assert played == []
    assert monitor.buffered == 20
    assert not monitor.started


def test_the_first_buffer_out_is_the_first_buffer_in_a_delay_ago():
    monitor, clock, played = _monitor()
    for index in range(30):
        clock.advance(1.0)
        monitor.feed(index)
    assert played[0] == 0
    assert played == list(range(len(played)))


def test_the_monitor_runs_one_delay_behind_the_analysis():
    monitor, clock, played = _monitor(delay_sec=14.0)
    period = 1.0
    lags = []
    for _ in range(40):
        clock.advance(period)
        before = len(played)
        monitor.feed(clock.monotonic())
        if len(played) > before:
            lags.append(clock.monotonic() - played[-1])

    assert lags, 'the monitor never started'
    assert all(14.0 - period <= lag <= 14.0 for lag in lags), lags


def test_a_stop_drops_the_tail_instead_of_playing_it_out():
    monitor, clock, played = _monitor()
    for index in range(20):
        clock.advance(1.0)
        monitor.feed(index)
    assert played

    monitor.silence()
    assert monitor.buffered == 0
    assert not monitor.started

    before = list(played)
    for index in range(100, 110):
        clock.advance(1.0)
        monitor.feed(index)
    assert played == before, 'the dropped tail played anyway'


def test_a_stop_re_arms_the_delay_rather_than_collapsing_it():
    monitor, clock, played = _monitor()
    for index in range(30):
        clock.advance(1.0)
        monitor.feed(index)
    monitor.silence()
    from_the_old_song = list(played)

    for index in range(200, 213):
        clock.advance(1.0)
        monitor.feed(index)
    assert played == from_the_old_song, 'the delay collapsed after the stop'

    clock.advance(1.0)
    monitor.feed(213)
    assert played[-1] == 200, 'the next song did not start a fresh delay'


def test_arming_twice_does_not_shorten_the_wait():
    monitor, clock, played = _monitor()
    clock.advance(13.0)
    monitor.arm()
    clock.advance(13.0)
    monitor.feed('early')
    assert played == []
    clock.advance(1.1)
    monitor.feed('late')
    assert played == ['early']


def _file_args(**overrides):
    import argparse

    from simulate.cli import add_simulate_subparser

    parser = argparse.ArgumentParser()
    add_simulate_subparser(parser.add_subparsers(dest='command'))
    argv = ['simulate', 'file', 'song.mp3']
    for flag, value in overrides.items():
        argv += [flag] if value is True else [flag, str(value)]
    return parser.parse_args(argv)


def test_a_plain_file_run_stays_fast_and_headless():
    from simulate.cli import paced

    assert not paced(_file_args())


def test_asking_for_the_headphones_paces_the_run_even_without_the_viewer():
    from simulate.cli import paced

    assert paced(_file_args(**{'-o': 17}))
    assert _file_args(**{'-o': 17}).output_device_index == 17


def test_the_viewer_paces_the_run_as_it_always_did():
    from simulate.cli import paced

    assert paced(_file_args(**{'--ui': True}))


def test_the_monitor_delay_is_the_one_the_lights_are_aligned_to():
    from simulate.runner import PLAYBACK_DELAY_SEC
    from lib.main import PLAYBACK_DELAY_SEC as LIVE_DELAY_SEC

    assert PLAYBACK_DELAY_SEC == LIVE_DELAY_SEC
