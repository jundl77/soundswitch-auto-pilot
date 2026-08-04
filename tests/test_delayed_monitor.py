import asyncio

import pytest

from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
from lib.clock import VirtualClock
from lib.delayed_monitor import DelayedMonitor
from simulate.runner import run_simulation


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


class _FiniteAudio:
    def __init__(self, buffers: int):
        self._left = buffers
        self._served = 0

    def start_streams(self):
        pass

    @property
    def exhausted(self) -> bool:
        return self._left <= 0

    def read(self):
        self._left -= 1
        self._served += 1
        return self._served - 1

    def close(self):
        pass


class _Idle:
    async def analyse(self, signal):
        return signal

    async def on_audio(self, signal):
        pass

    async def on_100ms_callback(self):
        pass

    async def on_1sec_callback(self):
        pass

    async def on_10sec_callback(self):
        pass


class _NoCommands:
    delay_sec = 0.0
    pending = False

    async def drain(self):
        pass


def _components(buffers: int) -> dict:
    idle = _Idle()
    return {'audio_client': _FiniteAudio(buffers), 'music_analyser': idle,
            'light_engine': idle, 'midi_client': idle,
            'command_queue': _NoCommands()}


def _buffers_for(seconds: float) -> int:
    return int(seconds * SAMPLE_RATE / BUFFER_SIZE)


def test_setup_latency_is_not_deducted_from_the_monitors_delay():
    monitor, clock, played = _monitor()
    clock.advance(30.0)
    asyncio.run(run_simulation(_components(_buffers_for(13.0)), float('inf'),
                               clock=clock, monitor=monitor))
    assert played == []


def test_the_headphones_join_one_full_delay_after_the_first_buffer():
    clock = VirtualClock()
    heard = []
    monitor = DelayedMonitor(
        14.0, lambda buf: heard.append((clock.monotonic(), buf)), clock=clock)
    clock.advance(30.0)
    start = clock.monotonic()

    asyncio.run(run_simulation(_components(_buffers_for(16.0)), float('inf'),
                               clock=clock, monitor=monitor))

    assert heard, 'the monitor never joined the room'
    at, first = heard[0]
    assert first == 0
    assert 14.0 <= at - start < 14.0 + 2 * BUFFER_SIZE / SAMPLE_RATE


def test_the_tail_the_room_has_not_heard_plays_out_when_the_file_ends():
    monitor, clock, played = _monitor()
    for index in range(20):
        clock.advance(0.5)
        monitor.feed(index)
    assert played == [], 'the fixture should still be inside the delay'

    asyncio.run(run_simulation(_components(0), 60.0, pace_real_time=True,
                               monitor=monitor))

    assert played == list(range(20))


def test_a_persistent_stop_cuts_the_tail_rather_than_playing_it_out_at_the_end():
    monitor, clock, played = _monitor()
    for index in range(20):
        clock.advance(0.5)
        monitor.feed(index)
    monitor.silence()

    asyncio.run(run_simulation(_components(0), 60.0, pace_real_time=True,
                               monitor=monitor))

    assert played == []


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


def test_the_retired_play_audio_flag_is_gone_from_the_parser():
    with pytest.raises(SystemExit):
        _file_args(**{'--play-audio': True})


def test_the_retired_play_audio_flag_leaves_no_plumbing_behind():
    assert not hasattr(_file_args(), 'play_audio')


def test_the_monitor_delay_is_the_one_the_lights_are_aligned_to():
    from simulate.runner import PLAYBACK_DELAY_SEC
    from lib.main import PLAYBACK_DELAY_SEC as LIVE_DELAY_SEC

    assert PLAYBACK_DELAY_SEC == LIVE_DELAY_SEC
