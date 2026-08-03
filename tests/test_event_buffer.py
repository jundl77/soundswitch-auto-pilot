import pytest

from lib.clock import VirtualClock
from lib.engine.event_buffer import EventBuffer


def test_timestamps_use_injected_clock():
    clock = VirtualClock()
    buf = EventBuffer(clock=clock)
    buf.start()
    clock.advance(12.5)
    buf.add_beat(bpm=128.0, change=False)
    snap = buf.snapshot()
    assert snap['now'] == 12.5
    assert snap['beats'][0]['t'] == 12.5


def _ui_session(minutes: float, look_ahead_sec: float = 14.0):
    clock = VirtualClock()
    buffer = EventBuffer(window_sec=float('inf'), clock=clock,
                         look_ahead_sec=look_ahead_sec)
    buffer.start()
    buffer.set_playing(True)
    for step in range(int(minutes * 120)):
        clock.advance(0.5)
        buffer.add_beat(bpm=128.0, change=False)
        if step % 60 == 0:
            buffer.set_intent(f'intent_{step}')
    return buffer, clock


def test_a_long_session_snapshots_a_window_but_reports_the_whole_track():
    buffer, _ = _ui_session(minutes=20.0)
    snapshot = buffer.snapshot()
    report = buffer.to_report()

    assert len(report['beats']) == 2400
    assert snapshot['beats_detected'] == 2400
    assert len(snapshot['beats']) < 150
    assert len(snapshot['intents']) < 10
    oldest = snapshot['beats'][0]['t']
    assert snapshot['now'] - oldest <= EventBuffer.SNAPSHOT_WINDOW_SEC + 14.0


def test_the_snapshot_stops_growing_once_the_window_is_full():
    short, _ = _ui_session(minutes=5.0)
    long, _ = _ui_session(minutes=20.0)
    assert len(long.snapshot()['beats']) == len(short.snapshot()['beats'])


def test_the_snapshot_reaches_back_far_enough_to_fill_the_display():
    buffer, _ = _ui_session(minutes=10.0, look_ahead_sec=14.0)
    snapshot = buffer.snapshot()
    reach = snapshot['now'] - snapshot['beats'][0]['t']
    assert reach >= EventBuffer.SNAPSHOT_WINDOW_SEC + 14.0 - 1.0


def test_the_sound_events_are_never_windowed():
    buffer, clock = _ui_session(minutes=20.0)
    assert buffer.snapshot()['sound_events'][0]['t'] == 0.0


def test_infinite_window_keeps_old_events():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    buf.set_intent('GROOVE')
    buf.add_effect('CH_A', 'AUTOLOOP')
    clock.advance(500.0)
    buf.set_intent('DROP')
    buf.add_effect('CH_B', 'AUTOLOOP')
    report = buf.to_report()
    assert [e['intent'] for e in report['intents']] == ['GROOVE', 'DROP']
    assert report['intents'][0]['t'] == 0.0
    assert [e['channel'] for e in report['effects']] == ['CH_A', 'CH_B']


def test_mark_end_freezes_timeline():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    clock.advance(10.0)
    buf.mark_end()
    clock.advance(2.5)
    buf.set_intent('DROP')
    report = buf.to_report()
    assert report['duration_sec'] == 10.0
    assert report['intents'][-1]['t'] == 10.0


def test_infinite_window_beats_are_unbounded():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    for _ in range(3005):
        buf.add_beat(bpm=128.0, change=False)
    assert buf.to_report()['metrics']['beats_detected'] == 3005


def test_default_window_prunes_old_effects():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=60.0, clock=clock)
    buf.start()
    buf.add_effect('CH_A', 'AUTOLOOP')
    clock.advance(1.0)
    buf.add_effect('CH_B', 'AUTOLOOP')
    clock.advance(500.0)
    buf.add_effect('CH_C', 'AUTOLOOP')
    report = buf.to_report()
    channels = [e['channel'] for e in report['effects']]
    assert 'CH_A' not in channels
    assert 'CH_B' in channels and 'CH_C' in channels


def test_a_beat_row_carries_only_what_the_pipeline_still_measures():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    buf.add_beat(128.0, False, rms=0.42)
    beat = buf.to_report()['beats'][0]
    assert set(beat) == {'t', 'bpm', 'strength', 'change', 'rms'}
    assert beat['rms'] == pytest.approx(0.42)


def test_add_beat_defaults_are_neutral():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    buf.add_beat(128.0, False)
    beat = buf.to_report()['beats'][0]
    assert beat['rms'] == 0.0
    assert beat['strength'] == 0.0


def test_report_records_look_ahead_offset():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock, look_ahead_sec=2.5)
    buf.start()
    buf.add_beat(128.0, False)
    assert buf.to_report()['metrics']['look_ahead_sec'] == pytest.approx(2.5)


def test_report_look_ahead_defaults_to_zero():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    assert buf.to_report()['metrics']['look_ahead_sec'] == 0.0
