"""Tests for EventBuffer virtual-clock timestamps and infinite window."""
from lib.clock import VirtualClock
from lib.engine.event_buffer import EventBuffer


def test_timestamps_use_injected_clock():
    clock = VirtualClock()
    buf = EventBuffer(clock=clock)
    buf.start()
    clock.advance(12.5)
    buf.add_beat(bpm=128.0, onset_density=4.0, change=False)
    snap = buf.snapshot()
    assert snap['now'] == 12.5
    assert snap['beats'][0]['t'] == 12.5


def test_infinite_window_keeps_old_events():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    buf.set_intent('GROOVE')
    buf.add_effect('CH_A', 'AUTOLOOP')
    clock.advance(500.0)  # far beyond the default 60 s window
    buf.set_intent('DROP')
    buf.add_effect('CH_B', 'AUTOLOOP')
    report = buf.to_report()
    assert [e['intent'] for e in report['intents']] == ['GROOVE', 'DROP']
    assert report['intents'][0]['t'] == 0.0
    assert [e['channel'] for e in report['effects']] == ['CH_A', 'CH_B']


def test_default_window_prunes_old_effects():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=60.0, clock=clock)
    buf.start()
    buf.add_effect('CH_A', 'AUTOLOOP')
    clock.advance(1.0)
    buf.add_effect('CH_B', 'AUTOLOOP')   # closes CH_A at end=1.0
    clock.advance(500.0)
    buf.add_effect('CH_C', 'AUTOLOOP')   # closes CH_B; CH_A end=1.0 < cutoff → pruned
    report = buf.to_report()
    channels = [e['channel'] for e in report['effects']]
    assert 'CH_A' not in channels
    assert 'CH_B' in channels and 'CH_C' in channels
