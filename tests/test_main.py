import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from lib.main import SoundSwitchAutoPilot


class _StopRun(Exception):
    ...


def _app(enable_ui: bool, event_buffer):
    app = object.__new__(SoundSwitchAutoPilot)
    app.disable_os2l = True
    app.enable_ui = enable_ui
    app._ui_port = 8050
    app._enable_playback = False
    app.event_buffer = event_buffer
    app.audio_client = MagicMock()
    app.audio_client.read.side_effect = _StopRun
    app.midi_client = MagicMock()
    app.os2l_client = MagicMock()
    app.overlay_client = MagicMock()
    app.section = None
    app.is_running = False
    return app


def _threads_started(monkeypatch) -> list:
    started = []

    def record(**kwargs):
        started.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(threading, 'Thread', record)
    return started


def _run(app):
    with pytest.raises(_StopRun):
        asyncio.run(app.run())


def test_report_without_ui_starts_no_dash_server(monkeypatch):
    started = _threads_started(monkeypatch)
    _run(_app(enable_ui=False, event_buffer=MagicMock()))
    assert started == []


def test_ui_starts_the_dash_server(monkeypatch):
    started = _threads_started(monkeypatch)
    _run(_app(enable_ui=True, event_buffer=MagicMock()))
    assert [t['target'].__name__ for t in started] == ['run_app']


def test_report_without_ui_still_starts_the_event_buffer(monkeypatch):
    _threads_started(monkeypatch)
    buffer = MagicMock()
    _run(_app(enable_ui=False, event_buffer=buffer))
    buffer.start.assert_called_once()


def test_no_buffer_and_no_ui_starts_neither(monkeypatch):
    started = _threads_started(monkeypatch)
    _run(_app(enable_ui=False, event_buffer=None))
    assert started == []


def test_a_raise_in_the_loop_still_blanks_the_rig_and_stops_every_thread(monkeypatch):
    _threads_started(monkeypatch)
    app = _app(enable_ui=False, event_buffer=None)
    app.section = MagicMock()

    _run(app)

    app.audio_client.close.assert_called_once()
    app.section.stop.assert_called_once()
    app.os2l_client.stop.assert_called_once()
    app.midi_client.stop.assert_called_once()
    app.overlay_client.stop.assert_called_once()
    assert app.is_running is False


def test_one_client_failing_to_close_does_not_strand_the_rest(monkeypatch):
    _threads_started(monkeypatch)
    app = _app(enable_ui=False, event_buffer=None)
    app.os2l_client.stop.side_effect = OSError('socket already gone')

    _run(app)

    app.midi_client.stop.assert_called_once()
    app.overlay_client.stop.assert_called_once()


def test_the_overlay_clear_is_transmitted_rather_than_queued(monkeypatch):
    _threads_started(monkeypatch)
    app = _app(enable_ui=False, event_buffer=None)

    _run(app)

    app.overlay_client.flush_messages.assert_called()

def _built_buffer(monkeypatch, *, enable_ui, report_path):
    from lib import section_chain

    for module, name in (
            ('lib.clients.pyaudio_client', 'PyAudioClient'),
            ('lib.clients.midi_client', 'MidiClient'),
            ('lib.clients.os2l_client', 'Os2lClient'),
            ('lib.clients.overlay_client', 'OverlayClient'),
            ('lib.analyser.music_analyser', 'MusicAnalyser'),
            ('lib.engine.light_engine', 'LightEngine'),
            ('lib.engine.effect_controller', 'EffectController')):
        monkeypatch.setattr(f'{module}.{name}', MagicMock())
    monkeypatch.setattr(section_chain, 'artifacts_present', lambda *a, **k: False)
    monkeypatch.setattr(asyncio, 'get_event_loop', MagicMock())

    app = SoundSwitchAutoPilot(midi_port_index=0, enable_ui=enable_ui,
                               report_path=report_path)
    return app.event_buffer


def test_a_session_report_keeps_the_whole_session(monkeypatch):
    from lib.clock import VirtualClock
    from lib.engine.event_buffer import EventBuffer

    assert _built_buffer(monkeypatch, enable_ui=False,
                         report_path='out.json')._window_sec == float('inf')

    def half_an_hour(window):
        clock = VirtualClock()
        buffer = EventBuffer(window_sec=window, clock=clock)
        buffer.start()
        for index in range(60):
            clock.advance(30.0)
            buffer.set_intent(f'intent-{index % 2}', song_sec=float(index))
        return buffer.to_report()['metrics']['intent_changes_count']

    assert half_an_hour(60.0) < 10, 'the window used not to prune here'
    assert half_an_hour(float('inf')) == 60


def test_the_live_view_keeps_its_rolling_window(monkeypatch):
    buffer = _built_buffer(monkeypatch, enable_ui=True, report_path=None)
    assert buffer._window_sec == 60.0
