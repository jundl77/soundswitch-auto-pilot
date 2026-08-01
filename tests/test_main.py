import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from lib.main import SoundSwitchAutoPilot


class _StopRun(Exception):
    """Raised out of the first audio read, to end `run`'s loop after setup."""


def _app(enable_ui: bool, event_buffer):
    """`run`'s setup path without the DSP stack the constructor builds.

    The constructor is 1.3 GB of encoder and four hardware clients; the
    question here is only which of two flags opens the Dash server.
    """
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
    """`--report` needs the event buffer; it does not need a web server.

    Gating the UI thread on the buffer's existence handed anyone who asked for
    a session report a Dash server on 8050 they never asked for -- a port
    taken, and a render loop competing with the audio thread on the venue box.
    """
    started = _threads_started(monkeypatch)
    _run(_app(enable_ui=False, event_buffer=MagicMock()))
    assert started == []


def test_ui_starts_the_dash_server(monkeypatch):
    started = _threads_started(monkeypatch)
    _run(_app(enable_ui=True, event_buffer=MagicMock()))
    assert [t['target'].__name__ for t in started] == ['run_app']


def test_report_without_ui_still_starts_the_event_buffer(monkeypatch):
    """The half of the old gate that was doing real work."""
    _threads_started(monkeypatch)
    buffer = MagicMock()
    _run(_app(enable_ui=False, event_buffer=buffer))
    buffer.start.assert_called_once()


def test_no_buffer_and_no_ui_starts_neither(monkeypatch):
    started = _threads_started(monkeypatch)
    _run(_app(enable_ui=False, event_buffer=None))
    assert started == []


def test_a_raise_in_the_loop_still_blanks_the_rig_and_stops_every_thread(monkeypatch):
    """The teardown used to sit after the loop, so anything that raised out of
    it skipped all five closes.

    `MidiClient.stop` is what sets the rig's intensities to zero, and
    `Os2lSender`'s thread is NOT a daemon -- so the default failure was a
    traceback, a venue rig left lit at whatever the last effect was, a resident
    encoder, and an interpreter that then blocked at exit forever while the
    OS2L thread went on driving VirtualDJ.
    """
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
    """Ordered so the rig is blanked last; a wire that throws on the way out
    must not be what leaves it lit."""
    _threads_started(monkeypatch)
    app = _app(enable_ui=False, event_buffer=None)
    app.os2l_client.stop.side_effect = OSError('socket already gone')

    _run(app)

    app.midi_client.stop.assert_called_once()
    app.overlay_client.stop.assert_called_once()


def test_the_overlay_clear_is_transmitted_rather_than_queued(monkeypatch):
    """`OverlayClient.stop` only sets a flush flag, and the only thing that
    ever flushed was `LightEngine.on_cycle` -- inside the loop that has already
    exited.  After a clean Ctrl-C the venue kept the last DMX frame."""
    _threads_started(monkeypatch)
    app = _app(enable_ui=False, event_buffer=None)

    _run(app)

    app.overlay_client.flush_messages.assert_called()

def _built_buffer(monkeypatch, *, enable_ui, report_path):
    """Construct the app with every hardware client and the 1.3 GB chain stubbed,
    and hand back the `EventBuffer` it chose to build."""
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
    """`--report` promises the session; the default rolling window pruned it.

    Every write prunes at twice the window, so a thirty-minute set reported six
    of its sixty intent changes and a beat count pinned at the deque cap -- and
    the branch's own soak artifact is one of those reports.  The simulation's
    `_session_buffer` had already taken the window off for exactly this reason;
    production never got the same treatment.
    """
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
    """The UI draws 30 s and never reads back past it, so it pays for the
    window rather than holding a whole night of blocks."""
    buffer = _built_buffer(monkeypatch, enable_ui=True, report_path=None)
    assert buffer._window_sec == 60.0
