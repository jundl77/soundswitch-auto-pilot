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
