import http.client
import json
import logging
import socket
import subprocess
import sys
import time

import pytest

from lib import ui_bridge
from lib.clock import VirtualClock
from lib.engine.event_buffer import EventBuffer


@pytest.fixture
def served():
    clock = VirtualClock()
    buffer = EventBuffer(window_sec=float('inf'), clock=clock,
                         look_ahead_sec=14.0)
    buffer.start()
    server = ui_bridge.SnapshotServer(buffer, 0)
    server.start()
    try:
        yield buffer, server, clock
    finally:
        server.stop()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind((ui_bridge.SNAPSHOT_HOST, 0))
        return probe.getsockname()[1]


def _free_ui_port() -> int:
    return _free_port() - 1


def _bound(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex((ui_bridge.SNAPSHOT_HOST, port)) == 0


def _get(port: int, path: str = ui_bridge.SNAPSHOT_PATH):
    connection = http.client.HTTPConnection(ui_bridge.SNAPSHOT_HOST, port,
                                            timeout=5)
    try:
        connection.request('GET', path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _poller(port: int):
    from simulate.visualizer_app import SnapshotPoller

    return SnapshotPoller(port=port, timeout=5.0)


def test_the_endpoint_serves_exactly_what_the_buffer_snapshots(served):
    buffer, server, clock = served
    buffer.add_beat(bpm=128.0, change=False)
    clock.advance(1.0)
    buffer.set_intent('drop', song_sec=1.0)
    buffer.set_decoder_state(classes=['drop'], posterior=[1.0],
                             observed_bar=3, committed_bar=2,
                             committed_label='drop', lag_bars=1,
                             chain_latency_sec=13.66)

    status, body = _get(server.port)
    assert status == 200
    assert json.loads(body) == buffer.snapshot()


def test_the_show_serves_nothing_else(served):
    _, server, _ = served
    assert _get(server.port, '/')[0] == 404


def test_every_field_the_panels_read_survives_the_json_round_trip(served):
    buffer, server, clock = served
    buffer.set_playing(True)
    buffer.add_beat(bpm=127.5, change=True, rms=0.1234)
    clock.advance(2.0)
    buffer.set_timing_log([{'label': 'beat', 'target_delta_sec': 14.0,
                            'actual_delta_sec': 14.004}])

    served_snapshot = json.loads(_get(server.port)[1])
    assert served_snapshot == buffer.snapshot()
    assert set(served_snapshot) == set(buffer.snapshot())


def test_the_poller_hands_the_viewer_what_the_show_published(served):
    buffer, server, _ = served
    buffer.add_beat(bpm=128.0, change=False)
    poller = _poller(server.port)
    assert poller.snapshot()['beats_detected'] == 1
    buffer.add_beat(bpm=128.0, change=False)
    assert poller.snapshot()['beats_detected'] == 2


def test_a_viewer_that_outlives_the_show_holds_its_last_frame(served, caplog):
    buffer, server, _ = served
    buffer.add_beat(bpm=128.0, change=False)
    poller = _poller(server.port)
    held = poller.snapshot()

    server.stop()
    # A stopped server keeps serving connections already open; a show that has
    # exited does not, so drop the keep-alive the way its death would.
    poller._connection.close()

    with caplog.at_level(logging.WARNING):
        assert poller.snapshot() == held
        assert poller.snapshot() == held
    assert sum('stopped answering' in record.message
               for record in caplog.records) == 1


def test_a_viewer_that_starts_before_the_show_renders_a_blank_frame():
    from simulate import visualizer_app as V

    snapshot = _poller(_free_port()).snapshot()
    assert snapshot == V.BLANK_SNAPSHOT
    assert V._build_timeline(snapshot) is not None
    assert V._build_metrics(snapshot)
    assert V._build_stage(snapshot)


def test_the_snapshot_port_is_derived_from_the_dash_port():
    assert ui_bridge.snapshot_port(8050) == 8051
    assert ui_bridge.snapshot_port(9000) == 9001


def test_the_viewer_is_launched_as_its_own_process_on_both_ports():
    command = ui_bridge.viewer_command(9000)
    assert command[:3] == [sys.executable, '-m', ui_bridge.VIEWER_MODULE]
    assert command[command.index('--port') + 1] == '9000'
    assert command[command.index('--snapshot-port') + 1] == '9001'


def test_killing_the_viewer_kills_the_whole_tree(tmp_path):
    # The venv launcher re-execs, so the recorded pid is a parent: a plain
    # terminate leaves the real viewer running and invisible.
    port = _free_port()
    child = tmp_path / 'child.py'
    child.write_text('import socket, sys\n'
                     's = socket.socket()\n'
                     's.bind(("127.0.0.1", int(sys.argv[1])))\n'
                     's.listen(64)\n'
                     'while True:\n'
                     '    s.accept()[0].close()\n')
    parent_py = tmp_path / 'parent.py'
    parent_py.write_text('import subprocess, sys, time\n'
                         'subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n'
                         'time.sleep(120)\n')
    parent = subprocess.Popen(
        [sys.executable, str(parent_py), str(child), str(port)])
    deadline = time.monotonic() + 20
    while not _bound(port) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _bound(port), 'the grandchild never came up'

    ui_bridge.kill_tree(parent)

    assert parent.poll() is not None
    deadline = time.monotonic() + 10
    while _bound(port) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _bound(port), 'the grandchild outlived the pid we killed'


def test_killing_a_viewer_that_already_died_is_not_an_error():
    process = subprocess.Popen([sys.executable, '-c', ''])
    process.wait()
    ui_bridge.kill_tree(process)


def test_a_port_the_show_cannot_bind_costs_it_the_viewer_and_nothing_else(
        served, caplog):
    _, server, _ = served
    with caplog.at_level(logging.WARNING):
        assert ui_bridge.start(object(), server.port - 1) is None
    assert 'no snapshot endpoint' in caplog.text
    assert _get(server.port)[0] == 200


def test_a_viewer_that_will_not_start_costs_the_show_nothing(monkeypatch,
                                                             caplog):
    monkeypatch.setattr(ui_bridge, 'viewer_command',
                        lambda port: ['this-command-does-not-exist'])
    with caplog.at_level(logging.WARNING):
        assert ui_bridge.start(object(), _free_ui_port()) is None
    assert 'did not start' in caplog.text


def test_a_viewer_dying_on_its_own_is_reported_once_and_the_show_runs_on(
        monkeypatch, caplog):
    monkeypatch.setattr(
        ui_bridge, 'viewer_command',
        lambda port: [sys.executable, '-c', 'raise SystemExit(3)'])
    clock = VirtualClock()
    buffer = EventBuffer(clock=clock)
    buffer.start()
    buffer.add_beat(bpm=128.0, change=False)

    with caplog.at_level(logging.WARNING):
        bridge = ui_bridge.start(buffer, _free_ui_port())
        assert bridge is not None
        bridge._watcher.join(timeout=20)

    try:
        assert 'the viewer process exited (3)' in caplog.text
        assert json.loads(_get(bridge._server.port)[1])['beats_detected'] == 1
    finally:
        bridge.stop()


def test_stopping_the_show_takes_the_viewer_and_the_endpoint_with_it(
        monkeypatch):
    monkeypatch.setattr(
        ui_bridge, 'viewer_command',
        lambda port: [sys.executable, '-c', 'import time; time.sleep(120)'])
    clock = VirtualClock()
    buffer = EventBuffer(clock=clock)
    buffer.start()

    ui_port = _free_ui_port()
    bridge = ui_bridge.start(buffer, ui_port)
    assert bridge is not None
    assert _bound(ui_bridge.snapshot_port(ui_port))

    bridge.stop()

    assert bridge.viewer.poll() is not None
    assert not _bound(ui_bridge.snapshot_port(ui_port))
