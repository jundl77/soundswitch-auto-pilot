"""The show's only UI surface: the event buffer over HTTP, and the viewer process."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SNAPSHOT_HOST = '127.0.0.1'
SNAPSHOT_PATH = '/snapshot'
VIEWER_MODULE = 'simulate.visualizer_app'
_REPO_ROOT = Path(__file__).resolve().parents[1]
_KILL_GRACE_SEC = 5.0


def snapshot_port(ui_port: int) -> int:
    return ui_port + 1


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:
        if self.path != SNAPSHOT_PATH:
            self.send_error(404)
            return
        body = json.dumps(self.server.event_buffer.snapshot()).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


class _Http(ThreadingHTTPServer):
    daemon_threads = True
    # A second show must fail to bind rather than quietly share the port.
    allow_reuse_address = False


class SnapshotServer:
    def __init__(self, event_buffer, port: int):
        self._http = _Http((SNAPSHOT_HOST, port), _Handler)
        self._http.event_buffer = event_buffer
        self._thread = threading.Thread(target=self._http.serve_forever,
                                        name='snapshot-server', daemon=True)

    @property
    def port(self) -> int:
        return self._http.server_address[1]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._http.shutdown()
        self._http.server_close()


def viewer_command(ui_port: int) -> list:
    return [sys.executable, '-m', VIEWER_MODULE,
            '--port', str(ui_port),
            '--snapshot-port', str(snapshot_port(ui_port))]


def _spawn(command: list) -> subprocess.Popen:
    detach = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
              if os.name == 'nt' else {'start_new_session': True})
    return subprocess.Popen(command, cwd=str(_REPO_ROOT), **detach)


def kill_tree(process: subprocess.Popen) -> None:
    """The venv's python.exe re-execs the real interpreter, so the pid is a parent."""
    if process.poll() is not None:
        return
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True)
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except OSError as error:
        logging.warning(f'[ui] could not signal the viewer ({error!r})')
    try:
        process.wait(timeout=_KILL_GRACE_SEC)
    except subprocess.TimeoutExpired:
        process.kill()


class UiBridge:
    def __init__(self, server: SnapshotServer, viewer: subprocess.Popen,
                 ui_port: int):
        self._server = server
        self._viewer = viewer
        self._ui_port = ui_port
        self._stopping = False
        self._watcher = threading.Thread(target=self._watch,
                                         name='viewer-watch', daemon=True)
        self._watcher.start()

    @property
    def viewer(self) -> subprocess.Popen:
        return self._viewer

    def _watch(self) -> None:
        code = self._viewer.wait()
        if not self._stopping:
            logging.warning(f'[ui] the viewer process exited ({code}) — the '
                            f'show is unaffected; relaunch it with '
                            f'`python -m {VIEWER_MODULE} --port {self._ui_port}`')

    def wait(self) -> None:
        self._viewer.wait()

    def stop(self) -> None:
        self._stopping = True
        kill_tree(self._viewer)
        self._server.stop()


def start(event_buffer, ui_port: int) -> UiBridge | None:
    try:
        server = SnapshotServer(event_buffer, snapshot_port(ui_port))
        server.start()
    except OSError as error:
        logging.warning(f'[ui] no snapshot endpoint on port '
                        f'{snapshot_port(ui_port)} ({error!r}) — the show runs '
                        f'without a viewer')
        return None
    try:
        viewer = _spawn(viewer_command(ui_port))
    except OSError as error:
        logging.warning(f'[ui] the viewer process did not start ({error!r}) — '
                        f'the show runs without it')
        server.stop()
        return None
    logging.info(f'[ui] visualizer → http://localhost:{ui_port} '
                 f'(pid {viewer.pid}, reading :{server.port}{SNAPSHOT_PATH})')
    return UiBridge(server, viewer, ui_port)
