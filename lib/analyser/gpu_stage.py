"""The GPU stage on its own thread, and what the show does when it stops."""
from __future__ import annotations

import logging
import sys
import threading
from collections import deque

import numpy as np

from lib.analyser.drift_watchdog import ShedLevel
from lib.analyser.mert_stream import RingOverrun
from lib.analyser.section_model import Drained
from lib.clock import SYSTEM_CLOCK, Clock

_QUEUE_PASSES = 4
_IDLE_WAIT_SEC = 0.05

# Orders of magnitude above a p95 pass, and Windows' own TDR gives up at 2 s.
_PASS_TIMEOUT_SEC = 5.0

_BACKOFF_SEC = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_LOG_INTERVAL_SEC = 30.0
_STATUS_INTERVAL_SEC = 30.0
_HEALTHY_PASSES = 10
_PASS_SAMPLES = 64
_STOP_JOIN_SEC = 2.0


def reserved_bytes():
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_reserved())
    except Exception:
        return None


class GpuStage:
    def __init__(self, posteriors, watchdog, *, clock: Clock = SYSTEM_CLOCK,
                 reinit=None, queue_passes: int = _QUEUE_PASSES,
                 pass_timeout_sec: float = _PASS_TIMEOUT_SEC) -> None:
        self.posteriors = posteriors
        self._watchdog = watchdog
        self._clock = clock
        self._reinit = reinit
        self._queue_passes = int(queue_passes)
        self._pass_timeout_sec = float(pass_timeout_sec)

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._queue: deque = deque()
        self._gap = False
        self._thread: threading.Thread | None = None
        self._running = False
        self._reset_requested = False
        self._pass_started_at: float | None = None
        self._shed = False
        self._attempts = 0
        self._clean = 0
        self._retry_at: float | None = None
        self._said: dict = {}
        self._suppressed: dict = {}
        self._status_at: float | None = None
        self._pass_sec: deque = deque(maxlen=_PASS_SAMPLES)

        self.passes = 0
        self.faults = 0
        self.overflows = 0
        self.reinits = 0
        self.resyncs = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._work, name="mert-gpu",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        thread, self._thread = self._thread, None
        if thread is None:
            return
        thread.join(_STOP_JOIN_SEC)
        if thread.is_alive():
            logging.warning('[gpu] the pass thread is still inside a pass and '
                            'was left to exit on its own')

    def push_audio(self, samples) -> Drained:
        self._check_for_a_hung_pass()
        if not self._reset_requested:
            self.posteriors.feed(samples)
            self._wake.set()
        return self._drain()

    def reset(self, ) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._take_reset()
            return
        self._reset_requested = True
        self._wake.set()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def shed(self) -> bool:
        return self._shed

    @property
    def idle(self) -> bool:
        return self._pass_started_at is None and not self.posteriors.due()

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def reset_pending(self) -> bool:
        return self._reset_requested

    def _drain(self) -> Drained:
        with self._lock:
            gap, self._gap = self._gap, False
            if self._reset_requested or self._watchdog.level is not ShedLevel.NONE:
                return Drained(gap, [])
            out = [item for group in self._queue for item in group]
            self._queue.clear()
        return Drained(gap, out)

    def _check_for_a_hung_pass(self) -> None:
        started = self._pass_started_at
        if started is None:
            return
        if self._clock.monotonic() - started > self._pass_timeout_sec:
            self._watchdog.report_fault('hung_pass')

    def _work(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception as error:
                self._fault('stage_error', error)
                self._sleep()

    def _tick(self) -> None:
        if self._reset_requested:
            self._take_reset()
            return
        shed = self._watchdog.level is not ShedLevel.NONE
        if shed != self._shed:
            self._enter_shed() if shed else self._leave_shed()
            return
        if shed:
            self._retry()
            self._sleep()
            return
        if not self.posteriors.due():
            self._sleep()
            return
        self._one_pass()

    def _sleep(self) -> None:
        self._wake.wait(_IDLE_WAIT_SEC)
        self._wake.clear()

    def _take_reset(self) -> None:
        self.posteriors.reset()
        with self._lock:
            self._queue.clear()
            self._gap = False
        self._attempts = 0
        self._clean = 0
        self._retry_at = None
        self._reset_requested = False

    def _enter_shed(self) -> None:
        with self._lock:
            self._queue.clear()
            self._gap = True
        self._say('shed', f'[gpu] shed: holding the intent, the hand-off queue '
                          f'is dropped and the decoder is reset '
                          f'({self.passes} passes so far)')
        self._shed = True

    def _leave_shed(self) -> None:
        self.resyncs += 1
        record = self.posteriors.resync()
        with self._lock:
            self._queue.clear()
            self._gap = True
        self._say('restored', f'[gpu] restored at the live edge: skipped '
                              f'{record.lost_sec:.2f}s / {record.cells_lost} '
                              f'cells, resuming at cell '
                              f'{record.first_cell_index}')
        self._shed = False

    def _one_pass(self) -> None:
        self._pass_started_at = self._clock.monotonic()
        try:
            produced = self.posteriors.run_pass()
        except RingOverrun as overrun:
            self._fault('ring_overrun', overrun)
            return
        except Exception as error:
            self._fault('pass_failed', error)
            return
        finally:
            took = self._clock.monotonic() - self._pass_started_at
            self._pass_started_at = None
        self._pass_sec.append(took)
        self.passes += 1
        self._clean += 1
        self._retry_at = None
        if self._clean >= _HEALTHY_PASSES:
            self._attempts = 0
            self._said.clear()
            self._suppressed.clear()
        if self._offer(produced):
            self._watchdog.report_healthy()
        self._status()

    def _offer(self, produced) -> bool:
        if not produced:
            return True
        with self._lock:
            room = len(self._queue) < self._queue_passes
            if room:
                self._queue.append(produced)
        if room:
            return True
        self.overflows += 1
        self._fault('queue_overflow',
                    f'{self._queue_passes} passes are waiting for a '
                    f'consumer that has stopped draining')
        return False

    def _fault(self, kind: str, detail) -> None:
        self.faults += 1
        self._clean = 0
        self._say(kind, f'[gpu] {kind}: {detail} — the show holds whatever '
                        f'intent it has (a quiet floor at start-up) and keeps '
                        f'beats (fault #{self.faults}, {self.reinits} reinit(s))')
        self._retry_at = self._clock.monotonic() + self._backoff()
        self._watchdog.report_fault(kind)

    def _backoff(self) -> float:
        return _BACKOFF_SEC[min(self._attempts, len(_BACKOFF_SEC) - 1)]

    def _retry(self) -> None:
        if self._watchdog.fault is None:
            return
        now = self._clock.monotonic()
        if self._retry_at is not None and now < self._retry_at:
            return
        self._attempts += 1
        self._retry_at = now + self._backoff()
        if self._attempts > 1 and self._reinit is not None and not self._reinitialise():
            return
        self._watchdog.report_healthy()

    def _reinitialise(self) -> bool:
        self.reinits += 1
        try:
            self.posteriors.set_encoder(self._reinit())
        except Exception as error:
            self._say('reinit_failed',
                      f'[gpu] reinit_failed: {error} — the show holds its '
                      f'intent and keeps beats (fault #{self.faults}, '
                      f'{self.reinits} reinit(s))')
            return False
        self._say('reinit', f'[gpu] extractor reinitialised (attempt '
                            f'{self._attempts}, {self.reinits} so far)')
        return True

    def _say(self, kind: str, message: str) -> None:
        now = self._clock.monotonic()
        last = self._said.get(kind)
        if last is not None and now - last < _LOG_INTERVAL_SEC:
            self._suppressed[kind] = self._suppressed.get(kind, 0) + 1
            return
        self._said[kind] = now
        more = self._suppressed.pop(kind, 0)
        logging.warning(message + (f' [+{more} more since the last line]'
                                   if more else ''))

    def _status(self) -> None:
        now = self._clock.monotonic()
        if self._status_at is not None and now - self._status_at < _STATUS_INTERVAL_SEC:
            return
        self._status_at = now
        if not self._pass_sec:
            return
        took = np.asarray(self._pass_sec, dtype=np.float64) * 1000.0
        reserved = reserved_bytes()
        logging.info(
            f'[gpu] {self.passes} passes | last {len(took)}: mean '
            f'{took.mean():.0f}ms max {took.max():.0f}ms | queue '
            f'{len(self._queue)}/{self._queue_passes}'
            + ('' if reserved is None
               else f' | cuda reserved {reserved / 1e6:.0f}MB'))
