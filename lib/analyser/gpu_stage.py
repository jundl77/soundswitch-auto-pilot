"""The GPU stage on its own thread, and what the show does when it stops.

B3: one encoder pass is ~81 ms, ~210 ms at p95 under contention, against a
5.805 ms buffer period -- and the audio input DROPS rather than queues. Run
inline, the show throws away fourteen buffers of audio once a second, in
periodic gouges rather than as smooth lag, which is precisely the failure the
drift watchdog was built to notice and cannot fix. So the pass and the student
step that follows it move off the audio loop:

    audio thread   resample -> ring write -> a monotonic sample index, and the
                   drain of whatever the GPU thread has finished
    GPU thread     one pass per hop, the student step per cell, whole passes
                   handed off through a bounded queue
    consumer       the audio thread again -- the decoder feed and the engine
                   commit stay where the queue, the MIDI client and the event
                   buffer already live, so no show state is shared across
                   threads at all

Overflow of that queue is a shed event, never a stall: the audio thread's
contract is that it cannot be made to wait for the GPU under any condition,
including the GPU being dead.

**A shed keeps feeding the ring**, which looks like waste and is the opposite.
The extractor's sample index IS song time and is what every cell is stamped
from, so a stage that stops taking audio comes back with a clock that disagrees
with the beat grid it is decoded against -- silently, and for the rest of the
song. Resampling 256 samples and writing them costs microseconds; what a shed is
about is not spending 81 ms on a GPU that cannot do it.

**Both edges of a gap clear.** Audio from before a gap must never decode as
current -- the lesson the onset chain taught. Entering a shed drops the hand-off
queue and tells the consumer to reset the decoder; leaving one resyncs the
extractor past the gap and starts the student cold, because its window and its
carried state describe audio from before it.

**A persistent fault must not become its own outage** (D11/#144): the retry
backs off to one attempt per half minute and stays there, the logging is
rate-limited, and the show runs on beats and the silence timer for as long as
the GPU stays away. There is no second classifier and none is wanted.
"""
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

# Four passes is four seconds of posteriors. The consumer drains every buffer,
# so reaching this at all means it has been away for longer than any song
# boundary's MIDI settle.
_QUEUE_PASSES = 4

# How long the thread sleeps when there is nothing due. Audio wakes it, so this
# only bounds how quickly it notices a shed, a restore or a stop.
_IDLE_WAIT_SEC = 0.05

# A pass is 81 ms and 210 ms at p95. Two orders of magnitude above that is not a
# slow pass, it is a driver reset or a dead context -- and Windows' own TDR
# gives up at 2 s.
_PASS_TIMEOUT_SEC = 5.0

# The first retry is immediate because the commonest fault, a ring overrun, is
# already fixed by the resync the restore performs. After that a GPU that is not
# coming back must cost one attempt per half minute, forever, and no more.
_BACKOFF_SEC = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_FAULT_LOG_INTERVAL_SEC = 30.0
_STATUS_INTERVAL_SEC = 30.0
_PASS_SAMPLES = 64
_STOP_JOIN_SEC = 2.0


def reserved_bytes():
    """CUDA's reserved pool, or None when nothing has imported torch.

    Read through `sys.modules` rather than by importing: on a CPU box, and in
    every test that never loads a real encoder, torch is 2.5 GB this module has
    no reason to pull in. The number is here because the WDDM trap is silent --
    the driver spills to host memory under pressure and raises no OOM at all, so
    a run that is quietly crawling looks exactly like one that is not.
    """
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
    """`PosteriorStream` with a thread in front of it and a watchdog behind it.

    Thread ownership, which the two stages below already state and this object
    is what makes true: `push_audio`, `drain`, `reset`, `start` and `stop` are
    the audio thread's; everything else belongs to the worker.
    """

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
        self._retry_at: float | None = None
        self._logged_fault: str | None = None
        self._logged_fault_at: float | None = None
        self._status_at: float | None = None
        self._pass_sec: deque = deque(maxlen=_PASS_SAMPLES)

        self.passes = 0
        self.faults = 0
        self.overflows = 0
        self.reinits = 0
        self.resyncs = 0

    # -- what the audio thread may touch ------------------------------------ #

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
        """One audio buffer in, whatever the GPU thread has finished out."""
        self._check_for_a_hung_pass()
        if not self._reset_requested:
            self.posteriors.feed(samples)
            self._wake.set()
        return self._drain()

    def reset(self, ) -> None:
        """A song boundary (D10), marshalled onto the worker.

        The ring cannot be zeroed under a snapshot in flight, so the request is
        handed over and the audio thread stops feeding until it has been taken.
        That costs at most one pass of audio -- which, at a boundary defined by
        0.3 s of silence, is silence.
        """
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
            # Idempotent: the watchdog latches one fault and logs one transition
            # however many buffers go by while the pass stays in there.
            self._watchdog.report_fault('hung_pass')

    # -- the worker --------------------------------------------------------- #

    def _work(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception as error:
                # A worker that dies takes the show's only classifier with it and
                # says nothing, which is the one outcome worse than a shed.
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
        self._retry_at = None
        self._reset_requested = False

    def _enter_shed(self) -> None:
        self._shed = True
        with self._lock:
            self._queue.clear()
            self._gap = True
        logging.warning(f'[gpu] shed: holding the intent, the hand-off queue is '
                        f'dropped and the decoder is reset '
                        f'({self.passes} passes so far)')

    def _leave_shed(self) -> None:
        self._shed = False
        self.resyncs += 1
        record = self.posteriors.resync()
        with self._lock:
            self._queue.clear()
            self._gap = True
        self._attempts = 0
        self._retry_at = None
        logging.warning(f'[gpu] restored at the live edge: skipped '
                        f'{record.lost_sec:.2f}s / {record.cells_lost} cells, '
                        f'resuming at cell {record.first_cell_index}')

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
        self._attempts = 0
        self._retry_at = None
        self._logged_fault = None
        self._offer(produced)
        self._watchdog.report_healthy()
        self._status()

    def _offer(self, produced) -> None:
        if not produced:
            return
        with self._lock:
            room = len(self._queue) < self._queue_passes
            if room:
                self._queue.append(produced)
        if not room:
            self.overflows += 1
            self._fault('queue_overflow',
                        f'{self._queue_passes} passes are waiting for a '
                        f'consumer that has stopped draining')

    # -- degradation -------------------------------------------------------- #

    def _fault(self, kind: str, detail) -> None:
        self.faults += 1
        self._log_fault(kind, detail)
        self._retry_at = self._clock.monotonic() + self._backoff()
        self._watchdog.report_fault(kind)

    def _backoff(self) -> float:
        return _BACKOFF_SEC[min(self._attempts, len(_BACKOFF_SEC) - 1)]

    def _retry(self) -> None:
        """Ask the watchdog to let a pass through again, on a capped backoff.

        Only a stage fault is the stage's to clear: shed by drift, the loop is
        behind and resuming the GPU is the last thing that helps.
        """
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
        """Rebuild the encoder, which is the only part a dead context invalidates.

        The ring, the schedule and the sample index are numpy and survive; a new
        stream would restart the clock the whole show is stamped against.
        """
        self.reinits += 1
        try:
            self.posteriors.set_encoder(self._reinit())
        except Exception as error:
            self._log_fault('reinit_failed', error)
            return False
        logging.warning(f'[gpu] extractor reinitialised (attempt '
                        f'{self._attempts}, {self.reinits} so far)')
        return True

    def _log_fault(self, kind: str, detail) -> None:
        now = self._clock.monotonic()
        if (kind == self._logged_fault and self._logged_fault_at is not None
                and now - self._logged_fault_at < _FAULT_LOG_INTERVAL_SEC):
            return
        self._logged_fault, self._logged_fault_at = kind, now
        logging.warning(f'[gpu] {kind}: {detail} — the show holds its intent '
                        f'and keeps beats (fault #{self.faults}, '
                        f'{self.reinits} reinit(s))')

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
