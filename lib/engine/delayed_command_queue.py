import time
import logging
from typing import Callable, Awaitable, List, Tuple

log = logging.getLogger(__name__)

CommandFactory = Callable[[], Awaitable[None]]


class DelayedCommandQueue:
    """
    Holds outgoing light commands and releases them after a fixed wall-clock delay.

    The audio playback stack (dmx-enttec-node/app_audio_receiver) delays audio output
    by `playback_delay_seconds`. To keep lights in sync, every command that drives
    hardware (MIDI, OS2L, overlay) is enqueued here and only dispatched once the same
    duration has elapsed.

    Analysis happens at wall-clock time T. The audience hears that audio at T + delay.
    Commands enqueued at T are released at T + delay, so lights change exactly when the
    audience hears the musical event that triggered them.

    All enqueuing and draining happens on the asyncio event loop — no locking needed.
    drain() should be called on every main-loop iteration (~5.8 ms cadence).
    """

    def __init__(self, delay_sec: float):
        self._delay_sec = delay_sec
        # (enqueue_time, fire_at, label, factory)
        self._queue: List[Tuple[float, float, str, CommandFactory]] = []
        self._timing_log: List[dict] = []

    @property
    def delay_sec(self) -> float:
        return self._delay_sec

    async def enqueue(self, label: str, factory: CommandFactory) -> None:
        """Schedule factory() to be called after delay_sec.
        Values used inside factory must be captured in a closure at call time."""
        enqueue_time = time.monotonic()
        fire_at = enqueue_time + self._delay_sec
        self._queue.append((enqueue_time, fire_at, label, factory))

    async def drain(self) -> None:
        """Execute all commands whose fire time has passed, in chronological order."""
        if not self._queue:
            return
        now = time.monotonic()
        due = [(et, ft, lbl, f) for et, ft, lbl, f in self._queue if ft <= now]
        if not due:
            return
        self._queue = [(et, ft, lbl, f) for et, ft, lbl, f in self._queue if ft > now]
        due.sort(key=lambda x: x[1])
        for enqueue_time, fire_at, label, factory in due:
            actual_fire_time = time.monotonic()
            actual_delta = actual_fire_time - enqueue_time
            error_ms = abs(actual_delta - self._delay_sec) * 1000
            log.debug(
                f'[cmd_queue] {label!r}  target={self._delay_sec:.3f}s  '
                f'actual={actual_delta:.3f}s  error={error_ms:.1f}ms'
            )
            self._timing_log.append({
                'label': label,
                'enqueue_time': enqueue_time,
                'target_fire_time': fire_at,
                'actual_fire_time': actual_fire_time,
                'target_delta_sec': self._delay_sec,
                'actual_delta_sec': actual_delta,
            })
            await factory()

    def get_timing_log(self) -> List[dict]:
        return list(self._timing_log)
