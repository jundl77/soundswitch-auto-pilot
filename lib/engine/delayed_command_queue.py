"""Hold each command until the audience hears the audio that caused it.

The delay is a property of the STREAM, not of the queue.  A beat is detected as
the audio arrives, so it waits the whole playback delay; a decoder decision is
already ~13.7 s old when it exists, so it waits what is left
(``playback_delay - chain_latency``, B1).  One constant for both would put the
OS2L wire a whole chain latency ahead of the room.

Because the two streams wait different amounts, ``fire_at`` is not monotone in
enqueue order across the queue -- only *within* a stream, which is what the
clamp preserves and what the ordering guarantee is stated in terms of.  The
queue is therefore kept sorted by fire time rather than by arrival, so the
head check stays a valid shortcut.
"""
import bisect
import logging
from collections import deque
from typing import Awaitable, Callable

from lib.clock import SYSTEM_CLOCK, Clock

log = logging.getLogger(__name__)

CommandFactory = Callable[[], Awaitable[None]]


class DelayedCommandQueue:
    # Unsynchronised: enqueue and drain must both run on the asyncio loop, never on a client thread.

    def __init__(self, delay_sec: float, clock: Clock = SYSTEM_CLOCK):
        self._delay_sec = delay_sec
        self._clock = clock
        self._queue: list = []
        self._sequence: int = 0
        self._last_fire_at: dict = {}
        self._timing_log: deque = deque(maxlen=2000)

    @property
    def delay_sec(self) -> float:
        return self._delay_sec

    @property
    def pending(self) -> int:
        return len(self._queue)

    async def enqueue(self, label: str, factory: CommandFactory,
                      delay_sec: float | None = None) -> None:
        enqueue_time = self._clock.monotonic()
        delay = self._delay_sec if delay_sec is None else float(delay_sec)
        # A stream's delay tracks a measurement that moves, so a later command
        # can ask for a shorter wait than an earlier one and overtake it.
        # Clamping per label keeps each stream in order without letting a
        # long-waiting beat drag the next intent out to its own fire time.
        fire_at = max(enqueue_time + delay,
                      self._last_fire_at.get(label, float('-inf')))
        self._last_fire_at[label] = fire_at
        self._sequence += 1
        bisect.insort(
            self._queue,
            (fire_at, self._sequence, enqueue_time, delay, label, factory),
            key=lambda item: (item[0], item[1]))

    async def drain(self) -> None:
        if not self._queue:
            return
        now = self._clock.monotonic()
        if self._queue[0][0] > now:
            return
        cut = bisect.bisect_right(self._queue, now, key=lambda item: item[0])
        due, self._queue = self._queue[:cut], self._queue[cut:]
        for fire_at, _, enqueue_time, delay, label, factory in due:
            actual_fire_time = self._clock.monotonic()
            actual_delta = actual_fire_time - enqueue_time
            error_ms = abs(actual_delta - delay) * 1000
            log.debug(
                f'[cmd_queue] {label!r}  target={delay:.3f}s  '
                f'actual={actual_delta:.3f}s  error={error_ms:.1f}ms'
            )
            self._timing_log.append({
                'label': label,
                'enqueue_time': enqueue_time,
                'target_fire_time': fire_at,
                'target_delta_sec': delay,
                'actual_fire_time': actual_fire_time,
                'actual_delta_sec': actual_delta,
            })
            await factory()

    def get_timing_log(self) -> list[dict]:
        return list(self._timing_log)
