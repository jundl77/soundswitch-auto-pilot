"""Hold each command until the audience hears the audio that caused it."""
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
        self.schedule(label, factory, delay_sec)

    def schedule(self, label: str, factory: CommandFactory,
                 delay_sec: float | None = None) -> None:
        enqueue_time = self._clock.monotonic()
        delay = self._delay_sec if delay_sec is None else float(delay_sec)
        fire_at = self._kept_in_stream_order(label, enqueue_time + delay)
        self._last_fire_at[label] = fire_at
        self._sequence += 1
        bisect.insort(
            self._queue,
            (fire_at, self._sequence, enqueue_time, delay, label, factory),
            key=lambda item: (item[0], item[1]))

    def _kept_in_stream_order(self, label: str, fire_at: float) -> float:
        return max(fire_at, self._last_fire_at.get(label, float('-inf')))

    def drop_pending(self, label: str, later_than: float,
                     inclusive: bool = False) -> int:
        def doomed(item) -> bool:
            return item[4] == label and (item[0] >= later_than if inclusive
                                         else item[0] > later_than)

        keep = [item for item in self._queue if not doomed(item)]
        dropped = len(self._queue) - len(keep)
        if not dropped:
            return 0
        self._queue = keep
        alive = [item[0] for item in keep if item[4] == label]
        if alive:
            self._last_fire_at[label] = max(alive)
        else:
            self._last_fire_at.pop(label, None)
        return dropped

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
            try:
                await factory()
            except Exception as error:
                log.exception(f'[cmd_queue] {label!r} failed ({error!r}) — '
                              f'dropping it and delivering the rest')

    def get_timing_log(self) -> list[dict]:
        return list(self._timing_log)
