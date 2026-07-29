import logging
from typing import Callable, Awaitable
from lib.clock import Clock, SYSTEM_CLOCK

log = logging.getLogger(__name__)

CommandFactory = Callable[[], Awaitable[None]]


class DelayedCommandQueue:
    # Unsynchronised: enqueue and drain must both run on the asyncio loop, never on a client thread.

    def __init__(self, delay_sec: float, clock: Clock = SYSTEM_CLOCK):
        self._delay_sec = delay_sec
        self._clock = clock
        self._queue: list[tuple[float, float, str, CommandFactory]] = []
        self._timing_log: list[dict] = []

    @property
    def delay_sec(self) -> float:
        return self._delay_sec

    @property
    def pending(self) -> int:
        return len(self._queue)

    async def enqueue(self, label: str, factory: CommandFactory) -> None:
        enqueue_time = self._clock.monotonic()
        fire_at = enqueue_time + self._delay_sec
        self._queue.append((enqueue_time, fire_at, label, factory))

    async def drain(self) -> None:
        if not self._queue:
            return
        now = self._clock.monotonic()
        # Head-only check is valid because a fixed delay on a monotonic clock keeps fire_at nondecreasing.
        if self._queue[0][1] > now:
            return
        due = [(et, ft, lbl, f) for et, ft, lbl, f in self._queue if ft <= now]
        if not due:
            return
        self._queue = [(et, ft, lbl, f) for et, ft, lbl, f in self._queue if ft > now]
        due.sort(key=lambda x: x[1])
        for enqueue_time, fire_at, label, factory in due:
            actual_fire_time = self._clock.monotonic()
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

    def get_timing_log(self) -> list[dict]:
        return list(self._timing_log)
