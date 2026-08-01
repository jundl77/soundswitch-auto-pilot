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
        self.schedule(label, factory, delay_sec)

    def schedule(self, label: str, factory: CommandFactory,
                 delay_sec: float | None = None) -> None:
        """The same thing from a synchronous caller.

        Nothing in here awaits, and the sound-boundary handlers the analyser
        calls are not coroutines; scheduling a task from them instead would put
        the enqueue at whatever await point came next, which is a report that
        depends on the loop's shape.
        """
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

    def drop_pending(self, label: str, later_than: float,
                     inclusive: bool = False) -> int:
        """Cancel this label's undelivered commands firing after a time.

        The supersede half of a single serialized stream: a command's fire time
        is the song instant it describes plus the playback delay, so a newer
        statement about later audio replaces whatever was queued for it.
        Without this, a beat-dropout ATMOSPHERIC enqueued for song instant N
        still lands after the committer has already said what N was -- and,
        being what the engine last decided, suppresses every repair.

        **Strictly after, because an equal fire time is not a restatement.**
        A chain older than the playback delay clamps every decision to `now`,
        so two consecutive bars arrive carrying one wall instant between them.
        In production the two reads are microseconds apart and both survive; in
        virtual time they are the same float, and dropping "at or after" then
        deleted the predecessor -- a real intent block lost on every
        slower-than-120-BPM track, in the simulation only.  Equal fire times
        keep their order through the queue's (fire time, sequence) sort, which
        is commit order.

        `inclusive` takes the equal fire times too, and exists for the one
        stream where equality IS a restatement: an effect refresh landing at
        the instant of an intent change is redundant, because the change
        re-picks the effect itself.  The caller says which meaning it holds,
        because the queue cannot know.

        The clamp goes with them: a cancelled command's fire time was the floor
        every later one in its stream was held to, and leaving it in place lets
        it order the stream from the grave.
        """
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
                # The batch was sliced off the queue before the first await, so
                # a raise here used to take every command behind it with it AND
                # propagate into the loop, which has no handler -- one transient
                # rtmidi or socket error killed the process with the rig lit.
                # Each command is its own statement about one instant; a wire
                # that refuses one of them is not a reason to drop the rest.
                log.exception(f'[cmd_queue] {label!r} failed ({error!r}) — '
                              f'dropping it and delivering the rest')

    def get_timing_log(self) -> list[dict]:
        return list(self._timing_log)
