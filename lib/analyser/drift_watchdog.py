"""Notices when the show's one sheddable stage should stop, from either side.

The ladder used to have two rungs, `SECTION_DETECTION` and `ONSET_DETECTION`,
and the NN integration deleted both tenants. What is left is the GPU feature
stage, so the ladder is one rung -- and `NN_SHED` is not a smaller show, it is
the degradation contract of #144: hold the intent, keep beats and the silence
timer, and say so loudly.

**Two inputs, one door.** Drift measures lost lead against a hardware-paced
input, which is the only thing that can see the loop failing to keep up. It is
structurally blind to the stage failing on its own: a CUDA fault, a driver
reset, a sleep/resume context loss or a hung pass all cost the audio loop
exactly nothing, so pacing stays perfect while the show holds one intent
forever. The stage reports those itself, and either input alone holds the door
shut -- clearing one is not clearing both.

The two arrive on different threads (drift from the audio loop, health from the
GPU thread) and each writes only its own half; the derived level is settled
under a lock so a transition cannot be logged twice or lost.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from enum import IntEnum

from lib.clock import SYSTEM_CLOCK, Clock

_WINDOW_SEC = 5.0
_ENTER_SEC = 0.15
# Positive, not negative: a hardware-paced input hands over exactly one buffer
# per buffer period, so the loop can never consume audio faster than it arrives
# and drift can never go negative however much headroom there is. Requiring
# negative drift to recover leaves the watchdog latched for the whole show.
_EXIT_SEC = 0.05

# A flapping stage crosses this door twice per pass, so the transition log is
# on a per-direction rate limit for the same reason every other WARNING here is.
# What is suppressed is COUNTED and carried into the next line: a quiet log and
# a healthy rig have to stay distinguishable.
_LOG_INTERVAL_SEC = 30.0


class ShedLevel(IntEnum):
    NONE = 0
    NN_SHED = 1


class DriftWatchdog:
    """Tracks lost lead and stage health, and picks a shed level from both.

    `observe()` is called once per processed audio buffer, from the same thread
    that does the processing. `report_fault` / `report_healthy` are called by
    the stage, from its own thread.
    """

    def __init__(self, buffer_sec: float, clock: Clock = SYSTEM_CLOCK,
                 window_sec: float = _WINDOW_SEC):
        self._buffer_sec = buffer_sec
        self._clock = clock
        self._window_sec = window_sec
        self._settling = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self._samples: deque[tuple[float, float]] = deque()
        self._stream_sec = 0.0
        self._level = ShedLevel.NONE
        self._drift_sec = 0.0
        self._drift_shed = False
        self._fault: str | None = None
        self.peak_drift_sec = 0.0
        self.total_drift_sec = 0.0
        self._first_wall: float | None = None
        self._calm_since: float | None = None
        self._said: dict = {}
        self._suppressed: dict = {}

    @property
    def level(self) -> ShedLevel:
        return self._level

    @property
    def fault(self) -> str | None:
        return self._fault

    def forgive(self, sec: float) -> None:
        """Deliberate stalls (the MIDI settle at a song boundary) are not lost lead."""
        if sec <= 0:
            return
        self._samples = deque((w + sec, s) for w, s in self._samples)
        if self._first_wall is not None:
            self._first_wall += sec
        if self._calm_since is not None:
            self._calm_since += sec

    @property
    def drift_sec(self) -> float:
        """Lead lost inside the rolling window. Negative means catching up."""
        return self._drift_sec

    # -- health, from the stage's own thread -------------------------------- #

    def report_fault(self, kind: str) -> ShedLevel:
        """The stage cannot run. Latched until it says otherwise."""
        if self._fault != kind:
            was = self._fault
            self._fault = kind
            self._settle(f'stage fault: {kind}' if was is None
                         else f'stage fault: {was} -> {kind}')
        return self._level

    def report_healthy(self) -> ShedLevel:
        """The stage completed work. Drift may still be holding the door."""
        if self._fault is not None:
            self._fault = None
            self._settle('stage healthy')
        return self._level

    # -- drift, from the audio loop ----------------------------------------- #

    def observe(self) -> ShedLevel:
        wall = self._clock.monotonic()
        self._stream_sec += self._buffer_sec

        if self._first_wall is None:
            self._first_wall = wall
        self.total_drift_sec = (wall - self._first_wall) - (self._stream_sec - self._buffer_sec)

        self._samples.append((wall, self._stream_sec))
        while len(self._samples) > 1 and wall - self._samples[0][0] > self._window_sec:
            self._samples.popleft()

        if len(self._samples) < 2:
            return self._level

        wall_span = wall - self._samples[0][0]
        stream_span = self._stream_sec - self._samples[0][1]
        self._drift_sec = wall_span - stream_span
        self.peak_drift_sec = max(self.peak_drift_sec, self._drift_sec)
        self._update_drift(wall)
        return self._level

    def _update_drift(self, wall: float) -> None:
        if self._drift_sec > _ENTER_SEC:
            self._calm_since = None
            if not self._drift_shed:
                self._drift_shed = True
                self._settle(f'drift {self._drift_sec:+.3f}s over a '
                             f'{self._window_sec:.0f}s window, peak '
                             f'{self.peak_drift_sec:.3f}s')
        elif self._drift_shed:
            if self._drift_sec >= _EXIT_SEC:
                self._calm_since = None
            elif self._calm_since is None:
                self._calm_since = wall
            elif wall - self._calm_since >= self._window_sec:
                self._drift_shed = False
                self._calm_since = None
                self._settle(f'pacing recovered ({self._drift_sec:+.3f}s)')

    def _settle(self, reason: str) -> None:
        with self._settling:
            target = (ShedLevel.NN_SHED if self._drift_shed or self._fault
                      else ShedLevel.NONE)
            if target is self._level:
                return
            direction = 'degrading' if target > self._level else 'recovering'
            message = (f'[drift] {direction}: {self._level.name} -> '
                       f'{target.name} ({reason})')
            self._level = target
            now = self._clock.monotonic()
            last = self._said.get(direction)
            if last is not None and now - last < _LOG_INTERVAL_SEC:
                self._suppressed[direction] = self._suppressed.get(direction, 0) + 1
                return
            self._said[direction] = now
            more = self._suppressed.pop(direction, 0)
            logging.warning(message + (f' [+{more} more since the last line]'
                                       if more else ''))
