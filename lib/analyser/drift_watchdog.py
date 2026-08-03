"""Notices when the show's one sheddable stage should stop, from either side."""

from __future__ import annotations

import logging
import threading
from collections import deque
from enum import IntEnum

from lib.clock import SYSTEM_CLOCK, Clock

_WINDOW_SEC = 5.0
_ENTER_SEC = 0.15
# Positive: a hardware-paced input hands over one buffer per period, so drift never goes negative.
_EXIT_SEC = 0.05

_LOG_INTERVAL_SEC = 30.0


class ShedLevel(IntEnum):
    NONE = 0
    NN_SHED = 1


class DriftWatchdog:
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
        if sec <= 0:
            return
        self._samples = deque((w + sec, s) for w, s in self._samples)
        if self._first_wall is not None:
            self._first_wall += sec
        if self._calm_since is not None:
            self._calm_since += sec

    @property
    def drift_sec(self) -> float:
        return self._drift_sec

    def report_fault(self, kind: str) -> ShedLevel:
        if self._fault != kind:
            was = self._fault
            self._fault = kind
            self._settle(f'stage fault: {kind}' if was is None
                         else f'stage fault: {was} -> {kind}')
        return self._level

    def report_healthy(self) -> ShedLevel:
        if self._fault is not None:
            self._fault = None
            self._settle('stage healthy')
        return self._level

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
