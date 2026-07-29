"""Notices when the analyser stops keeping up, and sheds work cheapest-loss-first."""

from __future__ import annotations

import logging
from collections import deque
from enum import IntEnum

from lib.clock import SYSTEM_CLOCK, Clock

_WINDOW_SEC = 5.0
_ENTER_SEC = (0.15, 0.75)
# Positive, not negative: a hardware-paced input hands over exactly one buffer
# per buffer period, so the loop can never consume audio faster than it arrives
# and drift can never go negative however much headroom there is. Requiring
# negative drift to recover leaves the watchdog latched for the whole show.
_EXIT_SEC = (0.05, 0.30)


class ShedLevel(IntEnum):
    NONE = 0
    SECTION_DETECTION = 1
    ONSET_DETECTION = 2


class DriftWatchdog:
    """Tracks lost lead over a rolling window and picks a shed level.

    `observe()` is called once per processed audio buffer, from the same thread
    that does the processing.
    """

    def __init__(self, buffer_sec: float, clock: Clock = SYSTEM_CLOCK,
                 window_sec: float = _WINDOW_SEC):
        self._buffer_sec = buffer_sec
        self._clock = clock
        self._window_sec = window_sec
        self.reset()

    def reset(self) -> None:
        self._samples: deque[tuple[float, float]] = deque()
        self._stream_sec = 0.0
        self._level = ShedLevel.NONE
        self._drift_sec = 0.0
        self.peak_drift_sec = 0.0
        self.total_drift_sec = 0.0
        self._first_wall: float | None = None
        self._calm_since: float | None = None

    @property
    def level(self) -> ShedLevel:
        return self._level

    @property
    def drift_sec(self) -> float:
        """Lead lost inside the rolling window. Negative means catching up."""
        return self._drift_sec

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
        self._update_level(wall)
        return self._level

    def _update_level(self, wall: float) -> None:
        target = self._level
        for level in (ShedLevel.ONSET_DETECTION, ShedLevel.SECTION_DETECTION):
            if self._drift_sec > _ENTER_SEC[level - 1]:
                target = max(target, level)
                break

        if target is self._level and self._level is not ShedLevel.NONE:
            if self._drift_sec < _EXIT_SEC[self._level - 1]:
                if self._calm_since is None:
                    self._calm_since = wall
                elif wall - self._calm_since >= self._window_sec:
                    target = ShedLevel(self._level - 1)
            else:
                self._calm_since = None

        if target is not self._level:
            self._calm_since = None
            direction = 'degrading' if target > self._level else 'recovering'
            logging.warning(
                f'[drift] {direction}: {self._level.name} -> {target.name} '
                f'(drift {self._drift_sec:+.3f}s over {self._window_sec:.0f}s window, '
                f'peak {self.peak_drift_sec:.3f}s)')
            self._level = target
