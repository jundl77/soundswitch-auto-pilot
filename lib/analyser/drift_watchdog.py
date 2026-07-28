"""Backpressure watchdog: notice when the analyser stops keeping up, loudly.

Live audio arrives at exactly 1x and never waits for us. The look-ahead lead is
a fixed budget handed out once at start-up, so a stretch of slower-than-real-time
processing spends it, and only running *faster* than real time wins it back.
Worse, the input side does not queue without bound — PortAudio's ring buffer
overflows and silently discards, so falling behind eventually costs audio rather
than latency, and nothing in the pipeline would say so.

The instrument is the cheapest one that answers the question: how much wall time
has passed that is not accounted for by audio processed. Measured over a rolling
window rather than since start-up, because dropped samples are gone — cumulative
lag can never fall again, so a watchdog built on it would latch after the first
hiccup and stay degraded for the rest of the show.

On breach the analyser sheds work in a fixed order, cheapest loss first. Beat
tracking is never shed: a show that has lost section detection is a duller show,
a show that has lost the beat is not a show.
"""

from __future__ import annotations

import logging
from collections import deque
from enum import IntEnum

from lib.clock import SYSTEM_CLOCK, Clock

# Rolling window the control signal is measured over. Long enough that a single
# slow buffer is invisible, short enough that a stall is noticed within seconds.
_WINDOW_SEC = 5.0

# Drift accumulated inside the window, in seconds of lost lead. Read as a
# fraction of the window these are shortfall rates: 0.15 s / 5 s = 3 %, and
# 0.75 s / 5 s = 15 %. Against the 2.5 s look-ahead a 3 % shortfall spends the
# whole lead in ~80 s and a 15 % one in ~17 s — so level 1 is "there is no
# headroom left, give up the cheapest thing now" and level 2 is "the lead is
# going within the minute".
_ENTER_SEC = (0.15, 0.75)
# Recovery requires winning lead back, not merely holding it. Leaving level 1
# demands NEGATIVE drift — the loop must actually be draining — because an
# exactly-real-time stretch preserves the loss rather than repairing it, and
# treating that as recovery would flap the level around the entry threshold.
_EXIT_SEC = (-0.25, 0.10)


class ShedLevel(IntEnum):
    """What the analyser has given up, in the order it gives it up."""

    NONE = 0
    SECTION_DETECTION = 1   # YAMNet off — the show loses section-change cues
    ONSET_DETECTION = 2     # + madmom onsets off — density features go stale


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
        # (wall, stream) pairs; stream time is exact by construction.
        self._samples: deque[tuple[float, float]] = deque()
        self._stream_sec = 0.0
        self._level = ShedLevel.NONE
        self._drift_sec = 0.0
        self.peak_drift_sec = 0.0
        self.total_drift_sec = 0.0
        self._last_wall: float | None = None

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

        if self._last_wall is not None:
            # Cumulative drift never forgets — it is the soak run's headline.
            self.total_drift_sec += max(0.0, (wall - self._last_wall) - self._buffer_sec)
        self._last_wall = wall

        self._samples.append((wall, self._stream_sec))
        while len(self._samples) > 1 and wall - self._samples[0][0] > self._window_sec:
            self._samples.popleft()

        if len(self._samples) < 2:
            return self._level  # one sample is not a span

        wall_span = wall - self._samples[0][0]
        stream_span = self._stream_sec - self._samples[0][1]
        self._drift_sec = wall_span - stream_span
        self.peak_drift_sec = max(self.peak_drift_sec, self._drift_sec)
        self._update_level()
        return self._level

    def _update_level(self) -> None:
        target = self._level
        # Escalate on the highest breached entry threshold...
        for level in (ShedLevel.ONSET_DETECTION, ShedLevel.SECTION_DETECTION):
            if self._drift_sec > _ENTER_SEC[level - 1]:
                target = max(target, level)
                break
        # ...and de-escalate one step at a time, so recovery is gradual.
        if target is self._level and self._level is not ShedLevel.NONE:
            if self._drift_sec < _EXIT_SEC[self._level - 1]:
                target = ShedLevel(self._level - 1)

        if target is not self._level:
            direction = 'degrading' if target > self._level else 'recovering'
            logging.warning(
                f'[drift] {direction}: {self._level.name} -> {target.name} '
                f'(drift {self._drift_sec:+.3f}s over {self._window_sec:.0f}s window, '
                f'peak {self.peak_drift_sec:.3f}s)')
            self._level = target
