"""The monitored output, held one look-ahead behind the analysis it came from."""

import logging
from collections import deque

from lib.clock import SYSTEM_CLOCK, Clock


class DelayedMonitor:
    def __init__(self, delay_sec: float, play, clock: Clock = SYSTEM_CLOCK):
        self._delay_sec = delay_sec
        self._play = play
        self._clock = clock
        self._buffered: deque = deque()
        self._ready_at: float = 0.0
        self._started: bool = False
        self.arm()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def buffered(self) -> int:
        return len(self._buffered)

    def arm(self) -> None:
        self._ready_at = self._clock.monotonic() + self._delay_sec
        self._started = False

    def silence(self) -> None:
        self._buffered.clear()
        self.arm()

    def drain(self) -> None:
        if self._buffered:
            self._play(self._buffered.popleft())

    def feed(self, audio) -> None:
        self._buffered.append(audio)
        if self._clock.monotonic() < self._ready_at:
            return
        if not self._started:
            self._started = True
            logging.info(f'[monitor] {self._delay_sec:.1f}s buffered — the '
                         f'headphones join the room')
        self._play(self._buffered.popleft())
