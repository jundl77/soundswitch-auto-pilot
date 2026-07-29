"""
Injectable time source for the pipeline.

Every component that measures time (windows, delays, cooldowns, timelines)
takes a Clock so that simulation can run faster than real-time on a virtual
clock while production runs on the system clock. Default is always
SYSTEM_CLOCK — production wiring never passes a clock explicitly.
"""
import datetime
import time


class Clock:
    """Time source interface: wall-clock datetime plus a monotonic float."""

    def now(self) -> datetime.datetime:
        raise NotImplementedError

    def monotonic(self) -> float:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> datetime.datetime:
        return datetime.datetime.now()

    def monotonic(self) -> float:
        # perf_counter, not monotonic: on Windows/CPython 3.11 `time.monotonic`
        # has a resolution of 15.625 ms — coarser than the 5.805 ms audio buffer
        # everything here is timed against, so it cannot resolve a single loop
        # iteration at all. perf_counter is the same kind of clock (monotonic,
        # undefined epoch, for measuring intervals) at ~100 ns.
        return time.perf_counter()


class VirtualClock(Clock):
    """Deterministic clock advanced manually by the simulation loop.

    Starts at a fixed epoch so runs are reproducible; monotonic() is the
    virtual elapsed seconds (song position in file simulation).
    """

    _EPOCH = datetime.datetime(2000, 1, 1)

    def __init__(self):
        self._elapsed_sec: float = 0.0

    def now(self) -> datetime.datetime:
        return self._EPOCH + datetime.timedelta(seconds=self._elapsed_sec)

    def monotonic(self) -> float:
        return self._elapsed_sec

    def advance(self, dt_sec: float) -> None:
        if dt_sec < 0:
            raise ValueError(f'cannot advance clock backwards ({dt_sec})')
        self._elapsed_sec += dt_sec


SYSTEM_CLOCK = SystemClock()
