"""Injectable time source, so simulation can run faster than real time."""
import datetime
import time


class Clock:
    def now(self) -> datetime.datetime:
        raise NotImplementedError

    def monotonic(self) -> float:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> datetime.datetime:
        return datetime.datetime.now()

    def monotonic(self) -> float:
        # perf_counter, not monotonic: on Windows/CPython `time.monotonic` has a
        # 15.625 ms resolution, coarser than the 5.805 ms audio buffer it times.
        return time.perf_counter()


class VirtualClock(Clock):
    """Deterministic clock advanced manually by the simulation loop."""

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
