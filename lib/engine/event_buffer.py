"""
Thread-safe store of pipeline events.

Written to from the asyncio pipeline thread (via stub clients), read from the
Dash server thread via snapshot(). The only shared state between threads.
"""

import threading
import time
from collections import deque


class EventBuffer:
    def __init__(self, window_sec: float = 60.0):
        self._lock = threading.Lock()
        self._window_sec = window_sec
        self._start_time: float | None = None
        self._is_playing: bool = False
        # Beats: high-frequency, bounded deque to avoid unbounded memory growth
        self._beats: deque[dict] = deque(maxlen=3000)
        # Effects: list so we can mutate the last entry to set 'end'
        self._effects: list[dict] = []
        self._timing_log: list[dict] = []

    def start(self) -> None:
        with self._lock:
            self._start_time = time.monotonic()

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def _now(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def add_beat(self, bpm: float, strength: float, change: bool) -> None:
        with self._lock:
            self._beats.append({
                't': self._now(), 'bpm': bpm, 'strength': strength, 'change': change,
            })

    def add_effect(self, channel: str, effect_type: str) -> None:
        with self._lock:
            now = self._now()
            # Close the previous open effect band
            if self._effects and 'end' not in self._effects[-1]:
                self._effects[-1]['end'] = now
            self._effects.append({'t': now, 'channel': channel, 'type': effect_type})
            # Prune entries well outside the window to prevent unbounded growth
            cutoff = now - self._window_sec * 2
            self._effects = [e for e in self._effects if e.get('end', now) >= cutoff]

    def set_playing(self, is_playing: bool) -> None:
        with self._lock:
            self._is_playing = is_playing

    def set_timing_log(self, log: list[dict]) -> None:
        with self._lock:
            self._timing_log = list(log)

    def snapshot(self) -> dict:
        """Thread-safe copy of recent state — called from Dash every 100 ms."""
        with self._lock:
            now = self._now()
            cutoff = now - self._window_sec
            return {
                'now': now,
                'is_playing': self._is_playing,
                'beats': [b for b in self._beats if b['t'] >= cutoff],
                'effects': [e for e in self._effects if e.get('end', now) >= cutoff],
                'current_effect': self._effects[-1] if self._effects else None,
                'bpm': self._beats[-1]['bpm'] if self._beats else 0.0,
                'beats_detected': len(self._beats),
            }

    def to_report(self, timing_log: list[dict] | None = None) -> dict:
        """Full serializable report for agentic evaluation or JSON export."""
        with self._lock:
            now = self._now()
            all_effects = list(self._effects)
            if all_effects and 'end' not in all_effects[-1]:
                all_effects[-1] = {**all_effects[-1], 'end': now}

            tlog = timing_log if timing_log is not None else self._timing_log
            errors_ms = [
                abs(e['actual_delta_sec'] - e['target_delta_sec']) * 1000
                for e in tlog
            ]
            durations = [e['end'] - e['t'] for e in all_effects if 'end' in e]
            unique_channels = {e['channel'] for e in all_effects}
            all_beats = list(self._beats)

            return {
                'duration_sec': now,
                'beats': all_beats,
                'effects': all_effects,
                'timing_log': tlog,
                'metrics': {
                    'beats_detected': len(all_beats),
                    'bpm_last': all_beats[-1]['bpm'] if all_beats else 0.0,
                    'timing_error_mean_ms': (
                        sum(errors_ms) / len(errors_ms) if errors_ms else 0.0
                    ),
                    'timing_error_max_ms': max(errors_ms) if errors_ms else 0.0,
                    'unique_effects_count': len(unique_channels),
                    'effect_changes_count': len(all_effects),
                    'avg_effect_duration_sec': (
                        sum(durations) / len(durations) if durations else 0.0
                    ),
                    'unique_channels': sorted(unique_channels),
                },
            }
