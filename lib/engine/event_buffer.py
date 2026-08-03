"""Thread-safe store of pipeline events — the only state the Dash thread shares."""

import threading
from collections import deque

from lib.clock import Clock, SYSTEM_CLOCK

_UNMEASURED_BEAT_STRENGTH = 0.0


class EventBuffer:
    def __init__(self, window_sec: float = 60.0, clock: Clock = SYSTEM_CLOCK,
                 look_ahead_sec: float = 0.0):
        self._lock = threading.Lock()
        self._window_sec = window_sec
        self._clock = clock
        self._look_ahead_sec = look_ahead_sec
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._is_playing: bool = False
        self._beats: deque[dict] = deque(maxlen=None if window_sec == float('inf') else 3000)
        self._effects: list[dict] = []
        self._intents: list[dict] = []
        self._timing_log: list[dict] = []
        self._current_intent: str | None = None
        self._sound_events: list[dict] = []
        self._decoder_state: dict = {}

    def start(self) -> None:
        with self._lock:
            self._start_time = self._clock.monotonic()

    def mark_end(self) -> None:
        with self._lock:
            self._end_time = self._clock.monotonic()

    def elapsed(self) -> float:
        return self._now()

    def _now(self) -> float:
        if self._start_time is None:
            return 0.0
        now = self._clock.monotonic()
        if self._end_time is not None and now > self._end_time:
            now = self._end_time
        return now - self._start_time

    def add_beat(self, bpm: float, change: bool, rms: float = 0.0) -> None:
        with self._lock:
            self._beats.append({
                't': self._now(), 'bpm': bpm,
                'strength': _UNMEASURED_BEAT_STRENGTH,
                'change': change,
                'rms': round(rms, 4),
            })

    def add_effect(self, channel: str, effect_type: str) -> None:
        with self._lock:
            now = self._now()
            if self._effects and 'end' not in self._effects[-1]:
                self._effects[-1]['end'] = now
            self._effects.append({'t': now, 'channel': channel, 'type': effect_type})
            cutoff = now - self._window_sec * 2
            self._effects = [e for e in self._effects if e.get('end', now) >= cutoff]

    def set_playing(self, is_playing: bool) -> None:
        with self._lock:
            self._is_playing = is_playing
            now = self._now()
            self._sound_events.append({'t': now, 'playing': is_playing})
            cutoff = now - self._window_sec * 2
            self._sound_events = [e for e in self._sound_events if e['t'] >= cutoff]

    def set_intent(self, intent: str, song_sec: float | None = None) -> None:
        with self._lock:
            if intent == self._current_intent:
                return
            self._current_intent = intent
            now = self._now()
            if self._intents and 'end' not in self._intents[-1]:
                self._intents[-1]['end'] = now
            block = {'t': now, 'intent': intent}
            if song_sec is not None:
                block['song_t'] = round(float(song_sec), 6)
            self._intents.append(block)
            cutoff = now - self._window_sec * 2
            self._intents = [e for e in self._intents if e.get('end', now) >= cutoff]

    def set_timing_log(self, log: list[dict]) -> None:
        with self._lock:
            self._timing_log = list(log)

    def set_decoder_state(self, **state) -> None:
        with self._lock:
            self._decoder_state = dict(state)

    @staticmethod
    def _delivery(log: list[dict]) -> dict:
        errors_ms = [abs(e['actual_delta_sec'] - e['target_delta_sec']) * 1000 for e in log]
        deltas = [e['actual_delta_sec'] for e in log]
        return {
            'samples': len(log),
            'mean_delta_sec': sum(deltas) / len(deltas) if deltas else None,
            'mean_error_ms': sum(errors_ms) / len(errors_ms) if errors_ms else None,
            'max_error_ms': max(errors_ms) if errors_ms else None,
        }

    @classmethod
    def _timing_stats(cls, log: list[dict]) -> dict:
        labels: dict = {}
        for entry in log:
            labels.setdefault(entry['label'], []).append(entry)
        return dict(cls._delivery(log),
                    by_label={label: cls._delivery(entries)
                              for label, entries in sorted(labels.items())})

    def snapshot(self) -> dict:
        with self._lock:
            now = self._now()
            cutoff = now - self._window_sec
            timing_stats = self._timing_stats(self._timing_log)
            return {
                'now': now,
                'look_ahead_sec': self._look_ahead_sec,
                'is_playing': self._is_playing and self._end_time is None,
                'beats': [b for b in self._beats if b['t'] >= cutoff],
                'effects': [e for e in self._effects if e.get('end', now) >= cutoff],
                'intents': [e for e in self._intents if e.get('end', now) >= cutoff],
                'current_effect': self._effects[-1] if self._effects else None,
                'bpm': self._beats[-1]['bpm'] if self._beats else 0.0,
                'beats_detected': len(self._beats),
                'intent': self._current_intent,
                'sound_events': [e for e in self._sound_events if e['t'] >= cutoff],
                'timing_stats': timing_stats,
                'decoder': dict(self._decoder_state),
            }

    def to_report(self, timing_log: list[dict] | None = None) -> dict:
        with self._lock:
            now = self._now()

            all_effects = list(self._effects)
            if all_effects and 'end' not in all_effects[-1]:
                all_effects[-1] = {**all_effects[-1], 'end': now}

            all_intents = list(self._intents)
            if all_intents and 'end' not in all_intents[-1]:
                all_intents[-1] = {**all_intents[-1], 'end': now}

            tlog = timing_log if timing_log is not None else self._timing_log
            errors_ms = [
                abs(e['actual_delta_sec'] - e['target_delta_sec']) * 1000
                for e in tlog
            ]
            durations = [e['end'] - e['t'] for e in all_effects if 'end' in e]
            unique_channels = {e['channel'] for e in all_effects}
            all_beats = list(self._beats)

            intent_distribution: dict[str, float] = {}
            for entry in all_intents:
                if 'end' not in entry:
                    continue
                dur = entry['end'] - entry['t']
                intent_distribution[entry['intent']] = (
                    intent_distribution.get(entry['intent'], 0.0) + dur
                )
            dominant_intent = (
                max(intent_distribution, key=intent_distribution.__getitem__)
                if intent_distribution else None
            )

            return {
                'duration_sec': now,
                'beats': all_beats,
                'effects': all_effects,
                'intents': all_intents,
                'timing_log': tlog,
                'metrics': {
                    'look_ahead_sec': self._look_ahead_sec,
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
                    'intent_changes_count': len(all_intents),
                    'unique_intents_count': len(intent_distribution),
                    'intent_distribution_sec': {
                        k: round(v, 2) for k, v in sorted(intent_distribution.items())
                    },
                    'dominant_intent': dominant_intent,
                },
            }
