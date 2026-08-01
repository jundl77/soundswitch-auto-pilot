"""Thread-safe store of pipeline events — the only state the Dash thread shares."""

import threading
from collections import deque

from lib.clock import Clock, SYSTEM_CLOCK


class EventBuffer:
    def __init__(self, window_sec: float = 60.0, clock: Clock = SYSTEM_CLOCK,
                 look_ahead_sec: float = 0.0):
        self._lock = threading.Lock()
        self._window_sec = window_sec
        self._clock = clock
        # Beats are stamped in song time, intent/effect blocks one look-ahead
        # later in audience time. Recorded so a report is self-describing.
        self._look_ahead_sec = look_ahead_sec
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._is_playing: bool = False
        # An infinite window promises complete reports, so the cap comes off.
        self._beats: deque[dict] = deque(maxlen=None if window_sec == float('inf') else 3000)
        self._effects: list[dict] = []
        self._intents: list[dict] = []
        self._timing_log: list[dict] = []
        self._current_intent: str | None = None
        self._sound_events: list[dict] = []

    def start(self) -> None:
        with self._lock:
            self._start_time = self._clock.monotonic()

    def mark_end(self) -> None:
        """Clamp later timestamps, so the look-ahead flush tail cannot stretch
        a report past the audio that produced it."""
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
                # Onset density scaled the marker size; the density chain is
                # gone and nothing has replaced it, so the channel is constant
                # rather than carrying a number nobody measured.
                'strength': 0.0,
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
        """``t`` is when the room sees it; ``song_t`` is the audio it describes.

        Both, because they are two different facts and only one of them can be
        recovered from the other.  A block's stamp is its fire time, and the
        delay behind it moves per command (B1) -- so no constant de-shift
        reaches song time, and a consumer that guesses one scores the show
        against the wrong part of the track.  The engine knows the instant
        exactly at commit time; this is where it says so.
        """
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

    def snapshot(self) -> dict:
        """Thread-safe copy of recent state — called from Dash every 100 ms."""
        with self._lock:
            now = self._now()
            cutoff = now - self._window_sec
            tlog = self._timing_log
            errors_ms = [abs(e['actual_delta_sec'] - e['target_delta_sec']) * 1000 for e in tlog]
            deltas = [e['actual_delta_sec'] for e in tlog]
            timing_stats = {
                'samples': len(tlog),
                'mean_delta_sec': sum(deltas) / len(deltas) if deltas else None,
                'mean_error_ms': sum(errors_ms) / len(errors_ms) if errors_ms else None,
                'max_error_ms': max(errors_ms) if errors_ms else None,
            }
            return {
                'now': now,
                'is_playing': self._is_playing,
                'beats': [b for b in self._beats if b['t'] >= cutoff],
                'effects': [e for e in self._effects if e.get('end', now) >= cutoff],
                'intents': [e for e in self._intents if e.get('end', now) >= cutoff],
                'current_effect': self._effects[-1] if self._effects else None,
                'bpm': self._beats[-1]['bpm'] if self._beats else 0.0,
                'beats_detected': len(self._beats),
                'intent': self._current_intent,
                'sound_events': [e for e in self._sound_events if e['t'] >= cutoff],
                'timing_stats': timing_stats,
            }

    def to_report(self, timing_log: list[dict] | None = None) -> dict:
        """Full serializable report for agentic evaluation or JSON export."""
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
                    # Subtract from an intent/effect block's bounds to reach song time.
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
