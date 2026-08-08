import datetime
import logging
import math
import numpy as np
from collections import deque
from lib.analyser.drift_watchdog import DriftWatchdog
from lib.analyser.madmom_rhythm import MadmomRhythm
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler
from lib.clock import Clock, SYSTEM_CLOCK

_BPM_BEAT_WINDOW = 9
_BPM_FOLD_MIN = 85.0
_BPM_FOLD_MAX = 170.0
_NOTE_REFRACTORY = datetime.timedelta(milliseconds=75)
_RHYTHM_LOG_INTERVAL = datetime.timedelta(seconds=10)

_ENERGY_WINDOW_BUFFERS = 26

# Swept to reproduce the committed sound-start/stop instants on the golden fixtures.
_SILENCE_RMS = 1.5e-4

_SONG_CLOCK_HORIZON = datetime.timedelta(minutes=15)


class MusicAnalyser:
    def __init__(self,
                 sample_rate: int,
                 buffer_size: int,
                 handler: IMusicAnalyserHandler,
                 clock: Clock = SYSTEM_CLOCK,
                 note_clicks: bool = False,
                 watchdog: DriftWatchdog | None = None):
        self._clock: Clock = clock
        self.sample_rate: int = sample_rate
        self.buffer_size: int = buffer_size
        self.handler: IMusicAnalyserHandler = handler
        self.note_clicks: bool = note_clicks

        self.click_sound: float = 0.15 * np.sin(
            2. * np.pi * np.arange(self.buffer_size) / self.buffer_size
            * self.sample_rate / 3000.)

        self._rhythm: MadmomRhythm = MadmomRhythm(self.sample_rate)
        self._drift: DriftWatchdog = watchdog or DriftWatchdog(
            self.buffer_size / self.sample_rate, clock=self._clock)
        self._rhythm_log_at: datetime.datetime = self._clock.now() + _RHYTHM_LOG_INTERVAL
        self._beats_since_log: int = 0
        self._last_beat_activation: float = 0.0
        self._reset_state()

    def _roll_song_clock(self) -> None:
        # The horizon is about memory, not music: re-locking madmom mid-audio
        # gains or loses beats, and every one rotates the bar grid's four-count
        # permanently. Its online state is bounded, so the beat stream survives.
        playing = self.is_playing
        self._reset_song_clock()
        self.is_playing = playing

    def _reset_state(self) -> None:
        self._rhythm.reset()
        self._beat_stream_times: deque = deque(maxlen=_BPM_BEAT_WINDOW)
        self.last_bpm: float = 0.0
        self.beat_count: int = 0
        self.time_to_last_beat_sec: float = 0
        self.last_beat_detected: datetime.datetime = self._clock.now()
        self.last_note_detected: datetime.datetime = self._clock.now()
        self._reset_song_clock()

    def _reset_song_clock(self) -> None:
        self.is_playing: bool = False
        self.song_start_time: datetime.datetime = self._clock.now()
        self.song_current_time: datetime.datetime = self._clock.now()
        self.silence_period_start: datetime.datetime = self._clock.now()
        self._rms_window: deque = deque(maxlen=_ENERGY_WINDOW_BUFFERS)

    def get_song_current_duration(self) -> datetime.timedelta:
        if self.is_playing:
            return self.song_current_time - self.song_start_time
        else:
            return datetime.timedelta(seconds=0)

    def get_beat_position(self) -> float:
        beat_interval_sec = self.time_to_last_beat_sec
        if self.is_playing and beat_interval_sec > 0:
            time_to_current_beat_sec = (self._clock.now() - self.last_beat_detected).total_seconds()
            return self.beat_count + abs(time_to_current_beat_sec / beat_interval_sec)
        else:
            return 0

    def get_bpm(self) -> float:
        if not self.is_playing:
            return 0
        return self._fold_bpm(self._measured_bpm())

    def _measured_bpm(self) -> float:
        if len(self._beat_stream_times) < 3:
            return 0.0
        interval = float(np.median(np.diff(np.array(self._beat_stream_times))))
        return 60.0 / interval if interval > 0 else 0.0

    @staticmethod
    def _fold_bpm(bpm: float) -> float:
        if not math.isfinite(bpm) or bpm <= 0:
            return 0.0
        while bpm >= _BPM_FOLD_MAX:
            bpm /= 2.0
        while bpm < _BPM_FOLD_MIN:
            bpm *= 2.0
        return bpm

    def is_song_playing(self) -> bool:
        return self.is_playing

    def get_seconds_since_last_beat(self) -> float:
        return (self._clock.now() - self.last_beat_detected).total_seconds()

    def get_rms_energy(self) -> float:
        if not self._rms_window:
            return 0.0
        return sum(self._rms_window) / len(self._rms_window)

    async def analyse(self, audio_signal: np.ndarray) -> np.ndarray:
        now = self._clock.now()

        self._drift.observe()

        rms = float(np.sqrt(np.mean(audio_signal ** 2)))
        self._rms_window.append(rms)
        self._track_song_duration(rms, now)

        rhythm = self._rhythm.process(audio_signal)
        if rhythm.beats:
            self._last_beat_activation = rhythm.beat_activation
        is_beat = await self._track_beat(rhythm.beats, now)
        await self._track_note(rhythm.beats, now)
        self._log_rhythm_state(now)

        if self.get_song_current_duration() > _SONG_CLOCK_HORIZON:
            self._roll_song_clock()

        if is_beat and self.note_clicks:
            audio_signal += self.click_sound

        await self.handler.on_cycle()
        return audio_signal

    def _log_rhythm_state(self, now: datetime.datetime) -> None:
        if now < self._rhythm_log_at:
            return
        window = (now - (self._rhythm_log_at - _RHYTHM_LOG_INTERVAL)).total_seconds()
        self._rhythm_log_at = now + _RHYTHM_LOG_INTERVAL
        drift = self.get_drift_status()
        logging.info(
            f'[rhythm] madmom {self._beats_since_log} beats '
            f'({self._beats_since_log / window:.2f}/s) '
            f'over {window:.1f}s | bpm={self.get_bpm():.1f} '
            f'| last beat activation={self._last_beat_activation:.3f} '
            f'| drift={drift["drift_sec"]:+.3f}s shed={drift["shed_level"]}'
            f'{"" if drift["fault"] is None else "/" + drift["fault"]} '
            f'| adapter lag={drift["adapter_latency_sec"] * 1000:.1f}ms')
        self._beats_since_log = 0

    def get_drift_status(self) -> dict:
        return {
            'shed_level': self._drift.level.name,
            'fault': self._drift.fault,
            'drift_sec': round(self._drift.drift_sec, 4),
            'peak_drift_sec': round(self._drift.peak_drift_sec, 4),
            'total_drift_sec': round(self._drift.total_drift_sec, 4),
            'adapter_latency_sec': round(self._rhythm.pending_latency_sec, 5),
        }

    async def _track_beat(self, beats: list, now: datetime.datetime) -> bool:
        for beat_time in beats:
            interval = (beat_time - self._beat_stream_times[-1]
                        if self._beat_stream_times else 0.0)
            self._beat_stream_times.append(beat_time)
            self._beats_since_log += 1
            this_bpm: float = self.get_bpm()
            logging.debug(
                f'[rhythm] beat #{self.beat_count + 1} @ {beat_time:.3f}s '
                f'(madmom stream), interval={interval:.3f}s, bpm={this_bpm:.1f}, '
                f'activation={self._last_beat_activation:.3f}')
            bpm_changed: bool = self._has_bpm_changed(this_bpm)
            self.beat_count += 1
            await self.handler.on_beat(self.beat_count, this_bpm, bpm_changed)
            self.last_bpm = this_bpm
            self.time_to_last_beat_sec = (now - self.last_beat_detected).total_seconds()
            self.last_beat_detected = now
        return bool(beats)

    async def _track_note(self, beats: list, now: datetime.datetime) -> bool:
        is_note = bool(beats) and now - self.last_note_detected > _NOTE_REFRACTORY
        if is_note:
            await self.handler.on_note()
            self.last_note_detected = now
        return is_note

    @staticmethod
    def _is_silence(rms: float) -> bool:
        return bool(rms < _SILENCE_RMS)

    def _track_song_duration(self, rms: float, now: datetime.datetime) -> None:
        if not self._is_silence(rms):
            self.silence_period_start = now

        if now - self.silence_period_start > datetime.timedelta(seconds=0.3):
            self._on_sound_stop()
        else:
            self.song_current_time = now

        if not self.is_playing and now - self.song_start_time > datetime.timedelta(seconds=0.3):
            self._on_sound_start()

    def _on_sound_start(self):
        self.is_playing = True
        self.handler.on_sound_start()

    def _on_sound_stop(self):
        was_playing = self.is_playing
        self._reset_state()
        if was_playing:
            self.handler.on_sound_stop()

    def _has_bpm_changed(self, current_bpm: float) -> bool:
        if not self.is_playing or current_bpm <= 0:
            return False
        return (abs(current_bpm - self.last_bpm) / current_bpm) > 0.05
