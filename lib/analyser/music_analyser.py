import datetime
import logging
import math
import aubio
import numpy as np
from collections import deque
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler
from lib.analyser.yamnet_change_detector import YamnetChangeDetector
from lib.clock import Clock, SYSTEM_CLOCK

_ONSET_DENSITY_WINDOW_SEC = 1.5  # rolling window for onset density calculation

# --- Kick strength -----------------------------------------------------------
# The mel FFT delays the sub-bass peak past aubio's beat, so the window straddles it.
_KICK_CAPTURE_PRE_BUFFERS  = 2
_KICK_CAPTURE_POST_BUFFERS = 6
_KICK_BACKGROUND_BUFFERS = 200   # median over ~1.2 s = the floor the kick sits on
_KICK_SMOOTHING_BEATS = 4
_KICK_MAX_RATIO = 10.0
_KICK_MIN_RMS = 0.005            # below this the track is silent and ratios are meaningless

# Kick presence unknown — below any presence threshold, so it reads as absent.
KICK_UNKNOWN = 1.0


class MusicAnalyser:
    def __init__(self,
                 sample_rate: int,
                 buffer_size: int,
                 handler: IMusicAnalyserHandler,
                 clock: Clock = SYSTEM_CLOCK,
                 note_clicks: bool = False):
        self._clock: Clock = clock
        self.sample_rate: int = sample_rate
        self.buffer_size: int = buffer_size
        self.handler: IMusicAnalyserHandler = handler
        self.note_clicks: bool = note_clicks
        self.yamnet_change_detector: YamnetChangeDetector = YamnetChangeDetector(self.sample_rate, self.buffer_size)

        # constants
        self.win_s: int = self.buffer_size * 4  # fft size
        self.win_s_small: int = self.buffer_size * 2  # fft size
        self.hop_s: int = self.buffer_size  # hop size
        self.mel_filters: int = 40  # slaney mel filterbank band count
        self._mel_band_indices: np.ndarray = np.arange(self.mel_filters)
        # Debug note-click amplitude: quiet enough to sit under the music.
        self.click_sound: float = 0.15 * np.sin(2. * np.pi * np.arange(self.hop_s) / self.hop_s * self.sample_rate / 3000.)

        self._reset_state()

    def _reset_state(self) -> None:
        # audio analysers
        self.tempo_o: aubio.tempo = aubio.tempo("default", self.win_s_small, self.hop_s, self.sample_rate)
        self.onset_o: aubio.onset = aubio.onset("default", self.win_s_small, self.hop_s, self.sample_rate)
        self.notes_o = aubio.notes("default", self.win_s_small, self.hop_s, self.sample_rate)
        self.pvoc_o: aubio.pvoc = aubio.pvoc(self.win_s, self.hop_s)
        self.energy_filter = aubio.filterbank(self.mel_filters, self.win_s)
        self.energy_filter.set_mel_coeffs_slaney(self.sample_rate)

        # tracking state
        self.yamnet_change_detector.reset()
        self.is_playing: bool = False
        self.song_start_time: datetime.datetime = self._clock.now()
        self.song_current_time: datetime.datetime = self._clock.now()
        self.silence_period_start: datetime.datetime = self._clock.now()
        self.last_bpm: float = 0.0
        self.beat_count: int = 0
        self.time_to_last_beat_sec: float = 0
        self.last_beat_detected: datetime.datetime = self._clock.now()
        self.last_note_detected: datetime.datetime = self._clock.now()
        # rolling window of onset timestamps for onset-density calculation
        self._onset_times: deque = deque(maxlen=500)
        # per-beat density samples for trend detection (maxlen=12 ≈ ~6s at 120 BPM)
        self._density_samples: deque = deque(maxlen=12)
        # Rolling mel-energy frames and RMS values for sub-bass / energy features.
        # maxlen≈26 ≈ 150 ms of buffers at 5.8 ms/buffer — long enough for a stable mean.
        self._mel_energies_window: deque = deque(maxlen=26)
        self._rms_window: deque = deque(maxlen=26)
        # Kick detection: beats are queued by buffer index and resolved once their
        # capture window has closed.
        self._all_sub_bass_samples: deque = deque(maxlen=_KICK_BACKGROUND_BUFFERS)
        self._kick_ratios: deque = deque(maxlen=_KICK_SMOOTHING_BEATS)
        self._pending_kick_beats: deque = deque()
        self._buffer_index: int = -1
        # Spectral centroid tracked per-buffer and at beat times (for trend detection).
        self._centroid_window: deque = deque(maxlen=26)
        self._beat_centroid_samples: deque = deque(maxlen=12)

    def start(self):
        self.yamnet_change_detector.start()

    def get_song_current_duration(self) -> datetime.timedelta:
        if self.is_playing:
            return self.song_current_time - self.song_start_time
        else:
            return datetime.timedelta(seconds=0)

    def get_beat_position(self) -> float:
        if self.is_playing and self.time_to_last_beat_sec > 0:
            time_to_current_beat_sec = (self._clock.now() - self.last_beat_detected).total_seconds()
            beat_percent_elapsed = time_to_current_beat_sec / self.time_to_last_beat_sec
            return self.beat_count + abs(beat_percent_elapsed)
        else:
            return 0

    def get_bpm(self) -> float:
        if self.is_playing:
            return self._fold_bpm(self.tempo_o.get_bpm())
        else:
            return 0

    @staticmethod
    def _fold_bpm(bpm: float) -> float:
        """Fold BPM into [85, 170) by octave halving/doubling; 0.0 if not finite or ≤ 0.

        aubio locks onto double/half tempo during warmup and on ambiguous material.
        """
        if not math.isfinite(bpm) or bpm <= 0:
            return 0.0
        while bpm >= 170.0:
            bpm /= 2.0
        while bpm < 85.0:
            bpm *= 2.0
        return bpm

    def is_song_playing(self) -> bool:
        return self.is_playing

    def get_onset_density(self) -> float:
        """Onsets per second over the last 1.5 seconds (rolling window)."""
        now = self._clock.now()
        cutoff = now - datetime.timedelta(seconds=_ONSET_DENSITY_WINDOW_SEC)
        while self._onset_times and self._onset_times[0] < cutoff:
            self._onset_times.popleft()
        return len(self._onset_times) / _ONSET_DENSITY_WINDOW_SEC

    def get_onset_density_trend(self) -> float:
        """Ratio of recent vs past onset density.

        Returns >1.0 when density is rising (buildup feel), <1.0 when falling.
        Returns 1.0 when there is insufficient data (< 4 samples).
        """
        samples = list(self._density_samples)
        if len(samples) < 4:
            return 1.0
        mid = len(samples) // 2
        past_mean = sum(samples[:mid]) / mid
        recent_mean = sum(samples[mid:]) / (len(samples) - mid)
        return recent_mean / past_mean if past_mean > 0 else 1.0

    def get_seconds_since_last_beat(self) -> float:
        """Seconds elapsed since the last detected beat."""
        return (self._clock.now() - self.last_beat_detected).total_seconds()

    def get_sub_bass_ratio(self) -> float:
        """Fraction of mel filterbank energy in sub-bass bands (bands 0–4, ~60–250 Hz).

        Returns 0.0 when no frames have been processed yet.  A high ratio (≥ 0.25)
        strongly suggests a kick drum or bass-heavy DROP rather than a hi-hat pattern.
        """
        if not self._mel_energies_window:
            return 0.0
        mean_energies = np.mean(np.array(list(self._mel_energies_window)), axis=0)
        total = float(np.sum(mean_energies)) + 1e-8
        sub_bass = float(np.sum(mean_energies[:5]))
        return sub_bass / total

    def get_rms_energy(self) -> float:
        """Mean RMS amplitude over the recent analysis window (last ~150 ms).

        Returns 0.0 when no frames have been processed yet.
        """
        if not self._rms_window:
            return 0.0
        return sum(self._rms_window) / len(self._rms_window)

    def get_kick_strength(self) -> float:
        """How strongly sub-bass energy spikes on the beat, averaged over recent beats.

        Each beat contributes its peak sub-bass in a window straddling it over the
        median sub-bass of the surrounding ~1.2 s; ~1.0 means no beat-locked bass.
        Returns KICK_UNKNOWN before the first measurement and below the silence gate.
        Lags by one beat (see `_resolve_pending_kicks`).
        """
        if not self._kick_ratios:
            return KICK_UNKNOWN
        if self.get_rms_energy() < _KICK_MIN_RMS:
            return KICK_UNKNOWN
        return sum(self._kick_ratios) / len(self._kick_ratios)

    def get_spectral_centroid_trend(self) -> float:
        """Trend of the spectral centroid across recent beats (ratio of recent vs. past half).

        > 1.0 means the centroid is rising — energy moving toward higher frequencies.
        This is the defining signature of a BUILDUP riser (sweep filter opening, noise sweep).
        < 1.0 means centroid is falling — energy concentrating in the bass (DROP incoming).
        Returns 1.0 (neutral) when fewer than 4 beat samples have been collected.
        """
        samples = list(self._beat_centroid_samples)
        if len(samples) < 4:
            return 1.0
        mid = len(samples) // 2
        past_mean   = sum(samples[:mid]) / mid
        recent_mean = sum(samples[mid:]) / (len(samples) - mid)
        return recent_mean / (past_mean + 1e-8)

    async def analyse(self, audio_signal: np.ndarray) -> np.ndarray:
        now = self._clock.now()

        rms = float(np.sqrt(np.mean(audio_signal ** 2)))
        self._rms_window.append(rms)
        energies = self._compute_mel_energies(audio_signal)
        self._track_song_duration(energies, now)

        await self._track_onset(audio_signal)
        await self._track_beat(audio_signal, now)
        is_note = await self._track_note(audio_signal, now)

        if self.get_song_current_duration() > datetime.timedelta(minutes=15):
            self._reset_state()

        if self.yamnet_change_detector.detect_change(audio_signal, self.get_song_current_duration()):
            await self.handler.on_section_change()

        if is_note and self.note_clicks:
            # Audible click for debug playback monitoring (returned buffer only —
            # feature extraction above already ran on the clean signal).
            audio_signal += self.click_sound

        await self.handler.on_cycle()
        return audio_signal

    async def _track_onset(self, audio_signal: np.ndarray) -> bool:
        is_onset: bool = self.onset_o(audio_signal)[0] > 0
        if is_onset:
            self._onset_times.append(self._clock.now())
            await self.handler.on_onset()
        return is_onset

    async def _track_beat(self, audio_signal: np.ndarray, now: datetime.datetime) -> bool:
        is_beat: bool = self.tempo_o(audio_signal)[0] > 0
        if is_beat:
            this_bpm: float = self.get_bpm()
            bpm_changed: bool = self._has_bpm_changed(this_bpm)
            self.beat_count += 1
            self._density_samples.append(self.get_onset_density())
            self._pending_kick_beats.append(self._buffer_index)
            if self._centroid_window:
                self._beat_centroid_samples.append(self._centroid_window[-1])
            await self.handler.on_beat(self.beat_count, this_bpm, bpm_changed)
            self.last_bpm = self.get_bpm()
            self.time_to_last_beat_sec = (now - self.last_beat_detected).total_seconds()
            self.last_beat_detected = now
        return is_beat

    def _resolve_pending_kicks(self) -> None:
        """Compute the kick ratio of every queued beat whose capture window has closed."""
        while (self._pending_kick_beats
               and self._buffer_index - self._pending_kick_beats[0] >= _KICK_CAPTURE_POST_BUFFERS):
            beat_index = self._pending_kick_beats.popleft()
            samples = list(self._all_sub_bass_samples)
            beat_pos = len(samples) - 1 - (self._buffer_index - beat_index)
            start = max(0, beat_pos - _KICK_CAPTURE_PRE_BUFFERS)
            end = min(len(samples), beat_pos + _KICK_CAPTURE_POST_BUFFERS + 1)
            if end <= start:
                continue
            peak = max(samples[start:end])
            background = float(np.median(samples))
            if background < 1e-8:
                continue
            self._kick_ratios.append(min(peak / background, _KICK_MAX_RATIO))

    async def _track_note(self, audio_signal: np.ndarray, now: datetime.datetime) -> bool:
        note = self.notes_o(audio_signal)
        is_note = note[0] > 0 and now - self.last_note_detected > datetime.timedelta(milliseconds=75)
        if is_note:
            await self.handler.on_note()
            self.last_note_detected = now
        return is_note

    def _compute_mel_energies(self, audio_signal: np.ndarray) -> np.ndarray:
        spec = self.pvoc_o(audio_signal)
        energies_out = self.energy_filter(spec)

        self._mel_energies_window.append(energies_out.copy())

        # Kick detection: raw sub-bass energy (not normalised — we want the actual spike magnitude).
        raw_sub_bass = float(np.sum(energies_out[:5]))
        self._buffer_index += 1
        self._all_sub_bass_samples.append(raw_sub_bass)
        self._resolve_pending_kicks()

        # Spectral centroid in mel-band index units (0–39).
        total_energy = float(np.sum(energies_out))
        centroid = float(np.dot(self._mel_band_indices, energies_out)) / (total_energy + 1e-8)
        self._centroid_window.append(centroid)

        return energies_out

    @staticmethod
    def _is_silence(energies: np.ndarray) -> bool:
        """All mel bands within ±1e-4 of zero (strict) — vectorized hot path."""
        return bool(np.all(np.abs(energies) < 0.0001))

    def _track_song_duration(self, energies: np.ndarray, now: datetime.datetime) -> None:
        is_silence_now: bool = self._is_silence(energies)

        # if it is silent now, we do not update silence_period_start in order to track the duration of the silence
        if not is_silence_now:
            self.silence_period_start = now

        # if there was sound, and then we had no sound for 0.3 seconds, set state to is not playing
        if now - self.silence_period_start > datetime.timedelta(seconds=0.3):
            self._on_sound_stop()
        else:
            self.song_current_time = now

        # if there was no sound, and then we had sound for 0.3 seconds, set state to is playing
        if not self.is_playing and now - self.song_start_time > datetime.timedelta(seconds=0.3):
            self._on_sound_start()

    def _on_sound_start(self):
        self.is_playing  = True
        self.yamnet_change_detector.reset()
        self.handler.on_sound_start()

    def _on_sound_stop(self):
        is_playing = self.is_playing
        self._reset_state()  # this sets self.is_playing to False, so we save the state before
        if is_playing:
            self.handler.on_sound_stop()

    def _has_bpm_changed(self, current_bpm: float) -> bool:
        if self.is_playing:
            # 5% change in bpm constitutes a change in bpm, defined arbitrarily
            return (abs(current_bpm - self.last_bpm) / current_bpm) > 0.05
        else:
            return False
