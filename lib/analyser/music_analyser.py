import datetime
import logging
import math
import aubio
import numpy as np
from collections import deque
from lib.analyser.drift_watchdog import DriftWatchdog, ShedLevel
from lib.analyser.madmom_rhythm import MadmomRhythm
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler
from lib.analyser.yamnet_change_detector import YamnetChangeDetector
from lib.clock import Clock, SYSTEM_CLOCK

_ONSET_DENSITY_WINDOW_SEC = 1.5
_BPM_BEAT_WINDOW = 9
_NOTE_REFRACTORY = datetime.timedelta(milliseconds=75)
_RHYTHM_LOG_INTERVAL = datetime.timedelta(seconds=10)

# ~150 ms of per-buffer samples at 5.805 ms/buffer, and ~6 s of per-beat ones.
_ENERGY_WINDOW_BUFFERS = 26
_TREND_WINDOW_BEATS = 12

# The mel FFT's group delay pushes the sub-bass peak past the reported beat, so
# the capture window straddles it.
_KICK_CAPTURE_PRE_BUFFERS = 2
_KICK_CAPTURE_POST_BUFFERS = 6
_KICK_BACKGROUND_BUFFERS = 200
_KICK_SMOOTHING_BEATS = 4
_KICK_MAX_RATIO = 10.0
_KICK_MIN_RMS = 0.005

# Sits below every presence threshold, so an unknown kick reads as absent.
KICK_UNKNOWN = 1.0
# Negative because a rate cannot be: no sparse passage can be mistaken for one
# where the detector was not running. Branch on it, never average it.
DENSITY_UNKNOWN = -1.0


def density_is_known(density: float) -> bool:
    return density >= 0.0


class MelFilterbank:
    """FFT into a 40-band Slaney mel bank — the spectral front-end."""

    BANDS = 40

    def __init__(self, sample_rate: int, buffer_size: int):
        self.sample_rate = sample_rate
        self.win_s = buffer_size * 4
        self.hop_s = buffer_size
        self._pvoc = aubio.pvoc(self.win_s, self.hop_s)
        self._filterbank = aubio.filterbank(self.BANDS, self.win_s)
        self._filterbank.set_mel_coeffs_slaney(sample_rate)

    def __call__(self, audio_signal: np.ndarray) -> np.ndarray:
        return self._filterbank(self._pvoc(audio_signal))


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

        self._mel_band_indices: np.ndarray = np.arange(MelFilterbank.BANDS)
        self.click_sound: float = 0.15 * np.sin(
            2. * np.pi * np.arange(self.buffer_size) / self.buffer_size
            * self.sample_rate / 3000.)

        # Constructed once and never rebuilt by _reset_state(), which fires on
        # every 0.3 s of silence: the rhythm stack is cleared in place there
        # rather than reloading eight pickled LSTMs mid-show, and everything
        # else here is LOOP-scoped — a song boundary says nothing about whether
        # the loop is keeping up.
        self._rhythm: MadmomRhythm = MadmomRhythm(self.sample_rate)
        self._drift: DriftWatchdog = DriftWatchdog(self.buffer_size / self.sample_rate,
                                                   clock=self._clock)
        self._section_detection_enabled: bool = True
        self._rhythm_log_at: datetime.datetime = self._clock.now() + _RHYTHM_LOG_INTERVAL
        self._beats_since_log: int = 0
        self._onsets_since_log: int = 0
        self._last_beat_activation: float = 0.0
        self._reset_state()

    def _reset_state(self) -> None:
        self.mel: MelFilterbank = MelFilterbank(self.sample_rate, self.buffer_size)
        self._rhythm.reset()

        # Beat instants in madmom's own stream time: if the input drops audio,
        # stream time still measures the music the detector actually heard.
        self._beat_stream_times: deque = deque(maxlen=_BPM_BEAT_WINDOW)
        # Set together, always: an emptied window is UNMEASURED until a full
        # window of audio has refilled it, on this path and on a shed-restore.
        self._onset_epoch_seen: int = self._rhythm.onset_epoch
        self._density_valid_from: datetime.datetime | None = (
            self._clock.now() + datetime.timedelta(seconds=_ONSET_DENSITY_WINDOW_SEC))

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
        self._onset_times: deque = deque(maxlen=500)
        self._density_samples: deque = deque(maxlen=_TREND_WINDOW_BEATS)
        self._mel_energies_window: deque = deque(maxlen=_ENERGY_WINDOW_BUFFERS)
        self._rms_window: deque = deque(maxlen=_ENERGY_WINDOW_BUFFERS)
        self._all_sub_bass_samples: deque = deque(maxlen=_KICK_BACKGROUND_BUFFERS)
        self._kick_ratios: deque = deque(maxlen=_KICK_SMOOTHING_BEATS)
        self._pending_kick_beats: deque = deque()
        self._buffer_index: int = -1
        self._centroid_window: deque = deque(maxlen=_ENERGY_WINDOW_BUFFERS)
        self._beat_centroid_samples: deque = deque(maxlen=_TREND_WINDOW_BEATS)

    def start(self):
        self.yamnet_change_detector.start()

    def get_song_current_duration(self) -> datetime.timedelta:
        if self.is_playing:
            return self.song_current_time - self.song_start_time
        else:
            return datetime.timedelta(seconds=0)

    def get_beat_position(self) -> float:
        # Read once: a song reset on the audio thread can zero this between the
        # guard and the division, and the OS2L sender that calls this has no
        # exception handler to survive it.
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
        """Median recent inter-beat interval; 0.0 until measurable.

        Zero during warmup is deliberate — the DROP branch gates on a tempo
        floor, so an unmeasured tempo must read as "not fast", never as a guess.
        """
        if len(self._beat_stream_times) < 3:
            return 0.0
        interval = float(np.median(np.diff(np.array(self._beat_stream_times))))
        return 60.0 / interval if interval > 0 else 0.0

    @staticmethod
    def _fold_bpm(bpm: float) -> float:
        """Fold into [85, 170) by octave halving/doubling; 0.0 if not measurable."""
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
        """Onsets per second over the last 1.5 s, or DENSITY_UNKNOWN.

        Unknown while the detector is shed and for one full window after it is
        restored: an empty window would report a low rate, not a missing one.
        """
        now = self._clock.now()
        enabled = self._rhythm.onsets_enabled
        if self._rhythm.onset_epoch != self._onset_epoch_seen:
            self._onset_epoch_seen = self._rhythm.onset_epoch
            self._onset_times.clear()
            self._density_samples.clear()
            self._density_valid_from = now + datetime.timedelta(
                seconds=_ONSET_DENSITY_WINDOW_SEC)
        if not enabled:
            return DENSITY_UNKNOWN
        if self._density_valid_from is not None:
            if now < self._density_valid_from:
                return DENSITY_UNKNOWN
            self._density_valid_from = None
        cutoff = now - datetime.timedelta(seconds=_ONSET_DENSITY_WINDOW_SEC)
        while self._onset_times and self._onset_times[0] < cutoff:
            self._onset_times.popleft()
        return len(self._onset_times) / _ONSET_DENSITY_WINDOW_SEC

    def get_onset_density_trend(self) -> float:
        """Recent vs past onset density; >1.0 rising, 1.0 until measurable."""
        samples = list(self._density_samples)
        if len(samples) < 4:
            return 1.0
        mid = len(samples) // 2
        past_mean = sum(samples[:mid]) / mid
        recent_mean = sum(samples[mid:]) / (len(samples) - mid)
        return recent_mean / past_mean if past_mean > 0 else 1.0

    def get_seconds_since_last_beat(self) -> float:
        return (self._clock.now() - self.last_beat_detected).total_seconds()

    def get_sub_bass_ratio(self) -> float:
        """Fraction of mel energy in bands 0-4 (~60-250 Hz); 0.0 until measurable."""
        if not self._mel_energies_window:
            return 0.0
        mean_energies = np.mean(np.array(list(self._mel_energies_window)), axis=0)
        total = float(np.sum(mean_energies)) + 1e-8
        sub_bass = float(np.sum(mean_energies[:5]))
        return sub_bass / total

    def get_rms_energy(self) -> float:
        if not self._rms_window:
            return 0.0
        return sum(self._rms_window) / len(self._rms_window)

    def get_kick_strength(self) -> float:
        """Peak beat-locked sub-bass over its own floor; ~1.0 means no kick.

        Lags one beat — see `_resolve_pending_kicks`.
        """
        if not self._kick_ratios:
            return KICK_UNKNOWN
        if self.get_rms_energy() < _KICK_MIN_RMS:
            return KICK_UNKNOWN
        return sum(self._kick_ratios) / len(self._kick_ratios)

    def get_spectral_centroid_trend(self) -> float:
        """Recent vs past centroid across beats; >1.0 is a riser, 1.0 until measurable."""
        samples = list(self._beat_centroid_samples)
        if len(samples) < 4:
            return 1.0
        mid = len(samples) // 2
        past_mean = sum(samples[:mid]) / mid
        recent_mean = sum(samples[mid:]) / (len(samples) - mid)
        return recent_mean / (past_mean + 1e-8)

    async def analyse(self, audio_signal: np.ndarray) -> np.ndarray:
        now = self._clock.now()

        shed = self._drift.observe()
        self._rhythm.set_onsets_enabled(shed < ShedLevel.ONSET_DETECTION)
        self._set_section_detection_enabled(shed < ShedLevel.SECTION_DETECTION)

        rms = float(np.sqrt(np.mean(audio_signal ** 2)))
        self._rms_window.append(rms)
        energies = self._compute_mel_energies(audio_signal)
        self._track_song_duration(energies, now)

        rhythm = self._rhythm.process(audio_signal)
        if rhythm.beats:
            self._last_beat_activation = rhythm.beat_activation
        await self._track_onset(rhythm.onsets, now)
        is_beat = await self._track_beat(rhythm.beats, now)
        await self._track_note(rhythm.onsets, now)
        self._log_rhythm_state(now)

        if self.get_song_current_duration() > datetime.timedelta(minutes=15):
            self._reset_state()

        if (self._section_detection_enabled
                and self.yamnet_change_detector.detect_change(
                    audio_signal, self.get_song_current_duration())):
            await self.handler.on_section_change()

        if is_beat and self.note_clicks:
            # Owner preference: the -d click marks the BEAT, not every onset.
            # Deliberately not the aubio note-trigger this replaced — beats are
            # ~2.1/s against ~3.6/s of onsets, so the monitor is sparser and
            # lands on the pulse a DJ is listening for.
            # Must stay last: every feature above ran on the clean signal.
            audio_signal += self.click_sound

        await self.handler.on_cycle()
        return audio_signal

    def _set_section_detection_enabled(self, enabled: bool) -> None:
        if enabled == self._section_detection_enabled:
            return
        if enabled:
            # While shed it was fed no audio, so everything it holds describes
            # music from before the gap.
            self.yamnet_change_detector.reset()
        logging.warning('[yamnet] section detection %s',
                        'restored' if enabled else 'SHED — no section changes will fire')
        self._section_detection_enabled = enabled

    def _log_rhythm_state(self, now: datetime.datetime) -> None:
        if now < self._rhythm_log_at:
            return
        window = (now - (self._rhythm_log_at - _RHYTHM_LOG_INTERVAL)).total_seconds()
        self._rhythm_log_at = now + _RHYTHM_LOG_INTERVAL
        drift = self.get_drift_status()
        logging.info(
            f'[rhythm] madmom {self._beats_since_log} beats '
            f'({self._beats_since_log / window:.2f}/s), '
            f'{self._onsets_since_log} onsets ({self._onsets_since_log / window:.2f}/s) '
            f'over {window:.1f}s | bpm={self.get_bpm():.1f} '
            f'| last beat activation={self._last_beat_activation:.3f} '
            f'| drift={drift["drift_sec"]:+.3f}s shed={drift["shed_level"]} '
            f'| adapter lag={drift["adapter_latency_sec"] * 1000:.1f}ms')
        self._beats_since_log = 0
        self._onsets_since_log = 0

    def get_drift_status(self) -> dict:
        return {
            'shed_level': self._drift.level.name,
            'drift_sec': round(self._drift.drift_sec, 4),
            'peak_drift_sec': round(self._drift.peak_drift_sec, 4),
            'total_drift_sec': round(self._drift.total_drift_sec, 4),
            'adapter_latency_sec': round(self._rhythm.pending_latency_sec, 5),
        }

    async def _track_onset(self, onsets: list, now: datetime.datetime) -> bool:
        for onset_time in onsets:
            self._onset_times.append(now)
            self._onsets_since_log += 1
            logging.debug(f'[rhythm] onset @ {onset_time:.3f}s (madmom stream)')
            await self.handler.on_onset()
        return bool(onsets)

    async def _track_beat(self, beats: list, now: datetime.datetime) -> bool:
        for beat_time in beats:
            interval = (beat_time - self._beat_stream_times[-1]
                        if self._beat_stream_times else 0.0)
            self._beat_stream_times.append(beat_time)
            self._beats_since_log += 1
            density = self.get_onset_density()
            this_bpm: float = self.get_bpm()
            logging.debug(
                f'[rhythm] beat #{self.beat_count + 1} @ {beat_time:.3f}s '
                f'(madmom stream), interval={interval:.3f}s, bpm={this_bpm:.1f}, '
                f'activation={self._last_beat_activation:.3f}')
            bpm_changed: bool = self._has_bpm_changed(this_bpm)
            self.beat_count += 1
            if density_is_known(density):
                self._density_samples.append(density)
            self._pending_kick_beats.append(self._buffer_index)
            if self._centroid_window:
                self._beat_centroid_samples.append(self._centroid_window[-1])
            await self.handler.on_beat(self.beat_count, this_bpm, bpm_changed)
            self.last_bpm = this_bpm
            self.time_to_last_beat_sec = (now - self.last_beat_detected).total_seconds()
            self.last_beat_detected = now
        return bool(beats)

    def _resolve_pending_kicks(self) -> None:
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

    async def _track_note(self, onsets: list, now: datetime.datetime) -> bool:
        is_note = bool(onsets) and now - self.last_note_detected > _NOTE_REFRACTORY
        if is_note:
            await self.handler.on_note()
            self.last_note_detected = now
        return is_note

    def _compute_mel_energies(self, audio_signal: np.ndarray) -> np.ndarray:
        energies_out = self.mel(audio_signal)

        self._mel_energies_window.append(energies_out.copy())

        raw_sub_bass = float(np.sum(energies_out[:5]))
        self._buffer_index += 1
        self._all_sub_bass_samples.append(raw_sub_bass)
        self._resolve_pending_kicks()

        total_energy = float(np.sum(energies_out))
        centroid = float(np.dot(self._mel_band_indices, energies_out)) / (total_energy + 1e-8)
        self._centroid_window.append(centroid)

        return energies_out

    @staticmethod
    def _is_silence(energies: np.ndarray) -> bool:
        return bool(np.all(np.abs(energies) < 0.0001))

    def _track_song_duration(self, energies: np.ndarray, now: datetime.datetime) -> None:
        if not self._is_silence(energies):
            self.silence_period_start = now

        if now - self.silence_period_start > datetime.timedelta(seconds=0.3):
            self._on_sound_stop()
        else:
            self.song_current_time = now

        if not self.is_playing and now - self.song_start_time > datetime.timedelta(seconds=0.3):
            self._on_sound_start()

    def _on_sound_start(self):
        self.is_playing = True
        self.yamnet_change_detector.reset()
        self.handler.on_sound_start()

    def _on_sound_stop(self):
        was_playing = self.is_playing
        self._reset_state()
        if was_playing:
            handler_started = self._clock.monotonic()
            self.handler.on_sound_stop()
            self._drift.forgive(self._clock.monotonic() - handler_started)

    def _has_bpm_changed(self, current_bpm: float) -> bool:
        if not self.is_playing or current_bpm <= 0:
            return False
        return (abs(current_bpm - self.last_bpm) / current_bpm) > 0.05
