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

_ONSET_DENSITY_WINDOW_SEC = 1.5  # rolling window for onset density calculation

# Beats used for the BPM estimate. Eight intervals is ~3.7 s at 128 BPM — long
# enough that one mistracked beat cannot move the median, short enough to follow
# a real tempo change within a phrase.
_BPM_BEAT_WINDOW = 9

# Debug beeps fire on onsets; without a refractory a dense passage would buzz.
_NOTE_REFRACTORY = datetime.timedelta(milliseconds=75)

# How often the rhythm front-end reports itself at INFO. Per-beat detail is at
# DEBUG: the engine already logs one INFO line per beat, and a second one from
# here would double the live log's volume for no extra information.
_RHYTHM_LOG_INTERVAL = datetime.timedelta(seconds=10)

# --- Kick strength -----------------------------------------------------------
# The mel FFT delays the sub-bass peak past the reported beat, so the window
# straddles it. This is a property of the filterbank's group delay, not of
# whichever detector reports the beat.
_KICK_CAPTURE_PRE_BUFFERS  = 2
_KICK_CAPTURE_POST_BUFFERS = 6
_KICK_BACKGROUND_BUFFERS = 200   # median over ~1.2 s = the floor the kick sits on
_KICK_SMOOTHING_BEATS = 4
_KICK_MAX_RATIO = 10.0
_KICK_MIN_RMS = 0.005            # below this the track is silent and ratios are meaningless

# Kick presence unknown — below any presence threshold, so it reads as absent.
KICK_UNKNOWN = 1.0

# Onset density unmeasured. Negative because a rate cannot be: no genuinely
# sparse passage can be mistaken for one where the detector was not running.
# Consumers must branch on it rather than average it — see `_classify_intent`.
DENSITY_UNKNOWN = -1.0


def density_is_known(density: float) -> bool:
    return density >= 0.0


class MelFilterbank:
    """The aubio spectral front-end: FFT into a 40-band Slaney mel bank.

    Extracted so it can be built and run on its own. Every trained model and
    every spectral feature depends on this exact bank, so tooling that wants to
    fingerprint it should not have to construct an analyser — and therefore load
    an eight-model beat ensemble and a TensorFlow graph — to reach two aubio
    objects.
    """

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

        # constants
        # `win_s_small` is gone with the aubio rhythm objects that were its only
        # consumers; the FFT size for the bank lives in MelFilterbank now.
        self._mel_band_indices: np.ndarray = np.arange(MelFilterbank.BANDS)
        # Debug click amplitude: quiet enough to sit under the music.
        self.click_sound: float = 0.15 * np.sin(
            2. * np.pi * np.arange(self.buffer_size) / self.buffer_size
            * self.sample_rate / 3000.)

        # Rhythm front-end. Built once and reset in place: _reset_state() runs
        # every 15 minutes and on every sound stop, and rebuilding would reload
        # eight pickled LSTMs mid-show.
        self._rhythm: MadmomRhythm = MadmomRhythm(self.sample_rate)
        self._drift: DriftWatchdog = DriftWatchdog(self.buffer_size / self.sample_rate,
                                                   clock=self._clock)
        # Shed state is loop-scoped, like the watchdog that drives it: a song
        # boundary says nothing about whether the loop is keeping up.
        self._section_detection_enabled: bool = True
        # Rhythm telemetry is loop-scoped too. Scheduling it per song meant that
        # on a quiet input — where _reset_state() fires every 0.3 s — the
        # heartbeat's deadline was pushed forward forever and it never printed,
        # which is exactly when a heartbeat is worth having.
        self._rhythm_log_at: datetime.datetime = self._clock.now() + _RHYTHM_LOG_INTERVAL
        self._beats_since_log: int = 0
        self._onsets_since_log: int = 0
        self._last_beat_activation: float = 0.0
        self._reset_state()

    def _reset_state(self) -> None:
        # Spectral front-end (aubio): the FFT and the 40-band mel filterbank
        # every trained model depends on. Rhythm is madmom's — see _rhythm.
        self.mel: MelFilterbank = MelFilterbank(self.sample_rate, self.buffer_size)

        # tracking state
        self._rhythm.reset()
        # The drift watchdog is deliberately NOT reset here. It measures the
        # LOOP's health, which has nothing to do with song boundaries, and
        # _reset_state() fires every time the input goes quiet for 0.3 s —
        # between tracks, during a stop, all night on a silent input. Resetting
        # it there emptied its window continuously, so it re-entered its first
        # shed level every second or so and logged a degradation each time.
        # Found by running the tool against a silent live input.
        # Beat instants in madmom's own stream time, which BPM is derived from.
        # Stream time rather than clock time on purpose: if the input ever drops
        # audio, stream time still measures the music the detector actually
        # heard, so the tempo estimate stays right about the track.
        self._beat_stream_times: deque = deque(maxlen=_BPM_BEAT_WINDOW)
        # A reset empties the onset window, so it starts a refill exactly as a
        # shed-restore does — one rule for both paths: an empty window is
        # UNMEASURED until a window's worth of audio has filled it.
        #
        # The bug this replaces: `_onset_epoch_seen` was re-seeded from the
        # counter `self._rhythm.reset()` had just bumped, which swallowed the
        # edge the density getter watches for, WITHOUT starting a refill. Every
        # song start and every 15-minute reset then published a confident 0.0
        # for a window with nothing in it — on the path that runs constantly on
        # a quiet input. Setting both together, here, is why that cannot recur.
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
        if not self.is_playing:
            return 0
        return self._fold_bpm(self._measured_bpm())

    def _measured_bpm(self) -> float:
        """Tempo from the median recent inter-beat interval; 0.0 until measurable.

        The median rather than madmom's own `tempo` attribute, which is one
        interval wide and jumps on any single mistracked beat. Zero during
        warmup is deliberate — the DROP branch gates on a tempo floor, so an
        unmeasured tempo reads as "not fast", never as a guess.
        """
        if len(self._beat_stream_times) < 3:
            return 0.0
        interval = float(np.median(np.diff(np.array(self._beat_stream_times))))
        return 60.0 / interval if interval > 0 else 0.0

    @staticmethod
    def _fold_bpm(bpm: float) -> float:
        """Fold BPM into [85, 170) by octave halving/doubling; 0.0 if not finite or ≤ 0.

        A tempo and its octave are the same tempo musically, and the DROP
        branch's floor assumes one band, so the fold keeps a half-tempo lock
        from reading as a low-energy passage. The cost is that genuinely fast
        genres report at half tempo (see the root CLAUDE.md's known issues).
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
        """Onsets per second over the last 1.5 seconds, or DENSITY_UNKNOWN.

        Unknown while the onset detector is shed, and for one full window after
        it is restored: an empty rolling window would otherwise report a low
        rate rather than a missing one, which is the same lie a second later.
        """
        now = self._clock.now()
        # Edge-detected here rather than at the call site that sheds, so the
        # answer is right no matter who flipped the switch.
        enabled = self._rhythm.onsets_enabled
        if self._rhythm.onset_epoch != self._onset_epoch_seen:
            self._onset_epoch_seen = self._rhythm.onset_epoch
            self._onset_times.clear()
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

        # Backpressure first: what gets shed is decided before the work is done,
        # not after it has already been paid for.
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
        await self._track_beat(rhythm.beats, now)
        is_note = await self._track_note(rhythm.onsets, now)
        self._log_rhythm_state(now)

        if self.get_song_current_duration() > datetime.timedelta(minutes=15):
            self._reset_state()

        if (self._section_detection_enabled
                and self.yamnet_change_detector.detect_change(
                    audio_signal, self.get_song_current_duration())):
            await self.handler.on_section_change()

        if is_note and self.note_clicks:
            # Audible click for debug playback monitoring (returned buffer only —
            # feature extraction above already ran on the clean signal, and the
            # rhythm adapter copies what it is given).
            audio_signal += self.click_sound

        await self.handler.on_cycle()
        return audio_signal

    def _set_section_detection_enabled(self, enabled: bool) -> None:
        """Shed or restore YAMNet section detection (the drift watchdog's first
        lever). Restoring clears the detector's buffers: while shed it was fed
        no audio, so everything it holds describes music from before the gap.
        """
        if enabled == self._section_detection_enabled:
            return
        if enabled:
            self.yamnet_change_detector.reset()
        logging.warning('[yamnet] section detection %s',
                        'restored' if enabled else 'SHED — no section changes will fire')
        self._section_detection_enabled = enabled

    def _log_rhythm_state(self, now: datetime.datetime) -> None:
        """Periodic proof that rhythm is flowing, and where it stands.

        Rate-bounded on purpose: a per-buffer line is 172 lines a second and a
        per-beat line duplicates what the engine already prints. What this adds
        is the front-end's own view — how many events actually came out of
        madmom, what the beat network thought of the last one, and whether the
        loop is keeping up with the input.
        """
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
        """What the backpressure watchdog currently sees — for logs and soaks."""
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
            # An unmeasured density must not enter the trend deque: the trend is
            # a ratio of two halves of it, so a sentinel would come back out as
            # a number.
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

    async def _track_note(self, onsets: list, now: datetime.datetime) -> bool:
        """The debug beep trigger. Onsets replace aubio's note detector, whose
        only consumer this ever was; the refractory keeps the click density the
        same as before rather than buzzing through a busy bar."""
        is_note = bool(onsets) and now - self.last_note_detected > _NOTE_REFRACTORY
        if is_note:
            await self.handler.on_note()
            self.last_note_detected = now
        return is_note

    def _compute_mel_energies(self, audio_signal: np.ndarray) -> np.ndarray:
        energies_out = self.mel(audio_signal)

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
        # An unmeasured tempo is not a tempo change — and it is the denominator,
        # so treating it as one would divide by zero during every warmup.
        if not self.is_playing or current_bpm <= 0:
            return False
        # 5% change in bpm constitutes a change in bpm, defined arbitrarily
        return (abs(current_bpm - self.last_bpm) / current_bpm) > 0.05
