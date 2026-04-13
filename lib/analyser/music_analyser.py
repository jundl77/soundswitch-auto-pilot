import datetime
import logging
import aubio
import numpy as np
from typing import Optional
from collections import deque
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler
from lib.analyser.yamnet_change_detector import YamnetChangeDetector
from lib.visualizer.visualizer import VisualizerUpdater, VisualizerData

_ONSET_DENSITY_WINDOW_SEC = 1.5  # rolling window for onset density calculation


class MusicAnalyser:
    def __init__(self,
                 sample_rate: int,
                 buffer_size: int,
                 handler: IMusicAnalyserHandler,
                 visualizer_updater: VisualizerUpdater):
        self.sample_rate: int = sample_rate
        self.buffer_size: int = buffer_size
        self.handler: IMusicAnalyserHandler = handler
        self.visualizer_updater: VisualizerUpdater = visualizer_updater
        self.yamnet_change_detector: YamnetChangeDetector = YamnetChangeDetector(self.sample_rate, self.buffer_size)

        # constants
        self.win_s: int = self.buffer_size * 4  # fft size
        self.win_s_small: int = self.buffer_size * 2  # fft size
        self.win_s_large: int = self.buffer_size * 8  # fft size
        self.hop_s: int = self.buffer_size  # hop size
        self.mfcc_filters: int = 40  # must be 40 for mfcc
        self.mfcc_coeffs: int = 13
        self.click_sound: float = 0.7 * np.sin(2. * np.pi * np.arange(self.hop_s) / self.hop_s * self.sample_rate / 3000.)

        self._reset_state()

    def _reset_state(self) -> None:
        # audio analysers
        self.pitch_o: aubio.pitch = aubio.pitch("default", self.win_s_large, self.hop_s, self.sample_rate)
        self.pitch_o.set_unit("hertz")
        self.tempo_o: aubio.tempo = aubio.tempo("default", self.win_s_small, self.hop_s, self.sample_rate)
        self.onset_o: aubio.onset = aubio.onset("default", self.win_s_small, self.hop_s, self.sample_rate)
        self.notes_o = aubio.notes("default", self.win_s_small, self.hop_s, self.sample_rate)
        self.pvoc_o: aubio.pvoc = aubio.pvoc(self.win_s, self.hop_s)
        self.mfcc_o: aubio.mfcc = aubio.mfcc(self.win_s, self.mfcc_filters, self.mfcc_coeffs, self.sample_rate)
        self.energy_filter = aubio.filterbank(self.mfcc_filters, self.win_s)
        self.energy_filter.set_mel_coeffs_slaney(self.sample_rate)


        # tracking state
        self.yamnet_change_detector.reset()
        self.is_playing: bool = False
        self.song_start_time: datetime.datetime = datetime.datetime.now()
        self.song_current_time: datetime.datetime = datetime.datetime.now()
        self.silence_period_start: datetime.datetime = datetime.datetime.now()
        self.last_mfcc_sample_time: datetime.datetime = datetime.datetime.now()
        self.mfccs = np.zeros([self.mfcc_coeffs,])
        self.energies = np.zeros((40,))
        self.last_bpm: float = 0.0
        self.beat_count: int = 0
        self.time_to_last_beat_sec: float = 0
        self.last_beat_detected: datetime.datetime = datetime.datetime.now()
        self.last_note_detected: datetime.datetime = datetime.datetime.now()
        # rolling window of onset timestamps for onset-density calculation
        self._onset_times: deque = deque(maxlen=500)
        # per-beat density samples for trend detection (maxlen=12 ≈ ~6s at 120 BPM)
        self._density_samples: deque = deque(maxlen=12)
        # Rolling mel-energy frames and RMS values for sub-bass / energy features.
        # maxlen≈26 ≈ 150 ms of buffers at 5.8 ms/buffer — long enough for a stable mean.
        self._mel_energies_window: deque = deque(maxlen=26)
        self._rms_window: deque = deque(maxlen=26)

    def start(self):
        self.yamnet_change_detector.start()

    def get_start_of_song(self) -> Optional[datetime.datetime]:
        if self.is_playing:
            return self.song_start_time
        else:
            return None

    def get_song_current_duration(self) -> datetime.timedelta:
        if self.is_playing:
            return self.song_current_time - self.song_start_time
        else:
            return datetime.timedelta(seconds=0)

    def get_beat_position(self) -> float:
        if self.is_playing and self.time_to_last_beat_sec > 0:
            time_to_current_beat_sec = (datetime.datetime.now() - self.last_beat_detected).total_seconds()
            beat_percent_elapsed = time_to_current_beat_sec / self.time_to_last_beat_sec
            return self.beat_count + abs(beat_percent_elapsed)
        else:
            return 0

    def get_bpm(self) -> float:
        if self.is_playing:
            return self.tempo_o.get_bpm()
        else:
            return 0

    def is_song_playing(self) -> bool:
        return self.is_playing

    def get_onset_density(self) -> float:
        """Onsets per second over the last 1.5 seconds (rolling window)."""
        now = datetime.datetime.now()
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
        return (datetime.datetime.now() - self.last_beat_detected).total_seconds()

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

    async def analyse(self, audio_signal: np.ndarray) -> np.ndarray:
        now = datetime.datetime.now()

        pitch_hz = self.pitch_o(audio_signal)[0]
        pitch_confidence = self.pitch_o.get_confidence()
        rms = float(np.sqrt(np.mean(audio_signal ** 2)))
        self._rms_window.append(rms)
        spec, mfccs, energies = self._compute_mfcc(audio_signal)
        self._track_song_duration(energies, now)

        is_onset: bool = await self._track_onset(audio_signal)
        is_beat: bool = await self._track_beat(audio_signal, now)
        is_note, note = await self._track_note(audio_signal, now)

        if self.get_song_current_duration() > datetime.timedelta(minutes=15):
            self._reset_state()

        if self.yamnet_change_detector.detect_change(audio_signal, self.get_song_current_duration()):
            await self.handler.on_section_change()

        if is_beat:
            #audio_signal += self.click_sound
            pass

        if is_note:
            audio_signal += self.click_sound
            pass

        await self.handler.on_cycle()
        return audio_signal

    async def _track_onset(self, audio_signal: np.ndarray) -> bool:
        is_onset: bool = self.onset_o(audio_signal)[0] > 0
        if is_onset:
            self._onset_times.append(datetime.datetime.now())
            await self.handler.on_onset()
        return is_onset

    async def _track_beat(self, audio_signal: np.ndarray, now: datetime.datetime) -> bool:
        is_beat: bool = self.tempo_o(audio_signal)[0] > 0
        if is_beat:
            this_bpm: float = self.get_bpm()
            bpm_changed: bool = self._has_bpm_changed(this_bpm)
            self.beat_count += 1
            self._density_samples.append(self.get_onset_density())
            await self.handler.on_beat(self.beat_count, this_bpm, bpm_changed)
            self.last_bpm = self.get_bpm()
            self.time_to_last_beat_sec = (now - self.last_beat_detected).total_seconds()
            self.last_beat_detected = now
        return is_beat

    async def _track_note(self, audio_signal: np.ndarray, now: datetime.datetime) -> tuple[bool, np.ndarray]:
        note = self.notes_o(audio_signal)
        is_note = note[0] > 0 and now - self.last_note_detected > datetime.timedelta(milliseconds=75)
        if is_note:
            await self.handler.on_note()
            self.last_note_detected = now
        return is_note, note

    def _compute_mfcc(self, audio_signal: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        spec = self.pvoc_o(audio_signal)
        mfcc_out = self.mfcc_o(spec)
        energies_out = self.energy_filter(spec)

        self.mfccs = np.vstack((self.mfccs, mfcc_out))
        self.energies = np.vstack([self.energies, energies_out])
        self._mel_energies_window.append(energies_out.copy())

        return spec, mfcc_out, energies_out

    def _track_song_duration(self, energies: np.ndarray, now: datetime.datetime) -> None:
        is_silence_now: bool = len([n for n in energies if -0.0001 < n < 0.0001]) == len(energies)

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

    def _midi_to_hz(self, notes: np.ndarray):
        """ taken from librosa, Get the frequency (Hz) of MIDI note(s) """
        return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))
