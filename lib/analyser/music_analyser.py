import datetime
import logging
import aubio
import numpy as np
from typing import Optional
from scipy.ndimage.filters import gaussian_filter1d
from lib.analyser.music_analyser_handler import IMusicAnalyserHandler
from lib.analyser.exp_filter import ExpFilter
from lib.clients.spotify_client import SpotifyTrackAnalysis
from lib.visualizer.visualizer import VisualizerUpdater, VisualizerData
from lib.effects.rgb_visualizer import RgbVisualizer


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

        self.rgb_visualizer = RgbVisualizer(num_mel_bins=self.mfcc_filters, num_pixels=60)
        self.mel_gain = ExpFilter(np.tile(1e-1, self.mfcc_filters), alpha_decay=0.01, alpha_rise=0.99)
        self.mel_smoothing = ExpFilter(np.tile(1e-1, self.mfcc_filters), alpha_decay=0.5, alpha_rise=0.99)

        # tracking state
        self.is_playing: bool = False
        self.spotify_track_analysis: Optional[SpotifyTrackAnalysis] = None
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
            time_to_current_beat_sec = (datetime.datetime.now() - self.last_beat_detected).microseconds / 1000 / 1000
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

    def inject_spotify_track_analysis(self, track_analysis: Optional[SpotifyTrackAnalysis]):
        self.spotify_track_analysis = track_analysis
        if self.spotify_track_analysis:
            self.beat_count = track_analysis.current_beat_count
            self.song_start_time = datetime.datetime.now() - datetime.timedelta(milliseconds=track_analysis.progress_ms)
            logging.info(f'[analyser] applied spotify adjustments: beat_count={self.beat_count}, song_start={self.song_start_time}')

    async def analyse(self, audio_signal: np.ndarray) -> np.ndarray:
        now = datetime.datetime.now()

        pitch_hz = self.pitch_o(audio_signal)[0]
        pitch_confidence = self.pitch_o.get_confidence()
        spec, mfccs, energies = self._compute_mfcc(audio_signal)
        self._track_song_duration(energies, now)

        is_onset: bool = await self._track_onset(audio_signal)
        is_beat: bool = await self._track_beat(audio_signal, now)
        is_note, note = await self._track_note(audio_signal, now)

        if self.get_song_current_duration() > datetime.timedelta(minutes=15):
            self._reset_state()

        # todo: uncomment and fix again
        # rgb_spec, rgb_energy, rgb_scroll = self._compute_rgb_visualizations(energies)
        # intensity_val = np.min([1, (np.mean(rgb_spec[0]) + np.mean(rgb_spec[1]) + np.mean(rgb_spec[2])) / 3 / 50])
        # if self.is_playing:
        #     await self.handler.on_cycle(intensity_val)

        # rgb_spec, rgb_energy, rgb_scroll = np.zeros((4,)), np.zeros((4,)), np.zeros((4,))
        # if is_onset:
        #     rgb_spec, rgb_energy, rgb_scroll = self._compute_rgb_visualizations(energies)

        if is_beat:
            audio_signal += self.click_sound
            pass

        if is_note:
            #audio_signal += self.click_sound
            pass

        # todo: uncomment again
        # if self.visualizer_updater is not None:
        #     data = VisualizerData(spec.norm, energies, rgb_spec, rgb_energy, rgb_scroll, mfccs, np.array([pitch_hz]), np.array([is_onset]), np.array([is_beat]), np.array([is_note]))
        #     self.visualizer_updater.update_data(data)

        return audio_signal

    async def _track_onset(self, audio_signal: np.ndarray) -> bool:
        is_onset: bool = self.onset_o(audio_signal)[0] > 0
        if is_onset:
            await self.handler.on_onset()
        return is_onset

    async def _track_beat(self, audio_signal: np.ndarray, now: datetime.datetime) -> bool:
        is_beat: bool = self.tempo_o(audio_signal)[0] > 0
        if is_beat:
            this_bpm: float = self.get_bpm()
            bpm_changed: bool = self._has_bpm_changed(this_bpm)
            self.beat_count += 1
            await self.handler.on_beat(self.beat_count, this_bpm, bpm_changed)
            self.last_bpm = self.get_bpm()
            self.time_to_last_beat_sec = (now - self.last_beat_detected).microseconds / 1000 / 1000
            self.last_beat_detected = now
        return is_beat

    async def _track_note(self, audio_signal: np.ndarray, now: datetime.datetime) -> tuple[bool, np.ndarray]:
        note = self.notes_o(audio_signal)
        is_note = note[0] > 0 and now - self.last_note_detected > datetime.timedelta(milliseconds=75)
        if is_note:
            logging.debug(f'[analyser] note {note}, frequency={self._midi_to_hz(note[0])}hz')
            self.last_note_detected = now
        return is_note, note

    def _compute_mfcc(self, audio_signal: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        spec = self.pvoc_o(audio_signal)
        mfcc_out = self.mfcc_o(spec)
        energies_out = self.energy_filter(spec)

        self.mfccs = np.vstack((self.mfccs, mfcc_out))
        self.energies = np.vstack([self.energies, energies_out])

        return spec, mfcc_out, energies_out

    def _compute_rgb_visualizations(self, energies: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
        # Scale data to values more suitable for visualization
        mel = np.atleast_2d(energies).T * energies.T
        mel = np.sum(mel, axis=0)
        mel = mel**2.0

        # Gain normalization
        self.mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= self.mel_gain.value
        mel = self.mel_smoothing.update(mel)

        rgb_spec = self.rgb_visualizer.visualize_spectrum(mel)
        rgb_energy = self.rgb_visualizer.visualize_energy(mel)
        rgb_scroll = self.rgb_visualizer.visualize_scroll(mel)

        return rgb_spec, rgb_energy, rgb_scroll

    def _track_song_duration(self, energies: np.ndarray, now: datetime.datetime) -> None:
        is_silence_now: bool = len([n for n in energies if -0.0001 < n < 0.0001]) == len(energies)

        # if it is silent now, we do not update silence_period_start in order to track the duration of the silence
        if not is_silence_now:
            self.silence_period_start = now

        # if there was sound, and then we had no sound for 0.3 seconds, set state to is not playing
        if now - self.silence_period_start > datetime.timedelta(seconds=0.3):
            if self.is_playing:
                self.handler.on_sound_stop()
            self._reset_state()  # sets is_playing to False
        else:
            self.song_current_time = now

        # if there was no sound, and then we had sound for 0.3 seconds, set state to is playing
        if not self.is_playing and now - self.song_start_time > datetime.timedelta(seconds=0.3):
            self.handler.on_sound_start()
            self.is_playing = True

    def _has_bpm_changed(self, current_bpm: float) -> bool:
        if self.is_playing:
            # 5% change in bpm constitutes a change in bpm, defined arbitrarily
            return (abs(current_bpm - self.last_bpm) / current_bpm) > 0.05
        else:
            return False

    def _midi_to_hz(self, notes: np.ndarray):
        """ taken from librosa, Get the frequency (Hz) of MIDI note(s) """
        return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))
