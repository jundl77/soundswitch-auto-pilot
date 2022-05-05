import numpy as np
import aubio
import datetime
import logging
from typing import List, Optional
from lib.analyser.music_analyser_handler import MusicAnalyserHandler


class MusicAnalyser:
    def __init__(self,
                 sample_rate: int,
                 buffer_size: int,
                 handler: MusicAnalyserHandler):
        self.sample_rate: int = sample_rate
        self.buffer_size: int = buffer_size
        self.handler: MusicAnalyserHandler = handler

        # constants
        self.tolerance: float = 0.8
        self.win_s: int = 1024  # fft size
        self.hop_s: int = self.buffer_size  # hop size
        self.mfcc_filters = 40  # must be 40 for mfcc
        self.mfcc_coeffs = 13
        self.click_sound: float = 0.7 * np.sin(2. * np.pi * np.arange(self.hop_s) / self.hop_s * self.sample_rate / 3000.)

        self._reset_state()

    def _reset_state(self) -> None:
        # audio analysers
        self.pitch_o: aubio.pitch = aubio.pitch("default", self.win_s, self.hop_s, self.sample_rate)
        self.pitch_o.set_unit("midi")
        self.pitch_o.set_tolerance(self.tolerance)
        self.tempo_o: aubio.tempo = aubio.tempo("default", self.win_s, self.hop_s, self.sample_rate)
        self.onset_o: aubio.onset = aubio.onset("default", self.win_s, self.hop_s, self.sample_rate)
        self.pvoc_o: aubio.pvoc = aubio.pvoc(self.win_s, self.hop_s)
        self.mfcc_o: aubio.mfcc = aubio.mfcc(self.win_s, self.mfcc_filters, self.mfcc_coeffs, self.sample_rate)
        self.energy_filter = aubio.filterbank(40, self.win_s)
        self.energy_filter.set_mel_coeffs_slaney(self.sample_rate)

        # tracking state
        self.is_playing: bool = False
        self.song_start_time: datetime.datetime = datetime.datetime.now()
        self.song_current_time: datetime.datetime = datetime.datetime.now()
        self.silence_period_start: datetime.datetime = datetime.datetime.now()
        self.last_mfcc_sample_time: datetime.datetime = datetime.datetime.now()
        self.mfccs = np.zeros([self.mfcc_coeffs,])
        self.energies = np.zeros((40,))

    def get_start_of_song(self) -> Optional[datetime.datetime]:
        if self.is_playing:
            return self.song_start_time
        else:
            return None

    def get_song_duration(self) -> datetime.timedelta:
        if self.is_playing:
            return self.song_current_time - self.song_start_time
        else:
            return datetime.timedelta(seconds=0)

    def get_bpm(self) -> float:
        if self.is_playing:
            return self.tempo_o.get_bpm()
        else:
            return 0

    def analyse(self, audio_signal: np.ndarray) -> np.ndarray:
        is_onset: bool = self._track_onset(audio_signal)
        is_beat: bool = self._track_beat(audio_signal)

        pitch = self.pitch_o(audio_signal)[0]
        confidence = self.pitch_o.get_confidence()

        if self.get_song_duration() > datetime.timedelta(minutes=15):
            self._reset_state()

        if is_beat:
            audio_signal += self.click_sound
        if is_onset:
            return audio_signal
        return np.zeros((1,))

    def _track_onset(self, audio_signal: np.ndarray) -> bool:
        is_onset: bool = self.onset_o(audio_signal)[0] > 0
        if is_onset:
            mfcc = self._compute_mfcc(audio_signal)
            self._track_song_duration(mfcc)
            self.handler.on_onset()
        return is_onset

    def _track_beat(self, audio_signal: np.ndarray) -> bool:
        is_beat: bool = self.tempo_o(audio_signal)[0] > 0
        if is_beat:
            this_beat: float = self.tempo_o.get_last_s()
            self.handler.on_beat(this_beat)

        return is_beat

    def _compute_mfcc(self, audio_signal: np.ndarray) -> np.ndarray:
        spec = self.pvoc_o(audio_signal)
        new_energies = self.energy_filter(spec)
        mfcc_out = self.mfcc_o(spec)

        self.mfccs = np.vstack((self.mfccs, mfcc_out))
        self.energies = np.vstack([self.energies, new_energies])

        return mfcc_out

    def _track_song_duration(self, mfcc: np.ndarray) -> None:
        is_silence_now: bool = len([n for n in mfcc[1:] if -0.001 < n < 0.001]) == len(mfcc) - 1
        now = datetime.datetime.now()

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
