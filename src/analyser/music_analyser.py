import numpy as np
import aubio
import datetime
from analyser.music_analyser_handler import MusicAnalyserHandler
from typing import List


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

        self._init_state()

    def _init_state(self):
        # audio analysers
        self.pitch_o: aubio.pitch = aubio.pitch("default", self.win_s, self.hop_s, self.sample_rate)
        self.pitch_o.set_unit("midi")
        self.pitch_o.set_tolerance(self.tolerance)
        self.tempo_o: aubio.tempo = aubio.tempo("default", self.win_s, self.hop_s, self.sample_rate)
        self.onset_o: aubio.onset = aubio.onset("default", self.win_s, self.hop_s, self.sample_rate)
        self.pvoc_o: aubio.pvoc = aubio.pvoc(self.win_s, self.hop_s)
        self.mfcc_o: aubio.mfcc = aubio.mfcc(self.win_s, self.mfcc_filters, self.mfcc_coeffs, self.sample_rate)

        # state
        self.is_playing: bool = False
        self.song_start_time: datetime.datetime = datetime.datetime.now()
        self.song_current_time: datetime.datetime = datetime.datetime.now()
        self.silence_period_start: datetime.datetime = datetime.datetime.now()

        self.bpm: float = 0
        self.beats: List[float] = []
        self.mfccs = np.zeros([self.mfcc_coeffs,])

    def get_start_of_song(self) -> datetime.datetime:
        return self.song_start_time

    def get_song_duration(self) -> datetime.timedelta:
        return self.song_current_time - self.song_start_time

    def get_bpm(self) -> float:
        return self.bpm

    def analyse(self, audio_signal: np.ndarray) -> np.ndarray:
        is_onset = self.onset_o(audio_signal)
        if is_onset:
            self.handler.on_onset()

        is_beat = self.tempo_o(audio_signal)
        if is_beat:
            this_beat: float = self.tempo_o.get_last_s()
            self.beats.append(this_beat)
            self.bpm = self._beats_to_bpm(self.beats)
            self.handler.on_beat(this_beat)

        spec = self.pvoc_o(audio_signal)
        mfcc_out = self.mfcc_o(spec)
        self.mfccs = np.vstack((self.mfccs, mfcc_out))
        self._track_song_duration(mfcc_out)

        pitch = self.pitch_o(audio_signal)[0]
        confidence = self.pitch_o.get_confidence()

        if is_beat:
            audio_signal += self.click_sound
        return audio_signal

    def _track_song_duration(self, mfcc) -> None:

        is_silence_now: bool = len([n for n in mfcc[1:] if -0.001 < n < 0.001]) == len(mfcc) - 1

        now = datetime.datetime.now()

        # if it is silent now, we do not update silence_period_start in order to track the duration of the silence
        if not is_silence_now:
            self.silence_period_start = now

        # if there was sound, and then we had no sound for 0.3 seconds, set state to is not playing
        if now - self.silence_period_start > datetime.timedelta(seconds=0.3):
            if self.is_playing:
                self.handler.on_sound_stop()
            self._init_state()  # sets is_playing to False
        else:
            self.song_current_time = now

        # if there was no sound, and then we had sound for 0.3 seconds, set state to is playing
        if not self.is_playing and now - self.song_start_time > datetime.timedelta(seconds=0.3):
            self.handler.on_sound_start()
            self.is_playing = True

    def _fbeats_to_bpm(self, beats: List[float]) -> float:
        # if enough beats are found, convert to periods then to bpm
        if len(beats) > 1:
            if len(beats) < 4:
                print("few beats found in audio")
            bpms = 60. / np.diff(beats)
            return np.median(bpms)
        else:
            print("not enough beats found in audio")
            return 0
