import numpy as np
import aubio
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
        self.tolerance: float = 0.8
        self.win_s: int = 1024  # fft size
        self.hop_s: int = self.buffer_size  # hop size

        self.pitch_o: aubio.pitch = aubio.pitch("default", self.win_s, self.hop_s, self.sample_rate)
        self.pitch_o.set_unit("midi")
        self.pitch_o.set_tolerance(self.tolerance)

        self.tempo_o: aubio.tempo = aubio.tempo("default", self.win_s, self.hop_s, self.sample_rate)
        self.onset_o: aubio.onset = aubio.onset("default", self.win_s, self.hop_s, self.sample_rate)

        self.beats: List[float] = []

        self.click_sound: float = 0.7 * np.sin(2. * np.pi * np.arange(self.hop_s) / self.hop_s * self.sample_rate / 3000.)

    def analyse(self, audio_signal: np.ndarray) -> np.ndarray:
        is_onset = self.onset_o(audio_signal)
        if is_onset:
            self.handler.on_onset()

        is_beat = self.tempo_o(audio_signal)
        if is_beat:
            this_beat: float = self.tempo_o.get_last_s()
            self.beats.append(this_beat)
            self.handler.on_beat(this_beat, self._beats_to_bpm(self.beats))

        pitch = self.pitch_o(audio_signal)[0]
        confidence = self.pitch_o.get_confidence()

        if is_beat:
            audio_signal += self.click_sound
        return audio_signal

    def _beats_to_bpm(self, beats: List[float]) -> float:
        # if enough beats are found, convert to periods then to bpm
        if len(beats) > 1:
            if len(beats) < 4:
                print("few beats found in audio")
            bpms = 60. / np.diff(beats)
            return np.median(bpms)
        else:
            print("not enough beats found in audio")
            return 0
