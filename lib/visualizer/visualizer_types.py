import numpy as np


class VisualizerData:
    def __init__(self,
                 spectogram: np.ndarray,
                 energies: np.ndarray,
                 freq_curve: np.ndarray,
                 energy_curve: np.ndarray,
                 scroll_curve: np.ndarray,
                 chroma: np.array,
                 pitch_hz: np.ndarray,
                 is_onset: np.ndarray,
                 is_beat: np.ndarray,
                 is_note: np.ndarray):
        self.spectogram: np.ndarray = spectogram
        self.energies: np.ndarray = energies
        self.freq_curve: np.ndarray = freq_curve
        self.energy_curve: np.ndarray = energy_curve
        self.scroll_curve: np.ndarray = scroll_curve
        self.chroma: np.ndarray = chroma
        self.pitch_hz: np.ndarray = pitch_hz
        self.is_onset: np.ndarray = is_onset
        self.is_beat: np.ndarray = is_beat
        self.is_note: np.ndarray = is_note
