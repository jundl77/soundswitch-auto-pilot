import librosa
import librosa.display
import numpy as np
from matplotlib import axes, figure
from lib.visualizer.visualizer_types import VisualizerData

# BUFFER_SIZE = 128
# SAMPLE_RATE = 44100


class SelfSimilarityMatrix:
    def __init__(self,
                 ax: axes.Axes,
                 data_attr: str,
                 hop_s: int,
                 title: str,
                 x_label: str,
                 y_label: str):
        self.ax: axes.Axes = ax
        self.data_attr: str = data_attr
        self.hop_s: int = hop_s

        self.image = None
        self.ax.title.set_text(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.label_outer()

    def render(self, visualizer_data: VisualizerData):
        #chroma = librosa.feature.chroma_cqt(y=samples, sr=samplerate, hop_length=hop_s)
        block_orig = visualizer_data.spectogram
        block = np.gradient(block_orig)[0]
        #chroma = librosa.feature.chroma_cqt(y=block, sr=SAMPLE_RATE, hop_length=BUFFER_SIZE)
        freq_stack = librosa.feature.stack_memory(block, n_steps=10, delay=3)
        #recurrence = librosa.segment.recurrence_matrix(chroma_stack)
        recurrence = librosa.segment.recurrence_matrix(freq_stack, metric='cosine')
        lag_pad = librosa.segment.recurrence_to_lag(recurrence, pad=False)

        if self.image is None:
            self.image = self.ax.imshow(lag_pad.T, aspect="auto")
            self.image.set_clim(0.0, 1.0)
        else:
            self.image.set_data(lag_pad.T)

