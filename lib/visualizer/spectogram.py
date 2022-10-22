import numpy as np
from matplotlib import axes
from lib.visualizer.visualizer_types import VisualizerData

N_BUCKETS = 100  # num of buckets on x


class Spectogram:
    def __init__(self,
                 ax: axes.Axes,
                 data_attr: str,
                 n_features: int,
                 title: str,
                 x_label: str,
                 y_label: str):
        self.ax: axes.Axes = ax
        self.data_attr: str = data_attr
        self.n_features = n_features  # number of rows displayed from the data set, e.g. number of frequency bands displayed starting from 0
        self.amp = np.zeros((N_BUCKETS, self.n_features))
        self.vmax, self.vmin = 1.0, 0.0

        self.image = self.ax.imshow(self.amp.T, aspect="auto")
        self.ax.title.set_text(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.label_outer()

    def render(self, visualizer_data: VisualizerData):
        if self.data_attr == 'freq_grad':
            block_orig = getattr(visualizer_data, 'spectogram')
            block = np.gradient(block_orig)[0]
            #block = (block - np.min(block)) / np.ptp(block)
        else:
            block = getattr(visualizer_data, self.data_attr)
        is_beat: bool = np.max(visualizer_data.is_beat, axis=0)[0]
        is_note: bool = np.max(visualizer_data.is_note, axis=0)[0]

        self.amp[-1] = np.mean(block, axis=0)[0:self.n_features]
        if self.vmax < np.max(self.amp[-1]):
            self.vmax = np.max(self.amp[-1])

        if is_beat:
            self.amp[-1][self.n_features - 10] = self.vmax / 10

        if is_note:
            self.amp[-1][self.n_features - 20] = self.vmax / 10

        self.image.set_clim(self.vmin, self.vmax)
        self.image.set_data(self.amp.T[::-1])

        self.amp[0:-1] = self.amp[1::]

