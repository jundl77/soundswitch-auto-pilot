import numpy as np
from matplotlib import axes
from lib.visualizer.visualizer_types import VisualizerData

N_BUCKETS = 100  # num of buckets on x


class CurveVisualizer:
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
        self.vmax, self.vmin = 1.0, 0.0

        self.x_axis = np.arange(self.n_features)
        zeros = np.zeros((N_BUCKETS, self.n_features))
        self.image = self.ax.imshow(zeros, aspect="auto")
        self.image.set_clim(self.vmin, self.vmax)
        self.ax.title.set_text(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.label_outer()

    def render(self, visualizer_data: VisualizerData):
        block = getattr(visualizer_data, self.data_attr)
        block_mean = np.mean(block, axis=0)[0:self.n_features]

        data = block
        # if self.vmax < np.max(block_mean):
        #     self.vmax = np.max(block_mean)
        # self.image.set_clim(self.vmin, self.vmax)

        # block_mean *= ((N_BUCKETS - 1)/block_mean.max())
        # data = np.zeros((N_BUCKETS, self.n_features))
        # for i in range(self.n_features):
        #     j = N_BUCKETS - 1 - int(block_mean[i])
        #     data[j][i] = 1

        self.image.set_data(data)

