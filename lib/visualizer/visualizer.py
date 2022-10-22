import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from multiprocessing import Process
from multiprocessing.connection import Client, Listener
from lib.visualizer.visualizer_types import VisualizerData
from lib.visualizer.spectogram import Spectogram
from lib.visualizer.curve_visualizer import CurveVisualizer
from lib.visualizer.self_similarity_matrix import SelfSimilarityMatrix

TCP_CONNECTION_PORT = 5599
N_SAMPLES = 16
plt.style.use('dark_background')


class VisualizerUpdater:
    def __init__(self):
        self.sending_thread = Thread(target=self._run_sending_thread)
        self.is_running = False
        self.data_buffer: VisualizerData = None

    def connect(self):
        time.sleep(1)
        self.is_running = True
        self.sending_thread.start()

    def stop(self):
        logging.info(f'[visualizer_updater] stopping sending thread')
        if self.is_running:
            self.is_running = False
            self.sending_thread.join()

    def update_data(self, data: VisualizerData):
        if self.data_buffer is None:
            self.data_buffer = data
            return

        self.data_buffer.spectogram = np.vstack((self.data_buffer.spectogram, data.spectogram))
        self.data_buffer.energies = np.vstack((self.data_buffer.energies, data.energies))
        self.data_buffer.freq_curve = np.vstack((self.data_buffer.freq_curve, data.freq_curve))
        self.data_buffer.energy_curve = np.vstack((self.data_buffer.energy_curve, data.energy_curve))
        self.data_buffer.scroll_curve = np.vstack((self.data_buffer.scroll_curve, data.scroll_curve))
        self.data_buffer.chroma = np.vstack((self.data_buffer.chroma, data.chroma))
        self.data_buffer.pitch_hz = np.vstack((self.data_buffer.pitch_hz, data.pitch_hz))
        self.data_buffer.is_onset = np.vstack((self.data_buffer.is_onset, data.is_onset))
        self.data_buffer.is_beat = np.vstack((self.data_buffer.is_beat, data.is_beat))
        self.data_buffer.is_note = np.vstack((self.data_buffer.is_note, data.is_note))

    def _run_sending_thread(self):
        logging.info(f'[visualizer_updater] started sending thread')

        logging.info(f'[visualizer_updater] attempting to connect..')
        client: Client = Client(('localhost', TCP_CONNECTION_PORT))
        logging.info(f'[visualizer_updater] connected to visualizer successfully')

        while self.is_running:
            if self.data_buffer is not None and self.data_buffer.spectogram.shape[0] > N_SAMPLES:
                client.send(self.data_buffer)
                self.data_buffer.spectogram = self.data_buffer.spectogram[-1:0]
                self.data_buffer.energies = self.data_buffer.energies[-1:0]
                self.data_buffer.freq_curve = self.data_buffer.freq_curve[-1:0]
                self.data_buffer.energy_curve = self.data_buffer.energy_curve[-1:0]
                self.data_buffer.scroll_curve = self.data_buffer.scroll_curve[-1:0]
                self.data_buffer.chroma = self.data_buffer.chroma[-1:0]
                self.data_buffer.pitch_hz = self.data_buffer.pitch_hz[-1:0]
                self.data_buffer.is_onset = self.data_buffer.is_onset[-1:0]
                self.data_buffer.is_beat = self.data_buffer.is_beat[-1:0]
                self.data_buffer.is_note = self.data_buffer.is_note[-1:0]
            time.sleep(0.01)
        plt.close()


class Visualizer:
    def __init__(self,
                 show_freq: bool,
                 show_freq_gradient: bool,
                 show_energy: bool,
                 show_freq_curve: bool,
                 show_ssm: bool):
        self.show_freq: bool = show_freq
        self.show_freq_gradient: bool = show_freq_gradient
        self.show_energy: bool = show_energy
        self.show_freq_curve: bool = show_freq_curve
        self.show_ssm: bool = show_ssm

        self.ui_process = Process(target=self._run_ui_process)
        self.is_running: bool = False
        self.fps = 1.0

        self.spec_fig = None
        self.curve_fig = None
        self.ssm_fig = None

        if self.show_freq:
            self.spec_fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            self.frequency_spec = Spectogram(ax=ax1, data_attr='spectogram', n_features=200,
                                             title='Time vs Frequency', x_label="Time", y_label="Frequency")
            if self.show_freq_gradient:
                self.freq_gradient_spec = Spectogram(ax=ax2, data_attr='freq_grad', n_features=200,
                                                     title='Time vs Freq. Gradient', x_label="Time", y_label="Freq. Gradient")
            if self.show_energy:
                self.energy_spec = Spectogram(ax=ax3, data_attr='energies', n_features=40,
                                              title='Time vs Energies', x_label="Time", y_label="Energies")
        if self.show_freq_curve:
            self.curve_fig, (curve_ax1) = plt.subplots(1, 1)
            self.freq_curve = CurveVisualizer(ax=curve_ax1, data_attr='freq_curve', n_features=4,
                                              title='Frequency vs Power', x_label="Frequency", y_label="Power")
        if self.show_ssm:
            self.ssm_fig, ssm_ax = plt.subplots(1, 1)
            self.ssm = SelfSimilarityMatrix(ssm_ax, 'chroma', 12, 'title', 'xlabel', 'ylabel')

    def show(self):
        logging.info(f'[ui] starting ui thread')
        self.is_running = True
        self.ui_process.start()

    def stop(self):
        logging.info(f'[ui] stopping ui thread')
        if self.is_running:
            self.is_running = False
            self.ui_process.terminate()
            self.ui_process.join()
            self.ui_process.close()

    def _run_ui_process(self):
        listener: Listener = Listener(('localhost', TCP_CONNECTION_PORT))
        logging.info(f'[visualizer] listening for new connections on port {TCP_CONNECTION_PORT}')

        connection = listener.accept()
        logging.info(f'[visualizer] new connection accepted from {listener.last_accepted}')

        plt.plot()
        plt.show(block=False)

        while self.is_running:
            render_start_ts = time.time()

            msg: VisualizerData = connection.recv()
            self._render(msg)

            time_diff = time.time() - render_start_ts
            self.fps = 1.0 / (time_diff + 1e-16)
        plt.close()

    def _render(self, visualizer_data: VisualizerData):
        if self.show_freq:
            self.frequency_spec.render(visualizer_data)
        if self.show_freq_gradient:
            self.freq_gradient_spec.render(visualizer_data)
        if self.show_energy:
            self.energy_spec.render(visualizer_data)
        if self.show_freq_curve:
            self.freq_curve.render(visualizer_data)
        if self.show_ssm:
            self.ssm.render(visualizer_data)

        if self.spec_fig is not None:
            self.spec_fig.suptitle(f"fps: {self.fps:0.1f} Hz")
            self.spec_fig.canvas.draw()

        if self.curve_fig is not None:
            self.curve_fig.suptitle(f"fps: {self.fps:0.1f} Hz")
            self.curve_fig.canvas.draw()

        if self.ssm_fig is not None:
            self.ssm_fig.suptitle(f"fps: {self.fps:0.1f} Hz")
            self.ssm_fig.canvas.draw()

        plt.pause(0.001)
