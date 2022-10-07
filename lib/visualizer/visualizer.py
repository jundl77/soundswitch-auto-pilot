import time
import logging
from threading import Thread
from multiprocessing import Process
from multiprocessing.connection import Client, Listener
import numpy as np
import matplotlib.pyplot as plt


TCP_CONNECTION_PORT = 5599
N_SAMPLES = 16
plt.style.use('dark_background')


class VisualizerData:
    def __init__(self,
                 spectogram: np.ndarray,
                 mfccs: np.ndarray,
                 energies: np.ndarray,
                 pitch_hz: np.ndarray,
                 is_onset: np.ndarray,
                 is_beat: np.ndarray,
                 is_note: np.ndarray):
        self.spectogram: np.ndarray = spectogram
        self.mfccs: np.ndarray = mfccs
        self.energies: np.ndarray = energies
        self.pitch_hz: np.ndarray = pitch_hz
        self.is_onset: np.ndarray = is_onset
        self.is_beat: np.ndarray = is_beat
        self.is_note: np.ndarray = is_note


class VisualizerUpdater:
    def __init__(self):
        self.sending_thread = Thread(target=self._run_sending_thread)
        self.is_running = False
        self.data_buffer: VisualizerData = None

    def connect(self):
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
        self.data_buffer.mfccs = np.vstack((self.data_buffer.mfccs, data.mfccs))
        self.data_buffer.energies = np.vstack((self.data_buffer.energies, data.energies))
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
                sending_buffer = VisualizerData(self.data_buffer.spectogram[:N_SAMPLES],
                                                self.data_buffer.mfccs[:N_SAMPLES],
                                                self.data_buffer.energies[:N_SAMPLES],
                                                self.data_buffer.pitch_hz[:N_SAMPLES],
                                                self.data_buffer.is_onset[:N_SAMPLES],
                                                self.data_buffer.is_beat[:N_SAMPLES],
                                                self.data_buffer.is_note[:N_SAMPLES])
                client.send(sending_buffer)
                self.data_buffer.spectogram = self.data_buffer.spectogram[N_SAMPLES:]
                self.data_buffer.mfccs = self.data_buffer.mfccs[N_SAMPLES:]
                self.data_buffer.energies = self.data_buffer.energies[N_SAMPLES:]
                self.data_buffer.pitch_hz = self.data_buffer.pitch_hz[N_SAMPLES:]
                self.data_buffer.is_onset = self.data_buffer.is_onset[N_SAMPLES:]
                self.data_buffer.is_beat = self.data_buffer.is_beat[N_SAMPLES:]
                self.data_buffer.is_note = self.data_buffer.is_note[N_SAMPLES:]
            time.sleep(0.01)
        plt.close()


class Visualizer:
    def __init__(self):
        self.ui_process = Process(target=self._run_ui_process)
        self.is_running: bool = False
        self.render_start_ts: float = 0
        self.frame_count: int = 0

        self.n_fft = N_SAMPLES
        self.n_plot_tf = 120  # num of buckets on x
        self.n_freqs = self.n_fft // 2 + 1
        self.f_max_idx = 100  # 1 < f_max_idx < n_freqs
        self.window = np.hamming(self.n_fft)
        self.amp = np.zeros((self.n_plot_tf, self.f_max_idx))
        self.fps = 1.0
        self.fig, self.ax = plt.subplots()
        self.image = self.ax.imshow(self.amp.T, aspect="auto")
        self.ax.set_xlabel(f"Time frame")
        self.ax.set_ylabel(f"Frequency")
        self.fig.colorbar(self.image)
        self.vmax, self.vmin = 1.0, 0.0

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
            msg: VisualizerData = connection.recv()
            self._render(msg)
        plt.close()

    def _render(self, visualizer_data: VisualizerData):
        self.render_start_ts = time.time()
        block = visualizer_data.spectogram
        if block.shape[0] != self.n_fft:
            return

        self.amp[-1] = np.mean(block, axis=0)[0:self.f_max_idx]
        if self.vmax < np.max(self.amp[-1]):
            self.vmax = np.max(self.amp[-1])
        self.image.set_clim(self.vmin, self.vmax)
        self.image.set_data(self.amp.T[::-1])

        plt.title(f"fps: {self.fps:0.1f} Hz")
        self.fig.canvas.draw()
        plt.pause(0.001)

        self.amp[0:-1] = self.amp[1::]

        now = time.time()
        time_diff = now - self.render_start_ts
        self.fps = 1.0 / (time_diff + 1e-16)
