import pyaudio
import numpy as np
import logging


class PyAudioClient:
    def __init__(self,
                 sample_rate: int,
                 buffer_size: int,
                 input_device_index: int = None,
                 output_device_index: int = None):
        self.sample_rate: int = sample_rate
        self.buffer_size: int = buffer_size
        self.input_device_index: int = input_device_index
        self.output_device_index: int = output_device_index
        self.py_audio: pyaudio.PyAudio = pyaudio.PyAudio()
        self.stream_in: pyaudio.Stream = None
        self.stream_out: pyaudio.Stream = None

    def list_devices(self):
        print('=== Pyaudio devices ===')
        for i in range(0, self.py_audio.get_device_count()):
            print(f'index: {i}, device: {self.py_audio.get_device_info_by_index(i)["name"]}')

    def support_output(self) -> bool:
        return self.stream_out is not None

    def start_streams(self, start_stream_out: bool = False) -> None:
        if self.input_device_index is None:
            default_input_device_info = self.py_audio.get_default_input_device_info()
            self.input_device_index = default_input_device_info['index']
            logging.info(f"[pyaudio] using default sound input device: {default_input_device_info['name']}")
        else:
            device_name = self.py_audio.get_device_info_by_index(self.input_device_index)["name"]
            logging.info(f"[pyaudio] using sound input device: {device_name} (index: {self.input_device_index})")

        if self.output_device_index is None:
            default_output_device_info = self.py_audio.get_default_output_device_info()
            self.output_device_index = default_output_device_info['index']
            logging.info(f"[pyaudio] using default sound output device: {default_output_device_info['name']}")
        else:
            device_name = self.py_audio.get_device_info_by_index(self.output_device_index)["name"]
            logging.info(f"[pyaudio] using sound output device: {device_name} (index: {self.output_device_index})")

        self.stream_in = self.py_audio.open(format=pyaudio.paFloat32,
                                            channels=1,
                                            rate=self.sample_rate,
                                            input_device_index=self.input_device_index,
                                            input=True,
                                            frames_per_buffer=self.buffer_size)
        self.stream_in.start_stream()
        if start_stream_out:
            self.stream_out = self.py_audio.open(format=pyaudio.paFloat32,
                                                 channels=1,
                                                 output_device_index=self.output_device_index,
                                                 rate=self.sample_rate,
                                                 output=True,
                                                 frames_per_buffer=self.buffer_size)
            self.stream_out.start_stream()

    def read(self) -> np.ndarray:
        assert self.stream_in is not None, "stream_in was None"
        audio_buffer = self.stream_in.read(self.buffer_size, exception_on_overflow=False)
        return np.fromstring(audio_buffer, dtype=np.float32)

    def play(self, audio_buffer: np.ndarray) -> None:
        assert self.stream_out is not None, "stream_out was None"
        self.stream_out.write(audio_buffer.tobytes())

    def close(self) -> None:
        if self.stream_in is not None:
            self.stream_in.stop_stream()
            self.stream_in.close()
        if self.stream_out is not None:
            self.stream_out.stop_stream()
            self.stream_out.close()
