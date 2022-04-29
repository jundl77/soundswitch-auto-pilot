import pyaudio
import numpy as np


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

    def support_output(self) -> bool:
        return self.stream_out is not None

    def start_streams(self, start_stream_out: bool = False) -> None:
        if self.input_device_index is None:
            default_input_device_info = self.py_audio.get_default_input_device_info()
            self.input_device_index = default_input_device_info['index']
            print(f"Using default input device: {default_input_device_info['name']}")
        else:
            print(f"Using input device with index: {self.input_device_index}")

        if self.output_device_index is None:
            default_output_device_info = self.py_audio.get_default_output_device_info()
            self.output_device_index = default_output_device_info['index']
            print(f"Using default output device: {default_output_device_info['name']}")
        else:
            print(f"Using output device with index: {self.output_device_index}")

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
