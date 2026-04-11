"""
Fake audio client for simulation. Two modes:

  BeepAudioClient  — generates a synthetic 120 BPM (configurable) metronome at known
                     timestamps. Use this for deterministic timing validation: beats
                     occur at predictable times, so we can assert that downstream
                     light commands fire at exactly beep_time + delay_sec.

  FileAudioClient  — decodes an audio file (MP3/WAV/FLAC) with librosa and replays it
                     at real-time speed. Use this for realistic pipeline testing with
                     actual music; timing validation is still possible because the queue
                     timing log records enqueue vs fire times independent of input.

Both implement the same interface as PyAudioClient.read() / start_streams() / close().
"""

import time
import logging
import numpy as np
from typing import List, Optional

log = logging.getLogger(__name__)

SAMPLE_RATE = 44100
BUFFER_SIZE = 256


# ---------------------------------------------------------------------------
# Beep audio client
# ---------------------------------------------------------------------------

class BeepAudioClient:
    """
    Generates a click at the start of each beat (configurable BPM) embedded in
    a near-silent buffer. Records the wall-clock time of each generated click so
    the simulation can compare against when the corresponding command fired.
    """

    def __init__(self, sample_rate: int, buffer_size: int, bpm: float = 120.0):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.bpm = bpm
        self._samples_per_beat = sample_rate * 60.0 / bpm
        self._total_samples = 0
        self._start_time: Optional[float] = None
        self._click = self._make_click()
        # (sample_index, wall_clock_time) for each generated click
        self.click_log: List[dict] = []

    def _make_click(self) -> np.ndarray:
        """10 ms Hann-windowed 1 kHz sine burst — easily detected by Aubio tempo."""
        n = int(self.sample_rate * 0.01)
        t = np.arange(n, dtype=np.float32)
        tone = np.sin(2 * np.pi * 1000 * t / self.sample_rate)
        return (tone * np.hanning(n)).astype(np.float32)

    def list_devices(self): pass
    def support_output(self) -> bool: return False
    def start_streams(self, start_stream_out: bool = False):
        self._start_time = time.monotonic()

    def play(self, audio_buffer: np.ndarray): pass
    def close(self): pass

    def read(self) -> np.ndarray:
        """Return the next 256-sample buffer, throttled to real-time speed."""
        buf = np.random.normal(0, 0.0005, self.buffer_size).astype(np.float32)

        # Embed a click whenever a beat boundary falls inside this buffer
        buf_start = self._total_samples
        beat_idx = int(buf_start / self._samples_per_beat)
        next_beat_sample = int((beat_idx + 1) * self._samples_per_beat)
        offset = next_beat_sample - buf_start
        if 0 <= offset < self.buffer_size:
            end = min(offset + len(self._click), self.buffer_size)
            length = end - offset
            buf[offset:end] += self._click[:length] * 0.8
            wall_time = self._start_time + next_beat_sample / self.sample_rate
            self.click_log.append({'sample': next_beat_sample, 'wall_time': wall_time})
            log.debug(f'[fake_audio] click at sample={next_beat_sample}, t={next_beat_sample / self.sample_rate:.3f}s')

        self._total_samples += self.buffer_size

        # Throttle to real-time
        expected_time = self._start_time + self._total_samples / self.sample_rate
        sleep_sec = expected_time - time.monotonic()
        if sleep_sec > 0:
            time.sleep(sleep_sec)

        return buf


# ---------------------------------------------------------------------------
# File audio client
# ---------------------------------------------------------------------------

class FileAudioClient:
    """Decodes an audio file and feeds 256-sample buffers at real-time speed."""

    def __init__(self, sample_rate: int, buffer_size: int, path: str):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.path = path
        self._audio: Optional[np.ndarray] = None
        self._pos = 0
        self._start_time: Optional[float] = None

    def list_devices(self): pass
    def support_output(self) -> bool: return False

    def start_streams(self, start_stream_out: bool = False):
        import librosa
        log.info(f'[fake_audio] loading {self.path} ...')
        audio, _ = librosa.load(self.path, sr=self.sample_rate, mono=True)
        self._audio = audio.astype(np.float32)
        self._pos = 0
        self._start_time = time.monotonic()
        log.info(f'[fake_audio] loaded {len(self._audio) / self.sample_rate:.1f}s of audio')

    def play(self, audio_buffer: np.ndarray): pass
    def close(self): pass

    def read(self) -> np.ndarray:
        end = self._pos + self.buffer_size
        if end > len(self._audio):
            # Pad last buffer with silence; simulation runner stops before this
            buf = np.zeros(self.buffer_size, dtype=np.float32)
            remaining = len(self._audio) - self._pos
            if remaining > 0:
                buf[:remaining] = self._audio[self._pos:]
        else:
            buf = self._audio[self._pos:end].copy()
        self._pos = min(end, len(self._audio))

        # Throttle to real-time
        expected_time = self._start_time + self._pos / self.sample_rate
        sleep_sec = expected_time - time.monotonic()
        if sleep_sec > 0:
            time.sleep(sleep_sec)

        return buf

    @property
    def duration_sec(self) -> float:
        return len(self._audio) / self.sample_rate if self._audio is not None else 0.0
