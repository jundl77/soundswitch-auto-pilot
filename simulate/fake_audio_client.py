"""
FileAudioClient — the simulation audio source. Decodes a real audio file
(MP3/WAV/FLAC) with librosa and feeds 256-sample buffers; caches the decoded
array as <path>.<samplerate>.npy (valid while newer than the source) so repeat
runs skip the multi-second decode.

It never throttles: read() returns immediately. Pacing (real-time or virtual)
is the simulation runner's responsibility. Implements the same interface as
PyAudioClient.read() / start_streams() / close() plus an `exhausted` property.
"""

import os
import logging
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File audio client
# ---------------------------------------------------------------------------

class FileAudioClient:
    """Decodes an audio file and feeds 256-sample buffers (no throttling).

    Takes no clock: it is a pure sample pump — read() walks a position,
    nothing here consults time.
    """

    def __init__(self, sample_rate: int, buffer_size: int, path: str):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.path = path
        self._audio: np.ndarray | None = None
        self._pos = 0

    def list_devices(self): pass
    def support_output(self) -> bool: return False

    def start_streams(self, start_stream_out: bool = False):
        cache_path = f'{self.path}.{self.sample_rate}.npy'
        src_mtime = os.path.getmtime(self.path)
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) > src_mtime:
            log.info(f'[fake_audio] loading decode cache {cache_path}')
            self._audio = np.load(cache_path)
        else:
            import librosa
            log.info(f'[fake_audio] decoding {self.path} ...')
            audio, _ = librosa.load(self.path, sr=self.sample_rate, mono=True)
            self._audio = audio.astype(np.float32)
            np.save(cache_path, self._audio)
            log.info(f'[fake_audio] decode cache written → {cache_path}')
        self._pos = 0
        log.info(f'[fake_audio] loaded {len(self._audio) / self.sample_rate:.1f}s of audio')

    def play(self, audio_buffer: np.ndarray): pass
    def close(self): pass

    @property
    def exhausted(self) -> bool:
        return self._audio is not None and self._pos >= len(self._audio)

    def read(self) -> np.ndarray:
        end = self._pos + self.buffer_size
        if end > len(self._audio):
            # Pad last buffer with silence
            buf = np.zeros(self.buffer_size, dtype=np.float32)
            remaining = len(self._audio) - self._pos
            if remaining > 0:
                buf[:remaining] = self._audio[self._pos:]
        else:
            buf = self._audio[self._pos:end].copy()
        self._pos = min(end, len(self._audio))
        return buf

    @property
    def duration_sec(self) -> float:
        return len(self._audio) / self.sample_rate if self._audio is not None else 0.0
