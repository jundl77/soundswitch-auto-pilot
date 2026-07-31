"""madmom's online beat tracker, adapted to the live pipeline's buffer cadence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# madmom's online models are trained at 100 fps; changing it invalidates the
# networks' learned time constants.
FPS = 100
SAMPLE_RATE = 44100
HOP_SIZE = SAMPLE_RATE // FPS
FRAME_SIZE = 2048


@dataclass
class RhythmEvents:
    beats: list[float] = field(default_factory=list)
    beat_activation: float = 0.0


class _BeatStage:
    def __init__(self):
        from madmom.features.beats import (DBNBeatTrackingProcessor,
                                           RNNBeatProcessor)
        from madmom.processors import BufferProcessor
        self._rnn = RNNBeatProcessor(online=True, origin='stream',
                                     num_frames=1, fps=FPS)
        self._dbn = DBNBeatTrackingProcessor(fps=FPS, online=True)
        self._buffer = BufferProcessor(buffer_size=FRAME_SIZE)
        self.reset()

    def reset(self) -> None:
        self._dbn.reset()
        self._buffer(np.zeros(FRAME_SIZE, dtype=np.float32))
        self._primed = False
        self.last_activation = 0.0

    def __call__(self, hop: np.ndarray) -> np.ndarray:
        frame = self._buffer(hop)
        activation = np.atleast_1d(self._rnn(frame, reset=not self._primed)).flatten()[-1:]
        self._primed = True
        self.last_activation = float(activation[0])
        return self._dbn.process_online(activation, reset=False)


class MadmomRhythm:
    """Buffers in, rhythm events out. Owns all madmom state and framing."""

    def __init__(self, sample_rate: int, beat_stage=None):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f'madmom online models are trained at {SAMPLE_RATE} Hz; '
                f'refusing sample rate {sample_rate} rather than resampling')
        self._beats = beat_stage if beat_stage is not None else _BeatStage()
        self._pending = np.zeros(0, dtype=np.float32)
        self._hops = 0

    @property
    def pending_latency_sec(self) -> float:
        """Audio held back waiting for a whole hop. Bounded by one hop."""
        return len(self._pending) / SAMPLE_RATE

    def reset(self) -> None:
        """Return to the constructed state without rebuilding the models."""
        self._beats.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._hops = 0

    def process(self, audio_buffer: np.ndarray) -> RhythmEvents:
        """Feed one audio buffer; return whatever fired inside it."""
        from madmom.audio.signal import Signal

        self._pending = (audio_buffer.astype(np.float32) if len(self._pending) == 0
                         else np.concatenate((self._pending, audio_buffer)))
        events = RhythmEvents()
        while len(self._pending) >= HOP_SIZE:
            hop, self._pending = self._pending[:HOP_SIZE], self._pending[HOP_SIZE:]
            hop = Signal(hop, sample_rate=SAMPLE_RATE, num_channels=1)
            # Our own hop clock, never madmom's: the decoder times events from an
            # internal frame counter that only advances for frames it is handed,
            # so a reset re-bases it and the two streams stop agreeing.
            now = self._hops / FPS
            self._hops += 1
            if len(np.atleast_1d(self._beats(hop))):
                events.beats.append(now)
                events.beat_activation = getattr(self._beats, 'last_activation', 0.0)
        return events
