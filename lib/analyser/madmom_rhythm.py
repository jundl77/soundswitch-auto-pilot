"""madmom's online rhythm stack, adapted to the live pipeline's buffer cadence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

# madmom's online models are trained at 100 fps; changing it invalidates the
# networks' learned time constants.
FPS = 100
SAMPLE_RATE = 44100
HOP_SIZE = SAMPLE_RATE // FPS
FRAME_SIZE = 2048

# Matched to the onset RATE of the aubio detector this replaced, not madmom's
# default (0.50) — every density constant in lib/engine/light_engine.py is
# denominated in that rate. Measured by training/onset_operating_point.py.
ONSET_THRESHOLD = 0.35


@dataclass
class RhythmEvents:
    beats: list[float] = field(default_factory=list)
    onsets: list[float] = field(default_factory=list)
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


class _OnsetStage:
    def __init__(self, threshold: float = ONSET_THRESHOLD):
        from madmom.features.onsets import (OnsetPeakPickingProcessor,
                                            RNNOnsetProcessor)
        from madmom.processors import BufferProcessor
        self._rnn = RNNOnsetProcessor(online=True, origin='stream',
                                      num_frames=1, fps=FPS)
        self._picker = OnsetPeakPickingProcessor(online=True, fps=FPS,
                                                 threshold=threshold)
        self._buffer = BufferProcessor(buffer_size=FRAME_SIZE)
        self.reset()

    def reset(self) -> None:
        self._picker.reset()
        self._buffer(np.zeros(FRAME_SIZE, dtype=np.float32))
        self._primed = False

    def __call__(self, hop: np.ndarray) -> np.ndarray:
        frame = self._buffer(hop)
        activation = self._rnn(frame, reset=not self._primed)
        self._primed = True
        return self._picker.process_online(
            np.atleast_1d(activation).flatten()[-1:], reset=False)


class MadmomRhythm:
    """Buffers in, rhythm events out. Owns all madmom state and framing."""

    def __init__(self, sample_rate: int, beat_stage=None, onset_stage=None):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f'madmom online models are trained at {SAMPLE_RATE} Hz; '
                f'refusing sample rate {sample_rate} rather than resampling')
        self._beats = beat_stage if beat_stage is not None else _BeatStage()
        self._onsets = onset_stage if onset_stage is not None else _OnsetStage()
        self._onsets_enabled = True
        self._onset_epoch = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._hops = 0

    @property
    def onsets_enabled(self) -> bool:
        return self._onsets_enabled

    @property
    def onset_epoch(self) -> int:
        """Bumped whenever the onset chain's history is discarded.

        A counter, not a flag: a shed and a restore between two consumer polls
        would net out to "nothing happened".
        """
        return self._onset_epoch

    @property
    def pending_latency_sec(self) -> float:
        """Audio held back waiting for a whole hop. Bounded by one hop."""
        return len(self._pending) / SAMPLE_RATE

    def set_onsets_enabled(self, enabled: bool) -> None:
        if enabled == self._onsets_enabled:
            return
        if enabled:
            # While shed the chain saw no audio: its frame buffer and recurrent
            # state still describe pre-gap music, and feeding it post-gap audio
            # would splice the two inside one frame.
            self._onsets.reset()
            self._onset_epoch += 1
        logging.warning(
            '[madmom] onset detection %s',
            'restored' if enabled
            else 'SHED — density becomes UNMEASURED, not zero (see DENSITY_UNKNOWN)')
        self._onsets_enabled = enabled

    def reset(self) -> None:
        """Return to the constructed state without rebuilding the models."""
        self._beats.reset()
        self._onsets.reset()
        self._onset_epoch += 1
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
            # Our own hop clock, never madmom's: each decoder times events from
            # an internal frame counter that only advances for frames it is
            # handed, so a shed chain re-bases and the two streams stop agreeing.
            now = self._hops / FPS
            self._hops += 1
            if len(np.atleast_1d(self._beats(hop))):
                events.beats.append(now)
                events.beat_activation = getattr(self._beats, 'last_activation', 0.0)
            if self._onsets_enabled and len(np.atleast_1d(self._onsets(hop))):
                events.onsets.append(now)
        return events
