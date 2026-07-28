"""madmom's online rhythm stack, adapted to the live pipeline's buffer cadence.

The live loop reads 256-sample buffers (5.805 ms at 44.1 kHz); madmom's online
models are trained at 441-sample hops (10 ms). That mismatch is the whole reason
this module exists: it accumulates buffers into whole hops, runs both online
chains on each hop, and hands back the events that fired inside the buffer just
processed. Nothing above it needs to know a hop size.

Everything here is causal. The beat chain is a unidirectional LSTM ensemble
feeding a DBN decoded with the forward algorithm — a beat is reported at a frame
already consumed, never one still to come. There is deliberately no downbeat
chain: madmom's downbeat tracker is whole-sequence Viterbi over a bidirectional
RNN and has no online mode at all, so importing it would quietly make this
pipeline non-causal.

The one free parameter is the onset peak-picking threshold. aubio's onset
detector had its own baked in; madmom's needs one, and every onset-density
constant in the light engine is denominated in the rate it produces. See
`ONSET_THRESHOLD` for how it was set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

# madmom's online models are trained at 100 frames per second. This is not a
# tuning knob: changing it invalidates the networks' learned time constants.
FPS = 100
SAMPLE_RATE = 44100
HOP_SIZE = SAMPLE_RATE // FPS       # 441 samples = 10 ms
# The RNNs read a right-aligned window ending at the current hop, so the frame
# size costs no latency — it only sets how much past context each frame carries.
FRAME_SIZE = 2048

# Onset peak-picking threshold, set by matching the median onset RATE of the
# aubio stream it replaces rather than by taking madmom's library default —
# every density constant in lib/engine/light_engine.py is expressed against
# that rate, so moving it would make the migration's deltas unreadable.
# Measured over 17 WHOLE tracks by training/onset_operating_point.py, evidence
# committed beside it: aubio's median is 6.077/s and this lands within 0.028/s
# of it, per-track ratio p10/p50/p90 = 0.85 / 1.03 / 1.28. madmom's own default
# (0.50) would have come in 13 % low.
#
# Whole tracks, not prefixes, and that is load-bearing: the same sweep matches
# at 0.30 on 90 s prefixes and 0.40 on 240 s ones, because the two detectors'
# rate ratio rises with the density of the material and a track's opening is
# its sparsest. Calibrating on a prefix calibrates against an intro.
ONSET_THRESHOLD = 0.40


@dataclass
class RhythmEvents:
    """What fired inside one audio buffer. Times are madmom stream seconds."""

    beats: list[float] = field(default_factory=list)
    onsets: list[float] = field(default_factory=list)
    # The beat network's output at the hop that produced the last beat. This is
    # a raw activation, NOT a calibrated confidence — the DBN decides beats from
    # the whole activation sequence, so a low value here can still be a correct
    # beat. Carried for debug logging, deliberately not used for any decision.
    beat_activation: float = 0.0


class _BeatStage:
    """madmom's online beat chain, one hop at a time."""

    def __init__(self):
        from madmom.features.beats import (DBNBeatTrackingProcessor,
                                           RNNBeatProcessor)
        from madmom.processors import BufferProcessor
        # The full 8-model ensemble: the single-model shortcut madmom's own
        # docstring offers was measured at 0.813 agreement with the offline
        # decode against the ensemble's 0.968, which is not worth the CPU it
        # gives back.
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
    """madmom's online onset chain, one hop at a time."""

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
        # Stages are injectable so the framing logic can be tested without
        # loading eight pickled LSTMs per test.
        self._beats = beat_stage if beat_stage is not None else _BeatStage()
        self._onsets = onset_stage if onset_stage is not None else _OnsetStage()
        self._onsets_enabled = True
        self._pending = np.zeros(0, dtype=np.float32)

    @property
    def pending_latency_sec(self) -> float:
        """Audio held back waiting for a whole hop — this adapter's own share of
        the look-ahead budget. Bounded by one hop minus one sample."""
        return len(self._pending) / SAMPLE_RATE

    def set_onsets_enabled(self, enabled: bool) -> None:
        """Shed or restore the onset chain (the drift watchdog's lever)."""
        if enabled != self._onsets_enabled:
            logging.warning('[madmom] onset detection %s',
                            'restored' if enabled else 'SHED — density features go stale')
            self._onsets_enabled = enabled

    def reset(self) -> None:
        """Return to the constructed state without rebuilding the models."""
        self._beats.reset()
        self._onsets.reset()
        # The partial hop goes too: carrying it across a sound stop would splice
        # the tail of one track onto the head of the next inside one frame.
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, audio_buffer: np.ndarray) -> RhythmEvents:
        """Feed one audio buffer; return whatever fired inside it."""
        from madmom.audio.signal import Signal

        self._pending = (audio_buffer.astype(np.float32) if len(self._pending) == 0
                         else np.concatenate((self._pending, audio_buffer)))
        events = RhythmEvents()
        while len(self._pending) >= HOP_SIZE:
            hop, self._pending = self._pending[:HOP_SIZE], self._pending[HOP_SIZE:]
            hop = Signal(hop, sample_rate=SAMPLE_RATE, num_channels=1)
            beats = np.atleast_1d(self._beats(hop)).tolist()
            if beats:
                events.beats.extend(beats)
                events.beat_activation = getattr(self._beats, 'last_activation', 0.0)
            if self._onsets_enabled:
                events.onsets.extend(np.atleast_1d(self._onsets(hop)).tolist())
        return events
