"""The live MERT feature stage: 44.1 kHz in, pooled label cells out."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.signal import resample_poly

from lib.audio_config import SAMPLE_RATE as SOURCE_SAMPLE_RATE

ENCODER_SAMPLE_RATE = 24000
ENCODER_SAMPLES_PER_FRAME = 320
ENCODER_RECEPTIVE_FIELD = 400


def encoder_samples(seconds: float, sample_rate: int = ENCODER_SAMPLE_RATE) -> int:
    return int(math.floor(seconds * sample_rate / ENCODER_SAMPLES_PER_FRAME)) \
        * ENCODER_SAMPLES_PER_FRAME


def encoder_frames(n_samples: int) -> int:
    if int(n_samples) < ENCODER_RECEPTIVE_FIELD:
        return 0
    return 1 + (int(n_samples) - ENCODER_RECEPTIVE_FIELD) // ENCODER_SAMPLES_PER_FRAME


def encoder_frame_times(n_frames: int, *, offset_samples: int,
                        sample_rate: int = ENCODER_SAMPLE_RATE) -> np.ndarray:
    starts = np.arange(int(n_frames), dtype=np.float64) * ENCODER_SAMPLES_PER_FRAME
    return (offset_samples + starts + ENCODER_RECEPTIVE_FIELD / 2.0) / float(sample_rate)


def frame_selection(n_frames: int, *, offset_samples: int, lo_sec: float,
                    hi_sec: float, sample_rate: int = ENCODER_SAMPLE_RATE):
    times = encoder_frame_times(n_frames, offset_samples=offset_samples,
                                sample_rate=sample_rate)
    keep = np.flatnonzero((times >= lo_sec) & (times < hi_sec))
    return times[keep], keep


# `resample_poly`'s FIR spans 37 input samples at 80/147; one down-block covers it.
_CONTEXT_BLOCKS = 2
_WORK_BLOCKS = 20


class Flushed(RuntimeError):
    ...


class StreamingResampler:
    def __init__(self, source_rate: int = SOURCE_SAMPLE_RATE,
                 target_rate: int = ENCODER_SAMPLE_RATE) -> None:
        divisor = math.gcd(int(source_rate), int(target_rate))
        self.up = int(target_rate) // divisor
        self.down = int(source_rate) // divisor
        self._context = self.down * _CONTEXT_BLOCKS
        self._work = self.down * _WORK_BLOCKS
        self.reset()

    def reset(self) -> None:
        self._pending = np.zeros(self._context, dtype=np.float32)
        self._flushed = False

    @property
    def identity(self) -> bool:
        return self.up == 1 and self.down == 1

    def push(self, samples) -> np.ndarray:
        if self._flushed:
            raise Flushed("the resampler has been flushed; reset it first")
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self.identity:
            return np.ascontiguousarray(block)
        if len(block):
            self._pending = np.concatenate([self._pending, block])
        return self._emit(self._work)

    def flush(self) -> np.ndarray:
        if self._flushed or self.identity:
            self._flushed = True
            return np.zeros(0, dtype=np.float32)
        remaining = len(self._pending) - self._context
        take = int(math.ceil(remaining / self.down)) * self.down
        pad = take - remaining + self._context
        self._pending = np.concatenate(
            [self._pending, np.zeros(max(pad, 0), dtype=np.float32)])
        out = self._emit(self.down, exact=remaining)
        self._flushed = True
        return out

    def _emit(self, block: int, exact: int | None = None) -> np.ndarray:
        usable = len(self._pending) - 2 * self._context
        take = (usable // block) * block
        if take <= 0:
            return np.zeros(0, dtype=np.float32)
        window = self._pending[:take + 2 * self._context]
        filtered = resample_poly(window, self.up, self.down)
        head = self._context * self.up // self.down
        count = (take * self.up // self.down if exact is None
                 else int(math.ceil(exact * self.up / self.down)))
        self._pending = self._pending[take:]
        return np.ascontiguousarray(filtered[head:head + count], dtype=np.float32)


class RingOverrun(RuntimeError):
    ...


class SampleRing:
    def __init__(self, capacity: int) -> None:
        self._buffer = np.zeros(int(capacity), dtype=np.float32)
        self._written = 0
        self._reserved = 0

    @property
    def written(self) -> int:
        return self._written

    @property
    def capacity(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._reserved = 0
        self._buffer[:] = 0.0
        self._written = 0

    def write(self, samples) -> None:
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        count = len(block)
        if not count:
            return
        capacity = self.capacity
        dropped = 0
        if count >= capacity:
            dropped = count - capacity
            block = block[-capacity:]
            count = capacity
        head = (self._written + dropped) % capacity
        end = head + count
        self._reserved = self._written + dropped + count
        if end <= capacity:
            self._buffer[head:end] = block
        else:
            split = capacity - head
            self._buffer[head:] = block[:split]
            self._buffer[:end - capacity] = block[split:]
        self._written = self._reserved

    def snapshot(self, start: int, end: int) -> np.ndarray:
        if end > self._written:
            raise ValueError(f"samples [{start}, {end}) are not written yet "
                             f"({self._written} so far)")
        if start < 0 or end < start:
            raise ValueError(f"bad span [{start}, {end})")
        self._still_held(start)
        capacity = self.capacity
        head = start % capacity
        count = end - start
        if head + count <= capacity:
            out = np.array(self._buffer[head:head + count], dtype=np.float32)
        else:
            split = capacity - head
            out = np.concatenate([self._buffer[head:],
                                  self._buffer[:count - split]]).astype(np.float32)
        self._still_held(start)
        return out

    def _still_held(self, start: int) -> None:
        if self._reserved - start > self.capacity:
            raise RingOverrun(f"samples from {start} have been overwritten; the "
                              f"ring holds {self.capacity} and is at "
                              f"{self._reserved}")


def pass_schedule(n_samples: int, *, length: int, hop: int, margin: int):
    lo = 0
    step = 1
    while lo < n_samples:
        end = min(step * hop, n_samples)
        step += 1
        final = end >= n_samples
        hi = n_samples if final else end - margin
        if hi <= lo:
            if final:
                break
            continue
        yield max(0, end - length), end, (lo, hi)
        lo = hi


class CellAccumulator:
    def __init__(self, n_layers: int, dim: int, label_frame_sec: float) -> None:
        self._shape = (int(n_layers), int(dim))
        self._label_frame_sec = float(label_frame_sec)
        self.reset()

    def reset(self) -> None:
        self._sums: dict = {}
        self._counts: dict = {}
        self._next = 0
        self._last = None

    @property
    def next_index(self) -> int:
        return self._next

    def skip_to(self, index: int) -> int:
        index = int(index)
        if index <= self._next:
            return 0
        for key in [key for key in self._sums if key < index]:
            del self._sums[key]
            del self._counts[key]
        skipped = index - self._next
        self._next = index
        self._last = None
        return skipped

    def add(self, stacked: np.ndarray, times: np.ndarray, lo_sec: float,
            hi_sec: float) -> None:
        keep = (times >= lo_sec) & (times < hi_sec)
        if not keep.any():
            return
        rows = np.asarray(stacked)[keep].astype(np.float64)
        cell = np.floor(np.asarray(times)[keep]
                        / self._label_frame_sec).astype(np.int64)
        inside = cell >= self._next
        if not inside.any():
            return
        rows, cell = rows[inside], cell[inside]
        base = int(cell.min())
        span = int(cell.max()) - base + 1
        sums = np.zeros((span,) + self._shape, dtype=np.float64)
        counts = np.zeros(span, dtype=np.int64)
        np.add.at(sums, cell - base, rows)
        np.add.at(counts, cell - base, 1)
        for offset in np.flatnonzero(counts):
            index = base + int(offset)
            if index in self._sums:
                self._sums[index] += sums[offset]
                self._counts[index] += int(counts[offset])
            else:
                self._sums[index] = sums[offset]
                self._counts[index] = int(counts[offset])

    def drain(self, hi_sec: float, *, final: bool = False) -> list:
        if final:
            limit = (max(self._sums) + 1) if self._sums else self._next
        else:
            limit = int(math.floor(hi_sec / self._label_frame_sec))
        out = []
        while self._next < limit:
            index = self._next
            count = self._counts.pop(index, 0)
            if count:
                row = (self._sums.pop(index) / count).astype(np.float32)
                self._last = row
            elif self._last is not None:
                row = self._last
            else:
                row = self._back_fill(limit)
                if row is None:
                    break
                self._last = row
            out.append((index, row))
            self._next += 1
        return out

    def _back_fill(self, limit: int):
        reached = [index for index in sorted(self._sums) if index < limit]
        if not reached:
            return None
        first = reached[0]
        return (self._sums[first] / self._counts[first]).astype(np.float32)


DEFAULT_MODEL_ID = "m-a-p/MERT-v1-330M"
DEFAULT_MODEL_REVISION = "5240c2708a5acaee1007f43fb9735c7dcd0b78c9"
DEFAULT_ENCODER_SHA = "decfecaef6d14868"


@dataclass(frozen=True)
class StreamGeometry:
    model_id: str
    layers: tuple
    margin_sec: float
    hop_sec: float
    buffer_sec: float
    label_frame_sec: float
    encoder_sha: str | None = None
    revision: str = DEFAULT_MODEL_REVISION

    @property
    def margin_samples(self) -> int:
        return encoder_samples(self.margin_sec)

    @property
    def hop_samples(self) -> int:
        return encoder_samples(self.hop_sec)

    @property
    def buffer_samples(self) -> int:
        return encoder_samples(self.buffer_sec)

    @property
    def worst_case_future_sec(self) -> float:
        return self.margin_sec + self.hop_sec


def load_stream_geometry(affine_path, *, label_frame_sec: float,
                         model_id: str = DEFAULT_MODEL_ID,
                         revision: str = DEFAULT_MODEL_REVISION,
                         encoder_sha: str | None = DEFAULT_ENCODER_SHA
                         ) -> StreamGeometry:
    with np.load(affine_path) as archive:
        if "geometry" not in archive.files:
            raise ValueError(f"{affine_path} carries no geometry record")
        record = json.loads(str(archive["geometry"]))
        layers = tuple(int(layer) for layer in archive["layers"]) \
            if "layers" in archive.files else ()
    if not record:
        raise ValueError(f"{affine_path} carries an empty geometry record")
    if int(record.get("causal", 0)) != 1:
        raise ValueError(f"{affine_path} was fitted on non-causal features")
    if not layers:
        raise ValueError(f"{affine_path} names no encoder layers")
    return StreamGeometry(model_id=model_id, layers=layers,
                          margin_sec=float(record["margin_sec"]),
                          hop_sec=float(record["hop_sec"]),
                          buffer_sec=float(record["buffer_sec"]),
                          label_frame_sec=float(label_frame_sec),
                          encoder_sha=encoder_sha, revision=revision)


def load_input_affine(affine_path):
    with np.load(affine_path) as archive:
        return (np.asarray(archive["mean"], dtype=np.float32),
                np.asarray(archive["std"], dtype=np.float32))


def check_encoder_sha(actual: str, expected: str | None) -> None:
    if not expected:
        raise ValueError("refusing an unpinned encoder: there is no weights "
                         "hash to check the fetched model against")
    if actual != expected:
        raise RuntimeError(f"encoder weights hash {actual} is not the "
                           f"{expected} the features were extracted with")


def best_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def state_dict_sha(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to("cpu").numpy().tobytes())
    return digest.hexdigest()[:16]


class MertEncoder:
    def __init__(self, model, extractor, model_sha: str, layers, *,
                 device: str, fp16: bool) -> None:
        import torch

        self._model = model
        self.sample_rate = int(extractor.sampling_rate)
        self.do_normalize = bool(getattr(extractor, "do_normalize", False))
        self.layers = tuple(int(layer) for layer in layers)
        self.model_sha = model_sha
        self.device = device
        self._dtype = torch.float16 if fp16 else torch.float32
        self.dim = int(model.config.hidden_size)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def encode(self, segment, *, offset_samples: int, lo_sec: float,
               hi_sec: float):
        import torch

        segment = np.ascontiguousarray(segment, dtype=np.float32)
        if self.do_normalize:
            segment = (segment - segment.mean()) / (segment.std() + 1e-7)
        with torch.no_grad():
            x = torch.from_numpy(segment).to(self._dtype).to(self.device)[None]
            hidden = self._model(x, output_hidden_states=True).hidden_states
            n_frames = int(hidden[self.layers[0]].shape[1])
            if n_frames != encoder_frames(len(segment)):
                raise RuntimeError(
                    f"the encoder produced {n_frames} frames for "
                    f"{len(segment)} samples, not the "
                    f"{encoder_frames(len(segment))} its conv stack implies")
            times, keep = frame_selection(
                n_frames, offset_samples=offset_samples, lo_sec=lo_sec,
                hi_sec=hi_sec, sample_rate=self.sample_rate)
            index = torch.from_numpy(keep).to(self.device)
            stacked = torch.stack(
                [hidden[layer][0].index_select(0, index)
                 for layer in self.layers], dim=1).float().cpu().numpy()
            del hidden, x
        return stacked, times


def load_encoder(geometry: StreamGeometry, *, device: str, fp16: bool = True,
                 expected_sha: str | None = None) -> MertEncoder:
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    model = AutoModel.from_pretrained(geometry.model_id,
                                      revision=geometry.revision,
                                      trust_remote_code=True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        geometry.model_id, revision=geometry.revision, trust_remote_code=True)
    model.eval()
    model_sha = state_dict_sha(model)
    check_encoder_sha(model_sha, expected_sha or geometry.encoder_sha)
    if fp16:
        model = model.half()
    return MertEncoder(model.to(device), extractor, model_sha, geometry.layers,
                       device=device, fp16=fp16)


def _checked_encoder(encoder):
    if int(encoder.sample_rate) != ENCODER_SAMPLE_RATE:
        raise ValueError(f"the encoder speaks {encoder.sample_rate} Hz while "
                         f"the geometry sizes the ring, the hop and the margin "
                         f"at {ENCODER_SAMPLE_RATE}")
    return encoder


def _at_offline_sidecar_precision(row: np.ndarray) -> np.ndarray:
    return row.reshape(-1).astype(np.float16).astype(np.float32)


class Cell(NamedTuple):
    index: int
    time_sec: float
    features: np.ndarray
    audio_seen_sec: float


class Resync(NamedTuple):
    lost_samples: int
    lost_sec: float
    first_cell_index: int
    cells_lost: int


class MertStream:
    def __init__(self, encoder, *, geometry: StreamGeometry,
                 source_rate: int = SOURCE_SAMPLE_RATE) -> None:
        self.geometry = geometry
        self._encoder = _checked_encoder(encoder)
        self._resampler = StreamingResampler(source_rate, encoder.sample_rate)
        self._ring = SampleRing(geometry.buffer_samples + geometry.hop_samples)
        self._cells = CellAccumulator(encoder.n_layers, encoder.dim,
                                      geometry.label_frame_sec)
        self._passes = 0
        self._lo = 0
        self._flushed = False

    def set_encoder(self, encoder) -> None:
        encoder = _checked_encoder(encoder)
        if (encoder.n_layers, encoder.dim) != (self._encoder.n_layers,
                                               self._encoder.dim):
            raise ValueError(f"the replacement encoder emits "
                             f"{encoder.n_layers}x{encoder.dim} features, not "
                             f"the {self._encoder.n_layers}x{self._encoder.dim} "
                             f"the accumulator and the student were built for")
        self._encoder = encoder

    @property
    def samples_seen(self) -> int:
        return self._ring.written

    @property
    def passes(self) -> int:
        return self._passes

    def reset(self) -> None:
        self._resampler.reset()
        self._ring.reset()
        self._cells.reset()
        self._passes = 0
        self._lo = 0
        self._flushed = False

    def push_audio(self, samples) -> None:
        if self._flushed:
            raise Flushed("the stage was flushed at a song boundary; reset it "
                          "before pushing the next song's audio")
        self._ring.write(self._resampler.push(samples))

    def due(self) -> bool:
        return (not self._flushed
                and self._ring.written >= self._next_end())

    def run_pass(self) -> list:
        end = self._next_end()
        if self._flushed or self._ring.written < end:
            return []
        cells = self._encode_span(end, end - self.geometry.margin_samples)
        self._passes += 1
        return cells

    def resync(self) -> Resync:
        if self._flushed:
            raise Flushed("the stage was flushed; there is no live edge to "
                          "rejoin until it is reset")
        hop = self.geometry.hop_samples
        passes = max(0, self._ring.written // hop - 1)
        end = (passes + 1) * hop
        rate = float(self._encoder.sample_rate)
        lo = max(self._lo, end - self.geometry.margin_samples - hop, 0)
        lost = lo - self._lo
        skipped = self._cells.skip_to(
            math.ceil(lo / rate / self.geometry.label_frame_sec))
        self._passes = passes
        self._lo = lo
        return Resync(lost, lost / rate, self._cells.next_index, skipped)

    def flush(self) -> list:
        if self._flushed:
            raise Flushed("the stage has already been flushed; reset it first")
        self._ring.write(self._resampler.flush())
        end = self._ring.written
        self._flushed = True
        return self._encode_span(end, end, final=True)

    def _next_end(self) -> int:
        return (self._passes + 1) * self.geometry.hop_samples

    def _encode_span(self, end: int, hi: int, *, final: bool = False) -> list:
        if hi <= self._lo and not final:
            return []
        rate = float(self._encoder.sample_rate)
        start = max(0, end - self.geometry.buffer_samples)
        if hi > self._lo:
            segment = self._ring.snapshot(start, end)
            stacked, times = self._encoder.encode(
                segment, offset_samples=start, lo_sec=self._lo / rate,
                hi_sec=hi / rate)
            self._cells.add(stacked, times, self._lo / rate, hi / rate)
            self._lo = hi
        return [self._cell(index, row, end / rate)
                for index, row in self._cells.drain(hi / rate, final=final)]

    def _cell(self, index: int, row: np.ndarray, seen_sec: float) -> Cell:
        return Cell(index, (index + 1) * self.geometry.label_frame_sec,
                    _at_offline_sidecar_precision(row), seen_sec)
