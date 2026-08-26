"""The tracker stage: artifact verification, framing, and how it dies."""
import json
from pathlib import Path

import numpy as np
import pytest

from lib.analyser import bar_tracker
from lib.analyser.bar_tracker import (BarTrackerStream, CHUNK_FRAMES,
                                      MARGIN_FRAMES, STRIDE_FRAMES,
                                      _PASS_PAD_SRC, _SRC_PER_FRAME,
                                      _STRIDE_SRC)

RECORD = {
    "checkpoint": "model.ckpt",
    "sha256": "",
    "geometry": dict(bar_tracker.GEOMETRY),
}


def _generation(tmp_path: Path, *, sha=None, geometry=None) -> Path:
    root = tmp_path / "l9"
    tracker = root / bar_tracker.TRACKER_DIR
    tracker.mkdir(parents=True)
    checkpoint = tracker / "model.ckpt"
    checkpoint.write_bytes(b"not a real checkpoint")
    record = dict(RECORD)
    record["sha256"] = sha if sha is not None else bar_tracker._sha256(checkpoint)
    if geometry is not None:
        record["geometry"] = geometry
    (tracker / bar_tracker.RECORD_NAME).write_text(json.dumps(record),
                                                   encoding="utf-8")
    return root


def test_absent_artifacts_read_as_not_present(tmp_path):
    assert not bar_tracker.tracker_present(tmp_path / "l9")


def test_a_record_without_its_checkpoint_reads_as_not_present(tmp_path):
    root = _generation(tmp_path)
    (root / bar_tracker.TRACKER_DIR / "model.ckpt").unlink()
    assert not bar_tracker.tracker_present(root)


def test_the_record_verifies(tmp_path):
    root = _generation(tmp_path)
    assert bar_tracker.tracker_present(root)
    record = bar_tracker.load_record(root)
    assert record["checkpoint"] == "model.ckpt"


def test_a_sha_mismatch_is_fatal(tmp_path):
    root = _generation(tmp_path, sha="0" * 64)
    with pytest.raises(RuntimeError, match="sha"):
        bar_tracker.load_record(root)


def test_a_geometry_mismatch_is_fatal(tmp_path):
    geometry = dict(bar_tracker.GEOMETRY, stride_sec=5.0)
    root = _generation(tmp_path, geometry=geometry)
    with pytest.raises(RuntimeError, match="geometry"):
        bar_tracker.load_record(root)


class _FlatModel:
    """Stands in for BeatThis: constant logits, records window shapes."""

    def __init__(self, value=-8.0):
        self.value = value
        self.windows = []

    def __call__(self, x):
        import torch

        self.windows.append(x.shape)
        return {"downbeat": torch.full((x.shape[0], x.shape[1]), self.value)}


class _DyingModel:
    def __call__(self, x):
        raise RuntimeError("the card is gone")


def _stream(model) -> BarTrackerStream:
    return BarTrackerStream(model, device="cpu")


def _feed(stream, n_samples, chunk=4096, rng=None):
    rng = rng or np.random.default_rng(0)
    out = []
    fed = 0
    while fed < n_samples:
        take = min(chunk, n_samples - fed)
        stream.feed(rng.standard_normal(take).astype(np.float32) * 0.1)
        fed += take
        while stream.due():
            out.extend(stream.run_pass())
    return out


def test_chunks_are_contiguous_margin_safe_and_stamped(caplog):
    stream = _stream(_FlatModel())
    chunks = _feed(stream, 4 * _STRIDE_SRC + _PASS_PAD_SRC)
    assert [c.frame_lo for c in chunks] == [0, 119, 244, 369]
    assert [c.frame_hi for c in chunks] == [119, 244, 369, 494]
    for p, c in enumerate(chunks, start=1):
        end_f = p * STRIDE_FRAMES
        assert c.frame_hi == end_f - MARGIN_FRAMES
        assert c.avail_sec == (end_f * _SRC_PER_FRAME + _PASS_PAD_SRC) / 44100.0
        assert len(c.db_logits) == c.frame_hi - c.frame_lo
    assert stream.passes == 4
    # every model window is the full trained length
    assert all(shape == (1, CHUNK_FRAMES, 128)
               for shape in stream._model.windows)


def test_a_forward_failure_kills_the_tracker_quietly(caplog):
    stream = _stream(_DyingModel())
    chunks = _feed(stream, 2 * _STRIDE_SRC + _PASS_PAD_SRC)
    assert chunks == []
    assert not stream.alive
    assert not stream.due()
    assert any("counting grid" in message for message in caplog.messages)


def test_reset_restarts_the_frame_grid():
    stream = _stream(_FlatModel())
    _feed(stream, 2 * _STRIDE_SRC + _PASS_PAD_SRC)
    stream.reset()
    chunks = _feed(stream, _STRIDE_SRC + _PASS_PAD_SRC)
    assert [(c.frame_lo, c.frame_hi) for c in chunks] == [(0, 119)]


def test_resync_skips_to_the_live_edge_and_leaves_the_hole(caplog):
    stream = _stream(_FlatModel())
    _feed(stream, _STRIDE_SRC + _PASS_PAD_SRC)
    # a stall: audio keeps arriving, no passes run
    rng = np.random.default_rng(1)
    stream.feed(rng.standard_normal(6 * _STRIDE_SRC).astype(np.float32))
    stream.resync()
    chunks = _feed(stream, 2 * _STRIDE_SRC)
    assert chunks, "the tracker must resume at the live edge"
    assert chunks[0].frame_lo >= 7 * STRIDE_FRAMES - MARGIN_FRAMES
    assert all(b.frame_lo == a.frame_hi for a, b in zip(chunks, chunks[1:]))
