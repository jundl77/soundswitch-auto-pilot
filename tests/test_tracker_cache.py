"""The tracker chunk cache: record, replay, and named misses."""
import numpy as np
import pytest

from lib.analyser.bar_tracker import TrackerChunk
from simulate import tracker_cache

RECORD = {"sha256": "f313307e" * 8,
          "geometry": {"sample_rate": 22050, "fps": 50, "chunk_frames": 1500,
                       "margin_frames": 6, "stride_sec": 2.5}}


class _FakeTracker:
    alive = True
    pass_sec = []

    def __init__(self):
        self.fed = 0
        self.passes = 0
        self.resets = 0
        self.resyncs = 0

    def feed(self, samples):
        self.fed += len(samples)

    def due(self):
        return self.fed >= (self.passes + 1) * 1000

    def run_pass(self):
        if not self.due():
            return []
        p = self.passes
        self.passes += 1
        if p == 1:
            return []          # a pass that emitted nothing
        lo, hi = p * 10, p * 10 + 10
        return [TrackerChunk(lo, hi,
                             np.arange(lo, hi, dtype=np.float32), p + 0.5)]

    def reset(self):
        self.resets += 1

    def resync(self):
        self.resyncs += 1


def _key(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"x" * 100)
    return audio, tracker_cache.cache_key(RECORD, source_rate=44100,
                                          audio_path=audio,
                                          decode_path="librosa")


def _record_session(path, key):
    recorder = tracker_cache.TrackerRecorder(_FakeTracker(), path, key)
    produced = []
    for _ in range(5):
        recorder.feed(np.zeros(1000, dtype=np.float32))
        while recorder.due():
            produced.extend(recorder.run_pass())
    recorder.save()
    return produced


def test_replay_reproduces_the_recorded_chunks_at_the_recorded_triggers(tmp_path):
    audio, key = _key(tmp_path)
    path = tracker_cache.sidecar_path(audio, "librosa")
    produced = _record_session(path, key)

    replay, reason = tracker_cache.open_replay(path, key, expected_samples=5000)
    assert reason == "hit"
    got = []
    for _ in range(5):
        replay.feed(np.zeros(1000, dtype=np.float32))
        while replay.due():
            got.extend(replay.run_pass())
    assert len(got) == len(produced)
    for a, b in zip(got, produced):
        assert (a.frame_lo, a.frame_hi, a.avail_sec) == \
            (b.frame_lo, b.frame_hi, b.avail_sec)
        assert np.array_equal(a.db_logits, b.db_logits)
    assert replay.alive


def test_misses_are_named(tmp_path):
    audio, key = _key(tmp_path)
    path = tracker_cache.sidecar_path(audio, "librosa")
    _record_session(path, key)

    assert tracker_cache.open_replay(path.with_name("x.npz"), key)[1] == "miss_new"
    changed = dict(key, checkpoint_sha256="0" * 64)
    assert tracker_cache.open_replay(path, changed)[1] == "miss_checkpoint"
    changed = dict(key, geometry=dict(key["geometry"], stride_sec=5.0))
    assert tracker_cache.open_replay(path, changed)[1] == "miss_geometry"
    changed = dict(key, decode="ffmpeg")
    assert tracker_cache.open_replay(path, changed)[1] == "miss_decode_path"
    assert tracker_cache.open_replay(path, key,
                                     expected_samples=9999)[1] == "miss_truncated"


def test_a_truncated_recording_refuses_to_overrun(tmp_path):
    from simulate.cell_cache import TruncatedRecording

    audio, key = _key(tmp_path)
    path = tracker_cache.sidecar_path(audio, "librosa")
    _record_session(path, key)
    replay, _ = tracker_cache.open_replay(path, key)
    replay.feed(np.zeros(5000, dtype=np.float32))
    with pytest.raises(TruncatedRecording):
        replay.feed(np.zeros(1, dtype=np.float32))


def test_the_archive_bytes_are_reproducible(tmp_path):
    audio, key = _key(tmp_path)
    path_a = tmp_path / "a.npz"
    path_b = tmp_path / "b.npz"
    _record_session(path_a, key)
    _record_session(path_b, key)
    assert path_a.read_bytes() == path_b.read_bytes()
