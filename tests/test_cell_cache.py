"""D12: the extractor's cells, cached beside the audio and replayed.

The fake stream here emits a schedule a real one could produce -- passes of
differing width, an empty pass, cells that do not start at zero -- because the
replay's whole job is reproducing the *grouping* as well as the values.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from lib.analyser.mert_stream import Cell, StreamGeometry
from simulate import cell_cache

GEOMETRY = StreamGeometry(model_id="m-a-p/MERT-v1-330M", layers=(4, 8, 12),
                          margin_sec=3.0, hop_sec=1.0, buffer_sec=30.0,
                          label_frame_sec=0.5, encoder_sha="decfecaef6d14868")

# One pass per entry: how many cells it emits.  The empty one is not decoration
# -- a pass that emits nothing still has to advance the replay's cursor, or the
# `while due()` loop above it spins forever.
SCHEDULE = (2, 1, 0, 3)
DIM = 6


class FakeStream:
    """A MertStream-shaped source with a schedule fixed up front."""

    def __init__(self, schedule=SCHEDULE, hop_samples: int = 100) -> None:
        self.geometry = GEOMETRY
        self._schedule = tuple(schedule)
        self._hop = hop_samples
        self.reset()

    def reset(self) -> None:
        self._written = 0
        self._passes = 0
        self._next_index = 0

    def push_audio(self, samples) -> None:
        self._written += len(samples)

    def due(self) -> bool:
        return (self._passes < len(self._schedule)
                and self._written >= (self._passes + 1) * self._hop)

    def run_pass(self) -> list:
        count = self._schedule[self._passes]
        self._passes += 1
        seen = self._passes * self._hop / 100.0
        cells = []
        for _ in range(count):
            index = self._next_index
            self._next_index += 1
            features = np.full(DIM, index, dtype=np.float32)
            cells.append(Cell(index, (index + 1) * GEOMETRY.label_frame_sec,
                              features, seen))
        return cells


def drive(stream, buffers: int = 6, buffer_size: int = 100) -> list:
    """The consumer's own loop: feed, then drain every pass that came due."""
    groups = []
    for _ in range(buffers):
        stream.push_audio(np.zeros(buffer_size, dtype=np.float32))
        while stream.due():
            groups.append(stream.run_pass())
    return groups


def as_tuples(groups) -> list:
    return [[(c.index, c.time_sec, c.features.tolist(), c.audio_seen_sec)
             for c in group] for group in groups]


@pytest.fixture
def audio(tmp_path) -> Path:
    path = tmp_path / "song.mp3"
    path.write_bytes(b"not really an mp3, but it has a size and an mtime")
    return path


@pytest.fixture
def key(audio) -> dict:
    return cell_cache.cache_key(GEOMETRY, source_rate=44100, audio_path=audio,
                                decode_path="librosa")


def record(audio: Path, key: dict, stream=None) -> Path:
    path = cell_cache.sidecar_path(audio, key["decode"])
    recorder = cell_cache.Recorder(stream or FakeStream(), path, key)
    drive(recorder)
    recorder.save()
    return path


# --------------------------------------------------------------------------- #
# The replay
# --------------------------------------------------------------------------- #


def test_a_replay_reproduces_the_cells_pass_for_pass(audio, key):
    live = drive(FakeStream())
    path = record(audio, key)
    replay, reason = cell_cache.open_replay(path, key)
    assert reason == "hit"
    assert as_tuples(drive(replay)) == as_tuples(live)


def test_the_pass_grouping_survives_and_not_just_the_cell_order(audio, key):
    """Flattening would still pass a cell-for-cell check and would hand the
    student a different number of steps per buffer."""
    path = record(audio, key)
    replay, _ = cell_cache.open_replay(path, key)
    assert [len(group) for group in drive(replay)] == list(SCHEDULE)


def test_a_reset_mid_run_is_replayed_where_it_happened(audio, key):
    """The engine resets the chain at both song boundaries, so a recording that
    ignored resets would replay a schedule the live stream never ran."""

    def run(stream):
        groups = []
        for buffer in range(6):
            if buffer == 3:
                stream.reset()
            stream.push_audio(np.zeros(100, dtype=np.float32))
            while stream.due():
                groups.append(stream.run_pass())
        return groups

    path = cell_cache.sidecar_path(audio, key["decode"])
    recorder = cell_cache.Recorder(FakeStream(), path, key)
    live = as_tuples(run(recorder))
    recorder.save()
    replay, _ = cell_cache.open_replay(path, key)
    assert as_tuples(run(replay)) == live


def test_an_exhausted_replay_reports_nothing_due_rather_than_looping(audio, key):
    """The `while due()` loop above it must terminate at the end of the
    recording.  Pushing PAST the recording is a different question and a
    different answer -- see the truncation test below; here the replay has been
    fed exactly what it was recorded over."""
    path = record(audio, key)
    replay, _ = cell_cache.open_replay(path, key)
    drive(replay)
    assert not replay.due()
    assert replay.run_pass() == []


def test_a_warm_run_cannot_reach_the_gpu_from_here(audio, key):
    """The point of the cache.  Asserted on the module rather than on a run,
    because "it happened not to load one this time" is not the claim."""
    path = record(audio, key)
    replay, _ = cell_cache.open_replay(path, key)
    assert not hasattr(replay, "_encoder")
    assert not hasattr(replay, "set_encoder")
    source = Path(cell_cache.__file__).read_text(encoding="utf-8")
    assert "torch" not in source and "transformers" not in source


# --------------------------------------------------------------------------- #
# The key, and every miss it names
# --------------------------------------------------------------------------- #


def test_the_decode_path_is_in_the_filename_not_only_in_the_key(audio):
    """#161: the sim decodes with librosa and the corpus with ffmpeg, and the
    two disagree on 13.2% of near-boundary decisions.  Two decodes cannot
    collide when they cannot name the same file."""
    librosa = cell_cache.sidecar_path(audio, "librosa")
    ffmpeg = cell_cache.sidecar_path(audio, "ffmpeg")
    assert librosa != ffmpeg
    assert librosa.parent == audio.parent


def test_a_sidecar_cut_under_another_decode_is_a_named_miss(audio, key):
    path = record(audio, key)
    wanted = dict(key, decode="ffmpeg")
    replay, reason = cell_cache.open_replay(path, wanted)
    assert replay is None
    assert reason == "miss_decode_path"


@pytest.mark.parametrize("field,value,reason", [
    ("schema", "mert-cells/0", "miss_schema"),
    ("extractor", "0000000000000000", "miss_extractor"),
    ("source_rate", 48000, "miss_source_rate"),
    ("audio_size", 999, "miss_audio_changed"),
    ("audio_mtime", 1.0, "miss_audio_changed"),
])
def test_a_moved_top_level_field_is_its_own_named_miss(audio, key, field, value,
                                                       reason):
    path = record(audio, key)
    replay, found = cell_cache.open_replay(path, dict(key, **{field: value}))
    assert replay is None
    assert found == reason


@pytest.mark.parametrize("group,field,value,reason", [
    ("encoder", "encoder_sha", "0000000000000000", "miss_encoder"),
    ("encoder", "layers", [4, 8], "miss_encoder"),
    ("encoder", "revision", "deadbeef", "miss_encoder"),
    ("framing", "margin_sec", 5.0, "miss_framing"),
    ("framing", "hop_sec", 2.0, "miss_framing"),
    ("framing", "buffer_sec", 20.0, "miss_framing"),
    ("framing", "label_frame_sec", 0.25, "miss_framing"),
])
def test_a_moved_geometry_field_is_its_own_named_miss(audio, key, group, field,
                                                      value, reason):
    path = record(audio, key)
    wanted = dict(key)
    wanted[group] = dict(key[group], **{field: value})
    replay, found = cell_cache.open_replay(path, wanted)
    assert replay is None
    assert found == reason


def test_no_sidecar_is_a_miss_and_not_an_error(audio, key):
    path = cell_cache.sidecar_path(audio, key["decode"])
    replay, reason = cell_cache.open_replay(path, key)
    assert replay is None
    assert reason == "miss_new"


def test_an_unreadable_sidecar_is_a_miss_and_not_an_error(audio, key):
    path = cell_cache.sidecar_path(audio, key["decode"])
    path.write_bytes(b"truncated garbage")
    replay, reason = cell_cache.open_replay(path, key)
    assert replay is None
    assert reason == "miss_unreadable"


def test_the_key_carries_the_geometry_the_features_were_framed_under(key):
    assert key["encoder"]["encoder_sha"] == GEOMETRY.encoder_sha
    assert key["encoder"]["layers"] == list(GEOMETRY.layers)
    assert key["framing"] == {"margin_sec": 3.0, "hop_sec": 1.0,
                              "buffer_sec": 30.0, "label_frame_sec": 0.5}


def test_the_audio_key_moves_when_the_file_does(audio, key):
    audio.write_bytes(b"a different recording entirely, of a different length")
    moved = cell_cache.cache_key(GEOMETRY, source_rate=44100, audio_path=audio,
                                 decode_path="librosa")
    assert moved["audio_size"] != key["audio_size"]


# --------------------------------------------------------------------------- #
# The bytes
# --------------------------------------------------------------------------- #


def test_the_archive_bytes_are_a_pure_function_of_the_cells(audio, key):
    """The same discipline as the posterior sidecars: fixed member order, fixed
    epoch, no compression, so the file depends on nothing but its contents."""
    first = record(audio, key).read_bytes()
    second = record(audio, key).read_bytes()
    assert first == second


def test_the_stored_key_is_the_key_that_was_asked_for(audio, key):
    path = record(audio, key)
    with np.load(path) as archive:
        assert json.loads(str(archive["key"])) == key


def test_a_recording_that_emitted_nothing_still_writes_a_usable_sidecar(audio, key):
    path = cell_cache.sidecar_path(audio, key["decode"])
    recorder = cell_cache.Recorder(FakeStream(schedule=()), path, key)
    drive(recorder)
    recorder.save()
    replay, reason = cell_cache.open_replay(path, key)
    assert reason == "hit"
    assert drive(replay) == []


# --------------------------------------------------------------------------- #
# The wiring
# --------------------------------------------------------------------------- #


def test_a_client_that_names_no_file_gets_no_cache():
    """A microphone names neither a file nor a decoder.  Asked first, because
    the plan resolves the corpus artifacts and must not on a live input."""
    from simulate import runner

    class Microphone:
        pass

    assert runner._cell_cache_plan(Microphone()) is None


def test_a_file_client_declares_which_decoder_made_its_samples():
    from simulate.fake_audio_client import FileAudioClient

    assert FileAudioClient.decode_path == "librosa"


@pytest.mark.integration
def test_a_warm_chain_loads_no_encoder(nn_artifacts, anchor_mp3, monkeypatch):
    """The claim the cache exists to make, asserted where it can fail: the
    encoder loader is replaced with one that raises."""
    from lib import section_chain
    from lib.analyser import mert_stream

    monkeypatch.setattr(mert_stream, "load_encoder", _refuse)
    replay, reason = cell_cache.open_replay(
        cell_cache.sidecar_path(anchor_mp3, "librosa"),
        cell_cache.cache_key(section_chain.read_geometry().stream,
                             source_rate=44100, audio_path=anchor_mp3,
                             decode_path="librosa"))
    if replay is None:
        pytest.skip(f"no warm sidecar for the anchor track ({reason}); run "
                    f"`python auto_pilot simulate file` on it first")
    chain = section_chain.build_section_chain(extractor=lambda _: replay)
    assert chain.stream.stream is replay


def _refuse(*args, **kwargs):
    raise AssertionError("a warm run loaded the encoder")


# --------------------------------------------------------------------------- #
# What the key has to carry beyond the geometry
# --------------------------------------------------------------------------- #


def test_the_key_carries_the_arithmetic_the_cells_were_computed_under(key):
    """Device and precision are resolved at build time, not requested, and
    cuda-fp16 and cpu-fp32 are different numbers for the same audio."""
    assert set(key["backend"]) == {"device", "precision"}
    assert key["backend"]["precision"] in ("fp16", "fp32")


def test_a_sidecar_recorded_under_another_backend_is_a_named_miss(audio, key):
    """Without this a sidecar recorded on a CPU box replays as the GPU path's
    answer -- a different measurement, with nothing anywhere to say so."""
    path = record(audio, key)
    other = json.loads(json.dumps(key))
    other["backend"] = {"device": "cpu", "precision": "fp32"}
    replay, reason = cell_cache.open_replay(path, other)
    assert replay is None and reason == "miss_backend"


def test_the_builder_and_the_key_read_the_backend_from_one_place():
    """They cannot drift apart if there is only one answer."""
    from lib import section_chain

    assert section_chain.resolve_backend("cpu", False) == {
        "device": "cpu", "precision": "fp32"}
    assert section_chain.resolve_backend("cuda")["precision"] == "fp16"


# --------------------------------------------------------------------------- #
# A recording that stopped early
# --------------------------------------------------------------------------- #


def test_a_recording_records_how_much_audio_it_covers(audio, key):
    path = record(audio, key)
    with np.load(path) as archive:
        assert int(archive["total_pushed"]) == 6 * 100


def test_a_recording_shorter_than_this_run_is_a_named_miss(audio, key):
    """`run_simulation` stops on a duration_sec bound as readily as on the end
    of the file, and saves an identically keyed sidecar either way."""
    path = record(audio, key, stream=FakeStream())
    replay, reason = cell_cache.open_replay(path, key, expected_samples=6 * 100)
    assert replay is not None and reason == "hit"

    replay, reason = cell_cache.open_replay(path, key,
                                            expected_samples=6 * 100 + 1)
    assert replay is None and reason == "miss_truncated"


def test_a_replay_pushed_past_its_recording_refuses_instead_of_going_quiet(
        audio, key):
    """The guarantee for a caller that cannot say how long the run will be:
    serving nothing for the rest of a track is a silent wrong show."""
    path = record(audio, key)
    replay, reason = cell_cache.open_replay(path, key)
    assert reason == "hit"

    drive(replay, buffers=6)
    with pytest.raises(cell_cache.TruncatedRecording):
        replay.push_audio(np.zeros(100, dtype=np.float32))


def test_a_replay_over_exactly_its_recording_is_not_refused(audio, key):
    path = record(audio, key)
    replay, _reason = cell_cache.open_replay(path, key)
    assert as_tuples(drive(replay, buffers=6)) == as_tuples(drive(FakeStream()))


# --------------------------------------------------------------------------- #
# Publishing the sidecar
# --------------------------------------------------------------------------- #


def test_the_temp_name_is_the_writer_s_own(audio, key, monkeypatch):
    """The corpus batch and the benchmark both run pools over the same files.
    One fixed `.part` beside the audio let two writers publish each other's
    bytes -- and on Windows raised PermissionError out of a finished run."""
    seen = []
    real = Path.write_bytes

    def spy(self, data):
        seen.append(self.name)
        return real(self, data)

    monkeypatch.setattr(Path, "write_bytes", spy)
    record(audio, key)
    assert len(seen) == 1
    assert str(os.getpid()) in seen[0] and seen[0].endswith(".part")


def test_a_sidecar_that_cannot_be_published_does_not_fail_the_run(
        audio, key, monkeypatch, caplog):
    """A cache is an optimisation: the simulation that just finished is
    complete whether or not its cells reach the disk."""
    def refuse(self, data):
        raise PermissionError(13, "used by another process")

    monkeypatch.setattr(Path, "write_bytes", refuse)
    with caplog.at_level("WARNING"):
        record(audio, key)
    assert any("could not publish" in message for message in caplog.messages)
    assert not cell_cache.sidecar_path(audio, key["decode"]).exists()


def test_an_archive_whose_arrays_disagree_is_a_named_miss(audio, key):
    """Caught at load, where it costs a re-record, rather than as an IndexError
    out of the middle of a simulation minutes into a batch."""
    path = record(audio, key)
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["cell_index"] = arrays["cell_index"][:-1]
    cell_cache._write_archive(path, arrays)

    replay, reason = cell_cache.open_replay(path, key)
    assert replay is None and reason == "miss_schema"


def test_an_edit_to_the_extractor_moves_the_key():
    """The mechanism, not a restatement of the field.

    Every other part of the key describes the encoder's WEIGHTS and the
    geometry they were pooled under; nothing described the code between them,
    and the only lever was a schema string nothing coupled to the module.  So
    an extractor edit that left the geometry alone replayed stale cells and the
    benchmark's checksum gate printed MATCHES with the extractor never called.
    """
    import hashlib

    sources = cell_cache._EXTRACTOR_SOURCES
    assert any(path.name == "mert_stream.py" for path in sources),         'the module that turns audio into cells is not in the identity'
    assert any(path.name == "cell_cache.py" for path in sources),         'the module that replays them is not in the identity'

    before = cell_cache.extractor_sha()
    edited = [path.read_bytes() for path in sources]
    for index, path in enumerate(sources):
        digest = hashlib.sha256()
        for other in range(len(sources)):
            digest.update(edited[other] + (b"# edit" if other == index else b""))
        assert digest.hexdigest()[:16] != before,             f'an edit to {path.name} does not move the key'
