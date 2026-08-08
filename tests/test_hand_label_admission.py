"""Tests for the hand-label admission path (``training/hand_label_admission.py``).

Three things are pinned because each is silent when it breaks.  **Precedence**:
a hand label must win over the published annotation everywhere the dataset
reads sections, while the frozen benchmark keeps reading its pinned slice -- a
hand label on an eval-set track re-scoring the benchmark would be invisible in
every gate until the numbers moved.  **Format**: the generated beat grid must
be indistinguishable from a published one to ``parse_beat_csv`` and its
consumers, or a hand track trains on a silently different grid.  **Coverage**:
the split assignment and the artist-exclusion guard must treat a hand id like
any corpus id, or hand tracks would sit outside the contamination protections
without anything saying so.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
for _path in (str(TRAINING_DIR), str(TRAINING_DIR / "raveform")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_clean_manifest as gate  # noqa: E402
import hand_label_admission as admission  # noqa: E402
import run_eval_set  # noqa: E402
from build_training_table import load_sections_by_track  # noqa: E402
from nn.dataset import (  # noqa: E402
    SPLIT_NAMES,
    TrackRef,
    assign_split,
    candidate_tracks,
    excluded_artist_names,
    partition,
)
from raveform_fetch_annotations import (  # noqa: E402
    load_all_tracks,
    load_hand_tracks,
    load_tracks,
    merge_hand_tracks,
    parse_beat_csv,
)

PUBLISHED_SECTIONS = [
    {"name": "intro", "start": 0.0, "end": 60.0},
    {"name": "drop", "start": 60.0, "end": 140.0},
    {"name": "end", "start": 140.0, "end": 150.0},
]

HAND_SECTIONS = [
    {"name": "intro", "start": 0.0, "end": 30.0, "strength": "major"},
    {"name": "breakdown", "start": 30.0, "end": 80.0, "strength": "minor"},
    {"name": "drop", "start": 80.0, "end": 120.0, "strength": "major"},
]


def published_record(key="0001.native00001", youtube="native00001",
                     title="Some Artist - Some Track", duration=150.0):
    return {"key": key, "id": youtube, "title": title, "duration": duration,
            "sections": PUBLISHED_SECTIONS}


def hand_record(identifier="hand-ab12cd34ef56", title="Hand Artist - Hand Track",
                duration=120.0, sections=None):
    return {"schema": 1, "source": "hand_label", "id": identifier,
            "title": title, "audio": f"{identifier}.mp3", "duration": duration,
            "sections": sections if sections is not None else HAND_SECTIONS}


def make_corpus(tmp_path, published=(), hand=()):
    data_dir = tmp_path / "raveform"
    (data_dir / "annotations" / "beats").mkdir(parents=True)
    (data_dir / "audio").mkdir()
    with open(data_dir / "annotations" / "segments.json", "w",
              encoding="utf-8") as handle:
        json.dump(list(published), handle)
    for record in hand:
        path = data_dir / "annotations" / f"{record['id']}.hand.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return data_dir


# --------------------------------------------------------------------------- #
# The loader and the precedence rule
# --------------------------------------------------------------------------- #


def test_load_hand_tracks_reads_a_committed_label(tmp_path):
    record = hand_record()
    data_dir = make_corpus(tmp_path, [published_record()], [record])

    loaded = load_hand_tracks(data_dir)

    assert len(loaded) == 1
    assert loaded[0]["key"] == record["id"]
    assert loaded[0]["title"] == record["title"]
    assert loaded[0]["sections"] == record["sections"]


def test_load_hand_tracks_ignores_everything_but_hand_json(tmp_path):
    data_dir = make_corpus(tmp_path, [published_record()], [])
    (data_dir / "annotations" / "beats" / "0001.native00001.beat.csv").write_text(
        "time,downbeat,section\n0.5,1,intro\n", encoding="utf-8")

    assert load_hand_tracks(data_dir) == []


def test_load_hand_tracks_refuses_a_renamed_file_by_name(tmp_path):
    data_dir = make_corpus(tmp_path, [], [hand_record()])
    original = data_dir / "annotations" / "hand-ab12cd34ef56.hand.json"
    original.rename(data_dir / "annotations" / "hand-999999999999.hand.json")

    with pytest.raises(RuntimeError, match="renamed"):
        load_hand_tracks(data_dir)


def test_load_hand_tracks_refuses_unparsable_json_by_name(tmp_path):
    data_dir = make_corpus(tmp_path, [], [])
    bad = data_dir / "annotations" / "hand-ab12cd34ef56.hand.json"
    bad.write_text("{not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hand-ab12cd34ef56.hand.json"):
        load_hand_tracks(data_dir)


def test_load_hand_tracks_refuses_a_record_missing_fields_by_name(tmp_path):
    record = hand_record()
    del record["duration"]
    data_dir = make_corpus(tmp_path, [], [record])

    with pytest.raises(RuntimeError, match="duration"):
        load_hand_tracks(data_dir)


def test_load_all_tracks_appends_a_hand_only_track(tmp_path):
    data_dir = make_corpus(tmp_path, [published_record()], [hand_record()])

    merged = load_all_tracks(data_dir)

    assert [track["key"] for track in merged] == [
        "0001.native00001", "hand-ab12cd34ef56"]


def test_a_hand_label_on_a_native_id_wins_over_the_published_sections(tmp_path):
    override = hand_record(identifier="native00001", duration=149.5)
    data_dir = make_corpus(tmp_path, [published_record()], [override])

    merged = load_all_tracks(data_dir)

    assert len(merged) == 1
    track = merged[0]
    assert track["key"] == "0001.native00001"
    assert track["title"] == "Some Artist - Some Track"
    assert track["sections"] == override["sections"]
    assert track["duration"] == 149.5
    assert track["source"] == "hand_label"


def test_merge_without_hand_labels_changes_nothing():
    published = [published_record()]
    assert merge_hand_tracks(published, []) == published


def test_load_tracks_stays_published_only(tmp_path):
    data_dir = make_corpus(tmp_path, [published_record()], [hand_record()])

    assert [track["key"] for track in load_tracks(data_dir)] == ["0001.native00001"]


# --------------------------------------------------------------------------- #
# The generated beat grid
# --------------------------------------------------------------------------- #


BEATS = [(0.5, 1.0), (1.0, 2.0), (1.5, 3.0), (2.0, 4.0), (2.5, 1.0)]


def test_beat_rows_carry_position_and_section():
    sections = [{"name": "intro", "start": 0.75, "end": 2.2},
                {"name": "drop", "start": 2.2, "end": 2.4}]

    rows = admission.beat_rows(BEATS, sections)

    assert rows == [
        (0.5, 1, "start"),
        (1.0, 2, "intro"),
        (1.5, 3, "intro"),
        (2.0, 4, "intro"),
        (2.5, 1, "end"),
    ]


def test_the_written_grid_matches_the_published_format(tmp_path):
    path = tmp_path / "hand-ab12cd34ef56.beat.csv"
    admission.write_beat_csv(path, [(0.11, 1, "intro"), (0.5862, 2, "intro"),
                                    (1.06239999, 3, "drop")])

    data = path.read_bytes()
    assert data == b"time,downbeat,section\n0.11,1,intro\n0.5862,2,intro\n1.0624,3,drop\n"
    assert parse_beat_csv(path) == [(0.11, 1, "intro"), (0.5862, 2, "intro"),
                                    (1.0624, 3, "drop")]


def test_the_hand_grid_flags_downbeats_for_the_labeller(tmp_path):
    path = tmp_path / "hand-ab12cd34ef56.hand.beat.csv"
    admission.write_hand_grid(path, [(0.5, 1, "intro"), (1.0, 2, "intro"),
                                     (1.5, 4, "intro")])

    assert path.read_bytes() == b"time,downbeat\n0.5,1\n1,0\n1.5,0\n"


# --------------------------------------------------------------------------- #
# admit()
# --------------------------------------------------------------------------- #


def measured(record, mp3_path, decoded=None):
    duration = float(record["duration"])
    return gate.CheckResult(
        record["id"] if record["id"].startswith("hand-") else "0001.native00001",
        record["id"], str(mp3_path), duration + 0.05,
        decoded if decoded is not None else duration, duration,
        gate.STATUS_OK, "")


def admission_corpus(tmp_path, monkeypatch, record=None, beats=None):
    record = record or hand_record()
    data_dir = make_corpus(tmp_path, [published_record()], [record])
    mp3 = data_dir / "audio" / record["audio"]
    mp3.write_bytes(b"mp3")

    calls = []

    def fake_detect(audio):
        calls.append(Path(audio))
        return beats if beats is not None else BEATS

    monkeypatch.setattr(admission, "detect_beats", fake_detect)
    monkeypatch.setattr(
        admission.gate, "check_track",
        lambda job: measured(record, job.mp3_path))
    return data_dir, record, mp3, calls


def manifest_rows(data_dir):
    with open(data_dir / "manifest.csv", "r", encoding="utf-8",
              newline="") as handle:
        return list(csv.DictReader(handle))


def clean_rows(data_dir):
    with open(data_dir / gate.CLEAN_MANIFEST_FILE, "r", encoding="utf-8",
              newline="") as handle:
        return list(csv.DictReader(handle))


def test_admit_makes_the_track_dataset_complete(tmp_path, monkeypatch):
    data_dir, record, _mp3, calls = admission_corpus(tmp_path, monkeypatch)

    message = admission.admit(record["id"], corpus=data_dir)

    grid = data_dir / "annotations" / "beats" / "hand-ab12cd34ef56.beat.csv"
    assert parse_beat_csv(grid) == admission.beat_rows(BEATS, record["sections"])
    assert (data_dir / "annotations" / "beats"
            / "hand-ab12cd34ef56.hand.beat.csv").exists()

    rows = manifest_rows(data_dir)
    assert [row["track_id"] for row in rows] == [
        "0001.native00001", "hand-ab12cd34ef56"]
    hand_row = rows[1]
    assert hand_row["youtube_id"] == record["id"]
    assert hand_row["n_sections"] == "3"
    assert hand_row["total_sec"] == "120.000"

    clean = clean_rows(data_dir)
    assert [row["track_id"] for row in clean] == ["hand-ab12cd34ef56"]
    assert clean[0]["status"] == gate.STATUS_OK
    assert clean[0]["decoded_duration_sec"] == "120.000"

    assert calls == [data_dir / "audio" / record["audio"]]
    expected_split = assign_split(record["id"])
    assert expected_split in message
    assert "generated" in message


def test_admit_is_idempotent_and_never_redetects(tmp_path, monkeypatch):
    data_dir, record, _mp3, calls = admission_corpus(tmp_path, monkeypatch)

    first = admission.admit(record["id"], corpus=data_dir)
    snapshot = {path.name: path.read_bytes()
                for path in data_dir.rglob("*") if path.is_file()}
    second = admission.admit(record["id"], corpus=data_dir)

    assert len(calls) == 1
    assert "kept" in second and "generated" in first
    after = {path.name: path.read_bytes()
             for path in data_dir.rglob("*") if path.is_file()}
    assert after == snapshot


def test_admit_preserves_existing_clean_manifest_rows(tmp_path, monkeypatch):
    data_dir, record, _mp3, _calls = admission_corpus(tmp_path, monkeypatch)
    native = gate.CheckResult("0001.native00001", "native00001", "x.mp3",
                              150.0, 150.0, 150.0, gate.STATUS_OK, "")
    gate.write_clean_manifest(data_dir, [native])
    before = clean_rows(data_dir)

    admission.admit(record["id"], corpus=data_dir)

    after = clean_rows(data_dir)
    assert after[0] == before[0]
    assert [row["track_id"] for row in after] == [
        "0001.native00001", "hand-ab12cd34ef56"]


def test_admit_records_a_gate_failure_and_refuses(tmp_path, monkeypatch):
    data_dir, record, _mp3, _calls = admission_corpus(tmp_path, monkeypatch)
    truncated = gate.CheckResult(
        record["id"], record["id"], "x.mp3", 120.05, 40.0, 120.0,
        gate.STATUS_CORRUPT, "truncated")
    monkeypatch.setattr(admission.gate, "check_track", lambda job: truncated)

    with pytest.raises(RuntimeError, match="corrupt"):
        admission.admit(record["id"], corpus=data_dir)

    assert clean_rows(data_dir)[0]["status"] == gate.STATUS_CORRUPT


def test_admit_refuses_a_label_outside_the_vocabulary(tmp_path, monkeypatch):
    record = hand_record(sections=[
        {"name": "chorus", "start": 0.0, "end": 120.0, "strength": "major"}])
    data_dir, record, _mp3, _calls = admission_corpus(
        tmp_path, monkeypatch, record=record)

    with pytest.raises(RuntimeError, match="chorus"):
        admission.admit(record["id"], corpus=data_dir)


def test_admit_refuses_missing_audio(tmp_path, monkeypatch):
    data_dir, record, mp3, _calls = admission_corpus(tmp_path, monkeypatch)
    mp3.unlink()

    with pytest.raises(RuntimeError, match="missing audio"):
        admission.admit(record["id"], corpus=data_dir)


def test_admit_refuses_a_track_with_no_hand_label(tmp_path, monkeypatch):
    data_dir, _record, _mp3, _calls = admission_corpus(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="no hand label"):
        admission.admit("native00001", corpus=data_dir)


def test_admit_keeps_a_published_beat_grid(tmp_path, monkeypatch):
    override = hand_record(identifier="native00001", duration=149.5)
    override["audio"] = "native00001.mp3"
    data_dir, record, _mp3, calls = admission_corpus(
        tmp_path, monkeypatch, record=override)
    published_grid = (data_dir / "annotations" / "beats"
                      / "0001.native00001.beat.csv")
    published_grid.write_text(
        "time,downbeat,section\n0.5,1,intro\n1,2,intro\n", encoding="utf-8")
    before = published_grid.read_bytes()

    admission.admit(record["id"], corpus=data_dir)

    assert calls == []
    assert published_grid.read_bytes() == before
    hand_grid = (data_dir / "annotations" / "beats"
                 / "native00001.hand.beat.csv")
    assert hand_grid.read_bytes() == b"time,downbeat\n0.5,1\n1,0\n"


def test_admit_never_touches_segments_json(tmp_path, monkeypatch):
    data_dir, record, _mp3, _calls = admission_corpus(tmp_path, monkeypatch)
    segments = data_dir / "annotations" / "segments.json"
    before = segments.read_bytes()

    admission.admit(record["id"], corpus=data_dir)

    assert segments.read_bytes() == before
    assert not (data_dir / "checksums.sha256").exists()
    assert not (data_dir / "splits.json").exists()


# --------------------------------------------------------------------------- #
# Splits, the artist guard, and the benchmark
# --------------------------------------------------------------------------- #


def test_assign_split_covers_hand_ids():
    split = assign_split("hand-ab12cd34ef56")
    assert split in SPLIT_NAMES
    assert assign_split("hand-ab12cd34ef56") == split


def test_partition_places_a_hand_id_additively():
    frozen = {"train": ["native00001"], "val": [], "test": []}
    candidates = [
        TrackRef("0001.native00001", "native00001", "Some Artist - Some Track"),
        TrackRef("hand-ab12cd34ef56", "hand-ab12cd34ef56",
                 "Hand Artist - Hand Track"),
    ]

    result = partition(candidates, eval_ids=frozenset(),
                       artist_names=frozenset(), existing=frozen)

    assert "native00001" in result["train"]
    placed = set(result["train"]) | set(result["val"]) | set(result["test"])
    assert "hand-ab12cd34ef56" in placed
    assert assign_split("hand-ab12cd34ef56") == next(
        split for split in SPLIT_NAMES if "hand-ab12cd34ef56" in result[split])


def test_the_artist_guard_reads_a_hand_title():
    candidates = [TrackRef("hand-ab12cd34ef56", "hand-ab12cd34ef56",
                           "Greg Downey - Rewired")]
    names = excluded_artist_names([{"title": "Greg Downey - Come To Me"}])

    result = partition(candidates, eval_ids=frozenset(), artist_names=names)

    assert result["excluded_artist"] == ["hand-ab12cd34ef56"]


def test_candidate_tracks_include_a_hand_only_track(tmp_path):
    record = hand_record()
    data_dir = make_corpus(tmp_path, [published_record()], [record])
    with open(data_dir / gate.CLEAN_MANIFEST_FILE, "w", encoding="utf-8",
              newline="") as handle:
        handle.write(",".join(gate.CLEAN_MANIFEST_HEADER) + "\n")
        handle.write("hand-ab12cd34ef56,hand-ab12cd34ef56,x.mp3,"
                     "120.05,120.0,120.0,ok,\n")
    candidates, no_annotation, _unlabeled = candidate_tracks(data_dir)

    assert candidates == [TrackRef("hand-ab12cd34ef56", "hand-ab12cd34ef56",
                                   "Hand Artist - Hand Track")]
    assert no_annotation == []


def test_the_priors_refit_reads_a_hand_track(tmp_path):
    from nn.priors import corpus_bar_runs

    record = hand_record()
    data_dir = make_corpus(tmp_path, [published_record()], [record])
    beats = [(0.5 * index, index % 4 + 1) for index in range(240)]
    admission.write_beat_csv(
        data_dir / "annotations" / "beats" / "hand-ab12cd34ef56.beat.csv",
        admission.beat_rows(beats, record["sections"]))

    sequences, skipped = corpus_bar_runs(data_dir, ["hand-ab12cd34ef56"])

    assert skipped == []
    assert [label for label, _bars in sequences[0]] == [
        "intro", "breakdown", "drop"]


def test_the_benchmark_fallback_ignores_hand_labels(tmp_path):
    override = hand_record(identifier="native00001", duration=149.5)
    data_dir = make_corpus(tmp_path, [published_record()], [override])

    sections = run_eval_set.load_sections(
        data_dir, labels=tmp_path / "no-such-labels.json")

    assert sections["0001.native00001"] == [
        (0.0, 60.0, "intro"), (60.0, 140.0, "drop"), (140.0, 150.0, "end")]


def test_the_table_join_reads_hand_wins(tmp_path):
    override = hand_record(identifier="native00001", duration=149.5)
    data_dir = make_corpus(tmp_path, [published_record()], [override])

    hand_wins = load_sections_by_track(data_dir)
    published_only = load_sections_by_track(data_dir, include_hand=False)

    assert hand_wins["0001.native00001"][0] == (0.0, 30.0, "intro")
    assert published_only["0001.native00001"] == [
        (0.0, 60.0, "intro"), (60.0, 140.0, "drop"), (140.0, 150.0, "end")]
