"""Tests for the third-party quarantine (``training/third_party/``).

The load-bearing property is **invisibility**: a converted RWC, SALAMI or
Harmonix record sits in ``annotations/`` beside the published and hand labels
and must reach none of the dataset flow -- not the annotation loaders, not the
training table's join, not the manifest, the clean manifest, the splits or the
checksums.  That holds today because every corpus reader globs ``*.hand.json``
or reads ``segments.json``, which is a fact about other modules' globs rather
than about this package, so it is pinned here: a rename in either place is
silent, and the failure it produces is third-party labels quietly training the
show.  The second property is that the record itself is trustworthy -- the
converters run unattended over thousands of files -- and the third is that a
source shipping an expert beat grid never has it regenerated.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
for _path in (str(REPO_ROOT), str(TRAINING_DIR), str(TRAINING_DIR / "raveform"),
              str(TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_clean_manifest as gate  # noqa: E402
import tp_admission as admission  # noqa: E402
import tp_record  # noqa: E402
from build_training_table import load_sections_by_track  # noqa: E402
from lib.label_space import SECTION_LABELS  # noqa: E402
from raveform_fetch_annotations import (  # noqa: E402
    load_all_tracks,
    load_hand_tracks,
    parse_beat_csv,
)

PUBLISHED_SECTIONS = [
    {"name": "intro", "start": 0.0, "end": 60.0},
    {"name": "drop", "start": 60.0, "end": 140.0},
    {"name": "end", "start": 140.0, "end": 150.0},
]

HAND_SECTIONS = [
    {"name": "intro", "start": 0.0, "end": 30.0, "strength": "major"},
    {"name": "drop", "start": 30.0, "end": 120.0, "strength": "major"},
]

BEATS = [(0.5, 1.0), (1.0, 2.0), (1.5, 3.0), (2.0, 4.0), (2.5, 1.0)]


def rwc_sections():
    # The 23.15 -> 30.0 gap is deliberate: unlabelled audio between two
    # annotated sections is legal and must survive validation unclosed.
    return [
        {"name": "intro", "start": 0.04, "end": 10.26},
        {"name": "chorus A", "start": 10.26, "end": 23.15, "key_shift": -10},
        {"name": "ending", "start": 30.0, "end": 40.0},
    ]


def published_record(key="0001.native00001", youtube="native00001",
                     title="Some Artist - Some Track", duration=150.0):
    return {"key": key, "id": youtube, "title": title, "duration": duration,
            "sections": PUBLISHED_SECTIONS}


def hand_record(identifier="hand-ab12cd34ef56", title="Hand Artist - Hand Track",
                duration=120.0):
    return {"schema": 1, "source": "hand_label", "id": identifier,
            "title": title, "audio": f"{identifier}.mp3", "duration": duration,
            "sections": HAND_SECTIONS}


def rwc_record(identifier="rwc-p001", native_id="RWC_P001",
               title="Eien no replica", duration=207.168, sections=None,
               masked=None):
    return {
        "schema": tp_record.RECORD_SCHEMA,
        "source": "rwc",
        "id": identifier,
        "native_id": native_id,
        "title": title,
        "artist": "Kazuo Nishi",
        "genre": "J-pop",
        "audio": f"{identifier}.wav",
        "duration": duration,
        "label_vocabulary": "rwc_aist_chorus",
        "beats": "rwc_preprocessed",
        "sections": rwc_sections() if sections is None else sections,
        "masked": [] if masked is None else masked,
        "provenance": {"converter": "rwc_convert",
                       "converted_utc": "2026-08-18T09:00:00Z"},
    }


def harmonix_record(identifier="hx-0001", native_id="0001_thisgirl",
                    duration=180.5):
    return {
        "schema": tp_record.RECORD_SCHEMA,
        "source": "harmonix",
        "id": identifier,
        "native_id": native_id,
        "title": "This Girl",
        "artist": "Kungs",
        "genre": "",
        "audio": f"{identifier}.mp3",
        "duration": duration,
        "label_vocabulary": "harmonix_segments",
        "beats": "harmonix_expert",
        "sections": [{"name": "verse", "start": 0.0, "end": 40.0},
                     {"name": "chorus", "start": 40.0, "end": 100.0}],
        "masked": [{"start": 100.0, "end": 104.0}],
        "provenance": {"converter": "harmonix_convert",
                       "converted_utc": "2026-08-18T09:00:00Z"},
    }


SALAMI_SECTIONS = [
    {"name": "Bridge", "start": 10.0, "end": 20.0},
    {"name": "Chorus", "start": 30.0, "end": 40.0},
]

# 20.0 is the mask's own start, 30.0 its end: the two half-open edges, plus a
# beat before the first section and one past the last.
SALAMI_BEATS = [(5.0, 1.0), (15.0, 2.0), (20.0, 3.0), (25.0, 4.0),
                (30.0, 1.0), (35.0, 2.0), (45.0, 3.0)]


def salami_record(identifier="salami-14", duration=50.0):
    return {
        "schema": tp_record.RECORD_SCHEMA,
        "source": "salami",
        "id": identifier,
        "native_id": "14",
        "title": "Some Salami Track",
        "artist": "Some Artist",
        "genre": "",
        "audio": f"{identifier}.mp3",
        "duration": duration,
        "label_vocabulary": "salami_functions",
        "beats": "madmom_offline",
        "sections": [dict(section) for section in SALAMI_SECTIONS],
        "masked": [{"start": 20.0, "end": 30.0}],
        "provenance": {"converter": "salami_convert",
                       "converted_utc": "2026-08-18T09:00:00Z"},
    }


def make_corpus(tmp_path, published=(), hand=(), third=()):
    data_dir = tmp_path / "raveform"
    (data_dir / "annotations" / "beats").mkdir(parents=True)
    (data_dir / "audio").mkdir()
    with open(data_dir / "annotations" / "segments.json", "w",
              encoding="utf-8") as handle:
        json.dump(list(published), handle)
    for record in hand:
        path = data_dir / "annotations" / f"{record['id']}.hand.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for record in third:
        tp_record.write_record(
            tp_record.annotation_path(data_dir, record["id"], record["source"]),
            record)
        audio = tp_record.audio_path(data_dir, record)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"audio")
    return data_dir


def admission_corpus(tmp_path, monkeypatch, third=None, beats=None):
    third = [rwc_record()] if third is None else list(third)
    data_dir = make_corpus(tmp_path, [published_record()], [hand_record()], third)

    calls = []

    def fake_detect(audio):
        calls.append(Path(audio))
        return BEATS if beats is None else beats

    monkeypatch.setattr(admission, "detect_beats", fake_detect)
    monkeypatch.setattr(admission.gate, "check_track", measured)
    return data_dir, third, calls


def measured(job):
    duration = float(job.annotation_duration_sec)
    return gate.CheckResult(job.track_id, job.youtube_id, job.mp3_path,
                            duration + 0.05, duration, duration,
                            gate.STATUS_OK, "")


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_rows(data_dir):
    return read_rows(admission.tp_dir(data_dir) / admission.TP_MANIFEST_FILE)


def clean_rows(data_dir):
    return read_rows(admission.tp_dir(data_dir) / admission.TP_CLEAN_MANIFEST_FILE)


# --------------------------------------------------------------------------- #
# The quarantine
# --------------------------------------------------------------------------- #


def test_the_corpus_loaders_never_see_a_third_party_record(tmp_path):
    data_dir = make_corpus(tmp_path, [published_record()], [hand_record()],
                           [rwc_record(), harmonix_record()])

    keys = [track["key"] for track in load_all_tracks(data_dir)]

    assert keys == ["0001.native00001", "hand-ab12cd34ef56"]
    assert [record["key"] for record in load_hand_tracks(data_dir)] == [
        "hand-ab12cd34ef56"]


def test_the_training_table_join_never_sees_a_third_party_record(tmp_path):
    data_dir = make_corpus(tmp_path, [published_record()], [hand_record()],
                           [rwc_record()])

    sections = load_sections_by_track(data_dir)

    assert set(sections) == {"0001.native00001", "hand-ab12cd34ef56"}


def test_the_records_are_loadable_through_their_own_reader(tmp_path):
    data_dir = make_corpus(tmp_path, [published_record()], [],
                           [rwc_record(), harmonix_record()])

    assert [record["id"] for record in tp_record.load_records(data_dir, "rwc")] == [
        "rwc-p001"]
    assert [record["id"] for record in
            tp_record.load_all_third_party(data_dir)] == ["hx-0001", "rwc-p001"]
    assert tp_record.load_records(data_dir, "salami") == []


CORPUS_ARTIFACTS = {
    "manifest.csv": b"track_id,youtube_id,n_sections,total_sec\n"
                    b"0001.native00001,native00001,3,150.000\n",
    "clean_manifest.csv": b"track_id,youtube_id,mp3_path,ffprobe_duration_sec,"
                          b"decoded_duration_sec,annotation_duration_sec,status,"
                          b"detail\n0001.native00001,native00001,x.mp3,150.050,"
                          b"150.000,150.000,ok,\n",
    "splits.json": b'{"train": ["native00001"], "val": [], "test": []}\n',
    "checksums.sha256": b"0000  audio/native00001.mp3\n",
}


def test_admit_leaves_every_corpus_artifact_byte_identical(tmp_path, monkeypatch):
    data_dir, third, _calls = admission_corpus(tmp_path, monkeypatch)
    for name, payload in CORPUS_ARTIFACTS.items():
        (data_dir / name).write_bytes(payload)
    segments = data_dir / "annotations" / "segments.json"
    before = segments.read_bytes()
    audio_before = sorted(path.name for path in (data_dir / "audio").iterdir())

    admission.admit(third[0]["id"], "rwc", corpus=data_dir)

    for name, payload in CORPUS_ARTIFACTS.items():
        assert (data_dir / name).read_bytes() == payload
    assert segments.read_bytes() == before
    assert sorted(path.name for path in (data_dir / "audio").iterdir()) == audio_before
    assert (admission.tp_dir(data_dir) / admission.TP_MANIFEST_FILE).exists()
    assert (admission.tp_dir(data_dir) / admission.TP_CLEAN_MANIFEST_FILE).exists()


# --------------------------------------------------------------------------- #
# validate_record
# --------------------------------------------------------------------------- #


def test_validate_record_accepts_a_converted_record():
    tp_record.validate_record(rwc_record())
    tp_record.validate_record(harmonix_record())


def test_validate_record_accepts_a_gap_between_sections():
    record = rwc_record()

    tp_record.validate_record(record)

    assert record["sections"][1]["end"] < record["sections"][2]["start"]


def test_validate_record_never_checks_the_show_vocabulary():
    record = rwc_record()

    names = {section["name"] for section in record["sections"]}

    assert names - set(SECTION_LABELS) == {"chorus A", "ending"}
    tp_record.validate_record(record)


def test_validate_record_accepts_a_record_with_no_masked_spans():
    record = rwc_record()
    del record["masked"]

    tp_record.validate_record(record)


@pytest.mark.parametrize("mutate, match", [
    (lambda record: record.update(schema=2), "schema"),
    (lambda record: record.update(source="acousticbrainz"),
     "not a third-party source"),
    (lambda record: record.pop("native_id"), "record lacks native_id"),
    (lambda record: record.update(id="p001"), "must start with"),
    (lambda record: record.update(id="rwc-P001"), "lowercase"),
    (lambda record: record.update(id="rwc-p 001"), "whitespace"),
    (lambda record: record.update(id="rwc-sub/p001"), "path separator"),
    (lambda record: record.update(duration=0.0), "not positive"),
    (lambda record: record.update(duration=-1.0), "not positive"),
    (lambda record: record.update(sections=[]), "labels nothing"),
    (lambda record: record["sections"][0].update(name=" intro"),
     "stray whitespace"),
    (lambda record: record["sections"][0].update(name=""), "name is empty"),
    (lambda record: record["sections"][1].update(start=5.0), "overlapping"),
    (lambda record: record.update(sections=list(reversed(record["sections"]))),
     "unsorted"),
    (lambda record: record["sections"][-1].update(end=9000.0),
     "past the record duration"),
    (lambda record: record["sections"][0].update(end=0.04),
     "is not before end"),
    (lambda record: record.update(masked=[{"start": 10.0, "end": 5.0}]),
     "masked 0"),
    (lambda record: record.pop("provenance"), "record lacks provenance"),
    (lambda record: record.update(provenance={"converter": "rwc_convert"}),
     "provenance lacks converted_utc"),
])
def test_validate_record_rejects_by_name(mutate, match):
    record = rwc_record()
    mutate(record)

    with pytest.raises(ValueError, match=match):
        tp_record.validate_record(record)


def test_read_record_refuses_a_renamed_file(tmp_path):
    data_dir = make_corpus(tmp_path, [], [], [rwc_record()])
    original = tp_record.annotation_path(data_dir, "rwc-p001", "rwc")
    renamed = tp_record.annotation_path(data_dir, "rwc-p999", "rwc")
    original.rename(renamed)

    with pytest.raises(RuntimeError, match="renamed"):
        tp_record.read_record(renamed)


def test_write_record_round_trips_through_read_record(tmp_path):
    record = rwc_record()
    path = tp_record.write_record(tmp_path / "rwc-p001.rwc.json", record)

    assert path.read_bytes().endswith(b"\n")
    assert tp_record.read_record(path) == record
    assert tp_record.labelled_span(record) == (0.04, 40.0)


# --------------------------------------------------------------------------- #
# admit()
# --------------------------------------------------------------------------- #


def test_admit_generates_a_beat_grid_and_a_quarantined_manifest_row(
        tmp_path, monkeypatch):
    data_dir, third, calls = admission_corpus(tmp_path, monkeypatch)
    record = third[0]

    message = admission.admit(record["id"], "rwc", corpus=data_dir)

    grid = data_dir / "annotations" / "beats" / "rwc-p001.beat.csv"
    assert parse_beat_csv(grid) == admission.beat_rows(BEATS, record["sections"])
    assert calls == [tp_record.audio_path(data_dir, record)]

    rows = manifest_rows(data_dir)
    assert [row["track_id"] for row in rows] == ["rwc-p001"]
    assert rows[0]["source"] == "rwc"
    assert rows[0]["native_id"] == "RWC_P001"
    assert rows[0]["n_sections"] == "3"
    assert rows[0]["n_masked"] == "0"
    assert rows[0]["labelled_sec"] == "33.110"
    assert rows[0]["total_sec"] == "207.168"

    clean = clean_rows(data_dir)
    assert [row["track_id"] for row in clean] == ["rwc-p001"]
    assert clean[0]["source"] == "rwc"
    assert clean[0]["status"] == gate.STATUS_OK
    assert clean[0]["decoded_duration_sec"] == "207.168"

    assert "generated" in message
    assert "quarantined" in message
    assert "split" not in message


def test_admit_keeps_an_expert_beat_grid_and_never_runs_madmom(
        tmp_path, monkeypatch):
    data_dir, third, calls = admission_corpus(tmp_path, monkeypatch)
    grid = data_dir / "annotations" / "beats" / "rwc-p001.beat.csv"
    grid.write_text("time,downbeat,section\n0.04,1,intro\n0.54,2,intro\n",
                    encoding="utf-8")
    before = grid.read_bytes()

    message = admission.admit(third[0]["id"], "rwc", corpus=data_dir)

    assert calls == []
    assert grid.read_bytes() == before
    assert "kept" in message


def test_admit_is_idempotent(tmp_path, monkeypatch):
    data_dir, third, calls = admission_corpus(tmp_path, monkeypatch)

    first = admission.admit(third[0]["id"], "rwc", corpus=data_dir)
    snapshot = {str(path.relative_to(data_dir)): path.read_bytes()
                for path in data_dir.rglob("*") if path.is_file()}
    second = admission.admit(third[0]["id"], "rwc", corpus=data_dir)

    assert len(calls) == 1
    assert "generated" in first and "kept" in second
    after = {str(path.relative_to(data_dir)): path.read_bytes()
             for path in data_dir.rglob("*") if path.is_file()}
    assert after == snapshot


def test_admit_records_a_gate_failure_and_refuses(tmp_path, monkeypatch):
    data_dir, third, _calls = admission_corpus(tmp_path, monkeypatch)
    truncated = gate.CheckResult("rwc-p001", "rwc", "x.wav", 207.168, 40.0,
                                 207.168, gate.STATUS_CORRUPT, "truncated")
    monkeypatch.setattr(admission.gate, "check_track", lambda job: truncated)

    with pytest.raises(RuntimeError, match="recorded, not admitted"):
        admission.admit(third[0]["id"], "rwc", corpus=data_dir)

    row = clean_rows(data_dir)[0]
    assert row["status"] == gate.STATUS_CORRUPT
    assert row["detail"] == "truncated"
    assert not (admission.tp_dir(data_dir) / admission.TP_MANIFEST_FILE).exists()


def test_admit_refuses_missing_audio(tmp_path, monkeypatch):
    data_dir, third, _calls = admission_corpus(tmp_path, monkeypatch)
    tp_record.audio_path(data_dir, third[0]).unlink()

    with pytest.raises(RuntimeError, match="missing audio"):
        admission.admit(third[0]["id"], "rwc", corpus=data_dir)


def test_admit_refuses_a_track_with_no_record(tmp_path, monkeypatch):
    data_dir, _third, _calls = admission_corpus(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="no salami record"):
        admission.admit("salami-0001", "salami", corpus=data_dir)


def test_the_manifest_covers_every_source_sorted_by_track_id(
        tmp_path, monkeypatch):
    data_dir, third, _calls = admission_corpus(
        tmp_path, monkeypatch, third=[rwc_record(), harmonix_record()])

    for record in third:
        admission.admit(record["id"], record["source"], corpus=data_dir)

    rows = manifest_rows(data_dir)
    assert [row["track_id"] for row in rows] == ["hx-0001", "rwc-p001"]
    assert [row["source"] for row in rows] == ["harmonix", "rwc"]
    assert rows[0]["artist"] == "Kungs"
    assert rows[0]["genre"] == ""
    assert rows[0]["n_masked"] == "1"
    assert rows[0]["labelled_sec"] == "100.000"
    assert rows[0]["total_sec"] == "180.500"
    assert [row["track_id"] for row in clean_rows(data_dir)] == [
        "hx-0001", "rwc-p001"]


def test_a_masked_span_reaches_the_grid_as_its_own_sentinel_not_as_end():
    record = salami_record()

    sections = [section for _time, _position, section
                in admission.third_party_beat_rows(SALAMI_BEATS, record)]

    assert sections == ["start", "Bridge", "masked", "masked", "Chorus",
                        "Chorus", "end"]


def test_a_beat_at_a_masks_start_is_masked_and_one_at_its_end_is_not():
    record = salami_record()

    rows = admission.third_party_beat_rows([(20.0, 1.0), (29.999, 2.0),
                                            (30.0, 3.0)], record)

    assert [section for _time, _position, section in rows] == [
        admission.MASKED_SENTINEL, admission.MASKED_SENTINEL, "Chorus"]


def test_a_record_with_no_masked_key_produces_exactly_the_hand_path_rows():
    record = rwc_record()
    del record["masked"]

    assert admission.third_party_beat_rows(BEATS, record) == admission.beat_rows(
        BEATS, record["sections"])


def test_admit_writes_the_masked_sentinel_into_the_generated_grid(
        tmp_path, monkeypatch):
    data_dir, third, _calls = admission_corpus(
        tmp_path, monkeypatch, third=[salami_record()], beats=SALAMI_BEATS)

    admission.admit(third[0]["id"], "salami", corpus=data_dir)

    grid = data_dir / "annotations" / "beats" / "salami-14.beat.csv"
    assert parse_beat_csv(grid) == admission.third_party_beat_rows(
        SALAMI_BEATS, third[0])


# --------------------------------------------------------------------------- #
# --repair-sections
# --------------------------------------------------------------------------- #


def stale_grid(data_dir, record, beats=SALAMI_BEATS):
    # Exactly what the admission run in flight wrote: the hand path's rows,
    # with every masked beat carrying the tail sentinel.
    grid = tp_record.beat_csv_path(data_dir, record)
    grid.parent.mkdir(parents=True, exist_ok=True)
    admission.write_beat_csv(grid, admission.beat_rows(beats,
                                                       record["sections"]))
    return grid


def repair(data_dir, ids=(), source="salami", all_records=True):
    argv = [*ids, "--source", source, "--data-dir", str(data_dir),
            "--repair-sections"]
    if all_records:
        argv.append("--all")
    return admission.main(argv)


def test_repair_sections_rewrites_a_stale_grid_without_running_the_tracker(
        tmp_path, monkeypatch, capsys):
    data_dir, third, calls = admission_corpus(
        tmp_path, monkeypatch, third=[salami_record()], beats=SALAMI_BEATS)
    grid = stale_grid(data_dir, third[0])
    assert [row[2] for row in parse_beat_csv(grid)].count("end") == 3

    code = repair(data_dir)

    assert code == 0
    assert calls == []
    assert parse_beat_csv(grid) == admission.third_party_beat_rows(
        SALAMI_BEATS, third[0])
    out = capsys.readouterr().out
    assert "examined 1 grids, changed 1, 2 beats moved from end to masked" in out


def test_repair_sections_leaves_a_correct_grid_byte_identical(
        tmp_path, monkeypatch, capsys):
    data_dir, third, _calls = admission_corpus(
        tmp_path, monkeypatch, third=[salami_record()], beats=SALAMI_BEATS)
    grid = tp_record.beat_csv_path(data_dir, third[0])
    grid.parent.mkdir(parents=True, exist_ok=True)
    admission.write_beat_csv(grid, admission.third_party_beat_rows(
        SALAMI_BEATS, third[0]))
    before = grid.read_bytes()

    repair(data_dir)

    assert grid.read_bytes() == before
    assert "changed 0" in capsys.readouterr().out


def test_repair_sections_leaves_the_time_and_downbeat_columns_byte_identical(
        tmp_path, monkeypatch):
    data_dir, third, _calls = admission_corpus(
        tmp_path, monkeypatch, third=[salami_record()], beats=SALAMI_BEATS)
    grid = tp_record.beat_csv_path(data_dir, third[0])
    grid.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately not the writer's own formatting: a repair that re-derived
    # these columns instead of carrying their text would round them here.
    grid.write_text(
        "time,downbeat,section\n"
        "5.000000,1,start\n15.000000,2,Bridge\n20.000000,3,end\n"
        "25.000000,4,end\n30.000000,1,Chorus\n35.000000,2,Chorus\n"
        "45.000000,3,end\n", encoding="utf-8")

    repair(data_dir)

    lines = grid.read_text(encoding="utf-8").splitlines()[1:]
    columns = [line.split(",") for line in lines]
    assert [(row[0], row[1]) for row in columns] == [
        ("5.000000", "1"), ("15.000000", "2"), ("20.000000", "3"),
        ("25.000000", "4"), ("30.000000", "1"), ("35.000000", "2"),
        ("45.000000", "3")]
    assert [row[2] for row in columns] == [
        "start", "Bridge", "masked", "masked", "Chorus", "Chorus", "end"]


def test_repair_sections_skips_an_id_whose_grid_is_absent(
        tmp_path, monkeypatch, capsys):
    data_dir, third, calls = admission_corpus(
        tmp_path, monkeypatch, third=[salami_record()], beats=SALAMI_BEATS)
    grid = tp_record.beat_csv_path(data_dir, third[0])

    code = repair(data_dir)

    assert code == 0
    assert calls == []
    assert not grid.exists()
    out = capsys.readouterr().out
    assert "skipped salami-14: no beat grid to repair" in out
    assert "examined 0 grids, changed 0" in out


def test_repair_grid_refuses_a_record_with_no_grid(tmp_path, monkeypatch):
    data_dir, third, _calls = admission_corpus(
        tmp_path, monkeypatch, third=[salami_record()], beats=SALAMI_BEATS)

    with pytest.raises(RuntimeError, match="never creates one"):
        admission.repair_grid(data_dir, third[0])


def test_the_cli_keeps_going_after_a_failing_track(tmp_path, monkeypatch, capsys):
    data_dir, _third, _calls = admission_corpus(
        tmp_path, monkeypatch, third=[rwc_record(), rwc_record(
            identifier="rwc-p002", native_id="RWC_P002")])
    tp_record.audio_path(data_dir, rwc_record(identifier="rwc-p002")).unlink()

    code = admission.main(["--source", "rwc", "--all",
                           "--data-dir", str(data_dir)])

    captured = capsys.readouterr().out
    assert code == 1
    assert "admitted 1, failed 1" in captured
    assert "FAILED rwc-p002" in captured
    # The manifest is a census of converted records; the clean manifest is the
    # ledger of what actually passed the gate, so only it thins out.
    assert [row["track_id"] for row in manifest_rows(data_dir)] == [
        "rwc-p001", "rwc-p002"]
    assert [row["track_id"] for row in clean_rows(data_dir)] == ["rwc-p001"]
