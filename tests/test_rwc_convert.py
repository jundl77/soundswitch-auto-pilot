"""Tests for the RWC converter (``training/third_party/rwc_convert.py``).

Every input fact this converter reads is one a naive parse gets wrong quietly.
The times are 100 Hz integer frames, so a reader that treats them as seconds
produces a plausible timeline a hundred times too long.  The labels are wrapped
in quotes that are literally in the file, so an unstripped one is a label
nothing downstream matches.  The key shift is parenthesised, signed and not
fixed width.  The first section does not always start at zero, RWC-Pop is not
always 4/4, and the CHORUS files are an exact partition -- so a gap or an
overlap is damaged input rather than something to smooth over.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
for _path in (str(TRAINING_DIR), str(TRAINING_DIR / "raveform"),
              str(TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rwc_convert  # noqa: E402
import tp_record  # noqa: E402

# The first three lines of RM-P001.CHORUS.TXT: a start of 4 frames rather than
# 0, an interior space in the label, and the two key-shift extremes.
P001_HEAD = [
    '4\t1026\t"intro"',
    '1026\t2315\t"chorus A"\t(+1)',
    '2315\t3604\t"chorus B"\t(-10)',
]

METADATA_HEADER = (
    "RWCID;CollID;PieceNo;CDNo;TrackNo;Title;Artist;SingerInformation;"
    "SingingLanguage;Tempo;Variation;LiveInstruments;DrumInformation;Composer;"
    "CompositionType;GenreMain;GenreSub;audio_start;audio_end;duration"
)

DEFAULT_BEATS = [(0.02, 1), (0.5, 2), (10.5, 3), (11.0, 4), (25.0, 1),
                 (39.5, 2)]


def chorus_text(lines, terminator="\r\n"):
    return terminator.join(lines) + terminator


def write_chorus(archive, number, lines, terminator="\r\n"):
    path = rwc_convert.chorus_path(archive, number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(chorus_text(lines, terminator).encode("ascii"))
    return path


def metadata_row(number, title="Eien no replica", artist="Kazuo Nishi",
                 genre="J-pop", tempo="135", duration="40.0"):
    return ";".join([
        rwc_convert.native_id(number), "RWC-MDB-P-2001", str(number), "1",
        str(number), title, artist, "male", "japanese", tempo, "", "drums",
        "real", "A Composer", "popular", "Popular Music", genre, "0.0",
        duration, duration])


def write_metadata(preprocessed, rows):
    path = rwc_convert.metadata_path(preprocessed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([METADATA_HEADER, *rows]) + "\n",
                    encoding="utf-8")
    return path


def write_beats(preprocessed, number, beats):
    path = rwc_convert.beats_path(preprocessed, number)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["t;beat"] + [f"{time};{position}" for time, position in beats]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_audio(root, number, payload=b"RIFFfake"):
    path = root / f"{rwc_convert.native_id(number)}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def make_rig(tmp_path, lines=None, beats=None, number=1, duration="40.0"):
    archive = tmp_path / "archive"
    preprocessed = tmp_path / "preprocessed"
    audio = tmp_path / "wav"
    corpus = tmp_path / "corpus"
    (corpus / "annotations").mkdir(parents=True)

    write_chorus(archive, number, lines if lines is not None else P001_HEAD)
    write_metadata(preprocessed, [metadata_row(number, duration=duration)])
    write_beats(preprocessed, number,
                beats if beats is not None else DEFAULT_BEATS)
    write_audio(audio, number)
    return archive, preprocessed, audio, corpus


def run(rig, extra=()):
    archive, preprocessed, audio, corpus = rig
    pieces = [] if "--pieces" in extra else ["--pieces", "1"]
    return rwc_convert.main([
        "--archive", str(archive), "--preprocessed", str(preprocessed),
        "--audio", str(audio), "--data-dir", str(corpus), *pieces, *extra])


def load_record(corpus, number=1):
    path = tp_record.annotation_path(corpus, rwc_convert.track_id(number),
                                     "rwc")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_report(corpus, name=rwc_convert.PARTIAL_REPORT_FILE):
    # Every rig here names its pieces, so every rig here is a partial run.
    with open(tp_record.third_party_dir(corpus) / name, "r",
              encoding="utf-8") as handle:
        return json.load(handle)


def beat_grid_rows(corpus, number=1):
    path = (corpus / "annotations" / "beats"
            / f"{rwc_convert.track_id(number)}.beat.csv")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def snapshot(corpus):
    files = {}
    for path in sorted(corpus.rglob("*")):
        if not path.is_file() or path.name in (
                rwc_convert.REPORT_FILE, rwc_convert.PARTIAL_REPORT_FILE):
            continue
        name = path.relative_to(corpus).as_posix()
        if name.endswith(".rwc.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["provenance"].pop("converted_utc")
            files[name] = json.dumps(record, sort_keys=True)
        else:
            files[name] = path.read_bytes()
    return files


# --------------------------------------------------------------------------- #
# The CHORUS parse
# --------------------------------------------------------------------------- #


def test_the_first_lines_of_rm_p001_parse_as_hundredths_of_a_second():
    sections = rwc_convert.parse_chorus(chorus_text(P001_HEAD))

    assert [(section.name, section.start, section.end)
            for section in sections] == [
        ("intro", 0.04, 10.26),
        ("chorus A", 10.26, 23.15),
        ("chorus B", 23.15, 36.04),
    ]


def test_a_first_start_of_seventy_nine_frames_is_preserved_not_zeroed():
    sections = rwc_convert.parse_chorus(chorus_text([
        '79\t1000\t"intro"', '1000\t2000\t"verse A"']))

    assert sections[0].start == 0.79


def test_a_key_shift_is_a_signed_int_and_absent_from_a_three_field_line():
    sections = rwc_convert.parse_chorus(chorus_text(P001_HEAD))

    assert sections[0].key_shift is None
    assert sections[1].key_shift == 1
    assert sections[2].key_shift == -10

    entries = rwc_convert.section_dicts(sections)
    assert "key_shift" not in entries[0]
    assert entries[1]["key_shift"] == 1
    assert entries[2]["key_shift"] == -10


def test_the_quotes_are_stripped_and_an_interior_space_survives_verbatim():
    sections = rwc_convert.parse_chorus(chorus_text([
        '0\t100\t"pre-chorus"', '100\t200\t"chorus A"',
        '200\t300\t"nothing"']))

    assert [section.name for section in sections] == [
        "pre-chorus", "chorus A", "nothing"]


def test_crlf_input_with_a_trailing_blank_line_parses_cleanly():
    assert len(rwc_convert.parse_chorus(chorus_text(P001_HEAD) + "\r\n")) == 3


def test_lf_input_parses_the_same_as_crlf():
    assert (rwc_convert.parse_chorus(chorus_text(P001_HEAD, "\n"))
            == rwc_convert.parse_chorus(chorus_text(P001_HEAD, "\r\n")))


def test_a_gap_in_the_partition_is_refused():
    with pytest.raises(rwc_convert.ConvertError, match="gap"):
        rwc_convert.parse_chorus(chorus_text([
            '0\t100\t"intro"', '150\t200\t"verse A"']))


def test_an_overlap_in_the_partition_is_refused():
    with pytest.raises(rwc_convert.ConvertError, match="overlap"):
        rwc_convert.parse_chorus(chorus_text([
            '0\t100\t"intro"', '80\t200\t"verse A"']))


def test_a_zero_length_span_is_refused():
    with pytest.raises(rwc_convert.ConvertError, match="empty or reversed"):
        rwc_convert.parse_chorus(chorus_text(['100\t100\t"intro"']))


def test_a_two_field_line_is_refused():
    with pytest.raises(rwc_convert.ConvertError, match="tab-separated"):
        rwc_convert.parse_chorus(chorus_text(['0\t100']))


def test_a_five_field_line_is_refused():
    with pytest.raises(rwc_convert.ConvertError, match="tab-separated"):
        rwc_convert.parse_chorus(chorus_text(['0\t100\t"intro"\t(+1)\tx']))


def test_a_decimal_time_is_refused():
    with pytest.raises(rwc_convert.ConvertError, match="not integers"):
        rwc_convert.parse_chorus(chorus_text(['0.0\t100\t"intro"']))


def test_an_unquoted_label_is_refused():
    with pytest.raises(rwc_convert.ConvertError, match="quote"):
        rwc_convert.parse_chorus(chorus_text(['0\t100\tintro']))


def test_a_key_shift_without_parentheses_is_refused():
    with pytest.raises(rwc_convert.ConvertError, match="key shift"):
        rwc_convert.parse_chorus(chorus_text(['0\t100\t"intro"\t+1']))


def test_repeated_intro_lines_are_kept_as_the_file_states_them():
    sections = rwc_convert.parse_chorus(chorus_text([
        '0\t100\t"intro"', '100\t200\t"verse A"', '200\t300\t"intro"']))

    assert [section.name for section in sections] == [
        "intro", "verse A", "intro"]


# --------------------------------------------------------------------------- #
# The emitted record
# --------------------------------------------------------------------------- #


def test_a_converted_piece_emits_a_record_that_validates(tmp_path):
    rig = make_rig(tmp_path)
    corpus = rig[3]

    assert run(rig) == 0

    record = load_record(corpus)
    tp_record.validate_record(record)
    assert record["schema"] == tp_record.RECORD_SCHEMA
    assert record["source"] == "rwc"
    assert record["id"] == "rwc-p001"
    assert record["native_id"] == "RWC_P001"
    assert record["title"] == "Eien no replica"
    assert record["artist"] == "Kazuo Nishi"
    assert record["genre"] == "J-pop"
    assert record["audio"] == "rwc-p001.wav"
    assert record["duration"] == 40.0
    assert record["label_vocabulary"] == "rwc_aist_chorus"
    assert record["beats"] == "rwc_preprocessed"
    assert record["masked"] == []
    assert record["sections"][0] == {"name": "intro", "start": 0.04,
                                     "end": 10.26}
    assert record["sections"][1]["key_shift"] == 1


def test_the_provenance_names_its_three_sources(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    provenance = load_record(rig[3])["provenance"]
    assert provenance["converter"] == "rwc_convert"
    assert provenance["chorus_file"] == "RM-P001.CHORUS.TXT"
    assert provenance["beats_file"] == "RWC_P001.csv"
    assert provenance["metadata_file"] == "metadata.csv"
    assert provenance["tempo_bpm"] == 135.0
    assert len(provenance["chorus_sha256"]) == 64
    assert len(provenance["beats_sha256"]) == 64
    assert "research use only" in provenance["license"]
    assert provenance["converted_utc"].endswith("Z")


def test_the_audio_is_copied_under_the_track_id(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    copied = rig[3] / "third_party" / "audio" / "rwc" / "rwc-p001.wav"
    assert copied.read_bytes() == b"RIFFfake"


def test_no_audio_skips_the_copy_and_still_emits_the_record(tmp_path):
    archive, preprocessed, _audio, corpus = make_rig(tmp_path)

    code = rwc_convert.main([
        "--archive", str(archive), "--preprocessed", str(preprocessed),
        "--data-dir", str(corpus), "--pieces", "1", "--no-audio"])

    assert code == 0
    assert load_record(corpus)["audio"] == "rwc-p001.wav"
    assert not (corpus / "third_party" / "audio").exists()


def test_a_missing_audio_file_fails_that_piece_and_not_the_batch(tmp_path):
    rig = make_rig(tmp_path)
    archive, preprocessed, _audio, corpus = rig
    write_chorus(archive, 2, P001_HEAD)
    write_beats(preprocessed, 2, DEFAULT_BEATS)
    write_metadata(preprocessed, [metadata_row(1), metadata_row(2)])

    assert run(rig, ["--pieces", "1", "2"]) == 1

    report = load_report(corpus)
    assert report["converted"] == 1
    assert report["failed"] == 1
    assert report["failures"][0]["id"] == "rwc-p002"
    assert "no audio file" in report["failures"][0]["reason"]
    assert (corpus / "annotations" / "rwc-p001.rwc.json").exists()
    assert not (corpus / "annotations" / "rwc-p002.rwc.json").exists()


def test_a_broken_partition_is_reported_as_a_failure_not_repaired(tmp_path):
    rig = make_rig(tmp_path, lines=['0\t100\t"intro"', '150\t200\t"verse A"'])
    corpus = rig[3]

    assert run(rig) == 1

    report = load_report(corpus)
    assert report["converted"] == 0
    assert report["failed"] == 1
    assert report["failures"][0]["id"] == "rwc-p001"
    assert "gap" in report["failures"][0]["reason"]
    assert not (corpus / "annotations" / "rwc-p001.rwc.json").exists()


# --------------------------------------------------------------------------- #
# The beat grid
# --------------------------------------------------------------------------- #


def test_the_beat_grid_is_the_published_three_column_format(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    assert beat_grid_rows(rig[3]) == [
        ["time", "downbeat", "section"],
        ["0.02", "1", "start"],
        ["0.5", "2", "intro"],
        ["10.5", "3", "chorus A"],
        ["11", "4", "chorus A"],
        ["25", "1", "chorus B"],
        ["39.5", "2", "end"],
    ]


def test_a_beat_position_of_five_survives_and_is_counted(tmp_path):
    rig = make_rig(tmp_path, beats=[(0.02, 1), (0.5, 2), (10.5, 3),
                                    (11.0, 4), (25.0, 5)])
    corpus = rig[3]
    run(rig)

    assert beat_grid_rows(corpus)[5] == ["25", "5", "chorus B"]
    report = load_report(corpus)
    assert report["odd_meter_count"] == 1
    assert report["odd_meter_pieces"] == [
        {"id": "rwc-p001", "max_position": 5}]


def test_a_four_four_track_is_not_counted_as_odd_meter(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    report = load_report(rig[3])
    assert report["odd_meter_count"] == 0
    assert report["odd_meter_pieces"] == []


def test_an_existing_beat_grid_is_kept(tmp_path):
    rig = make_rig(tmp_path)
    grid = rig[3] / "annotations" / "beats" / "rwc-p001.beat.csv"
    grid.parent.mkdir(parents=True, exist_ok=True)
    grid.write_text("time,downbeat,section\n0.25,1,intro\n", encoding="utf-8")

    run(rig)

    assert grid.read_text(encoding="utf-8") == (
        "time,downbeat,section\n0.25,1,intro\n")


# --------------------------------------------------------------------------- #
# The report and idempotency
# --------------------------------------------------------------------------- #


def test_the_report_counts_labels_seconds_and_nothing_sections(tmp_path):
    rig = make_rig(tmp_path, lines=['0\t1000\t"intro"', '1000\t2000\t"chorus A"',
                                    '2000\t2001\t"nothing"',
                                    '2001\t4000\t"chorus A"'])
    run(rig)

    report = load_report(rig[3])
    assert report["converted"] == 1
    assert report["failed"] == 0
    assert report["failures"] == []
    assert report["labels"] == {"chorus A": 2, "intro": 1, "nothing": 1}
    assert report["nothing_sections"] == 1
    assert report["labelled_sec"] == 40.0
    assert report["audio_sec"] == 40.0


def test_labelled_seconds_are_reported_against_audio_seconds(tmp_path):
    rig = make_rig(tmp_path, duration="90.0")
    run(rig)

    report = load_report(rig[3])
    assert report["labelled_sec"] == 36.0
    assert report["audio_sec"] == 90.0


def test_limit_converts_only_the_first_n_pieces(tmp_path):
    rig = make_rig(tmp_path)
    archive, preprocessed, audio, corpus = rig
    for number in (2, 3):
        write_chorus(archive, number, P001_HEAD)
        write_beats(preprocessed, number, DEFAULT_BEATS)
        write_audio(audio, number)
    write_metadata(preprocessed, [metadata_row(n) for n in (1, 2, 3)])

    assert run(rig, ["--pieces", "1", "2", "3", "--limit", "2"]) == 0

    report = load_report(corpus)
    assert report["pieces_requested"] == 2
    assert report["converted"] == 2
    assert not (corpus / "annotations" / "rwc-p003.rwc.json").exists()


def test_an_os_error_on_one_piece_is_recorded_and_the_batch_continues(
        tmp_path, monkeypatch):
    rig = make_rig(tmp_path)
    archive, preprocessed, audio, corpus = rig
    write_chorus(archive, 2, P001_HEAD)
    write_beats(preprocessed, 2, DEFAULT_BEATS)
    write_audio(audio, 2)
    write_metadata(preprocessed, [metadata_row(1), metadata_row(2)])

    real = rwc_convert.copy_audio

    def flaky(source, target):
        if source.stem.endswith("001"):
            raise OSError(28, "No space left on device", str(target))
        return real(source, target)

    monkeypatch.setattr(rwc_convert, "copy_audio", flaky)

    assert run(rig, ["--pieces", "1", "2"]) == 1

    report = load_report(corpus)
    assert report["converted"] == 1
    assert report["failed"] == 1
    assert report["failures"][0]["id"] == "rwc-p001"
    assert "No space left on device" in report["failures"][0]["reason"]
    assert not (corpus / "annotations" / "rwc-p001.rwc.json").exists()
    assert (corpus / "annotations" / "rwc-p002.rwc.json").exists()


def test_a_second_conversion_leaves_the_record_byte_identical(
        tmp_path, monkeypatch):
    rig = make_rig(tmp_path)
    corpus = rig[3]
    monkeypatch.setattr(rwc_convert, "_now_utc", lambda: "2026-01-01T00:00:00Z")
    run(rig)
    path = corpus / "annotations" / "rwc-p001.rwc.json"
    before = path.read_bytes()

    monkeypatch.setattr(rwc_convert, "_now_utc", lambda: "2027-02-03T04:05:06Z")
    assert run(rig) == 0

    assert path.read_bytes() == before
    assert json.loads(before)["provenance"]["converted_utc"] == (
        "2026-01-01T00:00:00Z")


def test_a_changed_record_takes_the_fresh_timestamp(tmp_path, monkeypatch):
    rig = make_rig(tmp_path)
    archive, preprocessed, _audio, corpus = rig
    monkeypatch.setattr(rwc_convert, "_now_utc", lambda: "2026-01-01T00:00:00Z")
    run(rig)

    write_metadata(preprocessed, [metadata_row(1, title="Renamed")])
    monkeypatch.setattr(rwc_convert, "_now_utc", lambda: "2027-02-03T04:05:06Z")
    run(rig)

    record = load_record(corpus)
    assert record["title"] == "Renamed"
    assert record["provenance"]["converted_utc"] == "2027-02-03T04:05:06Z"


def test_a_limited_run_writes_the_partial_report_and_spares_the_full_one(
        tmp_path):
    rig = make_rig(tmp_path)
    corpus = rig[3]
    full = tp_record.third_party_dir(corpus) / rwc_convert.REPORT_FILE
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text('{"converted": 100}\n', encoding="utf-8")

    assert run(rig, ["--pieces", "1", "--limit", "1"]) == 0

    partial = load_report(corpus)
    assert partial["partial"] is True
    assert partial["report_file"] == rwc_convert.PARTIAL_REPORT_FILE
    assert partial["converted"] == 1
    assert full.read_text(encoding="utf-8") == '{"converted": 100}\n'


def test_the_whole_set_writes_the_full_report_name():
    assert rwc_convert.report_file(
        list(range(1, rwc_convert.PIECE_COUNT + 1))) == rwc_convert.REPORT_FILE
    assert rwc_convert.report_file([1, 2, 3]) == rwc_convert.PARTIAL_REPORT_FILE


def test_conversion_is_idempotent(tmp_path):
    rig = make_rig(tmp_path)
    corpus = rig[3]

    assert run(rig) == 0
    first = load_report(corpus)
    before = snapshot(corpus)

    assert run(rig) == 0
    second = load_report(corpus)

    assert snapshot(corpus) == before
    for key in ("converted", "failed", "labels", "labelled_sec", "audio_sec",
                "nothing_sections", "odd_meter_pieces"):
        assert first[key] == second[key]


def test_a_re_run_does_not_re_copy_audio_of_the_same_size(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)
    copied = rig[3] / "third_party" / "audio" / "rwc" / "rwc-p001.wav"
    copied.write_bytes(b"SENTINE!")

    run(rig)

    assert copied.read_bytes() == b"SENTINE!"
