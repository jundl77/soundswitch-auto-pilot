"""Tests for the Harmonix converter (``training/third_party/harmonix_convert.py``).

Every fact this converter reads is one a plausible parse gets wrong in silence.
The segments delimiter is a single SPACE and the repository's own README says
tab -- a tab split yields one field per line, so a reader that trusts the docs
sees an empty timeline rather than an error.  The checkout arrives CRLF on a
machine with ``core.autocrlf=true`` while the blobs are LF, so a label read
without stripping is ``chorus\\r`` and matches nothing.  ``end`` is a tail
sentinel, not a section, and it is absent on two tracks and doubled on four --
a parser that assumes exactly one drops a real segment on the first shape and
invents a zero-length one on the second.  The bar positions run past four and
sometimes open at two; folding either would fabricate downbeats.
"""
import csv
import json
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
for _path in (str(TRAINING_DIR), str(TRAINING_DIR / "raveform"),
              str(TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import harmonix_convert  # noqa: E402
import tp_record  # noqa: E402

NATIVE = "0001_12step"
TRACK = "hx-0001_12step"
DURATION = "152.0"

# The head of segments/0001_12step.txt, verbatim, plus its tail sentinel.
SEGMENTS = ["0.0 intro", "19.3873291063 verse", "52.0233019687 chorus",
            "148.054524769 end"]
BEATS = ["0.0\t1\t1", "0.530973\t2\t1", "2.123892\t1\t2"]
ODD_BEATS = ["0.0\t1\t1", "0.530973\t2\t1", "7.058786\t5\t1",
             "7.3108855\t6\t1"]

METADATA_HEADER = ["File", "Title", "Artist", "Release", "Duration", "BPM",
                   "Ratio Bars in 4", "Time Signature", "Genre",
                   "MusicBrainz Id", "Acoustid Id"]


def text_of(lines, terminator="\r\n"):
    return terminator.join(lines) + terminator


def write_lines(path, lines, terminator="\r\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text_of(lines, terminator).encode("utf-8"))
    return path


def write_segments(dataset, native, lines, terminator="\r\n"):
    return write_lines(harmonix_convert.segments_path(dataset, native), lines,
                       terminator)


def write_beats(dataset, native, lines, terminator="\r\n"):
    return write_lines(harmonix_convert.beats_path(dataset, native), lines,
                       terminator)


def metadata_row(native, title="1, 2 Step", artist="Ciara", release="Goodies",
                 duration=DURATION, bpm="113", genre="R&B"):
    return [native, title, artist, release, duration, bpm, "100.0", "4|4",
            genre, "0408655f-189f-371b-9c41-ec861e1a7810", ""]


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_metadata(dataset, rows):
    return write_csv(dataset / harmonix_convert.METADATA_FILE,
                     METADATA_HEADER, rows)


def write_urls(dataset, natives):
    return write_csv(
        dataset / harmonix_convert.URLS_FILE, ["File", "URL"],
        [[native, f"http://www.youtube.com/watch?v=iBHNgV6_zn{index}"]
         for index, native in enumerate(natives)])


def write_scores(dataset, natives):
    return write_csv(dataset / harmonix_convert.SCORES_FILE, ["File", "score"],
                     [[native, "0.9984119782214156"] for native in natives])


def make_rig(tmp_path, segments=None, beats=None, natives=(NATIVE,),
             metadata=None, duration=DURATION, title="1, 2 Step"):
    harmonixset = tmp_path / "harmonixset"
    dataset = harmonix_convert.dataset_dir(harmonixset)
    corpus = tmp_path / "corpus"
    (corpus / "annotations").mkdir(parents=True)

    for native in natives:
        write_segments(dataset, native,
                       SEGMENTS if segments is None else segments)
        write_beats(dataset, native, BEATS if beats is None else beats)
    write_metadata(dataset, metadata if metadata is not None else [
        metadata_row(native, duration=duration, title=title)
        for native in natives])
    write_urls(dataset, natives)
    write_scores(dataset, natives)
    return harmonixset, corpus


def run(rig, extra=()):
    harmonixset, corpus = rig
    return harmonix_convert.main([
        "--harmonixset", str(harmonixset), "--data-dir", str(corpus), *extra])


def load_record(corpus, native=NATIVE):
    path = tp_record.annotation_path(corpus, harmonix_convert.track_id(native),
                                     "harmonix")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_report(corpus, name=None):
    name = name or harmonix_convert.REPORT_FILE
    with open(tp_record.third_party_dir(corpus) / name, "r",
              encoding="utf-8") as handle:
        return json.load(handle)


def beat_grid_rows(corpus, native=NATIVE):
    path = (corpus / "annotations" / "beats"
            / f"{harmonix_convert.track_id(native)}.beat.csv")
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def place_audio(corpus, native=NATIVE, payload=b"ID3fake"):
    path = harmonix_convert.corpus_audio_path(corpus, native)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def snapshot(corpus):
    files = {}
    for path in sorted(corpus.rglob("*")):
        if not path.is_file() or path.name in (
                harmonix_convert.REPORT_FILE,
                harmonix_convert.PARTIAL_REPORT_FILE):
            continue
        name = path.relative_to(corpus).as_posix()
        if name.endswith(".harmonix.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["provenance"].pop("converted_utc")
            files[name] = json.dumps(record, sort_keys=True)
        else:
            files[name] = path.read_bytes()
    return files


# --------------------------------------------------------------------------- #
# The segments parse
# --------------------------------------------------------------------------- #


def test_segments_are_space_separated_not_tab_separated():
    parse = harmonix_convert.parse_segments(text_of(SEGMENTS), 152.0)

    assert [(section.name, section.start, section.end)
            for section in parse.sections] == [
        ("intro", 0.0, 19.3873291063),
        ("verse", 19.3873291063, 52.0233019687),
        ("chorus", 52.0233019687, 148.054524769),
    ]


def test_a_tab_separated_segments_line_is_refused():
    try:
        harmonix_convert.parse_segments(text_of(["0.0\tintro"]), 10.0)
    except harmonix_convert.ConvertError as error:
        assert "1 space-separated fields" in str(error)
    else:
        raise AssertionError("a tab-delimited line parsed")


def test_crlf_and_lf_input_parse_identically():
    assert (harmonix_convert.parse_segments(text_of(SEGMENTS, "\n"), 152.0)
            == harmonix_convert.parse_segments(text_of(SEGMENTS), 152.0))


def test_no_label_carries_a_carriage_return():
    parse = harmonix_convert.parse_segments(text_of(SEGMENTS), 152.0)

    assert all("\r" not in section.name for section in parse.sections)


def test_the_end_sentinel_terminates_the_last_segment_and_is_not_a_section():
    parse = harmonix_convert.parse_segments(text_of(SEGMENTS), 152.0)

    assert [section.name for section in parse.sections] == [
        "intro", "verse", "chorus"]
    assert parse.sections[-1].end == 148.054524769
    assert parse.end_source == harmonix_convert.END_FROM_SENTINEL
    assert parse.duplicate_end is False


def test_a_duplicated_end_row_is_redundant_and_the_first_one_terminates():
    parse = harmonix_convert.parse_segments(text_of([
        "0.0 intro", "138.481749 chorus", "153.492761 end",
        "157.241784438 end"]), 154.392)

    assert [(section.name, section.end) for section in parse.sections] == [
        ("intro", 138.481749), ("chorus", 153.492761)]
    assert parse.duplicate_end is True
    assert parse.end_source == harmonix_convert.END_FROM_SENTINEL


def test_a_missing_end_row_closes_the_final_segment_on_the_duration():
    parse = harmonix_convert.parse_segments(text_of([
        "0.0 intro", "223.050684 outro"]), 246.004)

    assert parse.sections[-1] == ("outro", 223.050684, 246.004)
    assert parse.end_source == harmonix_convert.END_FROM_DURATION


def test_a_duration_before_the_final_start_is_clamped_never_reversed():
    parse = harmonix_convert.parse_segments(text_of([
        "0.0 intro", "247.427962 instrumental"]), 100.0)

    assert parse.sections[-1].end > parse.sections[-1].start


def test_adjacent_identical_labels_are_stored_unmerged_and_counted():
    parse = harmonix_convert.parse_segments(text_of([
        "0.0 chorus", "10.0 chorus", "20.0 verse", "30.0 verse",
        "40.0 end"]), 45.0)

    assert [section.name for section in parse.sections] == [
        "chorus", "chorus", "verse", "verse"]
    assert harmonix_convert.adjacent_identical(parse.sections) == 2


def test_silence_is_a_real_label_and_stays_a_section():
    parse = harmonix_convert.parse_segments(text_of([
        "0.0 silence", "5.0 intro", "40.0 end"]), 45.0)

    assert parse.sections[0] == ("silence", 0.0, 5.0)


def test_a_non_advancing_segment_time_is_refused():
    try:
        harmonix_convert.parse_segments(text_of([
            "10.0 intro", "10.0 verse", "20.0 end"]), 30.0)
    except harmonix_convert.ConvertError as error:
        assert "does not advance" in str(error)
    else:
        raise AssertionError("a repeated boundary time parsed")


def test_a_file_of_nothing_but_a_sentinel_is_refused():
    try:
        harmonix_convert.parse_segments(text_of(["0.0 end"]), 30.0)
    except harmonix_convert.ConvertError as error:
        assert "no section" in str(error)
    else:
        raise AssertionError("a sentinel-only file parsed")


# --------------------------------------------------------------------------- #
# The beats parse
# --------------------------------------------------------------------------- #


def test_beats_are_tab_separated_and_the_bar_number_is_dropped():
    assert harmonix_convert.parse_beats(text_of(BEATS)) == [
        (0.0, 1), (0.530973, 2), (2.123892, 1)]


def test_a_space_separated_beats_line_is_refused():
    try:
        harmonix_convert.parse_beats(text_of(["0.0 1 1"]))
    except harmonix_convert.ConvertError as error:
        assert "tab-separated" in str(error)
    else:
        raise AssertionError("a space-delimited beat line parsed")


def test_a_bar_position_of_six_is_carried_verbatim():
    assert harmonix_convert.parse_beats(text_of(ODD_BEATS))[-1] == (7.3108855, 6)


# --------------------------------------------------------------------------- #
# The emitted record
# --------------------------------------------------------------------------- #


def test_a_converted_track_emits_a_record_that_validates(tmp_path):
    rig = make_rig(tmp_path)
    corpus = rig[1]

    assert run(rig) == 0

    record = load_record(corpus)
    tp_record.validate_record(record)
    assert record["schema"] == tp_record.RECORD_SCHEMA
    assert record["source"] == "harmonix"
    assert record["id"] == TRACK
    assert record["native_id"] == NATIVE
    assert record["artist"] == "Ciara"
    assert record["release"] == "Goodies"
    assert record["genre"] == "R&B"
    assert record["audio"] == f"{TRACK}.mp3"
    assert record["duration"] == 152.0
    assert record["label_vocabulary"] == "harmonix_function"
    assert record["beats"] == "harmonix_expert"
    assert record["masked"] == []
    assert record["sections"][0] == {"name": "intro", "start": 0.0,
                                     "end": 19.3873291063}
    assert len(record["sections"]) == 3


def test_a_quoted_title_carrying_a_comma_survives_the_csv_reader(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    assert load_record(rig[1])["title"] == "1, 2 Step"


def test_the_provenance_names_its_sources_and_the_alignment_score(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    provenance = load_record(rig[1])["provenance"]
    assert provenance["converter"] == "harmonix_convert"
    assert provenance["segments_file"] == f"{NATIVE}.txt"
    assert provenance["beats_file"] == f"{NATIVE}.txt"
    assert provenance["metadata_file"] == "metadata.csv"
    assert len(provenance["segments_sha256"]) == 64
    assert len(provenance["beats_sha256"]) == 64
    assert provenance["bpm"] == 113.0
    assert provenance["time_signature"] == "4|4"
    assert provenance["ratio_bars_in_4"] == 100.0
    assert provenance["dtw_score"] == 0.9984119782214156
    assert provenance["youtube_id"] == "iBHNgV6_zn0"
    assert provenance["final_segment_end_source"] == "end_sentinel"
    assert provenance["license"].startswith("annotations MIT")
    assert provenance["converted_utc"].endswith("Z")


def test_a_track_without_an_end_row_records_where_its_final_end_came_from(
        tmp_path):
    rig = make_rig(tmp_path, segments=["0.0 intro", "100.0 outro"])
    corpus = rig[1]

    assert run(rig) == 0

    record = load_record(corpus)
    assert record["sections"][-1]["end"] == 152.0
    assert record["provenance"]["final_segment_end_source"] == (
        "metadata_duration")
    assert load_report(corpus)["no_end_sentinel_ids"] == [TRACK]


def test_the_converter_copies_no_audio_and_only_names_it(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    assert load_record(rig[1])["audio"] == f"{TRACK}.mp3"
    assert not (rig[1] / "third_party" / "audio").exists()


# --------------------------------------------------------------------------- #
# The beat grid
# --------------------------------------------------------------------------- #


def test_the_beat_grid_is_the_published_three_column_format(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    assert beat_grid_rows(rig[1]) == [
        ["time", "downbeat", "section"],
        ["0", "1", "intro"],
        ["0.531", "2", "intro"],
        ["2.1239", "1", "intro"],
    ]


def test_the_grid_carries_the_start_and_end_sentinels_around_the_labels(
        tmp_path):
    rig = make_rig(tmp_path,
                   segments=["5.0 intro", "10.0 chorus", "20.0 end"],
                   beats=["1.0\t1\t1", "7.0\t2\t1", "30.0\t3\t1"])
    run(rig)

    assert beat_grid_rows(rig[1])[1:] == [
        ["1", "1", "start"],
        ["7", "2", "intro"],
        ["30", "3", "end"],
    ]


def test_a_bar_position_of_six_reaches_the_grid_and_is_reported(tmp_path):
    rig = make_rig(tmp_path, beats=ODD_BEATS)
    corpus = rig[1]
    run(rig)

    assert beat_grid_rows(corpus)[4] == ["7.3109", "6", "intro"]
    report = load_report(corpus)
    assert report["odd_meter_count"] == 1
    assert report["odd_meter_tracks"] == [
        {"id": TRACK, "max_position": 6}]


def test_a_four_four_track_is_not_counted_as_odd_meter(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    report = load_report(rig[1])
    assert report["odd_meter_count"] == 0
    assert report["odd_meter_tracks"] == []


def test_a_track_opening_mid_bar_keeps_its_pickup_and_is_reported(tmp_path):
    rig = make_rig(tmp_path, beats=["0.0\t2\t1", "0.530973\t3\t1",
                                    "2.123892\t4\t1"])
    corpus = rig[1]
    run(rig)

    assert beat_grid_rows(corpus)[1] == ["0", "2", "intro"]
    report = load_report(corpus)
    assert report["mid_bar_start_count"] == 1
    assert report["mid_bar_start_tracks"] == [
        {"id": TRACK, "first_position": 2}]


def test_a_track_opening_on_a_downbeat_is_not_counted_as_mid_bar(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    report = load_report(rig[1])
    assert report["mid_bar_start_count"] == 0
    assert report["mid_bar_start_tracks"] == []


def test_an_existing_beat_grid_is_kept(tmp_path):
    rig = make_rig(tmp_path)
    grid = rig[1] / "annotations" / "beats" / f"{TRACK}.beat.csv"
    grid.parent.mkdir(parents=True, exist_ok=True)
    grid.write_text("time,downbeat,section\n0.25,1,intro\n", encoding="utf-8")

    run(rig)

    assert grid.read_text(encoding="utf-8") == (
        "time,downbeat,section\n0.25,1,intro\n")


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_the_report_counts_the_label_vocabulary_and_the_seconds(tmp_path):
    rig = make_rig(tmp_path, segments=[
        "0.0 intro", "10.0 chorus", "20.0 chorus", "30.0 silence",
        "40.0 end"])
    run(rig)

    report = load_report(rig[1])
    assert report["converted"] == 1
    assert report["failed"] == 0
    assert report["labels"] == {"chorus": 2, "intro": 1, "silence": 1}
    assert report["label_count"] == 3
    assert report["adjacent_identical_pairs"] == 1
    assert report["tracks_with_adjacent_identical"] == 1
    assert report["labelled_sec"] == 40.0
    assert report["audio_sec"] == 152.0


def test_the_report_carries_the_dtw_score_distribution(tmp_path):
    rig = make_rig(tmp_path)
    run(rig)

    report = load_report(rig[1])
    assert report["dtw_score"]["count"] == 1
    assert report["dtw_score"]["median"] == 0.998412
    assert report["dtw_score_missing"] == 0


def test_a_duplicated_end_track_is_named_in_the_report(tmp_path):
    rig = make_rig(tmp_path, segments=[
        "0.0 intro", "10.0 chorus", "20.0 end", "25.0 end"])
    run(rig)

    assert load_report(rig[1])["duplicate_end_ids"] == [TRACK]


def test_a_broken_track_fails_alone_and_the_batch_continues(tmp_path):
    rig = make_rig(tmp_path, natives=(NATIVE, "0003_6foot7foot"))
    harmonixset, corpus = rig
    write_segments(harmonix_convert.dataset_dir(harmonixset), NATIVE,
                   ["0.0\tintro", "10.0\tend"])

    assert run(rig) == 1

    report = load_report(corpus)
    assert report["converted"] == 1
    assert report["failed"] == 1
    assert report["failures"][0]["id"] == TRACK
    assert "space-separated" in report["failures"][0]["reason"]
    assert not (corpus / "annotations" / f"{TRACK}.harmonix.json").exists()
    assert (corpus / "annotations"
            / "hx-0003_6foot7foot.harmonix.json").exists()


def test_an_os_error_on_one_track_is_recorded_and_the_batch_continues(
        tmp_path, monkeypatch):
    rig = make_rig(tmp_path, natives=(NATIVE, "0003_6foot7foot"))
    corpus = rig[1]
    real = harmonix_convert.sha256

    def flaky(path):
        if NATIVE in path.name:
            raise OSError(28, "No space left on device", str(path))
        return real(path)

    monkeypatch.setattr(harmonix_convert, "sha256", flaky)

    assert run(rig) == 1

    report = load_report(corpus)
    assert report["converted"] == 1
    assert report["failed"] == 1
    assert report["failures"][0]["id"] == TRACK
    assert "No space left on device" in report["failures"][0]["reason"]


def test_a_missing_segments_file_is_a_bucket_not_a_failure(tmp_path):
    rig = make_rig(tmp_path, natives=(NATIVE, "0003_6foot7foot"))
    harmonixset, corpus = rig
    harmonix_convert.segments_path(harmonix_convert.dataset_dir(harmonixset),
                                   "0003_6foot7foot").unlink()

    assert run(rig) == 0

    report = load_report(corpus)
    assert report["converted"] == 1
    assert report["no_annotation_ids"] == ["0003_6foot7foot"]


def test_an_id_absent_from_the_metadata_is_reported_by_name(tmp_path):
    rig = make_rig(tmp_path)

    assert run(rig, ["--ids", NATIVE, "9999_nosuchtrack"]) == 0

    report = load_report(rig[1], harmonix_convert.PARTIAL_REPORT_FILE)
    assert report["converted"] == 1
    assert report["not_in_metadata"] == ["9999_nosuchtrack"]


def test_a_missing_annotation_does_not_make_a_whole_set_run_partial(tmp_path):
    rig = make_rig(tmp_path, natives=(NATIVE, "0003_6foot7foot"))
    harmonixset, corpus = rig
    harmonix_convert.segments_path(harmonix_convert.dataset_dir(harmonixset),
                                   "0003_6foot7foot").unlink()

    run(rig)

    report = load_report(corpus)
    assert report["partial"] is False
    assert report["no_annotation_ids"] == ["0003_6foot7foot"]


# --------------------------------------------------------------------------- #
# Selection, partial reports and idempotency
# --------------------------------------------------------------------------- #


def test_require_audio_converts_only_the_tracks_whose_mp3_has_landed(tmp_path):
    rig = make_rig(tmp_path, natives=(NATIVE, "0003_6foot7foot"))
    corpus = rig[1]
    place_audio(corpus, NATIVE)

    assert run(rig, ["--require-audio"]) == 0

    report = load_report(corpus, harmonix_convert.PARTIAL_REPORT_FILE)
    assert report["converted"] == 1
    assert report["no_audio_ids"] == ["0003_6foot7foot"]
    assert (corpus / "annotations" / f"{TRACK}.harmonix.json").exists()
    assert not (corpus / "annotations"
                / "hx-0003_6foot7foot.harmonix.json").exists()


def test_without_require_audio_a_track_with_no_mp3_still_converts(tmp_path):
    rig = make_rig(tmp_path)

    assert run(rig) == 0

    assert load_report(rig[1])["converted"] == 1


def test_limit_converts_only_the_first_n_tracks(tmp_path):
    rig = make_rig(tmp_path, natives=(NATIVE, "0003_6foot7foot",
                                      "0004_abc"))
    corpus = rig[1]

    assert run(rig, ["--limit", "2"]) == 0

    report = load_report(corpus, harmonix_convert.PARTIAL_REPORT_FILE)
    assert report["tracks_requested"] == 2
    assert report["converted"] == 2
    assert not (corpus / "annotations" / "hx-0004_abc.harmonix.json").exists()


def test_a_restricted_run_writes_the_partial_report_and_spares_the_full_one(
        tmp_path):
    rig = make_rig(tmp_path, natives=(NATIVE, "0003_6foot7foot"))
    corpus = rig[1]
    full = tp_record.third_party_dir(corpus) / harmonix_convert.REPORT_FILE
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text('{"converted": 912}\n', encoding="utf-8")

    assert run(rig, ["--ids", NATIVE]) == 0

    partial = load_report(corpus, harmonix_convert.PARTIAL_REPORT_FILE)
    assert partial["partial"] is True
    assert partial["report_file"] == harmonix_convert.PARTIAL_REPORT_FILE
    assert partial["converted"] == 1
    assert full.read_text(encoding="utf-8") == '{"converted": 912}\n'


def test_only_an_unrestricted_run_writes_the_full_report_name():
    assert harmonix_convert.report_file(None, None, False) == (
        harmonix_convert.REPORT_FILE)
    assert harmonix_convert.report_file(["0001_12step"], None, False) == (
        harmonix_convert.PARTIAL_REPORT_FILE)
    assert harmonix_convert.report_file(None, 25, False) == (
        harmonix_convert.PARTIAL_REPORT_FILE)
    assert harmonix_convert.report_file(None, None, True) == (
        harmonix_convert.PARTIAL_REPORT_FILE)


def test_a_second_conversion_leaves_the_record_byte_identical(
        tmp_path, monkeypatch):
    rig = make_rig(tmp_path)
    corpus = rig[1]
    monkeypatch.setattr(harmonix_convert, "_now_utc",
                        lambda: "2026-01-01T00:00:00Z")
    run(rig)
    path = corpus / "annotations" / f"{TRACK}.harmonix.json"
    before = path.read_bytes()

    monkeypatch.setattr(harmonix_convert, "_now_utc",
                        lambda: "2027-02-03T04:05:06Z")
    assert run(rig) == 0

    assert path.read_bytes() == before
    assert json.loads(before)["provenance"]["converted_utc"] == (
        "2026-01-01T00:00:00Z")


def test_a_changed_record_takes_the_fresh_timestamp(tmp_path, monkeypatch):
    rig = make_rig(tmp_path)
    harmonixset, corpus = rig
    monkeypatch.setattr(harmonix_convert, "_now_utc",
                        lambda: "2026-01-01T00:00:00Z")
    run(rig)

    write_metadata(harmonix_convert.dataset_dir(harmonixset),
                   [metadata_row(NATIVE, title="Renamed")])
    monkeypatch.setattr(harmonix_convert, "_now_utc",
                        lambda: "2027-02-03T04:05:06Z")
    run(rig)

    record = load_record(corpus)
    assert record["title"] == "Renamed"
    assert record["provenance"]["converted_utc"] == "2027-02-03T04:05:06Z"


def test_conversion_is_idempotent(tmp_path):
    rig = make_rig(tmp_path)
    corpus = rig[1]

    assert run(rig) == 0
    first = load_report(corpus)
    before = snapshot(corpus)

    assert run(rig) == 0
    second = load_report(corpus)

    assert snapshot(corpus) == before
    for key in ("converted", "failed", "labels", "labelled_sec", "audio_sec",
                "adjacent_identical_pairs", "odd_meter_tracks",
                "mid_bar_start_tracks", "no_end_sentinel_ids"):
        assert first[key] == second[key]
