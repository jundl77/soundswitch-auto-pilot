"""Tests for the SALAMI converter (``training/third_party/salami_convert.py``).

Every fixture line here is copied verbatim out of a real raw annotator file,
because the format's traps are all in what a line is allowed to carry at once:
one ``TIME<TAB>LABEL`` row interleaves the large-scale letter, the small-scale
letter, the function word and a parenthesised instrumentation annotation that
opens on one row and closes on a later one, in no fixed order.  A reader that
takes the first token gets a letter where a function belongs; one that reads
SALAMI's own parsed layer instead gets ``no_function`` written over the ``V'``
the annotator actually wrote, which is the whole reason the raw file is the
input.  Rows are start-only, so an end is the next boundary's start; ``Silence``
and ``End`` are sentinels, not sections; and SONG_DURATION is whole seconds, so
the annotation routinely ends after the duration the metadata claims.
"""
import json
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
for _path in (str(TRAINING_DIR), str(TRAINING_DIR / "raveform"),
              str(TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import salami_convert  # noqa: E402
import tp_record  # noqa: E402

# annotations/10/textfile1.txt, verbatim: a Silence sentinel, a row carrying all
# three layers plus an opening instrument paren, rows carrying only the
# small-scale letter, the closing paren riding a label, and the End sentinel.
SALAMI_10 = [
    "0.000000000\tSilence",
    "0.223492063\tA, a, Intro",
    "13.529160997\ta'",
    "26.809024943\tb",
    "53.162222222\tB, c, Verse, (voice",
    "66.339727891\tc",
    "79.588526077\td, voice)",
    "92.820453514\tC, b', Interlude",
    "314.269342403\tEnd",
]

# annotations/332/textfile1.txt, verbatim: the V' variant the parsed functions
# layer overwrites, and the two W/W' rows it overwrites with no_function.
SALAMI_332 = [
    "0.000000000\tSilence",
    "0.224943310\tI, Intro, (voice, b",
    "16.250362811\tV, Verse, c",
    "32.298027210\tW, d",
    "48.296802721\tV, Verse, e",
    "120.286349206\tV', Bridge, (rap, j",
    "232.300340136\tV', Outro, (rap, o",
    "264.271655328\tFade-out, rap), o",
    "285.174036281\tSilence",
    "287.129115646\tEnd",
]

METADATA_HEADER = (
    "SONG_ID,SOURCE,ANNOTATOR1,ANNOTATOR2,FILE_LOCATION,SONG_DURATION,EMPTY,"
    "SONG_TITLE,ARTIST,FORMAT,ANNOTATION_TIME1,ANNOTATION_TIME2,TEXTFILE1,"
    "TEXTFILE2,CLASS,GENRE,SUBMISSION_DATE1,SUBMISSION_DATE2,"
    "SONG_WAS_PRIVATE_FLAG,SONG_WAS_DISCARDED_FLAG,XEQS1,XEQS2")

# The 22 entries whose two sides are the same word; the rest of the shipped
# dictionary is semantic (hook -> Chorus) and must never be applied.
VOCAB_LINES = [
    "?\t",
    "A a\tA, a",
    "Bridge\tBridge",
    "Chorus\tChorus",
    "End\tEnd",
    "Fade-out\tFade-out",
    "Intro\tIntro",
    "Interlude\tInterlude",
    "Outro\tOutro",
    "Silence\tSilence",
    "Solo\tSolo",
    "Verse\tVerse",
    "bridge\tBridge",
    "chorus\tChorus",
    "end\tEnd",
    "fade-out\tFade-out",
    "hook\tChorus",
    "interlude\tInterlude",
    "intro\tIntro",
    "outro\tOutro",
    "silence\tSilence",
    "solo\tSolo",
    "verse\tVerse",
    "voice\tvoice",
]


def metadata_row(salami_id, duration="330", title="For_God_And_Country",
                 artist="The_Smashing_Pumpkins", song_class="popular",
                 genre="Alternative_Pop___Rock"):
    return ",".join([
        str(salami_id), "Codaich", "5", "8", "/srv/salami/data", duration, "",
        title, artist, "mp3", "37", "45", "textfile1.txt", "textfile2.txt",
        song_class, genre, "2010-06-14", "2010-06-04", "0", "FALSE", "X", "X"])


def write_metadata(salami, rows):
    path = salami_convert.metadata_path(salami)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([METADATA_HEADER, *rows]) + "\n",
                    encoding="utf-8")
    return path


def write_vocab(salami, lines=None):
    path = salami_convert.vocab_path(salami)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines if lines is not None else VOCAB_LINES)
                    + "\n", encoding="utf-8")
    return path


def write_textfile(salami, salami_id, annotator, lines,
                   terminator="\r\n"):
    path = salami_convert.textfile_path(salami, str(salami_id), annotator)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((terminator.join(lines) + terminator).encode("utf-8"))
    return path


def write_audio(salami, salami_id, payload=b"ID3fake"):
    path = salami_convert.source_audio_path(salami, str(salami_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def make_rig(tmp_path, lines=None, salami_id=10, duration="330",
             song_class="popular", second=None, audio=True):
    salami = tmp_path / "salami-data-public"
    corpus = tmp_path / "corpus"
    (corpus / "annotations").mkdir(parents=True)

    write_metadata(salami, [metadata_row(salami_id, duration=duration,
                                         song_class=song_class)])
    write_vocab(salami)
    write_textfile(salami, salami_id, 1,
                   lines if lines is not None else SALAMI_10)
    if second is not None:
        write_textfile(salami, salami_id, 2, second)
    if audio:
        write_audio(salami, salami_id)
    return salami, corpus


def run(rig, extra=()):
    salami, corpus = rig
    return salami_convert.main([
        "--salami-dir", str(salami), "--data-dir", str(corpus),
        "--no-audio", *extra])


def load_record(corpus, salami_id=10):
    path = tp_record.annotation_path(
        corpus, salami_convert.track_id(str(salami_id)), "salami")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_report(corpus, name=salami_convert.REPORT_FILE):
    with open(tp_record.third_party_dir(corpus) / name, "r",
              encoding="utf-8") as handle:
        return json.load(handle)


def parse(lines, vocab):
    return salami_convert.parse_annotation("\r\n".join(lines) + "\r\n", vocab)


def snapshot(corpus):
    files = {}
    for path in sorted(corpus.rglob("*")):
        if not path.is_file() or path.name in (
                salami_convert.REPORT_FILE,
                salami_convert.PARTIAL_REPORT_FILE):
            continue
        name = path.relative_to(corpus).as_posix()
        if name.endswith(".salami.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["provenance"].pop("converted_utc")
            files[name] = json.dumps(record, sort_keys=True)
        else:
            files[name] = path.read_bytes()
    return files


@pytest.fixture
def vocab(tmp_path_factory):
    root = tmp_path_factory.mktemp("vocab")
    write_vocab(root)
    return salami_convert.load_case_map(salami_convert.vocab_path(root))


# --------------------------------------------------------------------------- #
# The raw format
# --------------------------------------------------------------------------- #


def test_a_row_splits_into_its_three_layers_and_drops_the_instrumentation(
        vocab):
    rows = salami_convert.parse_rows(
        "53.162222222\tB, c, Verse, (voice\r\n79.588526077\td, voice)\r\n",
        vocab)

    assert rows[0] == salami_convert.Row(53.162222222, ("B",), ("c",),
                                         ("Verse",))
    assert rows[1] == salami_convert.Row(79.588526077, (), ("d",), ())


def test_a_section_ends_where_the_next_boundary_starts_not_at_its_own_row(
        vocab):
    result = parse(SALAMI_10, vocab)

    assert [(s["name"], s["start"], s["end"]) for s in result.sections] == [
        ("Intro", 0.223492063, 53.162222222),
        ("Verse", 53.162222222, 92.820453514),
        ("Interlude", 92.820453514, 314.269342403),
    ]


def test_a_small_scale_only_row_is_not_a_boundary(vocab):
    result = parse(SALAMI_10, vocab)

    starts = [section["start"] for section in result.sections]
    assert 13.529160997 not in starts
    assert 26.809024943 not in starts


def test_the_small_scale_letters_a_row_carries_are_kept_beside_the_section(
        vocab):
    result = parse(SALAMI_10, vocab)

    assert result.sections[0]["small_scale"] == "a"
    assert result.sections[2]["small_scale"] == "b'"


def test_a_track_with_no_musical_function_anywhere_fails_rather_than_guessing(
        vocab):
    # annotations/236/textfile1.txt: letters throughout, not one function word.
    with pytest.raises(salami_convert.ConvertError, match="no section"):
        parse(["0.000000000\tSilence",
               "0.114285714\ta, A, (violin",
               "46.161269841\ta', A'",
               "105.371428571\tb'', violin)",
               "145.038616780\tEnd"], vocab)


def test_a_line_that_is_not_time_tab_label_is_refused_by_line_number(vocab):
    with pytest.raises(salami_convert.ConvertError, match="line 2"):
        parse(["0.0\tSilence", "13.5 A, a, Intro", "20.0\tEnd"], vocab)


def test_a_non_numeric_time_is_refused_by_line_number(vocab):
    with pytest.raises(salami_convert.ConvertError, match="line 2"):
        parse(["0.0\tSilence", "later\tA, a, Intro", "20.0\tEnd"], vocab)


def test_rows_out_of_time_order_are_refused_rather_than_sorted(vocab):
    with pytest.raises(salami_convert.ConvertError, match="unsorted"):
        parse(["0.0\tSilence", "30.0\tA, a, Intro", "20.0\tB, b, Verse",
               "40.0\tEnd"], vocab)


def test_two_rows_at_the_same_instant_produce_no_zero_length_span(vocab):
    # annotations/1003/textfile1.txt opens with Z and Silence both at 0.0.
    result = parse(["0.000000000\tZ",
                    "0.000000000\tSilence",
                    "43.560634920\tA, a, Intro",
                    "66.992426000\tB, b, Verse",
                    "90.000000000\tEnd"], vocab)

    for span in list(result.sections) + list(result.masked):
        assert span["end"] > span["start"]


# --------------------------------------------------------------------------- #
# Masking: no_function and the sentinels
# --------------------------------------------------------------------------- #


def test_a_boundary_with_no_function_word_is_masked_and_is_not_a_section(
        vocab):
    result = parse(SALAMI_332, vocab)

    assert 32.298027210 not in [section["start"] for section in result.sections]
    masked = [span for span in result.masked if span["start"] == 32.29802721]
    assert masked[0]["reason"] == salami_convert.NO_FUNCTION
    assert masked[0]["end"] == 48.296802721


def test_the_letter_the_parsed_layer_overwrites_with_no_function_survives(
        vocab):
    result = parse(SALAMI_332, vocab)

    masked = [span for span in result.masked
              if span["reason"] == salami_convert.NO_FUNCTION]
    assert [span["large_scale"] for span in masked] == ["W"]


def test_a_v_prime_variant_survives_verbatim_beside_its_function(vocab):
    result = parse(SALAMI_332, vocab)

    variants = [(section["name"], section["large_scale"])
                for section in result.sections
                if section.get("large_scale", "").endswith("'")]
    assert variants == [("Bridge", "V'"), ("Outro", "V'")]


def test_silence_is_masked_as_non_musical_rather_than_stored_as_a_section(
        vocab):
    result = parse(SALAMI_332, vocab)

    assert "Silence" not in [section["name"] for section in result.sections]
    silence = [span for span in result.masked
               if span.get("function") == "Silence"]
    assert [(span["start"], span["end"], span["reason"]) for span in silence] \
        == [(0.0, 0.22494331, "non_musical"),
            (285.174036281, 287.129115646, "non_musical")]


def test_the_end_sentinel_closes_the_annotation_and_is_never_a_span(vocab):
    result = parse(SALAMI_10, vocab)

    assert result.last_time == 314.269342403
    assert result.sections[-1]["end"] == 314.269342403
    assert all(span["start"] < 314.269342403 for span in result.masked)


def test_masked_spans_leave_gaps_in_the_sections_and_the_gaps_stand(vocab):
    result = parse(SALAMI_332, vocab)

    assert result.sections[1]["end"] < result.sections[2]["start"]


def test_a_row_naming_two_functions_keeps_both_verbatim(vocab):
    result = parse(["0.0\tSilence",
                    "1.0\tA, a, Intro",
                    "10.0\tB, b, Solo, Interlude",
                    "20.0\tEnd"], vocab)

    assert result.sections[1]["name"] == "Solo, Interlude"


# --------------------------------------------------------------------------- #
# The case-only vocabulary transform
# --------------------------------------------------------------------------- #


def test_the_case_map_takes_only_the_entries_whose_two_sides_are_one_word(
        vocab):
    assert vocab["case"] == {
        "bridge": "Bridge", "chorus": "Chorus", "end": "End",
        "fade-out": "Fade-out", "interlude": "Interlude", "intro": "Intro",
        "outro": "Outro", "silence": "Silence", "solo": "Solo",
        "verse": "Verse"}


def test_a_semantic_dictionary_entry_is_never_applied(vocab):
    assert salami_convert.normalise("hook", vocab) == "hook"


def test_a_lowercase_function_word_is_recapitalised_to_the_dictionary_spelling(
        vocab):
    result = parse(["0.0\tsilence",
                    "1.0\tA, a, intro",
                    "10.0\tB, b, outro",
                    "20.0\tEnd"], vocab)

    assert [section["name"] for section in result.sections] == ["Intro",
                                                                "Outro"]
    assert result.masked[0]["function"] == "Silence"


def test_a_lowercase_letter_that_is_also_a_dictionary_word_stays_a_function(
        vocab):
    rows = salami_convert.parse_rows("1.0\tA, a, verse\r\n", vocab)

    assert rows[0].function == ("Verse",)
    assert rows[0].small == ("a",)


# --------------------------------------------------------------------------- #
# The annotator choice
# --------------------------------------------------------------------------- #


def test_textfile1_wins_when_both_annotators_parse(tmp_path):
    rig = make_rig(tmp_path, second=SALAMI_332)

    assert run(rig) == 0
    record = load_record(rig[1])
    assert record["provenance"]["annotator"] == 1
    assert record["provenance"]["annotation_file"] == "textfile1.txt"
    assert record["provenance"]["annotators_available"] == [1, 2]


def test_textfile2_is_the_fallback_when_textfile1_does_not_parse(tmp_path):
    rig = make_rig(tmp_path, lines=["0.0\tSilence", "not a row at all"],
                   second=SALAMI_332)

    assert run(rig) == 0
    record = load_record(rig[1])
    assert record["provenance"]["annotator"] == 2
    assert record["provenance"]["annotation_file"] == "textfile2.txt"


def test_both_annotators_failing_is_one_failure_naming_both_files(tmp_path):
    rig = make_rig(tmp_path, lines=["0.0\tnope"], second=["0.0\talso nope"])

    assert run(rig) == 1
    report = load_report(rig[1])
    assert report["failures"][0]["id"] == "salami-10"
    assert "textfile1.txt" in report["failures"][0]["reason"]
    assert "textfile2.txt" in report["failures"][0]["reason"]


def test_the_report_counts_the_tracks_that_offered_two_annotators(tmp_path):
    rig = make_rig(tmp_path, second=SALAMI_332)

    run(rig)
    report = load_report(rig[1])
    assert report["tracks_with_two_annotators"] == 1
    assert report["tracks_with_two_annotators_ids"] == ["10"]
    assert report["annotator2_used"] == []


# --------------------------------------------------------------------------- #
# The Live_Music_Archive exclusion
# --------------------------------------------------------------------------- #


def test_a_live_music_archive_track_is_excluded_by_default(tmp_path):
    rig = make_rig(tmp_path, song_class=salami_convert.LIVE_ARCHIVE_CLASS)

    assert run(rig) == 0
    report = load_report(rig[1])
    assert report["converted"] == 0
    assert report["excluded_live_music_archive"] == 1
    assert report["excluded_live_music_archive_ids"] == ["10"]
    assert not tp_record.annotation_path(rig[1], "salami-10",
                                         "salami").exists()


def test_the_include_flag_converts_a_live_music_archive_track(tmp_path):
    rig = make_rig(tmp_path, song_class=salami_convert.LIVE_ARCHIVE_CLASS)

    assert run(rig, ["--include-live-archive"]) == 0
    report = load_report(rig[1])
    assert report["converted"] == 1
    assert report["excluded_live_music_archive"] == 0
    assert load_record(rig[1])["salami_class"] == \
        salami_convert.LIVE_ARCHIVE_CLASS


# --------------------------------------------------------------------------- #
# Duration
# --------------------------------------------------------------------------- #


def test_a_metadata_duration_covering_the_annotation_is_used_as_is(tmp_path):
    rig = make_rig(tmp_path, duration="330")

    run(rig)
    record = load_record(rig[1])
    assert record["duration"] == 330.0
    assert record["provenance"]["duration_source"] == "metadata"


def test_a_blank_song_duration_falls_back_to_the_last_annotated_time(tmp_path):
    rig = make_rig(tmp_path, duration="")

    assert run(rig) == 0
    record = load_record(rig[1])
    assert record["duration"] == 314.269342403
    assert record["provenance"]["duration_source"] == "annotation_last_time"


def test_a_whole_second_duration_the_annotation_runs_past_is_extended(
        tmp_path):
    rig = make_rig(tmp_path, duration="314")

    run(rig)
    record = load_record(rig[1])
    assert record["duration"] == 314.269342403
    assert record["provenance"]["duration_source"] == \
        "annotation_last_time_over_metadata"


def test_a_non_numeric_song_duration_fails_that_track_by_name(tmp_path):
    rig = make_rig(tmp_path, duration="about six minutes")

    assert run(rig) == 1
    report = load_report(rig[1])
    assert report["failures"][0]["id"] == "salami-10"
    assert "SONG_DURATION" in report["failures"][0]["reason"]


# --------------------------------------------------------------------------- #
# The record, the beats and the audio
# --------------------------------------------------------------------------- #


def test_the_emitted_record_passes_the_third_party_validator(tmp_path):
    rig = make_rig(tmp_path, lines=SALAMI_332, duration="330")

    assert run(rig) == 0
    tp_record.validate_record(load_record(rig[1]))


def test_the_record_reads_back_through_the_shared_reader(tmp_path):
    rig = make_rig(tmp_path)

    run(rig)
    records = tp_record.load_records(rig[1], "salami")
    assert [record["id"] for record in records] == ["salami-10"]


def test_the_track_id_is_the_lowercase_prefixed_salami_id(tmp_path):
    rig = make_rig(tmp_path, salami_id=1000, duration="")
    write_textfile(rig[0], 1000, 1, SALAMI_10)
    write_audio(rig[0], 1000)

    assert salami_convert.track_id("1000") == "salami-1000"
    assert run(rig) == 0
    assert load_record(rig[1], 1000)["native_id"] == "1000"


def test_no_beat_csv_is_written_and_the_record_says_the_grid_is_pending(
        tmp_path):
    rig = make_rig(tmp_path)

    run(rig)
    record = load_record(rig[1])
    assert record["beats"] == "madmom_offline_pending"
    assert not tp_record.beat_csv_path(rig[1], record).exists()
    assert not (rig[1] / "annotations" / "beats").exists()


def test_the_audio_is_copied_under_the_track_id_when_copying_is_asked_for(
        tmp_path):
    salami, corpus = make_rig(tmp_path)
    write_audio(salami, 10, payload=b"ID3realbytes")

    assert salami_convert.main([
        "--salami-dir", str(salami), "--data-dir", str(corpus)]) == 0
    record = load_record(corpus)
    assert record["audio"] == "salami-10.mp3"
    assert tp_record.audio_path(corpus, record).read_bytes() == b"ID3realbytes"


def test_no_audio_leaves_the_corpus_audio_tree_alone(tmp_path):
    rig = make_rig(tmp_path)

    run(rig)
    assert not (tp_record.third_party_dir(rig[1]) / "audio").exists()


def test_a_track_with_no_audio_on_disk_is_reported_and_not_converted(tmp_path):
    rig = make_rig(tmp_path, audio=False)

    assert run(rig) == 0
    report = load_report(rig[1])
    assert report["no_audio_ids"] == ["10"]
    assert report["converted"] == 0


def test_a_metadata_row_with_no_annotation_directory_is_reported(tmp_path):
    salami, corpus = make_rig(tmp_path)
    write_metadata(salami, [metadata_row(10), metadata_row(99)])
    write_audio(salami, 99)

    assert run((salami, corpus)) == 0
    report = load_report(corpus)
    assert report["no_annotation_ids"] == ["99"]
    assert report["converted"] == 1


# --------------------------------------------------------------------------- #
# The batch and the report
# --------------------------------------------------------------------------- #


def test_a_second_identical_run_changes_nothing_on_disk(tmp_path):
    rig = make_rig(tmp_path, second=SALAMI_332)

    run(rig)
    before = snapshot(rig[1])
    run(rig)
    assert snapshot(rig[1]) == before


def test_a_restricted_run_writes_the_partial_report_and_not_the_full_one(
        tmp_path):
    rig = make_rig(tmp_path)

    assert run(rig, ["--ids", "10"]) == 0
    report = load_report(rig[1], salami_convert.PARTIAL_REPORT_FILE)
    assert report["partial"] is True
    assert not (tp_record.third_party_dir(rig[1])
                / salami_convert.REPORT_FILE).exists()


def test_a_limited_run_is_partial_too(tmp_path):
    rig = make_rig(tmp_path)

    assert run(rig, ["--limit", "1"]) == 0
    assert load_report(rig[1],
                       salami_convert.PARTIAL_REPORT_FILE)["partial"] is True


def test_a_full_run_writes_the_full_report_marked_not_partial(tmp_path):
    rig = make_rig(tmp_path)

    assert run(rig) == 0
    assert load_report(rig[1])["partial"] is False


def test_the_report_counts_the_label_vocabulary_and_the_masked_seconds(
        tmp_path):
    rig = make_rig(tmp_path, lines=SALAMI_332, duration="330")

    run(rig)
    report = load_report(rig[1])
    assert report["labels"] == {"Bridge": 1, "Fade-out": 1, "Intro": 1,
                                "Outro": 1, "Verse": 2}
    assert report["large_scale_labels"] == {"I": 1, "V": 2, "V'": 2}
    assert report["masked_spans"] == 3
    assert report["masked_sec"] == round(
        0.22494331 + (48.296802721 - 32.29802721)
        + (287.129115646 - 285.174036281), 3)


def test_an_id_absent_from_the_metadata_is_reported_not_crashed_on(tmp_path):
    rig = make_rig(tmp_path)

    assert run(rig, ["--ids", "10", "4242"]) == 0
    assert load_report(
        rig[1], salami_convert.PARTIAL_REPORT_FILE)["not_in_metadata"] == \
        ["4242"]


def test_one_failing_track_does_not_stop_the_batch(tmp_path):
    salami, corpus = make_rig(tmp_path)
    write_metadata(salami, [metadata_row(10), metadata_row(20)])
    write_textfile(salami, 20, 1, ["0.0\tnot a row"])
    write_audio(salami, 20)

    assert run((salami, corpus)) == 1
    report = load_report(corpus)
    assert report["converted"] == 1
    assert [failure["id"] for failure in report["failures"]] == ["salami-20"]
