"""Tests for the full-corpus validator (training/raveform_validate.py).

The validator answers one question the owner asked in absolute terms: is every
single annotated track either present-and-correct on disk, or precisely recorded
as unobtainable?  A bug here does not crash -- it produces a confident report
that quietly under-counts the corpus, which is worse than no report at all.

So the parts pinned here are the ones that decide the answer:

* the five-way classification, and in particular that a track which failed once
  and succeeded later is judged by what is on disk, not by the failure log;
* the convergence predicate, which is the gate's whole verdict;
* the orphan sweep, which must see ``.mp3`` and nothing else (the audio
  directory also holds deliberate ``.npy`` decode caches);
* the annotation cross-check, which is what makes "the corpus is complete"
  mean "complete against the annotations" and not merely "1,423 files";
* the checksum baseline, which is the only integrity reference this corpus can
  ever have -- YouTube publishes no canonical hashes.
"""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import build_clean_manifest as gate  # noqa: E402  (needs the path insert above)
from raveform_validate import (  # noqa: E402
    CHECKSUMS_FILE,
    STATUS_CORRUPT,
    STATUS_DURATION_MISMATCH,
    STATUS_MISSING,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    VALIDATION_JSON,
    VALIDATION_TXT,
    Failure,
    TrackVerdict,
    annotation_durations,
    check_annotations,
    classify_track,
    convergence,
    find_orphans,
    load_failures,
    main,
    overall_verdict,
    prune_corrupt,
    render_text_report,
    sha256_file,
    tally,
    validate,
    write_checksums,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


# --------------------------------------------------------------------------- #
# Fixtures: a miniature corpus with the same file layout as the real one
# --------------------------------------------------------------------------- #


def _sine_mp3(path: Path, seconds: float) -> Path:
    """A real, fully decodable mp3 of a given length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:a", "libmp3lame", "-b:a", "64k", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _verdict(track_id, youtube_id, status, **overrides) -> TrackVerdict:
    fields = {
        "track_id": track_id,
        "youtube_id": youtube_id,
        "status": status,
        "detail": "",
        "mp3_path": "",
        "ffprobe_duration_sec": None,
        "decoded_duration_sec": None,
        "annotation_duration_sec": 300.0,
        "failure_reason": "",
        "sha256": "",
    }
    fields.update(overrides)
    return TrackVerdict(**fields)


def _write_corpus(root: Path, tracks: list) -> Path:
    """Write manifest.csv + annotations/ for ``[(track_id, yt_id, duration)]``."""
    (root / "annotations" / "beats").mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(parents=True, exist_ok=True)

    with open(root / "manifest.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("track_id", "youtube_id", "n_sections", "total_sec"))
        for track_id, youtube_id, duration in tracks:
            writer.writerow((track_id, youtube_id, 2, f"{duration:.3f}"))

    records = [
        {
            "key": track_id,
            "id": youtube_id,
            "title": track_id,
            "duration": duration,
            "sections": [
                {"name": "intro", "start": 0.0, "end": duration / 2},
                {"name": "drop", "start": duration / 2, "end": duration},
            ],
        }
        for track_id, youtube_id, duration in tracks
    ]
    with open(root / "annotations" / "segments.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle)

    for track_id, _youtube_id, duration in tracks:
        with open(
            root / "annotations" / "beats" / f"{track_id}.beat.csv",
            "w", encoding="utf-8", newline="",
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("time", "downbeat", "section"))
            writer.writerow(("0.000", "1", "intro"))
            writer.writerow((f"{duration / 2:.3f}", "1", "drop"))
    return root


# --------------------------------------------------------------------------- #
# classify_track -- the five-way decision
# --------------------------------------------------------------------------- #


def test_a_clean_decode_is_ok():
    status, _detail = classify_track(gate.STATUS_OK, "", None)
    assert status == STATUS_OK


def test_a_clean_decode_of_the_wrong_length_is_a_duration_mismatch():
    status, detail = classify_track(gate.STATUS_MISMATCH, "decoded 200 s vs annotation 400 s", None)
    assert status == STATUS_DURATION_MISMATCH
    assert "200" in detail  # the evidence is carried through, not discarded


def test_an_undecodable_file_is_corrupt():
    status, detail = classify_track(gate.STATUS_CORRUPT, "truncated: header claims ...", None)
    assert status == STATUS_CORRUPT
    assert detail


def test_an_absent_file_with_a_recorded_failure_is_unavailable():
    failure = Failure("abc123", "unavailable", "ERROR: Video unavailable", "2026-07-26T12:00:00Z")
    status, detail = classify_track(None, "", failure)
    assert status == STATUS_UNAVAILABLE
    assert "unavailable" in detail
    assert "Video unavailable" in detail


def test_an_absent_file_with_no_recorded_failure_is_missing():
    # The dangerous case: nobody ever tried, or the attempt was lost.  It must
    # never be silently folded into "unavailable".
    status, detail = classify_track(None, "", None)
    assert status == STATUS_MISSING
    assert detail  # says why this is not the same as "unobtainable"


def test_disk_outranks_the_failure_log():
    # 12 real tracks failed once and succeeded on a later cycle; failed.jsonl is
    # append-only, so both records exist.  The file on disk is the truth.
    failure = Failure("abc123", "bot_check", "ERROR: Sign in to confirm", "2026-07-26T12:00:00Z")
    status, _detail = classify_track(gate.STATUS_OK, "", failure)
    assert status == STATUS_OK


# --------------------------------------------------------------------------- #
# load_failures
# --------------------------------------------------------------------------- #


def test_the_most_recent_failure_record_wins(tmp_path):
    path = tmp_path / "failed.jsonl"
    path.write_text(
        json.dumps({"youtube_id": "a", "reason": "bot_check", "error": "sign in", "timestamp": "t1"})
        + "\n"
        + json.dumps({"youtube_id": "a", "reason": "unavailable", "error": "gone", "timestamp": "t2"})
        + "\n",
        encoding="utf-8",
    )
    failures = load_failures(path)
    assert failures["a"].reason == "unavailable"
    assert failures["a"].error == "gone"


def test_a_torn_last_line_does_not_lose_the_earlier_failures(tmp_path):
    # failed.jsonl is written by a process that can be hard-killed.
    path = tmp_path / "failed.jsonl"
    path.write_text(
        json.dumps({"youtube_id": "a", "reason": "unavailable", "error": "gone"}) + "\n"
        + '{"youtube_id": "b", "rea',
        encoding="utf-8",
    )
    failures = load_failures(path)
    assert set(failures) == {"a"}


def test_an_absent_failure_log_is_empty_not_an_error(tmp_path):
    assert load_failures(tmp_path / "nope.jsonl") == {}


# --------------------------------------------------------------------------- #
# find_orphans -- audio/ holds more than mp3s
# --------------------------------------------------------------------------- #


def test_an_mp3_with_no_manifest_row_is_an_orphan(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "known.mp3").write_bytes(b"x")
    (audio / "stray.mp3").write_bytes(b"x")
    assert find_orphans(tmp_path, {"known"}) == ["stray"]


def test_decode_caches_beside_the_audio_are_not_orphans(tmp_path):
    # The eval set deliberately keeps *.npy decode caches in audio/.  Sweeping
    # them in would invent orphans on every run and mask the real ones.
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "known.mp3").write_bytes(b"x")
    (audio / "known.mp3.44100.npy").write_bytes(b"x")
    (audio / "known.mp3.part").write_bytes(b"x")
    assert find_orphans(tmp_path, {"known"}) == []


def test_an_absent_audio_directory_yields_no_orphans(tmp_path):
    assert find_orphans(tmp_path, {"known"}) == []


# --------------------------------------------------------------------------- #
# check_annotations -- completeness against the annotations, not the file count
# --------------------------------------------------------------------------- #


def test_a_complete_corpus_has_no_annotation_issues(tmp_path):
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 300.0), ("0002.bbb", "bbb", 400.0)])
    rows = gate.load_manifest_rows(tmp_path)
    from raveform_fetch_annotations import load_tracks

    assert check_annotations(tmp_path, rows, load_tracks(tmp_path)) == []


def test_a_manifest_row_with_no_annotation_is_reported(tmp_path):
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 300.0), ("0002.bbb", "bbb", 400.0)])
    from raveform_fetch_annotations import load_tracks

    tracks = [t for t in load_tracks(tmp_path) if t["key"] != "0002.bbb"]
    issues = check_annotations(tmp_path, gate.load_manifest_rows(tmp_path), tracks)
    assert any("0002.bbb" in issue for issue in issues)


def test_an_annotation_with_no_manifest_row_is_reported(tmp_path):
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 300.0)])
    from raveform_fetch_annotations import load_tracks

    tracks = load_tracks(tmp_path) + [
        {"key": "0009.zzz", "id": "zzz", "duration": 10.0, "sections": [
            {"name": "intro", "start": 0.0, "end": 10.0}]},
    ]
    issues = check_annotations(tmp_path, gate.load_manifest_rows(tmp_path), tracks)
    assert any("0009.zzz" in issue for issue in issues)


def test_a_youtube_id_that_disagrees_with_the_manifest_is_reported(tmp_path):
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 300.0)])
    from raveform_fetch_annotations import load_tracks

    tracks = load_tracks(tmp_path)
    tracks[0]["id"] = "different"
    issues = check_annotations(tmp_path, gate.load_manifest_rows(tmp_path), tracks)
    assert any("different" in issue for issue in issues)


def test_a_missing_beat_grid_is_reported(tmp_path):
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 300.0)])
    (tmp_path / "annotations" / "beats" / "0001.aaa.beat.csv").unlink()
    from raveform_fetch_annotations import load_tracks

    issues = check_annotations(tmp_path, gate.load_manifest_rows(tmp_path), load_tracks(tmp_path))
    assert any("beat" in issue.lower() for issue in issues)


def test_an_unparsable_beat_grid_is_reported(tmp_path):
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 300.0)])
    (tmp_path / "annotations" / "beats" / "0001.aaa.beat.csv").write_text(
        "time,downbeat,section\nnot-a-number,1,intro\n", encoding="utf-8"
    )
    from raveform_fetch_annotations import load_tracks

    issues = check_annotations(tmp_path, gate.load_manifest_rows(tmp_path), load_tracks(tmp_path))
    assert any("0001.aaa" in issue for issue in issues)


def test_a_record_with_no_key_is_reported_rather_than_crashing_the_run(tmp_path):
    # The check must survive exactly the malformed input it exists to describe:
    # asking for a keyless record's beat-grid path raises, and the exception
    # would escape before a single artifact was written -- a validation run that
    # produces no report at all, on the one corpus that most needs one.
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 300.0)])
    from raveform_fetch_annotations import load_tracks

    tracks = load_tracks(tmp_path) + [{"id": "zzz", "duration": 10.0, "sections": []}]
    issues = check_annotations(tmp_path, gate.load_manifest_rows(tmp_path), tracks)
    assert any("does not parse" in issue for issue in issues)


def test_two_records_under_one_key_are_reported(tmp_path):
    # The second silently replaces the first everywhere downstream, so the
    # corpus is a track short with nothing looking wrong.
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 300.0)])
    from raveform_fetch_annotations import load_tracks

    tracks = load_tracks(tmp_path)
    issues = check_annotations(tmp_path, gate.load_manifest_rows(tmp_path), tracks + tracks)
    assert any("duplicate" in issue for issue in issues)


def test_a_duration_that_disagrees_with_the_manifest_is_reported(tmp_path):
    # manifest total_sec is the annotation duration rounded to ms; anything
    # beyond that means the two files describe different tracks.
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 300.0)])
    from raveform_fetch_annotations import load_tracks

    tracks = load_tracks(tmp_path)
    tracks[0]["duration"] = 305.0
    issues = check_annotations(tmp_path, gate.load_manifest_rows(tmp_path), tracks)
    assert any("duration" in issue.lower() for issue in issues)


def test_millisecond_rounding_in_the_manifest_is_not_an_issue(tmp_path):
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 429.9639229025)])
    from raveform_fetch_annotations import load_tracks

    assert check_annotations(tmp_path, gate.load_manifest_rows(tmp_path), load_tracks(tmp_path)) == []


# --------------------------------------------------------------------------- #
# annotation_durations -- the reference length comes from the annotation record
# --------------------------------------------------------------------------- #


def test_durations_come_from_the_annotation_record_at_full_precision():
    tracks = [{"key": "0001.aaa", "id": "aaa", "duration": 429.9639229025, "sections": []}]
    assert annotation_durations(tracks)["0001.aaa"] == 429.9639229025


# --------------------------------------------------------------------------- #
# convergence -- the verdict
# --------------------------------------------------------------------------- #


def test_a_fully_accounted_corpus_converges():
    counts = tally([
        _verdict("0001.a", "a", STATUS_OK),
        _verdict("0002.b", "b", STATUS_DURATION_MISMATCH),
        _verdict("0003.c", "c", STATUS_UNAVAILABLE),
    ])
    converged, statement = convergence(counts, manifest_tracks=3)
    assert converged
    assert "3" in statement


def test_a_single_missing_track_blocks_convergence():
    counts = tally([
        _verdict("0001.a", "a", STATUS_OK),
        _verdict("0002.b", "b", STATUS_MISSING),
    ])
    converged, statement = convergence(counts, manifest_tracks=2)
    assert not converged
    assert "MISSING" in statement


def test_a_single_corrupt_track_blocks_convergence():
    counts = tally([
        _verdict("0001.a", "a", STATUS_OK),
        _verdict("0002.b", "b", STATUS_CORRUPT),
    ])
    converged, _statement = convergence(counts, manifest_tracks=2)
    assert not converged


def test_an_undercount_blocks_convergence_even_with_nothing_bad():
    # Every track judged fine, but the manifest has more rows than we judged:
    # the arithmetic, not the statuses, is what proves nothing was dropped.
    counts = tally([_verdict("0001.a", "a", STATUS_OK)])
    converged, _statement = convergence(counts, manifest_tracks=2)
    assert not converged


# --------------------------------------------------------------------------- #
# The duration tolerance boundary
# --------------------------------------------------------------------------- #
#
# The tolerance is max(+-10 s, +-3%) and it is the single number separating "we
# have this track" from "this is a different recording".  Exercised through the
# validator's own surface -- the gate's verdict mapped into a validator status --
# because that composition is what actually decides a track's fate.


def _status_for(decoded: float, annotation: float) -> str:
    """The validator's verdict for a file that decodes cleanly to ``decoded``."""
    # header == decoded, so the truncation check passes and only the annotation
    # comparison can be what decides.
    gate_status, detail = gate.classify("", decoded, decoded, annotation)
    return classify_track(gate_status, detail, None)[0]


def test_the_absolute_floor_admits_a_track_exactly_at_the_boundary():
    # 3% of 300 s is 9 s, so the 10 s floor governs.  Inclusive: at exactly the
    # tolerance the track is still ours.
    assert _status_for(310.0, 300.0) == STATUS_OK
    assert _status_for(290.0, 300.0) == STATUS_OK


def test_the_absolute_floor_rejects_a_track_just_past_the_boundary():
    assert _status_for(310.01, 300.0) == STATUS_DURATION_MISMATCH
    assert _status_for(289.99, 300.0) == STATUS_DURATION_MISMATCH


def test_the_relative_term_admits_a_long_edit_exactly_at_the_boundary():
    # 3% of 1000 s is 30 s, well past the floor, so the proportional term
    # governs -- a 20-minute DJ set gets proportional slack.
    assert _status_for(1030.0, 1000.0) == STATUS_OK
    assert _status_for(970.0, 1000.0) == STATUS_OK


def test_the_relative_term_rejects_a_long_edit_just_past_the_boundary():
    assert _status_for(1030.1, 1000.0) == STATUS_DURATION_MISMATCH
    assert _status_for(969.9, 1000.0) == STATUS_DURATION_MISMATCH


def test_the_two_regimes_meet_where_three_percent_equals_ten_seconds():
    # Below ~333 s the floor is wider than 3%, above it the reverse; the
    # crossover must not open a gap where neither term applies.
    assert gate.duration_tolerance(333.0) == pytest.approx(10.0)
    assert gate.duration_tolerance(400.0) == pytest.approx(12.0)
    assert _status_for(343.0, 333.0) == STATUS_OK
    assert _status_for(343.1, 333.0) == STATUS_DURATION_MISMATCH


# --------------------------------------------------------------------------- #
# overall_verdict -- convergence is not the whole story
# --------------------------------------------------------------------------- #


def test_a_converged_corpus_with_nothing_else_wrong_is_all_clear():
    all_clear, statement = overall_verdict(True, orphans=[], annotation_issues=[])
    assert all_clear
    assert "ALL CLEAR" in statement


def test_an_annotation_issue_denies_the_all_clear_even_when_converged():
    # The silent-failure case the whole tool exists to prevent, one layer down:
    # every mp3 present and correct, but a track's beat grid never arrived, so
    # it cannot be trained on.  The five buckets cannot see that -- they only
    # ever describe audio -- so it must not be able to hide behind convergence.
    all_clear, statement = overall_verdict(
        True, orphans=[], annotation_issues=["0001.a: beat grid missing"]
    )
    assert not all_clear
    assert "annotation" in statement


def test_an_orphan_denies_the_all_clear_even_when_converged():
    all_clear, statement = overall_verdict(True, orphans=["stray"], annotation_issues=[])
    assert not all_clear
    assert "orphan" in statement


def test_failing_to_converge_denies_the_all_clear():
    all_clear, statement = overall_verdict(False, orphans=[], annotation_issues=[])
    assert not all_clear
    assert "converge" in statement


# --------------------------------------------------------------------------- #
# checksums -- the local integrity baseline
# --------------------------------------------------------------------------- #


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "x.bin"
    payload = b"raveform" * 5000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_checksums_cover_the_ok_files_only_and_are_sha256sum_readable(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    for name in ("b", "a", "bad"):
        (audio / f"{name}.mp3").write_bytes(name.encode())
    verdicts = [
        _verdict("0002.b", "b", STATUS_OK, mp3_path=str(audio / "b.mp3"),
                 sha256=hashlib.sha256(b"b").hexdigest()),
        _verdict("0001.a", "a", STATUS_OK, mp3_path=str(audio / "a.mp3"),
                 sha256=hashlib.sha256(b"a").hexdigest()),
        _verdict("0003.bad", "bad", STATUS_DURATION_MISMATCH, mp3_path=str(audio / "bad.mp3"),
                 sha256=hashlib.sha256(b"bad").hexdigest()),
    ]
    path, count = write_checksums(tmp_path, verdicts)
    assert count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    # `sha256sum -c` format: hash, two spaces, path relative to the data dir.
    assert lines == [
        f"{hashlib.sha256(b'a').hexdigest()}  audio/a.mp3",
        f"{hashlib.sha256(b'b').hexdigest()}  audio/b.mp3",
    ]
    assert path.name == CHECKSUMS_FILE


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_the_text_report_names_every_unavailable_and_mismatched_track():
    payload = {
        "generated_at_utc": "2026-07-27T00:00:00Z",
        "data_dir": "C:/corpus",
        "manifest_tracks": 3,
        "counts": {STATUS_OK: 1, STATUS_DURATION_MISMATCH: 1, STATUS_UNAVAILABLE: 1,
                   STATUS_MISSING: 0, STATUS_CORRUPT: 0},
        "unavailable_by_reason": {"unavailable": 1},
        "converged": True,
        "convergence_statement": "converged: 3 accounted for",
        "all_clear": True,
        "verdict_statement": "ALL CLEAR: converged, no orphans, no annotation issues",
        "tolerance": {"abs_sec": 10.0, "rel": 0.03},
        "checksums": {"file": CHECKSUMS_FILE, "algorithm": "sha256", "files": 1},
        "orphans": [],
        "annotation_issues": [],
        "tracks": [
            {"track_id": "0001.a", "youtube_id": "a", "status": STATUS_OK, "detail": "",
             "decoded_duration_sec": 300.0, "annotation_duration_sec": 300.0,
             "ffprobe_duration_sec": 300.0, "failure_reason": "", "mp3_path": "", "sha256": "d"},
            {"track_id": "0002.b", "youtube_id": "b", "status": STATUS_DURATION_MISMATCH,
             "detail": "decoded 200.000 s vs annotation 400.000 s",
             "decoded_duration_sec": 200.0, "annotation_duration_sec": 400.0,
             "ffprobe_duration_sec": 200.0, "failure_reason": "", "mp3_path": "", "sha256": ""},
            {"track_id": "0003.c", "youtube_id": "c", "status": STATUS_UNAVAILABLE,
             "detail": "unavailable: ERROR: Video unavailable",
             "decoded_duration_sec": None, "annotation_duration_sec": 500.0,
             "ffprobe_duration_sec": None, "failure_reason": "unavailable", "mp3_path": "",
             "sha256": ""},
        ],
    }
    text = render_text_report(payload)
    assert "0003.c" in text and "Video unavailable" in text
    assert "0002.b" in text and "200.000" in text and "400.000" in text
    assert "converged" in text.lower()
    assert "ALL CLEAR" in text


# --------------------------------------------------------------------------- #
# prune_corrupt -- the only thing that ever deletes audio, and only on request
# --------------------------------------------------------------------------- #


def test_pruning_removes_corrupt_files_and_their_archive_lines(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "good.mp3").write_bytes(b"good")
    (audio / "bad.mp3").write_bytes(b"bad")
    (tmp_path / "downloaded.txt").write_text("youtube good\nyoutube bad\n", encoding="utf-8")

    verdicts = [
        _verdict("0001.good", "good", STATUS_OK, mp3_path=str(audio / "good.mp3")),
        _verdict("0002.bad", "bad", STATUS_CORRUPT, mp3_path=str(audio / "bad.mp3")),
    ]
    pruned = prune_corrupt(tmp_path, verdicts)

    assert pruned == ["bad"]
    assert (audio / "good.mp3").exists()
    assert not (audio / "bad.mp3").exists()
    # The archive line has to go too, or yt-dlp answers "already recorded" and
    # the track can never be re-fetched.
    archive = (tmp_path / "downloaded.txt").read_text(encoding="utf-8")
    assert "good" in archive and "bad" not in archive


def test_pruning_leaves_a_duration_mismatch_on_disk(tmp_path):
    # A wrong-length track is a human's judgement call, not a bad byte stream.
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "odd.mp3").write_bytes(b"odd")
    verdicts = [_verdict("0001.odd", "odd", STATUS_DURATION_MISMATCH,
                         mp3_path=str(audio / "odd.mp3"))]
    assert prune_corrupt(tmp_path, verdicts) == []
    assert (audio / "odd.mp3").exists()


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_a_whole_miniature_corpus_is_accounted_for(tmp_path):
    tracks = [
        ("0001.aaa", "aaa", 6.0),   # on disk, right length      -> OK
        ("0002.bbb", "bbb", 90.0),  # on disk, wrong length      -> DURATION_MISMATCH
        ("0003.ccc", "ccc", 6.0),   # on disk, not audio at all  -> CORRUPT
        ("0004.ddd", "ddd", 6.0),   # absent, recorded failure   -> UNAVAILABLE
        ("0005.eee", "eee", 6.0),   # absent, no record at all   -> MISSING
    ]
    _write_corpus(tmp_path, tracks)
    audio = tmp_path / "audio"
    _sine_mp3(audio / "aaa.mp3", 6.0)
    _sine_mp3(audio / "bbb.mp3", 6.0)          # annotation says 90 s
    (audio / "ccc.mp3").write_bytes(b"<html>not an mp3</html>" * 100)
    (audio / "orphan.mp3").write_bytes(b"x")   # no manifest row
    (audio / "aaa.mp3.44100.npy").write_bytes(b"x")  # deliberate decode cache
    (tmp_path / "failed.jsonl").write_text(
        json.dumps({"youtube_id": "ddd", "reason": "unavailable",
                    "error": "ERROR: Video unavailable", "timestamp": "t"}) + "\n",
        encoding="utf-8",
    )

    payload = validate(tmp_path, workers=1, checksums=True)

    assert payload["counts"] == {
        STATUS_OK: 1,
        STATUS_DURATION_MISMATCH: 1,
        STATUS_CORRUPT: 1,
        STATUS_MISSING: 1,
        STATUS_UNAVAILABLE: 1,
    }
    assert payload["converged"] is False          # one MISSING, one CORRUPT
    assert payload["orphans"] == ["orphan"]       # the .npy is not swept in
    assert payload["annotation_issues"] == []
    assert len(payload["tracks"]) == len(tracks)

    # Both report formats and the checksum baseline are on disk.
    assert (tmp_path / VALIDATION_JSON).exists()
    assert (tmp_path / VALIDATION_TXT).exists()
    written = json.loads((tmp_path / VALIDATION_JSON).read_text(encoding="utf-8"))
    assert written["counts"] == payload["counts"]
    checksums = (tmp_path / CHECKSUMS_FILE).read_text(encoding="utf-8").splitlines()
    assert checksums == [f"{sha256_file(audio / 'aaa.mp3')}  audio/aaa.mp3"]


@needs_ffmpeg
def test_a_complete_corpus_converges_end_to_end(tmp_path):
    tracks = [("0001.aaa", "aaa", 6.0), ("0002.bbb", "bbb", 6.0)]
    _write_corpus(tmp_path, tracks)
    _sine_mp3(tmp_path / "audio" / "aaa.mp3", 6.0)
    _sine_mp3(tmp_path / "audio" / "bbb.mp3", 6.0)

    payload = validate(tmp_path, workers=1, checksums=True)

    assert payload["counts"][STATUS_OK] == 2
    assert payload["converged"] is True
    assert payload["all_clear"] is True
    assert "converged" in render_text_report(payload).lower()


@needs_ffmpeg
def test_a_missing_beat_grid_converges_but_is_not_all_clear(tmp_path):
    # Every mp3 present and correct, so the five buckets are perfect -- and the
    # track still cannot be trained on.  Convergence must stay true (it is an
    # honest statement about the audio) while the overall verdict, and the exit
    # code that follows it, must not.
    tracks = [("0001.aaa", "aaa", 6.0), ("0002.bbb", "bbb", 6.0)]
    _write_corpus(tmp_path, tracks)
    _sine_mp3(tmp_path / "audio" / "aaa.mp3", 6.0)
    _sine_mp3(tmp_path / "audio" / "bbb.mp3", 6.0)
    (tmp_path / "annotations" / "beats" / "0002.bbb.beat.csv").unlink()

    payload = validate(tmp_path, workers=1, checksums=False)

    assert payload["counts"][STATUS_OK] == 2
    assert payload["converged"] is True
    assert payload["all_clear"] is False
    assert len(payload["annotation_issues"]) == 1


# --------------------------------------------------------------------------- #
# The exit code -- what a supervisor or CI step actually branches on
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_the_exit_code_follows_the_all_clear_not_convergence(tmp_path):
    # The distinction only exists because convergence alone can be true while
    # the validator's own annotation check has findings.  If the exit code
    # tracked convergence instead, a corpus with a missing beat grid would
    # report success to every automated caller -- so this is the assertion that
    # makes the whole all_clear/converged split worth having.
    tracks = [("0001.aaa", "aaa", 6.0), ("0002.bbb", "bbb", 6.0)]
    _write_corpus(tmp_path, tracks)
    _sine_mp3(tmp_path / "audio" / "aaa.mp3", 6.0)
    _sine_mp3(tmp_path / "audio" / "bbb.mp3", 6.0)
    (tmp_path / "annotations" / "beats" / "0002.bbb.beat.csv").unlink()

    code = main(["--data-dir", str(tmp_path), "--workers", "1", "--skip-checksums"])

    payload = json.loads((tmp_path / VALIDATION_JSON).read_text(encoding="utf-8"))
    assert payload["converged"] is True   # the audio really does add up
    assert code == 1                      # ...and the run is still not a success


@needs_ffmpeg
def test_a_clean_corpus_exits_zero(tmp_path):
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 6.0)])
    _sine_mp3(tmp_path / "audio" / "aaa.mp3", 6.0)
    assert main(["--data-dir", str(tmp_path), "--workers", "1", "--skip-checksums"]) == 0


@needs_ffmpeg
def test_an_unaccounted_track_exits_nonzero(tmp_path):
    # The MISSING case: in the manifest, no audio, no recorded attempt.
    _write_corpus(tmp_path, [("0001.aaa", "aaa", 6.0), ("0002.bbb", "bbb", 6.0)])
    _sine_mp3(tmp_path / "audio" / "aaa.mp3", 6.0)
    assert main(["--data-dir", str(tmp_path), "--workers", "1", "--skip-checksums"]) == 1
