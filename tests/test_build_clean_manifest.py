"""Tests for the cleanliness gate (training/raveform/build_clean_manifest.py).

The gate decides which audio is allowed to become training data.  A bug here is
silent: it either admits a truncated/corrupt file (poisoning every downstream
row with beats that do not exist in the annotation) or quietly drops hundreds of
good tracks.  So the classification rules, the tolerance boundary and the
"don't touch a file the downloader may still be writing" guard are all pinned.

ffmpeg/ffprobe are exercised for real where the point of the test is that a byte
stream really is (or is not) decodable; the pure logic is tested without them.
"""
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

RAVEFORM_DIR = Path(__file__).resolve().parents[1] / "training" / "raveform"
if str(RAVEFORM_DIR) not in sys.path:
    sys.path.insert(0, str(RAVEFORM_DIR))

from build_clean_manifest import (  # noqa: E402  (needs the path insert above)
    _run,
    ABS_TOLERANCE_SEC,
    CLEAN_MANIFEST_HEADER,
    MIN_AGE_SEC,
    REL_TOLERANCE,
    STATUS_CORRUPT,
    STATUS_MISMATCH,
    STATUS_OK,
    ManifestRow,
    TrackJob,
    audio_path,
    check_track,
    classify,
    decode,
    duration_tolerance,
    is_settled,
    load_manifest_rows,
    require_tools,
    run_checks,
    select_candidates,
    summarise,
    write_clean_manifest,
)

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _age(path: Path, seconds: float) -> None:
    """Backdate a file's mtime so the recency guard sees it as settled."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def _sine_mp3(path: Path, seconds: float) -> Path:
    """A real, fully decodable mp3 of a given length."""
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


# --------------------------------------------------------------------------- #
# duration_tolerance
# --------------------------------------------------------------------------- #


def test_tolerance_is_the_absolute_floor_for_short_tracks():
    # 3% of 120 s is 3.6 s, well under the 10 s floor.
    assert duration_tolerance(120.0) == ABS_TOLERANCE_SEC


def test_tolerance_is_relative_for_long_tracks():
    # A 20-minute DJ edit deserves proportional slack, not a flat 10 s.
    assert duration_tolerance(1200.0) == pytest.approx(1200.0 * REL_TOLERANCE)


def test_tolerance_crossover_is_where_the_two_rules_agree():
    crossover = ABS_TOLERANCE_SEC / REL_TOLERANCE
    assert duration_tolerance(crossover) == pytest.approx(ABS_TOLERANCE_SEC)
    assert duration_tolerance(crossover * 2) > ABS_TOLERANCE_SEC


def test_tolerance_of_a_zero_duration_is_the_floor():
    assert duration_tolerance(0.0) == ABS_TOLERANCE_SEC


# --------------------------------------------------------------------------- #
# classify -- corrupt beats everything
# --------------------------------------------------------------------------- #


def _consistent(duration: float, annotation: float):
    """classify() for a file whose header and decoder agree on ``duration``."""
    return classify("", duration, duration, annotation)


def test_decode_failure_is_corrupt_even_when_the_duration_matches():
    status, detail = classify("moov atom not found", 300.0, 300.0, 300.0)
    assert status == STATUS_CORRUPT
    assert "moov atom not found" in detail


def test_missing_ffprobe_duration_is_corrupt():
    status, _detail = classify("", 300.0, None, 300.0)
    assert status == STATUS_CORRUPT


def test_missing_decoded_duration_is_corrupt():
    status, _detail = classify("", None, 300.0, 300.0)
    assert status == STATUS_CORRUPT


def test_zero_length_audio_is_corrupt():
    # ffmpeg can "decode" a header-only file cleanly; a 0 s result is still junk.
    assert _consistent(0.0, 300.0)[0] == STATUS_CORRUPT


def test_negative_ffprobe_duration_is_corrupt():
    assert _consistent(-1.0, 300.0)[0] == STATUS_CORRUPT


# --------------------------------------------------------------------------- #
# classify -- truncation: the header lies, the decoder does not
# --------------------------------------------------------------------------- #


def test_a_file_that_decodes_short_of_its_header_is_corrupt():
    # The interrupted-download shape: the Xing header still advertises the full
    # length, the decoder runs out of bytes early, and ffmpeg reports no error.
    status, detail = classify("", 12.0, 60.0, 60.0)
    assert status == STATUS_CORRUPT
    assert "truncated" in detail


def test_truncation_outranks_a_duration_mismatch():
    # Both rules fire (12 s is short of the 60 s header AND of the annotation);
    # the byte-level defect is the more specific and more useful diagnosis.
    assert classify("", 12.0, 60.0, 60.0)[0] == STATUS_CORRUPT


def test_a_track_whose_header_and_decode_agree_is_not_called_truncated():
    # Wrong video, but intact bytes: that is a mismatch, not corruption.
    status, _detail = classify("", 200.0, 200.0, 400.0)
    assert status == STATUS_MISMATCH


def test_header_decode_disagreement_within_tolerance_is_not_truncation():
    # Encoder padding puts the decoded length a few tens of ms under the header.
    assert classify("", 299.95, 300.0, 300.0)[0] == STATUS_OK


def test_a_header_shorter_than_the_decode_is_also_corrupt():
    # The reverse disagreement is just as much a broken container.
    assert classify("", 300.0, 60.0, 300.0)[0] == STATUS_CORRUPT


# --------------------------------------------------------------------------- #
# classify -- the tolerance boundary
# --------------------------------------------------------------------------- #


def test_exact_duration_match_is_ok():
    assert _consistent(300.0, 300.0)[0] == STATUS_OK


def test_delta_exactly_at_the_absolute_tolerance_is_ok():
    # Inclusive boundary: "within max(+-10 s, +-3%)" includes the endpoint.
    assert _consistent(300.0 + ABS_TOLERANCE_SEC, 300.0)[0] == STATUS_OK
    assert _consistent(300.0 - ABS_TOLERANCE_SEC, 300.0)[0] == STATUS_OK


def test_delta_just_beyond_the_absolute_tolerance_is_a_mismatch():
    assert _consistent(300.0 + ABS_TOLERANCE_SEC + 0.01, 300.0)[0] == STATUS_MISMATCH
    assert _consistent(300.0 - ABS_TOLERANCE_SEC - 0.01, 300.0)[0] == STATUS_MISMATCH


def test_delta_inside_the_relative_tolerance_is_ok_for_a_long_track():
    # 15 s off a 20-minute track is 1.25% -- fine, though it exceeds the floor.
    assert _consistent(1200.0 + 15.0, 1200.0)[0] == STATUS_OK


def test_delta_just_beyond_the_relative_tolerance_is_a_mismatch():
    tolerance = 1200.0 * REL_TOLERANCE
    assert _consistent(1200.0 + tolerance, 1200.0)[0] == STATUS_OK
    assert _consistent(1200.0 + tolerance + 0.01, 1200.0)[0] == STATUS_MISMATCH


def test_mismatch_detail_reports_the_delta_and_the_tolerance():
    _status, detail = _consistent(100.0, 300.0)
    assert "200" in detail  # the delta, so a human can see how far off it is


# --------------------------------------------------------------------------- #
# Subprocess text decoding
# --------------------------------------------------------------------------- #


def test_a_tool_that_writes_undecodable_bytes_does_not_kill_the_worker():
    """ffmpeg echoes the offending file's tag text when it complains, and the
    corpus is full-of-the-world track titles.  Left to the locale this raises
    UnicodeDecodeError out of `subprocess.run` -- inside a pool worker, taking
    a whole batch down over one byte in a filename.  The encoding is pinned and
    undecodable bytes become U+FFFD, so the complaint still reaches the log."""
    proc = _run([sys.executable, "-c",
                 r"import sys; sys.stdout.buffer.write(b'\xff\x81 done')"])

    assert proc.stdout.endswith(" done")
    assert "�" in proc.stdout


# --------------------------------------------------------------------------- #
# is_settled / select_candidates -- the live-downloader guard
# --------------------------------------------------------------------------- #


def test_a_file_the_downloader_just_wrote_is_not_settled(tmp_path):
    fresh = tmp_path / "fresh.mp3"
    fresh.write_bytes(b"x")
    assert is_settled(fresh, now=time.time(), min_age_sec=MIN_AGE_SEC) is False


def test_a_file_older_than_the_min_age_is_settled(tmp_path):
    old = tmp_path / "old.mp3"
    old.write_bytes(b"x")
    _age(old, MIN_AGE_SEC + 5)
    assert is_settled(old, now=time.time(), min_age_sec=MIN_AGE_SEC) is True


def test_a_file_exactly_at_the_min_age_is_not_yet_settled(tmp_path):
    path = tmp_path / "edge.mp3"
    path.write_bytes(b"x")
    assert is_settled(path, now=path.stat().st_mtime + MIN_AGE_SEC, min_age_sec=MIN_AGE_SEC) is False


def test_a_missing_file_is_not_settled(tmp_path):
    assert is_settled(tmp_path / "nope.mp3", now=time.time(), min_age_sec=MIN_AGE_SEC) is False


def test_select_candidates_splits_missing_fresh_and_settled(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    settled = audio / "bbb.mp3"
    settled.write_bytes(b"x")
    _age(settled, 3600)
    fresh = audio / "ccc.mp3"
    fresh.write_bytes(b"x")

    rows = [
        ManifestRow("0001.aaa", "aaa", 100.0),   # never downloaded
        ManifestRow("0002.bbb", "bbb", 200.0),   # settled -> checked
        ManifestRow("0003.ccc", "ccc", 300.0),   # still being written -> skipped
    ]
    jobs, missing, too_recent = select_candidates(rows, tmp_path, now=time.time())

    assert [job.track_id for job in jobs] == ["0002.bbb"]
    assert missing == 1
    assert too_recent == 1


def test_select_candidates_is_sorted_by_track_id(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    rows = []
    for youtube_id, track_id in (("zzz", "0009.zzz"), ("aaa", "0001.aaa"), ("mmm", "0005.mmm")):
        path = audio / f"{youtube_id}.mp3"
        path.write_bytes(b"x")
        _age(path, 3600)
        rows.append(ManifestRow(track_id, youtube_id, 100.0))

    jobs, _missing, _fresh = select_candidates(rows, tmp_path, now=time.time())
    assert [job.track_id for job in jobs] == ["0001.aaa", "0005.mmm", "0009.zzz"]


def test_audio_path_follows_the_downloader_naming(tmp_path):
    assert audio_path(tmp_path, "kfJQCu-Jbec") == tmp_path / "audio" / "kfJQCu-Jbec.mp3"


# --------------------------------------------------------------------------- #
# check_track -- real ffmpeg
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_a_stub_mp3_that_ffmpeg_rejects_is_corrupt(tmp_path):
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"this is not an mp3, it is a sentence." * 64)

    result = check_track(TrackJob("0001.junk", "junk", str(junk), 300.0))

    assert result.status == STATUS_CORRUPT
    assert result.detail  # the reason must survive to the report


@needs_ffmpeg
def test_an_empty_file_is_corrupt(tmp_path):
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")

    assert check_track(TrackJob("0001.empty", "empty", str(empty), 300.0)).status == STATUS_CORRUPT


@needs_ffmpeg
def test_a_real_mp3_matching_its_annotation_is_ok(tmp_path):
    song = _sine_mp3(tmp_path / "good.mp3", 3.0)

    result = check_track(TrackJob("0001.good", "good", str(song), 3.0))

    assert result.status == STATUS_OK
    assert result.ffprobe_duration_sec == pytest.approx(3.0, abs=0.5)
    assert result.decoded_duration_sec == pytest.approx(3.0, abs=0.5)
    assert result.detail == ""


@needs_ffmpeg
@pytest.mark.parametrize(
    "source", ["sine=frequency=440:duration=60", "anoisesrc=d=60:c=pink"]
)
def test_a_truncated_mp3_is_rejected(tmp_path, source):
    """THE test: the failure mode that neither earlier check could see.

    An mp3 cut on a frame boundary -- an interrupted download -- decodes with
    exit 0 and empty stderr, and its Xing header still advertises the original
    length.  Only the decoded length gives it away.  Two different signals,
    because the first version of this gate passed a truncated file whose bit
    reservoir happened to produce an error on one particular track.
    """
    full = tmp_path / "full.mp3"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", source,
            "-c:a", "libmp3lame", "-b:a", "128k", str(full),
        ],
        check=True,
        capture_output=True,
    )
    truncated = tmp_path / "truncated.mp3"
    truncated.write_bytes(full.read_bytes()[: full.stat().st_size // 5])

    result = check_track(TrackJob("0001.trunc", "trunc", str(truncated), 60.0))

    assert result.status == STATUS_CORRUPT, (
        f"a 20% prefix was admitted as {result.status} "
        f"(header {result.ffprobe_duration_sec}, decoded {result.decoded_duration_sec})"
    )
    # The header keeps claiming the full length -- that is exactly why it cannot
    # be trusted on its own.
    assert result.ffprobe_duration_sec == pytest.approx(60.0, abs=1.0)
    assert result.decoded_duration_sec < 20.0


@needs_ffmpeg
def test_decode_measures_what_the_decoder_actually_produced(tmp_path):
    song = _sine_mp3(tmp_path / "measured.mp3", 4.0)

    complaint, decoded = decode(str(song))

    assert complaint == ""
    assert decoded == pytest.approx(4.0, abs=0.2)


@needs_ffmpeg
def test_a_real_mp3_far_shorter_than_its_annotation_is_a_mismatch(tmp_path):
    # The realistic failure: yt-dlp grabbed a trailer/edit, not the full track.
    song = _sine_mp3(tmp_path / "short.mp3", 3.0)

    result = check_track(TrackJob("0001.short", "short", str(song), 400.0))

    assert result.status == STATUS_MISMATCH


@needs_ffmpeg
def test_check_track_leaves_no_files_behind(tmp_path):
    song = _sine_mp3(tmp_path / "clean.mp3", 2.0)
    before = sorted(p.name for p in tmp_path.iterdir())

    check_track(TrackJob("0001.clean", "clean", str(song), 2.0))

    assert sorted(p.name for p in tmp_path.iterdir()) == before


# --------------------------------------------------------------------------- #
# run_checks
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_run_checks_returns_results_in_track_id_order(tmp_path):
    jobs = []
    for track_id, name in (("0009.z", "z"), ("0001.a", "a"), ("0005.m", "m")):
        song = _sine_mp3(tmp_path / f"{name}.mp3", 1.0)
        jobs.append(TrackJob(track_id, name, str(song), 1.0))

    results = run_checks(jobs, workers=1)
    assert [r.track_id for r in results] == ["0001.a", "0005.m", "0009.z"]
    assert {r.status for r in results} == {STATUS_OK}


def test_run_checks_of_nothing_is_empty():
    assert run_checks([], workers=8) == []


# --------------------------------------------------------------------------- #
# manifest I/O
# --------------------------------------------------------------------------- #


def _write_manifest(data_dir: Path, rows: list) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "manifest.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("track_id", "youtube_id", "n_sections", "total_sec"))
        writer.writerows(rows)


def test_load_manifest_rows_reads_the_existing_manifest(tmp_path):
    _write_manifest(tmp_path, [("0002.kfJQCu-Jbec", "kfJQCu-Jbec", 11, "429.964")])

    rows = load_manifest_rows(tmp_path)

    assert rows == [ManifestRow("0002.kfJQCu-Jbec", "kfJQCu-Jbec", 429.964)]


def test_load_manifest_rows_rejects_an_empty_manifest(tmp_path):
    _write_manifest(tmp_path, [])
    with pytest.raises(RuntimeError):
        load_manifest_rows(tmp_path)


def test_write_clean_manifest_emits_the_agreed_schema(tmp_path):
    results = run_checks([], workers=1)  # empty is legal; header still written
    path = write_clean_manifest(tmp_path, results)

    with open(path, "r", encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == list(CLEAN_MANIFEST_HEADER)


def test_write_clean_manifest_writes_every_status_and_sorts_by_track_id(tmp_path):
    from build_clean_manifest import CheckResult

    results = [
        CheckResult("0009.z", "z", "C:/audio/z.mp3", 300.0, 300.0, 300.0, STATUS_OK, ""),
        CheckResult("0001.a", "a", "C:/audio/a.mp3", None, None, 300.0, STATUS_CORRUPT, "boom"),
        CheckResult(
            "0005.m", "m", "C:/audio/m.mp3", 100.0, 100.0, 300.0, STATUS_MISMATCH, "-200.0 s"
        ),
    ]
    path = write_clean_manifest(tmp_path, results)

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["track_id"] for row in rows] == ["0001.a", "0005.m", "0009.z"]
    assert [row["status"] for row in rows] == [STATUS_CORRUPT, STATUS_MISMATCH, STATUS_OK]
    assert rows[0]["ffprobe_duration_sec"] == ""  # unknown, not a fake 0
    assert rows[0]["decoded_duration_sec"] == ""
    assert rows[2]["ffprobe_duration_sec"] == "300.000"
    assert rows[2]["decoded_duration_sec"] == "300.000"
    assert rows[2]["annotation_duration_sec"] == "300.000"


def test_write_clean_manifest_persists_the_rejection_reason(tmp_path):
    # A quarantined row is only actionable if the reason survives the run.
    from build_clean_manifest import CheckResult

    results = [
        CheckResult("0001.a", "a", "C:/audio/a.mp3", 60.0, 12.0, 60.0, STATUS_CORRUPT, "truncated: ..."),
        CheckResult("0002.b", "b", "C:/audio/b.mp3", 300.0, 300.0, 300.0, STATUS_OK, ""),
    ]
    path = write_clean_manifest(tmp_path, results)

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["detail"] == "truncated: ..."
    assert rows[1]["detail"] == ""


def test_write_clean_manifest_is_byte_identical_across_runs(tmp_path):
    from build_clean_manifest import CheckResult

    results = [
        CheckResult("0002.b", "b", "C:/audio/b.mp3", 210.5, 210.4, 211.0, STATUS_OK, ""),
        CheckResult("0001.a", "a", "C:/audio/a.mp3", 300.0, 300.0, 300.0, STATUS_OK, ""),
    ]
    first = write_clean_manifest(tmp_path, results).read_bytes()
    second = write_clean_manifest(tmp_path, list(reversed(results))).read_bytes()
    assert first == second


def test_write_clean_manifest_leaves_no_part_file(tmp_path):
    write_clean_manifest(tmp_path, [])
    assert list(tmp_path.glob("*.part")) == []


# --------------------------------------------------------------------------- #
# summarise
# --------------------------------------------------------------------------- #


def test_summarise_counts_each_status(tmp_path):
    from build_clean_manifest import CheckResult

    results = [
        CheckResult("0001.a", "a", "a.mp3", 300.0, 300.0, 300.0, STATUS_OK, ""),
        CheckResult("0002.b", "b", "b.mp3", 300.0, 300.0, 300.0, STATUS_OK, ""),
        CheckResult("0003.c", "c", "c.mp3", 100.0, 100.0, 300.0, STATUS_MISMATCH, ""),
        CheckResult("0004.d", "d", "d.mp3", None, None, 300.0, STATUS_CORRUPT, "boom"),
    ]
    counts = summarise(results)
    assert counts[STATUS_OK] == 2
    assert counts[STATUS_MISMATCH] == 1
    assert counts[STATUS_CORRUPT] == 1


# --------------------------------------------------------------------------- #
# require_tools
# --------------------------------------------------------------------------- #


def test_require_tools_accepts_a_tool_that_exists():
    require_tools(("python",))


def test_require_tools_names_the_missing_tool():
    # Without the preflight, a missing binary reads as "the whole corpus is
    # corrupt" -- the most misleading possible failure for this script.
    with pytest.raises(RuntimeError, match="definitely-not-a-real-tool"):
        require_tools(("definitely-not-a-real-tool",))
