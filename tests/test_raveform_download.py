"""Tests for the raveform downloader's pure state logic (training/raveform_download.py).

These cover the three decisions that make an unattended multi-hour run
recoverable: how a failure is bucketed, which failures a re-run retries, and how
an id is taken back out of the download archive.  All of them are pure or
tmp_path-local -- nothing here touches the network or the real corpus.
"""
import json
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from raveform_download import (  # noqa: E402  (needs the path insert above)
    BLOCK_REASONS,
    INTERRUPT_REASON,
    KNOWN_REASONS,
    RETRY_HINT,
    RETRYABLE_REASONS,
    _REASON_PATTERNS,
    archive_path,
    build_command,
    classify_error,
    failed_path,
    forget_download,
    read_archive_ids,
    read_failed_reasons,
    read_manifest_ids,
)


def write_archive(data_dir: Path, text: str) -> Path:
    path = archive_path(data_dir)
    path.write_text(text, encoding="utf-8", newline="")
    return path


def write_failures(data_dir: Path, records: list) -> Path:
    path = failed_path(data_dir)
    lines = [record if isinstance(record, str) else json.dumps(record) for record in records]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    return path


# --------------------------------------------------------------------------- #
# classify_error -- reason buckets
# --------------------------------------------------------------------------- #


def test_sign_in_wall_is_a_bot_check():
    assert classify_error("ERROR: Sign in to confirm you're not a bot") == "bot_check"


def test_curly_apostrophe_sign_in_wall_is_a_bot_check():
    # YouTube emits the typographic apostrophe; a straight-quote-only pattern
    # would silently bucket every block as `other` and defeat --retry-reasons.
    assert classify_error("ERROR: Please confirm you’re not a bot") == "bot_check"


def test_rate_limit_is_a_bot_check():
    assert classify_error("ERROR: HTTP Error 429: Too Many Requests") == "bot_check"


def test_cookie_advice_is_a_bot_check():
    assert classify_error("Use --cookies-from-browser or --cookies for the authentication") == "bot_check"


def test_age_restriction_is_its_own_bucket():
    assert classify_error("ERROR: Sign in to view this age-restricted video") == "age_restricted"


def test_geo_block_is_its_own_bucket():
    assert classify_error("ERROR: This video is not available in your country") == "geo_blocked"


def test_removed_video_is_unavailable():
    assert classify_error("ERROR: Video unavailable. This video has been removed") == "unavailable"


def test_private_video_is_unavailable():
    assert classify_error("ERROR: Private video. Sign in if you've been granted access") == "unavailable"


def test_copyright_takedown_is_its_own_bucket():
    blob = "ERROR: This video contains content from WMG, who has blocked it on copyright grounds"
    assert classify_error(blob) == "copyright"


def test_unrecognised_text_is_other():
    assert classify_error("ERROR: unable to download video data: <urlopen error timed out>") == "other"


def test_empty_text_is_other():
    assert classify_error("") == "other"


def test_matching_is_case_insensitive():
    assert classify_error("ERROR: VIDEO UNAVAILABLE") == "unavailable"


def test_a_credential_wall_outranks_the_unavailable_text_it_carries():
    # YouTube appends "Video unavailable" to sign-in walls.  Bucketing that as
    # `unavailable` would file a retryable block as a dead video, and the track
    # would never be re-fetched.
    blob = "ERROR: Video unavailable. Sign in to confirm you're not a bot"
    assert classify_error(blob) == "bot_check"


def test_an_age_gate_outranks_the_sign_in_text_it_carries():
    # The REAL yt-dlp wording.  It contains "sign in to confirm", which is a
    # bot_check pattern, so bucket order decides -- and getting it wrong means
    # every retry pass re-polls a video we will never sign in for, and five in a
    # row abort a healthy run.
    blob = (
        "ERROR: [youtube] LuT4EqmJmnU: Sign in to confirm your age. "
        "This video may be inappropriate for some users."
    )
    assert classify_error(blob) == "age_restricted"


def test_a_real_bot_check_is_still_a_bot_check():
    # The twin of the test above: the age-gate patterns must not swallow the
    # actual credential wall, whose wording differs only in its last three words.
    blob = (
        "ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm you're not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication."
    )
    assert classify_error(blob) == "bot_check"


def test_a_media_url_403_is_its_own_bucket_not_other():
    # The REAL wording, and 56% of the failures on the first full corpus sweep.
    # As `other` it was invisible to every "just retry the blocks" hint, and the
    # 62 tracks behind it were recovered only because the operator widened the
    # retry list by hand.
    blob = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
    assert classify_error(blob) == "http_403"


def test_a_403_is_retryable_and_aborts_a_sustained_run():
    # It describes the client, not the video, so it must be re-attempted -- and
    # a run of them back to back means the run is broken, not the manifest.
    assert "http_403" in RETRYABLE_REASONS
    assert "http_403" in BLOCK_REASONS


def test_the_id_stays_behind_the_separator(tmp_path):
    # 19 ids in the manifest begin with "-" (e.g. -DWkf03g4Gc), and all 19 are on
    # disk. Without the "--" guard yt-dlp would parse them as options and every
    # one of those tracks would be unfetchable.
    argv = build_command(tmp_path, "-DWkf03g4Gc")
    assert argv[-2:] == ["--", "-DWkf03g4Gc"]


def test_the_non_pattern_reasons_are_all_retryable():
    # These four never come from the pattern table -- they are recorded when the
    # run itself misbehaves (no output, empty output, a timeout) or when no
    # pattern matched at all.  `other` in particular is the whole argument of
    # the 403 fix: an error we have no pattern for is a gap in our knowledge,
    # not evidence that a video is gone, and dropping it from the retry set is
    # what stranded 62 recoverable tracks.
    assert {"other", "empty_output", "missing_output", "timeout"} <= RETRYABLE_REASONS


def test_the_retry_hint_names_every_recoverable_reason():
    # The hint used to say "bot_check (add ,timeout if any timed out)", which
    # would have abandoned 62 recoverable tracks.  It is now generated from the
    # same set the code reasons about, so it cannot drift narrower again.
    for reason in RETRYABLE_REASONS:
        assert reason in RETRY_HINT
    assert RETRY_HINT.startswith("--retry-reasons ")


def test_every_reason_the_table_can_emit_is_a_known_reason():
    # --retry-reasons is validated against KNOWN_REASONS, so a bucket missing
    # from it would be unselectable: its failures could never be retried.
    for reason, _patterns in _REASON_PATTERNS:
        assert reason in KNOWN_REASONS
    assert RETRYABLE_REASONS <= KNOWN_REASONS


def test_no_pattern_appears_in_two_buckets():
    # A phrase listed under two reasons makes bucket ORDER the silent arbiter of
    # meaning -- which is precisely how the private-video, age-gate and 403
    # mis-classifications survived.  One phrase, one intended bucket.
    seen = {}
    for reason, patterns in _REASON_PATTERNS:
        for pattern in patterns:
            if pattern in seen and seen[pattern] != reason:
                raise AssertionError(
                    f"pattern {pattern!r} is claimed by both {seen[pattern]!r} and {reason!r}"
                )
            seen[pattern] = reason


def test_a_bare_not_available_message_is_unavailable():
    # YouTube does not always put "unavailable" next to "video"; this wording
    # would otherwise fall through to `other` and be re-polled by every retry
    # pass, which is the opposite of what the reason buckets are for.
    blob = "ERROR: [youtube] A6O2p64sucM: This video is not available"
    assert classify_error(blob) == "unavailable"


def test_a_geo_block_is_not_swallowed_by_the_not_available_pattern():
    # "This video is not available in your country" contains the bare phrase
    # too; geo_blocked is matched first and must keep winning.
    blob = "ERROR: This video is not available in your country"
    assert classify_error(blob) == "geo_blocked"


def test_a_private_video_outranks_the_cookie_advice_it_carries():
    # The real yt-dlp text, verbatim in shape: a private video error ends with
    # the standard cookie advice.  Bucketing it as `bot_check` would make every
    # retry pass re-poll a video we will never be granted access to, and enough
    # of them in a row would trip the consecutive-block guard on a healthy run.
    blob = (
        "ERROR: [youtube] CJL6uHqLyfU: Private video. Sign in if you've been granted "
        "access to this video. Use --cookies-from-browser or --cookies for the "
        "authentication."
    )
    assert classify_error(blob) == "unavailable"


def test_copyright_ranks_below_unavailable():
    # Both phrases appear together on takedowns; `unavailable` is listed first
    # and must win, so the ordering of _REASON_PATTERNS is load-bearing.
    blob = "ERROR: Video unavailable due to a copyright claim"
    assert classify_error(blob) == "unavailable"


# --------------------------------------------------------------------------- #
# classify_error -- interruption is no longer a classification
# --------------------------------------------------------------------------- #


def test_yt_dlp_interrupt_text_is_not_classified_as_an_interrupt():
    # Interruption is decided by our own SIGINT flag, never by the child's text.
    # If this ever bucketed as INTERRUPT_REASON again, a genuine error carrying
    # the word would abort the run at the same index on every re-run.
    assert classify_error("ERROR: Interrupted by user") == "other"


def test_postprocessing_abort_text_is_not_classified_as_an_interrupt():
    blob = "ERROR: Postprocessing: ffmpeg exited with code 255"
    assert classify_error(blob) != INTERRUPT_REASON
    assert classify_error(blob) == "other"


def test_no_pattern_bucket_yields_the_interrupt_reason():
    assert INTERRUPT_REASON not in {reason for reason, _patterns in _REASON_PATTERNS}


def test_the_interrupt_reason_is_not_a_recordable_reason():
    # It is never written to failed.jsonl, so it must not be offered to
    # --retry-reasons either -- there would be nothing to select.
    assert INTERRUPT_REASON not in KNOWN_REASONS


def test_every_bucket_classify_error_can_return_is_a_known_reason():
    # Guards adding a new pattern bucket without registering it: --retry-reasons
    # would then reject a reason that genuinely appears in failed.jsonl.
    buckets = {reason for reason, _patterns in _REASON_PATTERNS} | {"other"}
    assert buckets <= KNOWN_REASONS


def test_only_refusals_of_this_client_abort_the_run():
    # A dead video must never count towards --max-consecutive-blocks, or a run
    # of deleted tracks would stop the corpus build for no reason.  What may
    # count is a refusal aimed at *us* -- a credential wall or a 403 on the
    # media URL -- because a run of those means the run is broken, not the
    # manifest.  Stated as the rule rather than as a literal set, so adding a
    # refusal bucket does not require editing the test that guards the rule.
    permanent = {"unavailable", "age_restricted", "geo_blocked", "copyright"}
    assert BLOCK_REASONS.isdisjoint(permanent)
    assert BLOCK_REASONS <= RETRYABLE_REASONS  # never abort on something we won't retry
    assert BLOCK_REASONS <= KNOWN_REASONS
    assert "bot_check" in BLOCK_REASONS


# --------------------------------------------------------------------------- #
# read_failed_reasons
# --------------------------------------------------------------------------- #


def test_read_failed_reasons_of_a_missing_file_is_empty(tmp_path):
    assert read_failed_reasons(failed_path(tmp_path)) == {}


def test_read_failed_reasons_reads_one_record_per_id(tmp_path):
    write_failures(
        tmp_path,
        [
            {"youtube_id": "aaaaaaaaaaa", "reason": "unavailable"},
            {"youtube_id": "bbbbbbbbbbb", "reason": "bot_check"},
        ],
    )
    assert read_failed_reasons(failed_path(tmp_path)) == {
        "aaaaaaaaaaa": "unavailable",
        "bbbbbbbbbbb": "bot_check",
    }


def test_the_most_recent_failure_wins(tmp_path):
    # failed.jsonl is append-only: a track rate-limited first and later found
    # deleted is deleted, and must not be retried by --retry-reasons bot_check.
    write_failures(
        tmp_path,
        [
            {"youtube_id": "aaaaaaaaaaa", "reason": "bot_check"},
            {"youtube_id": "aaaaaaaaaaa", "reason": "unavailable"},
        ],
    )
    assert read_failed_reasons(failed_path(tmp_path)) == {"aaaaaaaaaaa": "unavailable"}


def test_a_missing_reason_field_becomes_other(tmp_path):
    write_failures(tmp_path, [{"youtube_id": "aaaaaaaaaaa"}])
    assert read_failed_reasons(failed_path(tmp_path)) == {"aaaaaaaaaaa": "other"}


def test_a_non_string_reason_becomes_other(tmp_path):
    write_failures(tmp_path, [{"youtube_id": "aaaaaaaaaaa", "reason": None}])
    assert read_failed_reasons(failed_path(tmp_path)) == {"aaaaaaaaaaa": "other"}


def test_a_torn_last_line_does_not_lose_the_earlier_records(tmp_path):
    # A hard kill mid-append leaves half a line; the run must still resume.
    write_failures(
        tmp_path,
        [
            json.dumps({"youtube_id": "aaaaaaaaaaa", "reason": "unavailable"}),
            '{"youtube_id": "bbbbbbbbbbb", "rea',
        ],
    )
    assert read_failed_reasons(failed_path(tmp_path)) == {"aaaaaaaaaaa": "unavailable"}


def test_blank_lines_and_records_without_an_id_are_ignored(tmp_path):
    write_failures(
        tmp_path,
        [
            "",
            json.dumps({"reason": "other"}),
            json.dumps({"youtube_id": "", "reason": "other"}),
            json.dumps({"youtube_id": "aaaaaaaaaaa", "reason": "timeout"}),
        ],
    )
    assert read_failed_reasons(failed_path(tmp_path)) == {"aaaaaaaaaaa": "timeout"}


# --------------------------------------------------------------------------- #
# forget_download
# --------------------------------------------------------------------------- #


def test_forget_download_removes_exactly_the_target_id(tmp_path):
    write_archive(
        tmp_path,
        "youtube aaaaaaaaaaa\nyoutube bbbbbbbbbbb\nyoutube ccccccccccc\n",
    )
    assert forget_download(tmp_path, "bbbbbbbbbbb") is True
    assert archive_path(tmp_path).read_text(encoding="utf-8") == (
        "youtube aaaaaaaaaaa\nyoutube ccccccccccc\n"
    )
    assert read_archive_ids(archive_path(tmp_path)) == {"aaaaaaaaaaa", "ccccccccccc"}


def test_forget_download_removes_every_line_for_that_id(tmp_path):
    # A leftover duplicate would keep yt-dlp answering "already recorded", which
    # is exactly the trap this function exists to clear.
    write_archive(tmp_path, "youtube aaaaaaaaaaa\nyoutube aaaaaaaaaaa\nyoutube bbbbbbbbbbb\n")
    assert forget_download(tmp_path, "aaaaaaaaaaa") is True
    assert read_archive_ids(archive_path(tmp_path)) == {"bbbbbbbbbbb"}


def test_forget_download_of_an_absent_id_changes_nothing(tmp_path):
    original = "youtube aaaaaaaaaaa\nyoutube bbbbbbbbbbb\n"
    write_archive(tmp_path, original)
    assert forget_download(tmp_path, "zzzzzzzzzzz") is False
    assert archive_path(tmp_path).read_text(encoding="utf-8") == original


def test_forget_download_of_a_missing_archive_is_a_no_op(tmp_path):
    assert forget_download(tmp_path, "aaaaaaaaaaa") is False
    assert not archive_path(tmp_path).exists()


def test_forget_download_does_not_match_an_id_as_a_prefix(tmp_path):
    # Ids are compared whole-token; a substring match would silently evict a
    # different, healthy track from the archive.
    write_archive(tmp_path, "youtube aaaaaaaaaaa\nyoutube aaaaaaaaaaaXY\n")
    assert forget_download(tmp_path, "aaaaaaaaaaa") is True
    assert read_archive_ids(archive_path(tmp_path)) == {"aaaaaaaaaaaXY"}


def test_forget_download_tolerates_blank_lines(tmp_path):
    write_archive(tmp_path, "youtube aaaaaaaaaaa\n\nyoutube bbbbbbbbbbb\n")
    assert forget_download(tmp_path, "aaaaaaaaaaa") is True
    assert read_archive_ids(archive_path(tmp_path)) == {"bbbbbbbbbbb"}


def test_forget_download_leaves_no_partial_file_behind(tmp_path):
    # The rewrite goes through a temp file so a kill cannot truncate the
    # archive; on success that temp file must be gone, not left as litter that
    # a later run could mistake for state.
    write_archive(tmp_path, "youtube aaaaaaaaaaa\nyoutube bbbbbbbbbbb\n")
    forget_download(tmp_path, "aaaaaaaaaaa")
    assert list(tmp_path.glob("downloaded.txt.part")) == []
    assert archive_path(tmp_path).exists()


# --------------------------------------------------------------------------- #
# read_manifest_ids
# --------------------------------------------------------------------------- #


def test_read_manifest_ids_keeps_order_and_drops_duplicates(tmp_path):
    # The archive is only re-read at startup, so a duplicate id would be
    # downloaded twice in one run -- wasted bandwidth against YouTube.
    (tmp_path / "manifest.csv").write_text(
        "track_id,youtube_id,n_sections,total_sec\n"
        "0002.aaaaaaaaaaa,aaaaaaaaaaa,11,429.964\n"
        "0003.bbbbbbbbbbb,bbbbbbbbbbb,13,542.720\n"
        "0004.aaaaaaaaaaa,aaaaaaaaaaa,11,429.964\n"
        "0005.,,0,0.000\n",
        encoding="utf-8",
        newline="",
    )
    assert read_manifest_ids(tmp_path) == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
