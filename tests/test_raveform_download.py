import json
import sys
from pathlib import Path

RAVEFORM_DIR = Path(__file__).resolve().parents[1] / "training" / "raveform"
if str(RAVEFORM_DIR) not in sys.path:
    sys.path.insert(0, str(RAVEFORM_DIR))

from raveform_download import (  # noqa: E402
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


def test_sign_in_wall_is_a_bot_check():
    assert classify_error("ERROR: Sign in to confirm you're not a bot") == "bot_check"


def test_curly_apostrophe_sign_in_wall_is_a_bot_check():
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
    blob = "ERROR: Video unavailable. Sign in to confirm you're not a bot"
    assert classify_error(blob) == "bot_check"


def test_an_age_gate_outranks_the_sign_in_text_it_carries():
    blob = (
        "ERROR: [youtube] LuT4EqmJmnU: Sign in to confirm your age. "
        "This video may be inappropriate for some users."
    )
    assert classify_error(blob) == "age_restricted"


def test_a_real_bot_check_is_still_a_bot_check():
    blob = (
        "ERROR: [youtube] dQw4w9WgXcQ: Sign in to confirm you're not a bot. "
        "Use --cookies-from-browser or --cookies for the authentication."
    )
    assert classify_error(blob) == "bot_check"


def test_a_media_url_403_is_its_own_bucket_not_other():
    blob = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
    assert classify_error(blob) == "http_403"


def test_a_403_is_retryable_and_aborts_a_sustained_run():
    assert "http_403" in RETRYABLE_REASONS
    assert "http_403" in BLOCK_REASONS


def test_the_id_stays_behind_the_separator(tmp_path):
    # 19 manifest ids begin with "-"; without "--" yt-dlp parses them as options.
    argv = build_command(tmp_path, "-DWkf03g4Gc")
    assert argv[-2:] == ["--", "-DWkf03g4Gc"]


def test_the_non_pattern_reasons_are_all_retryable():
    assert {"other", "empty_output", "missing_output", "timeout"} <= RETRYABLE_REASONS


def test_the_retry_hint_names_every_recoverable_reason():
    for reason in RETRYABLE_REASONS:
        assert reason in RETRY_HINT
    assert RETRY_HINT.startswith("--retry-reasons ")


def test_every_reason_the_table_can_emit_is_a_known_reason():
    for reason, _patterns in _REASON_PATTERNS:
        assert reason in KNOWN_REASONS
    assert RETRYABLE_REASONS <= KNOWN_REASONS


def test_no_pattern_appears_in_two_buckets():
    seen = {}
    for reason, patterns in _REASON_PATTERNS:
        for pattern in patterns:
            if pattern in seen and seen[pattern] != reason:
                raise AssertionError(
                    f"pattern {pattern!r} is claimed by both {seen[pattern]!r} and {reason!r}"
                )
            seen[pattern] = reason


def test_a_bare_not_available_message_is_unavailable():
    blob = "ERROR: [youtube] A6O2p64sucM: This video is not available"
    assert classify_error(blob) == "unavailable"


def test_a_geo_block_is_not_swallowed_by_the_not_available_pattern():
    blob = "ERROR: This video is not available in your country"
    assert classify_error(blob) == "geo_blocked"


def test_a_private_video_outranks_the_cookie_advice_it_carries():
    blob = (
        "ERROR: [youtube] CJL6uHqLyfU: Private video. Sign in if you've been granted "
        "access to this video. Use --cookies-from-browser or --cookies for the "
        "authentication."
    )
    assert classify_error(blob) == "unavailable"


def test_copyright_ranks_below_unavailable():
    blob = "ERROR: Video unavailable due to a copyright claim"
    assert classify_error(blob) == "unavailable"


def test_yt_dlp_interrupt_text_is_not_classified_as_an_interrupt():
    assert classify_error("ERROR: Interrupted by user") == "other"


def test_postprocessing_abort_text_is_not_classified_as_an_interrupt():
    blob = "ERROR: Postprocessing: ffmpeg exited with code 255"
    assert classify_error(blob) != INTERRUPT_REASON
    assert classify_error(blob) == "other"


def test_no_pattern_bucket_yields_the_interrupt_reason():
    assert INTERRUPT_REASON not in {reason for reason, _patterns in _REASON_PATTERNS}


def test_the_interrupt_reason_is_not_a_recordable_reason():
    assert INTERRUPT_REASON not in KNOWN_REASONS


def test_every_bucket_classify_error_can_return_is_a_known_reason():
    buckets = {reason for reason, _patterns in _REASON_PATTERNS} | {"other"}
    assert buckets <= KNOWN_REASONS


def test_only_refusals_of_this_client_abort_the_run():
    permanent = {"unavailable", "age_restricted", "geo_blocked", "copyright"}
    assert BLOCK_REASONS.isdisjoint(permanent)
    assert BLOCK_REASONS <= RETRYABLE_REASONS
    assert BLOCK_REASONS <= KNOWN_REASONS
    assert "bot_check" in BLOCK_REASONS


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
    write_archive(tmp_path, "youtube aaaaaaaaaaa\nyoutube aaaaaaaaaaaXY\n")
    assert forget_download(tmp_path, "aaaaaaaaaaa") is True
    assert read_archive_ids(archive_path(tmp_path)) == {"aaaaaaaaaaaXY"}


def test_forget_download_tolerates_blank_lines(tmp_path):
    write_archive(tmp_path, "youtube aaaaaaaaaaa\n\nyoutube bbbbbbbbbbb\n")
    assert forget_download(tmp_path, "aaaaaaaaaaa") is True
    assert read_archive_ids(archive_path(tmp_path)) == {"bbbbbbbbbbb"}


def test_forget_download_leaves_no_partial_file_behind(tmp_path):
    write_archive(tmp_path, "youtube aaaaaaaaaaa\nyoutube bbbbbbbbbbb\n")
    forget_download(tmp_path, "aaaaaaaaaaa")
    assert list(tmp_path.glob("downloaded.txt.part")) == []
    assert archive_path(tmp_path).exists()


def test_read_manifest_ids_keeps_order_and_drops_duplicates(tmp_path):
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
