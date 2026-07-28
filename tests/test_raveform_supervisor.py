"""Tests for the download supervisor (training/raveform/raveform_supervisor.py).

The supervisor is operational tooling -- it waits, relaunches the downloader,
and reads its exit code -- so most of it is wall-clock and subprocess work that
is not worth simulating.  Two pure functions are worth pinning, because both
decide what a multi-hour unattended run actually does:

``_retry_reasons``
    Which failures a supervised refresh reclaims.  This is the one that already
    went wrong: the supervisor kept its own hardcoded copy of the list, the
    downloader grew an ``http_403`` bucket, and the copy never heard about it --
    a refresh would have re-recorded every 403 and then retried none of them,
    stranding 62 obtainable tracks while the branch looked correct.  It now
    reads the constant out of the downloader it is about to launch, and these
    tests pin both that derivation and its fallback.

``parse_cooldowns``
    The length of its result IS the cycle cap, so a parsing slip is the
    difference between a patient retry schedule and an unbounded one.
"""
import sys
from pathlib import Path

import pytest

RAVEFORM_DIR = Path(__file__).resolve().parents[1] / "training" / "raveform"
if str(RAVEFORM_DIR) not in sys.path:
    sys.path.insert(0, str(RAVEFORM_DIR))

from raveform_supervisor import (  # noqa: E402  (needs the path insert above)
    DOWNLOADER_FILE,
    _RETRY_REASONS_FALLBACK,
    _retry_reasons,
    parse_cooldowns,
    preflight_failed_state,
    terminal_state,
)
from raveform_download import RETRYABLE_REASONS  # noqa: E402


def _downloader(data_dir: Path, body: str) -> Path:
    path = data_dir / DOWNLOADER_FILE
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# _retry_reasons -- derived from the downloader that will actually run
# --------------------------------------------------------------------------- #


def test_the_reasons_come_from_the_downloader_beside_the_corpus(tmp_path):
    # Deliberately NOT the real set: the point is that the value is read from
    # the file on disk, not baked in here or imported from the branch.
    _downloader(tmp_path, 'RETRYABLE_REASONS = frozenset({"zebra", "alpha"})\n')
    assert _retry_reasons(tmp_path) == "alpha,zebra"


def test_the_reasons_are_sorted_so_a_relaunch_is_reproducible(tmp_path):
    _downloader(tmp_path, 'RETRYABLE_REASONS = frozenset({"c", "a", "b"})\n')
    assert _retry_reasons(tmp_path) == "a,b,c"


def test_the_real_downloader_yields_the_real_recoverable_set():
    # The pair that matters in production: the supervisor's argument and the
    # downloader's own notion of what is worth retrying must be the same set.
    assert _retry_reasons(RAVEFORM_DIR) == ",".join(sorted(RETRYABLE_REASONS))
    assert "http_403" in _retry_reasons(RAVEFORM_DIR)
    assert "other" in _retry_reasons(RAVEFORM_DIR)


def test_an_absent_downloader_falls_back_rather_than_crashing(tmp_path):
    # A supervised run must not die at launch because the probe failed; the
    # fallback is the same list, just frozen.
    assert _retry_reasons(tmp_path) == _RETRY_REASONS_FALLBACK


def test_a_downloader_without_the_constant_falls_back(tmp_path):
    # An older downloader predating RETRYABLE_REASONS.
    _downloader(tmp_path, "SOMETHING_ELSE = 1\n")
    assert _retry_reasons(tmp_path) == _RETRY_REASONS_FALLBACK


def test_a_broken_downloader_falls_back(tmp_path):
    _downloader(tmp_path, "this is not python(\n")
    assert _retry_reasons(tmp_path) == _RETRY_REASONS_FALLBACK


def test_an_empty_reason_set_falls_back(tmp_path):
    # "" would mean --retry-reasons with no value: argparse would take the next
    # flag as its argument, or the run would reclaim nothing at all.
    _downloader(tmp_path, "RETRYABLE_REASONS = frozenset()\n")
    assert _retry_reasons(tmp_path) == _RETRY_REASONS_FALLBACK


def test_the_frozen_fallback_still_matches_the_downloader():
    # The fallback is a copy, and copies drift -- that is the whole bug this
    # function exists to prevent, so the copy itself gets a guard too.
    assert _RETRY_REASONS_FALLBACK == ",".join(sorted(RETRYABLE_REASONS))


def test_probing_the_downloader_writes_nothing_beside_the_corpus(tmp_path):
    # The supervisor writes only logs and state into the data directory.
    # Importing a file normally drops a __pycache__ next to it, which would
    # also read as corpus content to anything sweeping the directory.
    _downloader(tmp_path, 'RETRYABLE_REASONS = frozenset({"a"})\n')
    before = {path.name for path in tmp_path.iterdir()}
    _retry_reasons(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == before
    assert not (tmp_path / "__pycache__").exists()


def test_probing_restores_the_interpreters_bytecode_setting(tmp_path):
    _downloader(tmp_path, 'RETRYABLE_REASONS = frozenset({"a"})\n')
    for setting in (False, True):
        sys.dont_write_bytecode = setting
        try:
            _retry_reasons(tmp_path)
            assert sys.dont_write_bytecode is setting
        finally:
            sys.dont_write_bytecode = False


# --------------------------------------------------------------------------- #
# terminal_state -- supervisor.state is the channel monitors actually read
# --------------------------------------------------------------------------- #


def test_an_interrupt_is_reported_as_stopped_not_as_giving_up():
    # A human stopping the run, possibly seconds in with every cycle unspent,
    # must not be recorded as "the schedule ran out and YouTube still refuses".
    # supervisor.state is a single overwritten line and often the only thing
    # read, so the two words have to stay distinguishable.
    state = terminal_state(interrupted=True, cycles=5, on_disk=1387)
    assert state.startswith("STOPPED")
    assert "GAVE_UP" not in state
    assert "1387" in state  # the useful fact for whoever resumes


def test_an_exhausted_schedule_is_reported_as_giving_up():
    state = terminal_state(interrupted=False, cycles=5, on_disk=1387)
    assert state.startswith("GAVE_UP")
    assert "5" in state


def test_the_two_terminal_states_are_never_the_same_string():
    assert terminal_state(True, 5, 10) != terminal_state(False, 5, 10)


def test_a_preflight_bailout_is_not_reported_as_giving_up():
    # Nothing was attempted at all, and the problem is this machine, not
    # YouTube. GAVE_UP would point the reader at an owner decision about a
    # refusing service when the actual fault is a broken install.
    state = preflight_failed_state("yt-dlp not on PATH")
    assert state.startswith("FAILED")
    assert "GAVE_UP" not in state and "STOPPED" not in state
    assert "yt-dlp not on PATH" in state  # the reader should not need the log


def test_every_terminal_state_word_is_distinct():
    # supervisor.state is one overwritten line and often the only thing read, so
    # its four outcomes -- finished, stopped, exhausted, never-started -- must
    # not be confusable with one another.
    words = {
        terminal_state(True, 5, 10).split()[0],
        terminal_state(False, 5, 10).split()[0],
        preflight_failed_state("x").split()[0],
    }
    assert len(words) == 3
    assert "DONE" not in words  # the success word, written elsewhere, stays free


# --------------------------------------------------------------------------- #
# parse_cooldowns -- its length is the cycle cap
# --------------------------------------------------------------------------- #


def test_cooldowns_parse_in_order():
    assert parse_cooldowns("45,90,180") == (45.0, 90.0, 180.0)


def test_whitespace_and_trailing_separators_are_tolerated():
    # Hand-typed on a command line; a stray comma must not silently add a
    # zero-length cool-down (an instant relaunch into a hardening block).
    assert parse_cooldowns(" 45 , 90 , ") == (45.0, 90.0)


def test_a_single_cooldown_means_a_single_cycle():
    assert parse_cooldowns("30") == (30.0,)


def test_fractional_cooldowns_are_allowed():
    # Used when smoke-testing the escalation without waiting 45 minutes.
    assert parse_cooldowns("0.1,0.2") == (0.1, 0.2)


def test_an_empty_schedule_is_rejected():
    # Would mean zero cycles: the supervisor would exit having done nothing,
    # looking exactly like a run that finished.
    with pytest.raises(ValueError):
        parse_cooldowns("")
    with pytest.raises(ValueError):
        parse_cooldowns("  ,  ")


def test_a_negative_cooldown_is_rejected():
    # sleep_until() would return immediately and the "patience" the whole
    # supervisor exists to apply would be silently skipped.
    with pytest.raises(ValueError):
        parse_cooldowns("45,-1")


def test_a_non_numeric_cooldown_is_rejected():
    with pytest.raises(ValueError):
        parse_cooldowns("45,soon")
