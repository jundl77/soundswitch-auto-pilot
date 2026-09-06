"""Tests for the third-party downloader (``tp_download``) and its supervisor.

Everything here is a behaviour the raveform downloader paid for once already:
an interrupted track must never be recorded as failed (the record is what a
plain re-run skips), a permanent refusal must outrank a transient one (or a
dead video is re-polled forever and aborts a healthy sweep), and a wall of
refusals must stop the run rather than burn the work list into failure records.
The classifier is imported rather than copied, and the supervisor's retry set
is the downloader's own object -- both are asserted by identity so a future
copy-paste fails here instead of stranding recoverable failures in the field.

No test touches the network: yt-dlp is a monkeypatched ``subprocess.run``.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
for _path in (str(REPO_ROOT), str(TRAINING_DIR), str(TRAINING_DIR / "raveform"),
              str(TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import raveform_download  # noqa: E402
import tp_download as download  # noqa: E402
import tp_fetch_queue as queue  # noqa: E402
import tp_supervisor as supervisor  # noqa: E402

BOT_CHECK_TEXT = "ERROR: Sign in to confirm you're not a bot"
PRIVATE_TEXT = ("ERROR: Private video. Sign in if you've been granted access. "
                "Use --cookies-from-browser")
AGE_TEXT = "ERROR: Sign in to confirm your age. This video may be inappropriate"


class FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def make_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "raveform"
    (corpus / "third_party").mkdir(parents=True, exist_ok=True)
    return corpus


def make_row(track_id="hx-0001_a", source="harmonix", video="AAA"):
    return queue.QueueRow(track_id=track_id, source=source, youtube_id=video,
                          title="Title", artist="Artist",
                          annotation_duration_sec="142.47", note="align=0.99")


def make_queue(tmp_path: Path, corpus: Path, rows) -> Path:
    target = queue.queue_path(corpus)
    queue.write_queue(target, rows)
    return target


def output_path(command: list) -> Path:
    return Path(command[command.index("-o") + 1])


def stub_yt_dlp(monkeypatch, outcomes, attempts: list):
    def fake_run(command, **_kwargs):
        video = command[-1]
        attempts.append(video)
        outcome = outcomes(video) if callable(outcomes) else outcomes
        if outcome == "ok":
            target = output_path(command).with_suffix(".mp3")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\xff\xfb" + b"audio" * 100)
            return FakeCompleted(0, stdout="[ExtractAudio] Destination")
        return FakeCompleted(1, stderr=outcome)

    monkeypatch.setattr(download.subprocess, "run", fake_run)


def read_failures(corpus: Path) -> list:
    path = download.failed_path(corpus)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run_main(corpus: Path, *extra) -> int:
    return download.main(["--data-dir", str(corpus), "--pause", "0", *extra])


def test_a_successful_download_lands_the_mp3_and_records_the_track_id(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [make_row()])
    attempts = []
    stub_yt_dlp(monkeypatch, "ok", attempts)

    assert run_main(corpus) == 0
    assert attempts == ["AAA"]
    assert download.audio_file(corpus, "harmonix", "hx-0001_a").stat().st_size > 0
    assert download.read_archive_ids(download.archive_path(corpus)) == {"hx-0001_a"}
    assert read_failures(corpus) == []


def test_two_tracks_sharing_one_video_are_both_downloaded(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [
        make_row("hx-0001_a", video="SHARED"),
        make_row("hx-0002_b", video="SHARED"),
    ])
    attempts = []
    stub_yt_dlp(monkeypatch, "ok", attempts)

    assert run_main(corpus) == 0
    assert attempts == ["SHARED", "SHARED"]
    assert download.audio_file(corpus, "harmonix", "hx-0001_a").exists()
    assert download.audio_file(corpus, "harmonix", "hx-0002_b").exists()


def test_an_interrupted_track_is_never_recorded_as_failed(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [make_row("hx-0001_a"), make_row("hx-0002_b", video="BBB")])
    attempts = []
    stub_yt_dlp(monkeypatch, BOT_CHECK_TEXT, attempts)
    monkeypatch.setattr(download, "interrupt_requested", lambda: bool(attempts))

    assert run_main(corpus) == 130
    assert attempts == ["AAA"]
    assert read_failures(corpus) == []
    assert download.read_archive_ids(download.archive_path(corpus)) == set()


def test_a_permanent_refusal_outranks_the_transient_text_it_carries():
    assert download.classify_error is raveform_download.classify_error
    assert download.classify_error(PRIVATE_TEXT) == "unavailable"
    assert download.classify_error(AGE_TEXT) == "age_restricted"
    assert download.classify_error(BOT_CHECK_TEXT) == "bot_check"
    assert "unavailable" not in download.RETRYABLE_REASONS
    assert "age_restricted" not in download.RETRYABLE_REASONS
    assert "bot_check" in download.RETRYABLE_REASONS


def test_a_permanent_failure_does_not_count_toward_the_refusal_wall(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [make_row(f"hx-000{n}_x", video=f"V{n}")
                                  for n in range(1, 5)])
    attempts = []
    stub_yt_dlp(monkeypatch, PRIVATE_TEXT, attempts)

    assert run_main(corpus, "--max-consecutive-failures", "2") == 0
    assert len(attempts) == 4
    assert {record["reason"] for record in read_failures(corpus)} == {"unavailable"}


def test_a_wall_of_consecutive_refusals_aborts_the_run(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [make_row(f"hx-000{n}_x", video=f"V{n}")
                                  for n in range(1, 6)])
    attempts = []
    stub_yt_dlp(monkeypatch, BOT_CHECK_TEXT, attempts)

    assert run_main(corpus, "--max-consecutive-failures", "2") == 2
    assert len(attempts) == 2
    failures = read_failures(corpus)
    assert len(failures) == 2
    assert {record["reason"] for record in failures} == {"bot_check"}


def test_a_success_resets_the_consecutive_refusal_counter(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [make_row(f"hx-000{n}_x", video=f"V{n}")
                                  for n in range(1, 5)])
    attempts = []
    stub_yt_dlp(monkeypatch,
                lambda video: "ok" if video == "V2" else BOT_CHECK_TEXT,
                attempts)

    assert run_main(corpus, "--max-consecutive-failures", "2") == 2
    assert attempts == ["V1", "V2", "V3", "V4"]


def test_a_re_run_skips_the_tracks_already_downloaded(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [make_row("hx-0001_a", video="AAA"),
                                  make_row("hx-0002_b", video="BBB")])
    attempts = []
    stub_yt_dlp(monkeypatch, "ok", attempts)
    assert run_main(corpus, "--limit", "1") == 0
    assert attempts == ["AAA"]

    again = []
    stub_yt_dlp(monkeypatch, "ok", again)
    assert run_main(corpus) == 0
    assert again == ["BBB"]


def test_a_re_run_skips_a_recorded_failure_until_its_reason_is_retried(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [make_row("hx-0001_a", video="AAA")])
    attempts = []
    stub_yt_dlp(monkeypatch, BOT_CHECK_TEXT, attempts)
    assert run_main(corpus) == 0
    assert attempts == ["AAA"]

    skipped = []
    stub_yt_dlp(monkeypatch, BOT_CHECK_TEXT, skipped)
    assert run_main(corpus) == 0
    assert skipped == []

    retried = []
    stub_yt_dlp(monkeypatch, "ok", retried)
    assert run_main(corpus, "--retry-reasons", "bot_check") == 0
    assert retried == ["AAA"]


def test_an_mp3_already_on_disk_is_never_re_fetched(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [make_row("hx-0001_a")])
    existing = download.audio_file(corpus, "harmonix", "hx-0001_a")
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"already here")
    attempts = []
    stub_yt_dlp(monkeypatch, "ok", attempts)

    assert run_main(corpus) == 0
    assert attempts == []
    assert existing.read_bytes() == b"already here"


def test_a_zero_byte_mp3_is_deleted_and_recorded_as_empty_output(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    make_queue(tmp_path, corpus, [make_row("hx-0001_a")])

    def fake_run(command, **_kwargs):
        target = output_path(command).with_suffix(".mp3")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
        return FakeCompleted(0)

    monkeypatch.setattr(download.subprocess, "run", fake_run)

    assert run_main(corpus) == 0
    assert not download.audio_file(corpus, "harmonix", "hx-0001_a").exists()
    assert [record["reason"] for record in read_failures(corpus)] == ["empty_output"]
    assert download.read_archive_ids(download.archive_path(corpus)) == set()


def test_the_downloader_never_passes_a_cookie_or_credential_flag(tmp_path):
    corpus = make_corpus(tmp_path)
    command = download.build_command(corpus, make_row())
    joined = " ".join(command)
    assert "cookies" not in joined
    assert "--username" not in joined
    assert "--password" not in joined
    assert command[-2] == "--"
    assert command[-1] == "AAA"


def test_the_output_template_is_keyed_on_track_id_not_the_video_id(tmp_path):
    corpus = make_corpus(tmp_path)
    command = download.build_command(corpus, make_row("hx-0001_a", video="SHARED"))
    target = output_path(command)
    assert target.name == "hx-0001_a.%(ext)s"
    assert target.parent.name == "harmonix"
    assert "--download-archive" not in command


def test_the_supervisor_retry_set_is_the_downloaders_own_object():
    assert supervisor.RETRY_REASONS is download.RETRYABLE_REASONS
    assert download.RETRYABLE_REASONS is raveform_download.RETRYABLE_REASONS
    assert supervisor.retry_reasons_arg() == ",".join(sorted(download.RETRYABLE_REASONS))


def test_the_supervisor_writes_a_fresh_cycle_log_per_relaunch(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    launched = []

    def fake_popen(command, stdout=None, stderr=None, cwd=None):
        launched.append(Path(stdout.name).name)

        class Child:
            pid = 4242

            def wait(self):
                return supervisor.EXIT_BLOCKED

        return Child()

    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)
    command = supervisor.build_command("python", supervisor.DOWNLOADER, corpus, None)
    log_path = supervisor.work_dir(corpus) / supervisor.LOG_FILE
    state_path = supervisor.work_dir(corpus) / supervisor.STATE_FILE
    for cycle in (1, 2, 3):
        supervisor.run_cycle(command, corpus, cycle, log_path, state_path)

    assert launched == ["tp_download.cycle1.log", "tp_download.cycle2.log",
                        "tp_download.cycle3.log"]
    assert len(set(launched)) == 3


def test_the_supervisor_command_is_gentle_and_carries_the_retry_reasons(tmp_path):
    corpus = make_corpus(tmp_path)
    command = supervisor.build_command("python", supervisor.DOWNLOADER, corpus, None)
    assert command[command.index("--retry-reasons") + 1] == supervisor.retry_reasons_arg()
    assert float(command[command.index("--pause") + 1]) >= 5.0
    assert "cookies" not in " ".join(command)
