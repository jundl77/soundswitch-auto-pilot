#!/usr/bin/env python
"""Patiently resume the third-party download after a refusal wall, unattended.

The only lever is time: no credential workaround is ever attempted, here or in
the downloader.  The retry set is the downloader's own ``RETRYABLE_REASONS``
object rather than a copy of it -- a supervisor holding its own list is exactly
how the raveform ops pair once came within one refresh of stranding every
recoverable 403 (see CLAUDE.md).

Each relaunch writes a fresh cycle log.  Re-issuing a redirect to the same name
truncates it, which destroys the previous cycle's evidence at the moment it
matters most.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

_TRAINING_DIR = Path(__file__).resolve().parents[1]
for _path in (str(_TRAINING_DIR.parent), str(_TRAINING_DIR),
              str(_TRAINING_DIR / "raveform"), str(_TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tp_download  # noqa: E402
import tp_record  # noqa: E402
from corpus_root import corpus_dir  # noqa: E402

DOWNLOADER = Path(__file__).resolve().parent / "tp_download.py"
STATE_FILE = "tp_supervisor.state"
LOG_FILE = "tp_supervisor.log"
CYCLE_LOG_STEM = "tp_download.cycle"

RETRY_REASONS = tp_download.RETRYABLE_REASONS

# Cycle 1 starts at once; a fresh launch has no wall to wait out yet.
DEFAULT_COOLDOWNS = (0, 45, 90, 180, 180, 180)

GENTLE_PAUSE = "6"

EXIT_CLEAN = 0
EXIT_BLOCKED = 2

_POLL_SECONDS = 5.0

_interrupted = False


def _request_stop(_signum=None, _frame=None) -> None:
    global _interrupted
    _interrupted = True


def install_signal_handlers() -> None:
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            signal.signal(number, _request_stop)
        except (ValueError, OSError, AttributeError):
            pass


def retry_reasons_arg() -> str:
    return ",".join(sorted(RETRY_REASONS))


def work_dir(corpus: Path) -> Path:
    return tp_record.third_party_dir(Path(corpus))


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(log_path: Path, event: str) -> None:
    line = f"{_stamp()}  {event}"
    try:
        with open(log_path, "a", encoding="ascii", errors="replace",
                  newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
    except OSError:
        pass
    print(line, flush=True)


def write_state(state_path: Path, text: str) -> None:
    try:
        with open(state_path, "w", encoding="ascii", errors="replace",
                  newline="\n") as handle:
            handle.write(text + "\n")
            handle.flush()
    except OSError:
        pass


def count_mp3s(corpus: Path) -> int:
    try:
        return sum(1 for _ in (work_dir(corpus) / tp_record.AUDIO_DIR)
                   .glob("*/*.mp3"))
    except OSError:
        return -1


def hhmm(when: float) -> str:
    return time.strftime("%H:%M", time.localtime(when))


def sleep_until(deadline: float) -> None:
    while not _interrupted:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        time.sleep(min(_POLL_SECONDS, remaining))


def build_command(python: str, downloader: Path, corpus: Path,
                  queue: Path | None) -> list:
    command = [
        python, "-u", str(downloader),
        "--data-dir", str(corpus),
        "--pause", GENTLE_PAUSE,
        "--retry-reasons", retry_reasons_arg(),
    ]
    if queue is not None:
        command += ["--queue", str(queue)]
    return command


def run_cycle(command: list, corpus: Path, cycle: int, log_path: Path,
              state_path: Path) -> int:
    cycle_log = work_dir(corpus) / f"{CYCLE_LOG_STEM}{cycle}.log"
    cycle_err = work_dir(corpus) / f"{CYCLE_LOG_STEM}{cycle}.err.log"
    log(log_path, f"LAUNCH   cycle {cycle}: {' '.join(command)}")
    log(log_path, f"         child stdout -> {cycle_log.name}, "
                  f"stderr -> {cycle_err.name}")
    try:
        with open(cycle_log, "w", encoding="utf-8", errors="replace") as out, \
             open(cycle_err, "w", encoding="utf-8", errors="replace") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err,
                                     cwd=str(work_dir(corpus)))
            write_state(state_path, f"RUNNING pid={child.pid}")
            log(log_path, f"         child pid {child.pid}")
            while True:
                try:
                    return child.wait()
                except KeyboardInterrupt:
                    _request_stop()
                    continue
    except OSError as error:
        log(log_path, f"CHILD_EXIT code -1  (could not run the downloader: {error})")
        return -1


def parse_cooldowns(text: str) -> tuple:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("need at least one cool-down")
    if any(value < 0 for value in values):
        raise ValueError("cool-downs must not be negative")
    return tuple(values)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="corpus root (default: the resolved corpus dir)")
    parser.add_argument("--downloader", type=Path, default=DOWNLOADER,
                        help="downloader script (default: %(default)s)")
    parser.add_argument("--queue", type=Path, default=None,
                        help="fetch queue csv passed through to the downloader")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter for the child (default: the one running this)")
    parser.add_argument("--cooldowns",
                        default=",".join(str(m) for m in DEFAULT_COOLDOWNS),
                        help="comma list of cool-down minutes, one per cycle; its "
                             "length is the cycle cap (default: %(default)s)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    corpus = (args.data_dir or corpus_dir()).resolve()
    downloader = Path(args.downloader).resolve()
    work_dir(corpus).mkdir(parents=True, exist_ok=True)
    log_path = work_dir(corpus) / LOG_FILE
    state_path = work_dir(corpus) / STATE_FILE

    try:
        cooldowns = parse_cooldowns(args.cooldowns)
    except ValueError as error:
        parser.error(str(error))

    install_signal_handlers()

    log(log_path, "=" * 70)
    log(log_path, f"SUPERVISOR start  corpus {corpus}")
    log(log_path, f"         python {args.python}")
    log(log_path, f"         downloader {downloader}")
    log(log_path, f"         retry reasons {retry_reasons_arg()}")
    log(log_path, f"         cycles {len(cooldowns)}, cool-downs "
                  f"{'/'.join(f'{m:g}m' for m in cooldowns)}")
    log(log_path, f"         quarantine now {count_mp3s(corpus)} mp3 on disk")

    if not downloader.is_file():
        log(log_path, f"FATAL    downloader not found at {downloader}")
        write_state(state_path, "FAILED (preflight: downloader not found)")
        return 3
    if shutil.which("yt-dlp") is None:
        log(log_path, "FATAL    yt-dlp not on PATH -- every cycle would fail identically")
        write_state(state_path, "FAILED (preflight: yt-dlp not on PATH)")
        return 3
    if shutil.which("deno") is None and shutil.which("node") is None:
        log(log_path, "WARN     no JS runtime visible to this process -- yt-dlp will "
                      "warn that some formats may be missing")

    command = build_command(args.python, downloader, corpus, args.queue)

    for index, minutes in enumerate(cooldowns, start=1):
        if _interrupted:
            break

        if minutes > 0:
            deadline = time.time() + minutes * 60.0
            log(log_path, f"COOLDOWN until {hhmm(deadline)}  ({minutes:g} min, before "
                          f"cycle {index}/{len(cooldowns)})")
            write_state(state_path, f"COOLDOWN until {hhmm(deadline)}")
            sleep_until(deadline)
            if _interrupted:
                break

        code = run_cycle(command, corpus, index, log_path, state_path)
        on_disk = count_mp3s(corpus)
        if code == EXIT_CLEAN:
            log(log_path, f"CHILD_EXIT code 0  (swept the queue; {on_disk} mp3 on disk)")
            log(log_path, f"DONE     after {index} cycle(s); {on_disk} mp3 on disk")
            write_state(state_path, f"DONE mp3s={on_disk}")
            return 0
        if code == EXIT_BLOCKED:
            log(log_path, f"CHILD_EXIT code 2  (refusal guard tripped again; "
                          f"{on_disk} mp3 on disk)")
        else:
            log(log_path, f"CHILD_EXIT code {code}  (unexpected; treating as a block; "
                          f"{on_disk} mp3 on disk)")
        if _interrupted:
            break

    on_disk = count_mp3s(corpus)
    if _interrupted:
        log(log_path, f"STOPPED  by signal; {on_disk} mp3 on disk. State is resumable "
                      "-- re-run this supervisor to continue.")
        write_state(state_path, f"STOPPED (signal) mp3s={on_disk}")
        return 130

    log(log_path, f"GAVE_UP  after {len(cooldowns)} cycles; {on_disk} mp3 on disk. "
                  "YouTube is still refusing. No credential workaround was attempted "
                  "-- this is an owner decision.")
    write_state(state_path, f"GAVE_UP after {len(cooldowns)} cycles")
    return 1


if __name__ == "__main__":
    sys.exit(main())
