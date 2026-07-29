#!/usr/bin/env python
"""Patiently resume the raveform download after a bot-check wall, unattended."""

# A copy of this file also runs beside the corpus; keep the two byte-identical
# (see CLAUDE.md).  No credential workaround is ever attempted here -- the only
# lever is time.

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

DOWNLOADER_FILE = "raveform_download.py"
STATE_FILE = "supervisor.state"
LOG_FILE = "supervisor.log"
AUDIO_DIR = "audio"

DEFAULT_COOLDOWNS = (45, 90, 180, 180, 180)

GENTLE_SLEEP_MIN = "6"
GENTLE_SLEEP_MAX = "12"

_RETRY_REASONS_FALLBACK = "bot_check,empty_output,http_403,missing_output,other,timeout"


def _retry_reasons(data_dir: Path) -> str:
    path = data_dir / DOWNLOADER_FILE
    # Importing writes a .pyc beside the source, and this supervisor puts
    # nothing in the corpus directory but its own log and state files.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_raveform_download_probe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reasons = sorted(module.RETRYABLE_REASONS)
        if reasons:
            return ",".join(reasons)
    except Exception:
        pass
    finally:
        sys.dont_write_bytecode = previous
    return _RETRY_REASONS_FALLBACK

EXIT_CLEAN = 0
EXIT_BLOCKED = 2

_POLL_SECONDS = 5.0


_interrupted = False


def _request_stop(_signum=None, _frame=None) -> None:
    global _interrupted
    _interrupted = True


def install_signal_handlers() -> None:
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        handler = getattr(signal, name, None)
        if handler is None:
            continue
        try:
            signal.signal(handler, _request_stop)
        except (ValueError, OSError, AttributeError):
            pass


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(log_path: Path, event: str) -> None:
    line = f"{_stamp()}  {event}"
    try:
        with open(log_path, "a", encoding="ascii", errors="replace", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
    except OSError:
        pass
    print(line, flush=True)


def write_state(state_path: Path, text: str) -> None:
    try:
        with open(state_path, "w", encoding="ascii", errors="replace", newline="\n") as handle:
            handle.write(text + "\n")
            handle.flush()
    except OSError:
        pass


def count_mp3s(data_dir: Path) -> int:
    try:
        return sum(1 for _ in (data_dir / AUDIO_DIR).glob("*.mp3"))
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


def build_command(python: str, downloader: Path, data_dir: Path) -> list:
    return [
        python,
        str(downloader),
        "--data-dir", str(data_dir),
        "--sleep-min", GENTLE_SLEEP_MIN,
        "--sleep-max", GENTLE_SLEEP_MAX,
        "--retry-reasons", _retry_reasons(data_dir),
    ]


def run_cycle(command: list, data_dir: Path, cycle: int, log_path: Path, state_path: Path) -> int:
    cycle_log = data_dir / f"download.cycle{cycle}.log"
    cycle_err = data_dir / f"download.cycle{cycle}.err.log"
    log(log_path, f"LAUNCH   cycle {cycle}: {' '.join(command)}")
    log(log_path, f"         child stdout -> {cycle_log.name}, stderr -> {cycle_err.name}")
    try:
        with open(cycle_log, "w", encoding="utf-8", errors="replace") as out, \
             open(cycle_err, "w", encoding="utf-8", errors="replace") as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, cwd=str(data_dir))
            write_state(state_path, f"RUNNING pid={child.pid}")
            log(log_path, f"         child pid {child.pid}")
            while True:
                try:
                    return child.wait()
                except KeyboardInterrupt:
                    # Ctrl-C reaches the child too on Windows; let it finish
                    # tearing down so its own resume state stays consistent.
                    _request_stop()
                    continue
    except OSError as error:
        log(log_path, f"CHILD_EXIT code -1  (could not run the downloader: {error})")
        return -1


def preflight_failed_state(reason: str) -> str:
    return f"FAILED (preflight: {reason})"


def terminal_state(interrupted: bool, cycles: int, on_disk: int) -> str:
    if interrupted:
        return f"STOPPED (signal) mp3s={on_disk}"
    return f"GAVE_UP after {cycles} cycles"


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
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=here,
                        help="corpus root (default: this script's directory)")
    parser.add_argument("--downloader", type=Path, default=None,
                        help=f"downloader script (default: <data-dir>/{DOWNLOADER_FILE})")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter for the child (default: the one running this)")
    parser.add_argument("--cooldowns", default=",".join(str(m) for m in DEFAULT_COOLDOWNS),
                        help="comma list of cool-down minutes, one per cycle; its length is "
                             "the cycle cap (default: %(default)s)")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    data_dir = args.data_dir.resolve()
    downloader = (args.downloader or (data_dir / DOWNLOADER_FILE)).resolve()
    log_path = data_dir / LOG_FILE
    state_path = data_dir / STATE_FILE

    try:
        cooldowns = parse_cooldowns(args.cooldowns)
    except ValueError as error:
        parser.error(str(error))

    install_signal_handlers()

    log(log_path, "=" * 70)
    log(log_path, f"SUPERVISOR start  data-dir {data_dir}")
    log(log_path, f"         python {args.python}")
    log(log_path, f"         cycles {len(cooldowns)}, cool-downs "
                  f"{'/'.join(f'{m:g}m' for m in cooldowns)}")
    log(log_path, f"         corpus now {count_mp3s(data_dir)} mp3 on disk")

    if not downloader.is_file():
        log(log_path, f"FATAL    downloader not found at {downloader}")
        write_state(state_path, preflight_failed_state("downloader not found"))
        return 3
    if shutil.which("yt-dlp") is None:
        log(log_path, "FATAL    yt-dlp not on PATH -- every cycle would fail identically")
        write_state(state_path, preflight_failed_state("yt-dlp not on PATH"))
        return 3
    if shutil.which("deno") is None:
        log(log_path, "WARN     no deno on PATH -- yt-dlp will warn that some formats may be missing")

    command = build_command(args.python, downloader, data_dir)

    for index, minutes in enumerate(cooldowns, start=1):
        if _interrupted:
            break

        deadline = time.time() + minutes * 60.0
        log(log_path, f"COOLDOWN until {hhmm(deadline)}  ({minutes:g} min, before cycle "
                      f"{index}/{len(cooldowns)})")
        write_state(state_path, f"COOLDOWN until {hhmm(deadline)}")
        sleep_until(deadline)
        if _interrupted:
            break

        code = run_cycle(command, data_dir, index, log_path, state_path)
        on_disk = count_mp3s(data_dir)
        if code == EXIT_CLEAN:
            log(log_path, f"CHILD_EXIT code 0  (swept the manifest; {on_disk} mp3 on disk)")
            log(log_path, f"DONE     after {index} cycle(s); {on_disk} mp3 on disk")
            write_state(state_path, f"DONE mp3s={on_disk}")
            return 0
        if code == EXIT_BLOCKED:
            log(log_path, f"CHILD_EXIT code 2  (block guard tripped again; {on_disk} mp3 on disk)")
        else:
            log(log_path, f"CHILD_EXIT code {code}  (unexpected; treating as a block; "
                          f"{on_disk} mp3 on disk)")
        if _interrupted:
            break

    on_disk = count_mp3s(data_dir)
    if _interrupted:
        log(log_path, f"STOPPED  by signal; {on_disk} mp3 on disk. State is resumable -- "
                      "re-run this supervisor to continue.")
        write_state(state_path, terminal_state(True, len(cooldowns), on_disk))
        return 130

    log(log_path, f"GAVE_UP  after {len(cooldowns)} cycles; {on_disk} mp3 on disk. "
                  "YouTube is still refusing. No credential workaround was attempted -- "
                  "this is an owner decision.")
    write_state(state_path, terminal_state(False, len(cooldowns), on_disk))
    return 1


if __name__ == "__main__":
    sys.exit(main())
