#!/usr/bin/env python
"""Supervise a patient resume of the raveform download after a bot-check wall.

The first full run downloaded 449 tracks in 1h20m and then hit five consecutive
sign-in refusals, which tripped the downloader's ``--max-consecutive-blocks``
guard and stopped it cleanly.  That guard did its job: the state on disk is
intact and resumable.  What is needed now is *patience*, applied unattended.

This supervisor waits, relaunches the downloader as a child process, reads its
exit code, and decides whether to wait longer or stop.  It is operational
tooling rather than application code, but it is versioned on the branch with
everything else: it lived only in the gitignored corpus directory once, and an
unreviewable copy of it carrying a stale retry list is exactly what would have
stranded 62 obtainable tracks on the next refresh.  A copy still runs beside the
corpus -- that is what a supervised refresh launches -- and it must be kept
byte-identical to the branch (``cmp``); see CLAUDE.md.

**No credential workaround is attempted, ever.**  The only lever pulled here is
time: a longer pause before retrying, and gentler pacing (~3x the sleep between
videos) once retrying.  If YouTube still refuses after the last cycle, the run
stops and leaves the decision to the owner.  Cookies, logins and IP tricks are
out of scope by project rule.

**Escalation.**  Cool-downs grow 45 -> 90 -> 180 -> 180 -> 180 minutes.  A block
that survives 45 minutes is a different kind of block from one that clears
immediately, and retrying a hardening wall on a fixed short interval is how a
soft block becomes a long one.  The list length is also the cycle cap.

**Everything recoverable is reclaimed; dead videos are not.**  Each relaunch
passes ``--retry-reasons`` naming every reason that describes *this run* rather
than the video, so ids recorded as ``unavailable``, ``age_restricted``,
``geo_blocked`` or ``copyright`` stay skipped.  A deleted video will not come
back, and re-polling dead ids every cycle is precisely the pointless traffic
that earns a block in the first place.

This list used to be ``bot_check`` alone, which was wrong in an expensive way:
``HTTP Error 403`` failures were bucketed as ``other`` and made up 56% of the
corpus sweep's failures, so a supervised refresh would have stranded every one
of them.  The set is now taken from the downloader's own
``RETRYABLE_REASONS`` -- one definition, so this file cannot drift from it again.

**The first thing it does is wait.**  The wall was tripped minutes ago;
relaunching immediately would walk straight back into it.

Stdlib only.  Writes ``supervisor.log`` (append, one line per event) and
``supervisor.state`` (a single overwritten line, for at-a-glance checks).

Usage::

    python raveform_supervisor.py --data-dir <corpus root>
"""

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

# One cool-down per relaunch cycle, in minutes.  The length of this tuple IS the
# cycle cap: five attempts spread over roughly 11 hours of waiting.
DEFAULT_COOLDOWNS = (45, 90, 180, 180, 180)

# Roughly 3x the pacing of the run that tripped the wall (which used 2-5 s).
GENTLE_SLEEP_MIN = "6"
GENTLE_SLEEP_MAX = "12"

# Every reason that describes THIS RUN rather than the video -- see the module
# docstring.  Read from the downloader sitting beside us so the two can never
# disagree; the literal below is only the fallback for an older downloader that
# predates the constant, and it is that constant's current value.
_RETRY_REASONS_FALLBACK = "bot_check,empty_output,http_403,missing_output,other,timeout"


def _retry_reasons(data_dir: Path) -> str:
    """``--retry-reasons`` value, taken from the downloader we actually launch.

    A hardcoded copy of this list is exactly how the supervisor came to strand
    every recoverable 403: the downloader grew a new reason bucket and this file
    never heard about it.  So import the real constant from the very file this
    supervisor is about to run -- not from PATH, not from the branch, but from
    the copy on disk beside the corpus, which is the one whose behaviour matters.
    """
    path = data_dir / DOWNLOADER_FILE
    # Importing a file writes a .pyc beside it by default, and this supervisor
    # writes nothing into the corpus directory that is not a log or state file.
    # A __pycache__ appearing there would also read as corpus content to
    # anything sweeping the directory, so the interpreter is told to stay quiet
    # for the duration of the load and restored afterwards.
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
        # An older or unreadable downloader must not stop the run; the fallback
        # is the same list, just frozen.
        pass
    finally:
        sys.dont_write_bytecode = previous
    return _RETRY_REASONS_FALLBACK

# The downloader's contract.  Anything else -- a crash, a missing dependency, an
# OS-level kill -- is treated like a block: wait and try again, but it still
# consumes one of the five cycles so a hard-broken downloader cannot spin.
EXIT_CLEAN = 0
EXIT_BLOCKED = 2

# Cool-downs are hours long; poll often enough that Ctrl-C feels immediate.
_POLL_SECONDS = 5.0


# --------------------------------------------------------------------------- #
# Interrupt handling
# --------------------------------------------------------------------------- #

_interrupted = False


def _request_stop(_signum=None, _frame=None) -> None:
    global _interrupted
    _interrupted = True


def install_signal_handlers() -> None:
    """Best-effort: a stop request must not raise out of the middle of a wait."""
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        handler = getattr(signal, name, None)
        if handler is None:
            continue
        try:
            signal.signal(handler, _request_stop)
        except (ValueError, OSError, AttributeError):
            pass


# --------------------------------------------------------------------------- #
# Log and state
# --------------------------------------------------------------------------- #
#
# Both are best-effort by design.  This process must survive a full disk, a
# locked file or an antivirus scanner holding a handle: losing a log line is an
# annoyance, but crashing the loop would abandon a multi-hour recovery.


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(log_path: Path, event: str) -> None:
    """Append one timestamped ASCII event line.  Never raises."""
    line = f"{_stamp()}  {event}"
    try:
        with open(log_path, "a", encoding="ascii", errors="replace", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
    except OSError:
        pass
    # Also to stdout, which the detached launch redirects to supervisor.out.log.
    print(line, flush=True)


def write_state(state_path: Path, text: str) -> None:
    """Overwrite the single-line state file.  Never raises."""
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


# --------------------------------------------------------------------------- #
# Waiting
# --------------------------------------------------------------------------- #


def hhmm(when: float) -> str:
    return time.strftime("%H:%M", time.localtime(when))


def sleep_until(deadline: float) -> None:
    """Sleep in short slices so a stop request is noticed within seconds."""
    while not _interrupted:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        time.sleep(min(_POLL_SECONDS, remaining))


# --------------------------------------------------------------------------- #
# The child run
# --------------------------------------------------------------------------- #


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
    """Run the downloader to completion.  Returns its exit code (or -1)."""
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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def terminal_state(interrupted: bool, cycles: int, on_disk: int) -> str:
    """The last line written to ``supervisor.state``.

    ``supervisor.state`` is the at-a-glance channel -- a single overwritten line,
    often the only thing anyone reads -- so its two failure words must not be
    interchangeable.  ``GAVE_UP`` means the schedule ran out and YouTube is
    still refusing: the retry budget was spent and what remains is an owner
    decision.  ``STOPPED`` means a human interrupted it, possibly seconds in,
    with every cycle still unspent.  Reporting an interrupt as exhaustion
    invents a budget that was never actually used.
    """
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

    # Fail loudly on a missing downloader: silently "supervising" nothing for
    # eleven hours is far worse than not starting.
    if not downloader.is_file():
        log(log_path, f"FATAL    downloader not found at {downloader}")
        write_state(state_path, "GAVE_UP after 0 cycles")
        return 3
    if shutil.which("yt-dlp") is None:
        log(log_path, "FATAL    yt-dlp not on PATH -- every cycle would fail identically")
        write_state(state_path, "GAVE_UP after 0 cycles")
        return 3
    if shutil.which("deno") is None:
        # Not fatal, but a corpus fetched without a JS runtime may be missing
        # formats, so it must be visible in the log rather than discovered later.
        log(log_path, "WARN     no deno on PATH -- yt-dlp will warn that some formats may be missing")

    command = build_command(args.python, downloader, data_dir)

    for index, minutes in enumerate(cooldowns, start=1):
        if _interrupted:
            break

        # The wall was tripped moments ago, so every cycle -- including the
        # first -- waits before it knocks.
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
            # Counts against the cap exactly like a block, so a downloader that
            # is simply broken cannot loop forever.
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
