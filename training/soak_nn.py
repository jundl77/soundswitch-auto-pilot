"""The live soak: real audio through real hardware for half an hour, measured."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "training")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from eval_assets import committed_audio_path  # noqa: E402
from select_eval_set import EVAL_SET_FILE, load_eval_set  # noqa: E402

SAMPLE_INTERVAL_SEC = 1.0
BUCKET_SEC = 60.0
BUDGET_MS = 1000.0 * 256 / 44100

# One second of buffers, the count the madmom soak dropped.
WARMUP_BUFFERS = 172

DEFAULT_TRACK_COUNT = 5


def eval_tracks(count: int) -> list:
    document = load_eval_set(EVAL_SET_FILE)
    return list(document["tracks"])[:count]


def spread(values, scale: float = 1.0) -> dict:
    if len(values) == 0:
        return {}
    array = np.asarray(values, dtype=np.float64) * scale
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "p999": float(np.percentile(array, 99.9)),
        "max": float(array.max()),
    }


def buckets(stamps, values, width: float = BUCKET_SEC) -> list:
    if not stamps:
        return []
    out, start = [], 0
    edge = width
    for index, stamp in enumerate(stamps):
        if stamp < edge:
            continue
        out.append({"minute": round(edge / 60.0), "buffers": index - start,
                    "p99_ms": float(np.percentile(values[start:index], 99)),
                    "max_ms": float(np.max(values[start:index]))})
        start, edge = index, edge + width
    if len(stamps) - start > 1:
        out.append({"minute": round(edge / 60.0), "buffers": len(stamps) - start,
                    "p99_ms": float(np.percentile(values[start:], 99)),
                    "max_ms": float(np.max(values[start:]))})
    return out


def play(device: int, tracks: list, gap_sec: float, sample_rate: int,
         buffer_size: int) -> None:
    import pyaudio
    from simulate.fake_audio_client import FileAudioClient

    audio = pyaudio.PyAudio()
    name = audio.get_device_info_by_index(device)["name"]
    stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate,
                        output_device_index=device, output=True,
                        frames_per_buffer=buffer_size)
    silence = np.zeros(buffer_size, dtype=np.float32).tobytes()
    print(f"playing {len(tracks)} tracks into [{device}] {name}", flush=True)
    started = time.monotonic()
    try:
        for index, track in enumerate(tracks, start=1):
            path = committed_audio_path(track["youtube_id"])
            client = FileAudioClient(sample_rate, buffer_size, str(path))
            client.start_streams()
            print(f"  [{index}/{len(tracks)}] {track['track_id']} "
                  f"{client.duration_sec / 60:.1f}m  "
                  f"t+{time.monotonic() - started:.0f}s", flush=True)
            while not client.exhausted:
                stream.write(client.read().tobytes())
            for _ in range(int(gap_sec * sample_rate / buffer_size)):
                stream.write(silence)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    print(f"done, {(time.monotonic() - started) / 60:.1f} minutes", flush=True)


class Sampler(threading.Thread):
    def __init__(self, app, minutes: float):
        super().__init__(name="soak-sampler", daemon=True)
        self._app = app
        self._until = time.monotonic() + minutes * 60.0
        self.rows: list = []
        self.transitions: list = []

    def run(self) -> None:
        from lib.analyser.gpu_stage import reserved_bytes
        watchdog = self._app.drift_watchdog
        stage = None if self._app.section is None else self._app.section.stream
        started = time.monotonic()
        last = watchdog.level.name
        seen_running = False
        while time.monotonic() < self._until:
            time.sleep(SAMPLE_INTERVAL_SEC)
            if not self._app.is_running:
                if seen_running:
                    break
                continue
            seen_running = True
            now = round(time.monotonic() - started, 2)
            level = watchdog.level.name
            if level != last:
                self.transitions.append({"t": now, "from": last, "to": level,
                                         "fault": watchdog.fault})
                last = level
            row = {
                "t": now,
                "audio_sec": round(self._app.light_engine.audio_sec, 2),
                "drift_sec": round(watchdog.drift_sec, 4),
                "peak_drift_sec": round(watchdog.peak_drift_sec, 4),
                "total_drift_sec": round(watchdog.total_drift_sec, 3),
                "shed": level,
                "fault": watchdog.fault,
                "intent": (self._app.light_engine.current_intent.name
                           if self._app.light_engine.current_intent else None),
                "commits": self._app.light_engine.intent_commits,
            }
            if stage is not None:
                row.update(queued=stage.queued, passes=stage.passes,
                           faults=stage.faults, overflows=stage.overflows,
                           reinits=stage.reinits, resyncs=stage.resyncs,
                           reserved_bytes=reserved_bytes())
            self.rows.append(row)
        self._app.stop()


def _instrument(app) -> dict:
    record = {"stamps": [], "on_audio": [], "analyse": [], "drain": [],
              "loop": [], "between_reads": [], "boundaries": []}
    engine, analyser, queue = app.light_engine, app.music_analyser, app.command_queue
    on_audio, analyse, drain = engine.on_audio, analyser.analyse, queue.drain
    sound_start, sound_stop = engine.on_sound_start, engine.on_sound_stop
    read = app.audio_client.read
    started = time.perf_counter()
    top = [started]
    read_ended = [None]

    def tapped_read():
        now = time.perf_counter()
        if read_ended[0] is not None:
            record["between_reads"].append((now - read_ended[0]) * 1000.0)
        try:
            return read()
        finally:
            read_ended[0] = time.perf_counter()

    app.audio_client.read = tapped_read

    async def timed_on_audio(signal):
        top[0] = time.perf_counter()
        try:
            return await on_audio(signal)
        finally:
            record["on_audio"].append((time.perf_counter() - top[0]) * 1000.0)

    async def timed_analyse(signal):
        mark = time.perf_counter()
        try:
            return await analyse(signal)
        finally:
            record["analyse"].append((time.perf_counter() - mark) * 1000.0)

    async def timed_drain():
        mark = time.perf_counter()
        try:
            return await drain()
        finally:
            end = time.perf_counter()
            record["drain"].append((end - mark) * 1000.0)
            record["loop"].append((end - top[0]) * 1000.0)
            record["stamps"].append(end - started)

    def stamp(event, inner):
        def stamped():
            record["boundaries"].append(
                {"t": round(time.perf_counter() - started, 3), "event": event})
            return inner()
        return stamped

    engine.on_audio = timed_on_audio
    analyser.analyse = timed_analyse
    queue.drain = timed_drain
    engine.on_sound_start = stamp("sound_start", sound_start)
    engine.on_sound_stop = stamp("sound_stop", sound_stop)
    return record


def boundary_verdict(boundaries: list, transitions: list,
                     window_sec: float = 30.0) -> dict:
    near = []
    if not boundaries:
        return {"boundaries": 0, "sheds_near_a_boundary": 0, "detail": [],
                "passed": False,
                "why": "no track boundary was recorded, so nothing was tested"}
    for transition in transitions:
        if transition["to"] == "NONE":
            continue
        closest = min(boundaries, key=lambda b: abs(transition["t"] - b["t"]))
        gap = abs(transition["t"] - closest["t"])
        if gap <= window_sec:
            near.append({"shed_at": transition["t"], "boundary": closest,
                         "gap_sec": round(gap, 2)})
    return {"boundaries": len(boundaries), "sheds_near_a_boundary": len(near),
            "detail": near,
            "passed": not near}


async def _soak(args) -> dict:
    from lib.main import PLAYBACK_DELAY_SEC, SoundSwitchAutoPilot

    app = SoundSwitchAutoPilot(midi_port_index=args.midi_port,
                               input_device_index=args.input_device,
                               disable_os2l=args.no_os2l,
                               ui_port=args.ui_port,
                               report_path=args.report)
    timing = _instrument(app)
    sampler = Sampler(app, args.minutes)
    sampler.start()
    await app.run()

    warm = WARMUP_BUFFERS
    stamps, loop = timing["stamps"][warm:], timing["loop"][warm:]
    record = {
        "minutes": args.minutes,
        "input_device": args.input_device,
        "midi_port": args.midi_port,
        "playback_delay_sec": PLAYBACK_DELAY_SEC,
        "warmup_buffers_dropped": warm,
        "buffers": len(loop),
        "per_buffer_ms": {
            "engine.on_audio": spread(timing["on_audio"][warm:]),
            "analyser.analyse": spread(timing["analyse"][warm:]),
            "queue.drain": spread(timing["drain"][warm:]),
            "loop": spread(loop),
            "between_reads": spread(timing["between_reads"][warm:]),
        },
        "over_budget_pct": float(100 * np.mean(
            np.asarray(timing["between_reads"][warm:]) > BUDGET_MS))
        if len(timing["between_reads"]) > warm else None,
        "budget_ms": BUDGET_MS,
        "per_minute": buckets(stamps, loop),
        "boundaries": timing["boundaries"],
        "shed_transitions": sampler.transitions,
        "boundary_verdict": boundary_verdict(timing["boundaries"],
                                             sampler.transitions),
        "samples": sampler.rows,
    }
    if app.event_buffer is not None and args.report:
        report = app.event_buffer.to_report(app.command_queue.get_timing_log())
        Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    return record


def run_soak(args) -> dict:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_soak(args))
    finally:
        loop.close()


def summarise(record: dict) -> str:
    loop = record["per_buffer_ms"]["loop"]
    between = record["per_buffer_ms"]["between_reads"]
    tail = record["samples"][-1] if record["samples"] else {}
    lines = [
        f'  buffers            {record["buffers"]}',
        f'  per-buffer loop    mean {loop["mean"]:.3f}  p99 {loop["p99"]:.3f}  '
        f'p99.9 {loop["p999"]:.3f}  max {loop["max"]:.3f} ms',
        f'  read-to-read       mean {between["mean"]:.3f}  p99 {between["p99"]:.3f}  '
        f'p99.9 {between["p999"]:.3f}  max {between["max"]:.3f} ms  '
        f'({record["over_budget_pct"]:.2f}% over the '
        f'{record["budget_ms"]:.3f} ms budget)',
        f'  peak drift         {tail.get("peak_drift_sec")}s '
        f'(total {tail.get("total_drift_sec")}s)',
        f'  gpu passes         {tail.get("passes")} '
        f'({tail.get("faults")} faults, {tail.get("overflows")} overflows, '
        f'{tail.get("reinits")} reinits, {tail.get("resyncs")} resyncs)',
        f'  shed transitions   {len(record["shed_transitions"])}',
        f'  song boundaries    {record["boundary_verdict"]["boundaries"]}, '
        f'{record["boundary_verdict"]["sheds_near_a_boundary"]} shed near one '
        f'-- {"PASS" if record["boundary_verdict"]["passed"] else "FAIL"}',
        f'  intent commits     {tail.get("commits")}',
    ]
    return "\n".join(lines)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    player = sub.add_parser("play", help="eval-set tracks into an output device")
    player.add_argument("--device", type=int, required=True,
                        help="output device index from `auto_pilot list`")
    player.add_argument("--tracks", type=int, default=DEFAULT_TRACK_COUNT)
    player.add_argument("--gap", type=float, default=3.0,
                        help="silence between tracks; must clear the 0.3 s gate")

    soak = sub.add_parser("run", help="the pipeline, sampled, for a fixed time")
    soak.add_argument("--midi-port", type=int, required=True)
    soak.add_argument("--input-device", type=int, required=True)
    soak.add_argument("--minutes", type=float, default=33.0)
    soak.add_argument("--no-os2l", action="store_true",
                      help="required off a rig: Os2lClient.start RAISES after "
                           "10 s when no SoundSwitch answers the zeroconf query")
    soak.add_argument("--ui-port", type=int, default=8050,
                      help="--report alone starts Dash (main.py gates the UI "
                           "thread on the event buffer, not on --ui), so this "
                           "moves it off a port something else may hold")
    soak.add_argument("--report", default=None)
    soak.add_argument("--record", default=None)

    args = parser.parse_args(argv)

    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE

    if args.mode == "play":
        play(args.device, eval_tracks(args.tracks), args.gap,
             SAMPLE_RATE, BUFFER_SIZE)
        return 0

    record = run_soak(args)
    print()
    print(summarise(record))
    if args.record:
        Path(args.record).write_text(json.dumps(record, indent=2) + "\n")
        print(f"\n  wrote {args.record}")
    return 0 if record["boundary_verdict"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
