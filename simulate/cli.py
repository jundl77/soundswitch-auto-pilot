"""
Simulation CLI handlers — wired into 'auto_pilot simulate' subcommand.

MODES
  file            — FAST headless (default): run the whole file through the
                    pipeline on a virtual clock (~25-30× real-time), write the
                    JSON report, print the evaluation, exit 0=PASS / 1=FAIL.
  file --ui       — real-time paced run with the live Dash timeline (previous
                    default behavior; --play-audio available here).
  realtime        — capture from microphone in real time with Dash timeline.

EXAMPLES
  python auto_pilot simulate file samples/song.mp3
  python auto_pilot simulate file samples/song.mp3 --report out.json
  python auto_pilot simulate file samples/song.mp3 --ui --play-audio
  python auto_pilot simulate realtime --device-index 1
"""

import asyncio
import json
import random
import sys
import threading
import time

SAMPLE_RATE = 44100
BUFFER_SIZE = 256

# Fixed seed for the fast headless mode: effect selection is random by design,
# but fast-sim reports must be reproducible run-to-run.
FAST_SIM_RANDOM_SEED = 1337


def _run_pipeline(components, duration_sec: float, event_buffer, command_queue,
                  pace_real_time: bool):
    from simulate.runner import run_simulation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            run_simulation(components, duration_sec, pace_real_time=pace_real_time)
        )
    finally:
        event_buffer.set_timing_log(command_queue.get_timing_log())
        loop.close()


def _write_report_and_evaluate(event_buffer, command_queue, report_path: str) -> bool:
    from simulate.evaluator import evaluate, print_evaluation
    report = event_buffer.to_report(command_queue.get_timing_log())
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'[simulate] report written → {report_path}')
    result = evaluate(report)
    print_evaluation(result)
    return result['passed']


async def run_file(args):
    if args.play_audio and not args.ui:
        print('[simulate] error: --play-audio requires --ui '
              '(audio cannot play at fast-simulation speed)')
        sys.exit(2)
    if args.ui:
        _run_file_realtime_ui(args)
    else:
        await _run_file_fast(args)


async def _run_file_fast(args):
    """Fast headless mode: virtual clock, no UI, report + evaluation + exit code."""
    from lib.clock import VirtualClock
    from lib.engine.event_buffer import EventBuffer
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import build_visualizer_simulation, run_simulation

    random.seed(FAST_SIM_RANDOM_SEED)
    clock = VirtualClock()
    audio_client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, args.audio, clock=clock)
    # Infinite window: keep the entire song's events — reports must never prune.
    event_buffer = EventBuffer(window_sec=float('inf'), clock=clock)
    components, command_queue = build_visualizer_simulation(audio_client, event_buffer, clock=clock)

    event_buffer.start()
    wall_start = time.monotonic()
    await run_simulation(components, duration_sec=float('inf'), clock=clock)
    wall_elapsed = time.monotonic() - wall_start

    song_sec = audio_client.duration_sec
    speed = song_sec / wall_elapsed if wall_elapsed > 0 else 0.0
    print(f'[simulate] {song_sec:.1f}s of audio processed in {wall_elapsed:.1f}s ({speed:.1f}x real-time)')

    passed = _write_report_and_evaluate(event_buffer, command_queue, args.report)
    sys.exit(0 if passed else 1)


def _run_file_realtime_ui(args):
    """Real-time paced run with the live Dash timeline (previous default behavior)."""
    from lib.engine.event_buffer import EventBuffer
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import build_visualizer_simulation

    audio_client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, args.audio)
    event_buffer = EventBuffer()
    components, command_queue = build_visualizer_simulation(audio_client, event_buffer)

    try:
        import librosa
        duration_sec = librosa.get_duration(path=args.audio)
    except Exception:
        duration_sec = float('inf')

    event_buffer.start()

    thread = threading.Thread(
        target=_run_pipeline,
        args=(components, duration_sec, event_buffer, command_queue, True),
        daemon=True,
    )
    thread.start()

    if args.play_audio:
        try:
            import sounddevice as sd
            import librosa as lr
            audio_data, sr = lr.load(args.audio, sr=SAMPLE_RATE, mono=True)
            sd.play(audio_data, samplerate=sr)
            print('[simulate] audio playback started')
        except ImportError as e:
            print(f'[simulate] warning: {e} — audio playback skipped')

    from simulate.visualizer_app import run_app
    run_app(event_buffer, port=args.port)


def run_realtime(args):
    from lib.engine.event_buffer import EventBuffer
    from lib.clients.pyaudio_client import PyAudioClient
    from simulate.runner import build_visualizer_simulation
    from simulate.visualizer_app import run_app

    audio_client = PyAudioClient(
        sample_rate=SAMPLE_RATE,
        buffer_size=BUFFER_SIZE,
        input_device_index=args.device_index,
    )
    event_buffer = EventBuffer()
    components, command_queue = build_visualizer_simulation(audio_client, event_buffer)
    event_buffer.start()

    # Microphone input is hardware-paced — no artificial pacing needed.
    thread = threading.Thread(
        target=_run_pipeline,
        args=(components, float('inf'), event_buffer, command_queue, False),
        daemon=True,
    )
    thread.start()

    run_app(event_buffer, port=args.port)


def add_simulate_subparser(subparsers):
    """Register the 'simulate' subcommand and its sub-subcommands."""
    sim = subparsers.add_parser(
        'simulate',
        help='Run the pipeline against a file (fast, headless) or microphone (live UI)',
    )
    sub = sim.add_subparsers(dest='sim_mode', required=True)

    fp = sub.add_parser('file', help='Simulate from an audio file (fast headless by default)')
    fp.add_argument('audio', help='Path to audio file (MP3 / WAV / FLAC)')
    fp.add_argument('--ui', action='store_true',
                    help='Real-time paced run with live Dash timeline (instead of fast headless)')
    fp.add_argument('--play-audio', action='store_true',
                    help='Play audio from speakers (requires --ui and sounddevice)')
    fp.add_argument('--report', default='report.json',
                    help='Report output path for fast mode (default: report.json)')
    fp.add_argument('--port', type=int, default=8050, help='Dash server port (--ui only)')

    rp = sub.add_parser('realtime', help='Simulate from microphone in real time')
    rp.add_argument('--device-index', type=int, default=None,
                    help='PyAudio input device index (default: system default)')
    rp.add_argument('--port', type=int, default=8050, help='Dash server port')

    sim.set_defaults(func=simulate_cmd)


async def simulate_cmd(args):
    if args.sim_mode == 'file':
        await run_file(args)
    elif args.sim_mode == 'realtime':
        run_realtime(args)
