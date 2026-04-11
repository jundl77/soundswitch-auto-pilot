"""
Simulation CLI handlers — wired into 'auto_pilot simulate' subcommand.

MODES
  file      — feed an audio file through the pipeline; optional Dash timeline
  realtime  — capture from microphone in real time with Dash timeline

EXAMPLES
  python auto_pilot simulate file samples/song.mp3 --delay 2.5 --play-audio
  python auto_pilot simulate file samples/song.mp3 --no-ui --report report.json
  python auto_pilot simulate realtime --device-index 1 --delay 0.3

EXIT CODE (--no-ui / file mode only)
  0 = PASS
  1 = FAIL
"""

import asyncio
import json
import sys
import threading

SAMPLE_RATE = 44100
BUFFER_SIZE = 256


def _run_pipeline(components, duration_sec: float, event_buffer, command_queue):
    from simulate.runner import run_simulation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_simulation(components, duration_sec))
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


def run_file(args):
    from lib.engine.event_buffer import EventBuffer
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import build_visualizer_simulation

    audio_client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, args.audio)
    event_buffer = EventBuffer()
    components, command_queue = build_visualizer_simulation(audio_client, event_buffer, args.delay)

    try:
        import librosa
        duration_sec = librosa.get_duration(path=args.audio)
    except Exception:
        duration_sec = float('inf')

    event_buffer.start()

    thread = threading.Thread(
        target=_run_pipeline,
        args=(components, duration_sec, event_buffer, command_queue),
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

    if args.no_ui:
        print(f'[simulate] running headlessly for {duration_sec:.0f}s …')
        thread.join()
        passed = _write_report_and_evaluate(event_buffer, command_queue, args.report)
        sys.exit(0 if passed else 1)

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
    components, command_queue = build_visualizer_simulation(audio_client, event_buffer, args.delay)
    event_buffer.start()

    thread = threading.Thread(
        target=_run_pipeline,
        args=(components, float('inf'), event_buffer, command_queue),
        daemon=True,
    )
    thread.start()

    run_app(event_buffer, port=args.port)


def add_simulate_subparser(subparsers):
    """Register the 'simulate' subcommand and its sub-subcommands."""
    sim = subparsers.add_parser(
        'simulate',
        help='Run the pipeline against a file or microphone with a real-time visualizer',
    )
    sub = sim.add_subparsers(dest='sim_mode', required=True)

    fp = sub.add_parser('file', help='Simulate from an audio file')
    fp.add_argument('audio', help='Path to audio file (MP3 / WAV / FLAC)')
    fp.add_argument('--delay', type=float, default=0.0,
                    help='Lookahead delay in seconds')
    fp.add_argument('--play-audio', action='store_true',
                    help='Play audio from speakers (requires sounddevice)')
    fp.add_argument('--no-ui', action='store_true',
                    help='Headless: run to completion, write report, evaluate (exit 0=PASS, 1=FAIL)')
    fp.add_argument('--report', default='report.json',
                    help='Report output path (--no-ui only, default: report.json)')
    fp.add_argument('--port', type=int, default=8050, help='Dash server port')

    rp = sub.add_parser('realtime', help='Simulate from microphone in real time')
    rp.add_argument('--device-index', type=int, default=None,
                    help='PyAudio input device index (default: system default)')
    rp.add_argument('--delay', type=float, default=0.0,
                    help='Lookahead delay in seconds')
    rp.add_argument('--port', type=int, default=8050, help='Dash server port')

    sim.set_defaults(func=simulate_cmd)


async def simulate_cmd(args):
    if args.sim_mode == 'file':
        run_file(args)
    elif args.sim_mode == 'realtime':
        run_realtime(args)
