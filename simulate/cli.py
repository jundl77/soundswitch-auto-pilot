import asyncio
import json
import logging
import os
import sys
import time

from lib.audio_config import SAMPLE_RATE, BUFFER_SIZE


def _run_pipeline(components, duration_sec: float, event_buffer, command_queue,
                  pace_real_time: bool, report_path: str | None = None):
    from simulate import runner
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            runner.run_simulation(components, duration_sec,
                                  pace_real_time=pace_real_time)
        )
    finally:
        event_buffer.set_timing_log(command_queue.get_timing_log())
        if report_path:
            _write_report(event_buffer, command_queue, report_path)
        loop.close()


def _write_report(event_buffer, command_queue, report_path: str) -> dict:
    from simulate.evaluator import report_checksum
    report = event_buffer.to_report(command_queue.get_timing_log())
    report['checksum'] = report_checksum(report)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'[simulate] report written → {report_path}  (sha256 {report["checksum"][:16]})')
    return report


def _write_report_and_evaluate(event_buffer, command_queue, report_path: str) -> bool:
    from simulate.evaluator import evaluate, print_evaluation
    result = evaluate(_write_report(event_buffer, command_queue, report_path))
    print_evaluation(result)
    return result['passed']


async def run_file(args):
    if not os.path.isfile(args.audio):
        print(f'[simulate] error: audio file not found: {args.audio}')
        sys.exit(2)
    if args.play_audio and not args.ui:
        print('[simulate] error: --play-audio requires --ui '
              '(audio cannot play at fast-simulation speed)')
        sys.exit(2)
    if args.ui:
        _run_file_realtime_ui(args)
    else:
        await _run_file_fast(args)


async def _run_file_fast(args):
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import run_fast_simulation

    logging.getLogger().setLevel(logging.WARNING)

    wall_start = time.monotonic()
    audio_client, event_buffer, command_queue = await run_fast_simulation(
        FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, args.audio)
    )
    wall_elapsed = time.monotonic() - wall_start

    song_sec = audio_client.duration_sec
    speed = song_sec / wall_elapsed if wall_elapsed > 0 else 0.0
    print(f'[simulate] {song_sec:.1f}s of audio processed in {wall_elapsed:.1f}s ({speed:.1f}x real-time)')

    passed = _write_report_and_evaluate(event_buffer, command_queue, args.report)
    sys.exit(0 if passed else 1)


def _session_buffer(look_ahead_sec: float, clock=None):
    from lib.clock import SYSTEM_CLOCK
    from lib.engine.event_buffer import EventBuffer

    return EventBuffer(window_sec=float('inf'),
                       clock=clock or SYSTEM_CLOCK,
                       look_ahead_sec=look_ahead_sec)


def _with_viewer(event_buffer, port, run):
    from lib import ui_bridge

    ui = ui_bridge.start(event_buffer, port)
    try:
        run()
        if ui is not None:
            print('[simulate] the track is over — the viewer is still up on '
                  f'http://localhost:{port}; Ctrl-C to close it')
            ui.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if ui is not None:
            ui.stop()


def _run_file_realtime_ui(args):
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import build_simulation, PLAYBACK_DELAY_SEC

    audio_client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, args.audio)
    event_buffer = _session_buffer(PLAYBACK_DELAY_SEC)
    components, command_queue = build_simulation(audio_client, event_buffer,
                                                 threaded=True)

    try:
        import librosa
        duration_sec = librosa.get_duration(path=args.audio)
    except Exception:
        duration_sec = float('inf')

    event_buffer.start()

    if args.play_audio:
        try:
            import sounddevice as sd
            import librosa as lr
            audio_data, sr = lr.load(args.audio, sr=SAMPLE_RATE, mono=True)
            sd.play(audio_data, samplerate=sr)
            print('[simulate] audio playback started')
        except ImportError as e:
            print(f'[simulate] warning: {e} — audio playback skipped')

    _with_viewer(event_buffer, args.port,
                 lambda: _run_pipeline(components, duration_sec, event_buffer,
                                       command_queue, True, args.report))


def run_realtime(args):
    from lib.engine.event_buffer import EventBuffer
    from lib.clients.pyaudio_client import PyAudioClient
    from simulate.runner import build_simulation, PLAYBACK_DELAY_SEC

    audio_client = PyAudioClient(
        sample_rate=SAMPLE_RATE,
        buffer_size=BUFFER_SIZE,
        input_device_index=args.device_index,
    )
    event_buffer = EventBuffer(look_ahead_sec=PLAYBACK_DELAY_SEC)
    components, command_queue = build_simulation(audio_client, event_buffer,
                                                 threaded=True)
    event_buffer.start()

    _with_viewer(event_buffer, args.port,
                 lambda: _run_pipeline(components, float('inf'), event_buffer,
                                       command_queue, False))


def add_simulate_subparser(subparsers):
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
                    help='Report output path (default: report.json); under --ui '
                         'it is written when the track ends')
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
