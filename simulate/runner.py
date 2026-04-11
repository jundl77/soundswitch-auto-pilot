"""
Simulation runner for soundswitch-auto-pilot.

Replaces all hardware-touching clients with stubs, feeds synthetic or file-based audio
through the full MusicAnalyser → LightEngine → DelayedCommandQueue pipeline, and prints
a timing validation report at the end.

Usage examples:
  # Real music file, 2.5s delay (matching dmx-enttec-node config), with Dash UI
  python auto_pilot simulate file samples/song.mp3 --delay 2.5

  # Headless evaluation (no UI, writes report.json, exits 0=PASS / 1=FAIL)
  python auto_pilot simulate file samples/song.mp3 --no-ui --report report.json
"""

import asyncio
import datetime
import logging
import time

SAMPLE_RATE = 44100
BUFFER_SIZE = 256
TIMING_TOLERANCE_SEC = 0.050  # 50 ms


def build_simulation(audio_client, delay_sec: float):
    """Wire all components together with stub clients and return (app_components, command_queue)."""
    from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient, StubSpotifyClient
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.effect_controller import EffectController
    from lib.engine.light_engine import LightEngine
    from lib.analyser.music_analyser import MusicAnalyser

    midi_client = StubMidiClient()
    os2l_client = StubOs2lClient()
    overlay_client = StubOverlayClient()
    spotify_client = StubSpotifyClient()
    command_queue = DelayedCommandQueue(delay_sec)

    effect_controller = EffectController(midi_client)
    light_engine = LightEngine(
        midi_client, os2l_client, overlay_client,
        spotify_client, effect_controller, command_queue
    )
    spotify_client.set_engine(light_engine)

    music_analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, light_engine, visualizer_updater=None)
    light_engine.set_analyser(music_analyser)
    # Skip YAMNet loading — section detection disabled in simulation for speed.
    # To enable: call music_analyser.start() (requires internet on first run to download model).
    music_analyser.yamnet_change_detector.detect_change = lambda *a, **kw: False

    return {
        'audio_client': audio_client,
        'midi_client': midi_client,
        'os2l_client': os2l_client,
        'overlay_client': overlay_client,
        'command_queue': command_queue,
        'music_analyser': music_analyser,
        'light_engine': light_engine,
    }, command_queue


async def run_simulation(components: dict, duration_sec: float):
    """Main simulation loop — mirrors SoundSwitchAutoPilot.run() but time-bounded."""
    audio_client = components['audio_client']
    music_analyser = components['music_analyser']
    command_queue = components['command_queue']

    audio_client.start_streams()
    start = time.monotonic()

    last_100ms = datetime.datetime.now()
    last_1s = datetime.datetime.now()

    logging.info(f'[sim] starting simulation loop for {duration_sec:.1f}s')
    while time.monotonic() - start < duration_sec:
        now = datetime.datetime.now()
        audio_signal = audio_client.read()
        await music_analyser.analyse(audio_signal)
        await command_queue.drain()

        if now - last_100ms > datetime.timedelta(milliseconds=100):
            last_100ms = now
            await components['light_engine'].on_100ms_callback()
            await components['midi_client'].on_100ms_callback()

        if now - last_1s > datetime.timedelta(seconds=1):
            last_1s = now
            await components['light_engine'].on_1sec_callback()

    audio_client.close()
    logging.info('[sim] simulation complete')


def build_visualizer_simulation(audio_client, event_buffer, delay_sec: float):
    """Like build_simulation but the engine emits events to the shared EventBuffer."""
    from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient, StubSpotifyClient
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.effect_controller import EffectController
    from lib.engine.light_engine import LightEngine
    from lib.analyser.music_analyser import MusicAnalyser

    midi_client = StubMidiClient()
    os2l_client = StubOs2lClient()
    overlay_client = StubOverlayClient()
    spotify_client = StubSpotifyClient()
    command_queue = DelayedCommandQueue(delay_sec)

    effect_controller = EffectController(midi_client, event_buffer=event_buffer)
    light_engine = LightEngine(
        midi_client, os2l_client, overlay_client,
        spotify_client, effect_controller, command_queue,
        event_buffer=event_buffer,
    )
    spotify_client.set_engine(light_engine)

    music_analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, light_engine, visualizer_updater=None)
    light_engine.set_analyser(music_analyser)
    music_analyser.yamnet_change_detector.detect_change = lambda *a, **kw: False

    return {
        'audio_client': audio_client,
        'midi_client': midi_client,
        'os2l_client': os2l_client,
        'overlay_client': overlay_client,
        'command_queue': command_queue,
        'music_analyser': music_analyser,
        'light_engine': light_engine,
    }, command_queue


def print_timing_report(command_queue, tolerance_sec: float = TIMING_TOLERANCE_SEC):
    """Print a human-readable timing validation report."""
    log = command_queue.get_timing_log()
    if not log:
        print('\n[TIMING REPORT] No commands were dispatched.')
        return

    target = command_queue.delay_sec
    passed = 0
    worst_error_ms = 0.0

    print(f'\n{"─" * 72}')
    print(f'  TIMING REPORT   delay_target={target:.3f}s   tolerance=±{tolerance_sec * 1000:.0f}ms')
    print(f'{"─" * 72}')
    print(f'  {"label":<18} {"actual_delta":>12}  {"error":>8}  {"status":>6}')
    print(f'  {"─"*18} {"─"*12}  {"─"*8}  {"─"*6}')

    for entry in log:
        actual = entry['actual_delta_sec']
        error = actual - target
        error_ms = error * 1000
        ok = abs(error) <= tolerance_sec
        if ok:
            passed += 1
        worst_error_ms = max(worst_error_ms, abs(error_ms))
        status = '✓' if ok else '✗'
        print(f'  {entry["label"]:<18} {actual:>10.3f}s  {error_ms:>+7.1f}ms  {status:>6}')

    total = len(log)
    print(f'{"─" * 72}')
    print(f'  RESULT: {passed}/{total} within ±{tolerance_sec * 1000:.0f}ms  |  worst error: {worst_error_ms:.1f}ms')
    verdict = 'PASS' if passed == total else f'FAIL ({total - passed} command(s) out of tolerance)'
    print(f'  {verdict}')
    print(f'{"─" * 72}\n')
    return passed == total
