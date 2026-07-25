"""
Simulation runner for soundswitch-auto-pilot.

Replaces all hardware-touching clients with stubs and feeds synthetic or
file-based audio through the full MusicAnalyser → LightEngine →
DelayedCommandQueue pipeline.

Two pacing modes, selected by the caller:

  Virtual (default for `simulate file`): a VirtualClock is advanced by exactly
  one buffer duration (256/44100 s) per buffer — the run completes as fast as
  the CPU allows while every window/delay/threshold sees identical time to a
  real-time run. Deterministic.

  Real-time (`--ui` file mode): pace_real_time=True sleeps against the wall
  clock so the live Dash timeline scrolls in sync with actual playback.

Usage examples:
  # Fast headless evaluation (writes report.json, exits 0=PASS / 1=FAIL)
  python auto_pilot simulate file samples/song.mp3

  # Real music file with live Dash UI (real-time paced)
  python auto_pilot simulate file samples/song.mp3 --ui
"""

import asyncio
import datetime
import logging
import time

from lib.clock import Clock, SYSTEM_CLOCK, VirtualClock

SAMPLE_RATE = 44100
BUFFER_SIZE = 256
TIMING_TOLERANCE_SEC = 0.050  # 50 ms
# Must match LOOK_AHEAD_SEC in lib/main.py and playback_delay_seconds in dmx-enttec-node.
LOOK_AHEAD_SEC = 2.5


def build_simulation(audio_client, clock: Clock = SYSTEM_CLOCK):
    """Wire all components together with stub clients and return (app_components, command_queue)."""
    from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.effect_controller import EffectController
    from lib.engine.light_engine import LightEngine
    from lib.analyser.music_analyser import MusicAnalyser

    midi_client = StubMidiClient(clock=clock)
    os2l_client = StubOs2lClient(clock=clock)
    overlay_client = StubOverlayClient(clock=clock)
    command_queue = DelayedCommandQueue(LOOK_AHEAD_SEC, clock=clock)

    effect_controller = EffectController(midi_client, clock=clock)
    light_engine = LightEngine(
        midi_client, os2l_client, overlay_client,
        effect_controller, command_queue,
        look_ahead_sec=LOOK_AHEAD_SEC,
        clock=clock,
    )

    music_analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, light_engine,
                                   visualizer_updater=None, clock=clock)
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


def build_visualizer_simulation(audio_client, event_buffer, clock: Clock = SYSTEM_CLOCK):
    """Like build_simulation but the engine emits events to the shared EventBuffer."""
    from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.effect_controller import EffectController
    from lib.engine.light_engine import LightEngine
    from lib.analyser.music_analyser import MusicAnalyser

    midi_client = StubMidiClient(clock=clock)
    os2l_client = StubOs2lClient(clock=clock)
    overlay_client = StubOverlayClient(clock=clock)
    command_queue = DelayedCommandQueue(LOOK_AHEAD_SEC, clock=clock)

    effect_controller = EffectController(midi_client, event_buffer=event_buffer, clock=clock)
    light_engine = LightEngine(
        midi_client, os2l_client, overlay_client,
        effect_controller, command_queue,
        event_buffer=event_buffer,
        look_ahead_sec=LOOK_AHEAD_SEC,
        clock=clock,
    )

    music_analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, light_engine,
                                   visualizer_updater=None, clock=clock)
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


async def run_simulation(components: dict, duration_sec: float,
                         clock: Clock = SYSTEM_CLOCK,
                         pace_real_time: bool = False):
    """Main simulation loop — mirrors SoundSwitchAutoPilot.run().

    duration_sec is measured on `clock`: virtual (song) seconds under a
    VirtualClock, wall seconds otherwise. The loop also stops when the audio
    source reports exhaustion (end of file).

    With a VirtualClock the clock is advanced by exactly one buffer duration
    per buffer, then advanced LOOK_AHEAD_SEC past the end ("flush tail") so
    commands enqueued near the end still fire into the timing log/report.
    """
    audio_client = components['audio_client']
    music_analyser = components['music_analyser']
    command_queue = components['command_queue']

    is_virtual = isinstance(clock, VirtualClock)
    buffer_sec = BUFFER_SIZE / SAMPLE_RATE

    audio_client.start_streams()
    start_mono = clock.monotonic()
    wall_start = time.monotonic()
    buffers_fed = 0

    last_100ms = clock.now()
    last_1s = clock.now()

    logging.info(f'[sim] starting simulation loop for {duration_sec:.1f}s '
                 f'({"virtual" if is_virtual else "wall"} time)')
    while clock.monotonic() - start_mono < duration_sec and not audio_client.exhausted:
        audio_signal = audio_client.read()
        buffers_fed += 1
        if is_virtual:
            clock.advance(buffer_sec)
        if pace_real_time:
            deadline = wall_start + buffers_fed * buffer_sec
            sleep_sec = deadline - time.monotonic()
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)

        await music_analyser.analyse(audio_signal)
        await command_queue.drain()

        now = clock.now()
        if now - last_100ms > datetime.timedelta(milliseconds=100):
            last_100ms = now
            await components['light_engine'].on_100ms_callback()
            await components['midi_client'].on_100ms_callback()

        if now - last_1s > datetime.timedelta(seconds=1):
            last_1s = now
            await components['light_engine'].on_1sec_callback()

    if is_virtual:
        # Flush tail: advance past the look-ahead delay so pending commands fire.
        flush_until = clock.monotonic() + command_queue.delay_sec
        while clock.monotonic() < flush_until:
            clock.advance(buffer_sec)
            await command_queue.drain()

    audio_client.close()
    logging.info('[sim] simulation complete')


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
