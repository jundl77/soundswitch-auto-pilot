import asyncio
import datetime
import logging
import random
import time

from lib.audio_config import SAMPLE_RATE, BUFFER_SIZE
from lib.clock import Clock, SYSTEM_CLOCK, VirtualClock
from simulate import cell_cache

TIMING_TOLERANCE_SEC = 0.050
# Must match PLAYBACK_DELAY_SEC in lib/main.py and playback_delay_seconds in
# dmx-enttec-node. 14.0 per #154: the chain is 13.66 s at the corpus median bar
# and 14.08 s at p99, so this is the budget the decoder's lag_bars=2 needs.
PLAYBACK_DELAY_SEC = 14.0
FAST_SIM_RANDOM_SEED = 1337


def build_simulation(audio_client, event_buffer=None, clock: Clock = SYSTEM_CLOCK,
                     section: object | None = None, threaded: bool = False):
    """The whole pipeline on stub clients -- the identical code path to prod.

    ``section`` is the NN chain; absent, it is built from the shipped artifacts,
    and if those are not on this machine the sim runs the degradation state
    (beats, silence, a held intent) rather than refusing to start.  That is
    #144's contract, and it is what keeps every test that does not need the
    model runnable on a fresh clone.

    ``threaded`` is the real-time paths' switch (D3): the microphone and the
    ``--ui`` file run is paced by a wall clock, so the GPU stage belongs on its
    own thread there exactly as it does in production.  Fast simulation runs on
    a virtual clock and stays single-threaded, which is what makes its reports
    byte-identical run to run.
    """
    from simulate.stub_clients import StubMidiClient, StubOs2lClient, StubOverlayClient
    from lib.analyser.drift_watchdog import DriftWatchdog
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.effect_controller import EffectController
    from lib.engine.light_engine import LightEngine
    from lib.analyser.music_analyser import MusicAnalyser

    midi_client = StubMidiClient(clock=clock)
    os2l_client = StubOs2lClient(clock=clock)
    overlay_client = StubOverlayClient(clock=clock)
    command_queue = DelayedCommandQueue(PLAYBACK_DELAY_SEC, clock=clock)

    watchdog = None
    if threaded:
        watchdog = DriftWatchdog(BUFFER_SIZE / SAMPLE_RATE, clock=clock)
    if section is None:
        section = load_section_chain(watchdog=watchdog, audio_client=audio_client)

    effect_controller = EffectController(midi_client, event_buffer=event_buffer, clock=clock)
    light_engine = LightEngine(
        midi_client, os2l_client, overlay_client,
        effect_controller, command_queue,
        event_buffer=event_buffer,
        playback_delay_sec=PLAYBACK_DELAY_SEC,
        section_chain=None if section is None else section.stream,
        section_decoder=None if section is None else section.decoder,
        clock=clock,
    )

    music_analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, light_engine, clock=clock,
                                   watchdog=watchdog)
    light_engine.set_analyser(music_analyser)

    return {
        'audio_client': audio_client,
        'midi_client': midi_client,
        'os2l_client': os2l_client,
        'overlay_client': overlay_client,
        'command_queue': command_queue,
        'music_analyser': music_analyser,
        'light_engine': light_engine,
        'event_buffer': event_buffer,
        'section': section,
    }, command_queue


def load_section_chain(watchdog=None, audio_client=None):
    """The shipped chain, reset, or None on a machine that does not have it.

    Built once per process and reused, because the encoder is 1.3 GB of weights
    and a fresh one per simulation would dominate a suite that runs several.
    **Reset on every hand-out**, because sharing it otherwise carries the last
    track's ring, GRU state and bar grid into the next simulation -- which makes
    a report a function of what ran before it in the same process, and there is
    no song boundary between two simulations to do the job.

    A threaded chain is NOT cached: it owns a running thread and a watchdog
    belonging to one run, and handing that to the next simulation would share a
    shed level between two shows.  The real-time paths build one each and stop
    it; only the fast sim, which is where the cache pays, reuses.

    **A warm cell cache short-circuits all of that** (D12): the extractor is
    replayed, so no encoder is loaded, the chain is cheap to build per track,
    and there is nothing worth sharing between simulations.
    """
    global _SECTION_CHAIN
    from lib import section_chain

    if watchdog is not None:
        # The threaded paths exist to run the real GPU stage; a replay there
        # would prove nothing about the thread it is meant to exercise.
        if not _artifacts_or_degrade():
            return None
        return section_chain.build_section_chain(watchdog=watchdog)

    plan = _cell_cache_plan(audio_client)
    if plan is not None:
        replay, reason = cell_cache.open_replay(*plan)
        if replay is not None:
            logging.info(f'[sim] replaying cached extractor cells ← {plan[0].name}')
            return section_chain.build_section_chain(extractor=lambda _: replay)
        logging.info(f'[sim] extractor cells: {reason} — this run needs the GPU')

    if _SECTION_CHAIN is _UNBUILT:
        _SECTION_CHAIN = (section_chain.build_section_chain()
                          if _artifacts_or_degrade() else None)
    if _SECTION_CHAIN is None:
        return None
    _SECTION_CHAIN.stream.reset()
    _SECTION_CHAIN.decoder.reset()
    if plan is None:
        return _SECTION_CHAIN
    return cell_cache.recording_chain(_SECTION_CHAIN, *plan)


def _artifacts_or_degrade() -> bool:
    from lib import section_chain

    if section_chain.artifacts_present():
        return True
    logging.warning('[sim] no NN artifacts on this machine — running '
                    'the degradation state (beats, silence, held intent)')
    return False


def _cell_cache_plan(audio_client):
    """``(sidecar path, key)`` for a client that reads a file it can name.

    A microphone names neither a file nor a decoder, so it gets no cache and
    the question never reaches the artifacts.
    """
    from lib import section_chain

    path = getattr(audio_client, 'path', None)
    decode = getattr(audio_client, 'decode_path', None)
    if path is None or decode is None or not section_chain.artifacts_present():
        return None
    geometry = section_chain.read_geometry().stream
    return (cell_cache.sidecar_path(path, decode),
            cell_cache.cache_key(geometry, source_rate=SAMPLE_RATE,
                                 audio_path=path, decode_path=decode))


_UNBUILT = object()
_SECTION_CHAIN = _UNBUILT


async def run_fast_simulation_components(audio_client, duration_sec: float = float('inf'),
                                         seed: int = FAST_SIM_RANDOM_SEED):
    """The fast sim, handing back every client it built.

    The stub MIDI and OS2L clients hold wires the report does not carry, and the
    golden fixture pins those. One code path so a fixture cut here describes the
    same run the benchmark scores.
    """
    from lib.engine.event_buffer import EventBuffer

    random.seed(seed)
    clock = VirtualClock()
    event_buffer = EventBuffer(window_sec=float('inf'), clock=clock,
                               look_ahead_sec=PLAYBACK_DELAY_SEC)
    components, command_queue = build_simulation(audio_client, event_buffer, clock=clock)
    event_buffer.start()
    await run_simulation(components, duration_sec, clock=clock)
    return components, command_queue


async def run_fast_simulation(audio_client, duration_sec: float = float('inf'),
                              seed: int = FAST_SIM_RANDOM_SEED):
    components, command_queue = await run_fast_simulation_components(
        audio_client, duration_sec, seed)
    return components['audio_client'], components['event_buffer'], command_queue


async def run_simulation(components: dict, duration_sec: float,
                         clock: Clock = SYSTEM_CLOCK,
                         pace_real_time: bool = False):
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
    last_10s = clock.now()

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

        # Before `analyse`, which appends the debug click to the buffer it was
        # handed: the feature stage must read the audio the room hears.
        await components['light_engine'].on_audio(audio_signal)
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

        if now - last_10s > datetime.timedelta(seconds=10):
            last_10s = now
            await components['light_engine'].on_10sec_callback()

    event_buffer = components.get('event_buffer')
    if event_buffer is not None:
        event_buffer.mark_end()

    if is_virtual:
        flush_until = clock.monotonic() + command_queue.delay_sec
        while clock.monotonic() < flush_until:
            clock.advance(buffer_sec)
            await command_queue.drain()
    elif pace_real_time:
        while command_queue.pending:
            await asyncio.sleep(buffer_sec)
            await command_queue.drain()

    audio_client.close()
    section = components.get('section')
    if section is not None:
        section.stop()
    logging.info('[sim] simulation complete')


def print_timing_report(command_queue, tolerance_sec: float = TIMING_TOLERANCE_SEC):
    log = command_queue.get_timing_log()
    if not log:
        print('\n[TIMING REPORT] No commands were dispatched.')
        return

    passed = 0
    worst_error_ms = 0.0

    print(f'\n{"─" * 72}')
    # Per entry, not per queue: a beat and an intent wait different amounts (B1).
    print(f'  TIMING REPORT   playback_delay={command_queue.delay_sec:.3f}s   '
          f'tolerance=±{tolerance_sec * 1000:.0f}ms')
    print(f'{"─" * 72}')
    print(f'  {"label":<18} {"actual_delta":>12}  {"error":>8}  {"status":>6}')
    print(f'  {"─"*18} {"─"*12}  {"─"*8}  {"─"*6}')

    for entry in log:
        actual = entry['actual_delta_sec']
        error = actual - entry['target_delta_sec']
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
