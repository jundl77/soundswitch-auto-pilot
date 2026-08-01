"""Per-buffer CPU cost of the audio loop, and what the GPU thread does beside it.

Two readings, because the NN integration split the show across two threads and
one number can no longer describe it.

**The audio loop** is the thread that must never miss a buffer: live input
arrives at exactly 1x and the input side DROPS rather than queues, so falling
behind costs audio.  Its per-buffer work is `LightEngine.on_audio` (resample,
ring write, drain the hand-off, feed the decoder, commit),
`MusicAnalyser.analyse` (madmom and the silence gate) and the command queue's
drain.  Every one of those is timed here against the 5.805 ms buffer period.

**The GPU thread** runs one encoder pass per hop and hands whole passes over a
bounded queue.  Its latency does not have to fit inside a buffer -- it has a
whole hop -- so it is reported as a distribution beside the hop, and the queue
depth the audio loop sees is what says whether the hand-off is keeping up.

The loop is PACED at real time, which is not an optional nicety: the pass
schedule is driven by how much audio has arrived, so an unpaced feed overruns
the extractor's ring inside four passes and measures the shed path instead of
the show.  A run therefore costs its own `--seconds` in wall clock.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

SR, BUFFER = 44100, 256
BUDGET_MS = 1000.0 * BUFFER / SR

# Decision #18's bar, and the one the madmom migration was read against.  It is
# a bar on the AUDIO LOOP: the GPU thread is measured against its hop instead.
CORE_SHARE_BAR_PCT = 20.0


def _load(path: str, seconds: float) -> np.ndarray:
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from simulate.fake_audio_client import FileAudioClient
    client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, path)
    client.start_streams()
    wanted = int(seconds * SR / BUFFER)
    chunks = []
    while len(chunks) < wanted and not client.exhausted:
        chunks.append(client.read())
    return np.concatenate(chunks)


def _stats(label: str, times_ms: np.ndarray, budget_ms: float = BUDGET_MS) -> dict:
    row = {
        'label': label,
        'samples': int(times_ms.size),
        'mean_ms': float(times_ms.mean()),
        'p50_ms': float(np.percentile(times_ms, 50)),
        'p90_ms': float(np.percentile(times_ms, 90)),
        'p99_ms': float(np.percentile(times_ms, 99)),
        'max_ms': float(times_ms.max()),
        'core_share_pct': float(100 * times_ms.mean() / budget_ms),
        'realtime_factor': float(budget_ms / times_ms.mean()),
        'over_budget_pct': float(100 * (times_ms > budget_ms).mean()),
    }
    print(f'{label:26s} mean {row["mean_ms"]:6.3f}  p50 {row["p50_ms"]:6.3f}  '
          f'p90 {row["p90_ms"]:6.3f}  p99 {row["p99_ms"]:7.3f}  '
          f'max {row["max_ms"]:8.3f} ms | {row["core_share_pct"]:5.1f}% of a core '
          f'| {row["realtime_factor"]:6.1f}x realtime')
    return row


def _time_each(audio: np.ndarray, step) -> np.ndarray:
    times = []
    for i in range(0, len(audio) - BUFFER, BUFFER):
        buf = audio[i:i + BUFFER]
        start = time.perf_counter()
        step(buf)
        times.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(times)


def spread(values: list, scale: float = 1.0) -> dict:
    """The distribution of a sampled quantity, or an empty dict if none were."""
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64) * scale
    return {
        'samples': int(array.size),
        'mean': float(array.mean()),
        'p50': float(np.percentile(array, 50)),
        'p90': float(np.percentile(array, 90)),
        'p99': float(np.percentile(array, 99)),
        'max': float(array.max()),
    }


def verdict(core_share_pct: float, bar_pct: float = CORE_SHARE_BAR_PCT) -> str:
    return 'PASS' if core_share_pct <= bar_pct else 'OVER BAR'


# --------------------------------------------------------------------------- #
# The paced audio loop
# --------------------------------------------------------------------------- #


async def _paced_loop(audio: np.ndarray, out_rows: list) -> dict:
    """Feed a track through the production wiring at real time, timing it.

    Mirrors `lib/main.py`'s construction rather than importing the simulation
    runner: the runner's fast path is single-threaded on a virtual clock, and a
    virtual clock cannot measure what a thread costs.
    """
    from lib.analyser.drift_watchdog import DriftWatchdog
    from lib.analyser.gpu_stage import reserved_bytes
    from lib.analyser.music_analyser import MusicAnalyser
    from lib.engine.delayed_command_queue import DelayedCommandQueue
    from lib.engine.effect_controller import EffectController
    from lib.engine.light_engine import LightEngine
    from lib.main import PLAYBACK_DELAY_SEC
    from simulate.stub_clients import (StubMidiClient, StubOs2lClient,
                                       StubOverlayClient)
    from lib import section_chain

    watchdog = DriftWatchdog(BUFFER / SR)
    chain = section_chain.build_section_chain(watchdog=watchdog)
    stage = chain.stream

    pass_ms: list = []
    inner = stage.posteriors.run_pass

    def timed_pass():
        started = time.perf_counter()
        try:
            return inner()
        finally:
            pass_ms.append((time.perf_counter() - started) * 1000.0)

    stage.posteriors.run_pass = timed_pass

    midi, os2l, overlay = StubMidiClient(), StubOs2lClient(), StubOverlayClient()
    queue = DelayedCommandQueue(PLAYBACK_DELAY_SEC)
    engine = LightEngine(midi, os2l, overlay, EffectController(midi), queue,
                         playback_delay_sec=PLAYBACK_DELAY_SEC,
                         section_chain=chain.stream,
                         section_decoder=chain.decoder,
                         watchdog=watchdog)
    analyser = MusicAnalyser(SR, BUFFER, engine, watchdog=watchdog)
    engine.set_analyser(analyser)
    os2l.set_analyser(analyser)

    on_audio_ms, analyse_ms, drain_ms, loop_ms = [], [], [], []
    depths, reserved, pacing_ms = [], [], []
    period = BUFFER / SR
    buffers = (len(audio) - BUFFER) // BUFFER

    # The first call loads madmom's nets and warms CUDA; timing it would put a
    # one-off startup cost in a per-buffer distribution.
    await engine.on_audio(audio[:BUFFER])
    await analyser.analyse(audio[:BUFFER].copy())

    started = time.perf_counter()
    deadline = started
    for index in range(1, buffers):
        deadline += period
        slack = deadline - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        pacing_ms.append((time.perf_counter() - deadline) * 1000.0)

        buf = audio[index * BUFFER:(index + 1) * BUFFER]
        depths.append(stage.queued)

        top = time.perf_counter()
        await engine.on_audio(buf)
        mark = time.perf_counter()
        await analyser.analyse(buf.copy())
        after = time.perf_counter()
        await queue.drain()
        end = time.perf_counter()

        on_audio_ms.append((mark - top) * 1000.0)
        analyse_ms.append((after - mark) * 1000.0)
        drain_ms.append((end - after) * 1000.0)
        loop_ms.append((end - top) * 1000.0)

        if index % 1000 == 0:
            reserved.append({'audio_sec': round(index * period, 1),
                             'bytes': reserved_bytes()})

    elapsed = time.perf_counter() - started
    chain.stop()

    print()
    out_rows.append(_stats('engine.on_audio', np.asarray(on_audio_ms)))
    out_rows.append(_stats('analyser.analyse', np.asarray(analyse_ms)))
    out_rows.append(_stats('queue.drain', np.asarray(drain_ms)))
    loop_row = _stats('AUDIO LOOP (all three)', np.asarray(loop_ms))
    out_rows.append(loop_row)

    return {
        'wall_sec': round(elapsed, 2),
        'buffers': len(loop_ms),
        'loop': loop_row,
        'gpu_pass_ms': spread(pass_ms),
        'gpu_hop_ms': 1000.0 * section_chain.read_geometry().stream.hop_sec,
        'gpu_passes': int(stage.passes),
        'gpu_faults': int(stage.faults),
        'gpu_overflows': int(stage.overflows),
        'gpu_reinits': int(stage.reinits),
        'gpu_resyncs': int(stage.resyncs),
        'queue_depth': spread(depths),
        'queue_capacity': int(stage._queue_passes),
        'pacing_error_ms': spread(pacing_ms),
        'reserved_bytes_curve': reserved,
        'peak_drift_sec': round(watchdog.peak_drift_sec, 4),
        'shed_level_at_end': watchdog.level.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('track')
    ap.add_argument('--seconds', type=float, default=120.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--front-end-only', action='store_true',
                    help='time madmom alone and skip the paced loop (no GPU)')
    args = ap.parse_args()

    audio = _load(args.track, args.seconds)
    print(f'{len(audio)/SR:.1f}s of audio, buffer {BUFFER} '
          f'({BUDGET_MS:.3f} ms budget), single thread\n')
    rows = []

    from lib.analyser.madmom_rhythm import MadmomRhythm

    rhythm = MadmomRhythm(SR)
    rhythm.process(audio[:BUFFER])            # exclude first-call model warm-up
    rows.append(_stats('madmom rhythm', _time_each(audio, rhythm.process)))

    # The two rows this harness used to carry beside that one -- the mel
    # filterbank and the rhythm stack it replaced -- measured code the NN
    # integration deleted.  Their figures survive in docs/migration-evidence.md;
    # the harness stays for the stages that take their place.
    paced = None
    if not args.front_end_only:
        print(f'\npacing {args.seconds:.0f}s of audio through the production '
              f'wiring at real time (this takes that long)\n')
        paced = asyncio.run(_paced_loop(audio, rows))

    reference = paced['loop'] if paced else rows[0]
    share = reference['core_share_pct']
    print(f'\naudio loop: {reference["mean_ms"]:.3f} ms per buffer')
    print(f'  as a share of one core : {share:.1f}%')
    print(f'  headroom               : {BUDGET_MS / reference["mean_ms"]:.1f}x real time')
    print(f'  decision #18 bar (<= {CORE_SHARE_BAR_PCT:.0f}% of a core): '
          f'{verdict(share)}')

    if paced:
        gpu = paced['gpu_pass_ms']
        print(f'\nGPU thread: {paced["gpu_passes"]} passes, '
              f'mean {gpu["mean"]:.1f} ms  p99 {gpu["p99"]:.1f} ms  '
              f'max {gpu["max"]:.1f} ms  '
              f'({100 * gpu["mean"] / paced["gpu_hop_ms"]:.1f}% of the '
              f'{paced["gpu_hop_ms"]:.0f} ms hop)')
        depth = paced['queue_depth']
        print(f'hand-off  : depth mean {depth["mean"]:.3f}  max {depth["max"]:.0f} '
              f'of {paced["queue_capacity"]}, {paced["gpu_overflows"]} overflow(s)')
        print(f'peak drift: {paced["peak_drift_sec"]:+.3f}s, '
              f'shed level at end {paced["shed_level_at_end"]}')

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'buffer_budget_ms': BUDGET_MS, 'track': args.track,
             'rows': rows, 'paced': paced,
             'core_share_pct': share, 'verdict': verdict(share)},
            indent=2) + '\n')
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
