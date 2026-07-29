"""Per-buffer CPU cost of the analyser, old rhythm front-end against new.

The live loop has one buffer period (256/44100 = 5.805 ms) to do everything in.
This measures what the analyser spends of it, and splits the spend so a
regression can be attributed rather than merely noticed:

    filterbank  the aubio pvoc + mel bank that STAYS
    rhythm      madmom's beat and onset chains that REPLACED aubio's
    aubio-rhythm  the same job as aubio did it, for the ratio

Reported against decision #18's bar: forward cost / buffer period <= 20 % of one
core is the >= 5x-real-time pass. Tails matter as much as the mean here — a
loop whose p99 exceeds the buffer period is one that periodically eats its own
look-ahead — so p50/p90/p99/max are reported and not just the average.

Run it on an otherwise quiet box; a concurrent sweep will contaminate it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SR, BUFFER = 44100, 256
BUDGET_MS = 1000.0 * BUFFER / SR


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


def _stats(label: str, times_ms: np.ndarray) -> dict:
    row = {
        'label': label,
        'mean_ms': float(times_ms.mean()),
        'p50_ms': float(np.percentile(times_ms, 50)),
        'p90_ms': float(np.percentile(times_ms, 90)),
        'p99_ms': float(np.percentile(times_ms, 99)),
        'max_ms': float(times_ms.max()),
        'core_share_pct': float(100 * times_ms.mean() / BUDGET_MS),
        'realtime_factor': float(BUDGET_MS / times_ms.mean()),
        'over_budget_pct': float(100 * (times_ms > BUDGET_MS).mean()),
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('track')
    ap.add_argument('--seconds', type=float, default=120.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    audio = _load(args.track, args.seconds)
    print(f'{len(audio)/SR:.1f}s of audio, buffer {BUFFER} '
          f'({BUDGET_MS:.3f} ms budget), single thread\n')
    rows = []

    import aubio

    from lib.analyser.madmom_rhythm import MadmomRhythm
    from lib.analyser.music_analyser import MelFilterbank

    mel = MelFilterbank(SR, BUFFER)
    rows.append(_stats('aubio filterbank (STAYS)', _time_each(audio, mel)))

    onset = aubio.onset('default', BUFFER * 2, BUFFER, SR)
    tempo = aubio.tempo('default', BUFFER * 2, BUFFER, SR)
    notes = aubio.notes('default', BUFFER * 2, BUFFER, SR)

    def old_rhythm(buf):
        onset(buf)
        tempo(buf)
        notes(buf)

    rows.append(_stats('aubio rhythm (WAS)', _time_each(audio, old_rhythm)))

    rhythm = MadmomRhythm(SR)
    rhythm.process(audio[:BUFFER])            # exclude first-call model warm-up
    rows.append(_stats('madmom rhythm (IS)', _time_each(audio, rhythm.process)))

    fb, was, now = rows[0], rows[1], rows[2]
    old_total = fb['mean_ms'] + was['mean_ms']
    new_total = fb['mean_ms'] + now['mean_ms']
    print(f'\nanalyser front-end total: {old_total:.3f} ms -> {new_total:.3f} ms '
          f'({new_total/old_total:.1f}x)')
    print(f'  as a share of one core : {100*old_total/BUDGET_MS:.1f}% -> '
          f'{100*new_total/BUDGET_MS:.1f}%')
    print(f'  headroom               : {BUDGET_MS/old_total:.0f}x -> '
          f'{BUDGET_MS/new_total:.1f}x real time')
    bar = 20.0
    verdict = 'PASS' if 100 * new_total / BUDGET_MS <= bar else 'OVER BAR'
    print(f'  decision #18 bar (<= {bar:.0f}% of a core): {verdict}')

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'buffer_budget_ms': BUDGET_MS, 'rows': rows,
             'old_total_ms': old_total, 'new_total_ms': new_total,
             'verdict': verdict}, indent=2) + '\n')
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
