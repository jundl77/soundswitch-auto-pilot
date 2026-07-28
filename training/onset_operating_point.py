"""Set madmom's onset peak-picking threshold by matching the stream it replaces.

madmom's peak picker needs a threshold; aubio's `onset("default")` had one baked
in. Every onset-density constant in `lib/engine/light_engine.py` is denominated
in the RATE that detector produced, so taking madmom's library default would
silently move the rule engine's input distribution and make the migration
impossible to judge — the deltas would be a mix of "better beat source" and
"different onset budget" with no way to separate them.

Matching the median onset rate across a corpus is therefore the opposite of
retuning the rule engine: it holds the engine's input fixed so that what
changed is what actually changed.

Run:  python training/onset_operating_point.py --audio-dir DIR [--tracks 16]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SR = 44100
BUFFER = 256
THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.50, 0.60, 0.70)


def _load(path: Path, seconds: float) -> np.ndarray:
    cache = Path(f'{path}.{SR}.npy')
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        audio = np.load(cache)
    else:
        import librosa
        audio, _ = librosa.load(str(path), sr=SR, mono=True)
        audio = audio.astype(np.float32)
        np.save(cache, audio)
    return audio[: int(seconds * SR)]


def _aubio_onset_rate(audio: np.ndarray) -> float:
    """The rate the rule engine's density constants were calibrated against."""
    import aubio
    detector = aubio.onset('default', BUFFER * 2, BUFFER, SR)
    count = sum(1 for i in range(0, len(audio) - BUFFER, BUFFER)
                if detector(audio[i:i + BUFFER])[0] > 0)
    return count / (len(audio) / SR)


def _madmom_activations(audio: np.ndarray) -> np.ndarray:
    """One streaming pass; every threshold is then read off the same curve, so
    the sweep costs one inference rather than eleven."""
    from lib.analyser.madmom_rhythm import FRAME_SIZE, HOP_SIZE
    from madmom.audio.signal import Signal
    from madmom.features.onsets import RNNOnsetProcessor
    from madmom.processors import BufferProcessor

    rnn = RNNOnsetProcessor(online=True, origin='stream', num_frames=1, fps=100)
    buffer = BufferProcessor(buffer_size=FRAME_SIZE)
    buffer(np.zeros(FRAME_SIZE, dtype=np.float32))

    activations, primed = [], False
    for i in range(0, len(audio) - HOP_SIZE, HOP_SIZE):
        hop = Signal(audio[i:i + HOP_SIZE], sample_rate=SR, num_channels=1)
        act = rnn(buffer(hop), reset=not primed)
        primed = True
        activations.append(float(np.atleast_1d(act).flatten()[-1]))
    return np.asarray(activations, dtype=np.float32)


def _rates_by_threshold(activations: np.ndarray, duration: float) -> dict[str, float]:
    from madmom.features.onsets import OnsetPeakPickingProcessor
    rates = {}
    for threshold in THRESHOLDS:
        picker = OnsetPeakPickingProcessor(online=True, fps=100, threshold=threshold)
        picker.reset()
        count = sum(len(np.atleast_1d(
            picker.process_online(np.array([a]), reset=False))) for a in activations)
        rates[f'{threshold:.2f}'] = count / duration
    return rates


def one_track(args) -> dict:
    path, seconds = args
    audio = _load(Path(path), seconds)
    duration = len(audio) / SR
    started = time.perf_counter()
    activations = _madmom_activations(audio)
    return {
        'track': Path(path).name,
        'duration_sec': round(duration, 1),
        'aubio_rate': round(_aubio_onset_rate(audio), 4),
        'madmom_rates': {k: round(v, 4) for k, v in
                         _rates_by_threshold(activations, duration).items()},
        'realtime_factor': round(duration / (time.perf_counter() - started), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audio-dir', required=True)
    ap.add_argument('--extra', nargs='*', default=[],
                    help='additional audio files (e.g. the bundled sample)')
    ap.add_argument('--tracks', type=int, default=16)
    ap.add_argument('--seconds', type=float, default=90.0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # Deterministic, selection-free sample: sorted filenames, evenly spaced.
    # Not the eval set — that list lives on another branch and this measurement
    # must not depend on it.
    everything = sorted(Path(args.audio_dir).glob('*.mp3'))
    step = max(1, len(everything) // args.tracks)
    chosen = [str(p) for p in everything[::step][: args.tracks]] + list(args.extra)
    print(f'{len(chosen)} tracks x {args.seconds:.0f}s, {args.workers} workers')

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one_track, [(t, args.seconds) for t in chosen]))

    aubio_rates = np.array([r['aubio_rate'] for r in results])
    print(f'\naubio onset rate: median {np.median(aubio_rates):.3f}/s  '
          f'mean {aubio_rates.mean():.3f}  '
          f'p10 {np.percentile(aubio_rates, 10):.3f}  '
          f'p90 {np.percentile(aubio_rates, 90):.3f}')
    print(f'\n{"thr":>6} {"median":>8} {"mean":>8} {"vs aubio":>10} '
          f'{"per-track ratio p10/p50/p90":>32}')
    best = None
    for threshold in THRESHOLDS:
        key = f'{threshold:.2f}'
        rates = np.array([r['madmom_rates'][key] for r in results])
        ratios = rates / np.maximum(aubio_rates, 1e-9)
        delta = np.median(rates) - np.median(aubio_rates)
        print(f'{key:>6} {np.median(rates):8.3f} {rates.mean():8.3f} '
              f'{delta:+10.3f}   {np.percentile(ratios, 10):.2f} / '
              f'{np.median(ratios):.2f} / {np.percentile(ratios, 90):.2f}')
        if best is None or abs(delta) < abs(best[1]):
            best = (threshold, delta)

    print(f'\nrate-matched threshold: {best[0]:.2f} '
          f'(median delta {best[1]:+.3f}/s)')
    print(f"madmom's library default (0.50) would have given "
          f"{np.median([r['madmom_rates']['0.50'] for r in results]):.3f}/s")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {'tracks': results, 'chosen_threshold': best[0]}, indent=2) + '\n')
        print(f'wrote {args.out}')


if __name__ == '__main__':
    try:
        import psutil
        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001 — priority is a courtesy, not a requirement
        pass
    main()
