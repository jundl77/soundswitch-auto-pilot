"""Set madmom's onset peak-picking threshold by matching aubio's onset rate."""

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


def _load(path: Path, seconds: float | None) -> np.ndarray:
    cache = Path(f'{path}.{SR}.npy')
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        audio = np.load(cache)
    else:
        import librosa
        audio, _ = librosa.load(str(path), sr=SR, mono=True)
        audio = audio.astype(np.float32)
        np.save(cache, audio)
    return audio if seconds is None else audio[: int(seconds * SR)]


def _aubio_onset_rate(audio: np.ndarray) -> float:
    import aubio
    detector = aubio.onset('default', BUFFER * 2, BUFFER, SR)
    count = sum(1 for i in range(0, len(audio) - BUFFER, BUFFER)
                if detector(audio[i:i + BUFFER])[0] > 0)
    return count / (len(audio) / SR)


def _madmom_activations(audio: np.ndarray) -> np.ndarray:
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


# Splits this calibration must never read: tuning on them would make the
# migration's before/after evidence a measurement of its own training set.
_FORBIDDEN_SPLITS = ('test', 'eval_set', 'excluded_eval_set')


def _golden_fixture_ids() -> set[str]:
    fixture = (Path(__file__).parent.parent / 'tests' / 'fixtures'
               / 'pipeline_digest_baseline.json')
    if not fixture.exists():
        return set()
    return {Path(name).stem for name in json.loads(fixture.read_text())}


def _forbidden_ids(data_dir: Path) -> set[str]:
    manifest = data_dir / 'splits.json'
    if not manifest.exists():
        raise SystemExit(f'refusing to select tracks without {manifest} — '
                         f'cannot prove the sample is holdout-free')
    splits = json.loads(manifest.read_text())
    ids: set[str] = set()
    for name in _FORBIDDEN_SPLITS:
        ids |= set(splits.get(name, []))
    return ids | _golden_fixture_ids()


def _refuse_holdout_contact(paths: list[str], data_dir: Path) -> None:
    offenders = sorted(p for p in paths if Path(p).stem in _forbidden_ids(data_dir))
    if offenders:
        raise SystemExit(f'refusing: held-out tracks in the sample {offenders}')


def _matched_threshold(rows: list[dict]) -> float:
    import numpy as np
    aubio_median = float(np.median([r['aubio_rate'] for r in rows]))
    best = None
    for threshold in THRESHOLDS:
        key = f'{threshold:.2f}'
        delta = float(np.median([r['madmom_rates'][key] for r in rows])) - aubio_median
        if best is None or abs(delta) < abs(best[1]):
            best = (threshold, delta)
    return best[0]


def _stability(rows: list[dict], seed: int, resamples: int = 2000) -> dict:
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(rows)
    draws = []
    for _ in range(resamples):
        sample = [rows[i] for i in rng.integers(0, n, n)]
        draws.append(_matched_threshold(sample))
    draws = np.asarray(draws)
    values, counts = np.unique(draws, return_counts=True)
    order = np.argsort(-counts)
    return {
        'resamples': resamples,
        'seed': seed,
        'distribution': [(float(values[i]), float(counts[i] / resamples))
                         for i in order],
        'mode': float(values[int(np.argmax(counts))]),
        'p10': float(np.percentile(draws, 10)),
        'p90': float(np.percentile(draws, 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audio-dir', default=None,
                    help='corpus audio directory (not needed with --from-json)')
    ap.add_argument('--extra', nargs='*', default=[],
                    help='additional audio files (e.g. the bundled sample)')
    ap.add_argument('--tracks', type=int, default=16)
    # Whole tracks by default: the two detectors' rate ratio rises with the
    # density of the material, so a prefix calibrates against an intro.
    ap.add_argument('--seconds', type=float, default=None,
                    help='truncate each track (diagnostic only — see above)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20260729)
    ap.add_argument('--from-json', default=None,
                    help='re-analyse an existing sweep instead of measuring again')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.from_json:
        results = json.loads(Path(args.from_json).read_text())['tracks']
        print(f'{len(results)} tracks, re-analysed from {args.from_json}')
        _report(results, args.seed, args.out)
        return

    if not args.audio_dir:
        raise SystemExit('--audio-dir is required unless --from-json is given')

    data_dir = Path(args.audio_dir).parent
    forbidden = _forbidden_ids(data_dir)
    everything = sorted(Path(args.audio_dir).glob('*.mp3'))
    pool = [p for p in everything if p.stem not in forbidden]
    print(f'candidates {len(everything)}; excluded {len(everything) - len(pool)} '
          f'held-out ({", ".join(_FORBIDDEN_SPLITS)}); pool {len(pool)}')
    step = max(1, len(pool) // args.tracks)
    chosen = [str(p) for p in pool[::step][: args.tracks]] + list(args.extra)
    _refuse_holdout_contact(chosen, data_dir)
    span = 'whole tracks' if args.seconds is None else f'{args.seconds:.0f}s prefixes'
    print(f'{len(chosen)} tracks, {span}, {args.workers} workers')

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one_track, [(t, args.seconds) for t in chosen]))

    _report(results, args.seed, args.out)


def _report(results: list[dict], seed: int, out: str | None) -> None:
    import numpy as np
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

    stability = _stability(results, seed=seed)
    print(f'\nstability over {stability["resamples"]} track-resamples '
          f'(seed {seed}):')
    for threshold, share in stability['distribution']:
        bar = '#' * int(round(share * 40))
        print(f'  {threshold:.2f}  {share*100:5.1f}%  {bar}')
    print(f'  mode {stability["mode"]:.2f}, '
          f'80% interval [{stability["p10"]:.2f}, {stability["p90"]:.2f}]')
    if stability['p10'] != stability['p90']:
        print('  NOTE: the matched threshold is not resample-stable at this n — '
              'the interval, not the point estimate, is the honest result.')

    if out:
        Path(out).write_text(json.dumps(
            {'tracks': results, 'chosen_threshold': best[0],
             'stability': stability}, indent=2) + '\n')
        print(f'wrote {out}')


if __name__ == '__main__':
    try:
        import psutil
        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    main()
