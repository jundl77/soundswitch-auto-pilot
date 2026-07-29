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


# Splits this calibration must never read. `test` is the campaign holdout.
# `eval_set` and `excluded_eval_set` are the benchmark tracks whose before/after
# deltas are this PR's headline evidence — a constant tuned on them would make
# that evidence a measurement of its own training set.
_FORBIDDEN_SPLITS = ('test', 'eval_set', 'excluded_eval_set')


def _golden_fixture_ids() -> set[str]:
    """Tracks whose before/after deltas are the migration's headline evidence.

    They are not in any split manifest — the bundled sample is not corpus audio
    at all — so the manifest cannot protect them. Read from the fixture itself
    so the list cannot drift from the thing it describes.
    """
    fixture = (Path(__file__).parent.parent / 'tests' / 'fixtures'
               / 'pipeline_digest_baseline.json')
    if not fixture.exists():
        return set()
    return {Path(name).stem for name in json.loads(fixture.read_text())}


def _forbidden_ids(data_dir: Path) -> set[str]:
    """Every id this calibration is not allowed to read.

    A missing manifest is fatal rather than empty: silently treating "I could
    not find the holdout list" as "there is no holdout" is how test contact
    happens — it is how it happened here the first time.
    """
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
    """The threshold whose median rate is closest to aubio's, over these tracks."""
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
    """How much the matched threshold depends on WHICH tracks were drawn.

    This exists because it caught a real error: three successive 17-track pools
    produced 0.40, 0.44 and 0.35 — two full steps of the ladder — and each was
    reported as if it were the answer. A median over a small track sample is an
    estimate, and shipping a point estimate without its spread is how a sampling
    artifact becomes a constant.
    """
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
    # Whole tracks by default, and that is not a detail. Measured on 90 s
    # prefixes this sweep matches at 0.30 and on 240 s prefixes at 0.40: the
    # two detectors' rate ratio drifts upward with the density of the material,
    # and a track's opening minutes are its sparsest. A prefix therefore
    # calibrates against an intro, while the rule engine sees whole shows.
    ap.add_argument('--seconds', type=float, default=None,
                    help='truncate each track (diagnostic only — see above)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20260729)
    ap.add_argument('--from-json', default=None,
                    help='re-analyse an existing sweep instead of measuring again')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # Re-analysis path: the per-track rates are the expensive part and they are
    # already stored, so re-reading them costs a second instead of an hour. This
    # is how the stability record gets added to a sweep that predates it.
    if args.from_json:
        results = json.loads(Path(args.from_json).read_text())['tracks']
        print(f'{len(results)} tracks, re-analysed from {args.from_json}')
        _report(results, args.seed, args.out)
        return

    if not args.audio_dir:
        raise SystemExit('--audio-dir is required unless --from-json is given')

    # Deterministic, selection-free sample: sorted filenames, evenly spaced.
    # Not the eval set — that list lives on another branch and this measurement
    # must not depend on it.
    #
    # Held-out ids are excluded LOUDLY and then refused. The first version of
    # this script had no guard at all and the sorted-and-spaced selection duly
    # picked up two test tracks; a constant chosen with holdout contact cannot
    # be defended, however small the contribution. The exclusion is printed
    # rather than done quietly, because "the pool was smaller than you think" is
    # exactly the kind of thing that should not be discovered by reading source.
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
        # The stability record is STORED, not merely printed. A sweep whose
        # spread lives only in a terminal scrollback is a sweep whose spread
        # will be forgotten by the next person to quote its point estimate.
        Path(out).write_text(json.dumps(
            {'tracks': results, 'chosen_threshold': best[0],
             'stability': stability}, indent=2) + '\n')
        print(f'wrote {out}')


if __name__ == '__main__':
    try:
        import psutil
        psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001 — priority is a courtesy, not a requirement
        pass
    main()
