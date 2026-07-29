"""Compact, diffable digest of one fast-sim run — the migration's regression anchor.

A full sim report is thousands of beat rows, so diffing two of them tells you
*that* something moved but not *what class* of thing moved. This splits a report
into the two things a rhythm-source migration must keep apart:

  rhythm     — beat times, beat count, BPM, onset density. EXPECTED to change
               when the beat/onset source changes. Recorded so the size of the
               change is a number rather than an impression.

  filterbank — the aubio mel bank's output over the track on a FIXED TIME GRID,
               plus the report's schema. MUST NOT change.

  at_beats   — the same filterbank columns as the report carries them, i.e.
               sampled at beat instants. Reported as evidence, never gated.

The third section exists because the second one has to. The beat-sampled columns
cannot be a regression gate: they are filterbank output read at beat times, so
moving the beat grid moves them by construction, and a gate on them fires on the
expected change while a real filterbank regression hides inside it. Sampling the
bank on a fixed grid instead gives an anchor that a moved grid cannot perturb —
so it can still fail, under a moved grid, for the one reason it exists.

Usage:
    python training/pipeline_digest.py samples/song.mp3 [more.mp3 ...]
    python training/pipeline_digest.py --write tests/fixtures/... track.mp3 ...
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

# Beat-row columns produced by the rhythm source (allowed to move).
RHYTHM_COLUMNS = ('t', 'bpm', 'onset_density', 'strength', 'change')
# Beat-row columns produced by the aubio filterbank (must not move as a result
# of the migration — they can only shift because the beats they are sampled at
# shifted, which is why they are hashed separately from their timestamps).
SPECTRAL_COLUMNS = ('kick_strength', 'centroid_trend', 'sub_bass_ratio', 'rms')


def _hash(values) -> str:
    canonical = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def digest_report(report: dict, *, wall_elapsed: float | None = None) -> dict:
    """Reduce a sim report to the fields a migration is judged on."""
    beats = report['beats']
    metrics = report['metrics']

    densities = sorted(b['onset_density'] for b in beats)
    bpms = sorted(b['bpm'] for b in beats if b['bpm'] > 0)

    def _median(xs):
        if not xs:
            return 0.0
        mid = len(xs) // 2
        return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2

    digest = {
        # --- schema: the shape of the contract, must not move ---------------
        'schema': {
            'report_keys': sorted(report.keys()),
            'metric_keys': sorted(metrics.keys()),
            'beat_keys': sorted(beats[0].keys()) if beats else [],
        },
        # --- at_beats: the same columns as the report carries them ----------
        # Evidence only. Moves whenever the beat grid moves; see module docstring.
        'at_beats': {
            'columns_hash': _hash([[b[c] for c in SPECTRAL_COLUMNS] for b in beats]),
            'kick_strength_mean': round(
                sum(b['kick_strength'] for b in beats) / len(beats), 6) if beats else 0.0,
            'sub_bass_ratio_mean': round(
                sum(b['sub_bass_ratio'] for b in beats) / len(beats), 6) if beats else 0.0,
            'rms_mean': round(sum(b['rms'] for b in beats) / len(beats), 6) if beats else 0.0,
        },
        # --- rhythm: madmom's job now, expected to move ----------------------
        'rhythm': {
            'columns_hash': _hash([[b[c] for c in RHYTHM_COLUMNS] for b in beats]),
            'beats_detected': metrics['beats_detected'],
            'beat_times_hash': _hash([round(b['t'], 4) for b in beats]),
            'bpm_median': round(_median(bpms), 4),
            'onset_density_mean': round(metrics['onset_density_mean'], 6),
            'onset_density_median': round(_median(densities), 6),
        },
        # --- show: what the audience sees ------------------------------------
        'show': {
            'duration_sec': round(report['duration_sec'], 3),
            'intent_changes_count': metrics['intent_changes_count'],
            'effect_changes_count': metrics['effect_changes_count'],
            'unique_intents_count': metrics['unique_intents_count'],
            'dominant_intent': metrics['dominant_intent'],
            'intent_distribution_sec': metrics['intent_distribution_sec'],
            'timing_error_max_ms': round(metrics['timing_error_max_ms'], 4),
        },
    }
    if wall_elapsed is not None:
        digest['speed'] = {
            'wall_elapsed_sec': round(wall_elapsed, 3),
            'realtime_factor': round(report['duration_sec'] / max(wall_elapsed, 1e-9), 2),
        }
    return digest


def filterbank_fingerprint(path: str, seconds: float = 120.0) -> dict:
    """The aubio mel bank's output over a fixed time grid — the anchor.

    Runs the analyser's own mel path buffer by buffer, so it measures exactly
    what the pipeline computes, and reduces it on a one-second grid so the
    result depends on the audio and the bank alone. No beat, onset or tempo
    decision can reach it, which is the whole point: it stays sensitive to a
    filterbank regression while the beat grid underneath it moves.
    """
    import numpy as np

    from lib.analyser.music_analyser import MusicAnalyser
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from simulate.fake_audio_client import FileAudioClient

    class _Silent:
        def on_sound_start(self): pass
        def on_sound_stop(self): pass
        async def on_cycle(self): pass
        async def on_onset(self): pass
        async def on_beat(self, *a): pass
        async def on_note(self): pass
        async def on_section_change(self): pass

    client = FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, path)
    client.start_streams()
    analyser = MusicAnalyser(SAMPLE_RATE, BUFFER_SIZE, _Silent())

    per_second, rows = [], []
    buffers_per_second = SAMPLE_RATE // BUFFER_SIZE
    limit = int(seconds * buffers_per_second)
    for i in range(limit):
        if client.exhausted:
            break
        rows.append(analyser._compute_mel_energies(client.read()))
        if len(rows) == buffers_per_second:
            per_second.append(np.mean(np.asarray(rows, dtype=np.float32), axis=0))
            rows = []
    grid = np.asarray(per_second, dtype=np.float32)
    return {
        'grid_shape': list(grid.shape),
        'grid_hash': hashlib.sha256(grid.tobytes()).hexdigest()[:16],
        'grid_sum': round(float(grid.sum()), 4),
    }


async def digest_track(path: str) -> dict:
    """Run one track through the fast sim and digest the result."""
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from simulate.evaluator import report_checksum
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import run_fast_simulation

    wall_start = time.monotonic()
    _, event_buffer, command_queue = await run_fast_simulation(
        FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, path)
    )
    wall_elapsed = time.monotonic() - wall_start
    report = event_buffer.to_report(command_queue.get_timing_log())
    digest = digest_report(report, wall_elapsed=wall_elapsed)
    digest['filterbank'] = filterbank_fingerprint(path)
    digest['report_checksum'] = report_checksum(report)
    return digest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tracks', nargs='+')
    ap.add_argument('--write', metavar='PATH',
                    help='write the digests to PATH as a JSON object keyed by track name')
    args = ap.parse_args()

    digests = {}
    for track in args.tracks:
        name = Path(track).name
        print(f'--- {name}')
        digests[name] = asyncio.run(digest_track(track))
        print(json.dumps(digests[name], indent=2, sort_keys=True))

    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Speed is machine-dependent: it belongs in a report, never in a fixture
        # that a test compares against.
        for d in digests.values():
            d.pop('speed', None)
        out.write_text(json.dumps(digests, indent=2, sort_keys=True) + '\n')
        print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
