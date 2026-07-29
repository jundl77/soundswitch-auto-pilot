"""Cross-process determinism proof: same audio in, byte-identical report out."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path


def _report_for(track: str) -> dict:
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from simulate.evaluator import report_checksum
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import run_fast_simulation

    async def run():
        _, event_buffer, queue = await run_fast_simulation(
            FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, track))
        return event_buffer.to_report(queue.get_timing_log())

    report = asyncio.run(run())
    return {'checksum': report_checksum(report),
            'beats': report['metrics']['beats_detected'],
            'bpm_last': report['metrics']['bpm_last'],
            'intents': report['metrics']['intent_distribution_sec'],
            'effects': report['metrics']['effect_changes_count']}


def _spawn(track: str) -> dict:
    # A fresh interpreter is the point: hash seeds, import-time RNG and cached
    # models all survive an in-process rerun.
    out = subprocess.run(
        [sys.executable, __file__, '--emit', track],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tracks', nargs='+')
    ap.add_argument('--emit', action='store_true',
                    help=argparse.SUPPRESS)
    ap.add_argument('--runs', type=int, default=2)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.emit:
        print(json.dumps(_report_for(args.tracks[0])))
        return

    verdicts, failures = {}, []
    for track in args.tracks:
        name = Path(track).name
        results = [_spawn(track) for _ in range(args.runs)]
        identical = all(r == results[0] for r in results)
        verdicts[name] = {'identical': identical, **results[0],
                          'all_checksums': [r['checksum'] for r in results]}
        status = 'IDENTICAL' if identical else 'DIVERGED'
        print(f'{name:36s} {status}  sha {results[0]["checksum"][:16]}  '
              f'beats {results[0]["beats"]}')
        if not identical:
            failures.append(name)
            for i, r in enumerate(results):
                print(f'    run {i}: {r}')

    if args.out:
        Path(args.out).write_text(json.dumps(verdicts, indent=2, sort_keys=True) + '\n')
        print(f'wrote {args.out}')
    if failures:
        raise SystemExit(f'NOT DETERMINISTIC: {failures}')
    print(f'\nall {len(args.tracks)} tracks identical across '
          f'{args.runs} fresh processes')


if __name__ == '__main__':
    main()
