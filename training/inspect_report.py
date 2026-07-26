"""Inspect a simulation report: per-10s feature/intent bins + intent timeline.

Usage: python training/inspect_report.py report.json [--bin-sec 10]
"""
import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('report')
    ap.add_argument('--bin-sec', type=float, default=10.0)
    args = ap.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    beats = report['beats']
    intents = report['intents']
    duration = report['duration_sec']
    # Beat rows are song time; intent blocks are audience time (song + look-ahead).
    # Shift the blocks back so a bin's features and its intent describe the same
    # moment of music. Older reports predate the metric — assume no offset.
    look_ahead = report['metrics'].get('look_ahead_sec', 0.0)

    def dominant_intent_at(t0: float, t1: float) -> str:
        """Dominant intent over [t0, t1) in *song* time."""
        best, best_overlap = '-', 0.0
        for block in intents:
            start = block['t'] - look_ahead
            end = block.get('end', duration) - look_ahead
            overlap = min(end, t1) - max(start, t0)
            if overlap > best_overlap:
                best, best_overlap = block['intent'], overlap
        return best

    print(f'{"bin":>12}  {"rms":>6}  {"density":>7}  {"kick":>5}  {"beats":>5}  intent')
    t = 0.0
    while t < duration:
        t1 = min(t + args.bin_sec, duration)
        rows = [b for b in beats if t <= b['t'] < t1]
        if rows:
            rms = sum(b.get('rms', 0.0) for b in rows) / len(rows)
            den = sum(b['onset_density'] for b in rows) / len(rows)
            kick = sum(b.get('kick_strength', 1.0) for b in rows) / len(rows)
            print(f'{t:>5.0f}-{t1:<5.0f}  {rms:>6.3f}  {den:>7.2f}  {kick:>5.2f}  {len(rows):>5}  {dominant_intent_at(t, t1)}')
        else:
            print(f'{t:>5.0f}-{t1:<5.0f}  {"-":>6}  {"-":>7}  {"-":>5}  {0:>5}  {dominant_intent_at(t, t1)}')
        t = t1

    print(f'\nIntent timeline — raw audience time '
          f'(= song time + metrics.look_ahead_sec = {look_ahead:.2f}s); '
          f'the bin table above is de-shifted to song time:')
    for block in intents:
        end = block.get('end', duration)
        print(f"  {block['t']:>7.1f} - {end:<7.1f}  ({end - block['t']:>6.1f}s)  {block['intent']}")

    print('\nDistribution:', report['metrics']['intent_distribution_sec'])


if __name__ == '__main__':
    main()
