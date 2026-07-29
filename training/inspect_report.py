"""Inspect a simulation report: per-10s feature/intent bins + intent timeline."""
import argparse
import json

from lib.analyser.music_analyser import density_is_known


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
    # Intent blocks are stamped in audience time; shift them back to song time.
    look_ahead = report['metrics'].get('look_ahead_sec', 0.0)

    def dominant_intent_at(t0: float, t1: float) -> str:
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
            measured = [b['onset_density'] for b in rows
                        if density_is_known(b['onset_density'])]
            den = f'{sum(measured) / len(measured):>7.2f}' if measured else f'{"unmeas.":>7}'
            kick = sum(b.get('kick_strength', 1.0) for b in rows) / len(rows)
            print(f'{t:>5.0f}-{t1:<5.0f}  {rms:>6.3f}  {den}  {kick:>5.2f}  {len(rows):>5}  {dominant_intent_at(t, t1)}')
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
