"""Inspect a simulation report: per-10s rms/beat/intent bins + intent timeline."""
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
    # Intent blocks are stamped in audience time.  The delay behind each one is
    # per command now (B1), so no constant reaches song time -- a block waited
    # `playback_delay` only while the chain fit inside it, and on a slow track
    # it waited less.  The engine records the instant it means as `song_t`
    # precisely so nothing has to guess; `look_ahead_sec` is the fallback for
    # reports cut before that field existed.
    look_ahead = report['metrics'].get('look_ahead_sec', 0.0)

    def song_bounds(block: dict) -> tuple:
        shift = block['t'] - block['song_t'] if 'song_t' in block else look_ahead
        return block['t'] - shift, block.get('end', duration) - shift

    def dominant_intent_at(t0: float, t1: float) -> str:
        best, best_overlap = '-', 0.0
        for block in intents:
            start, end = song_bounds(block)
            overlap = min(end, t1) - max(start, t0)
            if overlap > best_overlap:
                best, best_overlap = block['intent'], overlap
        return best

    print(f'{"bin":>12}  {"rms":>6}  {"beats":>5}  intent')
    t = 0.0
    while t < duration:
        t1 = min(t + args.bin_sec, duration)
        rows = [b for b in beats if t <= b['t'] < t1]
        if rows:
            rms = sum(b.get('rms', 0.0) for b in rows) / len(rows)
            print(f'{t:>5.0f}-{t1:<5.0f}  {rms:>6.3f}  {len(rows):>5}  {dominant_intent_at(t, t1)}')
        else:
            print(f'{t:>5.0f}-{t1:<5.0f}  {"-":>6}  {0:>5}  {dominant_intent_at(t, t1)}')
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
