"""Compact, diffable digest of one fast-sim run — the NN integration's golden fixture."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

SURVIVING_BEAT_COLUMNS = ('t', 'bpm', 'change', 'rms')
RESCALED_BEAT_COLUMNS = ('strength',)
DOOMED_BEAT_COLUMNS = ('onset_density', 'kick_strength', 'centroid_trend',
                       'sub_bass_ratio')
DOOMED_METRIC_KEYS = ('onset_density_mean',)

# Matches the command-timing tolerance tests/test_simulation.py holds the queue to.
TIMING_ACCURACY_MAX_MS = 10.0

_EFFECT_MIDI_LABELS = ('set_autoloop', 'set_special_effect')
_MIDI_LABEL_TO_EFFECT_TYPE = {'set_autoloop': 'AUTOLOOP',
                              'set_special_effect': 'SPECIAL_EFFECT'}

_STREAM_NAMES = ('sound', 'os2l', 'midi', 'overlay')
_OVERLAY_LIGHT_BAR = 'LIGHT_BAR_24'


def _hash(values) -> str:
    canonical = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _median(xs):
    if not xs:
        return 0.0
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _normalise_streams(streams: dict | None) -> dict:
    streams = streams or {}
    return {name: list(streams.get(name, ())) for name in _STREAM_NAMES}


def _non_decreasing(values) -> bool:
    values = list(values)
    return all(a <= b for a, b in zip(values, values[1:]))


def _queue_errors_ms(report: dict) -> list:
    return [abs(e['actual_delta_sec'] - e['target_delta_sec']) * 1000
            for e in report['timing_log']]


def _queue_error_buffers(report: dict) -> int:
    import math

    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE

    period_ms = BUFFER_SIZE / SAMPLE_RATE * 1000
    return int(math.ceil(max(_queue_errors_ms(report), default=0.0) / period_ms))


def _sound_events(events: list) -> list:
    return [{'t': round(e['t'], 3), 'playing': e['playing']} for e in events]


def _os2l_digest(events: list) -> dict:
    beats = [e for e in events if e['label'] == 'beat']
    wire = sorted(set(beats[0]) - {'label', 'time'}) if beats else []
    return {
        'beats': len(beats),
        'wire_keys': wire,
        'stream_hash': _hash([[e['change'], e['pos'], round(e['bpm'], 4),
                               e['strength']] for e in beats]),
        'strengths': sorted({e['strength'] for e in beats}),
        'positions': [beats[0]['pos'], beats[-1]['pos']] if beats else [],
    }


def survives_digest(report: dict, streams: dict | None = None) -> dict:
    beats = report['beats']
    metrics = report['metrics']
    streams = _normalise_streams(streams)
    return {
        'beats_detected': metrics['beats_detected'],
        'beat_times_hash': _hash([round(b['t'], 4) for b in beats]),
        'beat_columns_hash': _hash([[b[c] for c in SURVIVING_BEAT_COLUMNS]
                                    for b in beats]),
        'bpm_median': round(_median(sorted(b['bpm'] for b in beats if b['bpm'] > 0)), 4),
        'bpm_last': metrics['bpm_last'],
        'rms_mean': round(sum(b['rms'] for b in beats) / len(beats), 6) if beats else 0.0,
        'duration_sec': round(report['duration_sec'], 3),
        'sound_events': _sound_events(streams['sound']),
        'os2l': _os2l_digest(streams['os2l']),
        'timing_accuracy': {
            'max_error_buffers': _queue_error_buffers(report),
        },
        'schema': {
            'report_keys': sorted(report.keys()),
            'metric_keys': sorted(set(metrics) - set(DOOMED_METRIC_KEYS)),
            'beat_keys': (sorted(set(beats[0]) - set(DOOMED_BEAT_COLUMNS))
                          if beats else []),
        },
    }


def _effects_match_report(midi: list, report_effects: list) -> bool:
    applied = [(_MIDI_LABEL_TO_EFFECT_TYPE[e['label']], e['channel'])
               for e in midi if e['label'] in _EFFECT_MIDI_LABELS]
    return applied == [(e['type'], e['channel']) for e in report_effects]


def _intent_at(intents: list, t: float) -> str | None:
    current = None
    for block in intents:
        if block['t'] <= t + 1e-9:
            current = block['intent']
    return current


def _channels_come_from_the_intents_pool(report: dict) -> bool:
    from lib.engine.effect_definitions import INTENT_EFFECTS

    pools = {intent.value: {effect.midi_channel.name for effect in effects}
             for intent, effects in INTENT_EFFECTS.items()}
    return all(effect['channel'] in pools.get(_intent_at(report['intents'],
                                                         effect['t']), ())
               for effect in report['effects'])


def relations_digest(report: dict, streams: dict | None = None) -> dict:
    streams = _normalise_streams(streams)
    light_bar = [e for e in streams['overlay'] if e['effect'] == _OVERLAY_LIGHT_BAR]
    errors_ms = _queue_errors_ms(report)
    return {
        'midi_arrives_in_enqueue_order': _non_decreasing(e['time']
                                                         for e in streams['midi']),
        'midi_matches_the_report_effects': _effects_match_report(streams['midi'],
                                                                 report['effects']),
        'midi_channels_come_from_the_intents_pool':
            _channels_come_from_the_intents_pool(report),
        'overlay_light_bar_fires': bool(light_bar),
        'overlay_arrives_in_enqueue_order': _non_decreasing(e['time']
                                                            for e in light_bar),
        'queue_error_within_tolerance': max(errors_ms, default=0.0) < TIMING_ACCURACY_MAX_MS,
    }


def informational_digest(report: dict, streams: dict | None = None) -> dict:
    metrics = report['metrics']
    streams = _normalise_streams(streams)
    midi = streams['midi']
    digest = {
        'intent_changes_count': metrics['intent_changes_count'],
        'effect_changes_count': metrics['effect_changes_count'],
        'unique_intents_count': metrics['unique_intents_count'],
        'dominant_intent': metrics['dominant_intent'],
        'intent_distribution_sec': metrics['intent_distribution_sec'],
        'midi': {
            'commands': len(midi),
            'labels': sorted({e['label'] for e in midi}),
            'ordering_hash': _hash([[e['label'], e['channel']] for e in midi]),
        },
        'overlay': {
            'light_bar_updates': sum(1 for e in streams['overlay']
                                     if e['effect'] == _OVERLAY_LIGHT_BAR),
        },
        'timing': {
            'target_delta_sec': sorted({round(e['target_delta_sec'], 4)
                                        for e in report['timing_log']}),
            'commands': len(report['timing_log']),
        },
    }
    return digest


def degradation_digest(report: dict, streams: dict | None = None) -> dict:
    from lib.engine.effect_definitions import LightIntent

    beats = report['beats']
    intents = report['intents']
    classified = [e for e in intents
                  if e['intent'] != LightIntent.ATMOSPHERIC.value]
    return {
        'beats_detected': report['metrics']['beats_detected'],
        'beat_times_hash': _hash([round(b['t'], 4) for b in beats]),
        'sound_events': _sound_events(_normalise_streams(streams)['sound']),
        'intent_blocks': len(intents),
        'classified_blocks': len(classified),
        'atmospheric_blocks': len(intents) - len(classified),
        'intents_held': sorted({e['intent'] for e in intents}),
        'effect_changes': report['metrics']['effect_changes_count'],
    }


def held_start_to_end(degradation: dict) -> bool:
    return (degradation['intent_blocks'] >= 1
            and degradation['classified_blocks'] <= 1
            and len([i for i in degradation['intents_held']
                     if i != 'atmospheric']) <= 1
            and degradation['effect_changes'] <= degradation['atmospheric_blocks'] + 1)


def digest_report(report: dict, streams: dict | None = None, *,
                  wall_elapsed: float | None = None) -> dict:
    digest = {
        'survives': survives_digest(report, streams),
        'relations': relations_digest(report, streams),
        'informational': informational_digest(report, streams),
        'degradation': degradation_digest(report, streams),
    }
    if wall_elapsed is not None:
        digest['speed'] = {
            'wall_elapsed_sec': round(wall_elapsed, 3),
            'realtime_factor': round(report['duration_sec'] / max(wall_elapsed, 1e-9), 2),
        }
    return digest


def check_digest(name: str, digest: dict, fixture: dict) -> list:
    if name not in fixture:
        return [f'{name}: not in the fixture']
    failures = []
    expected = fixture[name]['survives']
    if digest['survives'] != expected:
        moved = sorted(set(digest['survives']) | set(expected))
        moved = [k for k in moved if expected.get(k) != digest['survives'].get(k)]
        failures.append(f'{name}: survivors moved -> {", ".join(moved)}')
    broken = sorted(k for k, v in digest['relations'].items() if v is not True)
    if broken:
        failures.append(f'{name}: relations broken -> {", ".join(broken)}')
    return failures


def capture_streams(components: dict) -> dict:
    return {
        'sound': components['event_buffer'].sound_events(),
        'os2l': components['os2l_client'].events,
        'midi': components['midi_client'].events,
        'overlay': components['overlay_client'].events,
    }


async def digest_track(path: str) -> dict:
    from lib.audio_config import BUFFER_SIZE, SAMPLE_RATE
    from simulate.evaluator import report_checksum
    from simulate.fake_audio_client import FileAudioClient
    from simulate.runner import run_fast_simulation_components

    wall_start = time.monotonic()
    components, command_queue = await run_fast_simulation_components(
        FileAudioClient(SAMPLE_RATE, BUFFER_SIZE, path)
    )
    wall_elapsed = time.monotonic() - wall_start
    report = components['event_buffer'].to_report(command_queue.get_timing_log())
    digest = digest_report(report, capture_streams(components),
                           wall_elapsed=wall_elapsed)
    digest['report_checksum'] = report_checksum(report)
    return digest


def fixture_key(path: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_assets import opaque_name
    from select_eval_set import EVAL_SET_FILE, load_eval_set

    stem = Path(path).stem
    for track in load_eval_set(EVAL_SET_FILE)['tracks']:
        youtube_id = track['youtube_id']
        if opaque_name(youtube_id) == stem:
            return f'{youtube_id}.mp3'
    return Path(path).name


def _drop_machine_dependent_speed(digests: dict) -> None:
    for digest in digests.values():
        digest.pop('speed', None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tracks', nargs='+')
    ap.add_argument('--write', metavar='PATH',
                    help='write the digests to PATH as a JSON object keyed by track name')
    ap.add_argument('--check', metavar='PATH',
                    help='compare the survivors and relations against a committed '
                         'fixture and exit non-zero on any difference')
    args = ap.parse_args()

    digests = {}
    for track in args.tracks:
        name = fixture_key(track)
        print(f'--- {name}')
        digests[name] = asyncio.run(digest_track(track))
        print(json.dumps(digests[name], indent=2, sort_keys=True))

    if args.write:
        out = Path(args.write)
        out.parent.mkdir(parents=True, exist_ok=True)
        _drop_machine_dependent_speed(digests)
        out.write_text(json.dumps(digests, indent=2, sort_keys=True) + '\n',
                       newline='\n')
        print(f'\nwrote {out}')

    if args.check:
        fixture = json.loads(Path(args.check).read_text())
        failures = [failure for name, digest in digests.items()
                    for failure in check_digest(name, digest, fixture)]
        print('\n' + ('\n'.join(failures) if failures
                      else f'all {len(digests)} tracks match {args.check}'))
        raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
