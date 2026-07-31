"""Compact, diffable digest of one fast-sim run — the NN integration's golden fixture.

Cut on the pipeline as it stands at the branch point, so the demolition can prove
what it preserved rather than assert it.  Three blocks, and they are read
differently:

``survives``    what the integration is not allowed to move — the beat stream, the
                sound-start/stop instants, the OS2L beat wire, the MIDI ordering
                relation, queue accuracy, and the report keys that outlive the
                rule engine.  Compared for equality.
``degradation`` the projection onto what NN_SHED and the branch's own intermediate
                state may produce: beats, silence, one held intent, nothing else.
                ``is_degradation_state`` is the predicate, and it is asserted from
                both sides — False here, True after the demolition.
``show``        intent and effect counts.  Recorded so the demolition's effect on
                the show is visible; expected to move.

The aubio filterbank fingerprint that this file used to gate on is gone: the bank
is deleted by the integration, so a fixed-grid fingerprint of it can only fail for
the reason it was built to rule out.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from lib.analyser.music_analyser import density_is_known

# The beat row of a report, partitioned by what the NN integration does to it.
SURVIVING_BEAT_COLUMNS = ('t', 'bpm', 'change', 'rms')
# Survives as a key, not as a value: it is onset density rescaled into [0, 1],
# and the density chain is deleted underneath it.
RESCALED_BEAT_COLUMNS = ('strength',)
# Produced only to feed the threshold classifier, and deleted with it.
DOOMED_BEAT_COLUMNS = ('onset_density', 'kick_strength', 'centroid_trend',
                       'sub_bass_ratio')
DOOMED_METRIC_KEYS = ('onset_density_mean',)

# The queue's own dispatch error, which virtual time makes exact. Matches the
# tolerance tests/test_simulation.py holds the command timing to.
TIMING_ACCURACY_MAX_MS = 10.0

# The MIDI calls that must appear in the report's effect timeline. `set_color_override`
# and `clear_color_overrides` deliberately do not: they are not effect changes.
_EFFECT_MIDI_LABELS = ('set_autoloop', 'set_special_effect')
_MIDI_LABEL_TO_EFFECT_TYPE = {'set_autoloop': 'AUTOLOOP',
                              'set_special_effect': 'SPECIAL_EFFECT'}


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
    return {name: list(streams.get(name, ())) for name in ('sound', 'os2l', 'midi')}


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


def _midi_digest(events: list, report_effects: list) -> dict:
    times = [e['time'] for e in events]
    applied = [(_MIDI_LABEL_TO_EFFECT_TYPE[e['label']], e['channel'])
               for e in events if e['label'] in _EFFECT_MIDI_LABELS]
    recorded = [(e['type'], e['channel']) for e in report_effects]
    return {
        'commands': len(events),
        'labels': sorted({e['label'] for e in events}),
        'ordering_hash': _hash([[e['label'], e['channel']] for e in events]),
        'times_non_decreasing': all(a <= b for a, b in zip(times, times[1:])),
        'effects_match_report': applied == recorded,
    }


def survives_digest(report: dict, streams: dict | None = None) -> dict:
    """Everything the NN integration must leave exactly where it found it."""
    beats = report['beats']
    metrics = report['metrics']
    streams = _normalise_streams(streams)
    errors_ms = [abs(e['actual_delta_sec'] - e['target_delta_sec']) * 1000
                 for e in report['timing_log']]
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
        'midi': _midi_digest(streams['midi'], report['effects']),
        'timing_accuracy': {
            # The command count and the queue's target both move with the
            # look-ahead; the error the queue makes against its own target does not.
            'target_delta_sec': sorted({round(e['target_delta_sec'], 4)
                                        for e in report['timing_log']}),
            'max_error_ms': round(max(errors_ms), 4) if errors_ms else 0.0,
        },
        'schema': {
            'report_keys': sorted(report.keys()),
            'metric_keys': sorted(set(metrics) - set(DOOMED_METRIC_KEYS)),
            'beat_keys': (sorted(set(beats[0]) - set(DOOMED_BEAT_COLUMNS))
                          if beats else []),
        },
    }


def degradation_digest(report: dict, streams: dict | None = None) -> dict:
    """The report seen through the degradation contract, and nothing else."""
    beats = report['beats']
    intents = report['intents']
    return {
        'beats_detected': report['metrics']['beats_detected'],
        'beat_times_hash': _hash([round(b['t'], 4) for b in beats]),
        'sound_events': _sound_events(_normalise_streams(streams)['sound']),
        'intent_blocks': len(intents),
        'intents_held': sorted({e['intent'] for e in intents}),
        'effect_changes': report['metrics']['effect_changes_count'],
    }


def is_degradation_state(degradation: dict) -> bool:
    """Whether a digest describes a show that only ever held.

    The branch's intermediate state between demolition and rewire, and NN_SHED
    forever after, are the same contract: beats and the silence timer keep
    running, at most one intent is ever current, and no effect change follows the
    one that established it.
    """
    return (degradation['intent_blocks'] <= 1
            and len(degradation['intents_held']) <= 1
            and degradation['effect_changes'] <= 1)


def show_digest(report: dict) -> dict:
    """Evidence, never a gate: this is the half the demolition is meant to move."""
    metrics = report['metrics']
    densities = sorted(b['onset_density'] for b in report['beats']
                       if density_is_known(b['onset_density']))
    return {
        'intent_changes_count': metrics['intent_changes_count'],
        'effect_changes_count': metrics['effect_changes_count'],
        'unique_intents_count': metrics['unique_intents_count'],
        'dominant_intent': metrics['dominant_intent'],
        'intent_distribution_sec': metrics['intent_distribution_sec'],
        'onset_density_mean': round(metrics['onset_density_mean'], 6),
        'onset_density_median': round(_median(densities), 6),
    }


def digest_report(report: dict, streams: dict | None = None, *,
                  wall_elapsed: float | None = None) -> dict:
    digest = {
        'survives': survives_digest(report, streams),
        'degradation': degradation_digest(report, streams),
        'show': show_digest(report),
    }
    if wall_elapsed is not None:
        digest['speed'] = {
            'wall_elapsed_sec': round(wall_elapsed, 3),
            'realtime_factor': round(report['duration_sec'] / max(wall_elapsed, 1e-9), 2),
        }
    return digest


def capture_streams(components: dict) -> dict:
    """The three wires a report does not carry, read off the stub clients."""
    return {
        'sound': components['event_buffer'].snapshot()['sound_events'],
        'os2l': components['os2l_client'].events,
        'midi': components['midi_client'].events,
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
    """The name a track is filed under, which is the id a human can look up.

    Committed eval-set audio carries a name derived from the YouTube id and says
    nothing on its own, so the fixture would be three opaque strings nobody could
    match to a track. Derived rather than tabulated, the same way the audio path is.
    """
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('tracks', nargs='+')
    ap.add_argument('--write', metavar='PATH',
                    help='write the digests to PATH as a JSON object keyed by track name')
    ap.add_argument('--check', metavar='PATH',
                    help='compare the survivors against a committed fixture and '
                         'exit non-zero on any difference')
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
        # Machine-dependent: never in a fixture a test compares against.
        for d in digests.values():
            d.pop('speed', None)
        out.write_text(json.dumps(digests, indent=2, sort_keys=True) + '\n',
                       newline='\n')
        print(f'\nwrote {out}')

    if args.check:
        fixture = json.loads(Path(args.check).read_text())
        failures = []
        for name, digest in digests.items():
            if name not in fixture:
                failures.append(f'{name}: not in {args.check}')
            elif digest['survives'] != fixture[name]['survives']:
                moved = sorted(k for k, v in digest['survives'].items()
                               if fixture[name]['survives'].get(k) != v)
                failures.append(f'{name}: survivors moved -> {", ".join(moved)}')
        print('\n' + ('\n'.join(failures) if failures
                      else f'all {len(digests)} tracks match {args.check}'))
        raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
