"""Compact, diffable digest of one fast-sim run — the NN integration's golden fixture.

Cut on the pipeline as it stands at the branch point, so the demolition can prove
what it preserved rather than assert it.  Four blocks, and they are read
differently:

``survives``      values the integration is not allowed to move — the beat stream,
                  the sound-start/stop instants, the OS2L beat wire, the queue's
                  own error, and the report keys that outlive the rule engine.
                  Compared for equality.
``relations``     properties, not values.  Everything the MIDI show is made of is a
                  function of the classifier the demolition retires, so its
                  transcript cannot be a survivor; what must still hold is that
                  commands arrive in the order they were enqueued, that every lit
                  channel comes from the pool its intent names, that the wire and
                  the report agree, that the queue meets its own target, and that
                  the overlay light bar is still being fed.  Every entry is a
                  predicate and every one must be True.
``informational`` the MIDI transcript itself and everything else the classifier
                  decides.  Recorded and diffed, never gated.
``degradation``   the projection onto what the branch's intermediate state may
                  produce: beats, the surviving silence timer, one held intent and
                  nothing else.  ``held_start_to_end`` is the predicate, and it is
                  asserted from both sides — False here, True after the demolition.

The aubio filterbank fingerprint that this file used to gate on is gone: the bank
is deleted by the integration, so a fixed-grid fingerprint of it can only fail for
the reason it was built to rule out.

Nothing under ``lib/`` or ``simulate/`` is imported at module scope, and no doomed
column or metric is read without checking that it is there.  The instrument has to
run on both sides of the demolition, or the comparison it exists to make cannot be
made in the commit that needs it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

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
    """Everything the NN integration must leave exactly where it found it."""
    beats = report['beats']
    metrics = report['metrics']
    streams = _normalise_streams(streams)
    errors_ms = _queue_errors_ms(report)
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
            'max_error_ms': round(max(errors_ms), 4) if errors_ms else 0.0,
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
    """Every lit channel is one the intent current at that instant offers.

    Survives the demolition where a transcript cannot: the classifier decides
    which intent and the controller decides which channel within it, but the
    containment is structural.  An effect lit with no intent committed is a
    violation, not a pass -- that is the stage moving on nobody's decision.
    """
    from lib.engine.effect_definitions import INTENT_EFFECTS

    pools = {intent.value: {effect.midi_channel.name for effect in effects}
             for intent, effects in INTENT_EFFECTS.items()}
    return all(effect['channel'] in pools.get(_intent_at(report['intents'],
                                                         effect['t']), ())
               for effect in report['effects'])


def relations_digest(report: dict, streams: dict | None = None) -> dict:
    """The properties that hold whatever the classifier decides."""
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
    """Evidence, never a gate: this is the half the demolition is meant to move."""
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
            # Moves with the look-ahead, which the rewire raises to the decoder's
            # budget. Recorded so the two time bases stay readable, never gated.
            'target_delta_sec': sorted({round(e['target_delta_sec'], 4)
                                        for e in report['timing_log']}),
            'commands': len(report['timing_log']),
        },
    }
    return digest


def degradation_digest(report: dict, streams: dict | None = None) -> dict:
    """The report seen through the degradation contract, and nothing else."""
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
    """Whether a whole run classified at most once and then held.

    The branch's intermediate state between the demolition and the rewire: beats
    and the beat-absence silence timer keep running, so ATMOSPHERIC still fires
    and still re-rolls an effect from its own pool.  Counting those would demand
    a stage that stays dark from the first beat to the last, which is not what
    the plan describes.  So the count is of *classified* intents, and one effect
    change per committed block is what the arithmetic allows.

    This reads a whole run, and is not the mid-show shed check: a shed holds the
    intent it had, so a real degradation report is a busy show followed by a held
    one and reads False here.
    """
    return (degradation['classified_blocks'] <= 1
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
    """The whole gate, in one place, so the CLI and the suite cannot drift.

    Survivors are compared for equality; relations are held to True rather than
    to the fixture, because a relation that was already broken when the fixture
    was cut is a defect the fixture should not be able to bless.
    """
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
    """The four wires a report does not carry, read off the stub clients."""
    return {
        'sound': components['event_buffer'].snapshot()['sound_events'],
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
        # Machine-dependent: never in a fixture a test compares against.
        for d in digests.values():
            d.pop('speed', None)
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
