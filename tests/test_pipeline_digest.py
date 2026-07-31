import json
from pathlib import Path

import pytest

from tests.conftest import ANCHOR_YOUTUBE_ID
from training.pipeline_digest import (DOOMED_BEAT_COLUMNS, DOOMED_METRIC_KEYS,
                                      RESCALED_BEAT_COLUMNS,
                                      SURVIVING_BEAT_COLUMNS,
                                      TIMING_ACCURACY_MAX_MS, degradation_digest,
                                      digest_report, is_degradation_state,
                                      survives_digest)

BASELINE = Path(__file__).parent / 'fixtures' / 'pipeline_digest_baseline.json'

SAMPLE_SONG_NAME = f'{ANCHOR_YOUTUBE_ID}.mp3'


def _beat(t, **over):
    row = {'t': t, 'bpm': 128.0, 'onset_density': 4.0, 'strength': 0.4,
           'change': False, 'kick_strength': 2.5, 'centroid_trend': 1.0,
           'sub_bass_ratio': 0.3, 'rms': 0.1}
    row.update(over)
    return row


def _report(beats, **over):
    report = {
        'duration_sec': 10.0,
        'beats': beats,
        'effects': [],
        'intents': [],
        'timing_log': [],
        'metrics': {
            'look_ahead_sec': 2.5, 'beats_detected': len(beats), 'bpm_last': 128.0,
            'onset_density_mean': 4.0, 'timing_error_mean_ms': 0.0,
            'timing_error_max_ms': 0.0, 'unique_effects_count': 0,
            'effect_changes_count': 0, 'avg_effect_duration_sec': 0.0,
            'unique_channels': [], 'intent_changes_count': 0,
            'unique_intents_count': 0, 'intent_distribution_sec': {},
            'dominant_intent': None,
        },
    }
    report.update(over)
    return report


def _streams(sound=(), os2l=(), midi=()):
    return {'sound': list(sound), 'os2l': list(os2l), 'midi': list(midi)}


def _os2l(pos, **over):
    row = {'label': 'beat', 'time': float(pos), 'change': False, 'pos': pos,
           'bpm': 128.0, 'strength': 0.5}
    row.update(over)
    return row


def test_the_three_beat_column_sets_do_not_overlap():
    assert not set(SURVIVING_BEAT_COLUMNS) & set(RESCALED_BEAT_COLUMNS)
    assert not set(SURVIVING_BEAT_COLUMNS) & set(DOOMED_BEAT_COLUMNS)
    assert not set(RESCALED_BEAT_COLUMNS) & set(DOOMED_BEAT_COLUMNS)


def test_every_beat_column_is_classified():
    from lib.engine.event_buffer import EventBuffer

    buffer = EventBuffer(window_sec=float('inf'))
    buffer.start()
    buffer.add_beat(bpm=128.0, onset_density=4.0, change=False)
    columns = set(buffer.to_report()['beats'][0].keys())
    classified = (set(SURVIVING_BEAT_COLUMNS) | set(RESCALED_BEAT_COLUMNS)
                  | set(DOOMED_BEAT_COLUMNS))
    assert columns == classified, (
        f'unclassified beat columns: {columns ^ classified}')
    assert set(_beat(0.0).keys()) == columns


def test_a_moved_beat_time_moves_the_beat_stream_hash():
    a = survives_digest(_report([_beat(1.0), _beat(2.0)]))
    b = survives_digest(_report([_beat(1.05), _beat(2.05)]))
    assert a['beat_times_hash'] != b['beat_times_hash']


def test_a_doomed_column_cannot_move_the_survival_hash():
    a = survives_digest(_report([_beat(1.0)]))
    b = survives_digest(_report([_beat(1.0, kick_strength=9.9, onset_density=9.0,
                                       sub_bass_ratio=0.9, centroid_trend=3.0)]))
    assert a['beat_columns_hash'] == b['beat_columns_hash']


def test_a_surviving_column_does_move_the_survival_hash():
    a = survives_digest(_report([_beat(1.0)]))
    b = survives_digest(_report([_beat(1.0, rms=0.9)]))
    assert a['beat_columns_hash'] != b['beat_columns_hash']


def test_the_schema_block_names_only_keys_that_outlive_the_rule_engine():
    d = survives_digest(_report([_beat(1.0)]))
    assert set(d['schema']['beat_keys']).isdisjoint(DOOMED_BEAT_COLUMNS)
    assert set(d['schema']['metric_keys']).isdisjoint(DOOMED_METRIC_KEYS)
    assert 'look_ahead_sec' in d['schema']['metric_keys']
    assert 'metrics' in d['schema']['report_keys']


def test_sound_start_and_stop_instants_are_pinned():
    events = [{'t': 0.301, 'playing': True}, {'t': 200.9004, 'playing': False}]
    d = survives_digest(_report([]), _streams(sound=events))
    assert d['sound_events'] == [{'t': 0.301, 'playing': True},
                                 {'t': 200.9, 'playing': False}]


def test_the_os2l_beat_wire_shape_is_pinned_field_by_field():
    d = survives_digest(_report([]), _streams(os2l=[_os2l(1), _os2l(2)]))
    assert d['os2l']['beats'] == 2
    assert d['os2l']['wire_keys'] == ['bpm', 'change', 'pos', 'strength']
    assert d['os2l']['strengths'] == [0.5]
    assert d['os2l']['positions'] == [1, 2]


def test_a_moved_os2l_field_moves_the_wire_hash():
    a = survives_digest(_report([]), _streams(os2l=[_os2l(1)]))
    b = survives_digest(_report([]), _streams(os2l=[_os2l(1, bpm=140.0)]))
    assert a['os2l']['stream_hash'] != b['os2l']['stream_hash']


def test_midi_autoloops_match_the_report_effects_one_for_one_in_order():
    midi = [{'label': 'set_autoloop', 'time': 1.0, 'channel': 'A'},
            {'label': 'set_color_override', 'time': 1.0, 'channel': 'C1'},
            {'label': 'set_special_effect', 'time': 2.0, 'channel': 'STROBE'}]
    effects = [{'t': 1.0, 'channel': 'A', 'type': 'AUTOLOOP', 'end': 2.0},
               {'t': 2.0, 'channel': 'STROBE', 'type': 'SPECIAL_EFFECT', 'end': 3.0}]
    d = survives_digest(_report([], effects=effects), _streams(midi=midi))
    assert d['midi']['effects_match_report'] is True
    assert d['midi']['times_non_decreasing'] is True
    assert d['midi']['commands'] == 3


def test_a_midi_command_the_report_never_saw_breaks_the_match():
    midi = [{'label': 'set_autoloop', 'time': 1.0, 'channel': 'A'},
            {'label': 'set_autoloop', 'time': 2.0, 'channel': 'B'}]
    effects = [{'t': 1.0, 'channel': 'A', 'type': 'AUTOLOOP', 'end': 2.0}]
    d = survives_digest(_report([], effects=effects), _streams(midi=midi))
    assert d['midi']['effects_match_report'] is False


def test_midi_time_running_backwards_is_caught():
    midi = [{'label': 'set_autoloop', 'time': 2.0, 'channel': 'A'},
            {'label': 'set_autoloop', 'time': 1.0, 'channel': 'B'}]
    d = survives_digest(_report([]), _streams(midi=midi))
    assert d['midi']['times_non_decreasing'] is False


def test_timing_accuracy_is_the_queue_error_not_the_command_count():
    log = [{'label': 'beat', 'actual_delta_sec': 2.5040, 'target_delta_sec': 2.5},
           {'label': 'intent', 'actual_delta_sec': 2.4990, 'target_delta_sec': 2.5}]
    d = survives_digest(_report([], timing_log=log))
    assert d['timing_accuracy']['max_error_ms'] == pytest.approx(4.0, abs=1e-6)
    assert d['timing_accuracy']['max_error_ms'] < TIMING_ACCURACY_MAX_MS
    assert 'commands' not in d['timing_accuracy']


def test_a_held_show_is_the_degradation_state():
    report = _report([_beat(1.0), _beat(2.0)],
                     intents=[{'t': 3.5, 'intent': 'breakdown', 'end': 10.0}])
    report['metrics']['effect_changes_count'] = 1
    d = degradation_digest(report, _streams(sound=[{'t': 0.3, 'playing': True}]))
    assert is_degradation_state(d)
    assert d['intents_held'] == ['breakdown']
    assert d['beats_detected'] == 2
    assert d['sound_events'] == [{'t': 0.3, 'playing': True}]


def test_a_show_that_switched_intent_is_not_the_degradation_state():
    report = _report([_beat(1.0)],
                     intents=[{'t': 3.5, 'intent': 'breakdown', 'end': 6.0},
                              {'t': 6.0, 'intent': 'drop', 'end': 10.0}])
    report['metrics']['effect_changes_count'] = 1
    assert not is_degradation_state(degradation_digest(report))


def test_a_show_that_re_rolled_its_effect_is_not_the_degradation_state():
    report = _report([_beat(1.0)],
                     intents=[{'t': 3.5, 'intent': 'drop', 'end': 10.0}])
    report['metrics']['effect_changes_count'] = 4
    assert not is_degradation_state(degradation_digest(report))


def test_a_silent_show_that_never_committed_is_the_degradation_state():
    assert is_degradation_state(degradation_digest(_report([])))


def test_the_degradation_digest_carries_beats_silence_and_the_held_intent_only():
    d = degradation_digest(_report([_beat(1.0)]))
    assert set(d) == {'beats_detected', 'beat_times_hash', 'sound_events',
                      'intent_blocks', 'intents_held', 'effect_changes'}


def test_speed_is_reported_but_never_baselined():
    d = digest_report(_report([_beat(1.0)]), wall_elapsed=2.0)
    assert d['speed']['realtime_factor'] == 5.0
    for track in json.loads(BASELINE.read_text()).values():
        assert 'speed' not in track


def test_digest_survives_a_report_with_no_beats():
    d = digest_report(_report([]))
    assert d['survives']['beats_detected'] == 0
    assert d['survives']['schema']['beat_keys'] == []
    assert is_degradation_state(d['degradation'])


def test_density_aggregates_are_evidence_and_ignore_the_unmeasured_sentinel():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN

    measured = [_beat(float(i), onset_density=6.0) for i in range(4)]
    shed = [_beat(float(i + 4), onset_density=DENSITY_UNKNOWN) for i in range(4)]
    d = digest_report(_report(measured + shed))
    assert d['show']['onset_density_median'] == 6.0


def test_the_committed_fixture_covers_every_track_the_benchmark_subset_runs():
    import run_eval_set
    from select_eval_set import EVAL_SET_FILE, load_eval_set

    subset = run_eval_set.shortest_track_ids(load_eval_set(EVAL_SET_FILE), 3)
    fixture = json.loads(BASELINE.read_text())
    covered = {Path(name).stem for name in fixture}
    for track_id in subset:
        youtube_id = track_id.split('.')[-1]
        assert youtube_id in covered, (
            f'{youtube_id} runs in the integration suite but has no golden '
            f'fixture -- re-cut with training/pipeline_digest.py --write')


# Module-level cache rather than a fixture, so each test keeps its own event loop.
_digest_cache: dict = {}


async def _anchor_digest(anchor_mp3) -> dict:
    if not _digest_cache:
        from training.pipeline_digest import digest_track
        _digest_cache.update(await digest_track(anchor_mp3))
    return _digest_cache


@pytest.mark.integration
async def test_the_anchor_track_still_produces_its_committed_survivors(anchor_mp3):
    baseline = json.loads(BASELINE.read_text())[SAMPLE_SONG_NAME]
    actual = await _anchor_digest(anchor_mp3)
    assert actual['survives'] == baseline['survives']
    assert actual['survives']['midi']['effects_match_report'] is True
    assert actual['survives']['midi']['times_non_decreasing'] is True
    assert (actual['survives']['timing_accuracy']['max_error_ms']
            < TIMING_ACCURACY_MAX_MS)


@pytest.mark.integration
async def test_the_rule_engine_is_not_yet_the_degradation_state(anchor_mp3):
    """The other half of the D13 contract, asserted from the wrong side.

    A predicate that only ever returns True proves nothing.  Master's show
    switches intent dozens of times, so it must read False here; the demolition
    is what flips it, and this test is what says the flip was real.
    """
    actual = await _anchor_digest(anchor_mp3)
    assert not is_degradation_state(actual['degradation'])
    assert actual['degradation']['intent_blocks'] > 1
