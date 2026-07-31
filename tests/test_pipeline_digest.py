import json
from pathlib import Path

import pytest

from tests.conftest import ANCHOR_YOUTUBE_ID
from training.pipeline_digest import (DOOMED_BEAT_COLUMNS, DOOMED_METRIC_KEYS,
                                      RESCALED_BEAT_COLUMNS,
                                      SURVIVING_BEAT_COLUMNS,
                                      TIMING_ACCURACY_MAX_MS, check_digest,
                                      degradation_digest, digest_report,
                                      held_start_to_end, informational_digest,
                                      relations_digest, survives_digest)

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


def _demolished(report):
    """The same report with every column and metric the demolition deletes gone."""
    stripped = dict(report)
    stripped['beats'] = [{k: v for k, v in beat.items()
                          if k not in DOOMED_BEAT_COLUMNS} for beat in report['beats']]
    stripped['metrics'] = {k: v for k, v in report['metrics'].items()
                           if k not in DOOMED_METRIC_KEYS}
    return stripped


def _streams(sound=(), os2l=(), midi=(), overlay=()):
    return {'sound': list(sound), 'os2l': list(os2l), 'midi': list(midi),
            'overlay': list(overlay)}


def _os2l(pos, **over):
    row = {'label': 'beat', 'time': float(pos), 'change': False, 'pos': pos,
           'bpm': 128.0, 'strength': 0.5}
    row.update(over)
    return row


def _overlay(t, effect='LIGHT_BAR_24'):
    return {'label': 'overlay_update', 'time': t, 'effect': effect}


def _pool_channel(intent, index=0):
    from lib.engine.effect_definitions import INTENT_EFFECTS, LightIntent

    return INTENT_EFFECTS[LightIntent(intent)][index].midi_channel.name


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


# --------------------------------------------------------------------------- #
# The MIDI show is a relation, and the transcript that proves it is evidence
# --------------------------------------------------------------------------- #


def test_the_midi_transcript_is_not_gated_anywhere_in_survives():
    """The finding this split exists to close.

    Command counts, labels and the ordering hash are a function of the
    classifier the demolition retires; equality-gating them would fail Task 4 on
    exactly the fields it is required to move.
    """
    midi = [{'label': 'set_autoloop', 'time': 1.0, 'channel': 'A'}]
    d = digest_report(_report([]), _streams(midi=midi))
    assert 'midi' not in d['survives']
    assert json.dumps(d['survives'], sort_keys=True).find('ordering_hash') == -1
    assert set(d['informational']['midi']) == {'commands', 'labels', 'ordering_hash'}


def test_a_show_that_changed_nothing_but_its_intents_keeps_every_survivor():
    """The experiment the review ran: hold one intent, and only the show moves."""
    beats = [_beat(1.0), _beat(2.0)]
    streams = _streams(sound=[{'t': 0.3, 'playing': True}], os2l=[_os2l(1)],
                       overlay=[_overlay(1.0)])
    busy_effects = [{'t': 1.0, 'channel': _pool_channel('groove'),
                     'type': 'AUTOLOOP', 'end': 2.0},
                    {'t': 2.0, 'channel': _pool_channel('drop'),
                     'type': 'AUTOLOOP', 'end': 3.0}]
    busy_intents = [{'t': 1.0, 'intent': 'groove', 'end': 2.0},
                    {'t': 2.0, 'intent': 'drop', 'end': 3.0}]
    busy_midi = [{'label': 'set_autoloop', 'time': 1.0,
                  'channel': _pool_channel('groove')},
                 {'label': 'set_autoloop', 'time': 2.0,
                  'channel': _pool_channel('drop')}]

    busy = digest_report(_report(beats, effects=busy_effects, intents=busy_intents),
                         {**streams, 'midi': busy_midi})
    held = digest_report(
        _report(beats,
                effects=[{'t': 1.0, 'channel': _pool_channel('groove'),
                          'type': 'AUTOLOOP', 'end': 3.0}],
                intents=[{'t': 1.0, 'intent': 'groove', 'end': 3.0}]),
        {**streams, 'midi': busy_midi[:1]})

    assert busy['survives'] == held['survives']
    assert busy['relations'] == held['relations']
    assert busy['informational'] != held['informational']


def test_midi_autoloops_match_the_report_effects_one_for_one_in_order():
    midi = [{'label': 'set_autoloop', 'time': 1.0, 'channel': 'A'},
            {'label': 'set_color_override', 'time': 1.0, 'channel': 'C1'},
            {'label': 'set_special_effect', 'time': 2.0, 'channel': 'STROBE'}]
    effects = [{'t': 1.0, 'channel': 'A', 'type': 'AUTOLOOP', 'end': 2.0},
               {'t': 2.0, 'channel': 'STROBE', 'type': 'SPECIAL_EFFECT', 'end': 3.0}]
    d = digest_report(_report([], effects=effects), _streams(midi=midi))
    assert d['relations']['midi_matches_the_report_effects'] is True
    assert d['relations']['midi_arrives_in_enqueue_order'] is True
    assert d['informational']['midi']['commands'] == 3


def test_a_midi_command_the_report_never_saw_breaks_the_match():
    midi = [{'label': 'set_autoloop', 'time': 1.0, 'channel': 'A'},
            {'label': 'set_autoloop', 'time': 2.0, 'channel': 'B'}]
    effects = [{'t': 1.0, 'channel': 'A', 'type': 'AUTOLOOP', 'end': 2.0}]
    d = relations_digest(_report([], effects=effects), _streams(midi=midi))
    assert d['midi_matches_the_report_effects'] is False


def test_midi_time_running_backwards_is_caught():
    midi = [{'label': 'set_autoloop', 'time': 2.0, 'channel': 'A'},
            {'label': 'set_autoloop', 'time': 1.0, 'channel': 'B'}]
    d = relations_digest(_report([]), _streams(midi=midi))
    assert d['midi_arrives_in_enqueue_order'] is False


def test_every_lit_channel_comes_from_the_pool_its_intent_names():
    intents = [{'t': 1.0, 'intent': 'groove', 'end': 5.0}]
    effects = [{'t': 1.0, 'channel': _pool_channel('groove'), 'type': 'AUTOLOOP',
                'end': 5.0}]
    d = relations_digest(_report([], effects=effects, intents=intents))
    assert d['midi_channels_come_from_the_intents_pool'] is True


def test_a_channel_from_another_intents_pool_is_caught():
    intents = [{'t': 1.0, 'intent': 'groove', 'end': 5.0}]
    effects = [{'t': 1.0, 'channel': _pool_channel('drop'), 'type': 'AUTOLOOP',
                'end': 5.0}]
    d = relations_digest(_report([], effects=effects, intents=intents))
    assert d['midi_channels_come_from_the_intents_pool'] is False


def test_an_effect_lit_before_any_intent_was_committed_is_caught():
    effects = [{'t': 1.0, 'channel': _pool_channel('groove'), 'type': 'AUTOLOOP',
                'end': 5.0}]
    d = relations_digest(_report([], effects=effects))
    assert d['midi_channels_come_from_the_intents_pool'] is False


# --------------------------------------------------------------------------- #
# D6: the overlay light bar has a supplier, and it is the one thing that says so
# --------------------------------------------------------------------------- #


def test_the_overlay_light_bar_is_pinned_as_a_liveness_relation():
    d = relations_digest(_report([]), _streams(overlay=[_overlay(1.0), _overlay(2.0)]))
    assert d['overlay_light_bar_fires'] is True
    assert d['overlay_arrives_in_enqueue_order'] is True


def test_a_dark_overlay_light_bar_is_caught():
    """D6's failure mode: the onset chain goes, the re-source is forgotten."""
    d = relations_digest(_report([]), _streams(overlay=[]))
    assert d['overlay_light_bar_fires'] is False


def test_an_overlay_that_is_not_the_light_bar_does_not_count_as_the_light_bar():
    d = relations_digest(_report([]), _streams(overlay=[_overlay(1.0, 'STROBE')]))
    assert d['overlay_light_bar_fires'] is False


def test_overlay_time_running_backwards_is_caught():
    d = relations_digest(_report([]), _streams(overlay=[_overlay(2.0), _overlay(1.0)]))
    assert d['overlay_arrives_in_enqueue_order'] is False


# --------------------------------------------------------------------------- #
# Queue accuracy: the error is the survivor, the target it is measured from is not
# --------------------------------------------------------------------------- #


def test_timing_accuracy_is_the_queue_error_not_the_command_count():
    log = [{'label': 'beat', 'actual_delta_sec': 2.5040, 'target_delta_sec': 2.5},
           {'label': 'intent', 'actual_delta_sec': 2.4990, 'target_delta_sec': 2.5}]
    d = survives_digest(_report([], timing_log=log))
    assert d['timing_accuracy']['max_error_ms'] == pytest.approx(4.0, abs=1e-6)
    assert d['timing_accuracy']['max_error_ms'] < TIMING_ACCURACY_MAX_MS
    assert 'commands' not in d['timing_accuracy']


def test_the_queues_target_is_recorded_but_never_gated():
    """The look-ahead moves at the rewire; a survivor that is doomed is not one."""
    log = [{'label': 'beat', 'actual_delta_sec': 2.501, 'target_delta_sec': 2.5}]
    d = digest_report(_report([], timing_log=log))
    assert 'target_delta_sec' not in d['survives']['timing_accuracy']
    assert d['informational']['timing']['target_delta_sec'] == [2.5]


def test_a_moved_look_ahead_leaves_every_survivor_where_it_was():
    at_2_5 = [{'label': 'beat', 'actual_delta_sec': 2.501, 'target_delta_sec': 2.5}]
    at_16 = [{'label': 'beat', 'actual_delta_sec': 16.001, 'target_delta_sec': 16.0}]
    assert (survives_digest(_report([], timing_log=at_2_5))
            == survives_digest(_report([], timing_log=at_16)))


def test_a_queue_that_missed_its_own_target_breaks_the_relation():
    log = [{'label': 'beat', 'actual_delta_sec': 2.6, 'target_delta_sec': 2.5}]
    d = relations_digest(_report([], timing_log=log))
    assert d['queue_error_within_tolerance'] is False


# --------------------------------------------------------------------------- #
# The instrument outlives what it measures
# --------------------------------------------------------------------------- #


def test_the_digest_module_imports_nothing_the_demolition_deletes():
    """It has to run on both sides of Task 4, so it may not name a doomed symbol."""
    import ast

    import training.pipeline_digest as digest

    tree = ast.parse(Path(digest.__file__).read_text(encoding='utf-8'))
    top_level = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom))]
    modules = set()
    for node in top_level:
        if isinstance(node, ast.Import):
            modules |= {alias.name.split('.')[0] for alias in node.names}
        elif node.module:
            modules.add(node.module.split('.')[0])
    assert 'lib' not in modules and 'simulate' not in modules


def test_the_post_demolition_report_shape_still_computes_every_gated_field():
    """Doomed columns and metrics gone, and the instrument still measures."""
    report = _demolished(_report([_beat(1.0), _beat(2.0)],
                                 timing_log=[{'label': 'beat',
                                              'actual_delta_sec': 2.501,
                                              'target_delta_sec': 2.5}]))
    streams = _streams(sound=[{'t': 0.3, 'playing': True}], os2l=[_os2l(1)],
                       overlay=[_overlay(1.0)])
    intact = digest_report(_report([_beat(1.0), _beat(2.0)]), streams)
    d = digest_report(report, streams)

    assert set(d['survives']) == set(intact['survives'])
    assert set(d['relations']) == set(intact['relations'])
    assert d['survives']['beat_times_hash'] == intact['survives']['beat_times_hash']
    assert d['survives']['beat_columns_hash'] == intact['survives']['beat_columns_hash']
    assert d['survives']['schema'] == intact['survives']['schema']
    assert all(value is True for value in d['relations'].values())


def test_the_doomed_density_evidence_is_dropped_rather_than_faked():
    report = _demolished(_report([_beat(1.0)]))
    informational = informational_digest(report)
    assert 'onset_density_mean' not in informational
    assert 'onset_density_median' not in informational
    assert informational['intent_changes_count'] == 0


# --------------------------------------------------------------------------- #
# The degradation reading
# --------------------------------------------------------------------------- #


def test_a_held_show_is_the_degradation_state():
    report = _report([_beat(1.0), _beat(2.0)],
                     intents=[{'t': 3.5, 'intent': 'breakdown', 'end': 10.0}])
    report['metrics']['effect_changes_count'] = 1
    d = degradation_digest(report, _streams(sound=[{'t': 0.3, 'playing': True}]))
    assert held_start_to_end(d)
    assert d['intents_held'] == ['breakdown']
    assert d['beats_detected'] == 2
    assert d['sound_events'] == [{'t': 0.3, 'playing': True}]


def test_the_surviving_silence_timer_does_not_disqualify_the_degradation_state():
    """The plan's literal D13 state: beats, silence, one held intent.

    ATMOSPHERIC is not classified -- it fires from the beat-absence timer, which
    the demolition keeps.  A predicate that counted it would demand a stage that
    stays dark from first beat to last, which the plan never asks for.
    """
    report = _report([_beat(1.0)],
                     intents=[{'t': 3.5, 'intent': 'groove', 'end': 300.0},
                              {'t': 300.0, 'intent': 'atmospheric', 'end': 310.0}])
    report['metrics']['effect_changes_count'] = 2
    d = degradation_digest(report)
    assert held_start_to_end(d)
    assert d['intent_blocks'] == 2
    assert d['classified_blocks'] == 1
    assert d['atmospheric_blocks'] == 1


def test_a_show_that_switched_intent_is_not_the_degradation_state():
    report = _report([_beat(1.0)],
                     intents=[{'t': 3.5, 'intent': 'breakdown', 'end': 6.0},
                              {'t': 6.0, 'intent': 'drop', 'end': 10.0}])
    report['metrics']['effect_changes_count'] = 1
    assert not held_start_to_end(degradation_digest(report))


def test_a_show_that_re_rolled_its_effect_is_not_the_degradation_state():
    report = _report([_beat(1.0)],
                     intents=[{'t': 3.5, 'intent': 'drop', 'end': 10.0}])
    report['metrics']['effect_changes_count'] = 4
    assert not held_start_to_end(degradation_digest(report))


def test_a_mid_show_shed_is_not_this_predicate():
    """A shed that happens mid-show carries every intent committed before it.

    D11 holds the CURRENT intent, so a genuine NN_SHED report is a busy show
    followed by a held one, and reads False here.  Saying so is the point: this
    predicate is the whole-run reading, and reusing it as the shed check would
    have made a working shed look like a broken one.
    """
    report = _report([_beat(1.0)],
                     intents=[{'t': 1.0, 'intent': 'breakdown', 'end': 20.0},
                              {'t': 20.0, 'intent': 'drop', 'end': 40.0},
                              {'t': 40.0, 'intent': 'drop', 'end': 300.0}])
    report['metrics']['effect_changes_count'] = 3
    assert not held_start_to_end(degradation_digest(report))


def test_a_silent_show_that_never_committed_is_the_degradation_state():
    assert held_start_to_end(degradation_digest(_report([])))


def test_the_degradation_digest_carries_beats_silence_and_the_held_intent_only():
    d = degradation_digest(_report([_beat(1.0)]))
    assert set(d) == {'beats_detected', 'beat_times_hash', 'sound_events',
                      'intent_blocks', 'classified_blocks', 'atmospheric_blocks',
                      'intents_held', 'effect_changes'}


def test_speed_is_reported_but_never_baselined():
    d = digest_report(_report([_beat(1.0)]), wall_elapsed=2.0)
    assert d['speed']['realtime_factor'] == 5.0
    for track in json.loads(BASELINE.read_text()).values():
        assert 'speed' not in track


def test_digest_survives_a_report_with_no_beats():
    d = digest_report(_report([]))
    assert d['survives']['beats_detected'] == 0
    assert d['survives']['schema']['beat_keys'] == []
    assert held_start_to_end(d['degradation'])


def test_density_aggregates_are_evidence_and_ignore_the_unmeasured_sentinel():
    from lib.analyser.music_analyser import DENSITY_UNKNOWN

    measured = [_beat(float(i), onset_density=6.0) for i in range(4)]
    shed = [_beat(float(i + 4), onset_density=DENSITY_UNKNOWN) for i in range(4)]
    d = digest_report(_report(measured + shed))
    assert d['informational']['onset_density_median'] == 6.0


# --------------------------------------------------------------------------- #
# The committed fixture, and the comparison the CLI and the suite share
# --------------------------------------------------------------------------- #


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


def test_the_committed_fixture_holds_every_relation_true():
    for name, entry in json.loads(BASELINE.read_text()).items():
        broken = sorted(k for k, v in entry['relations'].items() if v is not True)
        assert not broken, f'{name}: {broken}'


def test_the_committed_fixture_pins_a_sound_stop_instant_somewhere():
    """D5 changes the RMS gate; a fixture with no stop event cannot see it move."""
    stops = {name: [e for e in entry['survives']['sound_events'] if not e['playing']]
             for name, entry in json.loads(BASELINE.read_text()).items()}
    assert any(stops.values()), (
        f'no committed track stops making sound: {stops} -- D5 has no stop '
        f'instant to match against')


def test_check_digest_reports_a_moved_survivor_and_a_broken_relation():
    fixture = {'t.mp3': {'survives': {'beats_detected': 10},
                         'relations': {'overlay_light_bar_fires': True}}}
    moved = {'survives': {'beats_detected': 11},
             'relations': {'overlay_light_bar_fires': False}}
    failures = check_digest('t.mp3', moved, fixture)
    assert any('beats_detected' in f for f in failures)
    assert any('overlay_light_bar_fires' in f for f in failures)


def test_check_digest_is_silent_when_the_survivors_hold():
    fixture = {'t.mp3': {'survives': {'beats_detected': 10},
                         'relations': {'overlay_light_bar_fires': True}}}
    same = {'survives': {'beats_detected': 10},
            'relations': {'overlay_light_bar_fires': True}}
    assert check_digest('t.mp3', same, fixture) == []


def test_check_digest_names_a_track_the_fixture_does_not_carry():
    assert check_digest('missing.mp3', {'survives': {}, 'relations': {}}, {})


# Module-level cache rather than a fixture, so each test keeps its own event loop.
_digest_cache: dict = {}


async def _anchor_digest(anchor_mp3) -> dict:
    if not _digest_cache:
        from training.pipeline_digest import digest_track
        _digest_cache.update(await digest_track(anchor_mp3))
    return _digest_cache


@pytest.mark.integration
async def test_the_anchor_track_still_produces_its_committed_survivors(anchor_mp3):
    fixture = json.loads(BASELINE.read_text())
    actual = await _anchor_digest(anchor_mp3)
    assert check_digest(SAMPLE_SONG_NAME, actual, fixture) == []
    assert actual['survives'] == fixture[SAMPLE_SONG_NAME]['survives']


@pytest.mark.integration
async def test_the_anchor_tracks_overlay_light_bar_is_actually_fed(anchor_mp3):
    """D6's guard, on real audio: a supplier that stopped supplying is invisible."""
    actual = await _anchor_digest(anchor_mp3)
    assert actual['relations']['overlay_light_bar_fires'] is True
    assert actual['relations']['overlay_arrives_in_enqueue_order'] is True
    assert actual['informational']['overlay']['light_bar_updates'] > 0


def _non_anchor_fixture_tracks() -> list:
    import run_eval_set
    from select_eval_set import EVAL_SET_FILE, load_eval_set

    data_dir = run_eval_set.corpus_dir()
    fixture = json.loads(BASELINE.read_text())
    return [(name, str(run_eval_set.audio_path(data_dir, Path(name).stem)))
            for name in sorted(fixture)
            if Path(name).stem != ANCHOR_YOUTUBE_ID]


@pytest.mark.integration
@pytest.mark.parametrize('name, mp3', _non_anchor_fixture_tracks())
async def test_every_fixture_track_still_produces_its_committed_survivors(name, mp3):
    """The `--check` command, run by the suite rather than by a human.

    The anchor never stops making sound, so it cannot see a D5 threshold that
    matches sound-START instants and mis-times the stops.  The stop instant that
    exists lives on a non-anchor track, and until now the only thing that read
    it was somebody remembering to run the CLI over all three.
    """
    from training.pipeline_digest import digest_track

    if not Path(mp3).exists():
        pytest.fail(f'committed fixture audio is missing: {mp3}')
    actual = await digest_track(mp3)
    assert check_digest(name, actual, json.loads(BASELINE.read_text())) == []


@pytest.mark.integration
async def test_the_rule_engine_is_not_yet_the_degradation_state(anchor_mp3):
    """The other half of the D13 contract, asserted from the wrong side.

    A predicate that only ever returns True proves nothing.  Master's show
    switches intent dozens of times, so it must read False here; the demolition
    is what flips it, and this test is what says the flip was real.
    """
    actual = await _anchor_digest(anchor_mp3)
    assert not held_start_to_end(actual['degradation'])
    assert actual['degradation']['classified_blocks'] > 1
