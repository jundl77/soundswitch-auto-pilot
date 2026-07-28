"""The digest's own contract: what it separates, and that the separation holds.

The migration is judged by comparing digests, so a digest that quietly mixed
rhythm columns into the spectral hash would let a filterbank regression hide
behind an expected beat change. These tests pin the split itself.
"""

import json
from pathlib import Path

import pytest

from training.pipeline_digest import (RHYTHM_COLUMNS, SPECTRAL_COLUMNS,
                                      digest_report)

BASELINE = Path(__file__).parent / 'fixtures' / 'pipeline_digest_baseline.json'
SAMPLE_SONG_NAME = 'generate_eric_prydz_192k.mp3'


def _beat(t, **over):
    row = {'t': t, 'bpm': 128.0, 'onset_density': 4.0, 'strength': 0.4,
           'change': False, 'kick_strength': 2.5, 'centroid_trend': 1.0,
           'sub_bass_ratio': 0.3, 'rms': 0.1}
    row.update(over)
    return row


def _report(beats):
    return {
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


def test_the_two_column_sets_do_not_overlap():
    """A column in both sets would be pinned and licensed to move at once."""
    assert not set(RHYTHM_COLUMNS) & set(SPECTRAL_COLUMNS)


def test_every_beat_column_is_classified():
    """A new beat column must be deliberately placed on one side of the split,
    not silently fall outside both and escape the digest entirely."""
    columns = set(_beat(0.0).keys())
    assert columns == set(RHYTHM_COLUMNS) | set(SPECTRAL_COLUMNS)


def test_a_moved_beat_time_does_not_touch_the_spectral_hash():
    """The migration moves beat times by design; that must not read as a
    filterbank regression."""
    a = digest_report(_report([_beat(1.0), _beat(2.0)]))
    b = digest_report(_report([_beat(1.05), _beat(2.05)]))
    assert a['rhythm']['beat_times_hash'] != b['rhythm']['beat_times_hash']
    assert a['spectral'] == b['spectral']


def test_a_moved_filterbank_column_does_not_hide_in_the_rhythm_hash():
    a = digest_report(_report([_beat(1.0)]))
    b = digest_report(_report([_beat(1.0, kick_strength=9.9)]))
    assert a['spectral']['columns_hash'] != b['spectral']['columns_hash']
    assert a['rhythm'] == b['rhythm']


def test_schema_records_the_report_contract():
    d = digest_report(_report([_beat(1.0)]))
    assert 'metrics' in d['schema']['report_keys']
    assert 'look_ahead_sec' in d['schema']['metric_keys']
    assert d['schema']['beat_keys'] == sorted(_beat(0.0).keys())


def test_speed_is_reported_but_never_baselined():
    """Wall-clock speed is machine-dependent — it may be measured, but a fixture
    that pinned it would fail on someone else's laptop for no reason."""
    d = digest_report(_report([_beat(1.0)]), wall_elapsed=2.0)
    assert d['speed']['realtime_factor'] == 5.0
    assert 'speed' not in json.loads(BASELINE.read_text())[SAMPLE_SONG_NAME]


def test_digest_survives_a_report_with_no_beats():
    d = digest_report(_report([]))
    assert d['rhythm']['beats_detected'] == 0
    assert d['schema']['beat_keys'] == []


@pytest.mark.integration
async def test_bundled_track_keeps_the_committed_report_schema():
    """The report is a published contract — the visualizer, the evaluator and
    the inspector all read it. Changing the beat source may move every number
    in it; it may not move a key.

    Note what this deliberately does NOT assert. The digest's `spectral`
    section is the filterbank's output *sampled at beat instants*, so it moves
    whenever the beat grid moves — which is the whole point of this migration.
    An equality assertion there would fail for the expected reason and prove
    nothing about the filterbank. The filterbank's own invariance is pinned
    directly, and beat-independently, in test_music_analyser.py.
    """
    from training.pipeline_digest import digest_track
    baseline = json.loads(BASELINE.read_text())[SAMPLE_SONG_NAME]
    sample = Path(__file__).parent.parent / 'samples' / SAMPLE_SONG_NAME
    actual = await digest_track(str(sample))
    assert actual['schema'] == baseline['schema']
