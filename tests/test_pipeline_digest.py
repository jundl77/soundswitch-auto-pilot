"""The digest's own contract: what it separates, and that the separation holds.

The migration is judged by comparing digests, so a digest that quietly mixed
rhythm columns into the spectral hash would let a filterbank regression hide
behind an expected beat change. These tests pin the split itself.
"""

import json
from pathlib import Path

import pytest

from tests.conftest import ANCHOR_YOUTUBE_ID
from training.pipeline_digest import (RHYTHM_COLUMNS, SPECTRAL_COLUMNS,
                                      digest_report)

BASELINE = Path(__file__).parent / 'fixtures' / 'pipeline_digest_baseline.json'

# The anchor moved off the bundled Generate mp3, which the eval set retired and
# deleted. It did NOT move to a freshly cut number: the same baseline was cut on
# master's code over three tracks, and two of them are eval-set tracks whose
# audio is committed. So the anchor keeps its pre-madmom provenance -- the whole
# point of it -- and gains a file that exists in a fresh clone. The shorter of
# the two, because this runs a whole track through the sim.
SAMPLE_SONG_NAME = f'{ANCHOR_YOUTUBE_ID}.mp3'


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
    not silently fall outside both and escape the digest entirely.

    The columns come from the REAL beat record, not from this file's synthetic
    one: checking a fixture against a fixture would let a column added to
    EventBuffer.add_beat slip past both.
    """
    from lib.engine.event_buffer import EventBuffer

    buffer = EventBuffer(window_sec=float('inf'))
    buffer.start()
    buffer.add_beat(bpm=128.0, onset_density=4.0, change=False)
    columns = set(buffer.to_report()['beats'][0].keys())
    assert columns == set(RHYTHM_COLUMNS) | set(SPECTRAL_COLUMNS), (
        f'unclassified beat columns: '
        f'{columns ^ (set(RHYTHM_COLUMNS) | set(SPECTRAL_COLUMNS))}')
    # And the local fixture must track the real one, or every other test here
    # is exercising a shape that no longer exists.
    assert set(_beat(0.0).keys()) == columns


def test_a_moved_beat_time_does_not_touch_the_beat_sampled_columns():
    """Moving WHEN a beat is reported, without moving the row it carries, must
    not read as a filterbank change."""
    a = digest_report(_report([_beat(1.0), _beat(2.0)]))
    b = digest_report(_report([_beat(1.05), _beat(2.05)]))
    assert a['rhythm']['beat_times_hash'] != b['rhythm']['beat_times_hash']
    assert a['at_beats'] == b['at_beats']


def test_a_moved_filterbank_column_does_not_hide_in_the_rhythm_hash():
    a = digest_report(_report([_beat(1.0)]))
    b = digest_report(_report([_beat(1.0, kick_strength=9.9)]))
    assert a['at_beats']['columns_hash'] != b['at_beats']['columns_hash']
    assert a['rhythm'] == b['rhythm']


def test_the_beat_sampled_columns_are_not_a_regression_gate():
    """They cannot be. A different beat GRID reads the same filterbank output
    at different instants, so these values move for the expected reason -- which
    is exactly how a real regression would hide inside them. The gate is
    `filterbank`, sampled on a fixed time grid; this section is evidence."""
    a = digest_report(_report([_beat(1.0), _beat(2.0)]))
    b = digest_report(_report([_beat(1.0), _beat(1.5), _beat(2.0)]))
    assert a['at_beats'] != b['at_beats']


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
async def test_anchor_track_keeps_the_committed_schema_and_filterbank(anchor_mp3):
    """The two things the migration is not allowed to move, against a baseline
    cut from master before any madmom code existed.

    The report is a published contract — the visualizer, the evaluator and the
    inspector all read it — so the beat source may move every number in it but
    not a key. And the filterbank anchor is sampled on a FIXED TIME GRID, so it
    is still able to fail under a completely rewritten beat grid, which is the
    one thing a beat-sampled anchor could never do.
    """
    from training.pipeline_digest import digest_track
    baseline = json.loads(BASELINE.read_text())[SAMPLE_SONG_NAME]
    actual = await digest_track(anchor_mp3)
    assert actual['schema'] == baseline['schema']
    assert actual['filterbank'] == baseline['filterbank']


@pytest.mark.integration
def test_the_filterbank_anchor_notices_a_changed_filterbank(anchor_mp3):
    """A gate nobody has seen fail is a gate nobody knows works. Perturbing the
    bank must move the anchor — under whatever beat grid."""
    from training.pipeline_digest import filterbank_fingerprint
    sample = anchor_mp3
    honest = filterbank_fingerprint(sample, seconds=10.0)

    import lib.analyser.music_analyser as ma
    original = ma.MelFilterbank.__call__

    def perturbed(self, audio_signal):
        energies = original(self, audio_signal)
        energies[0] *= 1.01
        return energies

    ma.MelFilterbank.__call__ = perturbed
    try:
        tampered = filterbank_fingerprint(sample, seconds=10.0)
    finally:
        ma.MelFilterbank.__call__ = original
    assert tampered['grid_hash'] != honest['grid_hash']


def test_density_aggregates_ignore_the_unmeasured_sentinel():
    """DENSITY_UNKNOWN is negative, so a shed stretch would drag the reported
    mean and median below zero -- a digest that reads as a catastrophic
    regression when what actually happened is that nothing was measured."""
    from lib.analyser.music_analyser import DENSITY_UNKNOWN

    measured = [_beat(float(i), onset_density=6.0) for i in range(4)]
    shed = [_beat(float(i + 4), onset_density=DENSITY_UNKNOWN) for i in range(4)]
    d = digest_report(_report(measured + shed))
    assert d['rhythm']['onset_density_median'] == 6.0
    assert d['rhythm']['onset_density_median'] > 0
