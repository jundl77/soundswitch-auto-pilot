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
    assert not set(RHYTHM_COLUMNS) & set(SPECTRAL_COLUMNS)


def test_every_beat_column_is_classified():
    from lib.engine.event_buffer import EventBuffer

    buffer = EventBuffer(window_sec=float('inf'))
    buffer.start()
    buffer.add_beat(bpm=128.0, onset_density=4.0, change=False)
    columns = set(buffer.to_report()['beats'][0].keys())
    assert columns == set(RHYTHM_COLUMNS) | set(SPECTRAL_COLUMNS), (
        f'unclassified beat columns: '
        f'{columns ^ (set(RHYTHM_COLUMNS) | set(SPECTRAL_COLUMNS))}')
    assert set(_beat(0.0).keys()) == columns


def test_a_moved_beat_time_does_not_touch_the_beat_sampled_columns():
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
    a = digest_report(_report([_beat(1.0), _beat(2.0)]))
    b = digest_report(_report([_beat(1.0), _beat(1.5), _beat(2.0)]))
    assert a['at_beats'] != b['at_beats']


def test_schema_records_the_report_contract():
    d = digest_report(_report([_beat(1.0)]))
    assert 'metrics' in d['schema']['report_keys']
    assert 'look_ahead_sec' in d['schema']['metric_keys']
    assert d['schema']['beat_keys'] == sorted(_beat(0.0).keys())


def test_speed_is_reported_but_never_baselined():
    d = digest_report(_report([_beat(1.0)]), wall_elapsed=2.0)
    assert d['speed']['realtime_factor'] == 5.0
    assert 'speed' not in json.loads(BASELINE.read_text())[SAMPLE_SONG_NAME]


def test_digest_survives_a_report_with_no_beats():
    d = digest_report(_report([]))
    assert d['rhythm']['beats_detected'] == 0
    assert d['schema']['beat_keys'] == []


@pytest.mark.integration
async def test_bundled_track_keeps_the_committed_schema_and_filterbank():
    from training.pipeline_digest import digest_track
    baseline = json.loads(BASELINE.read_text())[SAMPLE_SONG_NAME]
    sample = Path(__file__).parent.parent / 'samples' / SAMPLE_SONG_NAME
    actual = await digest_track(str(sample))
    assert actual['schema'] == baseline['schema']
    assert actual['filterbank'] == baseline['filterbank']


@pytest.mark.integration
def test_the_filterbank_anchor_notices_a_changed_filterbank():
    from training.pipeline_digest import filterbank_fingerprint
    sample = str(Path(__file__).parent.parent / 'samples' / SAMPLE_SONG_NAME)
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
    from lib.analyser.music_analyser import DENSITY_UNKNOWN

    measured = [_beat(float(i), onset_density=6.0) for i in range(4)]
    shed = [_beat(float(i + 4), onset_density=DENSITY_UNKNOWN) for i in range(4)]
    d = digest_report(_report(measured + shed))
    assert d['rhythm']['onset_density_median'] == 6.0
    assert d['rhythm']['onset_density_median'] > 0
