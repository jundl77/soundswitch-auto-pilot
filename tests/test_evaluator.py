from simulate.evaluator import evaluate_against_labels, _sweep_score


def _make_intents(*pairs):
    """Build intents list: (time, intent) pairs → [{'t', 'intent', 'end'}, ...]."""
    entries = [{'t': t, 'intent': name} for t, name in pairs]
    for i in range(len(entries) - 1):
        entries[i]['end'] = entries[i + 1]['t']
    if entries:
        entries[-1]['end'] = entries[-1]['t'] + 30.0
    return entries


def _make_labels(*triples):
    """Build labels list: (start, end, intent) → [{'start', 'end', 'intent'}, ...]."""
    return [{'start': s, 'end': e, 'intent': n} for s, e, n in triples]


def test_perfect_match_zero_offset():
    """Predicted transition exactly at GT boundary → offset = 0 ms."""
    intents = _make_intents((0.0, 'atmospheric'), (10.0, 'breakdown'))
    labels  = _make_labels((0.0, 10.0, 'atmospheric'), (10.0, 40.0, 'breakdown'))
    result  = evaluate_against_labels(intents, labels, look_ahead_sec=0.0)
    assert result['total_boundaries'] == 1
    assert result['missed_count'] == 0
    assert result['false_boundary_count'] == 0
    assert abs(result['boundaries'][0]['offset_ms']) < 1


def test_late_transition_positive_offset():
    """Predicted 1s after GT → offset_ms = +1000."""
    intents = _make_intents((0.0, 'atmospheric'), (11.0, 'breakdown'))
    labels  = _make_labels((0.0, 10.0, 'atmospheric'), (10.0, 40.0, 'breakdown'))
    result  = evaluate_against_labels(intents, labels, look_ahead_sec=0.0)
    assert abs(result['boundaries'][0]['offset_ms'] - 1000.0) < 1


def test_early_transition_negative_offset():
    """Predicted 1s before GT → offset_ms = -1000, passed=False."""
    intents = _make_intents((0.0, 'atmospheric'), (9.0, 'breakdown'))
    labels  = _make_labels((0.0, 10.0, 'atmospheric'), (10.0, 40.0, 'breakdown'))
    result  = evaluate_against_labels(intents, labels, look_ahead_sec=0.0)
    assert result['missed_count'] == 0
    assert abs(result['boundaries'][0]['offset_ms'] - (-1000.0)) < 1
    assert result['boundaries'][0]['passed'] is False


def test_missed_boundary_beyond_search_window():
    """No predicted transition within ±2s → missed."""
    intents = _make_intents((0.0, 'atmospheric'), (20.0, 'breakdown'))
    labels  = _make_labels((0.0, 10.0, 'atmospheric'), (10.0, 40.0, 'breakdown'))
    result  = evaluate_against_labels(intents, labels, look_ahead_sec=0.0)
    assert result['missed_count'] == 1
    assert result['boundaries'][0]['missed'] is True


def test_false_boundary_counted():
    """Predicted transitions not matching any GT boundary → false boundary."""
    intents = _make_intents(
        (0.0, 'atmospheric'), (10.0, 'breakdown'), (15.0, 'groove')
    )
    labels = _make_labels((0.0, 10.0, 'atmospheric'), (10.0, 40.0, 'breakdown'))
    result = evaluate_against_labels(intents, labels, look_ahead_sec=0.0)
    assert result['false_boundary_count'] == 1


def test_look_ahead_offset_correction():
    """When look_ahead_sec=2.5, intent at t=12.5 matches GT boundary at t=10.0."""
    intents = _make_intents((0.0, 'atmospheric'), (12.5, 'breakdown'))
    labels  = _make_labels((0.0, 10.0, 'atmospheric'), (10.0, 40.0, 'breakdown'))
    result  = evaluate_against_labels(intents, labels, look_ahead_sec=2.5)
    assert result['missed_count'] == 0
    assert abs(result['boundaries'][0]['offset_ms']) < 1


def test_sweep_score_zero_for_perfect():
    ta = {
        'mean_offset_ms': 0.0, 'max_offset_ms': 0.0,
        'missed_count': 0, 'false_boundary_count': 0,
    }
    assert _sweep_score(ta) == 0.0


def test_sweep_score_miss_penalty():
    ta = {
        'mean_offset_ms': 0.0, 'max_offset_ms': 0.0,
        'missed_count': 1, 'false_boundary_count': 0,
    }
    assert _sweep_score(ta) == 5000.0


def test_sweep_score_false_boundary_penalty():
    ta = {
        'mean_offset_ms': 0.0, 'max_offset_ms': 0.0,
        'missed_count': 0, 'false_boundary_count': 2,
    }
    assert _sweep_score(ta) == 1000.0


def test_sweep_score_weighted_combo():
    ta = {
        'mean_offset_ms': 100.0, 'max_offset_ms': 200.0,
        'missed_count': 0, 'false_boundary_count': 0,
    }
    # 0.6 * 100 + 0.4 * 200 = 60 + 80 = 140
    assert abs(_sweep_score(ta) - 140.0) < 0.01


from simulate.evaluator import _replay_feature_log


def _make_feature_log(beats: list[tuple]) -> list[dict]:
    """Build feature log: list of (audio_time_sec, bpm, onset_density, kick_strength) tuples.
    density_trend=1.0, centroid_trend=1.0, sub_bass_ratio=0.3 as defaults.
    """
    return [
        {
            'audio_time_sec': t,
            'bpm':            bpm,
            'onset_density':  density,
            'kick_strength':  kick,
            'density_trend':  1.0,
            'centroid_trend': 1.0,
            'sub_bass_ratio': 0.3,
        }
        for t, bpm, density, kick in beats
    ]


_BASE_CFG = {
    '_BREAKDOWN_MAX_DENSITY_ENTER': 3.0,
    '_BREAKDOWN_MAX_DENSITY_EXIT':  3.5,
    '_BUILDUP_MIN_TREND':           1.3,
    '_DROP_MIN_DENSITY_ENTER':      8.5,
    '_DROP_MIN_DENSITY_EXIT':       7.0,
    '_KICK_PRESENCE_THRESHOLD':     1.3,
    '_CENTROID_BUILDUP_TREND':      1.1,
    '_VOTE_BUFFER_SIZE':            1,   # single vote for fast convergence in tests
    '_MIN_DWELL_BEATS':             1,
}


def test_replay_detects_drop_transition():
    """High-density beats after a sparse section → DROP detected."""
    sparse = [(float(i) * 0.5, 128.0, 2.0, 1.2) for i in range(10)]
    dense  = [(5.0 + float(i) * 0.5, 128.0, 9.5, 2.5) for i in range(10)]
    log    = _make_feature_log(sparse + dense)
    result = _replay_feature_log(log, _BASE_CFG, look_ahead_sec=0.5)
    intents = [r['intent'] for r in result]
    assert 'drop' in intents


def test_replay_starts_atmospheric():
    """Replay always begins with atmospheric (initial state)."""
    log    = _make_feature_log([(float(i) * 0.5, 128.0, 2.0, 1.2) for i in range(5)])
    result = _replay_feature_log(log, _BASE_CFG, look_ahead_sec=0.5)
    assert result[0]['intent'] == 'atmospheric'


def test_replay_blocks_invalid_transition():
    """ATMOSPHERIC → DROP is an invalid transition and must be blocked."""
    log = _make_feature_log([(0.0, 128.0, 9.5, 2.5)])
    cfg = {**_BASE_CFG, '_VOTE_BUFFER_SIZE': 1, '_MIN_DWELL_BEATS': 0}
    result = _replay_feature_log(log, cfg, look_ahead_sec=0.5)
    intents = [r['intent'] for r in result]
    assert 'drop' not in intents
