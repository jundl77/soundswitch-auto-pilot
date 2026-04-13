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
