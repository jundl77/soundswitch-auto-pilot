"""
Agentic evaluator: scores a simulation report against configurable criteria.

Usage (headless / CI):
  python auto_pilot simulate file song.mp3 --no-ui --report report.json
  # exit code 0 = PASS, 1 = FAIL

Or in Python:
  from simulate.evaluator import evaluate, print_evaluation
  result = evaluate(report)   # report from EventBuffer.to_report()
  print_evaluation(result)

The report contains:
  beats[]         — {t, bpm, onset_density, strength, change}
  intents[]       — {t, intent, end}  (timestamped intent blocks)
  effects[]       — {t, channel, type, end}
  metrics         — aggregated stats including intent_distribution_sec
"""

from collections import deque

DEFAULT_CRITERIA: dict = {
    'timing_error_mean_ms': {
        'max': 15.0, 'weight': 0.20,
        'description': 'Mean delay error (ms)',
    },
    'beat_detection_rate': {
        'min': 10.0, 'weight': 0.20,
        'description': 'Beats detected per minute',
    },
    'unique_effects_count': {
        'min': 2, 'weight': 0.10,
        'description': 'Distinct MIDI channels used',
    },
    'effect_changes_count': {
        'min': 1, 'weight': 0.10,
        'description': 'Total effect changes observed',
    },
    'intent_changes_count': {
        'min': 2, 'weight': 0.20,
        'description': 'Intent transitions detected',
    },
    'unique_intents_count': {
        'min': 2, 'weight': 0.20,
        'description': 'Distinct intents classified',
    },
}


def evaluate(report: dict, criteria: dict | None = None) -> dict:
    """
    Score a simulation report from EventBuffer.to_report().

    Returns:
        {
          'passed': bool,
          'score':  float 0.0–1.0,
          'criteria': {key: {'value', 'passed', 'score', 'spec'}},
        }
    """
    criteria = criteria or DEFAULT_CRITERIA
    m = report['metrics']
    duration_min = max(report['duration_sec'] / 60.0, 1e-9)

    derived: dict[str, float] = {
        'timing_error_mean_ms': m['timing_error_mean_ms'],
        'beat_detection_rate':  m['beats_detected'] / duration_min,
        'unique_effects_count': float(m['unique_effects_count']),
        'effect_changes_count': float(m['effect_changes_count']),
        'intent_changes_count': float(m.get('intent_changes_count', 0)),
        'unique_intents_count': float(m.get('unique_intents_count', 0)),
    }

    results: dict = {}
    weighted_score = 0.0
    total_weight = 0.0

    for key, spec in criteria.items():
        value = derived.get(key, 0.0)
        weight = spec.get('weight', 1.0)
        if 'max' in spec:
            passed = value <= spec['max']
            score = max(0.0, 1.0 - value / spec['max']) if spec['max'] > 0 else float(passed)
        else:
            passed = value >= spec['min']
            score = min(1.0, value / spec['min']) if spec['min'] > 0 else float(passed)
        results[key] = {
            'value': round(value, 3),
            'passed': passed,
            'score': round(score, 3),
            'spec': spec,
        }
        weighted_score += score * weight
        total_weight += weight

    return {
        'passed': all(r['passed'] for r in results.values()),
        'score': round(weighted_score / total_weight if total_weight else 0.0, 3),
        'criteria': results,
    }


def print_evaluation(result: dict, transition_accuracy: dict | None = None) -> None:
    w = 72
    verdict = 'PASS' if result['passed'] else 'FAIL'
    print(f'\n{"─" * w}')
    print(f'  EVALUATION   score={result["score"]:.2f}   {verdict}')
    print(f'{"─" * w}')
    print(f'  {"criterion":<32} {"value":>10}  {"threshold":>12}  {"ok":>4}')
    print(f'  {"─"*32} {"─"*10}  {"─"*12}  {"─"*4}')
    for key, r in result['criteria'].items():
        spec = r['spec']
        threshold = f'≤ {spec["max"]}' if 'max' in spec else f'≥ {spec["min"]}'
        desc = spec.get('description', key)
        print(f'  {desc:<32} {r["value"]:>10.2f}  {threshold:>12}  {"✓" if r["passed"] else "✗":>4}')
    print(f'{"─" * w}')

    if transition_accuracy:
        ta = transition_accuracy
        print(f'\n  TRANSITION ACCURACY   '
              f'mean={ta["mean_offset_ms"]:.0f}ms  '
              f'max={ta["max_offset_ms"]:.0f}ms  '
              f'missed={ta["missed_count"]}  '
              f'false={ta["false_boundary_count"]}')
        print(f'  {"boundary":<34} {"GT":>8}  {"pred":>8}  {"offset":>9}  {"ok":>4}')
        print(f'  {"─"*34} {"─"*8}  {"─"*8}  {"─"*9}  {"─"*4}')
        for b in ta['boundaries']:
            label = f'{b["from_intent"]} → {b["to_intent"]}'
            pred  = f'{b["predicted_sec"]:.2f}s' if b['predicted_sec'] is not None else 'MISSED'
            off   = f'{b["offset_ms"]:+.0f}ms'  if b['offset_ms']    is not None else '—'
            ok    = '✓' if b['passed'] else ('—' if b['missed'] else '✗')
            print(f'  {label:<34} {b["ground_truth_sec"]:>7.2f}s  {pred:>8}  {off:>9}  {ok:>4}')
        if ta['false_boundaries']:
            times = ', '.join(f'{fb["t"]:.2f}s' for fb in ta['false_boundaries'])
            print(f'  false boundaries: {times}')
    print(f'{"─" * w}\n')


_SEARCH_WINDOW_SEC = 2.0
_PASS_THRESHOLD_MS = 100.0


def evaluate_against_labels(
    intents: list[dict],
    labels: list[dict],
    look_ahead_sec: float = 0.0,
    search_window_sec: float = _SEARCH_WINDOW_SEC,
) -> dict:
    """Compare predicted intent timeline against ground-truth labels.

    intents:        report['intents'] — list of {'t', 'intent', 'end'}.
    labels:         from _load_labels() — list of {'start', 'end', 'intent'}.
    look_ahead_sec: subtract from intent timestamps to align with raw audio time.
                    Pass LOOK_AHEAD_SEC when using EventBuffer intents; 0.0 for sweep replay.
    """
    # GT boundaries: N segments → N-1 transitions
    gt_boundaries = [
        {
            'gt_sec':      labels[i]['end'],
            'from_intent': labels[i]['intent'],
            'to_intent':   labels[i + 1]['intent'],
        }
        for i in range(len(labels) - 1)
    ]

    # Predicted transition times (all intent entries except the initial t=0 seed)
    pred_times = [
        e['t'] - look_ahead_sec
        for e in intents
        if e.get('t', 0.0) > 0.0
    ]

    matched_pred_indices: set[int] = set()
    boundaries_result = []

    for gt in gt_boundaries:
        T = gt['gt_sec']
        best_idx, best_dist = None, float('inf')
        for j, pt in enumerate(pred_times):
            if j in matched_pred_indices:
                continue
            d = abs(pt - T)
            if d < best_dist and d <= search_window_sec:
                best_dist, best_idx = d, j

        if best_idx is None:
            boundaries_result.append({
                'ground_truth_sec': T,
                'from_intent':      gt['from_intent'],
                'to_intent':        gt['to_intent'],
                'predicted_sec':    None,
                'offset_ms':        None,
                'missed':           True,
                'passed':           False,
            })
        else:
            matched_pred_indices.add(best_idx)
            offset_ms = (pred_times[best_idx] - T) * 1000.0
            boundaries_result.append({
                'ground_truth_sec': T,
                'from_intent':      gt['from_intent'],
                'to_intent':        gt['to_intent'],
                'predicted_sec':    round(pred_times[best_idx], 3),
                'offset_ms':        round(offset_ms, 1),
                'missed':           False,
                'passed':           abs(offset_ms) <= _PASS_THRESHOLD_MS,
            })

    false_boundaries = [
        {'t': round(pred_times[j], 3)}
        for j in range(len(pred_times))
        if j not in matched_pred_indices
    ]

    found   = [b for b in boundaries_result if not b['missed']]
    offsets = [abs(b['offset_ms']) for b in found]
    return {
        'boundaries':           boundaries_result,
        'mean_offset_ms':       round(sum(offsets) / len(offsets), 1) if offsets else 0.0,
        'max_offset_ms':        round(max(offsets), 1) if offsets else 0.0,
        'within_100ms_count':   sum(1 for b in boundaries_result if b['passed']),
        'missed_count':         sum(1 for b in boundaries_result if b['missed']),
        'false_boundary_count': len(false_boundaries),
        'false_boundaries':     false_boundaries,
        'total_boundaries':     len(gt_boundaries),
    }


def _sweep_classify_intent(
    bpm: float,
    onset_density: float,
    density_trend: float,
    current_intent_value: str | None,
    sub_bass_ratio: float,
    kick_strength: float,
    centroid_trend: float,
    cfg: dict,
) -> str:
    """Mirrors light_engine._classify_intent with explicit threshold dict.

    Priority: DROP → PEAK → BREAKDOWN → BUILDUP → GROOVE.
    Returns intent as a string value (matching LightIntent.value).
    """
    currently_drop      = current_intent_value == 'drop'
    currently_peak      = current_intent_value == 'peak'
    currently_breakdown = current_intent_value == 'breakdown'

    drop_threshold      = cfg['_DROP_MIN_DENSITY_EXIT']       if currently_drop      else cfg['_DROP_MIN_DENSITY_ENTER']
    peak_threshold      = 135.0                                if currently_peak      else 140.0
    breakdown_threshold = cfg['_BREAKDOWN_MAX_DENSITY_EXIT']   if currently_breakdown else cfg['_BREAKDOWN_MAX_DENSITY_ENTER']
    sub_bass_threshold  = cfg['_DROP_MIN_SUB_BASS_RATIO_EXIT'] if currently_drop      else cfg['_DROP_MIN_SUB_BASS_RATIO']

    kick_present = kick_strength >= cfg['_KICK_PRESENCE_THRESHOLD']

    if onset_density >= drop_threshold and bpm >= 100 and kick_present and sub_bass_ratio >= sub_bass_threshold:
        return 'drop'
    if bpm >= peak_threshold:
        return 'peak'
    if onset_density < breakdown_threshold:
        return 'breakdown'
    if not kick_present and onset_density < cfg['_BREAKDOWN_NO_KICK_MAX_DENSITY']:
        return 'breakdown'
    if density_trend >= cfg['_BUILDUP_MIN_TREND'] or centroid_trend >= cfg['_CENTROID_BUILDUP_TREND']:
        return 'buildup'
    return 'groove'


_INVALID_TRANSITIONS = frozenset({
    ('atmospheric', 'drop'),
    ('atmospheric', 'buildup'),
    ('atmospheric', 'peak'),
    ('peak',        'buildup'),
})


def _replay_feature_log(
    feature_log: list[dict],
    cfg: dict,
    look_ahead_sec: float = 2.5,
) -> list[dict]:
    """Replay feature log through the stability pipeline with given thresholds.

    Returns list of {'t': float, 'intent': str} representing intent changes,
    starting with {'t': 0.0, 'intent': 'atmospheric'}.

    Timestamps are in raw audio time (same reference as the GT label CSV).
    """
    vote_buffer_size = int(cfg['_VOTE_BUFFER_SIZE'])
    min_dwell_beats  = int(cfg['_MIN_DWELL_BEATS'])

    current_intent  = 'atmospheric'
    vote_buffer     = deque(maxlen=vote_buffer_size)
    beats_in_intent = 0
    predicted       = [{'t': 0.0, 'intent': current_intent}]

    # Pre-compute window boundaries with two pointers (O(n) total, not O(n²))
    left = 0
    for i, beat in enumerate(feature_log):
        t = beat['audio_time_sec']
        while feature_log[left]['audio_time_sec'] < t - look_ahead_sec:
            left += 1
        right = i + 1
        while right < len(feature_log) and feature_log[right]['audio_time_sec'] <= t + look_ahead_sec:
            right += 1
        window = feature_log[left:right]

        # Windowed features (mirrors _classify_windowed in light_engine.py)
        densities    = [b['onset_density'] for b in window]
        sorted_d     = sorted(densities)
        median_d     = sorted_d[len(sorted_d) // 2]
        mean_sub     = sum(b['sub_bass_ratio'] for b in window) / len(window)
        mean_kick    = sum(b['kick_strength']  for b in window) / len(window)
        mean_ctrd    = sum(b['centroid_trend'] for b in window) / len(window)
        mid          = len(densities) // 2
        past         = densities[:mid] if mid > 0 else densities
        future       = densities[mid:] if mid > 0 else densities
        past_mean    = sum(past) / len(past)
        future_mean  = sum(future) / len(future)
        window_trend = future_mean / past_mean if past_mean > 0 else 1.0

        raw = _sweep_classify_intent(
            beat['bpm'], median_d, window_trend,
            current_intent, mean_sub, mean_kick, mean_ctrd, cfg,
        )

        vote_buffer.append(raw)
        beats_in_intent += 1

        # Vote consensus check
        if len(vote_buffer) < vote_buffer_size:
            continue
        if len(set(vote_buffer)) != 1:
            continue
        if raw == current_intent:
            continue
        if beats_in_intent < min_dwell_beats:
            continue
        if (current_intent, raw) in _INVALID_TRANSITIONS:
            continue

        # Commit the intent change
        predicted.append({'t': t, 'intent': raw})
        current_intent  = raw
        beats_in_intent = 0
        vote_buffer.clear()

    return predicted


def _sweep_score(ta: dict) -> float:
    """Composite score. Lower is better.

    score = 0.6 * mean_offset_ms
          + 0.4 * max_offset_ms
          + missed_count         * 5000  (heavy penalty — wrong lights for whole section)
          + false_boundary_count * 500   (per spurious switch)
    """
    return (
        0.6 * ta['mean_offset_ms']
        + 0.4 * ta['max_offset_ms']
        + ta['missed_count'] * 5000.0
        + ta['false_boundary_count'] * 500.0
    )


_PARAM_RANGES: dict[str, tuple] = {
    '_BREAKDOWN_MAX_DENSITY_ENTER': (1.5,  5.0,  'float'),
    '_BUILDUP_MIN_TREND':           (1.1,  2.0,  'float'),
    '_DROP_MIN_DENSITY_ENTER':      (3.0,  6.0,  'float'),
    '_DROP_MIN_SUB_BASS_RATIO':     (0.20, 0.28, 'float'),  # entry gate
    '_DROP_MIN_SUB_BASS_RATIO_EXIT':(0.10, 0.22, 'float'),  # exit gate; enforced ≤ entry in sweep
    '_KICK_PRESENCE_THRESHOLD':     (0.5,  1.2,  'float'),
    '_CENTROID_BUILDUP_TREND':      (1.05, 1.5,  'float'),
    '_VOTE_BUFFER_SIZE':            (2,    8,    'int'),
    '_MIN_DWELL_BEATS':             (2,    10,   'int'),
}


def sweep_thresholds(
    feature_log: list[dict],
    labels: list[dict],
    look_ahead_sec: float = 2.5,
    n_samples: int = 10_000,
) -> list[dict]:
    """Sample n_samples random threshold configs, replay each against feature_log,
    score against labels. Returns top-50 configs sorted by score ascending.

    feature_log timestamps must be in raw audio time (same reference as labels).
    """
    import numpy as np

    rng = np.random.default_rng(42)
    results = []

    for _ in range(n_samples):
        cfg: dict = {}
        for param, (lo, hi, typ) in _PARAM_RANGES.items():
            if typ == 'int':
                cfg[param] = int(rng.integers(lo, hi + 1))
            else:
                cfg[param] = round(float(rng.uniform(lo, hi)), 3)
        # Derive locked hysteresis thresholds from swept entry values
        cfg['_BREAKDOWN_MAX_DENSITY_EXIT'] = round(cfg['_BREAKDOWN_MAX_DENSITY_ENTER'] + 0.5, 3)
        cfg['_DROP_MIN_DENSITY_EXIT']      = round(max(cfg['_DROP_MIN_DENSITY_ENTER'] - 1.5, 3.0), 3)
        # Enforce sub-bass hysteresis invariant: exit threshold ≤ entry threshold
        cfg['_DROP_MIN_SUB_BASS_RATIO_EXIT'] = min(cfg['_DROP_MIN_SUB_BASS_RATIO_EXIT'],
                                                    cfg['_DROP_MIN_SUB_BASS_RATIO'])
        # Fixed non-swept threshold
        cfg['_BREAKDOWN_NO_KICK_MAX_DENSITY'] = 6.0

        predicted = _replay_feature_log(feature_log, cfg, look_ahead_sec)
        ta        = evaluate_against_labels(predicted, labels, look_ahead_sec=0.0)
        score     = _sweep_score(ta)
        results.append({
            'score':               round(score, 1),
            'config':              cfg,
            'transition_accuracy': ta,
        })

    results.sort(key=lambda x: x['score'])
    return results[:50]
