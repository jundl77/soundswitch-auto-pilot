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
                'passed':           abs(offset_ms) <= 100.0,
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
