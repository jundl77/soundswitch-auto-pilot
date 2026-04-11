"""
Agentic evaluator: scores a simulation report against configurable criteria.

Usage (headless / CI):
  python visualize_auto_pilot file song.mp3 --no-ui --report report.json
  # exit code 0 = PASS, 1 = FAIL

Or in Python:
  from simulate.evaluator import evaluate, print_evaluation
  result = evaluate(report)   # report from EventBuffer.to_report()
  print_evaluation(result)
"""

DEFAULT_CRITERIA: dict = {
    'timing_error_mean_ms': {
        'max': 15.0, 'weight': 0.30,
        'description': 'Mean delay error (ms)',
    },
    'beat_detection_rate': {
        'min': 10.0, 'weight': 0.30,
        'description': 'Beats detected per minute',
    },
    'unique_effects_count': {
        'min': 2, 'weight': 0.20,
        'description': 'Distinct effect channels used',
    },
    'effect_changes_count': {
        'min': 1, 'weight': 0.20,
        'description': 'Total effect changes observed',
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


def print_evaluation(result: dict) -> None:
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
    print(f'{"─" * w}\n')
