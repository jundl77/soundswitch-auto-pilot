# Ground-Truth Evaluation & Agentic Threshold Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically evaluate pipeline transition accuracy against annotated ground-truth CSVs, add a `--sweep` flag that sweeps 10k threshold combos in seconds via feature log replay, and enable an agentic outer loop (Claude reads report, patches `light_engine.py`, re-runs to verify).

**Architecture:** (1) `MusicAnalyser` records per-beat raw features with audio timestamps during simulation. (2) `evaluate_against_labels()` compares the EventBuffer intent timeline against the GT CSV, computing per-boundary offset in ms. (3) `sweep_thresholds()` replays the saved feature log through a self-contained stability pipeline with different threshold combos, scores each against labels, writes top-50 to report.json. Claude drives the outer loop.

**Tech Stack:** Python 3.11, numpy (already in venv), Plotly/Dash (visualizer), pytest, uv.

---

## File Map

| File | What changes |
|---|---|
| `lib/analyser/music_analyser.py` | Add `feature_log` list + `_frame_count` counter; append per-beat entry in `_track_beat()` |
| `lib/engine/event_buffer.py` | Add `feature_log` kwarg to `to_report()`; include it in the returned dict |
| `simulate/runner.py` | No change needed — feature_log exposed via `components['music_analyser'].feature_log` |
| `simulate/evaluator.py` | Add `evaluate_against_labels()`, `_sweep_classify_intent()`, `_replay_feature_log()`, `_sweep_score()`, `sweep_thresholds()`; update `print_evaluation()` |
| `simulate/cli.py` | Add `--sweep` flag; pass `feature_log` and `labels` through to report/evaluation |
| `simulate/visualizer_app.py` | Change GT band rendering from solid to hatched (diagonal stripe shapes) |
| `tests/test_evaluator.py` | New file: unit tests for `evaluate_against_labels()`, `_sweep_score()`, `_replay_feature_log()` |

---

## Task 1: Feature log capture in MusicAnalyser

**Files:**
- Modify: `lib/analyser/music_analyser.py`
- Test: `tests/test_music_analyser.py` (extend existing file)

### Context

`MusicAnalyser._track_beat()` fires every detected beat and already snapshots sub-bass and centroid values. We add a `feature_log` list (never reset mid-simulation) and a `_frame_count` counter (incremented every `analyse()` call) to derive audio-position timestamps.

- [ ] **Step 1: Add `feature_log` and `_frame_count` to `__init__`**

In `lib/analyser/music_analyser.py`, add two attributes at the END of `__init__`, after `self._reset_state()`. They must NOT go inside `_reset_state()` — they must survive the periodic state reset at 15 min.

```python
# Feature log: per-beat raw features for threshold sweep.
# Intentionally NOT in _reset_state() — survives the 15-min MusicAnalyser reset.
self.feature_log: list[dict] = []
# Monotonically increasing sample count — gives us audio time independent of wall-clock.
self._frame_count: int = 0
```

- [ ] **Step 2: Increment `_frame_count` in `analyse()`**

At the very end of `analyse()`, before `return audio_signal` (line 232), add:

```python
        self._frame_count += self.buffer_size
```

- [ ] **Step 3: Append to `feature_log` in `_track_beat()`**

In `_track_beat()`, after line 252 (`self._beat_centroid_samples.append(...)`) and before `await self.handler.on_beat(...)`, add:

```python
            self.feature_log.append({
                'audio_time_sec': self._frame_count / self.sample_rate,
                'bpm':            this_bpm,
                'onset_density':  self.get_onset_density(),
                'density_trend':  self.get_onset_density_trend(),
                'kick_strength':  self.get_kick_strength(),
                'centroid_trend': self.get_spectral_centroid_trend(),
                'sub_bass_ratio': self.get_sub_bass_ratio(),
            })
```

- [ ] **Step 4: Write the failing test**

Append to `tests/test_music_analyser.py`:

```python
import numpy as np
import asyncio
from lib.analyser.music_analyser import MusicAnalyser

class _NoopHandler:
    async def on_beat(self, *a, **kw): pass
    async def on_onset(self): pass
    async def on_note(self): pass
    async def on_cycle(self): pass
    async def on_section_change(self): pass
    def on_sound_start(self): pass
    def on_sound_stop(self): pass

def test_feature_log_populated_on_beats():
    """feature_log grows when beats are detected, entries have correct keys."""
    analyser = MusicAnalyser(44100, 256, _NoopHandler(), visualizer_updater=None)
    # Inject a fake beat by patching _track_beat to run directly
    # We verify the initial state is empty.
    assert analyser.feature_log == []
    assert analyser._frame_count == 0

def test_frame_count_increments_on_analyse():
    """_frame_count increments by buffer_size each analyse() call."""
    import asyncio

    analyser = MusicAnalyser(44100, 256, _NoopHandler(), visualizer_updater=None)
    buf = np.zeros(256, dtype=np.float32)
    asyncio.get_event_loop().run_until_complete(analyser.analyse(buf))
    assert analyser._frame_count == 256
    asyncio.get_event_loop().run_until_complete(analyser.analyse(buf))
    assert analyser._frame_count == 512
```

- [ ] **Step 5: Run test to verify it fails**

```bash
uv run pytest tests/test_music_analyser.py::test_feature_log_populated_on_beats tests/test_music_analyser.py::test_frame_count_increments_on_analyse -v
```

Expected: `test_feature_log_populated_on_beats` PASSES (initial state check), `test_frame_count_increments_on_analyse` FAILS with `AssertionError: assert 0 == 256` (before Step 2 is applied).

Apply Steps 1–3 now, then re-run.

- [ ] **Step 6: Run test to verify it passes**

```bash
uv run pytest tests/test_music_analyser.py::test_feature_log_populated_on_beats tests/test_music_analyser.py::test_frame_count_increments_on_analyse -v
```

Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add lib/analyser/music_analyser.py tests/test_music_analyser.py
git commit -m "feat: add feature_log and audio_time_sec tracking to MusicAnalyser"
```

---

## Task 2: Thread feature_log into EventBuffer and cli report

**Files:**
- Modify: `lib/engine/event_buffer.py`
- Modify: `simulate/cli.py`

### Context

`EventBuffer.to_report()` currently takes only `timing_log`. We add an optional `feature_log` kwarg. The cli's `_write_report_and_evaluate()` function pulls feature_log from the analyser after the simulation thread completes and passes it into the report.

- [ ] **Step 1: Add `feature_log` kwarg to `EventBuffer.to_report()`**

In `lib/engine/event_buffer.py`, change the signature at line 114:

```python
    def to_report(self, timing_log: list[dict] | None = None, feature_log: list[dict] | None = None) -> dict:
```

In the returned dict (around line 150), add `'feature_log'` as a key directly after `'timing_log'`:

```python
                'timing_log': tlog,
                'feature_log': feature_log or [],
```

- [ ] **Step 2: Pass feature_log through `_run_pipeline` in cli.py**

In `simulate/cli.py`, `_run_pipeline()` currently only calls `event_buffer.set_timing_log()`. Add a line after it to store feature_log in the event_buffer for retrieval:

```python
def _run_pipeline(components, duration_sec: float, event_buffer, command_queue):
    from simulate.runner import run_simulation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_simulation(components, duration_sec))
    finally:
        event_buffer.set_timing_log(command_queue.get_timing_log())
        event_buffer.set_feature_log(components['music_analyser'].feature_log)
        loop.close()
```

- [ ] **Step 3: Add `set_feature_log()` and store in `EventBuffer`**

In `lib/engine/event_buffer.py`, in `__init__`, add:

```python
        self._feature_log: list[dict] = []
```

Add the method after `set_timing_log()`:

```python
    def set_feature_log(self, log: list[dict]) -> None:
        with self._lock:
            self._feature_log = list(log)
```

Update `to_report()` to use `self._feature_log` when no explicit `feature_log` kwarg is given:

```python
    def to_report(self, timing_log: list[dict] | None = None, feature_log: list[dict] | None = None) -> dict:
```

And in the body, change:

```python
                'feature_log': feature_log if feature_log is not None else self._feature_log,
```

- [ ] **Step 4: Update `_write_report_and_evaluate` in cli.py to pass labels**

Change the signature and body of `_write_report_and_evaluate`:

```python
def _write_report_and_evaluate(event_buffer, command_queue, report_path: str,
                                labels: list[dict] | None = None) -> bool:
    from simulate.evaluator import evaluate, print_evaluation, evaluate_against_labels
    report = event_buffer.to_report(command_queue.get_timing_log())
    if labels:
        from simulate.runner import LOOK_AHEAD_SEC
        report['transition_accuracy'] = evaluate_against_labels(
            report['intents'], labels, look_ahead_sec=LOOK_AHEAD_SEC
        )
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'[simulate] report written → {report_path}')
    result = evaluate(report)
    print_evaluation(result, report.get('transition_accuracy'))
    return result['passed']
```

Update the call site in `run_file()`:

```python
        passed = _write_report_and_evaluate(event_buffer, command_queue, args.report, labels=labels)
```

- [ ] **Step 5: Run fast unit tests to make sure nothing is broken**

```bash
uv run pytest -m "not integration" -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lib/engine/event_buffer.py simulate/cli.py
git commit -m "feat: thread feature_log and labels through report pipeline"
```

---

## Task 3: `evaluate_against_labels()` in evaluator.py

**Files:**
- Modify: `simulate/evaluator.py`
- Create: `tests/test_evaluator.py`

### Context

This is the core accuracy function. It takes the `intents[]` list from the report (each entry: `{'t': float, 'intent': str, 'end': float}`) and the `labels[]` list from the CSV (each entry: `{'start': float, 'end': float, 'intent': str}`). Returns a dict with per-boundary transition offsets.

**Critical timestamp alignment:** EventBuffer records intent changes at `time.monotonic() - start_time`. Because `FileAudioClient` throttles to real-time and the `DelayedCommandQueue` fires `_commit_intent` after `LOOK_AHEAD_SEC`, the EventBuffer intent timestamp = audio_time + LOOK_AHEAD_SEC. Ground-truth CSV timestamps are raw audio time (user annotated what they heard, no delay applied). We subtract `look_ahead_sec` from each intent timestamp before matching.

- [ ] **Step 1: Write the failing tests first**

Create `tests/test_evaluator.py`:

```python
from simulate.evaluator import evaluate_against_labels, _sweep_score


# ---------------------------------------------------------------------------
# evaluate_against_labels
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _sweep_score
# ---------------------------------------------------------------------------

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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_evaluator.py -v
```

Expected: `ImportError` — `evaluate_against_labels` and `_sweep_score` don't exist yet.

- [ ] **Step 3: Implement `evaluate_against_labels` and `_sweep_score` in `simulate/evaluator.py`**

Add after the existing `print_evaluation` function:

```python
_SEARCH_WINDOW_SEC = 2.0


def evaluate_against_labels(
    intents: list[dict],
    labels: list[dict],
    look_ahead_sec: float = 0.0,
    search_window_sec: float = _SEARCH_WINDOW_SEC,
) -> dict:
    """Compare predicted intent timeline against ground-truth labels.

    intents:       report['intents'] — list of {'t', 'intent', 'end'}.
    labels:        from _load_labels() — list of {'start', 'end', 'intent'}.
    look_ahead_sec: subtract from intent timestamps to align with raw audio time.
                   Pass LOOK_AHEAD_SEC when using EventBuffer intents; 0.0 for sweep replay.
    """
    # GT boundaries: N segments → N-1 transitions
    gt_boundaries = [
        {
            'gt_sec':       labels[i]['end'],
            'from_intent':  labels[i]['intent'],
            'to_intent':    labels[i + 1]['intent'],
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
          + missed_count       * 5000   (heavy penalty — wrong lights for whole section)
          + false_boundary_count * 500  (per spurious switch)
    """
    return (
        0.6 * ta['mean_offset_ms']
        + 0.4 * ta['max_offset_ms']
        + ta['missed_count'] * 5000.0
        + ta['false_boundary_count'] * 500.0
    )
```

- [ ] **Step 4: Update `print_evaluation()` to show transition accuracy when present**

Add a `transition_accuracy` parameter to `print_evaluation` and print it when provided. Replace the existing function:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_evaluator.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -m "not integration" -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add simulate/evaluator.py tests/test_evaluator.py
git commit -m "feat: add evaluate_against_labels and _sweep_score with full test coverage"
```

---

## Task 4: Standalone sweep classification replay

**Files:**
- Modify: `simulate/evaluator.py`
- Modify: `tests/test_evaluator.py`

### Context

`sweep_thresholds()` needs to replay the feature log through the stability pipeline with arbitrary threshold values, without touching `LightEngine` (no side effects, no I/O). We add two private helpers:

1. `_sweep_classify_intent()` — mirrors `light_engine._classify_intent()` but takes all thresholds as explicit kwargs. No module globals.
2. `_replay_feature_log()` — iterates the feature log, builds windowed features for each beat (±look_ahead_sec), calls `_sweep_classify_intent`, applies vote buffer + dwell + invalid-transition guard, returns predicted transition list.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluator.py`:

```python
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
    # Sparse section (breakdown), then dense (drop)
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
    # Single beat at high density immediately after start (atmospheric)
    log = _make_feature_log([(0.0, 128.0, 9.5, 2.5)])
    cfg = {**_BASE_CFG, '_VOTE_BUFFER_SIZE': 1, '_MIN_DWELL_BEATS': 0}
    result = _replay_feature_log(log, cfg, look_ahead_sec=0.5)
    intents = [r['intent'] for r in result]
    assert 'drop' not in intents
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_evaluator.py::test_replay_detects_drop_transition tests/test_evaluator.py::test_replay_starts_atmospheric tests/test_evaluator.py::test_replay_blocks_invalid_transition -v
```

Expected: `ImportError` — `_replay_feature_log` not yet defined.

- [ ] **Step 3: Add `_sweep_classify_intent` to `simulate/evaluator.py`**

Add after `_sweep_score`:

```python
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
    """
    currently_drop      = current_intent_value == 'drop'
    currently_peak      = current_intent_value == 'peak'
    currently_breakdown = current_intent_value == 'breakdown'

    drop_threshold      = cfg['_DROP_MIN_DENSITY_EXIT']      if currently_drop      else cfg['_DROP_MIN_DENSITY_ENTER']
    peak_threshold      = 135.0                               if currently_peak      else 140.0
    breakdown_threshold = cfg['_BREAKDOWN_MAX_DENSITY_EXIT']  if currently_breakdown else cfg['_BREAKDOWN_MAX_DENSITY_ENTER']

    kick_present = kick_strength >= cfg['_KICK_PRESENCE_THRESHOLD']

    if onset_density >= drop_threshold and bpm >= 100 and kick_present and sub_bass_ratio >= 0.0:
        return 'drop'
    if bpm >= peak_threshold:
        return 'peak'
    if onset_density < breakdown_threshold:
        return 'breakdown'
    if not kick_present and onset_density < 6.0:
        return 'breakdown'
    if density_trend >= cfg['_BUILDUP_MIN_TREND'] or centroid_trend >= cfg['_CENTROID_BUILDUP_TREND']:
        return 'buildup'
    return 'groove'
```

- [ ] **Step 4: Add `_replay_feature_log` to `simulate/evaluator.py`**

Add after `_sweep_classify_intent`:

```python
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
    from collections import deque

    vote_buffer_size = int(cfg['_VOTE_BUFFER_SIZE'])
    min_dwell_beats  = int(cfg['_MIN_DWELL_BEATS'])

    current_intent  = 'atmospheric'
    vote_buffer     = deque(maxlen=vote_buffer_size)
    beats_in_intent = 0
    predicted       = [{'t': 0.0, 'intent': current_intent}]

    for beat in feature_log:
        t = beat['audio_time_sec']

        # Build symmetric look-ahead/look-behind window
        window = [b for b in feature_log if abs(b['audio_time_sec'] - t) <= look_ahead_sec]

        # Windowed features (mirrors _classify_windowed)
        densities   = [b['onset_density'] for b in window]
        sorted_d    = sorted(densities)
        median_d    = sorted_d[len(sorted_d) // 2]
        mean_sub    = sum(b['sub_bass_ratio'] for b in window) / len(window)
        mean_kick   = sum(b['kick_strength']  for b in window) / len(window)
        mean_ctrd   = sum(b['centroid_trend'] for b in window) / len(window)
        mid         = len(densities) // 2
        past        = densities[:mid] if mid > 0 else densities
        future      = densities[mid:] if mid > 0 else densities
        past_mean   = sum(past) / len(past)
        future_mean = sum(future) / len(future)
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
        if not all(v == raw for v in vote_buffer):
            continue
        if raw == current_intent:
            continue
        if beats_in_intent < min_dwell_beats:
            continue
        if (current_intent, raw) in _INVALID_TRANSITIONS:
            continue

        # Commit
        predicted.append({'t': t, 'intent': raw})
        current_intent  = raw
        beats_in_intent = 0

    return predicted
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_evaluator.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add simulate/evaluator.py tests/test_evaluator.py
git commit -m "feat: add _replay_feature_log and _sweep_classify_intent for threshold sweep"
```

---

## Task 5: `sweep_thresholds()` and `--sweep` CLI flag

**Files:**
- Modify: `simulate/evaluator.py`
- Modify: `simulate/cli.py`
- Modify: `tests/test_evaluator.py`

### Context

`sweep_thresholds()` samples 10k random configs over the 7-parameter space, replays each against the feature log, scores against labels, and returns the top-50. The `--sweep` flag wires this into `simulate file --no-ui`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluator.py`:

```python
from simulate.evaluator import sweep_thresholds


def test_sweep_returns_sorted_top50():
    """sweep_thresholds returns at most 50 configs sorted by score ascending."""
    # Use a tiny feature log (5 beats) so the test runs fast.
    sparse = [(float(i) * 0.5, 128.0, 2.0, 1.2) for i in range(3)]
    dense  = [(2.0 + float(i) * 0.5, 128.0, 9.5, 2.5) for i in range(3)]
    log    = _make_feature_log(sparse + dense)
    labels = _make_labels((0.0, 2.0, 'atmospheric'), (2.0, 5.0, 'drop'))
    results = sweep_thresholds(log, labels, look_ahead_sec=0.5, n_samples=20)
    assert len(results) <= 50
    scores = [r['score'] for r in results]
    assert scores == sorted(scores)
    assert all('config' in r and 'transition_accuracy' in r for r in results)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest tests/test_evaluator.py::test_sweep_returns_sorted_top50 -v
```

Expected: `ImportError` — `sweep_thresholds` not defined.

- [ ] **Step 3: Implement `sweep_thresholds` in `simulate/evaluator.py`**

Add after `_replay_feature_log`:

```python
_PARAM_RANGES: dict[str, tuple] = {
    '_BREAKDOWN_MAX_DENSITY_ENTER': (1.5,  5.0,  'float'),
    '_BUILDUP_MIN_TREND':           (1.1,  2.0,  'float'),
    '_DROP_MIN_DENSITY_ENTER':      (6.0,  12.0, 'float'),
    '_KICK_PRESENCE_THRESHOLD':     (1.0,  2.0,  'float'),
    '_CENTROID_BUILDUP_TREND':      (1.05, 1.5,  'float'),
    '_VOTE_BUFFER_SIZE':            (1,    4,    'int'),
    '_MIN_DWELL_BEATS':             (1,    6,    'int'),
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
            v = float(rng.uniform(lo, hi + (1 if typ == 'int' else 0)))
            cfg[param] = int(v) if typ == 'int' else round(float(v), 3)
        # Derive locked hysteresis thresholds from swept entry values
        cfg['_BREAKDOWN_MAX_DENSITY_EXIT'] = round(cfg['_BREAKDOWN_MAX_DENSITY_ENTER'] + 0.5, 3)
        cfg['_DROP_MIN_DENSITY_EXIT']      = round(max(cfg['_DROP_MIN_DENSITY_ENTER'] - 1.5, 3.0), 3)

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_evaluator.py::test_sweep_returns_sorted_top50 -v
```

Expected: PASS.

- [ ] **Step 5: Add `--sweep` flag to `simulate file` in `simulate/cli.py`**

In `add_simulate_subparser()`, add to the `fp` (file) parser after the existing flags:

```python
    fp.add_argument('--sweep', action='store_true',
                    help='After simulation: sweep 10k threshold combos against ground-truth CSV. Requires --no-ui and a label CSV alongside the audio.')
```

- [ ] **Step 6: Wire sweep into `run_file()` in `simulate/cli.py`**

In `run_file()`, in the `if args.no_ui:` block, replace:

```python
        passed = _write_report_and_evaluate(event_buffer, command_queue, args.report, labels=labels)
        sys.exit(0 if passed else 1)
```

with:

```python
        passed = _write_report_and_evaluate(event_buffer, command_queue, args.report, labels=labels)
        if getattr(args, 'sweep', False):
            if not labels:
                print('[sweep] error: --sweep requires a ground-truth CSV alongside the audio file.')
                sys.exit(1)
            _run_sweep(event_buffer, labels, args.report)
        sys.exit(0 if passed else 1)
```

Add `_run_sweep` above `run_file`:

```python
def _run_sweep(event_buffer, labels: list[dict], report_path: str) -> None:
    from simulate.evaluator import sweep_thresholds
    from simulate.runner import LOOK_AHEAD_SEC
    import json

    feature_log = event_buffer._feature_log
    if not feature_log:
        print('[sweep] no feature log available — skipping sweep.')
        return

    print(f'[sweep] running 10 000-sample threshold sweep …')
    results = sweep_thresholds(feature_log, labels, look_ahead_sec=LOOK_AHEAD_SEC)

    # Merge sweep_results into existing report
    try:
        with open(report_path) as f:
            report = json.load(f)
    except Exception:
        report = {}
    report['sweep_results'] = results

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    best = results[0] if results else None
    score_str = f'{best["score"]:.1f}' if best else 'N/A'
    print(f'[sweep] done — 10 000 combos evaluated, best score {score_str}, written to {report_path}')
```

- [ ] **Step 7: Run full test suite**

```bash
uv run pytest -m "not integration" -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add simulate/evaluator.py simulate/cli.py tests/test_evaluator.py
git commit -m "feat: add sweep_thresholds and --sweep flag for threshold optimization"
```

---

## Task 6: Ground-truth overlay hatching in visualizer

**Files:**
- Modify: `simulate/visualizer_app.py`

### Context

The visualizer already has a two-row layout (PRED top / GT bottom) when labels are present (lines 162–180 of `visualizer_app.py`). The GT bands currently use `fillcolor=cfg['primary'], opacity=0.80` — same style as the predicted bands, making them look identical in the separate row. The user asked for hatching so the two rows are unmistakably different at a glance.

Plotly layout `shapes` don't support `fillpattern`. We implement hatching by drawing diagonal line shapes at fixed x-intervals within each GT band, in addition to a lower-opacity fill.

- [ ] **Step 1: Replace the GT band rendering block in `_build_timeline`**

Locate the `# Ground-truth bands` block in `simulate/visualizer_app.py` (lines 161–180). Replace the inner shape append with:

```python
    # Ground-truth bands (when a label file is loaded)
    if has_labels:
        for lbl in labels:
            t_start = max(lbl['start'], x0)
            t_end   = min(lbl['end'],   x1)
            if t_end <= t_start:
                continue
            cfg   = _intent_config(lbl['intent'])
            # Base fill: same color but lower opacity so it reads as "reference"
            shapes.append(dict(
                type='rect', xref='x', yref='paper',
                x0=t_start, x1=t_end, y0=gt_y0, y1=gt_y1,
                fillcolor=cfg['primary'], opacity=0.30, line_width=0,
            ))
            # Diagonal hatch lines (one every 1.5s across the band)
            stripe_x = t_start
            while stripe_x < t_end:
                shapes.append(dict(
                    type='line', xref='x', yref='paper',
                    x0=stripe_x, x1=min(stripe_x + 1.0, t_end),
                    y0=gt_y0,    y1=gt_y1,
                    line=dict(color=cfg['primary'], width=1.5),
                ))
                stripe_x += 1.5
            if t_end - t_start > 1.5:
                annotations.append(dict(
                    x=(t_start + t_end) / 2, y=(gt_y0 + gt_y1) / 2,
                    xref='x', yref='paper',
                    text=cfg['label'], showarrow=False,
                    font=dict(color='rgba(255,255,255,0.70)', size=10, family='monospace'),
                ))
```

- [ ] **Step 2: Run unit tests**

```bash
uv run pytest -m "not integration" -v
```

Expected: all pass (visualizer has no unit tests — this confirms nothing else broke).

- [ ] **Step 3: Commit**

```bash
git add simulate/visualizer_app.py
git commit -m "feat: hatched GT bands in visualizer for clear predicted vs ground-truth separation"
```

---

## Task 7: Agentic outer loop — run sweep, patch thresholds, verify

**Files:**
- Modify: `lib/engine/light_engine.py` (thresholds only — Claude patches based on sweep results)

### Context

This task is executed by Claude in-session. It does not involve writing new code architecture — it uses the infrastructure built in Tasks 1–6 to run the optimization loop against the annotated Eric Prydz track.

- [ ] **Step 1: Run simulation with sweep**

```bash
python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3 --no-ui --report report.json --sweep
```

Expected output ends with: `[sweep] done — 10 000 combos evaluated, best score X.X, written to report.json`

- [ ] **Step 2: Read `report.json` and inspect results**

Claude reads `report.json`, checks `transition_accuracy.boundaries` to see which transitions are worst, and reads `sweep_results[0].config` for the best threshold config.

- [ ] **Step 3: Apply best config to `lib/engine/light_engine.py`**

Claude edits the threshold constants in `lib/engine/light_engine.py` (lines 35–58) to match `sweep_results[0].config`. Hysteresis exit thresholds are also updated to match the locked offsets used in the sweep.

- [ ] **Step 4: Re-run simulation to verify (no sweep)**

```bash
python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3 --no-ui --report report_verified.json
```

Claude reads `report_verified.json` → `transition_accuracy` and checks whether the transition offsets improved compared to the pre-patch run.

- [ ] **Step 5: Iterate if needed**

If mean offset > 500ms or any boundaries are missed, Claude runs a focused local sweep (set `n_samples=5000`, narrow the ranges around the current best config) or manually adjusts specific thresholds based on which features are at fault (e.g., drop threshold too high → lower `_DROP_MIN_DENSITY_ENTER`).

- [ ] **Step 6: Commit final thresholds**

```bash
git add lib/engine/light_engine.py
git commit -m "tune: optimize thresholds against Eric Prydz annotation via sweep"
```

- [ ] **Step 7: Run full test suite to confirm nothing regressed**

```bash
uv run pytest -v
```

Expected: all pass.

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Auto-evaluate when CSV present | Task 2 (`_write_report_and_evaluate` with `labels`) |
| Feature log per-beat in report | Task 1 + Task 2 |
| `evaluate_against_labels()` | Task 3 |
| `_sweep_score` (0.6×mean + 0.4×max + penalties) | Task 3 |
| `_replay_feature_log` + `_sweep_classify_intent` | Task 4 |
| `sweep_thresholds()` with 10k samples | Task 5 |
| `--sweep` flag on `simulate file` | Task 5 |
| Sweep output to `report.json` sweep_results | Task 5 |
| Hatched GT overlay in visualizer | Task 6 |
| Agentic outer loop | Task 7 |
| Timestamp alignment (look_ahead_sec correction) | Task 3 (`look_ahead_sec` param) |

**Placeholder scan:** No TBDs, no "implement later", all code blocks are complete.

**Type consistency:** `evaluate_against_labels` takes `list[dict]` for both `intents` and `labels` throughout. `_replay_feature_log` returns `list[dict]` with `'t'` and `'intent'` keys — same shape as EventBuffer `intents[]` for consistent calling. `sweep_thresholds` returns `list[dict]` with `score`, `config`, `transition_accuracy` keys — referenced consistently in `_run_sweep`.
