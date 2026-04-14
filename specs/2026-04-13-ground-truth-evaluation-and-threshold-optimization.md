# Ground-Truth Evaluation & Agentic Threshold Optimization

**Date:** 2026-04-13  
**Status:** Approved for implementation

---

## Problem

The pipeline produces an intent timeline (predicted section labels with timestamps). We now have a ground-truth annotation CSV for at least one track. The evaluator is completely blind to this annotation — it scores only generic metrics (beat count, timing accuracy, intent diversity). Nothing measures whether the predicted transitions land where they should.

Sub-50 ms transition accuracy is the target. At 128 BPM one beat is ~468 ms, so this requires near-sample-perfect boundary detection. The stability pipeline (3-vote buffer, 4-beat dwell) introduces inherent lag that needs to be quantified and minimized through threshold optimization.

---

## Goal

1. Automatically compute transition accuracy whenever a ground-truth CSV exists alongside the audio file — no new flags, no separate command.
2. Build a fast inner loop (feature log replay) that lets the optimizer sweep thousands of threshold combinations in seconds rather than re-running the full audio pipeline.
3. Claude drives an outer optimization loop: sweep → patch `light_engine.py` → re-run full simulation to verify → iterate until transitions converge.

---

## Architecture

### Two loops

**Inner loop — feature log replay (milliseconds per combo)**

Run audio through `MusicAnalyser` once. Save per-beat raw features to the simulation report. The optimizer replays those features through `_classify_intent()` with different threshold combinations — no audio I/O, no Aubio, no DSP. Pure Python math. Thousands of combos per second.

**Outer loop — agentic verification (Claude)**

After the inner loop finds an optimal config, Claude patches `light_engine.py`, re-runs the full audio simulation, checks that transition offsets actually improved (not just overfitting to the feature log), and iterates if needed. The outer loop runs in the current conversation session.

---

## Data Model

### Ground-truth label format (existing CSV)

```
start_sec, end_sec, intent
0.0, 9.04, atmospheric
9.04, 39.115, breakdown
...
```

N segments → N−1 boundaries. Each boundary is a `(time_sec, from_intent, to_intent)` triple.

### Feature log (new field in `report.json`)

Added to `EventBuffer.to_report()` under key `feature_log`:

```json
[
  {
    "t": 9.32,
    "bpm": 128.1,
    "onset_density": 2.4,
    "density_trend": 1.05,
    "kick_strength": 1.8,
    "centroid_trend": 1.02,
    "sub_bass_ratio": 0.31
  },
  ...
]
```

One entry per beat, in beat order. Timestamps are in simulation elapsed time as recorded by `EventBuffer._now()`. **Implementation note:** verify that feature log timestamps and intent timeline timestamps share the same reference frame, and that this frame aligns with the ground-truth CSV (which is in listener/audio time). If `event_buffer.set_intent()` is called at detection time rather than after the command queue delay fires, a fixed offset correction of `LOOK_AHEAD_SEC` may be needed when matching predicted transitions to ground-truth boundaries. Stored in `MusicAnalyser` during the audio pass and flushed into the report at end of simulation.

### Transition accuracy result (new section in evaluation output)

```json
{
  "transition_accuracy": {
    "boundaries": [
      {
        "ground_truth_sec": 9.04,
        "from_intent": "atmospheric",
        "to_intent": "breakdown",
        "predicted_sec": 9.51,
        "offset_ms": 470,
        "passed": false
      },
      ...
    ],
    "mean_offset_ms": 312.0,
    "max_offset_ms": 890.0,
    "within_50ms_count": 0,
    "within_500ms_count": 4,
    "total_boundaries": 8
  }
}
```

`passed` per boundary: `|offset_ms| < 100 ms` — the target accuracy. A missed boundary (no predicted transition within ±2 s search window) is reported separately and carries a heavy score penalty (see Sweep Scoring below).

---

## Components

### 1. `evaluate_against_labels(intents, labels)` — `simulate/evaluator.py`

New function. Takes the `intents[]` list from the report and the loaded label segments. For each ground-truth boundary, finds the nearest predicted intent change and computes the offset. Returns the `transition_accuracy` dict above.

Matching strategy: for each GT boundary at time T, find the predicted transition `argmin |predicted_t - T|`. If no predicted transition exists within ±2 s, mark as `missed`. False boundaries (predicted transitions with no GT boundary within ±2 s) are counted separately.

Called automatically from `_write_report_and_evaluate()` whenever labels are present. Also called from `print_evaluation()` for console output.

### 2. Feature log capture — `lib/analyser/music_analyser.py`

`MusicAnalyser` accumulates a list of beat feature dicts during the audio pass. Each time `_on_beat()` fires, append the current feature vector. The list is exposed via `music_analyser.feature_log` for the runner to pull at end of simulation.

`simulate/runner.py::run_simulation()` reads `music_analyser.feature_log` after the loop ends and returns it alongside `command_queue`. The caller writes it into the report via `EventBuffer.to_report()`.

### 3. `sweep_thresholds(feature_log, labels, grid)` — `simulate/evaluator.py`

New function. Replays `feature_log` through `_classify_intent()` for each threshold combo in `grid`. Applies the full stability pipeline (vote buffer, dwell, invalid-transition guard) in pure Python — no side effects, no I/O. Scores each combo using `evaluate_against_labels()`. Returns a list of `(score, config_dict)` sorted by score ascending (lower is better).

The stability pipeline replay must be a self-contained Python function (not calling into `LightEngine`) to avoid side effects and keep it fast.

**Swept parameters and default grid ranges:**

| Parameter | Default | Sweep range | Steps |
|---|---|---|---|
| `_BREAKDOWN_MAX_DENSITY_ENTER` | 3.0 | 1.5 – 5.0 | 8 |
| `_BUILDUP_MIN_TREND` | 1.3 | 1.1 – 2.0 | 6 |
| `_DROP_MIN_DENSITY_ENTER` | 8.5 | 6.0 – 12.0 | 7 |
| `_KICK_PRESENCE_THRESHOLD` | 1.3 | 1.0 – 2.0 | 6 |
| `_CENTROID_BUILDUP_TREND` | 1.1 | 1.05 – 1.5 | 6 |
| `_VOTE_BUFFER_SIZE` | 3 | 1 – 4 | 4 |
| `_MIN_DWELL_BEATS` | 4 | 1 – 6 | 6 |

Total grid: ~8×6×7×6×6×4×6 ≈ 483,840 combos. Reduce with random sampling (10k points via Latin hypercube) for the first pass; refine around the top-10 configs with a dense local grid.

Hysteresis exit thresholds are locked to their entry counterparts with a fixed offset (e.g. exit = entry + 0.5) during the sweep to reduce dimensionality.

**Sweep scoring function:**

```
score = 0.6 × mean_offset_ms
      + 0.4 × max_offset_ms
      + missed_boundaries × 5000      # heavy penalty per missed boundary
      + false_boundaries × 500        # per spurious switch
```

Lower score is better. A config with any missed boundary will score at least 5000 — effectively ranked last unless all others also miss. False boundaries are penalized per occurrence to discourage flickery configs that hit a real boundary by accident.

**Sweep output:** Results are written to `report.json` under key `sweep_results` as a ranked list of the top-50 configs (score, all threshold values, boundary-by-boundary offset breakdown). No console output. Claude reads the report and applies the best config to `light_engine.py`.

### 4. `--sweep` flag on `simulate file`

`add_simulate_subparser()` gains one new flag: `--sweep`. When set:

1. Run the full audio simulation (feature log is always captured now).
2. After the simulation, call `sweep_thresholds(feature_log, labels, grid)`.
3. Write the top-50 configs into `report.json` under key `sweep_results`. No console output — Claude reads the file and applies the best config.
4. Print a single summary line: `[sweep] done — N combos evaluated, best score X.X, written to report.json`.

If no labels are found, `--sweep` exits with a clear error: "sweep requires a ground-truth CSV alongside the audio file."
`--sweep` requires `--no-ui` (sweep runs after the audio pass completes; the Dash UI blocks).

### 5. Ground-truth overlay in Dash visualizer — `simulate/visualizer_app.py`

When labels are present, draw them as a hatched band layer behind the predicted intent timeline. One hatched band per segment using the same intent color scheme. Hatching makes ground-truth unmistakably distinct from the solid predicted bars at a glance — no overlap ambiguity. No new interaction — purely informational.

---

## Evaluation Metrics

### Primary — transition accuracy

- **Mean transition offset (ms)**: lower is better. Target: <100 ms.
- **Max transition offset (ms)**: worst-case boundary. Target: <100 ms.
- **Missed boundaries**: GT boundaries with no predicted transition within ±2 s. Target: 0.
- **False boundaries**: predicted transitions with no GT boundary within ±2 s. Target: 0.
- **Composite sweep score**: `0.6 × mean_offset + 0.4 × max_offset + misses × 5000 + false × 500`.

### Secondary — frame accuracy

Slice both timelines at 100 ms resolution. Compute % frames where predicted intent == ground-truth intent. Broken down per intent class. Identifies which intents are chronically misclassified even when transitions are broadly correct.

### Existing metrics (unchanged)

Timing error, beat detection rate, intent diversity — all still computed and reported.

---

## Agentic Outer Loop (Claude)

After infrastructure is built, Claude drives the optimization in-session:

1. Run `python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3 --no-ui --report report.json --sweep`
2. Read `report.json` — inspect `transition_accuracy` and `sweep_results`.
3. Identify which boundaries are worst. Understand why (which feature is at fault: density spike? missing kick? centroid flat during buildup?).
4. Apply the best sweep config to `lib/engine/light_engine.py`.
5. Re-run full simulation (no sweep needed) to verify real-pipeline accuracy.
6. If transitions improved but not enough, run a focused local sweep around the current config.
7. Repeat until mean offset < 500 ms and no missed boundaries.
8. Document final threshold values in `lib/analyser/CLAUDE.md` rationale sections.

The outer loop is intentionally Claude-driven (not automated) because some decisions require domain judgment: a config that nails the drop boundary but misclassifies the entire groove section is worse than one that's 200 ms late on the drop but gets the groove right.

---

## File Changes

| File | Change |
|---|---|
| `simulate/evaluator.py` | Add `evaluate_against_labels()`, `sweep_thresholds()`, update `print_evaluation()` |
| `lib/analyser/music_analyser.py` | Add `feature_log` accumulation on each beat event |
| `lib/engine/event_buffer.py` | Accept and store `feature_log` in `to_report()` |
| `simulate/runner.py` | Extract `feature_log` from `music_analyser` after loop, pass to report |
| `simulate/cli.py` | Add `--sweep` flag; pass labels to `_write_report_and_evaluate`; auto-evaluate when labels present |
| `simulate/visualizer_app.py` | Draw ground-truth overlay lane when labels are present |

No new files. No new subcommands.

---

## Out of Scope

- Automatic write-back of optimal thresholds to `light_engine.py` (Claude does this step manually to apply domain judgment).
- Multi-track evaluation (one annotated track to start; extend when more labels exist).
- YAMNet section detection in simulation (remains disabled for speed; sweep does not include YAMNet parameters).
- ML model training / neural classification (the classifier remains rule-based with tuned thresholds).
