# Onset threshold — every draw taken, including the wrong ones

Evidence for `ONSET_THRESHOLD` in `lib/analyser/madmom_rhythm.py`. Committed
because the point estimate moved four times, and three of those moves were
mistakes in the *measurement*, not in the corpus. A reader who sees only the
final number cannot tell how firm it is; this file is what makes that visible.

Instrument: `training/onset_operating_point.py`. Raw sweep for the shipped draw:
`training/onset_operating_point.json`.

## The draws

| # | pool | n | span | aubio median | matched | why this draw was wrong |
|---|---|---|---|---|---|---|
| 1 | every track, sorted-and-spaced | 17 | 90 s prefixes | 6.656/s | **0.30** | prefixes: a track's opening is its sparsest, so this calibrated against intros |
| 2 | every track | 17 | 240 s prefixes | 6.258/s | **0.40** | still a prefix; also still holdout-contaminated |
| 3 | every track | 17 | whole | 6.077/s | **0.40** | **contained 2 held-out test tracks** (`s_iJfyng3Lo`, `VzMpzT1sxEw`) |
| 4 | test excluded | 17 | whole | 5.883/s | **0.44** | `eval_set` still selectable — the tracks whose deltas are this PR's headline evidence |
| 5 | test + eval_set + excluded_eval_set excluded | 17 | whole | 6.792/s | **0.35** | correct scope, but n too small to be resample-stable |
| **6** | **same scope, larger sample** | **49** | **whole** | **6.262/s** | **0.35** | **shipped** |

Draws 3, 4 and 5 differ *only* in which tracks were excluded, and they span two
full steps of the ladder. That is sampling variance in a median, not a property
of the corpus — which is exactly why draw 5's agreement with draw 6 is
corroboration rather than proof.

## Stability of the shipped draw

2000 track-resamples of draw 6, seed 20260729:

| threshold | share of resamples |
|---|---|
| **0.35** | **49.6 %** |
| 0.30 | 24.8 % |
| 0.40 | 20.2 % |
| 0.42 | 4.7 % |
| 0.25 | 0.5 % |
| 0.44 | 0.2 % |

Mode 0.35; 80 % interval **[0.30, 0.40]**.

**What the remaining uncertainty costs.** Across that interval the corpus median
onset rate runs 6.594/s (at 0.30) to 5.862/s (at 0.40) against aubio's 6.262/s —
about ±6 %. So the constant is not pinned to two decimals, but the band it sits
in is narrow in the units that matter to the rule engine, and no value in it
approaches madmom's library default (0.50 → 5.341/s, 15 % low).

## The shipped ladder (n = 49, whole tracks, holdout-free)

aubio: median 6.262/s, mean 6.208, p10 4.273, p90 8.043.

| thr | median | mean | vs aubio | per-track ratio p10/p50/p90 |
|---|---|---|---|---|
| 0.20 | 7.397 | 7.274 | +1.135 | 1.00 / 1.17 / 1.44 |
| 0.25 | 7.010 | 6.920 | +0.748 | 0.96 / 1.12 / 1.36 |
| 0.30 | 6.594 | 6.612 | +0.332 | 0.93 / 1.07 / 1.31 |
| **0.35** | **6.184** | **6.331** | **−0.078** | **0.89 / 1.04 / 1.24** |
| 0.40 | 5.862 | 6.052 | −0.400 | 0.84 / 0.99 / 1.17 |
| 0.42 | 5.777 | 5.933 | −0.485 | 0.81 / 0.97 / 1.14 |
| 0.44 | 5.668 | 5.811 | −0.594 | 0.79 / 0.95 / 1.12 |
| 0.46 | 5.513 | 5.689 | −0.749 | 0.77 / 0.93 / 1.09 |
| 0.50 | 5.341 | 5.441 | −0.921 | 0.74 / 0.89 / 1.03 |
| 0.60 | 4.668 | 4.753 | −1.593 | 0.64 / 0.77 / 0.89 |
| 0.70 | 3.895 | 4.018 | −2.367 | 0.53 / 0.64 / 0.77 |

## The rule this branch adopted from it

Any rule-engine constant calibrated from a track sample must show stability
across draws, or ship with an interval. A single-draw median is an estimate
wearing a constant's clothes.
