# Raveform Eval Set — Implementation Plan (replaces the Generate track as the sim benchmark)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze a small, diverse, expert-labeled eval set of Raveform tracks; make the simulation's integration tests run on it (label-aligned scores + determinism) instead of the Generate track; guarantee the set never enters NN training or validation.

**Owner directive (verbatim intent):** "build a small set of eval data from the raveform data, and replace the generate track — the simulation then runs on that eval dataset, which is excluded from training."

**Sequencing:** runs AFTER eval-pipeline Task 3 (needs the label-aligned evaluator + baseline machinery). The NN plan's Task 1 (`make_splits`) is amended to exclude these ids — already committed alongside this plan.

## Global Constraints

- Branch `eval_pipeline`, same worktree. No push without the coordinator.
- **Audio is never committed.** The eval set is committed as ids + labels + expectations (`training/eval_set.json` + baseline file); the mp3s live in the gitignored corpus. Downloaded YouTube audio does not enter git — the repo is public.
- Integration tests that need eval-set audio must fail with a CLEAR one-line message (not skip silently, not crash obscurely) when the corpus is absent: "eval-set audio missing — run training/raveform_download.py".
- Determinism contract unchanged: same track → byte-identical report, checksums stable.
- `uv run pytest` green at every task end; commits per task.

### Task A: Select and freeze the eval set

**Files:** Create `training/select_eval_set.py`; data+committed output `training/eval_set.json`.

- [ ] Selection from `clean_manifest.csv` ok-rows ∩ annotations, ~10 tracks, deterministic given the same inputs (seeded; selection inputs recorded in the output). Criteria, in order: (1) all five v1 classes present in the track's labels where possible (intro/buildup/breakdown/drop/outro), (2) BPM diversity (spread across the corpus BPM range, from the Raveform beat grids), (3) duration 3–8 min, (4) at least 8 section boundaries, (5) no two tracks from the same `track_id` prefix family. Print the per-track rationale table.
- [ ] `training/eval_set.json` (COMMITTED): `{"youtube_ids": [...], "selected_from": {"clean_rows": N, "seed": S}, "rationale": {id: one-liner}}`.
- [ ] Commit: `eval-set: select and freeze 10 expert-labeled benchmark tracks`

### Task B: Eval-set runner + integration repoint + Generate retirement

**Files:** Create `training/run_eval_set.py`; modify `tests/test_simulation.py`; delete `samples/generate_eric_prydz_192k.mp3` (and its `.npy` if present); modify root `CLAUDE.md` + `lib/analyser/CLAUDE.md` (benchmark description, intent level).

- [ ] `run_eval_set.py`: for each eval-set track — fast sim → report → label-aligned scores (import metric functions from `training/evaluate_against_labels.py`; v1 space + flicker) → per-track and aggregate table; `--write-baseline` writes `training/eval_set_baseline.json` (COMMITTED: per-track checksums + scores). Without the flag it compares against the committed baseline and exits nonzero on: any checksum change (determinism/behavior change detector) or any score below baseline minus tolerance (regression detector). This file is the successor of the old 0.97-plumbing-PASS gate.
- [ ] `tests/test_simulation.py` repoint: the module-level cached run + its 6 assertions move from the Generate track to the FIRST eval-set track (timing exactness, flush, duration match, speed, determinism/checksum stay as-is — they are track-agnostic); add one integration test running the 3-track head of the eval set through `run_eval_set` compare-mode. Clear failure message when audio is absent (per Global Constraints). Keep total integration wall time under ~60 s.
- [ ] Delete the Generate mp3 from `samples/` (git rm; the old reference table stays in the committed Stage-1 plan doc as history). If `samples/` becomes empty, remove the dir; update any path references (grep `generate_eric_prydz`).
- [ ] **Speed (owner directive):** eval-set tracks are exempt from the delete-the-.npy rule — their decode caches persist (~10 tracks ≈ under 1 GB) so benchmark runs and integration tests skip decoding entirely after first contact. `run_eval_set.py` prints per-track and aggregate ×-realtime; the integration suite must stay under ~60 s wall.
- [ ] Run `--write-baseline` once; commit the baseline. Full `uv run pytest` green.
- [ ] CLAUDE.md: benchmark = frozen expert-labeled eval set, excluded from all training/validation; how to regenerate the baseline (intent level, no numbers).
- [ ] Commit: `eval-set: sim benchmark runs on expert-labeled raveform tracks; retire the Generate anchor`
