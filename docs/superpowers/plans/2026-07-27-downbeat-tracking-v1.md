# Live Downbeat Tracking v1 (Offline) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and validate a downbeat/bar-phase tracker on the expert grids, good enough that bar-snapped commits using the *predicted* grid preserve the section decoder's boundary quality. Offline component only; runtime wiring is a later plan.

**Spec (binding):** `docs/superpowers/specs/2026-07-27-live-downbeat-tracking-design.md`.

**Tech stack:** everything already built — `training/nn/` (dataset/model/train/export/infer patterns), mel sidecars, frozen splits, CUDA pre-flight recipe, deterministic Viterbi machinery.

## Global Constraints

- Branch `downbeat_tracking` (create off `eval_pipeline` at its current head) in a NEW worktree `C:\Users\Julian\Projects\soundswitch-downbeat-worktree`. Never commit to master/eval_pipeline; no push until the coordinator says so. Data at the main-repo path as usual.
- The v2 retrain chain runs concurrently in the eval worktree and OWNS the GPU during its training window — coordinate: your training runs are short (≤10 min); if `nvidia-smi` shows the GPU busy with a training process, wait for it rather than contending.
- Splits: reuse `splits.json` exactly (train for fitting, val for any tuning, test read ONCE at the end). Eval-set ids/artists excluded already by construction.
- Determinism, TDD, per-task commits, CLAUDE.md policy — all as established.
- `uv run pytest -m "not integration"` green at every task end (run from your worktree).

### Task 1: Grid targets + dataset extension

**Files:** `training/nn/downbeat_dataset.py`, tests. Interfaces: reuse `WindowDataset`'s loading; add per-frame downbeat Gaussian targets (σ = 70 ms — the eval tolerance) and per-beat phase labels parsed from the beat CSVs (`downbeat` column ∈ {1,2,3,4}); masks for unannotated lead-ins/tails as in the section dataset. Validate against 3 real tracks (downbeat count per track ≈ bars; phases cycle 1→2→3→4).

### Task 2: Model + training (smoke-first, per standing directive)

**Files:** `training/nn/downbeat_model.py`, `downbeat_train.py`, tests. A ≤300k-param CRNN (same conv front-end pattern, smaller GRU) → per-frame downbeat logit. BCE with pos_weight from target sparsity; calibration reported. **10-track smoke with bitwise determinism proof + resume drill FIRST, then the full train-split run** (no separate review gate needed between smoke and full here — the machinery is proven; include both in one task report). TensorBoard runs under the existing logdir (`downbeat_v1` run names).

### Task 3: Bar-phase decoder + posterior sidecars

**Files:** `training/nn/downbeat_decoder.py`, `downbeat_infer.py`, tests. ONNX export (dynamo=False, golden test); sliding-window inference → downbeat activations aggregated per expert-beat instant (and per aubio-beat instant where a sim report exists); 4-state cyclic phase HMM with flip penalty, exact fixed-lag Viterbi, immutability property test (the Task 4 pattern: emitted phases never change); phase-confidence output. Deterministic sidecars for all val+test tracks.

### Task 4: Verdict

**Files:** `training/nn/evaluate_downbeat.py`, tests; docs (intent-level; the three CLAUDE.md touchpoints). Val: downbeat F1@±70ms, phase accuracy, flips/track; pick the flip penalty + lag on val. TEST (once): same metrics, PLUS the ablation that matters — re-run the section decoder on test tracks using the PREDICTED grid instead of the expert grid and report the boundary-F1/flicker delta (the number that says whether live bar-snapping works). Robustness probe on aubio-beat instants for tracks with sim reports. STOP after the test read.

---

**Acceptance for the plan:** test downbeat F1 ≥ 0.85, phase flips ≤ ~1/track median, and the predicted-grid section decode within a few points of the expert-grid decode on boundary-F1@±2s with flicker not worse. If unreachable after honest iteration on val, report BLOCKED with the gap analysis — do not lower gates.
