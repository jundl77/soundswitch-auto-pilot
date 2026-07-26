# NN Section Classifier v1 (Offline) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the CRNN acoustic model on the clean Raveform subset, decode with fixed-lag Viterbi, and beat the Stage-1 rule classifier on the label-aligned metrics (macro-F1, boundary-F1, drop recall/precision, flicker rate) on a held-out test split — entirely offline. Runtime integration is a later plan, gated on winning here.

**Architecture (binding spec):** `docs/superpowers/specs/2026-07-26-nn-section-classifier-design.md`. This plan implements its Phases 2–3: dataset builder → CRNN (mel-only v1, label head + boundary head) → ONNX export + posterior sidecars → priors fitting → fixed-lag Viterbi decoder → offline evaluation + sweep harness.

**Tech Stack:** torch (CUDA, RTX 3070) for training only; onnxruntime (CPU, pinned) for all inference; numpy for the decoder; existing eval-pipeline artifacts (`clean_manifest.csv`, `features/*.npz` mel sidecars, `training_table.csv.gz`, segments.json, Raveform beat CSVs, `training/evaluate_against_labels.py`).

## Global Constraints

- Branch `eval_pipeline`, worktree `C:\Users\Julian\Projects\soundswitch-eval-worktree`. Never commit to master; no push without the coordinator.
- DATA at `C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform\` (gitignored): model checkpoints under `models/`, posterior sidecars under `posteriors/`, splits file at `splits.json`. Committed code: `training/nn/` package + tests. No model binaries in git.
- Dependencies: add a `[project.optional-dependencies] training = [...]` extra to pyproject (torch, onnx, onnxruntime pinned exact versions). `uv sync --extra training` documented. Nothing in the default deps changes; the live pipeline gains no new imports in this plan.
- `lib/` and `simulate/` remain untouched (offline plan).
- Determinism: training is seeded but only reproducible-ish (CUDA); everything AFTER the checkpoint is exact — ONNX inference single-threaded (`intra_op_num_threads=1`), posterior sidecars written once and content-hashed, decoder pure numpy. The eval chain must be byte-stable given the sidecars.
- Label space: `label_v1` (5 classes: intro, buildup, breakdown, drop, outro). Loss-masked regions: leading `[0, first_section_start)`, trailing beyond last section end, and merged-run join frames for the boundary head.
- Splits: track-level 70/15/15 by deterministic hash of `youtube_id` (seed constant), the frozen eval set (`training/eval_set.json`) excluded from all splits, splits frozen in `splits.json` and NEVER regenerated implicitly — the 1,423-track retrain later must extend, not reshuffle, existing assignments. Test split is read by Task 6 ONLY.
- A downloader may still be writing to `audio/` — same 60 s mtime rule; but all inputs here come from sidecars/tables built by the eval pipeline, so contact with `audio/` should be zero.

---

### Task 1: Training extra + dataset builder

**Files:** Create `training/nn/__init__.py`, `training/nn/dataset.py`; modify `pyproject.toml`; test `tests/test_nn_dataset.py`.

**Interfaces:** `dataset.py` exposes `make_splits(data_dir, seed=1337) -> dict` (writes/reads `splits.json`, keys train/val/test → lists of youtube_ids; ids listed in `training/eval_set.json` are EXCLUDED from all three splits — they are the frozen sim benchmark and must never be trained or validated on; assert the exclusion in a test) and a `WindowDataset` (torch `Dataset`) yielding `(mel_window [W,40] float32, label_targets [W10] int64, label_mask [W10] bool, boundary_targets [W] float32, boundary_mask [W] bool)` where W = window frames (~16 s / 46 ms ≈ 348, exact value derived from `frame_sec` in the sidecars and exposed as a constant), W10 = label-head frames (~10 Hz pooling factor). Targets built from segments.json in `label_v1` space; Gaussian boundary targets σ = 0.5 s; masks per the constraints. Window sampling: random offsets within each track per epoch (window-offset augmentation); gain jitter ±3 dB on mel (additive in log domain).

- [ ] TDD the target/mask construction on synthetic sections (leading offset masked; boundary Gaussian peak at the right frame; merged-join boundary deletion; label pooling to 10 Hz majority).
- [ ] `make_splits` determinism test (same ids → same assignment; extension property: adding new ids never changes existing assignments).
- [ ] Verify against 3 real sidecars end-to-end (shapes, mask fractions plausible vs known leading-offset stats).
- [ ] `uv sync --extra training` works; torch sees CUDA (`torch.cuda.is_available()` printed in your report — if False, report BLOCKED, don't train on CPU silently).
- [ ] Commit: `nn: training extra + windowed dataset with masked v1 targets`

### Task 2: Model + training loop

**Files:** Create `training/nn/model.py`, `training/nn/train.py`; test `tests/test_nn_model.py` (shape/forward tests, CPU).

**Interfaces:** `model.py` → `SectionCRNN(n_mels=40, n_classes=5)`: conv stack (3 blocks: Conv2d 3×3 + BN + GELU, channels 32/64/64, freq-pool 2× each block, time preserved) → flatten freq → 1D conv (kernel 5) to 128 ch → biGRU hidden 128 (both directions) → label head (avg-pool time ×~4.6 to ≈10 Hz, linear → 5 logits/frame) + boundary head (linear → 1 logit/frame at full frame rate). Target ≤ ~1M params (print the count; if over, shrink GRU first). `train.py` CLI: focal loss (γ=2) with class weights from the train-split histogram + boundary BCE (pos_weight from target sparsity) + total-variation penalty λ·mean(|p_t − p_{t−1}|) on label softmax at non-boundary frames (λ=0.1 start); AdamW lr 3e-4, cosine decay, batch 32 windows, early stop on val macro-F1 (patience 10 epochs), seeded; per-epoch val: macro-F1 (frame), boundary PR-AUC, ECE per class. Checkpoints + a `training_report.json` (curves, final val metrics, ECE table) under `models/v1/`.

- [ ] Forward-shape and param-count tests on CPU.
- [ ] Train on the 3070. Sanity gates before calling it done: val frame macro-F1 > 0.55 (label priors alone give ~0.2-0.3; if below, iterate lr/λ/architecture within this task and document attempts), boundary PR-AUC > 0.3, ECE < 0.15 per majority classes. If unreachable after honest iteration, report BLOCKED with curves.
- [ ] Commit code (not checkpoints): `nn: SectionCRNN + calibrated two-head training`

### Task 3: ONNX export + posterior sidecars

**Files:** Create `training/nn/export_onnx.py`, `training/nn/infer.py`; test `tests/test_nn_onnx.py`.

**Interfaces:** export the checkpoint to `models/v1/model.onnx` (dynamic time axis); `infer.py` runs the sliding-window inference EXACTLY as the spec's runtime will: 16 s window, 100 ms hop over each track's mel sidecar, per-frame posterior averaging across overlapping windows, edge frames of each window excluded from aggregation (the spec's never-read-the-edge rule — exclude the outer 1 s each side); writes `posteriors/<youtube_id>.npz` (`label_post [T10,5]`, `boundary [T]`, `frame_sec`, `t0`, `model_sha`).

- [ ] Golden test: one uncached ONNX inference on a fixed synthetic mel window, tolerance 1e-5 against saved reference — exercised in CI (no cache).
- [ ] Torch-vs-ONNX parity on 3 real windows (max abs diff < 1e-4).
- [ ] Determinism: run sidecar generation twice for 3 tracks → byte-identical npz.
- [ ] Generate sidecars for ALL clean tracks (train/val/test — needed for sweeps and eval); report wall time and sizes.
- [ ] Commit: `nn: onnx export + deterministic sliding-window posterior sidecars`

### Task 4: Priors + fixed-lag Viterbi decoder

**Files:** Create `training/nn/priors.py`, `training/nn/decoder.py`; tests `tests/test_nn_decoder.py`.

**Interfaces:** `priors.py` fits from segments.json (train split ONLY): transition matrix in v1 space with structural facts hard (-inf: outro→anything-but-end-of-track, anything→intro; intro initial-only) and buildup→{breakdown,drop} forced near-uniform; per-class duration as min-floor + geometric tail fitted on bar counts (bars from Raveform beat CSVs, downbeat column), widened per spec (floor = corpus p05, tail from median). Writes `models/v1/priors.json`. `decoder.py` → `FixedLagViterbi(priors, lag_bars, class_prior_division=True, drop_miss_cost=…)`: consumes a track's posterior sidecar + its bar grid, emits immutable per-bar decisions at the configured lag; pure numpy; boundary head enters as a hazard multiplier on switch probabilities at that bar. Also a convenience `decode_track(npz_path, beat_csv_path, params) -> list[(t_bar, label)]`.

- [ ] TDD on synthetic posteriors: sticky behavior vs flicker; min-duration honored; -inf transitions never emitted; lag semantics (decision for bar B never changes after emission — property test); class-prior division effect; boundary hazard sharpens switch placement.
- [ ] Determinism: same inputs → identical output (trivially, but pin it).
- [ ] Commit: `nn: raveform priors + fixed-lag viterbi decoder (bar rate)`

### Task 5: Offline evaluation + sweep harness

**Files:** Create `training/nn/evaluate_v1.py`, `training/nn/sweep.py`.

**Interfaces:** `evaluate_v1.py`: decode every VAL-split track, convert decisions to an intent-style timeline, score with `training/evaluate_against_labels.py`'s metric functions (import, don't duplicate) in the `label_v1` space + flicker rate; produce side-by-side vs the rule baseline's numbers on the SAME val tracks; write `models/v1/eval_val.json`. `sweep.py`: grid/random sweep over decoder params (lag, stickiness/self-transition scale, drop_miss_cost, class-prior strength, boundary hazard weight) against cached posteriors — seconds per config, no GPU; best config by val macro-F1 subject to flicker ≤ baseline; writes `models/v1/decoder_config.json`.

- [ ] Sweep on val split only; document the chosen config and the val table (NN vs rule baseline per metric).
- [ ] Commit: `nn: offline eval + decoder sweep harness`

### Task 6: Test-split verdict + docs

**Files:** Modify `CLAUDE.md` + `lib/analyser/CLAUDE.md` (intent-level: the NN v1 exists offline, where artifacts live, what gates shipping); create `training/nn/CLAUDE.md` (package intent map).

- [ ] ONE evaluation of the frozen (model, decoder_config) on the TEST split. Report the full table vs the rule baseline: macro-F1, boundary-F1@±0.5 s/±2 s, drop recall/precision, flicker rate, per-class F1. No tuning after this run — if it loses, that is the honest result and the next iteration is a new versioned model, not a re-tuned test.
- [ ] Docs per CLAUDE.md policy (no hyperparameters in docs; point at `training/nn/`).
- [ ] Full `uv run pytest -m "not integration"` green (NN tests are unit-level; heavy steps live behind CLI scripts, not pytest).
- [ ] Commit: `nn v1: test-split verdict + docs`

---

## Self-Review Notes

- Task ordering is strict: 1→2→3→4→5→6; Tasks 4's priors and Task 1's splits both read segments.json but only train-split tracks (leakage guard stated in both).
- The plan trains on the ~460-track table; the eval-pipeline Task 2/3 must complete first (dependency, coordinator-sequenced).
- Artist-level leakage guarding is limited to deterministic track hashing (Raveform metadata may lack artist fields); the residual remix-leakage risk is accepted and documented in Task 6's docs step.
