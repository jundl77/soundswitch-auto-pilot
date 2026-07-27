# Live Downbeat Tracking — Design Spec

Status: designed 2026-07-27, following the v1 verdict's deployment-prerequisite finding. This is the last unbuilt component between the offline NN chain and runtime integration.

## Problem

The decoder commits at bar rate and the 200 ms transition-sync contract is delivered by snapping commits to downbeats (Raveform boundaries are downbeat-aligned by annotation policy; verified 97.4% at ±0.5 s on our corpus). Offline evaluation reads expert beat grids from the annotation bundle. Live, the engine has aubio beat instants with no bar phase — no downbeats, no grid. ANTICIPATION (blackout on the bar before a confident drop) has the same dependency.

## Design

**A small downbeat head + a deterministic bar-phase decoder, trained on our own expert grids, phase-labeling aubio's live beats.**

- **Training data:** the 1,387 validated tracks' Raveform beat CSVs (beat times + downbeat column — expert annotations, in-genre). Mel sidecars already exist for every track; targets are per-frame downbeat proximity (Gaussian-smeared, the proven recipe) plus per-beat phase labels derived from the grids.
- **Model:** a compact CRNN head (same family as SectionCRNN, smaller — target ≤300k params) on the same 16 s mel windows: per-frame downbeat activation. Reuses WindowDataset machinery, the training loop, calibration metrics, TensorBoard, ONNX export (dynamo=False), the CUDA pre-flight recipe. Trained on the same frozen splits (train only; eval-set and test quarantine inherited).
- **Decoder:** beats arrive from aubio (production) or the expert grid (parity tests). Each beat gets the downbeat activation aggregated near its instant; a 4-state bar-phase HMM (phase 1..4 at beat rate, cyclic transitions, small phase-flip penalty = hysteresis) decoded with exact fixed-lag Viterbi — same deterministic machinery as the section decoder, microseconds per beat. Output: a live bar grid (downbeat instants + current phase), immutable at the configured lag. 3/4 or broken bars are out of scope for v1 (corpus is 4/4; a phase-confidence output lets the engine fall back to beat-snapping when the grid is unsure).
- **Runtime path (future integration):** aubio beats → phase labeler → bar grid → decoder commit quantization + ANTICIPATION scheduling. CPU ONNX, sidecar-cached in the sim per the determinism contract. This plan builds and validates the component OFFLINE; wiring into lib/ is the runtime-integration plan's job.

## Evaluation (held-out, expert grids as truth)

- Downbeat F1 at ±70 ms (the field's standard tolerance) on the test split — target: ≥0.85 in-genre (the Raveform paper's offline baseline hits 0.965; we accept an online discount but need enough for bar-snapping to beat beat-snapping).
- Phase accuracy (fraction of beats assigned the correct 1..4) and phase-flip rate per track (the stability metric — flips are what would make the lights re-anchor mid-song).
- Ablation that matters for the show: section-boundary snap error using predicted grid vs expert grid — does the predicted grid preserve the decoder's boundary-F1?
- Robustness probe: aubio-beat-driven evaluation (not just expert-beat-driven) on tracks where aubio's grid drifts from the expert grid — the honest live condition.

## Constraints inherited

Deterministic everywhere post-checkpoint; no test-split contact before the one-shot verdict; frozen-split extension rules; audio never committed; CLAUDE.md intent-level docs; stacked branch `downbeat_tracking` off `eval_pipeline` in its own worktree (parallel with v2; merges after).
