# Live Downbeat Tracking — Design Spec

Status: designed 2026-07-27, following the v1 verdict's deployment-prerequisite finding. This is the last unbuilt component between the offline NN chain and runtime integration.

## Problem

The decoder commits at bar rate and the 200 ms transition-sync contract is delivered by snapping commits to downbeats (Raveform boundaries are downbeat-aligned by annotation policy; verified 97.4% at ±0.5 s on our corpus). Offline evaluation reads expert beat grids from the annotation bundle. Live, the engine has aubio beat instants with no bar phase — no downbeats, no grid. ANTICIPATION (blackout on the bar before a confident drop) has the same dependency.

## Design

**A small downbeat head + a deterministic bar-phase decoder, trained on our own expert grids, phase-labeling aubio's live beats.**

- **Training data:** the 1,387 validated tracks' Raveform beat CSVs (beat times + downbeat column — expert annotations, in-genre). Mel sidecars already exist for every track; targets are per-frame downbeat proximity (Gaussian-smeared, the proven recipe) plus per-beat phase labels derived from the grids.
- **Model:** a compact CRNN head (same family as SectionCRNN, smaller — target ≤300k params) on the same 16 s mel windows: per-frame downbeat activation. Reuses WindowDataset machinery, the training loop, calibration metrics, TensorBoard, ONNX export (dynamo=False), the CUDA pre-flight recipe. Trained on the same frozen splits (train only; eval-set and test quarantine inherited).
- **Decoder:** beats arrive from the production tracker (live) or the expert grid (parity tests). Each beat gets the downbeat activation aggregated near its instant; a 4-state bar-phase HMM (phase 1..4 at beat rate, cyclic transitions, small phase-flip penalty = hysteresis) decoded with fixed-lag Viterbi — same deterministic machinery as the section decoder, microseconds per beat. Both decoders are *exact within the set of paths that survive their commits*: the section decoder freezes a decision by pruning disagreeing trellis states to -inf, the bar-phase decoder by carrying the frontier's scores forward under the constraint. Neither is "exact Viterbi" in the unconstrained sense — a hard commit is by definition a restriction of the search — and the two agree closely in practice where they have been compared. Output: a live bar grid (downbeat instants + current phase), immutable at the configured lag. 3/4 or broken bars are out of scope for v1 (corpus is 4/4; a phase-confidence output lets the engine fall back to beat-snapping when the grid is unsure).
- **Amendment (2026-07-27): the phase state space is half-beat-resolvable.** Admitting the midpoint of each consecutive beat pair as a candidate instant doubles the cycle to eight positions and makes "the stream sits half a beat off the bar" a state the decoder can occupy and hold rather than an error it rounds away. Midpoints stay causal — the midpoint of beats *n* and *n+1* exists as soon as *n+1* arrives — so the fixed-lag discipline is unchanged, and the beat stream remains the anchor rather than being replaced by free-floating instants. `subdivision` is a decoder parameter, not a rebuild: a clean grid decodes at beat rate, where the extra candidates only add ways to be wrong.
- **Superseded (owner decisions #81/#133).** The acceptance gates below, and the v1 evaluation that read against them, bound to the aubio beat stream that madmom has since replaced. Those aubio-condition results are removed from the repo rather than carried as a dated record: the 0.85 gate is retired (above published *offline* SOTA on general music) and the recommended replacement is F1 ≥ 0.55 at a median of two phase flips per track or fewer. Successor numbers are committed in `docs/migration-evidence.md`; on madmom's stream the best candidate grid is subdivision 1. The machinery this spec describes is unchanged and beat-source-agnostic — it is what the next generation re-measures.

- **Runtime path (future integration):** aubio beats → phase labeler → bar grid → decoder commit quantization + ANTICIPATION scheduling. CPU ONNX, sidecar-cached in the sim per the determinism contract. This plan builds and validates the component OFFLINE; wiring into lib/ is the runtime-integration plan's job.

## Evaluation — LIVE CONDITION IS THE HEADLINE (owner directive: no offline assumptions)

Truth is always the expert grid. The INPUT condition is what distinguishes honest from flattering:

- **PRIMARY (deployment condition): aubio-beat-driven, corpus-wide.** Every clean track's sim report (`reports/*.npz.gz` cache) carries the production pipeline's own aubio beat instants — the exact stream the live engine produces, bit-for-bit. The evaluated system is: aubio beats + model activations → phase HMM → predicted downbeats, scored against expert downbeats at ±70 ms. This is the number the acceptance gates bind to.
- **Diagnostic upper bound: expert-beat-driven** — same pipeline fed expert beat instants. The gap between the two IS the aubio-degradation cost; report it per-track and in aggregate.
- **Aubio-vs-expert grid analysis first:** per-track beat alignment distribution (offset, jitter, missed/extra beats — aubio's documented sidechain dropouts). Tracks where aubio drifts hardest are the honest live condition and must be over-weighted in the qualitative review, not excluded.
- Phase accuracy + phase-flip rate per track (stability) — on the aubio-driven condition.
- The show ablation: section decoding on test tracks using the aubio-driven predicted grid vs the expert grid — boundary-F1/flicker delta. This is the go/no-go number for live bar-snapping.
- Degradation handling is a design requirement, not a hope: the phase HMM must COAST through missed beats (advance phase by tempo/time when aubio drops beats under sidechain compression) and tolerate aubio's timing jitter in the activation-aggregation window. Both behaviors get property tests.

**Known honest limitation (documented, not hidden):** the corpus is single tracks; live DJ sets have deck transitions and tempo rides the corpus cannot represent. Phase re-lock behavior after a simulated grid discontinuity gets a synthetic test; real-set validation is a runtime-phase task on the actual rig.

## Constraints inherited

Deterministic everywhere post-checkpoint; no test-split contact before the one-shot verdict; frozen-split extension rules; audio never committed; CLAUDE.md intent-level docs; stacked branch `downbeat_tracking` off `eval_pipeline` in its own worktree (parallel with v2; merges after).
