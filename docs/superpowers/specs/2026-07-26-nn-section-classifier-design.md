# Neural Section Classifier — Design Spec

Status: reviewed design (4-agent adversarial literature review, 2026-07-26). Supersedes the rule-cascade classifier as the section/intent brain. The owner delegated input/output design and authorized structured decoding if the review supported it; it did, unanimously.

## Goal

Classify EDM sections live (7 Raveform classes → LightIntent) with commits landing within ~200 ms of the audience-heard transition, deterministically reproducible in the fast sim, trained on the Raveform corpus.

## Division of labor (the core decision)

**The network is the acoustic model. The decoder is the committer. Neither does the other's job.**

- A CRNN integrates evidence *within* its window — including the look-ahead audio the audience hasn't heard. This is what resolves buildup fake-outs: it hears whether the drop actually lands.
- A fixed-lag Viterbi decoder (one state per canonical class) (HSMM-style: duration + transition priors) turns posteriors into *monotone, immutable, bar-quantized commits*. It owns stability, latency policy, and show-tunability.
- The engine's vote buffer, min-dwell, PEAK-promotion counter, and invalid-transition veto are **retired** — they were a hand-rolled degenerate HMM; the decoder subsumes them with parameters fit from data. The invalid-transition graph moves INTO the transition matrix as -inf entries (decoder picks the best *legal* path instead of silently holding).

Why not end-to-end smoothing (the owner's initial pick, reversed on review evidence):
1. Every shipping online MIR system keeps a structured decoder on neural activations (madmom DBN, BeatNet, BEAST; All-In-One retains the DBN for beats). Offline SOTA drops it only by buying 82–420 s receptive fields we cannot have live.
2. "Beat This!" (ISMIR 2024), the strongest no-decoder result, required training the net overconfident — which measurably *disrupts* decoder post-processing, making end-to-end a one-way door — and still lost the continuity metric (CMLt). Continuity is the light show's product metric.
3. Duration is our strongest prior (median drop ≈ 60.5 s ≈ 32 bars) and cannot live inside an 8–16 s window. Decode-time knobs (stickiness, dwell, drop sensitivity, asymmetric error costs) must be tunable per venue without a retrain.
4. Shibata et al. (ISMIR 2020): frame-wise neural labels without structure flicker in "musically unnatural" ways — the exact pathology the current rule stack was built to suppress.

## Input

- **Mel stream**: the pipeline's own 40-band aubio mel energies, `log1p`-compressed and normalized, mean-pooled 8 buffers → ~46 ms frames. Same computation live and in sim — parity is structural. 40 bands kept for v1 (revisit only if confusion matrices implicate sub-bass resolution).
- **Window: trailing ~16 s** of received audio (not 8 s). The decision frame sits ~8 s from the right edge — i.e. at audience-now — with near-symmetric left/right context. **Never read the window's last frames**: they have no right context and a cold backward-GRU state (the most miscalibrated position). The 8 s look-ahead is spent as *decode lag*, not as prediction horizon.
- **Embeddings**: demoted to an ablation arm. v1 trains mel-only; the YAMNet branch (already computed live) is added only if it beats mel-only on boundary-F@0.5 s on the held-out split. Stronger music-SSL embeddings (MuQ/MERT-class, distilled under the CPU budget) are the known biggest score lever — deferred, benchmarked later against the same protocol.

## Model

CRNN ≤ ~1M params (All-In-One reaches SOTA-class results at ~300K): conv front-end over the mel patch (frequency pooling) → biGRU across frames → two heads:
- **Label head** at ~10 Hz (pooled; SOTA labels at 8.33 Hz): calibrated softmax over the canonical class set (5 in v1 after the merges below: intro / buildup / breakdown / drop / outro). Trained with class weighting/focal loss for the 34:1 imbalance and a boundary-aware total-variation smoothness penalty (SongFormer practice). **Calibration is a first-class metric (ECE per class)** — the net must NOT be trained overconfident/self-smoothing, or the decoder is disrupted.
- **Boundary head** at frame rate: Gaussian-smeared boundary targets (σ ≈ 0.5 s — the annotation tolerance; targets deleted at merged-run joins). Feeds the decoder as its boundary/hazard observation. Boundaries decide *where*; segment-aggregated label posteriors decide *what*.

Inference: ONNX Runtime CPU, pinned version, `intra_op_num_threads=1`, one uncached golden-inference CI test with tolerance assertions; sim caches outputs as sidecars keyed by (model hash, track) per the locked determinism contract. Cadence ~100 ms over the sliding window; **per-frame posteriors are averaged across all overlapping windows covering that frame** (pyannote's fix for window flicker) before decoding.

## Decoder

Fixed-lag Viterbi (one state per canonical class) over the aggregated posteriors, decoding at **bar rate** (duration priors are meaningful in bars, and message passing is a few hundred ops/bar — microseconds):
- Transition matrix: fit from Raveform, with the structural facts hard (-inf illegal pairs; intro pure-initial; outros terminal). **Buildup→{breakdown, drop} set near-uniform** — the 608:521 corpus split is ~0.15 nats, uninformative; the look-ahead evidence decides the fork, not the prior.
- Duration model: per-class, fit from Raveform but **widened with heavy tails and a min-duration floor** — corpus medians are studio-master statistics; a DJ cutting a drop at 16 bars must not fight a peaked prior.
- Class-prior division (scaled likelihoods) handles residual imbalance at decode time — a runtime scalar sweepable against cached posteriors in seconds, not a retrain.
- Asymmetric error costs at the commit step (missing a drop ≠ spurious drop), adjustable per venue.
- **Freeze rule**: the decision for audience-time T is read once at fixed lag (≈ 5–6 s into the 8 s budget, leaving margin for quantization + inference cadence + actuation) and is immutable. v2: confidence-gated early commit (Narasimhan/Viola-style) once fixed-lag works.
- Commit quantization: next beat edge now; bar/phrase edges when Stage-2 downbeat tracking lands (Raveform boundaries are downbeat-aligned by annotation policy; DJ cue points cluster on 8/16-bar multiples). **The 200 ms requirement is met by grid-snapping, not by model localization** — no published system localizes section boundaries to ±200 ms from audio; the grid does it for us. Documented as such.
- Decode-lag budget assertion in the sim tests: lag + inference + send latency < look-ahead, with margin. Also: measure the physical rig's MIDI→light actuation latency — the only unbounded term in the budget.

## Labels & data

- Canonical mapping (drop `end`, altintro→intro, **altoutro→outro**, bridge→breakdown), adjacent runs merged; **cooldown merged into breakdown for v1** (positionally defined; undecidable in-window; revisit with track-elapsed-time features later).
- Leading-offset regions ([0, first section)) and trailing unlabeled audio: **loss-masked**, never mislabeled.
- Splits: track-level 70/15/15 with artist/remix leakage guarding; test untouched; the Eric Prydz reference track held out entirely.
- Train now on the ~460 clean tracks (augmentation: gain, EQ tilt, beat-grid phase jitter — the model trains on expert grids but ships on aubio's grid); the 1,423-track retrain is a drop-in comparison under the frozen eval protocol. Optional later: joint training with Harmonix via label mapping (evidence says it doesn't hurt).
- Window-offset augmentation + consistency penalty (same excerpt at multiple window positions) for position invariance.

## Evaluation (gates for shipping)

Against the label-aligned evaluator (eval-pipeline Task 3), on the held-out test split, the NN+decoder must beat the Stage-1 rule classifier on: macro-F1, boundary-F1@±0.5 s and ±2 s, drop recall/precision — **and on flicker rate** (state changes not matching ground-truth transitions per audience-minute), which joins the sim report as a first-class metric. Expect absolute numbers below the Raveform paper's offline 0.835 HR.5F — that is an 82 s-context offline ceiling, not our baseline.

## Phasing

1. **Data**: eval-pipeline Tasks 2–3 (running) + amendments: export per-track pooled-mel (+ YAMNet) sidecars during the batch sim; loss-mask offsets; cooldown merge in the label map.
2. **Model v1**: dataset builder → CRNN training on the 3070 → ONNX export → posterior sidecars → offline decode + eval vs baseline. No runtime integration until it wins offline.
3. **Decoder harness**: parameter sweeps over cached posteriors (no GPU in the loop) — stickiness, costs, lag.
4. **Runtime integration**: ONNX in the engine, decoder replaces vote/dwell/promotion stack, freeze rule, grid commits. ANTICIPATION: decoder sees high future drop mass → schedule blackout at the preceding bar edge.
5. **Later levers** (in expected-value order): stronger SSL embeddings under CPU budget; vocal head via Demucs distillation; confidence-gated commit lag; Harmonix joint training; 8 s→phrase-edge look-ahead extension with dmx-enttec-node.
