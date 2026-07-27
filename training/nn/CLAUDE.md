# Neural Section Classifier (`training/nn/`)

The offline half of the replacement for the hand-tuned rule classifier: a CRNN over the runtime's own mel stream, decoded by a fixed-lag Viterbi that owns stability and latency policy. It is trained, exported, decoded and scored entirely offline. **Nothing in this package runs in a show, and `lib/` and `simulate/` import nothing from it.**

This file is the map: what each module is for, where its output lands, and which rules must not be broken. The *reasoning* -- why the boundary head gets an extra mask, why the exporter is the deprecated one, why the comparison imports the baseline's own metric functions, what gates deployment -- lives in the root `CLAUDE.md` under "Neural section classifier", and the design spec is `docs/superpowers/specs/2026-07-26-nn-section-classifier-design.md`. Do not restate either here.

---

## The chain

```
clean_manifest.csv + mel sidecars + annotations
        |  dataset.py            -> splits.json, windowed loss-masked training set
        v
    model.py (SectionCRNN: label head + boundary head)
        |  train.py              -> checkpoints, training_report.json, TensorBoard
        v
    export_onnx.py               -> model.onnx (the interface; time axis stays symbolic)
        |  infer.py              -> one posterior sidecar per track
        v
    priors.py (corpus bar runs)  -> the structural graph, duration floors, hazards
        |  decoder.py            -> immutable per-bar class decisions
        v
    evaluate_v1.py / sweep.py    -> decoder_config.json, eval_val.json, eval_test.json
```

Each stage reads the previous stage's artifact off disk. That is deliberate: the expensive stages run once and every cheap stage downstream is re-runnable in seconds.

A **second chain is under construction beside it**: a downbeat head and a bar-phase decoder, because the section decoder commits at bar rate and live audio arrives with beats but no bar. It shares the sidecars, the splits and the window machinery, and it is the deployment prerequisite the v1 verdict named. Spec: `docs/superpowers/specs/2026-07-27-live-downbeat-tracking-design.md`; the live (aubio-driven) condition is the one its gates bind to.

```
downbeat_dataset.py (expert beat grids)  ->  downbeat_model.py + downbeat_train.py
        |  downbeat_infer.py             ->  downbeat.onnx, one activation sidecar per val/test track,
        v                                    with per-beat evidence for BOTH input conditions
    downbeat_decoder.py                  ->  an immutable bar grid: downbeat instants, phase, confidence
        |  evaluate_downbeat.py          ->  what aubio's stream can reach, what the decoder gets,
        v                                    and what a predicted grid costs the section decode
```

The decoder's two input conditions are the whole point of the evaluation design: `aubio` is the beat stream the live engine actually produces (lifted from the cached sim reports) and is what the gates bind to; `expert` is the annotator's grid and is the diagnostic upper bound. The gap between them is the aubio-degradation cost, and it is reported, never assumed away.

| Module | Role |
|---|---|
| `__init__.py` | puts `training/` on `sys.path` once, so the label vocabulary, mel geometry and artist parser are the corpus's own definitions rather than copies that can drift; also sets the CUDA determinism env var before anything imports torch |
| `dataset.py` | splits + windowed, loss-masked training items; owns the split rule and the benchmark exclusions |
| `downbeat_dataset.py` | the *second* head's supervision: expert beat grids -> per-frame downbeat targets and per-beat bar-phase labels, on the same windows |
| `downbeat_model.py` | `DownbeatCRNN` -- one head, a per-frame downbeat logit; the section model's conv front end at half the capacity |
| `downbeat_train.py` | the downbeat head's training loop; imports the section head's determinism contract, loader policy and calibration metrics rather than restating them, and scores itself on peak F1 at the +-70 ms tolerance instead of on frames |
| `downbeat_infer.py` | the downbeat head's inference artifact: checkpoint -> ONNX, then a sliding-window pass per track into one sidecar carrying the frame-rate activation *and* the evidence aggregated onto each beat instant of both input conditions; also owns the measurement that chose the aggregation window |
| `downbeat_decoder.py` | the bar-phase HMM: a cyclic position within the bar, committed at a fixed lag and never revised, coasting through aubio's dropouts and reporting how sure it is; owns the candidate grid and the activation-aggregation window, because in the runtime putting an activation onto a beat stream *is* decode-path work; pure numpy, because the runtime will import this one |
| `evaluate_downbeat.py` | the downbeat verdict: the aubio-vs-expert grid analysis that says what the live condition *can* reach and why the rest is out of reach, the val lever sweep, the one test read against a config frozen and hashed beforehand, and the show ablation that decodes sections on the predicted grid against the expert one |
| `downbeat_baselines.py` | the numbers a downbeat F1 has to be read against -- above all a *phase-blind* perfect beat detector, because beating noise only proves the head found beats -- plus the calibration metric's own floor and the bar-phase histogram that says which decoder knob matters |
| `compare_runs.py` | the determinism proof for either head: walks every logged number of two runs, not just the weight hash, because equal endpoints do not mean equal paths |
| `model.py` | `SectionCRNN` -- two heads, label logits on the pooled grid and boundary logits at frame rate |
| `train.py` | the training loop, its determinism contract, calibration metrics and TensorBoard |
| `export_onnx.py` | checkpoint -> ONNX, plus the single pinned single-threaded session every consumer must use |
| `infer.py` | sliding-window inference -> byte-reproducible posterior sidecars |
| `priors.py` | fits the decoder's structural priors from corpus bar runs |
| `decoder.py` | fixed-lag Viterbi over an explicit-duration HSMM; commits per bar and never revises |
| `evaluate_v1.py` | decode -> per-beat classes -> the side-by-side verdict against the rule classifier |
| `sweep.py` | the decoder parameter search under the selection constraint |

Tests are `tests/test_nn_*.py` and are unit-level throughout: the heavy steps (training, inference, the sweep) live behind CLI entry points, not behind pytest.

## Where the artifacts live

All of it is gitignored, under the data directory (`training/data/raveform/` by default). Only code and tests are committed.

| Path | What |
|---|---|
| `splits.json` | the frozen train/val/test assignment |
| `features/*.npz` | pooled log-mel sidecars (written by the eval pipeline, read here) |
| `models/v1/<run>/` | training checkpoints and per-run reports |
| `models/downbeat_v1/<run>/` | the downbeat head's checkpoints and per-run reports; same layout, same TensorBoard logdir |
| `models/downbeat_v1/downbeat.onnx` | the exported downbeat graph -- the interface every downbeat inference goes through |
| `downbeat_posteriors/*.npz` | one downbeat sidecar per val/test track: the frame-rate activation and the per-beat evidence for both input conditions, keyed on model *and* geometry |
| `models/downbeat_v1/downbeat_alignment_val.json` | the corpus aubio-vs-expert grid analysis: per-track offset and jitter, and every annotated downbeat labelled with why it is or is not reachable |
| `models/downbeat_v1/downbeat_sweep_val.json` | the val lever sweep, one row per (condition, config, refinement) with its own fingerprint |
| `models/downbeat_v1/downbeat_decoder_config.json` | the chosen bar-phase config, frozen and hashed *before* the test read |
| `models/downbeat_v1/downbeat_eval_{val,test}.json` | the verdict, both input conditions, plus the show ablation |
| `models/v1/model.onnx` | the exported graph -- the interface every inference goes through |
| `models/v1/priors.json` | the fitted structural priors |
| `posteriors/*.npz` | one posterior sidecar per track, keyed on model *and* window geometry |
| `models/v1/decoder_config.json` | the chosen decoder config and the search that chose it |
| `models/v1/eval_val.json` | the tuned verdict |
| `models/v1/eval_test.json` | the selection-clean verdict -- the test split is read once |

## Rules that must not be broken

- **`splits.json` is frozen and is never regenerated implicitly.** The assignment hashes `(seed, track id)` so a growing corpus only ever *adds*; a missing splits file is an error, not a cue to rebuild. The frozen benchmark and every track sharing an artist with it stay out of all splits.
- **The test split is read once, and *anything that reads truth to choose a value* is a tuning read.** `sweep.py` refuses it outright; `evaluate_v1.py` accepts it because the verdict needs it and defaults to val. The rule is about what a mode *does*, not what it is called: a diagnostic that reads annotations to fix a constant is tuning, and belongs to val. Guard such a mode on split **membership**, not on the flag that usually selects it — an explicit id list walks straight past a flag-level check. After a test read, the response to a bad result is a new versioned model with its own single read -- never a re-tune measured against the same split. **The choice must provably predate the read**: the downbeat verdict refuses to score test at all unless a frozen config file already exists on disk, and it records that file's own hash beside the numbers, so "we picked this afterwards" is a claim the artifact can refute.
- **The verdict imports the baseline's metric functions rather than reimplementing them.** Exactly two things differ between the two columns, and both are named in the module docstring. A third divergence is a bug.
- **Everything after the checkpoint is exact.** Inference is single-threaded through one pinned session, sidecars are written with a fixed archive layout, and the decoder is pure numpy -- so a report is byte-stable given its sidecars. Training itself is seeded and bitwise reproducible in a fresh process.
- **A checkpoint and a sidecar each describe their own geometry, and consumers re-check it.** A pooling-factor or window change loads cleanly and decodes at the wrong rate, so the mismatch must fail loudly instead.
- **A metric is reported with the baseline it must be read against.** Every trained checkpoint on disk is keyed by parameter *names*, so a refactor that renames one is a silent break -- pinned by a frozen key list, in a test that needs no corpus. The same principle applies to scores: a downbeat F1 is quoted against a phase-blind beat detector, never against noise, and a calibration error is quoted with its own floor. A number without its null is a decoration.
- **A sidecar generated for the test split carries inputs, never truth.** Beat *instants* define an input condition and belong in the file; the bar *phase* is what the verdict scores against and stays in the annotations. Generating test-split sidecars early is safe precisely because of that line.
- **Torch is an optional extra** (`uv sync --extra training`) and stays one. The default dependency set does not move, the live pipeline gains no imports, and the decode-and-score path needs no torch at all — the bar-phase decoder is the module that will be imported into `lib/`, so its independence from torch is pinned by a test that imports it in a torch-free process rather than left to review.
- **A commit and its look-ahead are one design, and the lag is counted in *candidates*.** The bar-phase decoder cannot act on evidence outside its lag window, so its hysteresis knob and its lag are a *pair*: a penalty that reads as sensible hysteresis at one lag reads as "never change your mind" at a shorter one. The trap is that densifying the candidate grid *shortens the wall-clock look-ahead at a fixed lag*, so a sweep that varies the grid without compensating is measuring the lag and calling it the grid. Hold wall-clock look-ahead constant when comparing grids, and sweep penalty, lag and grid together.
- **The decoder can only place a downbeat where its candidate grid has an instant, and it emits one per cycle whether the music agrees or not.** Two ceilings follow, and neither is the model's. *Coverage*: aubio's beats are half a beat off the music often enough that a beat-rate grid caps the live condition well below the gate, which is why the state space is half-beat-resolved. *Rate*: a cyclic decoder that does not flip emits exactly one downbeat per cycle of candidates, so a candidate stream that runs fast produces a bar grid that runs fast, and the surplus downbeats are false positives no phase model can retract. Together they bound F1 at `2 x coverage / (1 + rate)` for the stable decode the stability gate demands. Measure both before attributing a shortfall to the model or the decoder — the evaluator does, per track and corpus-wide.
- **A tempo estimate that feeds its own input has to be able to notice it is wrong.** Coasting records a period *derived* from the running estimate, so a bad estimate is self-sustaining: it was measured filling whole tracks with phantom candidates after aubio's double-tempo warmup seeded it short. The rule that breaks the loop is that coasting every arrival is not a dropout — a dropout is a hole in an otherwise arriving stream — so the estimate is dropped and re-seeded from what is actually observed. Any future recursive estimator on this path needs its own version of that check.
- **Stability is reported in the unit the grid is judged in, not the unit the decoder counts in.** At a half-beat candidate grid a phase-flip count also counts every beat aubio inserts or drops, so a perfectly steady bar grid over a noisy stream can read as hundreds of flips. What is gated is the bar position advancing by one beat between consecutive *real* beats; what is reported beside it is the emitted grid's own interval regularity. A stability number that moves when the input's beat count moves is measuring the input.
