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

| Module | Role |
|---|---|
| `__init__.py` | puts `training/` on `sys.path` once, so the label vocabulary, mel geometry and artist parser are the corpus's own definitions rather than copies that can drift; also sets the CUDA determinism env var before anything imports torch |
| `dataset.py` | splits + windowed, loss-masked training items; owns the split rule and the benchmark exclusions |
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
| `models/v1/model.onnx` | the exported graph -- the interface every inference goes through |
| `models/v1/priors.json` | the fitted structural priors |
| `posteriors/*.npz` | one posterior sidecar per track, keyed on model *and* window geometry |
| `models/v1/decoder_config.json` | the chosen decoder config and the search that chose it |
| `models/v1/eval_val.json` | the tuned verdict |
| `models/v1/eval_test.json` | the selection-clean verdict -- the test split is read once |

## Rules that must not be broken

- **`splits.json` is frozen and is never regenerated implicitly.** The assignment hashes `(seed, track id)` so a growing corpus only ever *adds*; a missing splits file is an error, not a cue to rebuild. The frozen benchmark and every track sharing an artist with it stay out of all splits.
- **The test split is read once.** `sweep.py` refuses it outright; `evaluate_v1.py` accepts it because the verdict needs it and defaults to val. After a test read, the response to a bad result is a new versioned model with its own single read -- never a re-tune measured against the same split.
- **The verdict imports the baseline's metric functions rather than reimplementing them.** Exactly two things differ between the two columns, and both are named in the module docstring. A third divergence is a bug.
- **Everything after the checkpoint is exact.** Inference is single-threaded through one pinned session, sidecars are written with a fixed archive layout, and the decoder is pure numpy -- so a report is byte-stable given its sidecars. Training itself is seeded and bitwise reproducible in a fresh process.
- **A checkpoint and a sidecar each describe their own geometry, and consumers re-check it.** A pooling-factor or window change loads cleanly and decodes at the wrong rate, so the mismatch must fail loudly instead.
- **Torch is an optional extra** (`uv sync --extra training`) and stays one. The default dependency set does not move, the live pipeline gains no imports, and the decode-and-score path needs no torch at all.
