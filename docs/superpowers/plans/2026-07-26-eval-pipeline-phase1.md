# Label-Aligned Evaluation Pipeline (Phase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the partially-downloaded Raveform corpus (~459 tracks and growing) into a clean, label-aligned per-beat training table, and measure the CURRENT classifier against expert labels — replacing the plumbing-only evaluator with musical ground truth. Phase 1 ends with baseline numbers; the fitter/model itself is a separate decision the owner makes after seeing them.

**Architecture:** Three scripts in `training/`, all runnable incrementally as more audio lands. (1) A cleanliness gate produces `clean_manifest.csv` from whatever audio exists. (2) A batch runner pipes each clean track through the UNMODIFIED fast sim (identical production pipeline) in parallel worker processes, joins each beat row to its canonical Raveform label by song time, and appends to a training table. (3) An evaluation harness scores intent timelines against labels (time-weighted confusion, macro-F1, boundary-F1) per song and corpus-wide. Everything is deterministic and re-runnable; new tracks only add rows.

**Tech Stack:** Python 3.11 (uv venv), stdlib + numpy (no pandas/pyarrow — the table is a flat CSV.gz), multiprocessing, existing `simulate.runner.run_fast_simulation` + `training/raveform_manifest.py` helpers.

## Global Constraints

- Branch: `eval_pipeline`, worktree `C:\Users\Julian\Projects\soundswitch-eval-worktree` (stacked on `stage1_classifier_measurement_fixes` + `raveform_data_pipeline` — both pushed, neither merged to master yet; this PR will re-target after they land). Never commit to master; do not push until the coordinator says so.
- DATA lives at the main repo absolute path `C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform\` (gitignored). Scripts committed; data never.
- **A downloader/supervisor is LIVE, writing into `audio/`.** Only touch `*.mp3` files whose mtime is > 60 s old; never touch `*.part`, `*.webm`, supervisor/downloader files, or any process.
- **Decode-cache discipline:** the sim writes `<mp3>.<samplerate>.npy` beside the audio (~7.7× mp3 size — the full corpus would be ~95 GiB). Batch workers MUST delete the `.npy` for a track immediately after its sim run. Verify no cache accumulation after the batch run.
- The pipeline under test is read-only: nothing in `lib/` or `simulate/` may change in this plan. If something there blocks you, report it — don't patch it.
- Label semantics (binding, from the validated corpus): canonical mapping = drop `end` sentinel, `altintro`→`intro`, `bridge`→`breakdown`; merge adjacent same-label runs AFTER mapping; audio in `[0, sections[0].start)` is UNLABELED (leading offset up to 35.9 s on 190 tracks) — exclude those beats from training/eval rows; one track (`1020.c1VBubZ2w3M`) has a negative-length final section (clamp); `total_sec` in manifest.csv is ms-rounded (no bit-equality assumptions).
- Beat rows are song-time; intent blocks are audience-time — realign with `metrics.look_ahead_sec` (in the report since commit 7d10bca).
- `uv run pytest -m "not integration"` green at every task end; commit per task.

---

### Task 1: Cleanliness gate → `clean_manifest.csv`

**Files:**
- Create: `training/build_clean_manifest.py`
- Test: `tests/test_build_clean_manifest.py`
- Data out: `<data-dir>/clean_manifest.csv`

**Interfaces:**
- Produces: `clean_manifest.csv` with header `track_id,youtube_id,mp3_path,ffprobe_duration_sec,decoded_duration_sec,annotation_duration_sec,status,detail` where status ∈ {ok, duration_mismatch, corrupt} (as built: decoded length measured from the ffmpeg pass guards against header-lying truncation; `detail` carries the rejection reason). Only `ok` rows feed Task 2; Task 2 reads `decoded_duration_sec` (equals header duration within tolerance on ok rows). Reuses `manifest.csv` + segments.json via `training/raveform_manifest.py` / `raveform_fetch_annotations.py` helpers — no re-parsing from scratch.

- [ ] For each `manifest.csv` row with an existing `audio/<youtube_id>.mp3` older than 60 s: run `ffmpeg -v error -i <file> -f null -` (empty stderr = decodes) and `ffprobe` duration vs the annotation record's `duration` within max(±10 s, ±3%). Classify ok / duration_mismatch / corrupt. Tracks not on disk are simply absent from the output (the supervisor is still fetching them) — print counts: present-ok / mismatch / corrupt / not-yet-downloaded.
- [ ] Parallelize the ffmpeg checks with a small process pool (they're I/O+subprocess bound; 8 workers). Deterministic output ordering (sort by track_id).
- [ ] Tests (tmp_path, no real corpus): a stub "mp3" that ffmpeg rejects → corrupt; classification thresholds at the tolerance boundary; mtime-recency exclusion. Mock subprocess where needed but test the classification logic for real.
- [ ] Run against the live data dir; paste the summary into your report. Expect ~450+ ok; investigate (don't hide) anything corrupt.
- [ ] Commit: `eval: cleanliness gate over the partial raveform corpus`

### Task 2: Batch sim → label-aligned training table

**Files:**
- Create: `training/build_training_table.py`
- Test: `tests/test_training_table_labels.py` (label-join logic only)
- Data out: `<data-dir>/training_table.csv.gz` + `<data-dir>/training_table.meta.json`

**Interfaces:**
- Consumes: `clean_manifest.csv` (ok rows), segments.json, `simulate.runner.run_fast_simulation`, `simulate.fake_audio_client.FileAudioClient`.
- Produces: one row per beat per track: `track_id, youtube_id, t_song, bpm, onset_density, kick_strength, kick_known, centroid_trend, sub_bass_ratio, rms, intent_at_beat, label_canonical, label_raw, label_v1, bar_position_unknown` — `label_v1` = canonical further merged per the NN spec: cooldown→breakdown, altoutro→outro (5-class space: intro/buildup/breakdown/drop/outro) — plus per-song z-scored copies of the continuous features (`*_z`, computed per track; mix-invariance for the future fitter). `intent_at_beat` = the committed intent block covering `t_song` after shifting blocks back by `metrics.look_ahead_sec`. `meta.json`: row/track counts, class histogram, git SHA of the pipeline, build timestamp.

- [ ] Worker function: run one track through `run_fast_simulation(FileAudioClient(...))` (exactly like `tests/test_simulation.py` does), take `event_buffer.to_report(...)`, THEN export the NN feature sidecar (next bullet) while the decode cache still exists, THEN delete the track's `.npy` cache, return the joined rows.
- [ ] **NN feature sidecar (spec: 2026-07-26-nn-section-classifier-design.md):** per track write `<data-dir>/features/<youtube_id>.npz` containing `mel` = float32 array [n_frames, 40] of `log1p`(aubio mel energies) mean-pooled over 8 consecutive 256-sample buffers (~46 ms/frame), plus `frame_sec` (scalar hop in seconds) and `t0` (first frame's song time). Build the aubio objects (pvoc win 1024, filterbank 40 slaney bands, hop 256) with the SAME constructor parameters as `MusicAnalyser._reset_state` — `lib/` stays untouched; instead add a PARITY unit test: instantiate a real `MusicAnalyser`, feed 50 random buffers, assert the exporter's per-buffer energies equal `_compute_mel_energies` outputs exactly. Read audio from the `.npy` decode cache the sim just created (same decoded samples — parity by construction). Label join: beat `t` (song-time) → canonical merged label run covering it; beats before `sections[0].start` → dropped (count them); beats past the last section end → dropped.
- [ ] `kick_known` = `rms >= 0.005` per the documented RMS-gate convention (cite `lib/analyser/CLAUDE.md`; import the constant if exposed, else document the coupling in one comment).
- [ ] ProcessPoolExecutor over ok-tracks (workers = cpu_count − 2), progress line every 10 tracks, per-track failures logged and skipped (never abort the batch), deterministic final ordering.
- [ ] `--limit N` for smoke runs. **Report cache (MANDATORY, owner speed directive):** every track's sim report is persisted to `<data-dir>/reports/<youtube_id>.json.gz` keyed by (pipeline git SHA, mp3 size+mtime) recorded inside it; on rebuild, tracks with a current cached report skip the sim AND the decode entirely — table rebuilds after the first pass must touch only new/changed tracks. `--force` re-runs everything. Print cache-hit/miss counts.
- [ ] Tests for the join: synthetic sections incl. leading offset, sentinel-adjacent merge, look-ahead shift of intent blocks, negative-length clamp.
- [ ] Full run over all ok tracks. Paste: track count, row count, class histogram, dropped-beat counts (leading/trailing), wall time, confirmation that `audio/` contains zero `.npy` afterward.
- [ ] Commit: `eval: batch sim runner + label-aligned training table`

### Task 3: The real evaluator — score intents against labels

**Files:**
- Create: `training/evaluate_against_labels.py`
- Test: `tests/test_label_evaluation.py` (metric math on synthetic timelines)
- Data out: `<data-dir>/baseline_eval.json` + printed report

**Interfaces:**
- Consumes: `training_table.csv.gz`.
- Produces: per-song and corpus-wide: (a) time-weighted confusion matrix intent × canonical label (full 6×7); (b) headline macro-F1 over the mapped space {DROP,PEAK}→drop, BUILDUP→buildup, BREAKDOWN→breakdown, ATMOSPHERIC→quiet(intro/outro/altoutro), GROOVE→cooldown (mapping lives in ONE dict at the top — the owner will iterate on it); (c) boundary-F1: intent-change instants vs label-boundary instants at ±2 s and ±4 s tolerances, with per-boundary-type breakdown (→drop boundaries reported separately — they're the show-critical ones); (d) drop recall/precision specifically (is the system in DROP/PEAK while the label is drop); (e) worst-15 songs by headline F1, listed with per-song confusion rows.

- [ ] Metric math unit-tested on synthetic timelines (perfect match → 1.0; constant-single-intent → known degenerate scores; boundary tolerance edges).
- [ ] **Flicker rate** (NN spec requirement, applies to the rule baseline too): intent changes NOT within tolerance of any ground-truth boundary, per audience-minute — reported per song and corpus-wide alongside the F1s. Continuity is the product metric.
- [ ] Report the headline metrics in BOTH the 5-class `label_v1` space (primary — this is the NN's target space) and the fuller canonical space (diagnostic).
- [ ] Run on the Task 2 table; paste the full corpus report AND the worst-15 list into your task report. These are THE baseline numbers for the fitter discussion — present them without spin: the current classifier was calibrated on one track, and this is its first contact with 450.
- [ ] Commit: `eval: label-aligned scoring — baseline vs raveform ground truth`

---

## STOP after Task 3

Phase 1 ends here by design. The fitter (threshold refit vs GBDT+HSMM vs neural) is an explicit owner decision to be made looking at the baseline numbers. Do not begin any model training.
