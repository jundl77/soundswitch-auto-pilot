# SoundSwitch Auto Pilot

Intelligent DJ lighting automation system that synchronizes stage effects to live music in real time. Analyzes audio â†’ detects beats/sections/energy â†’ classifies `LightIntent` â†’ controls SoundSwitch lighting via MIDI and OS2L protocol.

---

## CLAUDE.md Policy

**CLAUDE.md documents intent and architecture, not code.** This applies to every CLAUDE.md in this repo â€” root or subdirectory.

- Do not replicate threshold values, function signatures, or internal variable names.
- Do not duplicate content that is already expressed in code. Point to the file instead.
- CLAUDE.md sits one layer above the code. It explains *why* things work the way they do, not *what* specific values are set to.
- Every PR must update CLAUDE.md to reflect any architectural, interface, or behavioural changes â€” but only at the intent/meta level.

**CLAUDE.md is the agent's source of truth.** Any analysis ideas, design decisions, or specifications must be recorded here â€” not just in code or in PR descriptions. An agent must be able to understand what this system does, why it is designed the way it is, and what direction it is heading by reading CLAUDE.md alone, without scanning all source files. Keep it current at all times.

---

## Development Workflow

**All changes must go through a pull request.** Never commit directly to `master`.

Before opening a PR, all tests must pass:

```bash
# Fast unit tests (run these frequently during development)
uv run pytest -m "not integration"

# Full suite including unit + integration tests
uv run pytest

# Run a single test file
uv run pytest tests/test_delayed_command_queue.py -v
```

The integration tests in `tests/test_simulation.py` run real, expert-labelled music through the full pipeline (identical code path to production). One track pins the *mechanism* — command-timing exactness, flush behaviour, report duration, speed, byte-identical determinism, and the plumbing evaluator's verdict; three more go through `training/run_eval_set.py` in compare mode and pin the *behaviour* — per-track report checksums and label-aligned scores against the committed baseline. If they fail, the pipeline is broken or its output moved. There is one simulation mode — real audio files — paced either sped-up (default) or real-time (`--ui`).

**These tests run from a fresh clone with no downloads.** Both things they read — the ten eval-set mp3s and the labels they are scored against — are committed (see The benchmark). If either is ever pruned they fail with one line naming both places to get it back, a deliberate failure rather than a skip, because a benchmark nobody notices has been skipped is not a benchmark. Everything else still needs the gitignored corpus; it does not follow `git worktree add`, so a linked worktree finds the main checkout's copy automatically, and `$RAVEFORM_DATA_DIR` overrides.

### Testing philosophy

- **Coverage over completeness**: aim for broad, confident coverage of critical logic â€” not 100% line coverage. Tests should catch real regressions, not just pad numbers.
- **Test the logic, not the wiring**: unit tests target pure functions and isolated methods. Integration tests verify the full pipeline assembles correctly.
- **Missing deps**: if a package is declared in `pyproject.toml` but absent from the venv, run `uv sync --extra dev --extra visualizer` â€” do not mock it.
- **Every PR must pass `uv run pytest`** (the full suite, not just unit tests) before merge.

---

## What It Does

1. Reads audio from a microphone/line input
2. Extracts musical features via Aubio (BPM, onsets, notes, mel filterbank energies)
3. Detects musical section changes via a YAMNet TensorFlow embedding + cosine similarity outlier detection
4. Classifies audio energy as a `LightIntent` (ATMOSPHERIC / BREAKDOWN / GROOVE / BUILDUP / DROP / PEAK)
5. Selects and sends MIDI lighting effects to SoundSwitch based on intent; also sends OS2L beat events to VirtualDJ and DMX overlays via UDP

---

## Architecture

```
PyAudio â†’ MusicAnalyser (Aubio DSP) â†’ LightEngine (IMusicAnalyserHandler)
                   â†“                          â†“               â†“
        YamnetChangeDetector          EffectController    MIDI / OS2L / Overlay
                                             â†‘
                                       LightIntent
                              (BPM + onset density + sub-bass)
```

### Key Files

| Path | Role |
|---|---|
| `auto_pilot` | CLI entry point (`run MIDI_PORT`, `list`, `simulate`) |
| `lib/main.py` | `SoundSwitchAutoPilot` â€” async event loop, 100 ms / 1 s / 10 s callbacks |
| `lib/clock.py` | `Clock` abstraction â€” `SystemClock` (prod default) vs `VirtualClock` (fast sim); every time-based component takes an injectable clock |
| `lib/audio_config.py` | Canonical `SAMPLE_RATE` / `BUFFER_SIZE` â€” single source for live pipeline, simulation, and virtual-clock timing math |
| `lib/analyser/music_analyser.py` | `MusicAnalyser` â€” per-buffer DSP, beat/onset/note events, YAMNet trigger |
| `lib/analyser/yamnet_change_detector.py` | `YamnetChangeDetector` â€” TF Hub YAMNet embeddings, MAD outlier detection |
| `lib/analyser/CLAUDE.md` | Analysis pipeline detail: features, classification design, evaluation strategy |
| `lib/engine/light_engine.py` | `LightEngine` â€” routes DSP events â†’ intent â†’ MIDI / OS2L / overlay commands; all tuning constants live here |
| `lib/engine/effect_controller.py` | `EffectController` â€” maps `LightIntent` â†’ non-repetitive random MIDI channel selection |
| `lib/engine/effect_definitions.py` | `LightIntent` enum + `INTENT_EFFECTS` mapping (the single place to change intentâ†’MIDI routing) |
| `lib/engine/event_buffer.py` | Thread-safe beat/effect/intent store; read by Dash visualizer every 100 ms |
| `lib/clients/midi_client.py` | MIDI note-on/off to SoundSwitch; 90+ channels, delayed deactivation |
| `lib/clients/os2l_client.py` | zeroconf discovery of VirtualDJ; bidirectional OS2L JSON |
| `lib/clients/pyaudio_client.py` | Mono 44.1 kHz audio input (and optional debug output passthrough) |
| `lib/clients/overlay_client.py` | UDP binary DMX overlay (hardcoded IP â€” must match venue) |
| `simulate/visualizer_app.py` | Dash real-time visualizer: timeline, intent-based stage simulation, metrics |
| `simulate/runner.py` | Simulation runner â€” stub clients, full pipeline; virtual-clock fast mode (default) or real-time pacing for the live UI |
| `simulate/cli.py` | `auto_pilot simulate file|realtime` subcommands |
| `training/inspect_report.py` | Report inspector â€” per-10s feature/intent bins + intent timeline; the tool for checking a show against a track's structure |
| `training/run_eval_set.py` | The benchmark â€” the frozen eval set through the sim, scored against its labels; cuts and enforces `training/eval_set_baseline.json` |

---

## LightIntent System

`LightIntent` is the semantic bridge between audio analysis and lighting output. It lives in `lib/engine/effect_definitions.py`.

Six intents map to structural moments in an EDM track:

| Intent | Musical moment | MIDI pool |
|---|---|---|
| ATMOSPHERIC | Silence, intro, full breakdown, outro â€” no beats | BANK_2A/B/C |
| BREAKDOWN | Melodic, stripped, emotional â€” beats present but sparse | BANK_2C/D/E |
| GROOVE | Steady dance-floor mid-energy â€” main verse/groove loop | BANK_2F/G/H |
| BUILDUP | Rising tension pre-drop â€” onset density climbing | BANK_1A/B/C |
| DROP | Maximum impact â€” bass, kick, full arrangement | BANK_1D/E + STROBE |
| PEAK | Sustained maximum energy after the drop (engine promotion, see below) | BANK_1F/G/H |

For the specific thresholds and tuning constants that drive classification, see `lib/engine/light_engine.py` and `lib/analyser/CLAUDE.md`.

### How classification works

Classification uses BPM (octave-folded into one tempo band, so aubio's double/half-tempo locks cannot fake a high-energy moment), onset density (rhythmic busyness), onset density trend (rising vs. falling energy), kick strength (how far sub-bass rises above its own floor on the beat â€” distinguishes a kick drum from a hi-hat-only or pad-driven passage), and spectral centroid trend (rising centroid = riser/BUILDUP sweep). See `lib/analyser/CLAUDE.md` for the full feature breakdown and design rationale.

**Kick strength carries the classification.** On real material onset density barely varies across a track's sections â€” it says whether anything rhythmic is happening, not how big the moment is. The kick is what separates a drop from an intro, so it gates DROP and, by its absence, widens BREAKDOWN. Branches are tested DROP â†’ BUILDUP â†’ BREAKDOWN â†’ GROOVE; BUILDUP sits above BREAKDOWN because a riser strips the kick by design and would otherwise be swallowed by the kick-absence branch. An unmeasured or unmeasurable kick reads as *absent*, never assumed present.

**Windowed look-ahead:** the engine runs 2.5 s ahead of what the audience hears (matching `playback_delay_seconds` in dmx-enttec-node). Each beat is classified using a symmetric window of past *and* future beats, giving more confident classifications than a causal-only approach. This is why a single anomalous beat cannot flip the intent: it is outvoted by its neighbours via median density.

**Stability pipeline:** each commit runs windowed classification, then the PEAK promotion check, then three guards that can each veto an effect change â€” a vote buffer (consensus required), a minimum dwell check (can't switch away immediately), and an invalid-transition guard (musically impossible jumps blocked). `LightEngine._commit_intent` documents the full ordered sequence; see `lib/analyser/CLAUDE.md` for rationale and `lib/engine/light_engine.py` for constants. A change that reaches consensus is recorded on the intent timeline even when a guard then blocks the effect switch, so `intent_changes_count` reads the classifier's opinion and `effect_changes_count` reads the show.

**ATMOSPHERIC and PEAK are the two intents the beat classifier never returns.** ATMOSPHERIC fires from a beat-absence timer in the 100 ms callback; the first beat after silence immediately re-classifies and changes the effect.

**PEAK is an engine-level promotion, not a classification.** "Sustained maximum energy after the drop" is a temporal property that feature thresholds cannot express (a peak window and a drop window look identical), so the engine promotes an already-committed DROP to PEAK once it has survived a fixed number of commit-beats. While PEAK is current, DROP votes are absorbed so the pair cannot oscillate and the reported intent timeline keeps reading PEAK; PEAK also inherits DROP's hysteresis, so a density dip DROP would ride out cannot eject it. Any other consensus exits PEAK through the normal stability pipeline. Rationale in `lib/analyser/CLAUDE.md`, constant in `lib/engine/light_engine.py`.

**Look-ahead delay** (`LOOK_AHEAD_SEC`) must always match `playback_delay_seconds` in dmx-enttec-node. It is defined in `lib/main.py` and `simulate/runner.py`. Local debug audio playback is delayed by the same amount so headphone monitoring stays in sync.

**Fast simulation:** file simulation runs on a virtual clock driven by audio sample position instead of the wall clock â€” the full pipeline (identical code path to production) processes a track ~30â€“50Ã— faster than real-time and deterministically: the same file always produces byte-identical reports (RNG seeded, no wall-clock jitter). Beat timestamps are song-position seconds; intent/effect blocks are stamped when the audience hears them â€” one look-ahead delay after the beats that caused them â€” so expect intent blocks to trail the track structure by that delay when reading reports. The report records that offset in its metrics, so consumers can realign the two time bases without hardcoding the constant. The decoded audio is cached beside the source file (`*.npy`, gitignored) to skip repeat decodes. Real OS scheduler jitter is only observable in `--ui` / realtime modes, which still run on the system clock.

### DMX migration path

When moving away from SoundSwitch to direct DMX:
- Replace `EffectController._apply_autoloop(effect)` with a `_send_dmx(intent)` call
- Everything above (`YAMNet â†’ intent classification â†’ EventBuffer`) stays unchanged
- `INTENT_EFFECTS` dict in `effect_definitions.py` becomes the only thing to remove

---

## Running

```bash
# Install dependencies (requires uv: https://github.com/astral-sh/uv)
uv sync --extra dev --extra visualizer

# Offline neural-classifier work only (torch/CUDA, onnx, onnxruntime, tensorboard).
# Nothing in lib/ or simulate/ needs it.
uv sync --extra dev --extra visualizer --extra training

# The app is installed as an editable console script, so `uv run auto_pilot ...`
# and `python auto_pilot ...` are equivalent.

# List available MIDI and audio devices
python auto_pilot list

# Minimal run (MIDI port 0, default audio device)
python auto_pilot run 0

# Run with real-time Dash visualizer
python auto_pilot run 0 --ui

# Full options
python auto_pilot run 0 -i INPUT_DEVICE_IDX -o OUTPUT_DEVICE_IDX --no-os2l --ui

# Simulation (no hardware required) — any audio file
python auto_pilot simulate file path/to/song.mp3          # fast headless: full song in seconds, report + evaluation
python auto_pilot simulate file path/to/song.mp3 --ui     # real-time paced with live Dash timeline
python auto_pilot simulate realtime                       # microphone input with live Dash timeline

# Inspect a report: per-10s feature bins, intent timeline, distribution
python auto_pilot simulate file path/to/song.mp3 --report report.json
python training/inspect_report.py report.json

# The benchmark: the frozen eval set, scored against its labels
python training/run_eval_set.py                     # compare against the committed baseline
python training/run_eval_set.py --write-baseline    # re-cut it (see below before you do)

# Score the current classifier against the whole expert corpus (seconds; needs a built table)
python training/evaluate_against_labels.py --data-dir training/data/raveform

# The offline NN verdict: decoder search on val, then the side-by-side val table
# (needs posterior sidecars + priors; pure CPU, no torch on this path)
uv run python -m training.nn.sweep --data-dir training/data/raveform
uv run python -m training.nn.evaluate_v1 --data-dir training/data/raveform

# Tests
uv run pytest -m "not integration"   # fast unit tests only
uv run pytest                        # unit + integration
```

**Flags (`run`):**
- `-i / -o` â€” audio device indices from `list`; passing `-o` enables delayed audio monitoring on that device
- `-d` â€” debug: adds a click on detected notes to the monitored audio (implies monitoring on the default output if `-o` is not given)
- `--no-os2l` â€” disable VirtualDJ connection
- `--ui` â€” launch Dash real-time visualizer at http://localhost:8050
- `--ui-port N` â€” change visualizer port
- `--report FILE` â€” write JSON session report on exit

---

## ML / DSP Components

- **Aubio** â€” real-time BPM, onset, note, and mel filterbank energies. Tuned for low-latency real-time use.
- **YAMNet (TensorFlow Hub)** â€” Google's pre-trained audio classifier; used here for embeddings only (not tag predictions). Cosine similarity + MAD-based outlier detection finds section transitions. Degrades gracefully if model fails to load.

---

## Training Corpus (Raveform)

The direction of travel is to replace hand-tuned classification thresholds with a model trained on real, expert-labelled EDM structure. The corpus for that is **Raveform** (Hugging Face, `taejunkim/raveform`): 1,423 EDM tracks with expert section annotations plus a per-track beat grid. Only the annotations are distributed -- the audio is not, so each track is fetched per YouTube ID for research use.

The pipeline is a set of scripts, each resumable and safe to re-run. Acquisition through the clean manifest lives in `training/raveform/` and is stdlib-only; everything downstream lives in `training/` and `training/nn/`, and the training-table build additionally drives the project's own simulation pipeline (which it treats as read-only). None of it is an installed package -- each script puts the directories it needs on `sys.path` so siblings import by plain module name:

| Script | Role |
|---|---|
| `raveform/raveform_fetch_annotations.py` | pull and schema-validate the annotation archive from Hugging Face |
| `raveform/raveform_manifest.py` | annotations -> `manifest.csv`, plus the label / duration / transition statistics |
| `raveform/raveform_download.py` | manifest -> one mp3 per YouTube ID, sequential and resumable |
| `raveform/raveform_supervisor.py` | unattended patient resume: relaunches the downloader across escalating cool-downs after a refusal wall |
| `raveform/build_clean_manifest.py` | manifest + downloaded audio -> `clean_manifest.csv`, the trusted subset everything downstream reads |
| `raveform/raveform_validate.py` | manifest + audio + download state -> `validation_report.{json,txt}` and `checksums.sha256`: the acquisition-complete verdict |
| `build_training_table.py` | clean manifest + the unmodified fast sim -> `training_table.csv.gz` (one row per labelled beat), a sim report per track, and a pooled log-mel sidecar per track for the neural classifier |
| `evaluate_against_labels.py` | training table -> `baseline_eval.json` + a printed report: the current classifier scored against expert labels (confusion, per-class F1, boundary-F1, flicker, worst songs) |
| `select_eval_set.py` | clean manifest + annotations -> the frozen ten-track benchmark at `training/eval_set.json` (committed, tempo-spanning, structurally rich) |
| `eval_assets.py` | the eval set's committed artifacts: the derived opaque mp3 names, the sha-pinned label slice, and the `--cut` that re-makes both |
| `run_eval_set.py` | the frozen eval set -> per-track report checksums and label-aligned scores; cuts and enforces `training/eval_set_baseline.json` |
| `nn/dataset.py` | clean manifest + mel sidecars + annotations -> `splits.json` and the windowed, loss-masked training set the CRNN reads |
| `nn/model.py` | `SectionCRNN` -- the two-head acoustic model (label logits at ~10 Hz, boundary logits at frame rate) |
| `nn/train.py` | windowed dataset -> checkpoints, `training_report.json` and TensorBoard logs under `<data-dir>/models/v1/` |
| `nn/export_onnx.py` | a training checkpoint -> `<data-dir>/models/v1/model.onnx` (dynamic time axis), plus the single pinned onnxruntime session every consumer is required to go through |
| `nn/infer.py` | mel sidecars + `model.onnx` -> one posterior sidecar per track under `<data-dir>/posteriors/`, byte-identical on every regeneration |
| `nn/priors.py` | corpus bar runs -> the fitted structural graph, duration floors and hazards the decoder commits against (`<data-dir>/models/v1/priors.json`) |
| `nn/decoder.py` | posterior sidecar + bar grid -> immutable per-bar class decisions (fixed-lag Viterbi over an explicit-duration HSMM); owns stability and latency policy |
| `nn/evaluate_v1.py` | decoded timelines + training table -> the verdict for one split at `<data-dir>/models/v1/eval_<split>.json` (`eval_val.json` is the tuned reading, `eval_test.json` the selection-clean one): NN and rule classifier side by side, scored by `evaluate_against_labels`' own functions |
| `nn/sweep.py` | cached posteriors -> the decoder parameter search and `<data-dir>/models/v1/decoder_config.json` (best val macro-F1 subject to the baseline's flicker and the latency budget) |

`clean_manifest.csv` is the boundary between "audio we happen to have" and "audio we are willing to learn from". Only its `ok` rows may feed a training table or an evaluation run.

`training_table.csv.gz` is the corpus-wide version of what a single sim report already is: a feature row per beat, now carrying the expert label for the section that beat falls in, the show state the engine actually committed there, and a per-track z-scored copy of every continuous feature. It is the input to both the label-aligned baseline evaluation and the neural section classifier's dataset builder.

Decisions that belong here rather than in the code:

- **Canonical label vocabulary.** The published labels are a superset of the documented seven. `end` is a tail sentinel, not a musical section, and is dropped; `altintro` folds into `intro` and `bridge` into `breakdown` (too few examples to learn as a class); `altoutro` is kept. Adjacent same-label sections are merged *after* the drop and the fold, so a canonical "section" means one contiguous stretch of one label. The rationale and the mapping live in `raveform_manifest.py`.
- **Politeness over throughput.** Downloads are strictly sequential with a pause between videos -- pulling 1,423 tracks in parallel is indistinguishable from abuse. Bot checks are never worked around: no cookies, credentials or IP tricks. A refusal is recorded, reported, and left as an owner decision, and a run of consecutive refusals aborts rather than burning the manifest into failure records.
- **Resumability is the contract.** The downloader is safe to kill at any point: finished tracks and failures are both persisted, an interrupted track is never recorded as failed, and a track counts as downloaded only when a non-empty mp3 is actually on disk. See the module docstring in `raveform_download.py`.
- **The container header is not evidence.** An mp3 truncated on a frame boundary -- the normal shape of an interrupted download -- decodes with no error at all (the stream just ends) while its Xing header still advertises the original full length, because that header is written at encode time and never revised. Neither "ffmpeg exited clean" nor "the header duration matches" detects it, together or apart. So the gate measures how much audio the decoder actually emitted and judges on that: it must agree with the header (or the file is truncated) and with the annotation (or it is the wrong recording). The measurement is free -- it comes from ffmpeg's own progress output during the decode pass the gate already runs.
- **Cleanliness is a gate, not a cleanup.** A track is admitted only if it decodes without a single error line, decodes to the length it claims, and that length matches the annotation record. A truncated file is `corrupt` (the bytes are damaged); a fully-decodable file of the wrong length is `duration_mismatch` (the wrong video was fetched, so every beat-to-label join would drift). Rejected tracks are recorded with their reason rather than deleted -- a wrong-length track is worth a human's eye. The tolerance is deliberately loose (an absolute floor for normal tracks, proportional for long DJ edits) because it exists to catch a different recording, not encoder padding.
- **The gate reads a moving corpus.** It is designed to be re-run while the downloader is still fetching: tracks not yet on disk are simply absent from the output, and a file whose mtime is younger than the min age is left for the next run because it may still be mid-write. Nothing in `audio/` is ever written, moved or deleted by the gate.
- **Acquisition is done when the arithmetic says so, not when the downloader exits.** The gate is silent about what it never saw, so a corpus that is quietly short looks exactly like a complete one. `raveform_validate.py` closes that hole by reconciling every manifest row against the disk *and* the download state, into exactly one of OK / DURATION_MISMATCH / CORRUPT / MISSING / UNAVAILABLE, and declaring convergence only when `OK + DURATION_MISMATCH + UNAVAILABLE == manifest rows` **and** `MISSING == CORRUPT == 0`. Both halves matter: the zeroes say nothing is in a state we refuse to accept, the sum says nothing fell out of the accounting. `MISSING` is the bucket that exists purely to make "we lost track of this" impossible to confuse with "this is unobtainable" -- the latter must always carry a recorded yt-dlp reason and its error tail.
- **Complete means complete against the annotations, not "1,423 files".** The five buckets only ever describe audio, so two failures cannot appear in them by construction: an orphan mp3 that no manifest row claims, and an annotation that does not reconcile with the manifest -- a missing or unparsable beat grid, a disagreeing YouTube id, a duration that differs by more than the manifest's millisecond rounding. A track whose beat grid never arrived is as untrainable as one whose audio never arrived, and its mp3 looks perfect. So the validator reports two verdicts: convergence (the audio accounting, unchanged) and an **all-clear** that additionally requires no orphans and no annotation issues. The exit code follows the all-clear, because a corpus that adds up but does not reconcile is not one we are done with.
- **What the disk says outranks what the log says.** `failed.jsonl` is append-only, so a track refused on one cycle and fetched on the next is recorded in both it and `downloaded.txt`. Any track with audio on disk is judged by decoding it; the failure log is consulted only for tracks that are not there.
- **Checksums are a baseline, not a verification.** YouTube publishes no canonical hashes, so there is nothing to check the corpus *against*. `checksums.sha256` (sha256sum format, OK files only) records what passed the decode check on the day it passed, so a later re-validation can prove the bytes have not drifted. That baseline plus the decode gate is the entire correctness mechanism this corpus can have.
- **Recovery is selective, not blanket.** A plain downloader re-run skips every recorded failure and a `--retry-failed` re-run re-polls genuinely dead videos forever. So the failure reasons split into *permanent* (the video is gone, private, age-gated or geo-blocked) and *transient* (YouTube refused this client). Only the transient ones — plus `other`, the unclassified bucket, which is far likelier to be something unnamed than a video that vanished silently — are worth re-attempting, and the script derives its own retry hints from that same set so its advice can never be narrower than the advice that works.
- **A permanent condition always outranks a transient one.** YouTube's permanent errors habitually carry transient-looking text: a private video ends with "Use --cookies-from-browser", an age gate opens with "Sign in to confirm your age". Every mis-classification found in this pipeline was a transient bucket sitting above a permanent one and swallowing a dead video, and the cost was never the wrong label alone — retry passes re-poll it forever and a run of them aborts a healthy run. The bucket table is ordered permanent-first for that reason, and tests use the wording yt-dlp really emits, since a strawman fixture is what let those bugs survive.
- **A 403 on the media URL is our problem, not YouTube's policy.** It means a signature/nsig challenge could not be solved, which is a property of the toolchain, not of the video. It gets its own reason bucket rather than folding into `bot_check` because the two demand opposite remedies, and reporting one as the other sends the owner after credentials they do not need. **The remedy that was actually needed is patience**: all 68 recorded 403 events resolved on re-attempt, which is why they sit in a retryable bucket rather than with the dead videos. The diagnosis, for the record: a JS runtime that is installed but not *visible to the running process* explains only 19 of them (a shell started before the install carries a stale PATH — "installed" must be checked from the runner). The other 43 happened while a runtime *was* visible, and yt-dlp names the cause itself in four of those records — its remote challenge-solver components are opt-in and were not enabled. That was never acted on because it never had to be. If a future refresh hits a 403 wall that survives repeated patient re-runs, *then* it becomes a decision; nothing is pre-built for it here.
- **Running it detached, and stopping it.** The long sweeps run as detached OS processes that outlive the session. Two things about that are not guessable and cost real data when guessed wrong. **Stopping requires `taskkill /PID <pid> /T`** — the venv's `python.exe` is a trampoline that re-execs the real interpreter, so killing the recorded PID alone leaves the actual downloader orphaned and still fetching, invisible to the next run's bookkeeping. **Re-issuing the detached launch command truncates `download.log`** — the redirect reopens the file, so a relaunch silently destroys the previous run's evidence. Redirect to a new filename per cycle (the supervisor does this: `download.cycle<N>.log`).
- **The corpus data directory holds *ops copies* of the scripts, and they drift.** Both `raveform_download.py` and `raveform_supervisor.py` are copied next to the corpus, and a supervised refresh runs **those copies, not the branch**. So whenever either changes, refresh both and confirm it: `cmp <data-dir>/raveform_download.py training/raveform/raveform_download.py` and the same for `raveform_supervisor.py` — they must be byte-identical to the branch blob. This is not hypothetical. The supervisor used to live *only* in the gitignored data directory, unversioned and unreviewable, and the ops downloader beside it carried the pre-`http_403` classifier while the supervisor hardcoded `--retry-reasons bot_check`; the next refresh would have stranded every recoverable 403 while the branch looked correct. Both files are now on the branch, and the supervisor reads `RETRYABLE_REASONS` out of the downloader sitting beside it rather than keeping its own copy — but the copy step itself is still manual, which is why the `cmp` is written down.
- **A beat is labelled by its own section, never by a merged run.** Merging adjacent same-label sections answers "how long is a musical section"; it must not answer "what is playing at time t". A merged run's *span* can swallow a dropped `end` sentinel sitting between two members, and that time is explicitly not the surrounding label's. So the beat-to-label join looks up the individual published section (folded and clamped) and uses the merged runs only to find where the labelled region of a track starts and stops.
- **Unlabelled audio is dropped and counted, never absorbed.** Most tracks begin with a stretch of audio before the first annotated section, some end with audio past the last one, and the `end` sentinel is a third such region. Attributing any of it to a neighbouring section would teach a section the annotator never marked -- and would look exactly like a classifier error downstream. Every beat is accounted for: kept, leading, interior gap, or trailing.
- **Two label spaces ship in the same table.** `label_canonical` is the seven-class corpus vocabulary; `label_v1` merges `cooldown` into `breakdown` and `altoutro` into `outro` for the five-class space the neural classifier trains on (a cooldown is defined by *where* it sits, which no single analysis window can see). Keeping both means changing the model's class space costs a re-join, not a re-simulation. `label_raw` is kept alongside so any mapping decision stays auditable from the table alone.
- **Simulate once, join many times.** The batch is two stages: an expensive parallel pass that simulates each track and caches its report, and a cheap serial pass that rebuilds the whole table from those cached reports. Fixing the join or adding a column is then seconds of work rather than hours of re-simulation, the join stays pure and unit-testable, and the reports themselves are the artefact the evaluation harness reads.
- **The report cache is keyed on what the report actually depends on.** A cached report is reused only if the pipeline that produced it (`lib/` + `simulate/`) and the exact audio file (size *and* mtime) are unchanged, and its mel sidecar is still on disk; anything else is a miss with a named reason, and `--force` misses everything. The key is deliberately *not* repo HEAD -- keying on HEAD would discard the whole corpus cache on every commit to a script or a document, which is what the cache exists to prevent -- but uncommitted edits under those paths do invalidate it, so a pipeline change under test never hides behind stale reports. A rebuild with nothing changed therefore costs seconds, and a rebuild after twenty new downloads costs twenty tracks. Two *different* uncommitted edits are two different keys — a constant "dirty" marker would have given a whole afternoon of edits one cache entry, which is the state a pipeline is changed in most often. The mel sidecar is checked for the exporter generation that wrote it and not merely for existence, because an exporter change that keeps the frame rate and the band count changes every number in the file; sidecars from before that stamp are grandfathered rather than re-simulated.
- **The corpus stops short of the analyser's self-reset.** `MusicAnalyser` throws its rolling state away every 15 minutes, and the mel exporter has no such reset, so past that horizon a track's beats and its features describe the same audio from different states — and the join would produce wrong rows with no error and no counter. Tracks at or past it are dropped from the build with a line saying why. The corpus tops out 0.11 s under it, so this is a live edge rather than a hypothetical one.
- **The batch cleans up after itself, and only after itself.** The simulation caches decoded audio beside each mp3 at several times the mp3's size; over the full corpus that is far more disk than the corpus itself. Each worker deletes its own track's cache as soon as the features are extracted, in a `finally` so a failed track cannot leak one. Caches that existed before the run started are left where they were -- tidying is scoped to what this run created.
- **Feature parity with the runtime is enforced, not assumed.** The neural classifier trains on the mel stream the live pipeline computes, but the pipeline under evaluation is read-only, so the exporter rebuilds the same aubio objects instead of borrowing them. That duplication is only safe because a unit test feeds both sides the same buffers and demands identical energies: without it, an FFT-size change in the analyser would silently train the model on features the runtime never produces.
- **Prerequisites.** `yt-dlp` and `ffmpeg` on PATH, plus a JavaScript runtime (Deno or Node) *visible to the running process*. Installed is not the same as visible: a shell or detached process started before the install inherits a stale environment block, so check from the actual runner, not from a fresh terminal. Downloads **can** fail with `HTTP Error 403: Forbidden` — not *will*: the whole 1,387-track corpus was fetched under exactly this setup, and every 403 that occurred was cleared by patient re-attempts.

### Label-aligned evaluation

`evaluate_against_labels.py` replaces the plumbing-only verdict in `simulate/evaluator.py` (which only asks "did anything happen") with musical ground truth. It reads the training table and nothing else -- the intent timelines were already realigned into song time when the table was built, so the evaluator never re-derives them from reports.

- **The intent alphabet and the label alphabet are different languages, so the confusion matrix is not square.** Rows are the six intents the engine can commit, columns are the annotator's classes, cells are minutes of show. Flattening one alphabet into the other before looking at the matrix hides exactly the failures worth seeing. The mapping from intent to "the labels this intent is correct for" lives in a single dict at the top of the script and is the thing the owner iterates on.
- **Two spaces, one primary.** Everything is reported in the five-class `label_v1` space (the model's target space, so its numbers are what a model gets compared against) and again in the seven-class canonical space as a diagnostic. They disagree in a specific place: GROOVE's semantic home is `cooldown`, which v1 merges into `breakdown`, so v1 flatters GROOVE relative to canonical.
- **An intent cannot know where in the track it is.** ATMOSPHERIC describes quiet with no beat, which is `intro` at the start of a track and `outro` at the end -- the same sound, labelled by position. It is therefore scored correct against either, and when it is wrong its false positive is split across the classes it claimed so no single class absorbs the blame for an ambiguous prediction.
- **Both event streams are quantised the same way.** The table is per beat, so a label boundary is only observable as "the first beat carrying the new label" and an intent change as "the first beat carrying the new intent". Using one estimator on both sides makes the quantisation cancel when a change is correctly timed. The residual uncertainty is one beat period, which is why the strict tolerance tier sits at the resolution floor and means "within a beat", not "sample accurate".
- **Flicker is the product metric, and it is not boundary precision.** Flicker counts state changes with no real boundary anywhere near them, per audience-minute; boundary precision additionally punishes a correctly-placed decision made twice. An audience notices the first and not the second, so both are reported and they are not interchangeable.
- **"How often did it change" has two honest answers, and a model may only be scored against one of them.** The *intent stream* counts every committed `LightIntent` change - each one re-picks a lighting effect, so that is the show as the room experiences it and the owner's continuity number. The *class stream* maps into the label space first and then differences, so a DROP-to-PEAK move (different lights, same label class) is not counted. A model that predicts label classes emits a class stream by construction and physically cannot make the difference, so comparing it against intent-stream numbers would credit it for changes it is unable to make. Boundary-F1 and flicker are therefore reported for both streams everywhere, and the report says which is which.
- **A structural ceiling is a wall, not a target.** "Reachable classes over all classes" cannot be exceeded, but it also cannot be reached: the time sitting in classes the vocabulary cannot name is still predicted as *something*, so it lands as false positives on the classes that do exist. The evaluator reports the naive bound and, beside it, the best figure actually achievable - the optimum concentrates all of that damage on the single largest class rather than spreading it, because the objective is convex and its maximum over the allocation simplex is at a vertex. Spreading is the *worst* allocation, which is the trap an "equalise the marginal loss" reading falls straight into.
- **A beat with no committed intent is not a class.** Those beats are excluded from every cell and counted instead -- scoring them as errors blames the classifier for the engine's start-up, scoring them as correct flatters it. Their position matters more than their count: an *interior* gap would let the change stream close over a silence and read two commits as one change, so leading / interior / trailing are counted separately as a tripwire.

**The baseline.** The live numbers are in `training/data/raveform/baseline_eval.json` and are deliberately not copied here - the corpus is still growing, so any figure written down goes stale silently. Re-run the script for current values. What is durable is the shape of the result, and it is a starting line rather than a verdict on the architecture (the classifier was calibrated on one track and this was its first contact with hundreds):

- Macro-F1 lands around a fifth of the way to 1.0 in the v1 space, with DROP the one class that works, BREAKDOWN/GROOVE moderate, and BUILDUP near-random despite firing roughly the right *amount* of time - a placement failure, not a sensitivity one.
- **ATMOSPHERIC is never committed on mastered EDM at all** (see Known Issues), putting ~22% of labelled time out of reach. That alone caps macro-F1 well below 1.0 before a single classification is made, and it is the single biggest lever available.
- The engine changes intent several times more often than the music changes section, and the large majority of those changes are nowhere near a real boundary. Continuity, not accuracy, is the furthest from acceptable.

### The benchmark: the frozen eval set

The simulation used to be judged against one bundled track and a plumbing-only PASS verdict that asked "did anything happen at all". It is now judged against **ten expert-labelled Raveform tracks frozen in `training/eval_set.json`**, with `training/run_eval_set.py` as the gate and `training/eval_set_baseline.json` as the committed answer. The bundled Generate track has been retired; its historical measurements survive in the Stage-1 plan under `docs/superpowers/plans/`.

- **A benchmark that only runs on one laptop is not a benchmark.** The ten tracks' audio and labels are committed — the audio under names derived from the YouTube id (`training/eval_audio/`, opaque so a directory listing says nothing; derived so code finds a file with no lookup table to go stale), the labels as a verbatim, sha-pinned slice of the corpus annotation. Owner-authorised, and precisely these ten: the rest of the corpus stays gitignored and machine-local. A machine that has the corpus too still reads the committed copies, so every machine benchmarks the same bytes. `training/eval_assets.py` owns the derivation and re-cuts both after a re-freeze.
- **A benchmark that follows the corpus is not a benchmark.** The set is frozen: the selector refuses to overwrite it without `--force`, and the baseline records the eval set's own checksum so a re-freeze fails loudly instead of silently re-scoring a different ten tracks. Re-cutting the set and re-cutting the baseline is one change, never two.
- **That checksum is over the file's bytes, so the checkout must not rewrite them.** Git on Windows defaults to `core.autocrlf=true` and materialises LF as CRLF, which changes the hash while changing nothing about the benchmark — the guard then passed in the worktree the file was written in and failed in every fresh clone, making its verdict a fact about the machine. `.gitattributes` pins the frozen artifacts to `eol=lf` so every checkout agrees with the writers, which already emit LF unconditionally. An older clone made before that pin keeps its CRLF copies until they are re-checked-out; rewriting them to LF is the remedy and leaves the index untouched.
- **The benchmark is never learned from.** Neither the ten ids nor any track sharing an artist with one of them enters a training or validation split. This is stated twice on purpose — once here as policy, once in the split builder as code.
- **Two gates, and they mean different things.** The *report checksum* says the pipeline's behaviour moved: a deterministic run over fixed audio can only change if the code did. The *label-aligned scores* say whether the show got better or worse. A deliberate improvement trips both, and that is the workflow — read the table, decide the change is wanted, re-cut the baseline in the same commit. A regression with no checksum change is impossible and would mean the determinism contract is broken. Beside the scores, the *count facts* each row records (beats joined, label boundaries, seconds scored) are compared exactly — a score tolerance wide enough to be useful absorbs a run that measured a different number of things.
- **The ground truth is verified before anything is simulated.** A boundary that moves under a baseline cut before the move leaves every number comparable to nothing while the gate prints "matches". The committed label slice cannot move behind git's back, so what is checked of it is *provenance*: it records the checksum of the annotation file it was cut from, and that must be the one the eval set froze against. A machine falling back to the gitignored corpus annotation gets that file hashed on every run instead. Either way a mismatch is fatal. The manifest that chose *which* tracks are in the set is deliberately not checked: it grows with every download batch and feeds no score.
- **Scores are the corpus's scores, not the benchmark's own.** The runner reuses the training table's beat/label join (and therefore its look-ahead realignment) and the label-aligned evaluator's metric functions. A benchmark that computed its own numbers would eventually disagree with the corpus evaluation and nobody would know which was right.
- **The integration suite runs a subset, a human runs the set.** Three tracks fit a test-suite wall-time budget; ten do not. A subset run compares only its own tracks and deliberately does not compare the aggregate — an aggregate over three tracks is a different quantity. The full set is a manual command and takes a couple of minutes, or seconds across worker processes (parallel and serial produce identical bytes, which is checked by running both). The quoted integration wall time assumes more than one core; on a single-core machine the worker pool degrades to serial and the suite gets slower but stays inside its budget.
- **A subset may not overwrite the committed baseline, and the baseline is itself under test.** The two ways this tripwire could be disarmed without anything failing are a baseline cut from a partial run (the gate then compares the tracks it ran against the tracks in the file, so the rest silently stop being checked) and a guarded metric missing from the file (skipped rather than flagged). So a partial `--write-baseline` at the committed path is refused outright — `--allow-partial-baseline` is the deliberate override, an explicit `--baseline PATH` is the experiment — a missing metric is a failure rather than a skip, and a fast unit test reads the committed file and asserts it still covers the whole frozen set, was cut against the current one, and carries every gated number.
- **The eval set is exempt from the delete-your-decode-cache rule.** Everything else in the corpus discards the decoded `.npy` beside its mp3 because the corpus is thousands of tracks; the eval set is ten and is re-simulated on every test run, so its caches persist. They land beside the committed mp3s, where the repository's `*.npy` rule keeps them out of git. That is under a gigabyte and removes the decode from every run after the first.

### Neural section classifier -- dataset, model, decoder and the offline verdict (`training/nn/`)

The design spec is `docs/superpowers/specs/2026-07-26-nn-section-classifier-design.md`: a CRNN acoustic model over the pipeline's own mel stream, plus a fixed-lag Viterbi decoder that owns stability and latency policy. This package is the offline half. Only code lives in git; every artefact it produces (splits, checkpoints, ONNX graphs, posterior sidecars) lands in the gitignored data directory.

- **Splits are a pure function of the track id, so the corpus can keep growing.** The download is still running and the eventual 1,423-track retrain must be comparable with tonight's model, so the 70/15/15 assignment hashes `(seed, youtube_id)` rather than shuffling a list. Adding tracks may only ever *add*; nothing already placed can move. `splits.json` is the frozen record and wins over recomputation -- the hash decides only where new ids land.
- **The benchmark is excluded twice over.** The ten frozen eval-set ids never enter any split, and neither does any track sharing an *artist* with one of them. The second guard is the one that is easy to miss: producers have a sound, so a net that has heard six other tracks by an eval-set artist has partly memorised the benchmark. Artist matching is collaboration-aware -- a credit is split on `Feat.` / `&` / `vs` and any shared participant excludes, so a solo release by half of a featured pair is caught. It deliberately over-excludes (a band name containing `&` splits into two names); a handful of lost training tracks is cheap, a contaminated benchmark is not.
- **Unannotated audio is masked, never labelled.** The same rule the training table applies to beats applies here to frames: audio before the first published section, audio past the last section end, and the time of a dropped `end` sentinel carry no loss. A masked frame teaches nothing; a mislabelled one teaches the wrong thing.
- **The boundary head gets a fourth mask that the label head does not need.** Where two published sections fold to the same `label_v1` class, the join is a statement about section identity, not necessarily an audible event -- so its neighbourhood is *deleted* from the boundary loss rather than taught as a negative. A genuine transition that happens to sit inside a deleted neighbourhood keeps its target: deletion is for ambiguity, not for erasing known events. On the current corpus roughly three quarters of tracks have at least one such join, so this is a live path, not a corner case.
- **The window geometry is derived from the sidecars, not restated.** Frame rate, band count and pooling factor all come from the constants the sidecars were written with, and every sidecar load re-checks its own recorded frame rate against them -- a change to the mel front-end must break loudly rather than silently train the model on a different grid than the runtime produces.
- **Augmentation moves the window, not the truth.** Training draws a fresh window offset and a gain shift per item per epoch, seeded from `(seed, epoch, index)` so a run is reproducible and dataloader workers cannot collide; offsets stay aligned to the label-pooling factor so pooled targets are sliced rather than re-derived. Gain is applied as the additive shift in the log domain that it is, and clamped so the model never sees an input the encoder could not have produced.
- **Torch is an optional extra and stays that way.** `training` is a separate `[project.optional-dependencies]` group (`uv sync --extra training`); `lib/` and `simulate/` gain no new imports, and the dataset's target and mask logic is plain numpy so it stays testable on a checkout that never installs torch. The pinned versions and the uv index/environment configuration are validated end to end on this machine -- see `.superpowers/sdd/2026-07-26-nn-classifier-v1-offline/cuda-preflight-report.md` before changing any of them.
- **The convolutions pool frequency and never time.** The boundary head has to name the frame a section change lands on, to the tolerance the annotations carry; a time-strided front-end throws that away before the recurrence sees it. "Which bands are loud" survives pooling, "exactly when" does not. The recurrence is bidirectional for the reason the whole design exists: the window carries look-ahead the audience has not heard, and telling a buildup from a fake-out means hearing whether the drop actually lands.
- **Three loss terms, one failure each.** Class-weighted focal loss stops the majority class from owning the ambiguous middle *and* stops the easy interior of a long section from drowning out the frames either side of a transition. Boundary BCE takes its positive weight from the corpus's own sparsity rather than a guess. A total-variation penalty on the label posteriors buys the smoothness that flicker -- the metric the runtime is being replaced over -- is made of. That third term is *boundary aware*: a blanket smoothness penalty would teach the model to smear the one step the decoder needs, so it is weighted down where the annotation says a transition is happening and abstains entirely where the boundary target was deleted.
- **Calibration is a metric, not a side effect.** The decoder consumes posteriors, divides by class priors and multiplies along a path, so it reads every column of the softmax. A model that is confidently wrong is worse to it than one that is honestly unsure, and per-class ECE (one-vs-rest, not just on the argmax) is logged every epoch beside macro-F1 as a release gate.
- **Training is bitwise reproducible, and that is enforced rather than hoped for.** Two runs of the same config in fresh processes produce identical weight hashes and identical loss curves. Two ops would silently break it -- `gather` and boolean indexing both route through a nondeterministic `scatter_add` on the backward pass -- so neither appears in the loss path; masking multiplies by a float mask instead. `--resume` restores optimiser, scheduler and RNG state and the per-epoch shuffle is re-seeded from `(seed, epoch)`, so an interrupted run rejoins the *same* trajectory rather than a plausible one. There is a deliberate `--crash-after-epoch` drill for proving exactly that.
- **The schedule is derived from the dataset, never written down.** Steps per epoch and the cosine tail come from `len(dataset)`; the corpus is still growing, and a schedule computed against a stale window count either decays to zero mid-run or never arrives.
- **DataLoader workers are never persistent, and that is a correctness choice.** A persistent worker holds a *pickled copy* of the dataset, so `set_epoch` in the parent never reaches it and the augmentation freezes on whichever epoch the workers spawned at -- every later epoch re-draws identical offsets and gains, which looks exactly like a working run. Respawning per epoch re-pickles the dataset with the new epoch, so correctness costs the documented ~2.2 s Windows spawn per epoch. Measured, not assumed, and pinned by a test. `--num-workers 0` remains the default and is genuinely faster at this corpus size, so the cost is hypothetical.
- **A checkpoint describes its own geometry.** Every save carries the constructor arguments that decide tensor shapes, and resume refuses a mismatch. A `state_dict` alone is not self-describing: change the label pooling factor and the weights load *cleanly* into a model that decodes at the wrong frame rate. `models/v1/<run>/best.pt` is the artifact the whole plan is judged on and it outlives the CLI defaults that produced it. It is also a pickle, so the export path reads it under torch's restricted unpickler — a published checkpoint is a file that gets shared, and nothing the exporter needs from one is worth the right to execute it. Only resume relaxes that, for a file this trainer wrote minutes earlier into the gitignored corpus.
- **Hyperparameter provenance is recorded, not remembered.** The learning rate was specified against one batch size and runs at another; rather than silently "fixing" it, every run config and report carries a note saying so and naming the triage order if the run underperforms. The next person to read a training report is not the person who chose the number.
- **TensorBoard is part of the run, not an afterthought.** The trainer writes per-step losses, per-epoch val metrics, per-class F1 and ECE and a confusion image, and starts a detached server itself if one is not already listening -- the owner watches training live at `http://localhost:6006`.
- **The ONNX graph is the interface, and the exporter that produces it is deliberately the deprecated one.** Torch's new dynamo exporter handles this architecture *without raising* and silently freezes the time axis at whatever length it traced, so the graph then only runs at that length; the legacy TorchScript exporter is the only path that keeps the axis symbolic today, which is why torch is pinned. Because the failure is silent, the exporter reads the declared dimensions back out of the written file and refuses to publish a graph that lost them. The model is rebuilt from the checkpoint's own geometry block and checked back against it before any weight is loaded -- a pooling-factor mismatch changes no tensor shape, so it would load cleanly and decode at the wrong frame rate.
- **Every inference goes through one session definition, and it is single-threaded.** That is a determinism contract rather than a performance choice: a threaded reduction sums in whatever order the pool finishes in, and float addition is not associative. Throughput comes from running *tracks* in parallel over that same pinned session, so no track's numbers depend on how many workers ran -- which is a test, not a claim.
- **The offline pass is the runtime's own sliding window, not a whole-track forward pass.** The model is bidirectional over a fixed window, so a frame's posterior depends on where in the window it was read; a single long pass would be ~150x cheaper and would not be the same model. The window slides at the runtime's callback cadence (snapped to the pooled label grid, since an unaligned hop would land the label head's output between cells of the track-wide grid) and every posterior covering a frame is averaged, which is the standard fix for window flicker.
- **Never read the window's edge -- except where nothing else can.** The outer margin of a window has context on one side only and a cold recurrent state, so it is dropped before aggregation. The head and tail of a *track* are reachable by no window's interior, and there the first and last window donate their margin rather than leave the decoder an undefined posterior. That is arithmetic, not a policy exception, and each sidecar records how many windows voted on each frame so the thin ends are visible instead of looking like the confident middle.
- **The last window re-overlaps its predecessor on purpose, so aggregation averages and never concatenates.** A track length is not a whole number of hops; the final window is clamped back to the end of the track (the same rule the dataset uses in eval mode, so the tail -- almost all outro -- is not silently unreachable). Concatenating window outputs would double-count that tail and shift every frame after it.
- **The boundary score is stored raw, and it is not a probability.** Per-window normalisation was measured to destroy most of its ranking power, so nothing rescales it; and it was trained under a large positive weight against a ~2 % base rate, so the decoder must consume it as evidence, never as P(boundary).
- **A sidecar is keyed on the model *and* the window geometry, and its bytes are a pure function of its contents.** Reusing a cached posterior file after a hop or edge change would silently mix two geometries in one evaluation, so both are part of the cache key. The archive is written with a fixed member order, a fixed timestamp and no compression rather than through `np.savez`, whose reproducibility is an implementation default and whose compressed form folds the zlib build into the bytes -- a determinism claim resting on either is a claim about one machine. Both time bases are recorded in the file (the frame-rate boundary grid and the pooled label grid, which is stamped at the end of each group) so no consumer has to infer one from the other.
- **The comparison runs through the baseline's own scoring functions, with exactly two things changed.** A verdict measured by a second implementation of the metrics is a verdict about the two implementations. So the accumulators, the event matcher, the flicker rule, the time weighting and every derived property are imported from `evaluate_against_labels`, and only two things vary: the *claim map* (which label a prediction asserts) and the *sentinel* for "no prediction here". Both sides then aggregate with the same aggregator, and a divergence in arithmetic is structurally impossible rather than merely unlikely.
- **Undecoded is not the same as wrong, and saying so is not a courtesy to the model.** The decoder runs on a bar grid, so it says nothing before the first downbeat or past the final bar line. Those regions belong almost entirely to `intro` and `outro`, so scoring them as misdecodes would be a systematic charge against two specific classes for where the annotation grid starts. They reuse the evaluator's existing "no committed prediction" sentinel, which already excludes and reports rather than penalises. On the current corpus this is a near no-op -- the beat grid starts on a downbeat -- which is worth knowing precisely because it means the handling cannot be flattering the result.
- **A model that can name the class is held to a stricter standard than the vocabulary it replaces, and both readings are published.** ATMOSPHERIC is credited against `intro` OR `outro` because an intent cannot know its position in the arrangement; a model predicting in the label space can, so its primary score forgives nothing. That is a genuinely higher bar, so the same decode is also scored under the rule's exact ambiguity, and a third number restricts the macro-F1 to the classes both sides can contest. One table, three readings, so "the model only wins because the two sides were scored differently" is a question the artifact answers instead of inviting.
- **Two of the five classes are not contested at all, and that is a property of the harness rather than of the classifier.** ATMOSPHERIC -- the only intent covering `intro` and `outro` -- fires from a beat-ABSENCE timer, while the training table carries one row per DETECTED BEAT. The rows that exist are exactly the rows where that trigger cannot be active, so the baseline's zero on those two classes is guaranteed by construction, not measured. They carry roughly two thirds of the headline delta, so the restricted macro-F1 over the contested classes is the model-vs-model number and the one any per-track claim must be read from -- the full five-class reading cannot go negative per track and must never be quoted as evidence that a win is universal. The full reading is still primary and still fair as a description of what the *deployed system* does at beats; it is simply not a like-for-like model comparison.
- **The class stream is the only fair comparand, and for this model it is the only stream.** A label-space model cannot express DROP -> PEAK, so quoting the engine's intent-stream flicker against it would overstate the model. Under identity claims the model's two streams are provably the same stream, which the report asserts rather than assumes -- printing one number twice would look like corroboration.
- **Selection is constrained, not maximised.** The chosen config is the best val macro-F1 *subject to* flickering no more than the shipping classifier and committing inside the look-ahead budget. A model that is right more often while twitching more is not an improvement, and one that needs more future audio than the show has is not runnable. Longer lags are still measured -- the accuracy-versus-lag curve is the evidence for whether the budget should ever move -- but cannot be selected. When nothing clears the constraint the sweep raises rather than quietly returning the best ineligible config.
- **The sweep is joint where the axes interact, and deterministic everywhere.** Prior strength and drop-miss cost both push mass toward the rare expensive classes, and the boundary gain is meaningless without the neutral point it is measured from; a line search over either pair finds a compromise neither axis would pick. Those are full grids, the remaining axes are staged around the running winner, and a joint refinement then re-opens everything at once to catch what a staged search walks past. Nothing samples: every axis is an explicit tuple and the enumeration is a fixed-order product, so a winner is reproducible without replaying the search. It is cheap because bar observations depend only on the two knobs the sweep holds fixed -- so sidecars are read once and a config costs a decode plus a score -- and the cache is keyed on that pair rather than assuming it.
- **Sensitivity is measured by ablating the shipped config, not by reading a curve off the search.** The best result ever seen at a given knob value conflates that knob with whatever the rest of the config happened to be in the stage that produced it, so it cannot say what the knob cost. A final pass moves one axis at a time around the chosen config, reusing configs the search already measured so the anchor appears in every curve. That pass is also a search -- if it finds something better the selection takes it, and the artifact records whether the anchor survived.

**v1 exists, it won, and it is not running anything.** The chain was scored once against the held-out test split -- tracks no selection decision had ever seen -- and beat the shipping rule classifier on every metric the plan named, in both the all-classes and the contested-core reading, while committing several times fewer state changes. The verdict artifacts are `models/v1/eval_val.json` (the tuned reading) and `models/v1/eval_test.json` (the selection-clean one); each carries the sha256 of the model, priors, splits and table that produced it, so any figure traces to a chain rather than to a memory. The figures themselves are deliberately not copied into documentation -- the corpus is still growing and a written-down number goes stale in silence. `training/nn/CLAUDE.md` maps the package; the bullets above are the reasoning behind it.

- **The test split is read once, and that run is the record.** Everything else in this package was chosen on val -- the decoder config, the early-stopping epoch, and which of several training runs to export -- so the test figure is the only number no decision was permitted to see, and the acoustic-layer selection noise alone is comparable to most of what the decoder sweep was tuning. Tuning after reading it spends the one clean measurement the project has. A disappointing test result is therefore a *new versioned model*, never a re-tuned old one, and that model gets its own single read.
- **Stability is not accuracy, and the decoder buys the first with the second.** Committing few, long, confident runs is precisely what produces the large flicker win; on a track whose structure alternates faster than the fitted duration prior expects, the same property lets one run swallow several real sections. The worst tracks in *both* splits share that shape -- a couple of committed runs against an annotator's many -- and it is the dominant reason a track can lose to the rule classifier on the contested core. A twitchy classifier collects partial credit for passing through the right state; a committed one does not. This is priced rather than accidental: the decoder can trade the over-commitment back and pays macro-F1 for it, so removing the cost without paying elsewhere is acoustic-model work, not decoder work.
- **Held-out means held out from selection, not necessarily from the music.** Splits are assigned per track id, so a remix and its original can land on opposite sides -- the artist guard protects the frozen benchmark, and nothing yet groups a corpus track with its own alternate versions. The measured effect on the verdict is negligible (it is a couple of tracks, and the result is unchanged with them removed), but the v2 split should group by song rather than by track id.
- **Nothing here is runnable, and the gap is larger than an integration task.** Three things gate a live show, in descending order of size:
  1. **Live downbeat tracking does not exist.** The decoder is bar-rate -- every decision, duration floor and boundary read is expressed in bars -- and offline it is handed an expert-annotated bar grid. `lib/` has no downbeat tracker at all, and the training table records bar position as unknown for exactly this reason. So the measured numbers are what the chain achieves *given a correct grid*, and acquiring one live is unbuilt work that belongs in its own plan. The grid is not a neutral input and not an answer key either: annotated boundaries genuinely do sit on downbeats, but there are dozens of downbeat candidates per real boundary and choosing between them is the whole of the model's job.
  2. **The show's look-ahead must grow to the decoder's budget**, in lockstep with `playback_delay_seconds` in dmx-enttec-node (`LOOK_AHEAD_SEC` in `lib/main.py`). The two systems are not latency-matched today, so the offline table compares a system running at the current delay against one designed for a larger one; the lag sweep varies only the committer's latency and must not be read as "the NN still wins at today's budget".
  3. **v1 trained on the subset that had finished downloading.** The full-corpus retrain is v2, and the split assignment was built to make the two comparable -- new tracks may only be *added* to a split, never moved between them.

**Data location.** Everything lands in `training/data/raveform/` -- annotations, `manifest.csv`, `audio/`, the download state files, and the build outputs (`reports/`, `features/`, `training_table.csv.gz`, `splits.json`, and later `models/` and `posteriors/`) -- and is gitignored: the audio is ~13 GiB and is never committed. The committed `.gitignore` covers `training/data/`.

---

## Known Issues / Gotchas

- **Hardcoded overlay IP** â€” must change per venue in `overlay_client.py`.
- **YAMNet divide-by-zero**: safe (returns empty list) when MAD == 0, but worth noting.
- **MusicAnalyser full reset** every 15 min prevents rolling-window memory growth.
- **10 ms delays** between MIDI commands give SoundSwitch hardware time to settle.
- **Os2lSender** runs in a separate thread; the audio/DSP path is async on the main thread â€” mixing threading models requires care when touching shared state.
- **Beat dropout false ATMOSPHERIC**: aubio can miss beats during heavy sidechain compression. The beat-absence threshold guards against single-beat dropouts but not sustained compression artifacts.
- **ATMOSPHERIC never fires on mastered EDM.** Measured across the whole downloaded corpus: not one atmospheric block in any committed timeline, while ~22% of labelled time is `intro` or `outro`. ATMOSPHERIC is the only intent driven by a beat-absence timer rather than by classification, and mastered EDM intros and outros have beats -- so the timer never trips, and the two quiet label classes are unreachable no matter how the thresholds move. Whatever replaces the classifier has to be able to *say* "quiet", not merely notice that beats stopped.
- **Weak YAMNet changes are now always accepted** (previously gated on Spotify section proximity). May cause more false-positives in stable sections. The cooldown constant is the main guard.
- **Density trend warmup**: `get_onset_density_trend()` returns neutral until enough beat-density samples have been collected. BUILDUP cannot be detected during this initial window.
- **Sub-bass gate disabled**: `_DROP_MIN_SUB_BASS_RATIO` is set to 0.0 (gate open). Calibrate against real hi-hat-only vs. kick+bass passages before enabling.
- **Thresholds are fitted to one track**: the classifier's constants sit between populations measured on the retired Generate anchor. They are a hypothesis until re-measured on a wider corpus â€” see `lib/analyser/CLAUDE.md` (Known Limitations).
- **Fast genres fold to half tempo**: BPM octave folding puts drum & bass and faster material below DROP's BPM floor, so their drops cannot classify as DROP. Accepted for Stage 1.
- **Kick strength lags one beat**: a beat's kick value is not final until a few buffers after the beat fires (the filterbank's group delay), so each beat record carries the previous beat's measurement. Irrelevant to the multi-second classification window; relevant if you ever want a single-beat trigger.
- **Decode cache**: `simulate file` writes `<song>.<samplerate>.npy` beside the audio file (gitignored). Stale caches are detected by mtime; delete the `.npy` to force a re-decode. The ten eval-set tracks keep theirs on purpose (see The benchmark); everything else in the corpus is cleaned up by the batch that created it.
- **Integration tests are green on a fresh clone.** The eval-set audio and labels are committed, so `uv run pytest` needs no corpus and no downloads. Everything else under `training/` still does.
- **The eval-set baseline lags a deliberate pipeline change by one command.** Any change to `lib/` or `simulate/` that moves the reports fails `run_eval_set.py` until the baseline is re-cut. That is the gate working, not a flake — but it does mean a pipeline PR is two steps, and the second one must not be skipped.
