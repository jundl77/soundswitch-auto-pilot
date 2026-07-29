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

# Full suite including unit + integration tests (a few minutes: the integration
# tests run whole tracks through the real rhythm networks, so this is minutes,
# not seconds. It is not hung.)
uv run pytest

# Run a single test file
uv run pytest tests/test_delayed_command_queue.py -v
```

The integration tests in `tests/test_simulation.py` run the bundled sample track through the full pipeline (identical code path to production) and assert the evaluator's verdict, command-timing exactness, flush behaviour, report duration, speed, and byte-identical determinism. If they fail, the pipeline is broken. There is one simulation mode — real audio files — paced either sped-up (default) or real-time (`--ui`).

### Testing philosophy

- **Coverage over completeness**: aim for broad, confident coverage of critical logic â€” not 100% line coverage. Tests should catch real regressions, not just pad numbers.
- **Test the logic, not the wiring**: unit tests target pure functions and isolated methods. Integration tests verify the full pipeline assembles correctly.
- **Missing deps**: if a package is declared in `pyproject.toml` but absent from the venv, run `uv sync --extra dev --extra visualizer` â€” do not mock it.
- **Every PR must pass `uv run pytest`** (the full suite, not just unit tests) before merge.

---

## What It Does

1. Reads audio from a microphone/line input
2. Extracts rhythm via madmom's online neural trackers (beats, BPM, onsets) and spectral features via Aubio's mel filterbank
3. Detects musical section changes via a YAMNet TensorFlow embedding + cosine similarity outlier detection
4. Classifies audio energy as a `LightIntent` (ATMOSPHERIC / BREAKDOWN / GROOVE / BUILDUP / DROP / PEAK)
5. Selects and sends MIDI lighting effects to SoundSwitch based on intent; also sends OS2L beat events to VirtualDJ and DMX overlays via UDP

---

## Architecture

```
PyAudio â†’ MusicAnalyser (madmom rhythm + Aubio bank) â†’ LightEngine (IMusicAnalyserHandler)
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
| `lib/analyser/madmom_rhythm.py` | `MadmomRhythm` -- madmom's online beat/onset stack, adapted from the pipeline's buffer size to madmom's frame rate; the only place that framing mismatch exists |
| `lib/analyser/drift_watchdog.py` | `DriftWatchdog` -- measures the loop's lost lead against the live input and sheds work, cheapest loss first, when it stops keeping up |
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

Classification uses BPM (octave-folded into one tempo band, so a half- or double-tempo lock cannot fake a high-energy moment), onset density (rhythmic busyness), onset density trend (rising vs. falling energy), kick strength (how far sub-bass rises above its own floor on the beat â€” distinguishes a kick drum from a hi-hat-only or pad-driven passage), and spectral centroid trend (rising centroid = riser/BUILDUP sweep). See `lib/analyser/CLAUDE.md` for the full feature breakdown and design rationale.

**Kick strength carries the classification.** On real material onset density barely varies across a track's sections â€” it says whether anything rhythmic is happening, not how big the moment is. The kick is what separates a drop from an intro, so it gates DROP and, by its absence, widens BREAKDOWN. Branches are tested DROP â†’ BUILDUP â†’ BREAKDOWN â†’ GROOVE; BUILDUP sits above BREAKDOWN because a riser strips the kick by design and would otherwise be swallowed by the kick-absence branch. An unmeasured or unmeasurable kick reads as *absent*, never assumed present.

**Windowed look-ahead:** the engine runs 2.5 s ahead of what the audience hears (matching `playback_delay_seconds` in dmx-enttec-node). Each beat is classified using a symmetric window of past *and* future beats, giving more confident classifications than a causal-only approach. This is why a single anomalous beat cannot flip the intent: it is outvoted by its neighbours via median density.

**Stability pipeline:** each commit runs windowed classification, then the PEAK promotion check, then three guards that can each veto an effect change â€” a vote buffer (consensus required), a minimum dwell check (can't switch away immediately), and an invalid-transition guard (musically impossible jumps blocked). `LightEngine._commit_intent` documents the full ordered sequence; see `lib/analyser/CLAUDE.md` for rationale and `lib/engine/light_engine.py` for constants. A change that reaches consensus is recorded on the intent timeline even when a guard then blocks the effect switch, so `intent_changes_count` reads the classifier's opinion and `effect_changes_count` reads the show.

**ATMOSPHERIC and PEAK are the two intents the beat classifier never returns.** ATMOSPHERIC fires from a beat-absence timer in the 100 ms callback; the first beat after silence immediately re-classifies and changes the effect.

**PEAK is an engine-level promotion, not a classification.** "Sustained maximum energy after the drop" is a temporal property that feature thresholds cannot express (a peak window and a drop window look identical), so the engine promotes an already-committed DROP to PEAK once it has survived a fixed number of commit-beats. While PEAK is current, DROP votes are absorbed so the pair cannot oscillate and the reported intent timeline keeps reading PEAK; PEAK also inherits DROP's hysteresis, so a density dip DROP would ride out cannot eject it. Any other consensus exits PEAK through the normal stability pipeline. Rationale in `lib/analyser/CLAUDE.md`, constant in `lib/engine/light_engine.py`.

**Look-ahead delay** (`LOOK_AHEAD_SEC`) must always match `playback_delay_seconds` in dmx-enttec-node. It is defined in `lib/main.py` and `simulate/runner.py`. Local debug audio playback is delayed by the same amount so headphone monitoring stays in sync.

**Fast simulation:** file simulation runs on a virtual clock driven by audio sample position instead of the wall clock â€” the full pipeline (identical code path to production) processes a track several times faster than real-time and deterministically: the same file always produces byte-identical reports (RNG seeded, no wall-clock jitter). Beat timestamps are song-position seconds; intent/effect blocks are stamped when the audience hears them â€” one look-ahead delay after the beats that caused them â€” so expect intent blocks to trail the track structure by that delay when reading reports. The report records that offset in its metrics, so consumers can realign the two time bases without hardcoding the constant. The decoded audio is cached beside the source file (`*.npy`, gitignored) to skip repeat decodes. Real OS scheduler jitter is only observable in `--ui` / realtime modes, which still run on the system clock.

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

# Simulation (no hardware required)
python auto_pilot simulate file samples/song.mp3          # fast headless: full song in seconds, report + evaluation
python auto_pilot simulate file samples/song.mp3 --ui     # real-time paced with live Dash timeline
python auto_pilot simulate realtime                       # microphone input with live Dash timeline

# Inspect a report: per-10s feature bins, intent timeline, distribution
python auto_pilot simulate file samples/song.mp3 --report report.json
python training/inspect_report.py report.json

# Tests
uv run pytest -m "not integration"   # fast unit tests only
uv run pytest                        # unit + integration (minutes, not seconds)
```

**Flags (`run`):**
- `-i / -o` â€” audio device indices from `list`; passing `-o` enables delayed audio monitoring on that device
- `-d` â€” debug: adds a click on every detected BEAT to the monitored audio (implies monitoring on the default output if `-o` is not given). Beat-triggered by owner preference, not the note-triggered click aubio used to drive â€” roughly half the clicks, landing on the pulse rather than on every onset.
- `--no-os2l` â€” disable VirtualDJ connection
- `--ui` â€” launch Dash real-time visualizer at http://localhost:8050
- `--ui-port N` â€” change visualizer port
- `--report FILE` â€” write JSON session report on exit

---

## ML / DSP Components

- **madmom** -- all rhythm: beat tracking (a recurrent-network ensemble feeding a
  dynamic Bayesian network), BPM, and onsets. **Online mode only**, and that is
  load-bearing: madmom's offline decoders score better and cannot run live, so a
  number produced by one is a number the runtime can never reproduce. Chosen over
  aubio on a measured decoded comparison rather than reputation. The basis and
  the measured effect on the show are in `docs/migration-evidence.md` and
  `training/migration_deltas.json`; the onset operating point's raw sweep and its
  stability analysis are in `training/onset_operating_point.json`, with every
  draw taken (including the wrong ones) in `training/onset_operating_point_draws.md`.
- **Aubio** -- the 40-band Slaney mel filterbank and the FFT that feeds it, and
  nothing else. Every trained model and every spectral feature (kick strength,
  sub-bass ratio, centroid trend) is built on this exact bank, so it is held
  byte-stable by a golden test rather than reimplemented. The two-library split
  is interim by owner decision: the next retrain generation bakes off front-ends
  and may consolidate. Replacing the bank now would invalidate the trained stack
  for no measured gain; replacing the rhythm source roughly doubled downbeat F1
  at the operating point a show actually uses.
- **YAMNet (TensorFlow Hub)** â€” Google's pre-trained audio classifier; used here for embeddings only (not tag predictions). Cosine similarity + MAD-based outlier detection finds section transitions. Degrades gracefully if model fails to load.

---

## Training Corpus (Raveform)

The direction of travel is to replace hand-tuned classification thresholds with a model trained on real, expert-labelled EDM structure. The corpus for that is **Raveform** (Hugging Face, `taejunkim/raveform`): 1,423 EDM tracks with expert section annotations plus a per-track beat grid. Only the annotations are distributed -- the audio is not, so each track is fetched per YouTube ID for research use.

The pipeline is a set of stdlib-only scripts in `training/raveform/`, each resumable and safe to re-run. They are run as scripts, not imported as a package -- each adds its own directory to `sys.path` so the siblings import by plain module name:

| Script | Role |
|---|---|
| `raveform_fetch_annotations.py` | pull and schema-validate the annotation archive from Hugging Face |
| `raveform_manifest.py` | annotations -> `manifest.csv`, plus the label / duration / transition statistics |
| `raveform_download.py` | manifest -> one mp3 per YouTube ID, sequential and resumable |
| `raveform_supervisor.py` | unattended patient resume: relaunches the downloader across escalating cool-downs after a refusal wall |
| `build_clean_manifest.py` | manifest + downloaded audio -> `clean_manifest.csv`, the trusted subset everything downstream reads |
| `raveform_validate.py` | manifest + audio + download state -> `validation_report.{json,txt}` and `checksums.sha256`: the acquisition-complete verdict |

`clean_manifest.csv` is the boundary between "audio we happen to have" and "audio we are willing to learn from". Only its `ok` rows may feed a training table or an evaluation run.

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
- **Prerequisites.** `yt-dlp` and `ffmpeg` on PATH, plus a JavaScript runtime (Deno or Node) *visible to the running process*. Installed is not the same as visible: a shell or detached process started before the install inherits a stale environment block, so check from the actual runner, not from a fresh terminal. Downloads **can** fail with `HTTP Error 403: Forbidden` — not *will*: the whole 1,387-track corpus was fetched under exactly this setup, and every 403 that occurred was cleared by patient re-attempts.
- **Running it detached, and stopping it.** The long sweeps run as detached OS processes that outlive the session. Two things about that are not guessable and cost real data when guessed wrong. **Stopping requires `taskkill /PID <pid> /T`** — the venv's `python.exe` is a trampoline that re-execs the real interpreter, so killing the recorded PID alone leaves the actual downloader orphaned and still fetching, invisible to the next run's bookkeeping. **Re-issuing the detached launch command truncates `download.log`** — the redirect reopens the file, so a relaunch silently destroys the previous run's evidence. Redirect to a new filename per cycle (the supervisor does this: `download.cycle<N>.log`).
- **The corpus data directory holds *ops copies* of the scripts, and they drift.** Both `raveform_download.py` and `raveform_supervisor.py` are copied next to the corpus, and a supervised refresh runs **those copies, not the branch**. So whenever either changes, refresh both and confirm it: `cmp <data-dir>/raveform_download.py training/raveform/raveform_download.py` and the same for `raveform_supervisor.py` — they must be byte-identical to the branch blob. This is not hypothetical. The supervisor used to live *only* in the gitignored data directory, unversioned and unreviewable, and the ops downloader beside it carried the pre-`http_403` classifier while the supervisor hardcoded `--retry-reasons bot_check`; the next refresh would have stranded every recoverable 403 while the branch looked correct. Both files are now on the branch, and the supervisor reads `RETRYABLE_REASONS` out of the downloader sitting beside it rather than keeping its own copy — but the copy step itself is still manual, which is why the `cmp` is written down.

**Data location.** Everything lands in `training/data/raveform/` -- annotations, `manifest.csv`, `audio/`, and the download state files -- and is gitignored: the audio is ~13 GiB and is never committed. The committed `.gitignore` covers `training/data/`; until this branch merges to `master`, the main checkout relies on a local `.git/info/exclude` entry as a bridge.

---

## Known Issues / Gotchas

- **Hardcoded overlay IP** â€” must change per venue in `overlay_client.py`.
- **YAMNet divide-by-zero**: safe (returns empty list) when MAD == 0, but worth noting.
- **MusicAnalyser full reset** every 15 min prevents rolling-window memory growth.
- **10 ms delays** between MIDI commands give SoundSwitch hardware time to settle.
- **Os2lSender** runs in a separate thread; the audio/DSP path is async on the main thread â€” mixing threading models requires care when touching shared state.
- **Beat dropout false ATMOSPHERIC**: a beat tracker can miss beats during heavy sidechain compression. The beat-absence threshold guards against single-beat dropouts but not sustained compression artifacts. Measured far less often since the madmom migration, but not eliminated.
- **Weak YAMNet changes are now always accepted** (previously gated on Spotify section proximity). May cause more false-positives in stable sections. The cooldown constant is the main guard.
- **Density trend warmup**: `get_onset_density_trend()` returns neutral until enough beat-density samples have been collected. BUILDUP cannot be detected during this initial window.
- **Sub-bass gate disabled**: `_DROP_MIN_SUB_BASS_RATIO` is set to 0.0 (gate open). Calibrate against real hi-hat-only vs. kick+bass passages before enabling.
- **Thresholds are fitted to one track**: the classifier's constants sit between populations measured on the single bundled sample. They are a hypothesis until re-measured on a wider corpus â€” see `lib/analyser/CLAUDE.md` (Known Limitations).
- **Fast genres fold to half tempo**: BPM octave folding puts drum & bass and faster material below DROP's BPM floor, so their drops cannot classify as DROP. Accepted for Stage 1.
- **Kick strength lags one beat**: a beat's kick value is not final until a few buffers after the beat fires (the filterbank's group delay), so each beat record carries the previous beat's measurement. Irrelevant to the multi-second classification window; relevant if you ever want a single-beat trigger.
- **The rhythm front-end costs ~18x what aubio did** end to end (1.4% of one core
  to 25.7%, filterbank included); the rhythm half alone went 0.8% to 25.2%. On a
  strict reading of the campaign's >= 5x / <= 20% realtime bar, 25.7% is a MISS
  (3.9x). That bar was written for NN posterior generation, where 5x buys the
  rest of the stack its headroom on one core, and it was ruled not to bind a DSP
  front-end; the front-end's gates are instead sustained 1x whole-pipeline with
  headroom, the backpressure machinery, and a 30-minute soak, all of which pass.
  Both readings belong in any future discussion -- the miss is real under the
  original wording.
- **Fast simulation is ~12x slower** (46.3x real-time to 3.8x on the bundled
  track). Regenerating the whole corpus report cache is therefore ~41 CPU-hours
  against ~3.4 before, or roughly 3.5 hours of wall clock at 12 workers: track
  parallelism recovers the wall-clock, per-core throughput is what fell.
- **Backpressure is monitored, not assumed**: live audio arrives at exactly 1x and
  the input side DROPS rather than queues, so falling behind costs audio, not
  latency. The drift watchdog sheds section detection first and onsets second, and
  never beats. If a log shows sustained shedding, the box is too slow for the
  configuration -- that is the signal, not a nuisance warning.
- **Shedding degrades explicitly, it does not fake a measurement.** A shed onset
  detector reports density as UNKNOWN rather than zero, and the classifier holds
  the current intent instead of classifying on a sentinel -- zero density would
  otherwise pin the show to BREAKDOWN, with BUILDUP and DROP unreachable, and the
  report rows would be indistinguishable from a genuinely sparse passage.
  Restoring either shed component clears its buffers first, because everything
  they hold describes audio from before the gap.
- **madmom is CC BY-NC-SA** (models). Fine for a personal project; it forecloses a
  commercial turn without a JKU licence. aubio's GPL note stands.
- **madmom is pinned to a git SHA**, not a release: the last PyPI release cannot be
  imported on Python 3.10+. An upgrade is deliberate and `tests/test_madmom_contract.py`
  is what makes it a checked one.
- **Decode cache**: `simulate file` writes `<song>.<samplerate>.npy` beside the audio file (gitignored). Stale caches are detected by mtime; delete the `.npy` to force a re-decode.
