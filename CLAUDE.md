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

# Full suite including unit + integration tests (~8s)
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
| PEAK | Sustained maximum energy after the drop | BANK_1F/G/H |

For the specific thresholds and tuning constants that drive classification, see `lib/engine/light_engine.py` and `lib/analyser/CLAUDE.md`.

### How classification works

Classification uses BPM, onset density (rhythmic busyness), onset density trend (rising vs. falling energy), kick strength (beat-synchronous sub-bass ratio â€” distinguishes kick drum from hi-hat-only patterns), and spectral centroid trend (rising centroid = riser/BUILDUP sweep). See `lib/analyser/CLAUDE.md` for the full feature breakdown and design rationale.

**Windowed look-ahead:** the engine runs 2.5 s ahead of what the audience hears (matching `playback_delay_seconds` in dmx-enttec-node). Each beat is classified using a symmetric window of past *and* future beats, giving more confident classifications than a causal-only approach. This is why a single anomalous beat cannot flip the intent: it is outvoted by its neighbours via median density.

**Stability pipeline:** classification changes pass through three guards before triggering an effect change â€” a vote buffer (consensus required), a minimum dwell check (can't switch away immediately), and an invalid-transition guard (musically impossible jumps blocked). See `lib/analyser/CLAUDE.md` for rationale and `lib/engine/light_engine.py` for constants.

**ATMOSPHERIC** is the only intent not driven by the beat classifier. It fires from a beat-absence timer in the 100 ms callback. The first beat after silence immediately re-classifies and changes the effect.

**Look-ahead delay** (`LOOK_AHEAD_SEC`) must always match `playback_delay_seconds` in dmx-enttec-node. It is defined in `lib/main.py` and `simulate/runner.py`. Local debug audio playback is delayed by the same amount so headphone monitoring stays in sync.

**Fast simulation:** file simulation runs on a virtual clock driven by audio sample position instead of the wall clock â€” the full pipeline (identical code path to production) processes a track ~30â€“50Ã— faster than real-time and deterministically: the same file always produces byte-identical reports (RNG seeded, no wall-clock jitter). Beat timestamps are song-position seconds; intent/effect blocks are stamped when the audience hears them â€” one look-ahead delay after the beats that caused them â€” so expect intent blocks to trail the track structure by that delay when reading reports. The decoded audio is cached beside the source file (`*.npy`, gitignored) to skip repeat decodes. Real OS scheduler jitter is only observable in `--ui` / realtime modes, which still run on the system clock.

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

# Tests
uv run pytest -m "not integration"   # fast unit tests only
uv run pytest                        # unit + integration (~6s)
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

The pipeline is a set of stdlib-only scripts in `training/`, each resumable and safe to re-run:

| Script | Role |
|---|---|
| `raveform_fetch_annotations.py` | pull and schema-validate the annotation archive from Hugging Face |
| `raveform_manifest.py` | annotations -> `manifest.csv`, plus the label / duration / transition statistics |
| `raveform_download.py` | manifest -> one mp3 per YouTube ID, sequential and resumable |

A corpus validation pass (every downloaded track decodes, and its length agrees with its annotation) is the next step and does not exist yet.

Decisions that belong here rather than in the code:

- **Canonical label vocabulary.** The published labels are a superset of the documented seven. `end` is a tail sentinel, not a musical section, and is dropped; `altintro` folds into `intro` and `bridge` into `breakdown` (too few examples to learn as a class); `altoutro` is kept. Adjacent same-label sections are merged *after* the drop and the fold, so a canonical "section" means one contiguous stretch of one label. The rationale and the mapping live in `raveform_manifest.py`.
- **Politeness over throughput.** Downloads are strictly sequential with a pause between videos -- pulling 1,423 tracks in parallel is indistinguishable from abuse. Bot checks are never worked around: no cookies, credentials or IP tricks. A refusal is recorded, reported, and left as an owner decision, and a run of consecutive refusals aborts rather than burning the manifest into failure records.
- **Resumability is the contract.** The downloader is safe to kill at any point: finished tracks and failures are both persisted, an interrupted track is never recorded as failed, and a track counts as downloaded only when a non-empty mp3 is actually on disk. See the module docstring in `raveform_download.py`.
- **Prerequisites.** `yt-dlp` and `ffmpeg` on PATH, plus a JavaScript runtime (Deno) -- without one, yt-dlp warns that YouTube extraction is deprecated and some formats may be missing, which a full-corpus run cannot accept.

**Data location.** Everything lands in `training/data/raveform/` -- annotations, `manifest.csv`, `audio/`, and the download state files -- and is gitignored: the audio is ~13 GiB and is never committed. The committed `.gitignore` covers `training/data/`; until this branch merges to `master`, the main checkout relies on a local `.git/info/exclude` entry as a bridge.

---

## Known Issues / Gotchas

- **Hardcoded overlay IP** â€” must change per venue in `overlay_client.py`.
- **YAMNet divide-by-zero**: safe (returns empty list) when MAD == 0, but worth noting.
- **MusicAnalyser full reset** every 15 min prevents rolling-window memory growth.
- **10 ms delays** between MIDI commands give SoundSwitch hardware time to settle.
- **Os2lSender** runs in a separate thread; the audio/DSP path is async on the main thread â€” mixing threading models requires care when touching shared state.
- **Beat dropout false ATMOSPHERIC**: aubio can miss beats during heavy sidechain compression. The beat-absence threshold guards against single-beat dropouts but not sustained compression artifacts.
- **Weak YAMNet changes are now always accepted** (previously gated on Spotify section proximity). May cause more false-positives in stable sections. The cooldown constant is the main guard.
- **Density trend warmup**: `get_onset_density_trend()` returns neutral until enough beat-density samples have been collected. BUILDUP cannot be detected during this initial window.
- **Sub-bass gate disabled**: `_DROP_MIN_SUB_BASS_RATIO` is set to 0.0 (gate open). Calibrate against real hi-hat-only vs. kick+bass passages before enabling.
- **Decode cache**: `simulate file` writes `<song>.<samplerate>.npy` beside the audio file (gitignored). Stale caches are detected by mtime; delete the `.npy` to force a re-decode.
