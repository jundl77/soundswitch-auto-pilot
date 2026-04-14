# SoundSwitch Auto Pilot

Intelligent DJ lighting automation system that synchronizes stage effects to live music in real time. Analyzes audio → detects beats/sections/energy → classifies `LightIntent` → controls SoundSwitch lighting via MIDI and OS2L protocol.

---

## CLAUDE.md Policy

**CLAUDE.md documents intent and architecture, not code.** This applies to every CLAUDE.md in this repo — root or subdirectory.

- Do not replicate threshold values, function signatures, or internal variable names.
- Do not duplicate content that is already expressed in code. Point to the file instead.
- CLAUDE.md sits one layer above the code. It explains *why* things work the way they do, not *what* specific values are set to.
- Every PR must update CLAUDE.md to reflect any architectural, interface, or behavioural changes — but only at the intent/meta level.

**CLAUDE.md is the agent's source of truth.** Any analysis ideas, design decisions, or specifications must be recorded here — not just in code or in PR descriptions. An agent must be able to understand what this system does, why it is designed the way it is, and what direction it is heading by reading CLAUDE.md alone, without scanning all source files. Keep it current at all times.

---

## Development Workflow

**All changes must go through a pull request.** Never commit directly to `master`.

Before opening a PR, all tests must pass:

```bash
# Fast unit tests (run these frequently during development)
uv run pytest -m "not integration"

# Full suite including unit + integration tests (~15s)
uv run pytest

# Run a single test file
uv run pytest tests/test_delayed_command_queue.py -v
```

The integration tests in `tests/test_simulation.py` run the full pipeline without hardware. If they fail, the pipeline is broken.

### Testing philosophy

- **Coverage over completeness**: aim for broad, confident coverage of critical logic — not 100% line coverage. Tests should catch real regressions, not just pad numbers.
- **Test the logic, not the wiring**: unit tests target pure functions and isolated methods. Integration tests verify the full pipeline assembles correctly.
- **Missing deps**: if a package is declared in `pyproject.toml` but absent from the venv, run `uv sync --extra dev --extra visualizer` — do not mock it.
- **Every PR must pass `uv run pytest`** (the full suite, not just unit tests) before merge.

---

## What It Does

1. Reads audio from a microphone/line input
2. Extracts musical features via Aubio (pitch, BPM, onsets, notes, MFCCs, mel filterbank energies)
3. Detects musical section changes via a YAMNet TensorFlow embedding + cosine similarity outlier detection
4. Classifies audio energy as a `LightIntent` (ATMOSPHERIC / BREAKDOWN / GROOVE / BUILDUP / DROP / PEAK)
5. Selects and sends MIDI lighting effects to SoundSwitch based on intent; also sends OS2L beat events to VirtualDJ and DMX overlays via UDP

---

## Architecture

```
PyAudio → MusicAnalyser (Aubio DSP) → LightEngine (IMusicAnalyserHandler)
                   ↓                          ↓               ↓
        YamnetChangeDetector          EffectController    MIDI / OS2L / Overlay
                                             ↑
                                       LightIntent
                              (BPM + onset density + sub-bass)
```

### Key Files

| Path | Role |
|---|---|
| `auto_pilot` | CLI entry point (`run MIDI_PORT`, `list`, `simulate`) |
| `lib/main.py` | `SoundSwitchAutoPilot` — async event loop, 100 ms / 1 s / 10 s callbacks |
| `lib/analyser/music_analyser.py` | `MusicAnalyser` — per-buffer DSP, beat/onset/note events, YAMNet trigger |
| `lib/analyser/yamnet_change_detector.py` | `YamnetChangeDetector` — TF Hub YAMNet embeddings, MAD outlier detection |
| `lib/analyser/CLAUDE.md` | Analysis pipeline detail: features, classification design, evaluation strategy |
| `lib/engine/light_engine.py` | `LightEngine` — routes DSP events → intent → MIDI / OS2L / overlay commands; all tuning constants live here |
| `lib/engine/effect_controller.py` | `EffectController` — maps `LightIntent` → non-repetitive random MIDI channel selection |
| `lib/engine/effect_definitions.py` | `LightIntent` enum + `INTENT_EFFECTS` mapping (the single place to change intent→MIDI routing) |
| `lib/engine/event_buffer.py` | Thread-safe beat/effect/intent store; read by Dash visualizer every 100 ms |
| `lib/clients/midi_client.py` | MIDI note-on/off to SoundSwitch; 90+ channels, delayed deactivation |
| `lib/clients/os2l_client.py` | zeroconf discovery of VirtualDJ; bidirectional OS2L JSON |
| `lib/clients/pyaudio_client.py` | Mono 44.1 kHz audio input (and optional debug output passthrough) |
| `lib/clients/overlay_client.py` | UDP binary DMX overlay (hardcoded IP — must match venue) |
| `simulate/visualizer_app.py` | Dash real-time visualizer: timeline, intent-based stage simulation, metrics |
| `simulate/runner.py` | Simulation runner — stub clients, full pipeline, timing report |
| `simulate/cli.py` | `auto_pilot simulate file|realtime` subcommands |
| `lib/visualizer/` | Optional matplotlib spectrogram UI via multiprocessing + TCP |

---

## LightIntent System

`LightIntent` is the semantic bridge between audio analysis and lighting output. It lives in `lib/engine/effect_definitions.py`.

Six intents map to structural moments in an EDM track:

| Intent | Musical moment | MIDI pool |
|---|---|---|
| ATMOSPHERIC | Silence, intro, full breakdown, outro — no beats | BANK_2A/B/C |
| BREAKDOWN | Melodic, stripped, emotional — beats present but sparse | BANK_2C/D/E |
| GROOVE | Steady dance-floor mid-energy — main verse/groove loop | BANK_2F/G/H |
| BUILDUP | Rising tension pre-drop — onset density climbing | BANK_1A/B/C |
| DROP | Maximum impact — bass, kick, full arrangement | BANK_1D/E + STROBE |
| PEAK | Sustained maximum energy after the drop | BANK_1F/G/H |

For the specific thresholds and tuning constants that drive classification, see `lib/engine/light_engine.py` and `lib/analyser/CLAUDE.md`.

### How classification works

Classification uses BPM, onset density (rhythmic busyness), onset density trend (rising vs. falling energy), kick strength (beat-synchronous sub-bass ratio — distinguishes kick drum from hi-hat-only patterns), spectral centroid trend (rising centroid = riser/BUILDUP sweep), sub-bass ratio (low-frequency energy gate for DROP and compound BREAKDOWN), RMS energy (loudness gate for PEAK — currently disabled pending relative RMS), and spectral flux (frame-to-frame mel-energy change, available for future compound rules). See `lib/analyser/CLAUDE.md` for the full feature breakdown and design rationale.

**Compound BREAKDOWN rule:** a secondary BREAKDOWN path fires for sections where the density is above the normal sparse threshold but sub-bass is absent — e.g. a stripped melodic section that has some rhythmic activity but no kick/bass engagement. This rule governs both entry and stay, providing implicit hysteresis: when already in BREAKDOWN with low sub-bass, the rule keeps the system there even if density fluctuates above the normal exit threshold. The compound rule fires strictly above the normal BREAKDOWN entry threshold so the primary hysteresis boundary is not disrupted, and BUILDUP always takes priority over it.

**Windowed look-ahead:** the engine runs 2.5 s ahead of what the audience hears (matching `playback_delay_seconds` in dmx-enttec-node). Each beat is classified using a symmetric window of past *and* future beats, giving more confident classifications than a causal-only approach. This is why a single anomalous beat cannot flip the intent: it is outvoted by its neighbours via median density.

**Stability pipeline:** classification changes pass through three guards before triggering an effect change — a vote buffer (consensus required), a minimum dwell check (can't switch away immediately), and an invalid-transition guard (musically impossible jumps blocked). See `lib/analyser/CLAUDE.md` for rationale and `lib/engine/light_engine.py` for constants.

**ATMOSPHERIC** is the only intent not driven by the beat classifier. It fires from a beat-absence timer in the 100 ms callback. The first beat after silence immediately re-classifies and changes the effect.

**Look-ahead delay** (`LOOK_AHEAD_SEC`) must always match `playback_delay_seconds` in dmx-enttec-node. It is defined in `lib/main.py` and `simulate/runner.py`. Local debug audio playback is delayed by the same amount so headphone monitoring stays in sync.

### DMX migration path

When moving away from SoundSwitch to direct DMX:
- Replace `EffectController._apply_autoloop(effect)` with a `_send_dmx(intent)` call
- Everything above (`YAMNet → intent classification → EventBuffer`) stays unchanged
- `INTENT_EFFECTS` dict in `effect_definitions.py` becomes the only thing to remove

---

## Running

```bash
# Install dependencies (requires uv: https://github.com/astral-sh/uv)
uv sync --extra dev --extra visualizer

# List available MIDI and audio devices
python auto_pilot list

# Minimal run (MIDI port 0, default audio device)
python auto_pilot run 0

# Run with real-time Dash visualizer
python auto_pilot run 0 --ui

# Full options
python auto_pilot run 0 -i INPUT_DEVICE_IDX -o OUTPUT_DEVICE_IDX --no-os2l --ui

# Simulation (no hardware required)
python auto_pilot simulate file samples/song.mp3
python auto_pilot simulate realtime

# Tests
uv run pytest -m "not integration"   # fast unit tests only
uv run pytest                        # unit + integration (~15s)
```

**Flags (`run`):**
- `-i / -o` — audio device indices from `list`
- `-d` — debug: plays audio back with click on beats/notes
- `-v` — show matplotlib visualizer
- `--no-os2l` — disable VirtualDJ connection
- `--ui` — launch Dash real-time visualizer at http://localhost:8050
- `--ui-port N` — change visualizer port
- `--report FILE` — write JSON session report on exit

---

## ML / DSP Components

- **Aubio** — real-time pitch, BPM, onset, note, MFCC, mel filterbank energies. Tuned for low-latency real-time use.
- **YAMNet (TensorFlow Hub)** — Google's pre-trained audio classifier; used here for embeddings only (not tag predictions). Cosine similarity + MAD-based outlier detection finds section transitions. Degrades gracefully if model fails to load.

---

## Known Issues / Gotchas

- **Hardcoded overlay IP** — must change per venue in `overlay_client.py`.
- **YAMNet divide-by-zero**: safe (returns empty list) when MAD == 0, but worth noting.
- **MusicAnalyser full reset** every 15 min prevents rolling-window memory growth.
- **10 ms delays** between MIDI commands give SoundSwitch hardware time to settle.
- **Os2lSender** runs in a separate thread; the audio/DSP path is async on the main thread — mixing threading models requires care when touching shared state.
- **Beat dropout false ATMOSPHERIC**: aubio can miss beats during heavy sidechain compression. The beat-absence threshold guards against single-beat dropouts but not sustained compression artifacts.
- **Weak YAMNet changes are now always accepted** (previously gated on Spotify section proximity). May cause more false-positives in stable sections. The cooldown constant is the main guard.
- **Density trend warmup**: `get_onset_density_trend()` returns neutral until enough beat-density samples have been collected. BUILDUP cannot be detected during this initial window.
- **Sub-bass gate calibrated**: `_DROP_MIN_SUB_BASS_RATIO` and `_DROP_MIN_SUB_BASS_RATIO_EXIT` are calibrated against Eric Prydz "Generate" (128 BPM). They may need re-tuning on other tracks — run the `--sweep` flag against an annotated sample to re-optimize.
- **Compound BREAKDOWN rule**: a secondary BREAKDOWN path fires when density is in the range above the normal sparse threshold but sub-bass is absent — catching "stripped" sections that sit above the simple density cutoff. It governs both entry and stay (not just entry), which suppresses GROOVE↔BREAKDOWN oscillation when density fluctuates but bass remains off. The rule is bypassed for the density boundary case (strictly above entry threshold) so normal hysteresis is not disturbed. If the rule causes false BREAKDOWNs on a track where sub-bass is absent but the section is clearly GROOVE, lower the sub-bass ceiling constant in `light_engine.py`.
- **BUILDUP detection is weak for constant-density buildups**: Generate's buildup section has stable onset density (the energy rise is tonal/spectral, not rhythmic), so the density-trend trigger rarely fires. Centroid-trend works but requires a clear spectral sweep. Tracks where buildups manifest as density rises are detected reliably.
- **DROP→GROOVE transition missed for tight gaps**: when the post-drop groove section has sub-bass close to the DROP exit threshold (windowed mean above the exit gate), the system stays in DROP. This affects tracks like Generate where the groove is energetically similar to the drop. Relative RMS or a spectral flux gate would improve discrimination.
- **UDP overlay crash fixed**: `overlay_client.py` now catches `OSError` in `flush_messages()` and logs a warning rather than propagating the exception. The async event loop is no longer disrupted when the DMX hardware is unreachable.
