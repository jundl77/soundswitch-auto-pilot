# SoundSwitch Auto Pilot

Intelligent DJ lighting automation system that synchronizes stage effects to live music in real time. Analyzes audio → detects beats/sections/energy → classifies `LightIntent` → controls SoundSwitch lighting via MIDI and OS2L protocol.

---

## Development Workflow

**All changes must go through a pull request.** Never commit directly to `master`.

**Every PR must include an update to `CLAUDE.md`** reflecting any architectural, interface, or behavioural changes made in that PR.

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
- **Test the logic, not the wiring**: unit tests target pure functions and isolated methods (e.g. `_classify_intent`, `get_onset_density_trend`). Integration tests verify the full pipeline assembles correctly.
- **Missing deps**: if a package is declared in `pyproject.toml` but absent from the venv, run `uv sync` — do not mock it.
- **Every PR must pass `uv run pytest`** (the full suite, not just unit tests) before merge.

---

## What It Does

1. Reads audio from a microphone/line input at 44.1 kHz, 256-sample buffers (~5.8 ms windows)
2. Extracts musical features via Aubio (pitch, BPM, onsets, notes, MFCCs)
3. Detects musical section changes via a YAMNet TensorFlow embedding + cosine similarity outlier detection
4. Classifies audio energy as a `LightIntent` (ATMOSPHERIC / BREAKDOWN / GROOVE / BUILDUP / DROP / PEAK) from real-time BPM + onset density + density trend
5. Selects and sends MIDI lighting effects to SoundSwitch based on intent; also sends OS2L beat events to VirtualDJ and DMX overlays via UDP

---

## Architecture

```
PyAudio → MusicAnalyser (Aubio DSP) → LightEngine (IMusicAnalyserHandler)
                   ↓                          ↓               ↓
        YamnetChangeDetector          EffectController    MIDI / OS2L / Overlay
                                             ↑
                                       LightIntent
                              (BPM + onset density + trend)
```

### Key Files

| Path | Role |
|---|---|
| `auto_pilot` | CLI entry point (`run MIDI_PORT`, `list`, `simulate`) |
| `lib/main.py` | `SoundSwitchAutoPilot` — async event loop, 100 ms / 1 s / 10 s callbacks |
| `lib/analyser/music_analyser.py` | `MusicAnalyser` — per-buffer DSP, beat/onset/note events, YAMNet trigger |
| `lib/analyser/yamnet_change_detector.py` | `YamnetChangeDetector` — TF Hub YAMNet embeddings, MAD outlier detection, 10 s cooldown |
| `lib/engine/light_engine.py` | `LightEngine` — routes DSP events → intent → MIDI / OS2L / overlay commands |
| `lib/engine/effect_controller.py` | `EffectController` — maps `LightIntent` → non-repetitive random MIDI channel selection |
| `lib/engine/effect_definitions.py` | `LightIntent` enum + `INTENT_EFFECTS` mapping (the single place to change intent→MIDI routing) |
| `lib/engine/event_buffer.py` | Thread-safe beat/effect/intent store; read by Dash visualizer every 100 ms |
| `lib/clients/midi_client.py` | MIDI note-on/off to SoundSwitch; 90+ channels, delayed deactivation |
| `lib/clients/os2l_client.py` | zeroconf discovery of VirtualDJ; bidirectional OS2L JSON; 25 ms beat position updates |
| `lib/clients/pyaudio_client.py` | Mono 44.1 kHz audio input (and optional debug output passthrough) |
| `lib/clients/overlay_client.py` | UDP binary DMX overlay to 192.168.178.245:19001 (hardcoded — must match venue) |
| `simulate/visualizer_app.py` | Dash real-time visualizer: timeline, intent-based stage simulation, metrics |
| `simulate/runner.py` | Simulation runner — stub clients, full pipeline, timing report |
| `simulate/cli.py` | `auto_pilot simulate file|realtime` subcommands |
| `lib/visualizer/` | Optional matplotlib spectrogram UI via multiprocessing + TCP |

---

## LightIntent System

`LightIntent` is the semantic bridge between audio analysis and lighting output. It lives in `lib/engine/effect_definitions.py`.

### Intent classifier (in `lib/engine/light_engine.py`)

Classification uses three signals: BPM, onset density (onsets/sec over a 1.5 s rolling window), and onset density trend (ratio of recent vs past beats, from `get_onset_density_trend()`).

| Intent | Detection method | MIDI pool | Visualizer fixtures |
|---|---|---|---|
| ATMOSPHERIC | Beat absence > 2.5 s (via `on_100ms_callback`) | BANK_2A/B/C | 2 center (deep blue/violet) |
| BREAKDOWN | density < 3.0 onsets/s | BANK_2C/D/E | 3 center (purple/rose) |
| GROOVE | density ≥ 3.0 and trend < 1.3 | BANK_2F/G/H | 5 spread (teal/sky) |
| BUILDUP | density ≥ 3.0 and trend ≥ 1.3 (rising energy) | BANK_1A/B/C | 6 fixtures (amber/gold) |
| DROP | density ≥ 8.0 and BPM ≥ 100 | BANK_1D/E + STROBE | 8 all (crimson/magenta) |
| PEAK | BPM ≥ 138 | BANK_1F/G/H | 8 all (white-hot/red) |

**ATMOSPHERIC** is the only intent set outside `_classify_intent`: `on_100ms_callback` detects beat absence (> 2.5 s without a beat), fires ATMOSPHERIC once via `_atmospheric_sent` flag (not every 100 ms), and triggers a MIDI effect change. The first beat after ATMOSPHERIC immediately re-classifies and changes the MIDI effect.

### DMX migration path

When moving away from SoundSwitch to direct DMX:
- Replace `EffectController._apply_autoloop(effect)` with a `_send_dmx(intent)` call
- Everything above (`YAMNet → intent classification → EventBuffer`) stays unchanged
- `INTENT_EFFECTS` dict in `effect_definitions.py` becomes the only thing to remove

---

## Data Flow

```
Audio buffer (256 samples)
  → Aubio: pitch, BPM, onset confidence, MFCCs, mel energies
  → onset detected? → _onset_times.append(now) → on_onset()
  → beat detected? → _density_samples.append(density) → LightEngine.on_beat()
      → _classify_intent(bpm, onset_density, density_trend) → LightIntent
      → on first beat OR returning from ATMOSPHERIC: EffectController.change_effect(intent)
      → EventBuffer.set_intent() / add_beat()
      → OS2L beat (via DelayedCommandQueue if delay > 0)
  → note detected? → LightEngine.on_note() → DMX overlay
  → YAMNet buffer full (4096 samples)?
      → embed → cosine similarity → outlier? → section change
      → LightEngine.on_section_change()
          → _classify_intent(bpm, onset_density, density_trend) → EffectController.change_effect(intent)
  → every 100 ms: on_100ms_callback()
      → if no beat for > 2.5 s: set ATMOSPHERIC intent + fire MIDI effect once
```

---

## Running

```bash
# Install dependencies (requires uv: https://github.com/astral-sh/uv)
uv sync
uv sync --extra dev   # include pytest etc.

# List available MIDI and audio devices
python auto_pilot list

# Minimal run (MIDI port 0, default audio device)
python auto_pilot run 0

# Run with real-time Dash visualizer
python auto_pilot run 0 --ui

# Full options
python auto_pilot run 0 -i INPUT_DEVICE_IDX -o OUTPUT_DEVICE_IDX --no-os2l --ui

# Simulation (no hardware required)
python auto_pilot simulate file samples/song.mp3 --delay 0.3
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

## Key Constants & Tuning Knobs

| Constant | Location | Value | Meaning |
|---|---|---|---|
| `BUFFER_SIZE` | `pyaudio_client.py` | 256 | Audio frames per callback |
| `SAMPLE_RATE` | `music_analyser.py` | 44100 | Hz |
| `_ONSET_DENSITY_WINDOW_SEC` | `music_analyser.py` | 1.5 s | Rolling window for onset density |
| `_BREAKDOWN_MAX_DENSITY` | `light_engine.py` | 3.0 | onsets/s ceiling for BREAKDOWN |
| `_BUILDUP_MIN_TREND` | `light_engine.py` | 1.3 | Density trend ratio floor for BUILDUP |
| `_DROP_MIN_DENSITY` | `light_engine.py` | 8.0 | onsets/s floor for DROP |
| `_PEAK_MIN_BPM` | `light_engine.py` | 138 | BPM floor for PEAK |
| `_BEAT_ABSENCE_SEC` | `light_engine.py` | 2.5 s | Beat silence threshold for ATMOSPHERIC |
| `SECTION_CHANGE_COOLDOWN` | `yamnet_change_detector.py` | 10 s | Min gap between YAMNet-triggered changes |
| `APPLY_COLOR_OVERRIDE_INTERVAL_SEC` | `effect_controller.py` | 300 | Color override rotation every 5 min |
| OS2L beat update | `os2l_client.py` | 25 ms | Beat position broadcast interval |

---

## ML / DSP Components

- **Aubio** — real-time pitch, BPM, onset, note, MFCC. All tuned for 44.1 kHz / 256-sample windows.
- **YAMNet (TensorFlow Hub)** — Google's pre-trained audio classifier; used here for 1024-dim embeddings only (not tags). Cosine similarity + MAD-based outlier detection across a 2 s lookback finds section transitions. Degrades gracefully if model fails to load (logs a warning, section detection disabled).

---

## Known Issues / Gotchas

- **Hardcoded overlay IP** (`192.168.178.245:19001`) — must change per venue in `overlay_client.py`.
- **YAMNet divide-by-zero**: safe (returns empty list) when MAD == 0, but worth noting.
- **MusicAnalyser full reset** every 15 min prevents rolling-window memory growth.
- **10 ms delays** between MIDI commands give SoundSwitch hardware time to settle.
- **Os2lSender** runs in a separate thread; the audio/DSP path is async on the main thread — mixing threading models requires care when touching shared state.
- **Beat dropout false ATMOSPHERIC**: aubio can miss beats during heavy sidechain compression. The 2.5 s threshold (≈5 missed beats at 128 BPM) guards against single-beat dropouts but not sustained compression artifacts.
- **Weak YAMNet changes are now always accepted** (previously gated on Spotify section proximity). May cause more false-positives in stable sections. The 10 s cooldown is the main guard.
- **Density trend needs 4 beats to warm up**: `get_onset_density_trend()` returns 1.0 (neutral) until 4 beat-density samples have been collected. During this window BUILDUP cannot be detected via trend; it falls through to GROOVE.
