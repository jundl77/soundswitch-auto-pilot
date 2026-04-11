# SoundSwitch Auto Pilot

Intelligent DJ lighting automation system that synchronizes stage effects to live music in real time. Analyzes audio → detects beats/sections/energy → classifies `LightIntent` → controls SoundSwitch lighting via MIDI and OS2L protocol.

---

## Development Workflow

**All changes must go through a pull request.** Never commit directly to `master`.

Before opening a PR, all tests must pass:

```bash
# Fast unit tests (run these frequently during development)
pytest -m "not integration"

# Full suite including integration tests (~30s, requires aubio)
pytest

# Run a single test file
pytest tests/test_delayed_command_queue.py -v
```

The integration tests in `tests/test_simulation.py` run the full pipeline without hardware. If they fail, the pipeline is broken.

---

## What It Does

1. Reads audio from a microphone/line input at 44.1 kHz, 256-sample buffers (~5.8 ms windows)
2. Extracts musical features via Aubio (pitch, BPM, onsets, notes, MFCCs)
3. Detects musical section changes via a YAMNet TensorFlow embedding + cosine similarity outlier detection
4. Classifies audio energy as a `LightIntent` (CALM / GROOVE / ENERGY / PEAK) from real-time BPM
5. Selects and sends MIDI lighting effects to SoundSwitch based on intent; also sends OS2L beat events to VirtualDJ and DMX overlays via UDP

---

## Architecture

```
PyAudio → MusicAnalyser (Aubio DSP) → LightEngine (IMusicAnalyserHandler)
                   ↓                          ↓               ↓
        YamnetChangeDetector          EffectController    MIDI / OS2L / Overlay
                                             ↑
                                       LightIntent
                                    (BPM → CALM/GROOVE/ENERGY/PEAK)
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

### BPM → Intent mapping (in `lib/engine/light_engine.py`)

| Intent | BPM range | MIDI pool | Active fixtures (visualizer) |
|---|---|---|---|
| CALM   | < 90      | BANK_2A/B/C | 3 (blue)   |
| GROOVE | 90–119    | BANK_2D/E/F | 5 (teal)   |
| ENERGY | 120–144   | BANK_1A/B/C | 7 (amber)  |
| PEAK   | ≥ 145     | BANK_1D/E + STROBE | 8 (red) |

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
  → beat detected? → LightEngine.on_beat()
      → _bpm_to_intent(bpm) → LightIntent
      → on first beat: EffectController.change_effect(intent)
      → MIDI autoloop / OS2L beat
  → onset detected? → LightEngine.on_onset()
  → note detected? → LightEngine.on_note() → DMX overlay
  → YAMNet buffer full (4096 samples)?
      → embed → cosine similarity → outlier? → section change
      → LightEngine.on_section_change()
          → _bpm_to_intent(bpm) → EffectController.change_effect(intent)
          → select MIDI channel from intent pool (avoids repeats)
          → send MIDI note-on/off to SoundSwitch
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
python auto_pilot simulate realtime --bpm 128 --duration 30

# Tests
pytest -m "not integration"   # fast unit tests only
pytest                        # unit + integration (~30s)
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
| `_CALM_MAX_BPM` | `light_engine.py` | 90 | BPM threshold: CALM → GROOVE |
| `_GROOVE_MAX_BPM` | `light_engine.py` | 120 | BPM threshold: GROOVE → ENERGY |
| `_ENERGY_MAX_BPM` | `light_engine.py` | 145 | BPM threshold: ENERGY → PEAK |
| `SECTION_CHANGE_COOLDOWN` | `yamnet_change_detector.py` | 10 s | Min gap between YAMNet-triggered changes |
| `APPLY_COLOR_OVERRIDE_INTERVAL_SEC` | `effect_controller.py` | 300 | Color override rotation every 5 min |
| OS2L beat update | `os2l_client.py` | 25 ms | Beat position broadcast interval |

---

## ML / DSP Components

- **Aubio** — real-time pitch, BPM, onset, note, MFCC. All tuned for 44.1 kHz / 256-sample windows.
- **YAMNet (TensorFlow Hub)** — Google's pre-trained audio classifier; used here for 1024-dim embeddings only (not tags). Cosine similarity + MAD-based outlier detection across a 2 s lookback finds section transitions.

---

## Known Issues / Gotchas

- **Hardcoded overlay IP** (`192.168.178.245:19001`) — must change per venue in `overlay_client.py`.
- **YAMNet divide-by-zero**: safe (returns empty list) when MAD == 0, but worth noting.
- **MusicAnalyser full reset** every 15 min prevents rolling-window memory growth.
- **10 ms delays** between MIDI commands give SoundSwitch hardware time to settle.
- **Os2lSender** runs in a separate thread; the audio/DSP path is async on the main thread — mixing threading models requires care when touching shared state.
- **BPM-only intent classification is coarse** — BPM alone won't distWe ninguish a calm 130 BPM track from an energetic one. Next improvement: weight by aubio onset density or YAMNet embeddings directly.
- **Weak YAMNet changes are now always accepted** (previously gated on Spotify section proximity). May cause more false-positives in stable sections. The 10 s cooldown is the main guard.
