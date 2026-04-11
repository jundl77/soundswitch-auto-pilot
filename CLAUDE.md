# SoundSwitch Auto Pilot

Intelligent DJ lighting automation system that synchronizes stage effects to live music in real time. Analyzes audio → detects beats/sections/intensity → controls SoundSwitch lighting via MIDI and OS2L protocol. Optionally enriched with Spotify track metadata.

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
4. Enriches with Spotify metadata (genre, energy, BPM, section timestamps, beat strength) every 20 s
5. Classifies track intensity as LOW / MEDIUM / HIGH / HIP_HOP using Spotify features + genre heuristics
6. Selects and sends MIDI lighting effects to SoundSwitch; also sends OS2L beat events to VirtualDJ and DMX overlays via UDP

---

## Architecture

```
PyAudio → MusicAnalyser (Aubio DSP) → LightEngine (IMusicAnalyserHandler)
                   ↓                          ↓               ↓
        YamnetChangeDetector          EffectController    MIDI / OS2L / Overlay
                   ↓
         SpotifyClient (20 s poll) → LightShowClassifier
```

### Key Files

| Path | Role |
|---|---|
| `auto_pilot` | CLI entry point (`run MIDI_PORT` or `list`) |
| `lib/main.py` | `SoundSwitchAutoPilot` — async event loop, 100 ms / 1 s / 10 s callbacks |
| `lib/analyser/music_analyser.py` | `MusicAnalyser` — per-buffer DSP, beat/onset/note events, YAMNet trigger |
| `lib/analyser/change_detector.py` | `YamnetChangeDetector` — TF Hub YAMNet embeddings, MAD outlier detection, 10 s cooldown |
| `lib/analyser/light_show_classifier.py` | `LightShowClassifier` — maps Spotify audio features → intensity class |
| `lib/engine/light_engine.py` | `LightEngine` — routes DSP events → MIDI / OS2L / overlay commands |
| `lib/engine/effect_controller.py` | `EffectController` — non-repetitive random effect selection per section/intensity |
| `lib/clients/midi_client.py` | MIDI note-on/off to SoundSwitch; 90+ channels, delayed deactivation |
| `lib/clients/os2l_client.py` | zeroconf discovery of VirtualDJ; bidirectional OS2L JSON; 25 ms beat position updates |
| `lib/clients/spotify_client.py` | OAuth2 Spotipy; extracts sections, beats, audio features, genre |
| `lib/clients/pyaudio_client.py` | Mono 44.1 kHz audio input (and optional debug output passthrough) |
| `lib/clients/overlay_client.py` | UDP binary DMX overlay to 192.168.178.245:19001 (hardcoded — must match venue) |
| `lib/visualizer/` | Optional matplotlib spectrogram UI via multiprocessing + TCP |

---

## Data Flow

```
Audio buffer (256 samples)
  → Aubio: pitch, BPM, onset confidence, MFCCs, mel energies
  → beat detected? → LightEngine.on_beat() → MIDI velocity commands
  → onset detected? → LightEngine.on_onset()
  → note detected? → LightEngine.on_note()
  → YAMNet buffer full (4096 samples)?
      → embed → cosine similarity → outlier? → section change
      → EffectController.change_effect()
          → find current Spotify section (with -1 s lookahead offset)
          → select from LOW/MEDIUM/HIGH/HIP_HOP effect bank (avoids repeats)
          → send MIDI note-on/off to SoundSwitch
```

Spotify polls every 20 s: injects beat count / timing offsets back into `MusicAnalyser` for sync, and updates section timeline used by `EffectController`.

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

# Full options
python auto_pilot run 0 -i INPUT_DEVICE_IDX -o OUTPUT_DEVICE_IDX -v --no-os2l

# Simulation (no hardware required)
python simulate_auto_pilot --mode beep --bpm 120 --delay 0.3 --duration 15

# Tests
pytest -m "not integration"   # fast unit tests only
pytest                        # unit + integration (~30s)
```

**Flags:**
- `-i / -o` — audio device indices from `list`
- `-d` — debug: plays audio back with click on beats/notes
- `-v` — show matplotlib visualizer
- `--no-os2l` — disable VirtualDJ connection

**Spotify** (optional): create `spotify_details.json` with `spotify_client_id` and `spotify_client_secret`. First run opens browser for OAuth2 consent. Without this file, Spotify features are silently skipped.

---

## Key Constants & Tuning Knobs

| Constant | Location | Value | Meaning |
|---|---|---|---|
| `BUFFER_SIZE` | `pyaudio_client.py` | 256 | Audio frames per callback |
| `SAMPLE_RATE` | `music_analyser.py` | 44100 | Hz |
| `SPOTIFY_POLL_INTERVAL` | `spotify_client.py` | 20 s | How often Spotify is queried |
| `SECTION_CHANGE_COOLDOWN` | `change_detector.py` | 10 s | Min gap between YAMNet-triggered changes |
| `APPLY_COLOR_OVERRIDE_INTERVAL_SEC` | `effect_controller.py` | 300 | Color override rotation every 5 min |
| `SECTION_OFFSET_SEC` | `effect_controller.py` | -1.0 | Trigger effect 1 s before section starts |
| OS2L beat update | `os2l_client.py` | 25 ms | Beat position broadcast interval |

---

## ML / DSP Components

- **Aubio** — real-time pitch, BPM, onset, note, MFCC. All tuned for 44.1 kHz / 256-sample windows.
- **YAMNet (TensorFlow Hub)** — Google's pre-trained audio classifier; used here for 1024-dim embeddings only (not tags). Cosine similarity + MAD-based outlier detection across a 2 s lookback finds section transitions.
- **Spotify Audio Analysis API** — pre-computed beats, sections, and audio features (energy, danceability, etc.) used for sync and intensity classification. Beat strengths computed by averaging loudness-max within each second.

---

## Known Issues / Gotchas

- **Hardcoded overlay IP** (`192.168.178.245:19001`) — must change per venue in `overlay_client.py`.
- **YAMNet divide-by-zero**: safe (returns empty list) when MAD == 0, but worth noting.
- **MusicAnalyser full reset** every 15 min prevents rolling-window memory growth.
- **10 ms delays** between MIDI commands give SoundSwitch hardware time to settle.
- **SpotifyClient and Os2lSender** run in separate threads; the audio/DSP path is async on the main thread — mixing threading models requires care when touching shared state.
- **RgbVisualizer** exists but is unused (code commented out in main loop).
