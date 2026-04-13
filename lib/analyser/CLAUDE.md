# Analyser: Audio Feature Extraction and Classification

Deep-dive on the analysis pipeline. The main `CLAUDE.md` covers what each intent means and how to run the system; this document covers *why* the features and classifiers are designed the way they are.

See `music_analyser.py` for all implementation details and `lib/engine/light_engine.py` for all tuning constants.

---

## Features and Why They Were Chosen

**BPM** — the primary tempo discriminator. High BPM with moderate density → PEAK. Low onset activity at any BPM → BREAKDOWN or ATMOSPHERIC.

**Onset density** (onsets/sec, rolling window) — measures rhythmic busyness. A sparse arrangement has few onsets per second; a full drop with kick, bass, hi-hat, and percussion fires many per second. This is the primary BREAKDOWN/DROP discriminator.

**Onset density trend** (ratio of recent vs. past beats) — detects whether energy is rising. BUILDUP requires both sufficient density *and* a rising trend. A steady groove at the same density level stays in GROOVE even if the density is high. Needs a few beats to warm up before it carries signal.

**Sub-bass ratio** (mel filterbank bands 0–4 / total energy) — discriminates kick-and-bass patterns from hi-hat-heavy patterns. Both can produce high onset density; only one has heavy sub-bass. Currently used as a gate on the DROP classifier but disabled (threshold = 0) until calibrated against real tracks. The feature is computed and stored per-beat for future use. See `music_analyser.py` for the filterbank setup.

**RMS energy** — mean amplitude over a short rolling window. Stored in the beat record alongside the other features. Not yet used in classification, but available as a loudness proxy (PEAK is loud and sustained; a quiet passage at high BPM is something else). Future use: PEAK vs. GROOVE disambiguation.

**YAMNet embeddings** — 1024-dimensional audio embeddings (not tag predictions). Used to detect structural section changes via cosine similarity outlier detection across a rolling lookback. The cooldown constant in `yamnet_change_detector.py` controls how often section changes can fire.

---

## Classifier Design Decisions

### Why hysteresis (Schmitt trigger)?

A single threshold causes rapid back-and-forth switching when a signal hovers near the boundary. Separate entry/exit thresholds create a "dead zone" where the current intent is held until the signal clearly crosses to the other side. This is the same principle as a thermostat — you don't want the heating to toggle every second because the temperature is bouncing around the setpoint.

All entry/exit thresholds live in `lib/engine/light_engine.py`.

### Why a vote buffer?

Even with windowed median density, a single anomalous beat window can temporarily shift the classification. The vote buffer requires several consecutive identical classifications before committing — a single outlier window gets overruled by the surrounding ones. This is the intent-level equivalent of debouncing a button.

### Why a minimum dwell check?

Prevents the classifier from entering a new intent and immediately switching away from it before the window has settled. Without dwell, the system could enter DROP, detect a slightly lower density on the next beat, and snap back to GROOVE — too fast for any light effect to be meaningful.

### Why invalid-transition blocking?

Some intent transitions are musically impossible. You cannot go from dead silence (ATMOSPHERIC) directly to a full DROP — there must be some beats in between. The transition guard encodes this domain knowledge as a hard rule. The blocked transitions and the valid graph are defined in `lib/engine/light_engine.py`.

### Why symmetric windowed classification?

The engine operates ahead of what the audience hears (the look-ahead delay). By the time beat T is heard, the engine has already seen the beats that *follow* T. Using both past and future beats around T for classification gives:

- **Spike rejection**: a single high-density beat surrounded by normal beats stays at the median density of the window — no false DROP.
- **Earlier BUILDUP detection**: future beats confirm that energy really is rising at T, not just spiking once.

The look-ahead window half-width (`LOOK_AHEAD_SEC`) must match `playback_delay_seconds` in dmx-enttec-node. It is defined in `lib/main.py` and `simulate/runner.py`.

### Why is ATMOSPHERIC not in the classifier?

ATMOSPHERIC is detected by beat *absence*, not by any feature value. No density reading, BPM, or trend is meaningful when there are no beats. The 100 ms callback monitors elapsed time since the last beat and fires ATMOSPHERIC once the silence threshold is crossed. Everything else is purely beat-driven.

---

## Evaluation Strategy

### Running a simulation

```bash
python auto_pilot simulate file samples/song.mp3 --report report.json
```

The JSON report contains the full beat list, intent timeline, and timing log. Inspect:

- **`intent_distribution_sec`** — time spent in each intent. Does it match the track structure?
- **`intent_changes_count`** — should be in the tens for a 3-minute track. Much higher means flickering; much lower means stuck.
- **`dominant_intent`** — should reflect the character of the track.
- **`timing_error_max_ms`** — command queue accuracy. Should be well under 50 ms.

### Tuning workflow

1. Run simulation on a track with a known structure (e.g. the drop starts at T=90 s).
2. Inspect the intent timeline: does the DROP intent start and end where the drop does?
3. Adjust thresholds in `lib/engine/light_engine.py` and re-run.
4. Once the basic structure is reliable, enable and tune the sub-bass gate against hi-hat-only vs. kick+bass passages.

---

## Future Work

- **Sub-bass gate calibration**: find the ratio threshold that reliably separates kick+bass from hi-hat-only patterns on real tracks.
- **RMS energy in classification**: use as a PEAK confirmation signal (loud + high BPM = PEAK; quiet + high BPM = probably just tempo, not energy).
- **Spectral flux**: rate of change of the mel spectrum captures timbral shifts that onset density misses — useful for detecting timbral drops (e.g. a low-pass filter sweep releasing into the drop).
- **Labelled data**: once enough real-track simulations exist, label the intent timeline manually and use it to validate or calibrate thresholds systematically rather than by ear.
