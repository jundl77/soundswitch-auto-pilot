# Analyser: Audio Feature Extraction and Intent Classification

This document summarises the analysis pipeline, the features computed, the design decisions made, and the evaluation strategy for intent classification.

---

## Pipeline Overview

```
PyAudio buffer (256 samples, 5.8 ms)
    │
    ├── Aubio pitch        (yin, win=2048)
    ├── Aubio BPM/tempo    (win=512)
    ├── Aubio onset        (win=512) ──► onset_times deque
    ├── Aubio notes        (win=512)
    ├── Aubio PVOC + mel filterbank (win=1024, 40 bands)
    │       └── mel energies ──► _mel_energies_window deque (last ~150 ms)
    │       └── RMS ──────────► _rms_window deque (last ~150 ms)
    └── YAMNet (4096-sample buffer, async) ──► section-change detection
```

Every `analyse()` call computes all features synchronously on the audio thread. Callbacks (`on_beat`, `on_onset`, `on_section_change`) are fired from within this call.

---

## Features Computed

### BPM (`get_bpm()`)
- Source: `aubio.tempo` — phase vocoder energy envelope tracker.
- Updated every time a beat is detected.
- Used directly in the intent classifier.

### Onset Density (`get_onset_density()`)
- Onsets per second over the last 1.5 s (`_ONSET_DENSITY_WINDOW_SEC`).
- Measures *rhythmic busyness*: a single kick drum at 128 BPM ≈ 2 onsets/s; a full arrangement with hi-hats ≈ 6–12 onsets/s.
- The primary discriminator between BREAKDOWN (sparse) and DROP (dense).

### Onset Density Trend (`get_onset_density_trend()`)
- Ratio: `mean(recent half) / mean(past half)` of the per-beat density deque.
- > 1.0 → energy rising → BUILDUP signal.
- Returns 1.0 (neutral) until 4 samples are collected (≈ 2 s at 120 BPM).

### Sub-Bass Ratio (`get_sub_bass_ratio()`)
- Mel filterbank bands 0–4 cover approximately 60–250 Hz (sub-bass and bass).
- Ratio: `sum(bands 0–4) / sum(all 40 bands)` averaged over the last ~150 ms of frames.
- High values (≥ 0.25) indicate kick drum or heavy bass — strong DROP discriminator.
- Currently used as a gate in `_classify_intent` with threshold `_DROP_MIN_SUB_BASS_RATIO = 0.0` (gate disabled). Set to ~0.15–0.25 after calibrating against real music data.

### RMS Energy (`get_rms_energy()`)
- Mean RMS amplitude over the last ~150 ms of buffers.
- Available in the beat history (5-tuple) for future use. Not yet used directly in classification, but available as a loudness proxy for PEAK vs GROOVE disambiguation.

### YAMNet Embeddings (Section Detection)
- Google's audio classifier (TF Hub), used here only for 1024-dimensional embeddings — not tag predictions.
- Buffer accumulates 4096 samples (~93 ms), then computes a new embedding.
- **Outlier detection**: cosine similarity between the new embedding and the rolling mean of recent embeddings. If the similarity is a MAD outlier (> 3 MADs from median), a section change is fired.
- 10 s cooldown (`SECTION_CHANGE_COOLDOWN`) prevents bursts of changes.
- Degrades gracefully if TF Hub is unavailable (logs a warning, section detection disabled).

---

## Intent Classification

### `_classify_intent(bpm, onset_density, density_trend, current_intent, sub_bass_ratio)`

Pure function in `light_engine.py`. Maps the feature tuple → `LightIntent`.

**Hysteresis (Schmitt trigger)**: when `current_intent` is provided, each intent's *exit* threshold applies instead of the *entry* threshold. This prevents threshold-boundary oscillation ("flickering").

| Intent | Entry condition | Exit condition |
|---|---|---|
| ATMOSPHERIC | beat absence > 2.5 s (not in this function) | first beat detected |
| BREAKDOWN | density < 3.0 /s | density > 3.5 /s |
| GROOVE | density ≥ 3.0, trend < 1.3 | (falls through from other intents) |
| BUILDUP | density ≥ 3.0, trend ≥ 1.3 | trend < 1.3 |
| DROP | density ≥ 8.5, BPM ≥ 100, sub_bass ≥ threshold | density < 7.0 |
| PEAK | BPM ≥ 140 | BPM < 135 |

Priority order (DROP first, ATMOSPHERIC never from this function):
1. DROP (density spike wins over everything)
2. PEAK (high sustained BPM)
3. BREAKDOWN (sparse density)
4. BUILDUP (rising trend)
5. GROOVE (default)

### `_classify_windowed(window, bpm, current_intent)`

Classifies using a symmetric window of beats [T − look_ahead_sec, T + look_ahead_sec]:

- **Median density** (not mean) — robust to single-beat transient spikes. A genuine DROP must have sustained high density across all window beats.
- **Forward trend** = `mean(future half) / mean(past half)` — uses future beats to confirm whether energy is genuinely rising at beat T.
- **Mean sub-bass ratio** — averaged over all window beats; passed to `_classify_intent` for the DROP gate.
- `current_intent` forwarded to `_classify_intent` for hysteresis-aware thresholds.

---

## Stability Pipeline (in `_commit_intent`)

Applied on every delayed commit (once per beat, after `look_ahead_sec`):

```
classify_windowed(window) → candidate intent
    │
    ├─ vote buffer (rolling deque, size=3)
    │   └─ require full + unanimous buffer before proceeding
    │
    ├─ minimum dwell check
    │   └─ require ≥ 4 beats in current intent before switching
    │
    ├─ invalid-transition guard
    │   └─ block musically impossible jumps:
    │       ATMOSPHERIC → DROP / BUILDUP / PEAK
    │       PEAK → BUILDUP
    │
    └─ commit: change_effect(new_intent) + reset vote buffer + reset dwell counter
```

**Why three layers?**
- **Voting** kills single-beat noise (one anomalous beat can't flip the intent).
- **Hysteresis** kills threshold-boundary jitter (prevents rapid in/out oscillation near a boundary).
- **Dwell** kills rapid cycling (can't switch intent after just 1–2 beats in a new state).
- **Transition guard** kills genre-incoherent jumps (ATMOSPHERIC → DROP is physically impossible without a BREAKDOWN or GROOVE in between).

---

## ATMOSPHERIC Detection

ATMOSPHERIC is the only intent set outside `_classify_intent`. It is fired from `on_100ms_callback` when no beat has been detected for `_BEAT_ABSENCE_SEC = 2.5 s` (≈ 5 missed beats at 128 BPM).

When ATMOSPHERIC fires:
- Vote buffer is cleared (stale votes irrelevant).
- Dwell counter is reset to 0.
- `_atmospheric_sent` flag is set to prevent repeated firing.

When the next beat arrives after ATMOSPHERIC:
- `_classify_intent` is called immediately (no delay, no windowed classification) to rapidly re-engage a musical intent.
- Vote buffer and dwell counter are reset fresh.

---

## Evaluation Strategy

### Simulation mode
Run `python auto_pilot simulate file samples/song.mp3 --report report.json`. The JSON report contains:
- Full beat list with timestamps, BPM, onset density
- Full intent timeline with durations
- Timing log for command queue accuracy

### Metrics to watch
- `intent_distribution_sec`: time spent in each intent. Compare against expected musical structure.
- `intent_changes_count`: should be ~5–20 for a 3-min track. Much higher → flickering. Much lower → stuck.
- `dominant_intent`: should match the track's character (e.g. "drop" for a hard techno track).
- `timing_error_max_ms`: should be < 50 ms. Higher → command queue is lagging.

### Known limitations
- Onset density is averaged over 1.5 s, not per-beat. Sudden changes take 1.5 s to propagate.
- Density trend needs 4 beats to warm up; BUILDUP cannot be detected in the first 2–3 s.
- Sub-bass gate (`_DROP_MIN_SUB_BASS_RATIO`) is currently disabled (threshold = 0.0). Must be calibrated against real DROP vs hi-hat-only passages before enabling.
- YAMNet section changes fire independently of the beat-based classifier; they bypass the vote buffer and dwell check for section changes (by design — sections are infrequent and deliberate).

### Tuning workflow
1. Run `simulate file` on tracks with known structure (e.g. a track where the drop starts at T=90s).
2. Inspect the intent timeline in the JSON report: does the DROP intent align with the actual drop?
3. Adjust thresholds in `light_engine.py` constants and re-run.
4. Once drops are consistent, enable and tune `_DROP_MIN_SUB_BASS_RATIO` against hi-hat-only high-density passages.

---

## Future Work

- **Sub-bass gate calibration**: collect real data to find a ratio threshold that separates kick+bass from hi-hat-only patterns.
- **RMS energy**: currently stored in BeatRecord but not used in classification. Could distinguish PEAK (loud, sustained) from GROOVE (moderate volume).
- **Spectral flux**: rate of change of the mel spectrum — captures sudden timbral changes that onset density misses.
- **MFCC-based clustering**: offline, label clusters per intent; use nearest-centroid at runtime (interpretable, no training data needed beyond labelling).
