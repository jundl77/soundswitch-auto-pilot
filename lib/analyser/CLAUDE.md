# Analyser: Audio Feature Extraction and Classification

Deep-dive on the analysis pipeline. The main `CLAUDE.md` covers what each intent means and how to run the system; this document covers *why* the features and classifiers are designed the way they are.

See `music_analyser.py` for all implementation details and `lib/engine/light_engine.py` for all tuning constants.

---

## Features and Why They Were Chosen

**BPM** — the primary tempo discriminator. High BPM with moderate density → PEAK. Low onset activity at any BPM → BREAKDOWN or ATMOSPHERIC.

**Onset density** (onsets/sec, rolling window) — measures rhythmic busyness. A sparse arrangement has few onsets per second; a full drop with kick, bass, hi-hat, and percussion fires many per second. This is the primary BREAKDOWN/DROP discriminator.

**Onset density trend** (ratio of recent vs. past beats) — detects whether energy is rising. BUILDUP requires both sufficient density *and* a rising trend. A steady groove at the same density level stays in GROOVE even if the density is high. Needs a few beats to warm up before it carries signal.

**Sub-bass ratio** (mel filterbank bands 0–4 / total energy) — normalised fraction of energy in the bass register. Stored per-beat. A secondary signal; `kick_strength` (below) is more discriminating for DROP detection.

**Kick strength** — ratio of raw sub-bass energy *at beat timestamps* vs. the rolling mean sub-bass energy over all frames. A kick drum creates a concentrated sub-bass spike exactly on beat positions, pushing this ratio well above 1.0. A hi-hat-dominated pattern with no kick keeps the ratio near 1.0 because sub-bass is spread evenly. This is the primary DROP gate and the primary BREAKDOWN signal when density is moderate (stripped arrangement with no kick). See `music_analyser.py` for the implementation.

**Spectral centroid** (mel-band index units, 0–39) — centre of mass of the frequency spectrum. Low = bass-heavy; high = treble-heavy. Tracked per-buffer and at beat timestamps. The *trend* of the centroid across recent beats is the key feature: a rising centroid (energy moving toward higher frequencies) is the defining signature of a BUILDUP riser or sweep filter. A falling centroid (energy concentrating downward) signals a DROP approach. The trend is computed the same way as onset density trend: recent beats vs. past beats. See `music_analyser.py` for details.

**RMS energy** — mean amplitude over a short rolling window. Stored in the beat record. Not yet used in classification directly, but available as a loudness proxy. Future use: PEAK confirmation (loud + high BPM).

**YAMNet embeddings** — 1024-dimensional audio embeddings (not tag predictions). Used to detect structural section changes via cosine similarity outlier detection across a rolling lookback. The cooldown constant in `yamnet_change_detector.py` controls how often section changes can fire.

---

## Classifier Design Decisions

### Why kick strength as a feature rather than sub-bass ratio alone?

Sub-bass ratio (bands 0–4 / total energy) normalises by total energy, which means when a kick fires and total energy spikes, the ratio may not rise as dramatically as the raw energy does. More importantly, the ratio says nothing about *when* in the beat the sub-bass appears. Kick strength explicitly compares sub-bass *at beat timestamps* to the off-beat average, which directly tests whether the bass is rhythmically locked to the beat pattern — the defining feature of a kick drum. Hi-hat patterns have high onset density but their sub-bass is flat across the beat cycle; the ratio stays near 1.0.

### Why spectral centroid trend rather than just centroid value?

An absolute centroid value depends on the track and mix — a bass-heavy track has a low centroid throughout, and a bright track has a high centroid throughout. The *trend* is mix-invariant: it asks whether the centroid is rising or falling relative to its own recent history. A riser in any track will push the centroid upward regardless of where it starts. This makes the trend a reliable BUILDUP signal without requiring per-track calibration.

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

Now automated. Run:

```bash
python auto_pilot simulate file samples/song.mp3 --no-ui --report report.json --sweep
```

The `--sweep` flag runs a 10k-sample Latin-hypercube threshold sweep via fast feature-log replay, writes top-50 configs to `report.json` under `sweep_results`, and prints a summary. Claude then reads the report, applies the best config to `lib/engine/light_engine.py`, re-runs the full simulation to verify, and iterates. See `simulate/evaluator.py::sweep_thresholds` and `specs/2026-04-13-ground-truth-evaluation-and-threshold-optimization.md` for the full architecture.

### Threshold calibration status (as of 2026-04-13, Eric Prydz "Generate" 128 BPM)

Sweep-optimized against `samples/generate_eric_prydz_192k.csv` (8 GT boundaries).

**Current accuracy: 3/8 boundaries detected (offsets 440–600 ms), 5 missed, 1 false.**

| Boundary | Offset | Notes |
|---|---|---|
| atmospheric → breakdown (9.04s) | MISSED | Spurious initial PEAK from double-BPM aubio artifact delays BREAKDOWN entry |
| breakdown → buildup (39.12s) | +603ms | Detected; offset from 4-vote buffer + window alignment |
| buildup → drop (45.77s) | MISSED | DROP undetectable — see below |
| drop → groove (60.92s) | MISSED | Cannot exit DROP if never entered |
| groove → breakdown (76.06s) | +504ms | Detected |
| breakdown → buildup (106.48s) | MISSED | Insufficient density/trend signal |
| buildup → peak (128.90s) | MISSED | PEAK undetectable at 128 BPM — see below |
| peak → atmospheric (159.28s) | +439ms | Detected via beat-absence transition |

**Known hard limits for this track:**

- **DROP undetectable**: Max onset density in the actual drop section is 4.67 — indistinguishable from the breakdown density (~3.77). Eric Prydz's heavy sidechain compression suppresses onset spikes during the drop. The density feature cannot separate this section from breakdown without additional signal (RMS, spectral flux, or sub-bass power).
- **PEAK undetectable**: `_PEAK_MIN_BPM_ENTER=140` cannot be satisfied at 128 BPM. The PEAK classification needs a different gate (RMS energy sustained high, or density + kick sustained).
- **Transition lag ~500ms**: with `_VOTE_BUFFER_SIZE=4` at 128 BPM (0.47s/beat), 4 votes = ~1.9s of buffer lag. The look-ahead window (2.5s) partially compensates, but window alignment effects leave ~500ms residual lag for clean section boundaries.

---

## Future Work

- **RMS energy in classification**: critical for PEAK detection in tracks where BPM stays constant. "Loud + sustained kick" is a better PEAK gate than BPM alone.
- **Spectral flux**: rate of change of the mel spectrum captures timbral shifts that onset density misses — useful for sidechain-compressed drops where onset density is suppressed.
- **DROP sub-bass gate**: `_DROP_MIN_SUB_BASS_RATIO` is disabled (0.0). Calibrate against real kick+bass vs. hi-hat-only passages once per-beat sub-bass data is validated.
- **Kick strength calibration**: the Eric Prydz track has kick_mean=0.59–1.06 throughout — consistently below the old 1.3 threshold. With `_KICK_PRESENCE_THRESHOLD=1.173`, kick is detected only in a few sections. Measure on more tracks to find a robust cross-track value.
- **Multi-track sweep**: current sweep is calibrated against one track. Add more annotated tracks and sweep against all simultaneously to avoid overfitting to a single track's density profile.
- **Vote buffer lag**: with `_VOTE_BUFFER_SIZE=4`, the minimum structural transition lag is ~4 beats (~1.9s at 128 BPM). Reducing to 2–3 with a sharper dwell check could lower lag without introducing flickering; needs multi-track validation.