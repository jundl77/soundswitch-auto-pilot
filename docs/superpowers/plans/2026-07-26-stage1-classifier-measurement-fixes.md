# Stage 1: Classifier Measurement Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un-invert the intent classifier by fixing five empirically proven measurement bugs, so that high-energy track sections classify as DROP/GROOVE and low-energy sections as BREAKDOWN/ATMOSPHERIC — the opposite of today's behavior.

**Architecture:** All fixes are localized: feature measurement in `lib/analyser/music_analyser.py`, classification thresholds/branching in `lib/engine/light_engine.py`, and richer per-beat records flowing through `lib/engine/event_buffer.py` into the simulation report (which becomes the future training table). No new components, no pipeline changes. The deterministic fast sim (`auto_pilot simulate file`) is the measurement instrument for every fix.

**Tech Stack:** Python 3.12, aubio, numpy, pytest (async via existing conftest), uv.

## Global Constraints

- Branch: `stage1_classifier_measurement_fixes` (already created off master). Never commit to `master`.
- Every task ends with `uv run pytest -m "not integration"` green; the final task runs the full `uv run pytest`.
- Determinism is inviolate: re-running `uv run python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3` twice must print the same sha256. Report *content* will change with each fix (expected — the checksum value changes, but stays stable across reruns).
- CLAUDE.md policy: intent/architecture only — no threshold values or signatures duplicated into CLAUDE.md files.
- Commit after every task (small commits). Do NOT push until the final task.

## Empirical ground truth (sample track `samples/generate_eric_prydz_192k.mp3`, ~165 s)

Reference structure (librosa offline analysis, energy percentiles in parens):

| Song time | Section | Energy |
|---|---|---|
| 0–22.9 s | intro | pct 25 |
| 22.9–45.5 s | builds | pct 41–44 |
| 45.5–76.5 s | main groove | pct 80–81 |
| 76.5–90.8 s | breakdown | pct 7–13 |
| 90.8–128.4 s | rebuild | pct 53–64 |
| 128.4–158.8 s | final drop | pct 82 |
| 158.8 s+ | outro fade | pct 1–3 |

Measured engine-side baselines (instrumented run, pre-fix):
- onset_density: median 4.00, p90 4.67, max 5.33 (vs `_DROP_MIN_DENSITY_ENTER = 8.5` → DROP unreachable).
- kick_strength: median 0.84, ≥1.3 on only 7.8% of beats (vs offline ground-truth beat/background sub-bass ratio **2.59** in main groove, **2.82** in final drop, **1.04–1.11** in intro/breakdown). The snapshot misses the transient.
- kick_strength explodes to 139–1125 during the outro fade (denominator → 0).
- BPM: median 127.99 (excellent), except 6 warmup beats at 1.6–2.7 s reading **257.8** → false PEAK 0–4.2 s.
- Result: 93.7% of the track labeled BREAKDOWN; DROP/BUILDUP/ATMOSPHERIC never fired.

**Note on look-ahead offset when reading reports:** intent blocks are stamped at *audience* time — one look-ahead delay (2.5 s) after the beats that caused them. When comparing an intent timeline against the song-time table above, expect that constant shift.

---

### Task 1: BPM octave folding

**Files:**
- Modify: `lib/analyser/music_analyser.py` (method `get_bpm`, ~line 91)
- Test: `tests/test_music_analyser.py`

**Interfaces:**
- Produces: `MusicAnalyser._fold_bpm(bpm: float) -> float` (staticmethod) — folds any positive BPM into `[85.0, 170.0)` by octave halving/doubling; returns `0.0` for input `<= 0`. `get_bpm()` now returns folded values. All downstream consumers (`LightEngine.on_beat`, beat records, reports) automatically receive folded BPM.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_music_analyser.py`:

```python
class TestFoldBpm:
    def test_double_tempo_folds_down(self):
        # aubio warmup double-tempo lock: 257.8 must fold to 128.9
        assert MusicAnalyser._fold_bpm(257.8) == pytest.approx(128.9)

    def test_half_tempo_folds_up(self):
        assert MusicAnalyser._fold_bpm(64.0) == pytest.approx(128.0)

    def test_in_range_untouched(self):
        assert MusicAnalyser._fold_bpm(128.0) == pytest.approx(128.0)

    def test_boundary_170_folds_to_85(self):
        assert MusicAnalyser._fold_bpm(170.0) == pytest.approx(85.0)

    def test_zero_and_negative_return_zero(self):
        assert MusicAnalyser._fold_bpm(0.0) == 0.0
        assert MusicAnalyser._fold_bpm(-10.0) == 0.0
```

(Match the existing import style in that file; if it has no classes, write them as flat functions instead — follow the file's convention.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_music_analyser.py -v -k fold`
Expected: FAIL with `AttributeError: ... has no attribute '_fold_bpm'`

- [ ] **Step 3: Implement**

In `lib/analyser/music_analyser.py`, replace `get_bpm` and add `_fold_bpm`:

```python
    def get_bpm(self) -> float:
        if self.is_playing:
            return self._fold_bpm(self.tempo_o.get_bpm())
        else:
            return 0

    @staticmethod
    def _fold_bpm(bpm: float) -> float:
        """Fold BPM into [85, 170) by octave halving/doubling.

        aubio locks onto double/half tempo during warmup and on ambiguous
        material (observed: 257.8 BPM for the first ~6 beats of every track).
        EDM lives in one tempo octave; folding removes the ambiguity without
        touching the beat phase.
        """
        if bpm <= 0:
            return 0.0
        while bpm >= 170.0:
            bpm /= 2.0
        while bpm < 85.0:
            bpm *= 2.0
        return bpm
```

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest -m "not integration"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/analyser/music_analyser.py tests/test_music_analyser.py
git commit -m "fold BPM into [85,170) — kills aubio warmup double-tempo octave errors"
```

---

### Task 2: Kick strength — capture the transient, gate the silence

**Files:**
- Modify: `lib/analyser/music_analyser.py` (`_track_beat` ~line 221, `get_kick_strength` ~line 148)
- Test: `tests/test_music_analyser.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `get_kick_strength()` unchanged signature, corrected semantics: beat samples are now the **max** raw sub-bass over the last `_KICK_CAPTURE_BUFFERS` buffers at beat time (captures the transient instead of one trailing 5.8 ms buffer); the returned ratio is capped at 10.0 and gated to return 1.0 (unknown) when overall RMS is near-silent.

**Why:** the current code appends `self._all_sub_bass_samples[-1]` — a single 256-sample buffer snapshot that misses the kick transient (measured 0.84 vs true 2.6–2.8). And during fades the denominator collapses, producing ratios of 139–1125 that fired GROOVE on a silent outro.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_music_analyser.py` (adapt constructor usage to how existing tests in the file build a `MusicAnalyser`):

```python
class TestKickStrength:
    def _analyser(self):
        # Build the same way existing tests in this file do (stub handler).
        ...

    def test_beat_sample_captures_transient_peak(self):
        a = self._analyser()
        # Simulate 12 buffers of quiet sub-bass with one kick spike 5 buffers ago
        for v in [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 8.0, 1.0, 1.0, 1.0, 1.0]:
            a._all_sub_bass_samples.append(v)
        a._rms_window.append(0.2)  # clearly audible
        a._capture_beat_sub_bass()
        # The captured value must be the transient peak (8.0), not the last buffer (1.0)
        assert a._beat_sub_bass_samples[-1] == pytest.approx(8.0)

    def test_ratio_capped(self):
        a = self._analyser()
        a._rms_window.append(0.2)
        a._all_sub_bass_samples.extend([0.001] * 50)
        a._beat_sub_bass_samples.extend([5.0] * 5)
        assert a.get_kick_strength() == pytest.approx(10.0)

    def test_near_silence_returns_unknown(self):
        a = self._analyser()
        a._rms_window.append(0.001)  # -60 dBFS: fade-out / silence
        a._all_sub_bass_samples.extend([1e-6] * 50)
        a._beat_sub_bass_samples.extend([1e-3] * 5)
        assert a.get_kick_strength() == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_music_analyser.py -v -k kick`
Expected: FAIL (`_capture_beat_sub_bass` not defined; ratio uncapped)

- [ ] **Step 3: Implement**

In `music_analyser.py`, add a module-level constant near `_ONSET_DENSITY_WINDOW_SEC`:

```python
# Kick transient capture: max sub-bass over the last N buffers at beat time
# (~52 ms at 5.8 ms/buffer). aubio reports the beat a few buffers after the
# transient; a single trailing-buffer snapshot misses the kick spike entirely.
_KICK_CAPTURE_BUFFERS = 9
# Below this mean RMS the track is effectively silent (fade-out): sub-bass
# ratios become numerically meaningless, so kick presence reads as unknown.
_KICK_MIN_RMS = 0.005
```

Replace the beat-snapshot lines in `_track_beat`:

```python
            # Snapshot sub-bass and centroid at beat time for kick/centroid-trend features.
            self._capture_beat_sub_bass()
            if self._centroid_window:
                self._beat_centroid_samples.append(self._centroid_window[-1])
```

Add the method:

```python
    def _capture_beat_sub_bass(self) -> None:
        """Record the kick transient for this beat: the max raw sub-bass energy
        over the last _KICK_CAPTURE_BUFFERS buffers (not just the trailing one)."""
        if not self._all_sub_bass_samples:
            return
        recent = list(self._all_sub_bass_samples)[-_KICK_CAPTURE_BUFFERS:]
        self._beat_sub_bass_samples.append(max(recent))
```

Update `get_kick_strength` (keep the docstring, amend it for the cap/gate):

```python
        if not self._beat_sub_bass_samples or not self._all_sub_bass_samples:
            return 1.0
        if self.get_rms_energy() < _KICK_MIN_RMS:
            return 1.0
        beat_mean = sum(self._beat_sub_bass_samples) / len(self._beat_sub_bass_samples)
        all_mean = sum(self._all_sub_bass_samples) / len(self._all_sub_bass_samples)
        if all_mean < 1e-8:
            return 1.0
        return min(beat_mean / all_mean, 10.0)
```

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest -m "not integration"`
Expected: PASS

- [ ] **Step 5: Measure the fix on the sample track**

Run: `uv run python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3 --report report.json`
Then read `report.json` and eyeball `metrics` — this is a smoke check only; full verification comes in Task 6 once beat records carry kick_strength.

- [ ] **Step 6: Commit**

```bash
git add lib/analyser/music_analyser.py tests/test_music_analyser.py
git commit -m "kick_strength: capture transient peak over ~52ms window; cap ratio; silence gate"
```

---

### Task 3: Per-beat feature records in the report (the future training table)

**Files:**
- Modify: `lib/engine/event_buffer.py` (`add_beat`, line 59)
- Modify: `lib/engine/light_engine.py` (`on_beat`, the `add_beat` call ~line 262)
- Test: `tests/test_event_buffer.py`

**Interfaces:**
- Produces: `EventBuffer.add_beat(bpm: float, onset_density: float, change: bool, kick_strength: float = 1.0, centroid_trend: float = 1.0, sub_bass_ratio: float = 0.0, rms: float = 0.0) -> None`. Beat dicts in `snapshot()['beats']` and `to_report()['beats']` gain keys `kick_strength`, `centroid_trend`, `sub_bass_ratio`, `rms` (floats rounded to 4 decimals). Task 6's inspection script and all Stage-2 training work consume these keys.

**Why:** the report's `beats[]` is the training table for the upcoming Raveform work, and Task 6 needs these features to calibrate thresholds. Today only bpm/onset_density survive into the report.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_event_buffer.py` (match its existing construction style — it uses a `VirtualClock`):

```python
def test_add_beat_records_feature_columns():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    buf.add_beat(128.0, 4.2, False, kick_strength=2.61, centroid_trend=1.05,
                 sub_bass_ratio=0.31, rms=0.42)
    beat = buf.to_report()['beats'][0]
    assert beat['kick_strength'] == pytest.approx(2.61)
    assert beat['centroid_trend'] == pytest.approx(1.05)
    assert beat['sub_bass_ratio'] == pytest.approx(0.31)
    assert beat['rms'] == pytest.approx(0.42)


def test_add_beat_defaults_are_neutral():
    clock = VirtualClock()
    buf = EventBuffer(window_sec=float('inf'), clock=clock)
    buf.start()
    buf.add_beat(128.0, 4.2, False)
    beat = buf.to_report()['beats'][0]
    assert beat['kick_strength'] == 1.0
    assert beat['rms'] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_event_buffer.py -v -k feature`
Expected: FAIL with `TypeError: add_beat() got an unexpected keyword argument`

- [ ] **Step 3: Implement**

`event_buffer.py`:

```python
    def add_beat(self, bpm: float, onset_density: float, change: bool,
                 kick_strength: float = 1.0, centroid_trend: float = 1.0,
                 sub_bass_ratio: float = 0.0, rms: float = 0.0) -> None:
        with self._lock:
            self._beats.append({
                't': self._now(), 'bpm': bpm,
                'onset_density': onset_density,   # onsets/sec (aubio rolling window)
                'strength': min(1.0, onset_density / 10.0),  # 0–1 scaled for visualizer
                'change': change,
                # Full feature row — the sim report doubles as a training table.
                'kick_strength': round(kick_strength, 4),
                'centroid_trend': round(centroid_trend, 4),
                'sub_bass_ratio': round(sub_bass_ratio, 4),
                'rms': round(rms, 4),
            })
```

`light_engine.py` `on_beat` — replace the `add_beat` call:

```python
        if self.event_buffer:
            self.event_buffer.add_beat(bpm, onset_density, bpm_changed,
                                       kick_strength=kick_strength,
                                       centroid_trend=centroid_trend,
                                       sub_bass_ratio=sub_bass_ratio,
                                       rms=rms_energy)
```

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest -m "not integration"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/engine/event_buffer.py lib/engine/light_engine.py tests/test_event_buffer.py
git commit -m "beat records carry kick/centroid/sub-bass/rms — sim report becomes a feature table"
```

---

### Task 4: Classifier — reachable DROP, unshadowed BUILDUP, PEAK out of the pure classifier

**Files:**
- Modify: `lib/engine/light_engine.py` (constants block lines 35–51, `_classify_intent` lines 70–118)
- Test: `tests/test_classify_intent.py` (several existing tests must be updated — listed below)

**Interfaces:**
- Consumes: kick_strength semantics from Task 2 (true range: ~1.0–1.1 kick-absent, ~2.6–2.8 kick-present).
- Produces: `_classify_intent` same signature, but: (a) never returns `PEAK` (PEAK becomes an engine-level promotion, Task 5); (b) DROP thresholds are reachable; (c) BUILDUP is checked before the BREAKDOWN branches; (d) new constant `_BUILDUP_MIN_DENSITY`. Constants `_PEAK_MIN_BPM_ENTER`/`_PEAK_MIN_BPM_EXIT` are **deleted**. Task 5 depends on (a).

**Why (three proven defects):**
1. DROP unreachable: density maxes at 5.33 vs entry 8.5 — the detector saturates at ~2.5× dynamic range where thresholds assumed ~4×.
2. BUILDUP shadowed: the no-kick BREAKDOWN branch precedes it, and risers *strip the kick by design* — exactly when BUILDUP matters, it was unreachable.
3. PEAK-by-BPM is meaningless post-Task-1: folded BPM tops out below 170 and house sits at 122–130 forever; the only PEAK ever observed was the 257.8 warmup artifact. PEAK's real meaning ("sustained maximum energy after the drop") is temporal, so it moves to the engine (Task 5).

- [ ] **Step 1: Update the constants**

In `light_engine.py` replace the threshold block:

```python
_BREAKDOWN_MAX_DENSITY_ENTER = 3.0   # enter BREAKDOWN when density < this
_BREAKDOWN_MAX_DENSITY_EXIT  = 3.5   # exit BREAKDOWN when density exceeds this
_BUILDUP_MIN_TREND           = 1.3   # density trend ratio — rising ≥30% → BUILDUP
_BUILDUP_MIN_DENSITY         = 2.0   # BUILDUP needs some rhythmic floor (trend on 1→2 onsets is noise)
_DROP_MIN_DENSITY_ENTER      = 4.7   # enter DROP: sustained density ≥ this (p90 on ref track = 4.67)
_DROP_MIN_DENSITY_EXIT       = 4.2   # exit DROP when windowed median falls below this
_DROP_MIN_SUB_BASS_RATIO     = 0.0   # sub-bass gate for DROP (0.0 = disabled — kick_strength is the gate)
```

(`_PEAK_MIN_BPM_ENTER` / `_PEAK_MIN_BPM_EXIT` are deleted. `_KICK_PRESENCE_THRESHOLD` becomes `1.5` — midpoint-safe between measured 1.04–1.11 kick-absent and 2.59–2.82 kick-present. Keep `_BREAKDOWN_NO_KICK_MAX_DENSITY` and `_CENTROID_BUILDUP_TREND` as they are.)

These DROP values are *starting points*; Task 6 verifies them against measured per-section densities and adjusts if the final drop fails to classify DROP or the breakdown false-fires. Document any adjustment in the Task 6 commit message.

- [ ] **Step 2: Rewrite `_classify_intent`'s decision body**

New priority order: **DROP → BUILDUP → BREAKDOWN(sparse) → BREAKDOWN(no-kick) → GROOVE.**

```python
    currently_drop      = (current_intent == LightIntent.DROP)
    currently_breakdown = (current_intent == LightIntent.BREAKDOWN)

    drop_threshold      = _DROP_MIN_DENSITY_EXIT       if currently_drop      else _DROP_MIN_DENSITY_ENTER
    breakdown_threshold = _BREAKDOWN_MAX_DENSITY_EXIT  if currently_breakdown else _BREAKDOWN_MAX_DENSITY_ENTER

    kick_present = kick_strength >= _KICK_PRESENCE_THRESHOLD

    # DROP: sustained density + kick locked to beats + (optional) sub-bass gate
    if onset_density >= drop_threshold and bpm >= 100 and kick_present and sub_bass_ratio >= _DROP_MIN_SUB_BASS_RATIO:
        return LightIntent.DROP
    # BUILDUP: rising density trend OR rising spectral centroid (riser sweep).
    # Checked BEFORE the BREAKDOWN branches: risers strip the kick and thin the
    # arrangement right before the drop — the no-kick clamp must not swallow them.
    if onset_density >= _BUILDUP_MIN_DENSITY and (
            density_trend >= _BUILDUP_MIN_TREND or centroid_trend >= _CENTROID_BUILDUP_TREND):
        return LightIntent.BUILDUP
    # BREAKDOWN: very sparse density, or kick absent at moderate density
    if onset_density < breakdown_threshold:
        return LightIntent.BREAKDOWN
    if not kick_present and onset_density < _BREAKDOWN_NO_KICK_MAX_DENSITY:
        return LightIntent.BREAKDOWN
    return LightIntent.GROOVE
```

Update the function docstring: priority order, PEAK no longer produced here (engine promotes DROP→PEAK on dwell — see `LightEngine._commit_intent`), and the BUILDUP-before-BREAKDOWN rationale. Keep the signature including `bpm` (still gates DROP) and `kick_strength: float = 2.0` default.

- [ ] **Step 3: Update the existing tests that the new thresholds break**

In `tests/test_classify_intent.py`:
- Delete `test_peak_at_high_bpm_moderate_density`, `test_drop_beats_peak_at_high_bpm_high_density`, `test_peak_entry_threshold`, `test_peak_hysteresis_stays_in_peak_above_exit_threshold`, `test_peak_hysteresis_exits_below_exit_threshold`, and the `_PEAK_MIN_BPM_ENTER, _PEAK_MIN_BPM_EXIT` imports. Add one replacement:

```python
def test_peak_never_returned_by_pure_classifier():
    # PEAK is an engine-level promotion (sustained DROP), not a feature classification.
    for density in [0.5, 3.0, 4.5, 6.0, 9.0]:
        assert _classify_intent(160.0, density) != LightIntent.PEAK
```

- Every test that used density `5.0` as "moderate/GROOVE territory" now lands above the DROP entry (4.7). Change the density to `4.0` in: `test_buildup_on_rising_trend`, `test_no_buildup_without_rising_trend`, `test_buildup_trend_threshold_boundary`, `test_buildup_via_centroid_trend_without_density_trend`, `test_groove_when_centroid_trend_is_neutral`, `test_buildup_via_either_trend_signal`.
- `test_high_density_no_kick_above_breakdown_no_kick_max_stays_groove`: `_BREAKDOWN_NO_KICK_MAX_DENSITY + 0.5 = 6.5` is above DROP entry but kick is absent, so the existing assertion (`not in (DROP, BREAKDOWN)`) still holds — leave it.
- Windowed tests: `[4.0, 4.0, 9.5, 4.0, 4.0]` median 4.0 < 4.7 still not DROP — fine. `[9.0, 9.5, 10.0, 9.2, 8.8]` still DROP — fine. `test_windowed_stable_groove_not_classified_as_buildup` and `test_windowed_buildup_via_rising_centroid` use flat `4.5` — below 4.7 entry, fine.
- `test_windowed_buildup_detected_via_forward_context` uses `[3.0, 3.2, 5.0, 5.5, 6.0]`: median 5.0 ≥ 4.7 with default kick → now DROP. Change densities to `[2.5, 2.7, 3.8, 4.2, 4.4]` (median 3.8 < 4.7; forward trend (3.8+4.2+4.4)/3 ÷ (2.5+2.7)/2 = 4.13/2.6 = 1.59 ≥ 1.3 → BUILDUP).
- Add new-order regression tests:

```python
def test_buildup_wins_over_no_kick_breakdown():
    # Riser: kick stripped, moderate density, centroid rising → BUILDUP not BREAKDOWN
    no_kick = _KICK_PRESENCE_THRESHOLD - 0.1
    rising = _CENTROID_BUILDUP_TREND + 0.05
    assert _classify_intent(128.0, 4.0, kick_strength=no_kick,
                            centroid_trend=rising) == LightIntent.BUILDUP


def test_sparse_riser_below_density_floor_stays_breakdown():
    # Almost no onsets: trend is noise — BREAKDOWN even with rising centroid
    rising = _CENTROID_BUILDUP_TREND + 0.05
    assert _classify_intent(128.0, _BUILDUP_MIN_DENSITY - 0.5,
                            centroid_trend=rising) == LightIntent.BREAKDOWN
```

(import `_BUILDUP_MIN_DENSITY`.)

- [ ] **Step 4: Run the classifier tests, then the full unit suite**

Run: `uv run pytest tests/test_classify_intent.py -v` then `uv run pytest -m "not integration"`
Expected: PASS. Note: `tests/test_intent_stability.py` exercises the engine pipeline — if any of its scenarios relied on PEAK-by-BPM or density-5.0-as-GROOVE, adjust them with the same substitutions and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add lib/engine/light_engine.py tests/test_classify_intent.py tests/test_intent_stability.py
git commit -m "classifier: reachable DROP thresholds, BUILDUP before no-kick clamp, PEAK removed from pure classifier"
```

---

### Task 5: PEAK as engine-level promotion (sustained DROP)

**Files:**
- Modify: `lib/engine/light_engine.py` (`_commit_intent`, constants block)
- Test: `tests/test_intent_stability.py`

**Interfaces:**
- Consumes: Task 4's guarantee that `_classify_intent` never returns PEAK.
- Produces: engine behavior — a DROP that survives `_PEAK_PROMOTION_BEATS` consecutive commit-beats becomes PEAK; while in PEAK, DROP votes are absorbed (no PEAK→DROP→PEAK oscillation); any non-DROP consensus exits PEAK through the normal pipeline.

**Why:** PEAK's semantic is *temporal* — "sustained maximum energy after the drop". Feature thresholds can't express "after"; the engine's dwell counter can, in ~10 lines.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_intent_stability.py`, following the file's existing pattern for driving `_commit_intent` (it already has scenario helpers for the vote/dwell pipeline — reuse them):

```python
async def test_sustained_drop_promotes_to_peak():
    # Drive enough consecutive DROP-classified beats through _commit_intent to
    # pass vote consensus, commit DROP, then continue for _PEAK_PROMOTION_BEATS
    # more beats. The engine must promote DROP → PEAK exactly once.
    ...

async def test_peak_absorbs_drop_votes():
    # Once in PEAK, further DROP consensus must NOT switch back to DROP.
    ...

async def test_peak_exits_to_groove_on_consensus():
    # From PEAK, a GROOVE consensus (density easing) exits normally.
    ...
```

Write these as real tests using the same harness the file already uses for dwell/vote tests (build a `LightEngine` with stub clients exactly as neighboring tests do). The `...` above marks where you follow the file's established setup idiom — the assertions themselves must be concrete: check `engine._current_intent` after each phase.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_intent_stability.py -v -k peak`
Expected: FAIL (`_PEAK_PROMOTION_BEATS` not defined)

- [ ] **Step 3: Implement**

Constants block:

```python
# A DROP that survives this many commit-beats is promoted to PEAK
# ("sustained maximum energy after the drop"). ~15 s at 128 BPM.
_PEAK_PROMOTION_BEATS = 32
```

In `_commit_intent`, insert after `self._beats_in_current_intent += 1` (before the vote-consensus section):

```python
        # PEAK promotion: PEAK is not produced by the classifier — it is a DROP
        # that has lasted. Promote once the dwell counter shows sustained DROP.
        if (self._current_intent == LightIntent.DROP
                and self._beats_in_current_intent >= _PEAK_PROMOTION_BEATS):
            logging.info('[engine] [windowed] sustained DROP — promoting to PEAK')
            self._intent_vote_buffer.clear()
            self._beats_in_current_intent = 0
            self._current_intent = LightIntent.PEAK
            if self.event_buffer:
                self.event_buffer.set_intent(LightIntent.PEAK.value)
            await self.effect_controller.change_effect(LightIntent.PEAK)
            return
```

And in the post-consensus section, immediately after consensus is reached and BEFORE the `if self.event_buffer:` surfacing block, add PEAK's absorption rule. (Amended after review: placed before surfacing so the report/visualizer timeline holds `peak` through absorbed DROP votes — the timeline must reflect committed show state, and an absorbed vote is not a change.)

```python
        # PEAK absorbs DROP votes: easing back to plain DROP is not a show change.
        # Placed before the surface step so the intent timeline keeps reading
        # 'peak' — the lights are holding PEAK; the timeline must agree.
        if self._current_intent == LightIntent.PEAK and intent == LightIntent.DROP:
            return
```

PEAK must also inherit DROP's hysteresis in `_classify_intent` (amended after review: without this, a windowed density dip landing between DROP's exit and entry thresholds ejects PEAK where it would have held DROP — PEAK would be less sticky than the DROP it replaces):

```python
    # PEAK is sustained DROP — it keeps DROP's exit threshold.
    currently_drop = current_intent in (LightIntent.DROP, LightIntent.PEAK)
```

Tests must cover both amendments: the timeline stays `peak` during absorbed DROP votes, and a windowed density between exit and entry holds PEAK.

Check `_INVALID_TRANSITIONS`: `(PEAK, BUILDUP)` is already blocked — correct and unchanged.

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest -m "not integration"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/engine/light_engine.py tests/test_intent_stability.py
git commit -m "PEAK = sustained DROP: engine-level promotion after dwell, absorbs DROP votes"
```

---

### Task 6: Verification against ground truth + threshold calibration

**Files:**
- Create: `training/inspect_report.py`
- Test: manual verification runs (documented below) — no new pytest file; the integration suite is the automated gate.

**Interfaces:**
- Consumes: report `beats[]` feature columns from Task 3.
- Produces: `python training/inspect_report.py report.json` — prints (a) per-10s bins: mean rms, mean density, mean kick_strength, dominant intent; (b) the intent timeline with durations; (c) intent_distribution_sec. Used here for calibration and forever after for eyeballing any track.

- [ ] **Step 1: Write the inspection script**

```python
"""Inspect a simulation report: per-10s feature/intent bins + intent timeline.

Usage: python training/inspect_report.py report.json [--bin-sec 10]
"""
import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('report')
    ap.add_argument('--bin-sec', type=float, default=10.0)
    args = ap.parse_args()

    with open(args.report) as f:
        report = json.load(f)

    beats = report['beats']
    intents = report['intents']
    duration = report['duration_sec']

    def dominant_intent_at(t0: float, t1: float) -> str:
        best, best_overlap = '-', 0.0
        for block in intents:
            overlap = min(block.get('end', duration), t1) - max(block['t'], t0)
            if overlap > best_overlap:
                best, best_overlap = block['intent'], overlap
        return best

    print(f'{"bin":>12}  {"rms":>6}  {"density":>7}  {"kick":>5}  {"beats":>5}  intent')
    t = 0.0
    while t < duration:
        t1 = min(t + args.bin_sec, duration)
        rows = [b for b in beats if t <= b['t'] < t1]
        if rows:
            rms = sum(b.get('rms', 0.0) for b in rows) / len(rows)
            den = sum(b['onset_density'] for b in rows) / len(rows)
            kick = sum(b.get('kick_strength', 1.0) for b in rows) / len(rows)
            print(f'{t:>5.0f}-{t1:<5.0f}  {rms:>6.3f}  {den:>7.2f}  {kick:>5.2f}  {len(rows):>5}  {dominant_intent_at(t, t1)}')
        else:
            print(f'{t:>5.0f}-{t1:<5.0f}  {"-":>6}  {"-":>7}  {"-":>5}  {0:>5}  {dominant_intent_at(t, t1)}')
        t = t1

    print('\nIntent timeline (audience time = song time + look-ahead):')
    for block in intents:
        end = block.get('end', duration)
        print(f"  {block['t']:>7.1f} - {end:<7.1f}  ({end - block['t']:>6.1f}s)  {block['intent']}")

    print('\nDistribution:', report['metrics']['intent_distribution_sec'])


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the calibration loop**

```bash
uv run python auto_pilot simulate file samples/generate_eric_prydz_192k.mp3 --report report.json
uv run python training/inspect_report.py report.json
```

Compare against the ground-truth table at the top of this plan (remember the 2.5 s audience-time shift on intent blocks). **Acceptance criteria:**

1. kick_strength column: clearly separated populations — higher in main groove (45–76 s) and final drop (128–158 s) than in intro (0–20 s) and breakdown (76.5–90 s), with a gap wide enough to place `_KICK_PRESENCE_THRESHOLD` between them. First try Task 2's `_KICK_CAPTURE_BUFFERS` (6–12). **Known from Task 2's measurements:** the capture fix alone moved the median only 0.84 → 1.02 (ground truth 2.6–2.8) because the denominator — mean over ALL buffers — includes the on-beat kick spikes and this track's sustained rolling bass. If capture tuning can't separate the populations, change the denominator in `get_kick_strength()` to a background estimate that discounts beat spikes (e.g. median or lower-quartile of `_all_sub_bass_samples`), update Task 2's unit tests to the new semantics, and set `_KICK_PRESENCE_THRESHOLD` between the measured kick-absent and kick-present populations. The absolute 2.6–2.8 value is not the target — separation is; document the measured populations and chosen threshold in the commit.
2. The final drop (128.4–158.8 s) is dominantly DROP (PEAK acceptable in its tail). If it isn't, lower `_DROP_MIN_DENSITY_ENTER` toward the measured main-groove/drop density (keep entry > breakdown-section densities; keep exit ~0.5 below entry).
3. The breakdown (76.5–90.8 s) and intro (0–22.9 s) contain **no** DROP or PEAK blocks.
4. BREAKDOWN total ≤ 40% of the track (was 93.7%).
5. BUILDUP fires at least once in 22.9–45.5 s or 90.8–128.4 s (the builds/rebuild).
6. No PEAK block in the first 10 s (the 257.8-BPM warmup artifact is dead).
7. Intent changes: between 5 and 25 total (was 3; churn target ~4/min ±).
8. Determinism: run the simulate command twice — identical sha256 both times.

Iterate on the constants in `light_engine.py` until all eight hold. Record the final constants and the per-section measured densities/kick values in the commit message.

- [ ] **Step 3: Run the FULL suite (including integration)**

Run: `uv run pytest`
Expected: PASS. The integration tests assert evaluator PASS, timing, flush, duration, speed, and determinism on the new behavior. If `test_sample_song_evaluation_passes` fails on `intent_changes_count`/`unique_intents_count`, the classifier changes have made the show *worse*, not better — go back to Step 2, do not loosen the evaluator.

- [ ] **Step 4: Update the CLAUDE.md files (intent level only)**

- `lib/analyser/CLAUDE.md`: kick strength paragraph — transient-peak capture over a short pre-beat window (not a single-buffer snapshot), silence gating; BPM section — octave folding rationale; classifier design — new priority order with the risers-strip-the-kick rationale; PEAK — now an engine-level temporal promotion, not a feature threshold; note that beat records in reports now carry the full feature row (the training table).
- Root `CLAUDE.md`: Known Issues — delete the now-fixed bullets (“Sub-bass gate disabled … calibrate before enabling” stays; the beat-dropout and density-trend-warmup bullets stay); update the LightIntent table row for PEAK (“sustained DROP, promoted by dwell”); mention `training/inspect_report.py` in the analyser CLAUDE.md evaluation section.
- Do NOT copy threshold numbers into either file.

- [ ] **Step 5: Final commit**

```bash
git add training/inspect_report.py CLAUDE.md lib/analyser/CLAUDE.md lib/engine/light_engine.py
git commit -m "calibrate thresholds against reference structure; add report inspector; update docs"
```

Do not push and do not open a PR — the coordinator reviews the branch first.

---

## Self-Review Notes

- Task 4 deletes `_PEAK_MIN_BPM_*` while Task 5 re-adds PEAK via `_PEAK_PROMOTION_BEATS` — no orphan references remain (Task 4 Step 3 removes the test imports).
- Task 3's new `add_beat` kwargs default to neutral values, so `test_simulation.py` and the visualizer keep working unmodified.
- Task 6 is the only task whose “test” is a measured acceptance checklist rather than pytest — that is deliberate: the integration suite gates regressions automatically, while threshold calibration needs human-readable evidence.
