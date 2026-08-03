# NN runtime integration — the model takes the show, the rule engine leaves

**Branch** `nn_runtime_integration` off `master` (`c931bbd`, post-PR-8).
**Worktree** `C:\Users\Julian\Projects\soundswitch-nn-runtime-worktree`.
**Charter** `.superpowers/sdd/2026-08-01-nn-integration/charter.md` (gitignored tree;
its binding content is reproduced in this file, which is the committed artifact —
#108(c)).
**Owner rulings executed:** #141 (integration is the next phase; crispness@0.5 s is
the primary metric), #142 (rule engine retired, demolition cascade), #143 (YAMNet and
TensorFlow removed outright), #144 (no-fallback degradation accepted), and the
2026-08-01 UI directive — the Dash stage view must visibly move with the music under
the NN path, and that plus the prod-sim match is the phase's acceptance bar.

The one-line scope: **the fixed-lag decoder becomes the only thing permitted to say
what the lights are doing, and everything that existed to feed the hand-thresholded
classifier is deleted — including two whole dependencies.**

Shipping artifacts are frozen by #141(b): student `student_kd_t2_w05_s1234`, ONNX
sha256 `f1fe6ef7c3cc0dede24a7d572841b3eb2c381f123868f67dcf0e1d0298aa33b4`, geometry
F=3 / hop=1 s / backward 41 cells, decoder `reduced_plus_floors_x0.75`.

---

## Task 0 — census (done; these tables are the scope contract)

Verified by reading every file under `lib/` and `simulate/` at `c931bbd`, plus
`grep -rn aubio`, `grep -rlni yamnet|tensorflow` repo-wide.

### 0.1 Where features come from today

`lib/analyser/music_analyser.py` is the only feature producer. Every row is
classified by what happens to it.

| Feature / machinery | Produced by | Consumed by | Verdict |
|---|---|---|---|
| beat instants, beat count, `time_to_last_beat_sec` | `MadmomRhythm._BeatStage` | `on_beat` → OS2L, event buffer, commit timing | **SURVIVES** |
| `get_bpm` / `_fold_bpm` / `_beat_stream_times` | derived from the beat stream | OS2L wire, report, logs | **SURVIVES** (the DROP BPM gate that motivated the fold dies with the classifier; the fold stays because a folded tempo is what the OS2L wire has always carried) |
| `get_rms_energy` / `_rms_window` | per-buffer RMS | kick gate (dies), report column | **SURVIVES** — and gains a job, see 0.5 |
| `get_onset_density`, `get_onset_density_trend`, `_onset_times`, `_density_samples`, `_density_valid_from`, `_onset_epoch_seen`, `DENSITY_UNKNOWN`, `density_is_known` | `MadmomRhythm._OnsetStage` | `_classify_intent` only | **DELETE** |
| `get_kick_strength`, `_resolve_pending_kicks`, `_pending_kick_beats`, `_all_sub_bass_samples`, `_kick_ratios`, `KICK_UNKNOWN`, `_KICK_*` (6 constants) | mel bands 0–4 straddling the beat | `_classify_intent` only | **DELETE** |
| `get_sub_bass_ratio`, `_mel_energies_window` | mel bands 0–4 / total | `_classify_intent` only | **DELETE** |
| `get_spectral_centroid_trend`, `_centroid_window`, `_beat_centroid_samples`, `_mel_band_indices` | mel centroid across beats | `_classify_intent` only | **DELETE** |
| `MelFilterbank` (`aubio.pvoc` + `aubio.filterbank`) | the FFT + 40-band Slaney bank | the four rows above **and `_is_silence`** | **DELETE** — but see 0.5, it has one surviving consumer |
| `YamnetChangeDetector` (whole file), `_section_detection_enabled`, `_set_section_detection_enabled` | TF Hub YAMNet embeddings + MAD outliers | `handler.on_section_change()` only | **DELETE** (#143) |
| `_track_note` / `_NOTE_REFRACTORY` | fires off the **onset** stream | `LightEngine.on_note` → overlay 24-bar | **RE-SOURCE**, see 0.5 |
| `note_clicks` / `click_sound` (the `-d` beep) | fires off the **beat** stream already | monitored audio | **SURVIVES** unchanged |
| `_is_silence` / `_track_song_duration` / `_on_sound_start` / `_on_sound_stop` | filterbank energies | the silence→ATMOSPHERIC contract, all state resets | **SURVIVES**, re-based off RMS (0.5) |
| `DriftWatchdog` | loop pacing | shed ladder | **SURVIVES**, ladder rebuilt (0.4) |

`MadmomRhythm`: `_BeatStage` survives whole. `_OnsetStage`, `ONSET_THRESHOLD`,
`onsets_enabled`, `onset_epoch`, `set_onsets_enabled` and the whole
shed/restore/epoch protocol **DELETE** — the #131 "delete or replace the onset
chain?" question self-resolves, and ~11.2 % of a core comes back.

### 0.2 Where intents commit today

`lib/engine/light_engine.py`. This is the file the integration rewrites.

| Site | Role | Verdict |
|---|---|---|
| `_classify_intent` | the threshold branch | **DELETE** |
| `_classify_windowed`, `BeatRecord`, `_beat_history` | symmetric ±look-ahead window | **DELETE** |
| `_hold` | density-unknown fallback | **DELETE** (with its sentinel) |
| 8 threshold constants (`_BREAKDOWN_*`, `_BUILDUP_*`, `_DROP_*`, `_KICK_PRESENCE_THRESHOLD`, `_CENTROID_BUILDUP_TREND`) | tuning | **DELETE** |
| `_VOTE_BUFFER_SIZE` + `_intent_vote_buffer` | consensus guard | **DELETE** (#142) |
| `_MIN_DWELL_BEATS` + `_beats_in_current_intent` | dwell guard | **DELETE** (#142) |
| `_INVALID_TRANSITIONS` | transition guard | **DELETE** (#142 — the priors' `transition_allowed` is the successor, and it is fitted rather than chosen) |
| `_PEAK_PROMOTION_BEATS` + the DROP→PEAK promotion | run-length show device | **DECISION D8** — not named by #142; recommend keeping, re-denominated in decoder bars |
| `_commit_intent` | the whole stability pipeline | **REPLACED** by decoder commits |
| `on_section_change` | YAMNet-driven effect refresh | **DELETE**, behaviour replaced by D9 |
| `on_beat` | feature read + OS2L beat + event buffer + commit enqueue | **SLIMS** — keeps OS2L/event-buffer/bar-grid feed, loses every feature read |
| `on_100ms_callback` + `_BEAT_ABSENCE_SEC` | silence→ATMOSPHERIC | **SURVIVES** |
| `_apply_intent`, `_enqueue_or_apply`, `_publishable_bpm`, `on_sound_start/stop`, `on_cycle`, `on_note`, `on_1sec/10sec` | plumbing | **SURVIVE** (`on_note` re-sourced, 0.5) |

Everything downstream is untouched: `EffectController`, `effect_definitions`,
`DelayedCommandQueue`, `MidiClient`, `Os2lClient`, `OverlayClient`, `EventBuffer`
(minus four report columns), `PyAudioClient`, `Clock`, the visualizer.

### 0.3 What the rule engine's death takes with it, outside `lib/`

| Site | Verdict |
|---|---|
| `tests/test_classify_intent.py`, `tests/test_intent_stability.py` | **DELETE** whole |
| `tests/test_music_analyser.py` | density/kick/centroid/sub-bass/YAMNet cases delete; beat/BPM/silence/reset cases stay and gain the new silence gate |
| `tests/test_madmom_rhythm.py`, `tests/test_madmom_contract.py` | onset-stage and threshold cases delete |
| `tests/test_drift_watchdog.py` | rewritten for the new ladder |
| `tests/test_pipeline_digest.py` + `tests/fixtures/pipeline_digest_baseline.json` | the filterbank fingerprint gate loses its subject; re-cut around the NN path |
| `training/train.py` (aubio source/onset/tempo/pvoc/mfcc/filterbank) | **DELETE** — a standalone offline mel dump, superseded by MERT |
| `training/onset_operating_point.py` (+ `.json`, `_draws.md`) | **DELETE** — it exists to calibrate a chain being deleted |
| `training/measure_realtime.py` | aubio rows delete; the harness stays and gains the NN stages |
| `training/build_training_table.py::FeatureExtractor` (rebuilds aubio objects) | **DELETE**; the label-join / `realign_intents` half **STAYS** — `run_eval_set.py` depends on it |
| `training/multi_channel_yamnet.py`, `training/yamnet_testing.py` | **DELETE** (#143) |
| `pyproject.toml` | `aubio`, `tensorflow`, `tensorflow-hub` out of base deps; the `tensorflow-io-gcs-filesystem` override and the aubio `CFLAGS` build variable go with them |

**Disclosed consequence:** deleting the aubio front-end retires this repo's ability
to regenerate the v1/v2 **mel** model's inputs. That is intended (MERT supersedes
mel, #121(c) named aubio's GPL as the tree's largest licence exposure), but it is a
one-way door and belongs in the PR body, not in a diff nobody reads.

### 0.4 The shed ladder after the demolition

`ShedLevel` today is `NONE | SECTION_DETECTION | ONSET_DETECTION`. **Both tenants are
being deleted.** The ladder is rebuilt around its one remaining tenant, the GPU
feature stage — and gains a second input it never had: drift cannot see a CUDA fault,
a driver reset or a sleep/resume context loss (#143's named failure modes), all of
which are silent to a pacing measurement. So the watchdog takes *health* alongside
drift, and the ladder becomes `NONE | NN_SHED`, where `NN_SHED` **is** the
degradation contract of #144.

### 0.5 Three surviving components lose their supplier

Not obvious from any single file; each needs a decision, and each is a behaviour
change to something the demolition was supposed to leave alone.

1. **`_is_silence` runs on filterbank energies.** It is the trigger for the entire
   sound-start/sound-stop machinery — every state reset, the ATMOSPHERIC timer, the
   OS2L song boundary. Deleting the bank without replacing this deletes song
   detection. → **D5**: re-base on the already-computed RMS, gated by a golden
   fixture that pins sound-start/stop instants on the fixture tracks *before* the
   swap.
2. **`on_note` (the overlay 24-channel light bar) is driven by onsets.** The onset
   chain is being deleted. → **D6**: re-source on beats, matching the owner's
   earlier ruling that moved the `-d` click from onsets to beats. The bar advances
   ~2.1×/s instead of ~3.6×/s; that is the same UX change already accepted once.
3. **`GROOVE` becomes unreachable.** The model's class space is
   `(intro, buildup, breakdown, drop, outro)` — there is no groove. `BANK_2F/G/H`
   would silently stop being used. → **D7**.

### 0.6 The Dash visualizer's dependencies on deleted code

Owner directive (2026-08-01, binding): **the UI must visibly move with the music
under the NN path** — intent timeline *and* the stage-simulation lights. The
visualizer is therefore acceptance surface, not a nice-to-have. Its coupling to the
demolition is small, precise, and would all fail quietly rather than loudly.

| Site | Dependency | Verdict |
|---|---|---|
| `visualizer_app.py:133` `beat_size = max(16, b['strength'] * 40)` | `EventBuffer` computes `strength` from **onset density** | **BREAKS SILENTLY** — density is deleted, so `strength` becomes 0.0 and every beat marker clamps to the floor size. The chart still renders; it just stops meaning anything |
| `INTENT_CONFIG['groove']` (colour, slot set, decay, glow, legend) | `LightIntent.GROOVE` | dies with D7. `_intent_config` falls back to `_DEFAULT_CONFIG` for unknown keys, so nothing raises — the legend would just advertise an intent the show can no longer enter |
| `visualizer_app.py:241` `abs(mean_delta - 2.5) < 0.05` | **`LOOK_AHEAD_SEC` hardcoded as a literal** | B1 moves that number. The health dot would go amber for the whole show and nobody would know why. Must read the queue's own target |
| `EventBuffer.add_beat(kick_strength=…, centroid_trend=…, sub_bass_ratio=…)` | deleted features | stored but **never read by the UI** — only the report carries them. Dropping the kwargs is safe for the visualizer, and is a report-schema change (Task 12's anchors) |
| `simulate/runner.py:40` — `music_analyser.yamnet_change_detector.detect_change = lambda …: False` | the monkeypatch that kept TensorFlow out of simulation | deletes with YAMNet; `build_simulation` must instead construct the NN stages |
| `simulate/cli.py::_run_file_realtime_ui` / `run_realtime` | build the pipeline on the **system clock**, real-time paced | these are the paths the UI runs on, so **the GPU thread must work under `--ui`**, not only in headless fast sim. With Dash and `Os2lSender` this makes four threads |

Nothing else in `visualizer_app.py` touches a deleted symbol: the timeline reads
`intents` / `effects` / `sound_events`, the stage reads `intent` + effect slots, and
the metric strip reads bpm/beats/elapsed/timing — all of which survive.

**What replaces the deleted panels — minimal, per the directive.** Beat marker size
loses its density source and becomes constant; the freed visual channel is *not*
re-purposed speculatively. The one genuinely new panel worth having is the decoder's
own state, because it is what the show is now driven by and it is otherwise invisible:
current class posteriors (5 bars) and the committed-bar cursor with its lag. That is
one row of the metric strip and one small figure — see **D14**.

### 0.7 What master does not have yet

| Missing | Where it lives | Consequence |
|---|---|---|
| `floor_bars`, `outro_escape`, `temperature`, `temper()` in `DecodeParams` | `soundswitch-phase-b-worktree` | **Highest-risk latent defect in the whole integration.** Master's `load_decoder_config` filters by `dataclasses.fields(DecodeParams)`, so loading `reduced_plus_floors_x0.75` on master **silently yields a different decoder with no error**. Task 5 lands the decoder generation first, with a test that a config carrying an unknown key raises. |
| the streaming MERT extractor | `training/nn/ceiling/stream_extract.py` (offline, whole-wav, npz sidecars) | port to a live ring-buffer object |
| `OnlineCRNN` / `online_export` / the ONNX session factory | `training/nn/ceiling/`, `training/nn/export_onnx.py` | the live path needs the session factory's determinism pinning (1 intra / 1 inter op, sequential), not the training code |
| a **live bar grid** | nowhere — `decoder.bar_grid` reads downbeats out of a Raveform annotation CSV | **BLOCKER B2** |
| `transformers` as a declared dependency | nowhere (imported lazily by `extract.load_encoder`) | undeclared dependency on the live path |
| PR #9's downbeat machinery | branch `downbeat_tracking` (`ab9d264`) — **verified NOT an ancestor of master** despite #137's note | see B2 |

---

## The three findings that outrank the task list

These came out of the census arithmetic, not out of building anything. Two are
blockers with owner-facing decisions; the third is a design constraint that changes
the shape of the code.

### B1 — the look-ahead budget does not close, and it is out by ~13 s

The 7.9938 s figure is the model's **feature-to-posterior** future dependence:
F (3.0) + hop (1.0) + head future (43 cells × 0.09287981859410431 s = 3.9938 s).
The decoder's lag is **on top of it and was never inside the 8 s**: bar *b*'s
observation needs bar *b* to finish, and `lag_bars = 3` commits three bars later. At
the corpus median bar (1.875 s, 128 BPM 4/4) that is 1.875 + 5.625 = **7.5 s**.

    audio → final posterior      7.9938 s   (pinned by the shipped .onnx.json)
    posterior → committed bar    7.5    s   (lag_bars 3, median bar)
    ───────────────────────────────────────
    audio → committed intent    ~15.5   s

`LOOK_AHEAD_SEC` on master is **2.5 s** (`lib/main.py:14`, mirrored in
`simulate/runner.py:12`, matching `playback_delay_seconds` in dmx-enttec-node). The
charter's brief said "8 s, playback_delay_seconds untouched"; production is at 2.5,
and 8 would not be enough either. `lib/analyser/CLAUDE.md` already recorded this —
"the show's look-ahead must grow to the decoder's budget… the two systems are not
latency-matched today" — but the size of the gap had not been written down.

**And the queue's role inverts.** Today the engine runs *ahead* of the audience and
`DelayedCommandQueue` holds commands back by the full `LOOK_AHEAD_SEC`. With the NN
the engine runs *behind*. The correct relation is:

    queue_delay = playback_delay − chain_latency,  and it must be ≥ 0

Today `chain_latency ≈ 0`, so `queue_delay = playback_delay = 2.5` — which is why
the current code can conflate the two into one constant. It cannot survive the NN.

**Levers, all cheap to price, none free:**

| option | playback_delay needed | cost |
|---|---|---|
| ship `lag_bars = 3` as measured | ~15.5 s | the whole show, and the DJ's own monitoring, sits 15.5 s behind the decks |
| drop to `lag_bars = 0` | ~9.9 s | commits greedily off the frontier with no backtrace; **unmeasured** — the frontier was swept at lag 3 only |
| keep `playback_delay = 2.5` | — | intents land ~13 s late: a whole section late, not a "late transition". Not shippable, and #129's forgiveness of lateness does not stretch this far |

**Task 1a resolves this at zero GPU cost**: the accuracy/crispness-versus-lag curve
over the shipped student's *existing* val posteriors, `lag_bars ∈ {0,1,2,3,4}`,
producing a priced table — "each bar of lag you hand back costs X crispness@0.5 s
and Y contested macro; each bar you keep costs 1.875 s of show delay." That is a
decision-grade artifact for the owner and it is a few minutes of CPU.

### B2 — there is no live bar grid, and the decoder decides per bar

`decoder.bar_grid()` parses downbeats out of a Raveform annotation CSV. Every
frontier number — including the 0.6779 crispness@0.5 s the pick is being chosen on —
was measured on **expert downbeats**. Live, madmom's online stack gives beats with no
bar phase, and the live path is forbidden from importing the offline downbeat tracker
(`tests/test_madmom_contract.py:104` asserts it).

PR #9 built a downbeat head for exactly this; `git merge-base --is-ancestor` confirms
it is **not on master**, and its own re-measured figure on madmom's stream is 0.50
against a recommended 0.55 gate (#133) — it is not shipping-ready, and finishing it is
training work, which #141(c) parks.

**Task 1b resolves this at zero GPU cost too**, and it is the ablation PR #9's own
spec pre-registered as the go/no-go for live bar-snapping: re-decode the shipped
student's existing val posteriors on (i) the expert grid and (ii) a beat-derived
4-beat grid at each of the four possible phases, and report the crispness/macro
delta and its phase sensitivity. If the delta is small, live bars are 4 madmom beats
and B2 closes for free. If it is large, the integration needs the downbeat head and
that is an owner decision about scope, taken with a number in hand.

### B3 — the MERT stage cannot run on the audio thread

81 ms per pass at a 1 s cadence is 8.1 % GPU duty, which reads harmless. But the
audio input **drops rather than queues**, the buffer period is 5.805 ms, and 81 ms is
**~14 buffers**. Once per second the pipeline would throw away 14 buffers of audio;
under GPU contention p95 triples to ~210 ms (#124), i.e. ~36 buffers. Add the
student's ~15 ms burst (10.8 cells arrive together after each pass) and the inline
design loses ~1.7 % of all audio, in periodic gouges rather than as smooth lag —
precisely the failure the drift watchdog was built to notice and cannot fix.

So the GPU stage gets its own thread with a lock-free hand-off, and the audio thread's
only new work is a 24 kHz resample and a ring-buffer write. This is a threading-model
change to a pipeline whose own docs warn that mixing threading models "requires care
when touching shared state" (the `Os2lSender` precedent). **D3** fixes the shape.

---

## Design decisions taken before any code

**D1 — one module per stage, three modules total.**
`lib/analyser/mert_stream.py` (resample + ring buffer + encoder + cell accumulator),
`lib/analyser/section_model.py` (feature ring buffer, forward state, ONNX session,
per-cell posteriors), `lib/engine/section_decoder.py` (bar grid feed, `FixedLagViterbi`,
`Decision` → `LightIntent`). Nothing above a module knows the geometry below it, the
same principle `madmom_rhythm.py` established for the hop mismatch.

**D2 — the geometry is read from the shipped artifact, never retyped.**
`window_cells`, `future_cells`, `label_frame_sec`, `input_dim` and the sha come from
`online_step.onnx.json`. A test asserts the sha matches #141's value and that
`F + hop + future_sec ≤ 8.0` computed from the file. A constant copied into `lib/`
would drift silently the first time a model is re-exported.

**D3 — the GPU stage is a thread, and the hand-off is a queue of whole passes.**
Audio thread: resample to 24 kHz, write into a 30 s ring buffer, publish a monotonic
sample index. GPU thread: once per hop, snapshot the ring, encode, emit the cells
whose centres fall in `[prev_hi, T − F)`, push them. Consumer: student `step()` per
cell, then the decoder. The queue is bounded; overflow is a shed event, not a stall —
the audio thread must never block on the GPU.

**D4 — the resampler is part of train==deploy and must be measured, not assumed.**
Offline features were extracted from **ffmpeg**'s resampler. A live
`resample_poly(x, 80, 147)` is a different filter, and the model has never seen its
output. Nothing fails loudly if this is wrong — the posteriors just get quietly
worse. Task 6 compares one track's cells both ways and reports max cell delta and
argmax disagreements, exactly as `online_export.verify` does for torch-vs-ONNX.

**D5 — silence detection moves to RMS, behind a golden fixture cut first.**
The energy gate is `all(|mel| < 1e-4)`. RMS is already computed per buffer. The
threshold is chosen by matching sound-start/stop *instants* on the fixture tracks,
not by picking a plausible number — the same discipline that set the onset threshold
in the madmom migration (D6 there).

#### D5 as measured — `_SILENCE_RMS = 1.5e-4`, and the instant no RMS reading reproduces

Executed in Task 4 (`3a2f099`). `_track_song_duration`'s state machine was replayed
offline over the per-buffer RMS of all three fixture tracks, recorded through the
*unmodified* pipeline. The replay is checked rather than asserted: driven by the
recorded **mel** decisions it returns the committed instants exactly on all three.
Nine readings of that RMS — instantaneous, plus mean/rms/max/min pooled over 4, 8 and
26 buffers (26 being `get_rms_energy`'s own window, with `_reset_state`'s clear
modelled) — were then each swept against a 240-point log grid from 1e-7 to 5e-2. A
track's *feasible band* is the set of thresholds reproducing its committed
sound-start/stop instants exactly:

| reading | NyEKXA7_6z0 | PNpXKsge4xM | SBnxzXkc_qw | all three |
|---|---|---|---|---|
| instant | [2.445e-3, 2.445e-3] | [1e-7, 1.658e-4] | [9.03e-6, 2.583e-3] | **none** |
| mean4 / rms4 | empty | [1e-7, 1.406e-4] | [8.55e-6, 2.074e-3] | none |
| mean8 | empty | [1e-7, 1.406e-4] | [7.66e-6, 1.665e-3] | none |
| mean26 / rms26 | empty | [1e-7, 1.331e-4] | [6.15e-6, 1.858e-3] | none |
| max4 | empty | [1e-7, 1.658e-4] | [9.03e-6, 2.583e-3] | none |
| max26 | empty | [1e-7, 1.752e-4] | [9.03e-6, 3.399e-3] | none |
| min4 | empty | [1e-7, 1.331e-4] | [7.25e-6, 1.759e-3] | none |

**The binding pair is a contradiction, not a near miss.** `NyEKXA7_6z0` needs its
run-out called *silent* at a buffer RMS of ~1.6e-3; `PNpXKsge4xM` needs its fade-out
called *sound* at ~1.0e-4, or a stop appears 1.9 s before the audio ends. Fourteen
times apart, and no monotone reading of a waveform RMS can separate them, because the
retired gate was not measuring the same quantity: a Slaney-normalised band energy
under 1e-4 in *every one of 40 bands* is a statement about the spectrum, and a
broadband noise floor satisfies it at an RMS a tonal reverb tail does not.

**Chosen: 1.5e-4** — the middle of the flat part of the error curve ([1.4e-4, 1.66e-4]
all cost 168 ms, and nothing below 1.4e-4 costs less), with ~10 % headroom under the
constraint that binds it:

| T | NyEKXA7_6z0 stop | error | PNpXKsge4xM exact |
|---|---|---|---|
| 1e-5 | 362.760 | +575 ms | yes |
| 1e-4 | 362.446 | +261 ms | yes |
| **1.5e-4** | **362.353** | **+168 ms** | **yes** |
| 1.7e-4 | 362.353 | +168 ms | no — spurious stop at 255.71 |
| 2.4e-3 | 362.185 | 0 ms | no — spurious stop at 255.71 |

**The one survivor that moves.** `tests/fixtures/pipeline_digest_baseline.json` changes
by exactly one value: `NyEKXA7_6z0.survives.sound_events[1].t`, 362.185 → 362.353.
Everything else in `survives` on all three tracks is byte-identical, every beat-time
hash included (701 / 729 / 1050 beats, unchanged). The movement is late, never early;
it lands inside a run-out whose audio has already decayed past −70 dBFS, and the
track's digital silence begins at 362.5 either way. The brief forbids survivor
movement, so this was re-cut deliberately rather than absorbed: the alternative is
holding the stop instant exactly and paying for it with `PNpXKsge4xM` losing 1.9 s of
its outro to a *fabricated* song boundary. An inaudible 168 ms of lateness on one
run-out is the smaller lie, and reverting it is a one-line change if the ruling goes
the other way.

**D6 — the overlay light bar moves from onsets to beats.** Its supplier is being
deleted; beats are the ruling already taken for the `-d` click.

**D7 — GROOVE's effect pool merges into BREAKDOWN's rather than going dark.**
The class space has no groove. Deleting the intent would retire `BANK_2F/G/H` from
the show; folding the pool into BREAKDOWN keeps six banks in rotation behind the
class the corpus actually labels. `LightIntent.GROOVE` itself is removed — an enum
member no path can produce is a lie. **Flagged for owner confirmation**: this is an
audience-visible choice, not a mechanical consequence.

**D8 — PEAK survives as an engine-level run-length device, re-denominated in bars.**
#142 named the vote, dwell and transition guards; it did not name PEAK. PEAK is not
a classifier rule — it is "a committed DROP that has lasted", which the decoder
expresses cleanly as a drop run of N bars. Keeping it preserves `BANK_1F/G/H` and
the sustained-drop look. **Flagged for owner confirmation.**

**D9 — `on_section_change`'s effect refresh is replaced by boundary-head events.**
The audience-visible behaviour to preserve is "the effect refreshes inside a long
same-intent section". Class boundaries cannot express that (same class either side).
The model already emits a `boundary_logit` per cell, currently read only as a decoder
hazard. A peak in it *inside a held intent* re-rolls the effect from the current
pool. Same signal, no new model, and it is a strictly better trigger than YAMNet's
cosine outliers because it was trained on section boundaries.

**D10 — state resets ride the existing sound-stop machinery.** `_on_sound_stop` →
extractor ring buffer cleared, forward GRU state to `None`, decoder `reset()`. The
feature ring re-primes from `mean_frames` (corpus mean in raw units), never zeros —
zeros are not silence after the input affine, and the phase-b tests already encode
this contract.

**D11 — degradation holds, it does not guess.** `NN_SHED` means: stop consuming
posteriors, hold the current intent, keep beats, keep the silence timer, log loudly
at WARNING with a rate limit, attempt extractor reinit on a backoff, resume on
success. There is no second classifier and none is wanted (#144).

**D12 — the feature sidecar is the decode cache one layer up.** `simulate file`
caches per-pass extractor output beside the audio, keyed by
`(model_sha, layers, F, hop, chunk_sec, sample_rate, audio mtime)`. Cold runs need a
GPU; warm runs are pure CPU and byte-deterministic. The honest statement in the docs:
sim=prod holds bit-exactly for everything downstream of the extractor, and the
extractor is *replayed* — the same contract the `.npy` decode cache already has, and
the reason the determinism guarantee is stated as "given cached sidecars".

**D13 — demolition comes early, and the branch's intermediate state is the
degradation contract.** Between the demolition task and the rewire task the branch
produces beats + silence + held intent and nothing else. That is not a broken
half-state to be hurried through — it is exactly `NN_SHED`, and it gets its golden
fixtures there, where it is the only thing running.

**D14 — the visualizer is acceptance surface, and it gets exactly one new panel.**
Owner directive: the stage view must visibly move with the music under the NN. Three
of its couplings to deleted code fail *silently* (0.6), so each is fixed with a test
rather than by looking at the screen once. The single addition is a decoder-state
row — five class posteriors plus the committed-bar cursor and its lag — because the
thing now driving the show is otherwise entirely invisible, and because a stuck
decoder and a quiet passage look identical from the stage view alone. No other panel
is added: the directive says give a visual sense of what is happening, not build a
diagnostic console. The hardcoded `2.5` becomes a read of the queue's own target, so
the health indicator keeps working after B1 moves the delay.

---

## Measurements already on disk (they shaped this plan)

| quantity | value | source |
|---|---|---|
| student CPU | 1.406 ms/cell median, 1.5497 p95 → 66.06× realtime, 1.51 % of a core | `…/student_kd_t2_w05_s1234/realtime.json` |
| MERT GPU | 81 ms per 30 s pass at 1 s cadence = 8.1 % duty; ~210 ms p95 under contention | Phase A / #124 |
| MERT VRAM | 1.30 GB; usable budget on this box ~5.7 GB (desktop holds 2.5 of 8) | #124 |
| rest of stack | 0.66 of a core over 186 s | `online_export.REST_OF_STACK_CORE_FRACTION` |
| onset chain being deleted | ~11.2 % of a core | #131 |
| aubio filterbank being deleted | 0.028 ms/buffer ≈ 0.5 % of a core | madmom migration Task 7 |
| ONNX parity | max label delta 3.34e-6; 0 argmax disagreements over 400 free-running cells | `online_step.onnx.json` |
| decoded val (the show's expected quality) | contested 0.7099, all-class 0.7066, flicker 0.3206 vs deployed 0.6160 / 0.6404 / 0.4862 | #140 |

Net CPU expectation: 0.66 − 0.112 (onsets) − 0.005 (filterbank) − YAMNet + 0.015
(student) ≈ **0.55 of a core or less**, before the GPU thread. The demolition pays
for the model with change left over. This is a prediction; Task 15 measures it.

---

## The crispness-first frontier re-rank (#141(b), done, zero GPU)

Re-ranked all 9 rows of `…/student_kd_t2_w05_s1234/frontier.json` (val-215,
1417.35 audience-minutes) by `boundary_f1["0.5"]` instead of the published
`utility`, which used flicker@2.0 s only.

| rank by b@0.5 | config | **b@0.5** | b@2.0 | macro_contested | flicker@2.0 | switches/min | false-drop/min |
|---|---|---|---|---|---|---|---|
| 1 | **reduced_plus_floors_x0.75** ← the pick | **0.6779** | 0.7142 | **0.7099** | **0.3206** | 1.098 | 0.0430 |
| 2 | reduced_plus | 0.6551 | 0.6878 | 0.7019 | 0.3383 | 1.081 | 0.0416 |
| 3 | reduced_plus_floors_x1.25 | 0.5741 | 0.6170 | 0.6921 | 0.4010 | 1.061 | 0.0388 |
| 4 | shipped_frozen | 0.5444 | 0.5750 | 0.6839 | 0.4292 | 1.038 | 0.0452 |
| 5 | reduced_plus_floors_x1.5 | 0.5380 | 0.5667 | 0.6793 | 0.4390 | 1.039 | 0.0395 |
| 6 | reduced_plus_floors_x2 | 0.4620 | 0.4906 | 0.6380 | 0.4482 | 0.945 | 0.0437 |
| 7 | reduced_plus_dwell_1s | 0.0124 | 0.6754 | 0.6867 | 0.3516 | 1.081 | 0.0430 |
| 8 | reduced_plus_dwell_2s | 0.0098 | 0.0301 | 0.6760 | 1.0451 | 1.079 | 0.0430 |
| 9 | reduced_plus_dwell_4s | 0.0059 | 0.0177 | 0.6536 | 1.0577 | 1.078 | 0.0473 |

**The pick does not change, and it does not have to be argued for.**
`reduced_plus_floors_x0.75` is not merely first on crispness — it is simultaneously
first on `boundary_f1` at all three tolerances, first on contested macro, and lowest
on flicker@2.0. There is no trade to arbitrate: #130's exchange rate never engages,
and re-running the ranking with crispness in the utility (macro − 0.20 × relative
flicker@0.5, shipped as reference) reproduces the same order — 0.5539 vs 0.5401 for
runner-up. Margins: **+0.0228 over `reduced_plus`, +0.1335 over `shipped_frozen`.**

Two things worth carrying forward rather than filing:

- The floors ladder is **monotone** in crispness (x0.75 → x1.25 → x1.5 → x2 gives
  0.678 → 0.574 → 0.538 → 0.462). Shorter duration floors buy crispness directly.
  The ladder was never swept **below** 0.75 — that is a free follow-up, val-only.
- The dwell rows **collapse** at 0.5 s tolerance (0.006–0.012 against 0.678). This is
  the sharpest possible confirmation of #136(d)/#140: post-decoder dwell does not
  reduce switching, it mis-times it, and the 0.5 s lens shows the damage that the
  2 s lens partly hid. `DO NOT SHIP` stands, now with a crispness number behind it.

Caveat kept honest: the whole frontier is measured at `lag_bars = 3` **on expert
downbeats**. B1 and B2 both move numbers in this table, which is why Task 1 runs
before anything is built.

---

## Tasks

Each task ends in a commit and an appended report in
`.superpowers/sdd/2026-08-01-nn-integration/progress.md`. Estimates are agent-hours
excluding review rounds. TDD throughout; red before green.

### Task 1 — resolve B1 and B2, at zero GPU cost — **GATE** — 4 h
Nothing else starts until the owner has ruled on the look-ahead.
- **1a** lag sweep `{0,1,2,3,4}` over the shipped student's existing val posteriors:
  crispness@0.5 s, contested macro, flicker, switches/min, false-drop/min per lag,
  plus the show-delay each lag implies at the corpus median bar. One table.
- **1b** bar-grid ablation: same posteriors decoded on the expert grid vs a
  beat-derived 4-beat grid at each of the four phases; report the delta and the
  phase sensitivity. This is PR #9's pre-registered go/no-go, run at last.
- Deliverable: a decision package (both tables + a recommendation) into
  `progress.md` and the coordinator. **No code in `lib/` yet.**

### Task 2 — golden fixtures FIRST, re-scoped — 4 h
The old contract (NN-off reproduces today's show) is void (#142). The new one:
- pin what **survives** on three fixture tracks — beat count and beat-time hash,
  sound-start/stop instants, OS2L beat wire, MIDI command ordering, report schema
  keys, timing-log accuracy;
- pin the **degradation state** (beats + silence + held intent) as a first-class
  fixture, since D13 makes the branch pass through it;
- extend `training/pipeline_digest.py` accordingly; the filterbank fingerprint is
  retired **in this commit**, with the reason in the diff, not silently.
Red first: each new digest field must fail before it exists.

### Task 3 — dependency surface — 3 h
`torch`, `transformers`, `onnxruntime` become **base** dependencies (they run the
show); `tensorflow`, `tensorflow-hub`, `aubio` and their build shims leave.
`uv.lock` regenerated. A test asserts the live path imports neither `tensorflow` nor
`aubio`. Record in the file *why* torch is pinned and why `transformers` was
previously undeclared. Note in the PR: base install gains ~2.5 GB of CUDA wheels and
loses TensorFlow.

### Task 4 — the demolition — 8 h
Everything in census 0.1–0.3 marked DELETE, in one reviewable commit per layer
(analyser features → engine classifier → tests → offline scripts → deps). D5 and D6
land here because they are the two deletions that would otherwise remove behaviour
silently. After this task the branch is the degradation state and Task 2's fixture
for it must pass. `grep -rn aubio` and `grep -rni yamnet\|tensorflow` return only
historical documents. Skip count and collected count reconciled explicitly (#47/#59).

### Task 5 — land the decoder generation on master — 3 h
Port `floor_bars` / `outro_escape` / `temperature` / `temper()` / the `__post_init__`
coercion / the coverage guard from `phase_b_online` into `training/nn/decoder.py`.
**Red first on the latent defect**: a test that loading a config with a key
`DecodeParams` does not know **raises** — today it is silently dropped, which would
have shipped a decoder nobody chose. Commit the chosen config as a file rather than
synthesising `floors×0.75` at runtime.

### Task 6 — `lib/analyser/mert_stream.py`, TDD — 8 h
Resample (D4, with the ffmpeg-parity measurement), 30 s ring buffer at 24 kHz,
`pass_schedule` geometry, encoder load with sha check, cell accumulator with
forward-fill, `reset()`. Unit-tested against a fake encoder so tests stay fast; one
model-loading integration test proves the real stack streams, is deterministic across
two runs in one process, and reproduces the offline sidecar for a track prefix.
Asserts: no emitted cell sees more future than `F + hop`; startup uses the short
buffer it has rather than waiting.

### Task 7 — `lib/analyser/section_model.py`, TDD — 4 h
Feature ring buffer primed from `mean_frames`, forward state carried, the 46-cell
window, the pinned ONNX session (1 intra / 1 inter op, sequential), sha verification
that fails **at startup** rather than at the first beat, `reset()`. The phase-b
streaming-equivalence test is ported as the acceptance test.

### Task 8 — `lib/engine/section_decoder.py` + the bar grid — 6 h
Bar grid per Task 1b's ruling, `FixedLagViterbi.push()` per bar, `Decision` stream
out. The `_psi` ring buffer is bounded to `lag_bars + 1` (its docstring already says
only those are read — a live decoder must not grow a list for the length of a set).
Reset at song boundaries.

### Task 9 — class → `LightIntent`, and the engine rewire — 6 h
The map (intro/outro → ATMOSPHERIC, buildup → BUILDUP, breakdown → BREAKDOWN,
drop → DROP, D8's promotion → PEAK, D7 folding GROOVE's pool). `_commit_intent`
becomes "a decoder decision arrived". `on_beat` slims to OS2L + event buffer + grid
feed. The queue delay becomes `playback_delay − chain_latency` per B1, with the
chain latency *measured* by the pipeline and logged at startup, not assumed.

### Task 10 — threading model and backpressure — 8 h
D3's GPU thread and bounded hand-off. `DriftWatchdog` rebuilt: `NONE | NN_SHED`, with
drift **and** a stage-health input. Shed and restore both clear the extractor's ring
and the decoder's state — a stage fed no audio during a gap holds pre-gap music, the
lesson the onset chain already taught. Every transition logs.

### Task 11 — degradation and reinit, loud — 4 h
D11. Simulated fault injection for each of #143's four failure modes: VRAM spill,
sleep/resume context loss, a raised CUDA error, a hung pass. Each must produce hold +
loud log + reinit + resume, with a test per mode. Rate-limited logging so a persistent
fault does not become its own outage.

### Task 12 — sim = prod, and determinism — 6 h
D12's sidecar cache; `simulate file` runs the identical NN path. Two fresh processes,
same file, byte-identical reports on all three fixture tracks, cold and warm cache.
Any RNG in the new path named and pinned. New checksums recorded as the new anchors.

### Task 13 — boundary-triggered effect refresh — 3 h
D9. Threshold set by matching the refresh *rate* YAMNet used to produce, not by
picking a number — the same discipline as D5 and the onset threshold before it.

### Task 14 — realtime and soak — 4 h + 30 min wall
Per-buffer cost of the new `analyse()` vs the pre-branch baseline, mean/p95/max
against the 5.805 ms period; GPU-thread pass latency distribution; queue depth
distribution. Then 30+ minutes of live capture across multiple back-to-back tracks —
song boundaries exercised, not avoided. Report max backlog, per-buffer p99/tail, every
shed/recover, every reinit. The gate is #126's: the whole stack keeps up, not any
component's share.

### Task 15 — eval-set baseline, re-cut once — 4 h
`run_eval_set.py --write-baseline`, once, at the settled tip. Add crispness@0.5 s as
a headline column (#141a). Disclose every metric's delta against the rule engine's
committed baseline — this is the first labelled scoring of a neural show and the PR's
headline number.

### Task 16 — the UI, end to end — 5 h
Owner directive; this and the prod-sim match are the acceptance bar for the phase.
- fix all four silent couplings from census 0.6 (beat `strength`, the GROOVE entry,
  the hardcoded `2.5`, the YAMNet monkeypatch), each red-first — a test that fails
  because the beat markers went constant is worth more than a screenshot that looked
  fine;
- add D14's decoder-state row;
- make `build_simulation` construct the NN stages, and prove the GPU thread runs
  under real-time pacing, not only under the virtual clock;
- **run `simulate file --ui` on the integrated pipeline** and record evidence that
  the intent timeline advances *and* the stage lights change as intents change —
  the snapshot stream sampled over a full track, showing intent transitions reaching
  the stage slots, plus a captured frame per distinct intent;
- **run `run --ui` during Task 17's VB-cable session** and confirm the same live,
  with Dash, the GPU thread, `Os2lSender` and the audio loop all running together.

Acceptance: a track's UI session shows every intent the report says it committed,
the stage look changes at those instants, and the timing indicator is green.

### Task 17 — live validation — 4 h
VB-cable run on the real rig, with `--ui` up (Task 16); prod-sim match on identical
samples to the madmom migration's methodology (beat agreement, intent-block
agreement, the timing distribution). Discrepancies explained, not tolerated.

### Task 18 — docs, grep proofs, PR — 4 h
Root `CLAUDE.md` and `lib/analyser/CLAUDE.md` rewritten at intent level: the rule
engine's whole section goes, replaced by the model + decoder story. The two grep
proofs. PR body carries the census table, B1/B2 and how they were resolved, the
crispness re-rank, the realtime numbers, the new anchors, the dependency diff, the
one-way door on mel regeneration, and the follow-ups.

**Total ≈ 88 agent-hours** across 18 tasks, plus per-task review loops and a
whole-branch review before the PR (the madmom migration's final round found a blocker
that every scoped review had structurally missed — #108).

---

## Follow-ups this PR deliberately does not do

1. **Sweep the floors ladder below ×0.75.** Crispness is monotone in it across the
   whole measured range and the range simply stops at 0.75. Val-only, zero GPU.
2. **Export MERT to ONNX** and drop `torch`/`transformers` from the live path. Would
   shrink the install by ~2.5 GB and remove the last training-shaped dependency from
   the show; unmeasured, so it ships after the model does.
3. **The downbeat head** (PR #9), if Task 1b says the beat-derived grid costs too
   much. Training work, parked by #141(c).
4. **The look-ahead's *modelling* consequences.** B1 is resolved here as a plumbing
   and configuration question. Whether a different geometry would buy a shorter chain
   is a training question and stays parked.
5. **Corpus report-cache regeneration.** `pipeline_sha` invalidates it by
   construction the moment the report schema loses four columns. Expected, priced in
   the PR, run once after merge.
6. **`training/measure_realtime.py`'s aubio comparison rows** — they document a
   front-end that no longer exists; the harness survives, the rows go with Task 4.
