# Analyser: Audio Feature Extraction and Classification

Deep-dive on the analysis pipeline. The main `CLAUDE.md` covers what each intent means and how to run the system; this document covers *why* the features and classifiers are designed the way they are.

See `music_analyser.py` for all implementation details and `lib/engine/light_engine.py` for all tuning constants.

---

## Where the numbers come from: two libraries, one job each

The measured basis for this split, and its measured effect on the show, are in
`docs/migration-evidence.md` and `training/migration_deltas.json`.

**madmom owns rhythm** — beats, BPM and onsets — through its *online* processors
only. The offline decoders score better and cannot run live, so any number one of
them produced would be a number the runtime can never reproduce. Causality is not
a claim here but a tested property: no network in the live path may contain a
bidirectional layer, because a bidirectional layer cannot emit anything until the
audio has ended.

**aubio owns the spectral front-end** — one FFT and the 40-band Slaney mel bank —
and nothing else. Every trained model and every feature below that mentions the
filterbank is built on that exact bank, so it is held byte-stable by a golden
fixture rather than reimplemented. The two-library split is interim by owner
decision; the next retrain generation bakes off front-ends and may consolidate.

**The two live on different clocks, and the adapter reconciles them.** The
pipeline reads 256-sample buffers; madmom's online models are trained at
441-sample hops. One module owns that mismatch, and it stamps every rhythm event
from its own hop counter rather than from either decoder's internal frame count —
those counters advance only for frames their decoder was handed, so the moment
one chain is shed its clock stops while the other's runs on.

**Shedding degrades explicitly.** Under sustained backpressure the analyser gives
up section detection first and onsets second, and never beats. A shed onset
detector reports density as *unmeasured*, not as zero: zero is a measurement, and
since every branch of the classifier is a density comparison, a zero would not
degrade — it would classify, always the same way, for as long as the detector was
off. Restoring either shed component clears its buffers first, because everything
they hold describes audio from before the gap.

## Features and Why They Were Chosen

**BPM** — the primary tempo discriminator. It gates DROP: maximum-impact classification requires dance tempo. Low onset activity at any BPM → BREAKDOWN or ATMOSPHERIC.

BPM is *derived from the beat stream*, not read off a tempo estimator: it is the median of the recent inter-beat intervals, measured in the beat detector's own stream time. The median because a single mistracked beat must not move the reported tempo, and stream time because if the input ever drops audio the stream clock still measures the music the detector actually heard. Until two intervals exist it reports *unmeasured*, and the DROP branch gates on a tempo floor, so an unmeasured tempo reads as "not fast" rather than as a guess.

Reported BPM is *octave-folded* into a single tempo band before anyone sees it. A tempo and its octave are the same tempo musically, and the fold does not touch beat phase, only the number attached to it — so a half-tempo lock cannot make a busy passage read as a low-energy one. The cost is that genuinely fast genres (drum & bass above the fold ceiling) report at half tempo and fall below DROP's BPM floor — a known Stage 1 limitation, not a bug.

*Historical note worth keeping:* the fold was introduced because aubio locked onto double tempo during its warmup, reliably enough to fire a false PEAK before the music started. That specific failure is gone with the rhythm source, but the fold stays because the ambiguity it addresses is a property of tempo, not of any one tracker.

**Onset density** (onsets/sec, rolling window) — measures rhythmic busyness. A sparse arrangement has few onsets per second; a full drop with kick, bass, hi-hat, and percussion fires many per second.

Density is *unmeasured*, not zero, whenever the onset detector is shed under backpressure or has been restored for less than one full window — an empty rolling window would otherwise report a low rate rather than a missing one, which is the same error one second later. The sentinel is negative, because a rate cannot be, so no genuinely sparse passage can be mistaken for an unmeasured one; unlike the kick sentinel it *is* self-identifying in the training table. Consumers hold rather than classify on it.

Density is *quantized*: it is an onset count over a fixed window, so only multiples of one-over-the-window-length are reachable. Picking a density threshold means picking which bucket it falls between; two thresholds inside the same gap are the same threshold. It is also far less discriminating than it looks — on the reference track the windowed median sits in a single bucket across intro, groove, breakdown *and* final drop alike. Density says whether anything rhythmic is happening; it does not say how big the moment is. Kick strength does that.

**Onset density trend** (ratio of recent vs. past beats) — detects whether energy is rising. BUILDUP requires both sufficient density *and* a rising trend. A steady groove at the same density level stays in GROOVE even if the density is high. Needs a few beats to warm up before it carries signal.

**Sub-bass ratio** (mel filterbank bands 0–4 / total energy) — normalised fraction of energy in the bass register. Stored per-beat. A secondary signal; `kick_strength` (below) is more discriminating for DROP detection.

**Kick strength** — how far the sub-bass energy on the beat rises above the sub-bass floor around it. This is the feature that actually separates the sections of a track, and it is the primary DROP gate and the primary BREAKDOWN signal when density is moderate (stripped arrangement, no kick). Three measurement decisions make it work, each of which was wrong at some point and each of which flattened the feature to noise when it was:

- **The window straddles the beat, and mostly follows it.** A kick's energy does not appear in the mel filterbank at the instant the beat is reported: the filterbank runs over an FFT window several buffers long, so the sub-bass peak lands *after* the beat index by roughly that window's group delay. A backward-looking capture measures the bar before the kick and reports the background twice. The consequence is that a beat's own kick value is not final when the beat fires — it resolves a few buffers later, so the feature lags by one beat. Over a multi-second classification window that is immaterial, and it is worth far more than the lag costs.
- **The denominator is a median, not a mean.** A mean over recent frames includes the on-beat spikes it is supposed to be compared against, and on a track with a sustained rolling bassline it also includes that. The median reads the floor the kick sits on.
- **Numerator and denominator cover comparable spans.** Smoothing the beat side over many beats while the background side covers ~1 s makes the two lag each other, and at every section boundary the ratio reports the *transition* — a breakdown entry then reads as the strongest kick in the track. Keeping the smoothing short keeps the ratio a statement about the present.

Measured this way on the reference track, kicking sections and kick-free sections separate cleanly at the decile level — their medians sit far apart and the presence threshold falls between the kick-free ninth decile and the kicking first decile — but the extreme tails do cross it in both directions, a handful of beats each way. The separation is a property of the bulk of each population, not a guarantee about every beat, which is precisely why the classifier gates on a multi-beat window mean rather than on a single beat: averaging over the window pulls the crossing tails back to the correct side. Do not describe this feature as cleanly separable per beat, and do not build anything on a single beat's value.

Near-silence and no-measurement-yet both report a dedicated *unknown* value that is deliberately below any usable threshold: an unmeasured kick reads as an absent kick, so DROP is never entered without positive evidence. The classifier's own default for the parameter is that same sentinel, so a caller with no kick data cannot accidentally assert one. See `music_analyser.py` for the implementation.

Because the sentinel is a number in the same range as a real ratio, it is not self-identifying in the training table: the only rows that can carry it are those below the silence gate or before the first resolved measurement, so a Stage-2 consumer should derive a `kick_known` flag from each row's own RMS against that gate rather than testing the kick value against the sentinel — a genuine measurement can legitimately land on it.

**Spectral centroid** (mel-band index units, 0–39) — centre of mass of the frequency spectrum. Low = bass-heavy; high = treble-heavy. Tracked per-buffer and at beat timestamps. The *trend* of the centroid across recent beats is the key feature: a rising centroid (energy moving toward higher frequencies) is the defining signature of a BUILDUP riser or sweep filter. A falling centroid (energy concentrating downward) signals a DROP approach. The trend is computed the same way as onset density trend: recent beats vs. past beats. See `music_analyser.py` for details.

**RMS energy** — mean amplitude over a short rolling window. Stored in the beat record. Not yet used in classification directly, but available as a loudness proxy. Future use: PEAK confirmation (loud + high BPM).

**YAMNet embeddings** — 1024-dimensional audio embeddings (not tag predictions). Used to detect structural section changes via cosine similarity outlier detection across a rolling lookback. The cooldown constant in `yamnet_change_detector.py` controls how often section changes can fire.

---

## Classifier Design Decisions

### Why this priority order?

The classifier tests DROP first, then BUILDUP, then the two BREAKDOWN branches, and falls through to GROOVE. The order encodes which evidence outranks which:

- **DROP first** because it is the only branch that demands positive evidence rather than settling for the absence of it. In principle that evidence is two features, density *and* a beat-locked kick; in practice on the reference track the density gate is nearly a no-op — a clear majority of all beats in the track sit at or above the entry value, breakdown included — so the kick is doing essentially all of the work. Treat DROP as kick-gated with a density sanity check, not as a two-feature conjunction, until a wider corpus says otherwise. Nothing weaker should be able to pre-empt it.
- **BUILDUP before both BREAKDOWN branches** because a riser strips the kick and thins the arrangement *by design*. That is exactly the signature the kick-absence branch would swallow — it would relabel every build as a breakdown, which is the one moment the show must not go quiet. This ordering is a regression the tests pin explicitly.
- **BUILDUP's density floor is BREAKDOWN's ceiling.** Below the density at which we would call the arrangement a breakdown, a trend ratio is computed over one or two onsets and is noise. Leaving a gap between the two would open a sparse band where a noise trend fires BUILDUP and bypasses BREAKDOWN's hysteresis entirely.
- **GROOVE last**, as the default: beats are present, nothing else claimed them.

### Why does kick absence widen BREAKDOWN rather than force it?

With no kick, density is the only evidence left — and density is quantized coarsely enough that a single onset entering or leaving the rolling window can move it a whole bucket. So kick absence widens BREAKDOWN's band by a margin spanning more than one bucket, applied to the entry and exit thresholds alike. Widening both preserves the hysteresis dead zone; widening only one (the earlier design) removed it, and the result was a kick-less section flapping between BREAKDOWN and GROOVE every few seconds on ±1 onset.

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

### Why is PEAK not in the classifier either?

PEAK means "sustained maximum energy *after* the drop". The "after" is a temporal property, and a feature window has no way to express it: a peak section and the drop that preceded it look identical in density, kick, and centroid. Any threshold that separated them would either be unreachable (never firing) or would steal windows from DROP. So the classifier never returns PEAK — the engine promotes a *committed* DROP to PEAK once the dwell counter shows it has lasted, which is exactly the "sustained" part of the definition. While PEAK is current, DROP votes are absorbed, since easing from peak back to plain drop is not a show change and would otherwise let the two oscillate. Any other consensus exits PEAK through the normal stability pipeline. See `LightEngine._commit_intent`.

Because PEAK *is* the DROP musical state, two things follow. It inherits DROP's hysteresis: a density dip that DROP would ride out must not eject PEAK either, so the classifier applies DROP's exit threshold while PEAK is committed. And absorbed votes are never surfaced to the event buffer, so the reported intent timeline keeps reading PEAK for as long as the lights hold it — the timeline records committed show state, and an absorbed vote is not a change.

---

## Evaluation Strategy

### Running a simulation

```bash
python auto_pilot simulate file path/to/song.mp3 --report report.json
```

Fast headless mode is the default: the full track runs through the identical production pipeline on a virtual clock at a few times real time, deterministically — rerunning the same file yields an identical report, so threshold changes show up as clean diffs. Beat timestamps are song-position seconds; intent blocks are stamped at audience time (one look-ahead delay after the beats that caused them), so when comparing the intent timeline against the track structure, expect that constant delay. The report carries that offset in its metrics rather than leaving it implicit in the code, so a consumer can align the two time bases from the file alone — the inspector's bin table uses it to de-shift intents back to song time.

That constant offset is not quite uniform, and a consumer that subtracts it from every intent block will be wrong about some of them. Most commits go through the delayed command queue and are therefore stamped one look-ahead after the beat that caused them — audience time. But the commits the engine makes *immediately*, because a beat is its own confirmation — the first beat of a run, the first beat back after a sound stop, and beat-absence ATMOSPHERIC — write to the event buffer inside the callback, so those blocks are stamped in song time already. De-shifting one of them moves it a whole look-ahead too early, where it silently annexes the beats belonging to the intent before it. A consumer aligning the two time bases must decide per block, not per report: a block is a queue commit exactly when its timestamp minus the look-ahead lands on an actual beat. See `realign_intents` in `training/build_training_table.py` for the detection and its failure modes.

The JSON report contains the full beat list, intent timeline, and timing log. Every beat record carries the complete feature row the classifier saw — density, BPM, kick strength, centroid trend, sub-bass ratio, RMS — which makes the report a labelled feature table, not just a debug dump. It is the intended input for the hand-labelled dataset work; keep it that way when adding features.

Inspect:

- **`intent_distribution_sec`** — time spent in each intent. Does it match the track structure?
- **`intent_changes_count`** — should be in the tens for a 3-minute track. Much higher means flickering; much lower means stuck. Note it counts classifier *consensus* changes, which includes changes the dwell guard then blocked; `effect_changes_count` is what the lights actually did.
- **`dominant_intent`** — should reflect the character of the track.
- **`timing_error_max_ms`** — command queue accuracy. Should be well under 50 ms.

### Reading a report

```bash
python training/inspect_report.py report.json
```

Prints per-10-second bins (mean RMS, density, kick strength, beat count, dominant intent), the intent timeline with block durations, and the intent distribution. This is the tool for eyeballing whether a track's show follows its structure: the bins line up against the track's sections, and a feature column that reads flat across sections that sound nothing alike is a measurement bug, not a threshold that needs nudging. That distinction is the whole lesson of the Stage 1 calibration — every threshold on the branch was unreachable or inverted because a feature was being measured wrong, and no amount of threshold tuning would have fixed it.

### Tuning workflow

1. Run simulation on a track with a known structure (e.g. the drop starts at T=90 s).
2. Run `training/inspect_report.py` and check the *feature* columns first: do they separate sections that sound different? If not, fix the measurement before touching a threshold.
3. Inspect the intent timeline: does the DROP intent start and end where the drop does? Remember the constant look-ahead offset between the two.
4. Adjust thresholds in `lib/engine/light_engine.py` and re-run. The run is deterministic, so a threshold change shows up as a clean diff.
5. Once the basic structure is reliable, enable and tune the sub-bass gate against hi-hat-only vs. kick+bass passages.

### Scoring against expert labels

The workflow above tunes by ear against one track. `training/evaluate_against_labels.py` does the same job against the whole expertly annotated corpus: it scores the committed intent timeline against the Raveform section labels and prints a time-weighted confusion matrix, per-class F1, boundary-F1 at three tolerances, and a flicker rate. Current numbers live in `training/data/raveform/baseline_eval.json` (the corpus is still growing, so they are not copied into documentation); see the root `CLAUDE.md` for the metric design decisions. Two of its results bear directly on the classifier rather than on the harness:

- **ATMOSPHERIC is dead code on this material.** It is the only intent not driven by the beat classifier, and mastered EDM intros and outros have beats, so its timer never trips. Across the whole corpus it was committed zero times, which puts every `intro` and `outro` label permanently out of reach. (That sweep ran on the aubio beat stream. madmom finds more beats, not fewer, so the timer trips even less often — the finding survives the migration by construction, though the count has not been re-run.)
- **The engine changes intent several times more often than the music changes section**, and the large majority of those changes are nowhere near a real boundary. The stability pipeline (votes, dwell, invalid-transition guard) stops the timeline from flickering *within* a bar; it does not make the timeline follow structure. Note the evaluator reports that count two ways -- every intent change, and only the changes that alter the *label class* -- because a model predicting label classes cannot express a DROP-to-PEAK move and must be compared against the second.

### The regression gate

Scoring the whole corpus is the measurement; `training/run_eval_set.py` is the *gate*. It runs the ten frozen eval-set tracks (`training/eval_set.json`) through the same fast sim and the same join, and compares both the per-track report checksums and the same v1 metrics against a committed baseline. The consequence for threshold work: **a threshold change is expected to fail it.** That failure is the whole point — it prints which track moved, which metric moved, and in which direction, so a change made to fix one section of one track cannot quietly cost more elsewhere than it gained. Read the table, then re-cut the baseline in the same commit with `--write-baseline`.

The tracks span 117-174 BPM across five genres, which makes the gate the fastest available answer to the two limitations below: whether a threshold fitted to one track survives contact with another, and how much of the corpus sits above the tempo fold ceiling. Three of them run inside `uv run pytest`; the full ten are a manual command. See the root `CLAUDE.md` (The benchmark) for why the set is frozen and how it is kept out of training.

### The neural replacement exists offline

A trained section classifier (`training/nn/`) now scores better than everything on this page -- on tracks it has never seen, measured by the same functions that produce the baseline above. See `training/nn/CLAUDE.md` for the package map and the root `CLAUDE.md` for the verdict, its artifacts and its caveats. What it changes for this document:

- **The features here are still the runtime's features.** The model trains on the pipeline's own mel stream, so the front-end above is shared rather than superseded. What it replaces is the hand-thresholded branch and the three-stage stability pipeline: it predicts the label class directly, and a decoder owns stability and latency policy in one place instead of votes, dwell and a transition guard in three.
- **The two limitations recorded above are the model's whole motivation.** ATMOSPHERIC never fires on mastered EDM, which puts `intro` and `outro` permanently out of reach; and the engine changes intent far more often than the music changes section. The model answers both, and the second by a wide margin.
- **It does not run yet, and the blocker is not integration effort.** The decoder decides per bar and there is no live downbeat tracker; the show's look-ahead also has to grow to the decoder's budget. Until both exist, this classifier is what ships and its thresholds are still worth tuning.
- **Stability is a trade, not a free win -- do not import the model's preferences here.** The decoder's flicker advantage comes from committing long confident runs, and the same property lets one run swallow several sections on a track that alternates quickly. That is its known failure mode. The tuning workflow above optimises against the *opposite* failure, flapping, so a threshold change justified by "the NN commits harder" is unmeasured.

---

## Known Limitations

- **One reference track.** Every threshold currently in the engine was placed against the populations measured on a single track, which is no longer even in the repo — it was the Generate anchor, retired when the eval set replaced it. They separated its sections cleanly, but a threshold fitted to one track is a hypothesis, not a calibration. Re-measure the populations before trusting them on a new corpus — the measurement is cheap and deterministic, and the eval set is now the place to do it.
- **Tempo above the fold ceiling.** Fast genres (drum & bass and up) fold to half tempo and fall under DROP's BPM floor, so their drops cannot be classified as such. Accepted for Stage 1.
- **Section-boundary latency.** An intent change needs consensus across several beats, so a committed intent trails the audible transition by roughly the vote window — a second or so. That is the price of not flickering, and it is deliberate; the symmetric look-ahead window keeps the response centred on the transition rather than lagging a full window behind it.

## Future Work

- **Kick strength across a corpus**: on the reference track the presence threshold sits between the two populations' deciles, with a few tail beats crossing it. Measure whether that decile gap survives on other material — and how wide it really is — before treating the threshold as calibrated. Tune the kick-absence margin against passages where the kick drops out mid-groove.
- **Centroid trend calibration**: measure `get_spectral_centroid_trend()` during genuine buildup sections vs. steady grooves to validate `_CENTROID_BUILDUP_TREND`. The threshold is more reliable than sub-bass ratio but still needs real data.
- **RMS energy in classification**: use as a loudness confirmation for DROP (loud + high density = real drop; quiet + high density = busy but small arrangement). It could also gate the DROP → PEAK promotion so a fading drop is not promoted.
- **Spectral flux**: rate of change of the mel spectrum captures timbral shifts that onset density misses — useful for detecting timbral drops (e.g. a low-pass filter sweep releasing into the drop).
- **Labelled data**: once enough real-track simulations exist, label the intent timeline manually and use it to validate or calibrate thresholds systematically rather than by ear.
