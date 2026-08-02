# Analyser: the live stages between audio and a decision

Deep-dive on the analysis pipeline. The main `CLAUDE.md` covers what each intent
means, how the delay budget is put together and how to run the system; this
document covers *why* each stage measures what it measures and behaves the way it
does when it cannot.

What lives here:

| file | what it turns into what |
|---|---|
| `madmom_rhythm.py` | 256-sample buffers -> beat instants and a tempo |
| `mert_stream.py` | 44.1 kHz audio -> pooled label cells, one encoder pass per hop |
| `section_model.py` | one cell -> one class posterior and one boundary score |
| `gpu_stage.py` | the two above, on their own thread, and what the show does when they stop |
| `drift_watchdog.py` | pacing and stage health -> one shed level |
| `music_analyser.py` | the per-buffer loop that drives the rhythm stage, the RMS window and the silence gate |

The decoder that consumes all of this lives in `lib/engine/section_decoder.py`,
and the wiring that builds the chain out of the shipped artifacts lives in
`lib/section_chain.py`. Nothing above a module knows the geometry below it -- the
same principle `madmom_rhythm.py` established for its own framing mismatch.

---

## Rhythm: madmom, online only

**madmom owns rhythm** -- beats and the tempo derived from them -- through its
*online* processors only. The offline decoders score better and cannot run live,
so any number one of them produced would be a number the runtime can never
reproduce. Causality is not a claim here but a tested property: no network in the
live path may contain a bidirectional layer, because a bidirectional layer cannot
emit anything until the audio has ended. (The section model *is* bidirectional,
and that is not a contradiction: it is bidirectional over a bounded window of
audio the engine has already received, which is what the delay budget buys.)

**The two live on different clocks, and the adapter reconciles them.** The
pipeline reads 256-sample buffers; madmom's online models are trained at
441-sample hops. One module owns that mismatch, and it stamps every rhythm event
from its own hop counter rather than from a decoder's internal frame count.

**The onset stage is gone.** It existed only to produce the rule classifier's
density features, and it was the most expensive optional thing in the pipeline.
Deleting it returned roughly 11 % of a core, and it took the whole
shed/restore/epoch protocol with it -- the ladder's second rung had no tenant
left.

**BPM is derived from the beat stream, not read off a tempo estimator**: the
median of the recent inter-beat intervals, measured in the beat detector's own
stream time. The median because a single mistracked beat must not move the
reported tempo, and stream time because if the input ever drops audio the stream
clock still measures the music the detector actually heard. Until two intervals
exist it reports *unmeasured*, and the OS2L publisher holds the last real value
rather than putting a warm-up zero on the wire, which a consumer reads as a tempo
of zero rather than as "not known yet".

**Reported BPM is octave-folded**, and the reason has changed. It was introduced
because the retired tracker locked onto double tempo during warm-up, reliably
enough to fire a false PEAK before the music started, and it was load-bearing for
a tempo-gated DROP branch that no longer exists. It stays because a folded tempo
is what the OS2L wire has always carried and the ambiguity it addresses is a
property of tempo rather than of any one tracker.

**What the octave choice costs now is different, and larger.** The bar grid is
counted off this beat stream, so a *tracker* octave flip does not merely mislabel
a number -- it halves or doubles the bar length the committer quantises to. A
live run on a drum & bass track put a 30-second passage on the half-time grid and
moved a committed DROP by nearly one decoder bar against the same track's
simulation. Fast genres are the exposed case, as they were before, for a
different reason than before.

---

## RMS, and the silence gate that had to be re-based

RMS is a mean amplitude over a short rolling window, computed per buffer. It is
in every beat record, it is one of two continuous columns the training table
still carries, and since the demolition it has one more job: **it is the silence
gate**, which is the trigger for the entire sound-start/sound-stop machinery --
every state reset, the beat-absence timer, the OS2L song boundary, and the
decoder's and extractor's own resets.

The retired gate asked whether *every one of 40 mel bands* sat under a floor.
That is a statement about the spectrum, and it does not have an RMS equivalent:

- The threshold was chosen by replaying the sound-start/stop state machine over
  the recorded per-buffer RMS of three fixture tracks and sweeping a wide grid,
  not by picking a plausible level. The replay is *checked* rather than asserted:
  driven by the recorded mel decisions it reproduces the committed instants
  exactly on all three.
- **The binding pair is a contradiction, not a near miss.** One track needs its
  run-out called silent at a buffer RMS an order of magnitude above the level at
  which another track needs its fade-out still called sound. No monotone reading
  of a waveform RMS separates them -- not the instantaneous value, not mean, rms,
  max or min pooled over any of three window lengths -- because a broadband noise
  floor satisfies an all-bands-quiet test at a level a tonal reverb tail does
  not.
- **So one instant moved, deliberately, and it is recorded rather than
  absorbed.** The chosen threshold sits in the middle of the flat part of the
  error curve; the cost is a single sound-stop landing ~168 ms late, inside a
  run-out whose audio has already decayed past -70 dBFS and whose digital silence
  begins at the same moment either way. The alternative was holding that instant
  exactly and paying with another track losing nearly two seconds of its outro to
  a *fabricated* song boundary. An inaudible lateness is the smaller lie.

The sweep that produced this -- nine readings of the RMS against a 240-point log
grid, and the feasible band each fixture track admits -- is a session artifact
rather than a committed one, so what survives of it is the shape recorded above
and the constant in `music_analyser.py`, whose comment names the same reasoning.

---

## The MERT feature stage

`mert_stream.py` is the port of an offline whole-track extractor into the
geometry a live one can actually run. Pass `k` encodes the ring buffer ending at
`k * hop` and emits exactly the frames whose centre lies in
`[previous_hi, T - F)`, so an emitted frame sees between `F` and `F + hop`
seconds of future audio and a bounded amount of past. That future dependence is
the first half of the delay budget, and it is read off the shipped artifact, not
retyped.

Four things differ from the offline extractor, all because that one knows the
track length up front and a show does not:

- **The schedule is re-derived incrementally**, and the live driver is *asserted*
  against the offline generator over whole spans rather than trusted to agree
  with it at pass boundaries.
- **Cells are emitted as they complete**, not pooled into a track-wide array at
  the end. A cell is complete once no later pass can reach it. Gaps forward-fill
  from the last cell reached, because a zero row is not "no information" to a
  network -- it is a confident out-of-distribution input.
- **A track ending exactly on a hop boundary costs one extra encoder pass.** The
  driver has already run that pass as a regular one before it learns the stream
  stopped, so the flush re-encodes the same buffer to emit the residual margin.
  The emitted cells are identical either way.
- **The flush tail ends at the last cell an encoder frame reached.** Mid-stream
  cells are bit-identical with the offline sidecar; a flushed track can end a
  cell short or long of it.

**Cells are stamped at the END of their span**, which is the convention every
offline artifact uses. Every consumer computes from that one contract.

**The resample is part of train==deploy and is measured, not assumed** (D4). The
offline features were extracted through ffmpeg's resampler; a live polyphase
resample is a different filter, and the model has never seen its output. Nothing
fails loudly if this is wrong -- the posteriors just get quietly worse. The
streaming resampler is exact against a whole-array `resample_poly`, so the only
question left open is polyphase-versus-ffmpeg, and that is measured against a
committed track rather than argued about.

**The ring is bounded and an overrun is a typed shed event with a way back**, not
an exception the audio thread has to survive. The ring publishes a write *before*
it moves a sample, so a reader can never see a partially advanced buffer.

---

## The online student

The model is bidirectional over a bounded window: cell `t` is decided from a ring
holding a fixed span either side of it, plus the forward recurrence's carried
state, which is the whole past of the song at no extra cost. So **the live path
holds two things and nothing else**: a ring of feature cells and one state
tensor.

- **The ring is primed from the corpus mean, never from zeros** (D10). Zero raw
  features are a confident out-of-distribution input after the model's own input
  affine; the corpus mean is the one row it reads as no information, and it is
  what the offline whole-track pass pads its edges with.
- **The session is pinned single-threaded, and that is a determinism contract**
  rather than a performance choice: a threaded reduction sums in whatever order
  the pool finishes in, and float addition is not associative. Throughput, when
  it is wanted, comes from running tracks in parallel over separate sessions.
- **The graph is verified against its recorded sha at construction, and so is its
  geometry** -- against the shapes the graph itself declares and against its own
  internal arithmetic, because the sha covers the `.onnx` bytes and an edited
  sidecar is a wrong-geometry model that passes every hash it is asked for.
  Verified at construction and not at the first beat: a show that discovers its
  model is the wrong one halfway through a set has already played the wrong
  lights.
- **Nothing here is thread-safe and nothing here needs to be.** The ring, the
  carried state and the cell counter are owned end to end by whichever thread
  consumes cells. `reset` is not an exception -- a song boundary arriving on
  another thread has to be marshalled onto the owner.
- **A posterior is stamped by its cell, not by how many arrived.** Deriving the
  stamp from a push counter cannot know a gap happened, so after one shed every
  posterior would be early by the length of the shed -- and the decoder joins
  cells to bars *by time*, so the show would have read the wrong part of the
  track with nothing to say so.

---

## The GPU stage, and what the show does when it stops

The argument is arithmetic and it was made before any of this was built. One
encoder pass measured ~81 ms, and ~210 ms at p95 under GPU contention, against a
5.805 ms buffer period -- and the audio input **drops rather than queues**. Run
inline, the show would throw away roughly fourteen buffers of audio once a
second, in periodic gouges rather than as smooth lag: precisely the failure the
drift watchdog was built to notice and cannot fix. That is finding **B3**, and it
is why the pass and the student step that follows it moved off the audio loop.
(The pass measures slower than that on this box in a real-time run -- mean and
p99 both well above the design figure, and still comfortably inside the hop --
which strengthens the argument rather than weakening it. Both numbers are one
machine.)

    audio thread   resample -> ring write -> a monotonic sample index, and the
                   drain of whatever the GPU thread has finished
    GPU thread     one pass per hop, the student step per cell, whole passes
                   handed off through a bounded queue
    consumer       the audio thread again -- the decoder feed and the engine
                   commit stay where the queue, the MIDI client and the event
                   buffer already live, so no show state crosses a thread

**Overflow of the hand-off queue is a shed event, never a stall.** The audio
thread's contract is that it cannot be made to wait for the GPU under any
condition, including the GPU being dead.

**A shed keeps feeding the ring**, which looks like waste and is the opposite.
The extractor's sample index *is* song time and is what every cell is stamped
from, so a stage that stopped taking audio would come back with a clock that
disagrees with the beat grid it is decoded against -- silently, and for the rest
of the song. Resampling a buffer and writing it costs microseconds; what a shed
is about is not spending 81 ms on a GPU that cannot do it.

**Both edges of a gap clear.** Entering a shed drops the hand-off queue and tells
the consumer to reset the decoder; leaving one resyncs the extractor past the gap
and starts the student cold, because its window and its carried state describe
audio from before it. This is the lesson the deleted onset chain already taught,
applied to a more expensive stage.

**Degradation holds; it does not guess.** There is no second classifier and none
is wanted. The show runs on beats and the silence timer for as long as the GPU
stays away, the intent is held, the logging is rate-limited so a persistent fault
does not become its own outage, and reinit is attempted on a backoff that tops
out at one attempt per half minute.

Two things the fault drills found that are worth carrying:

- **Three of the four named failure modes are one mechanism reached by different
  exceptions.** A raised CUDA fault, an out-of-memory and a dead context cannot
  be told apart without reading the message, and a policy that branched on
  message text would be a policy about strings. The fourth -- a hung pass --
  raises nothing at all and leaves nothing on the GPU thread to notice it, so the
  audio thread times it out.
- **A restore must not reset the retry counter.** It did, once, and a permanently
  dead GPU then restarted from the immediate rung every time: fault, retry at
  once, fault, retry at once, forever -- an uncapped retry loop wearing a capped
  backoff's clothes, taking the fault log with it. A restore is the stage being
  let through to *try*; only a pass that returned resets the backoff.

---

## The shed ladder

`ShedLevel` is `NONE | NN_SHED`. The ladder used to have two rungs and the
integration deleted both tenants; what is left is the one expensive optional
stage, and `NN_SHED` is not a smaller show -- it is the degradation contract
above.

**Two inputs, one door.** Drift measures lost lead against a hardware-paced
input, which is the only thing that can see the loop failing to keep up. It is
structurally blind to the stage failing on its own: a CUDA fault, a driver reset,
a sleep/resume context loss or a hung pass all cost the audio loop exactly
nothing, so pacing stays perfect while the show holds one intent forever. The
stage reports those itself, and **either input alone holds the door shut** --
clearing one is not clearing both. The two arrive on different threads and each
writes only its own half; the derived level is settled under a lock so a
transition cannot be logged twice or lost.

Three details that each cost something before they were right:

- **The exit threshold is positive, not negative.** A hardware-paced input hands
  over exactly one buffer per buffer period, so the loop can never consume audio
  faster than it arrives and drift can never go negative however much headroom
  there is. Requiring negative drift to recover latches the watchdog for the
  whole show.
- **A flapping stage crosses the door twice per pass**, so the transition log is
  rate-limited per direction -- and what is suppressed is *counted* and carried
  into the next line, because a quiet log and a healthy rig have to stay
  distinguishable.
- **A stall the show chose is not lost lead.** The MIDI client blocks briefly at
  a song boundary to let the rig settle. That stall used to run inside the
  analyser, which forgave it; making the boundary room-aligned moved it into the
  command drain, where nothing did -- and it is over the watchdog's door, so
  every track change shed the GPU stage for about ten seconds and then re-warmed
  the decoder for a whole chain latency. All night, and invisible to a virtual
  clock, which does not advance while a real thread sleeps. The forgiveness
  follows the stall to where it actually runs, and is **bounded by the settle
  itself** -- forgiving the whole bracket would write off genuine lost lead at
  exactly the moment the loop is most likely to be in trouble.

---

## The per-stream delay model

Three streams leave this pipeline at three different ages, and the engine treats
them as three:

| stream | age when the engine sees it |
|---|---|
| beats | essentially the audio's own age |
| boundary scores (per cell) | one feature latency |
| bar decisions | one feature latency plus `(lag + 1)` bars |

The engine's queue therefore holds each command for the playback delay minus
*that statement's* measured age, rather than holding everything for one constant.
The consequences and the arithmetic are in the root `CLAUDE.md` ("The delay
model"); what belongs here is why this file cannot paper over it: **a bar is
formed long before it can be observed.** Beats arrive as the audio does and build
the grid; the cells a bar is decoded from arrive about eight seconds later.
`section_decoder.py`'s whole job is keeping those two facts apart, and every
bounded structure in it (the retained bar edges, the pending-cell window, the
trellis backtrace) exists so that nothing grows with the length of a set.

---

## Evaluation strategy

### Running a simulation

```bash
python auto_pilot simulate file path/to/song.mp3 --report report.json
```

Fast headless is the default: the full track runs through the identical
production pipeline on a virtual clock, deterministically. Beat timestamps are
song-position seconds; intent blocks are stamped at audience time and **carry
`song_t`, the instant of audio they describe** -- read that rather than
subtracting a constant, because the delay is per command now.

The report contains the beat list, the intent timeline and the timing log. It is
no longer a labelled feature table: the four rule-engine feature columns went
with the chains that produced them, and what a beat record carries is BPM and
RMS. The model's own features are not in the report at all -- they are the cells,
and those are cached beside the audio.

Inspect:

- **`intent_distribution_sec`** -- time spent in each intent. Does it match the
  track structure?
- **`intent_changes_count` / `effect_changes_count`** -- the first is the show's
  state changes, the second additionally counts boundary-triggered refreshes
  *inside* a held intent. They are no longer "consensus vs. what the lights did":
  every decision that reaches the engine is committed, so a gap between the two
  is refreshes, not vetoes.
- **`dominant_intent`** -- should reflect the character of the track.
- **`timing_log`** -- accuracy is per entry against *its own* target now, because
  the streams wait different amounts. A mean pooled across labels is a number no
  command ever targeted, so the report breaks it out per label.

### Reading a report

```bash
python training/inspect_report.py report.json
```

Prints per-10-second bins (mean RMS, beat count, dominant intent), the intent
timeline with block durations, and the intent distribution. This is still the
tool for eyeballing whether a track's show follows its structure -- but the
feature columns it used to print are gone, and with them the workflow of nudging
a threshold until they line up.

### There are no thresholds to tune here any more

The old loop was: read the feature bins, fix the measurement if a column reads
flat across sections that sound nothing alike, otherwise move a constant in
`light_engine.py` and re-run. That whole loop is retired. What replaces it:

1. **The decoder config is the tuning surface**, it is swept rather than nudged,
   and the shipped one is committed as a file rather than synthesised at runtime.
   A config carrying a key the decoder does not know **raises** -- it used to be
   silently dropped, which would have shipped a decoder nobody chose.
2. **`training/run_eval_set.py` is the gate**, and a deliberate change is
   expected to fail it. Read the printed table, then re-cut the baseline in the
   same commit.
3. **A new threshold anywhere in the show is measured against something, never
   picked.** The silence gate was fitted to committed sound-boundary instants;
   the effect-refresh threshold was fitted to a rate
   (`training/nn_boundary_refresh_rate.py`). Where a rate could not be recovered
   -- the retired YAMNet refresh, which simulation stubbed out from the day fast
   simulation landed and which therefore never appeared in a report, a fixture or
   a training table -- that is stated rather than reconstructed.

### Scoring against expert labels

`training/evaluate_against_labels.py` scores the committed intent timeline
against the Raveform section labels across the whole corpus, and
`training/run_eval_set.py` does the same on the frozen ten. See the root
`CLAUDE.md` for the metric design decisions and for what the current benchmark
says. Two of its findings bear on this pipeline rather than on the harness, and
both have moved:

- **ATMOSPHERIC is no longer dead code on this material.** It used to be the one
  intent not driven by classification, and mastered EDM intros and outros have
  beats, so its timer never tripped and every `intro`/`outro` label was out of
  reach. `intro` and `outro` are decoder classes now.
- **The show changes state far less often than it did**, and the benchmark
  measures that directly. The old stability pipeline stopped the timeline
  flickering *within* a bar without making it follow structure; the decoder's
  fitted duration model does both, and it pays for it in a way named under Known
  Limitations.

---

## Known Limitations

- **There is no live downbeat tracker, and the bar grid is counted.** Bars are
  four beats from the first detected beat. Measured on the production beat
  stream that costs about 0.14 crispness@0.5 s against an expert grid, and **all
  of it is placement** -- the class decisions are nearly grid-invariant; they
  land at a displaced instant. The cause is phase slips (a median of two per
  track), not beat timing, whose correlation with the damage is approximately
  zero. This is a phase-*tracking* problem, not a one-shot phase *decision*: an
  oracle frozen phase covers only about two thirds of a track, and a
  boundary-logit phase vote lost to plain counting on every configuration tried.
  An offline downbeat head and bar-phase decoder do exist in `training/nn/`,
  unwired and parked: their v1 scoring ran on the aubio beat stream and was
  removed when madmom replaced it (owner decisions #81/#133), so
  `training/nn/CLAUDE.md` carries the removal note and `docs/migration-evidence.md`
  the successor numbers.
- **Section-boundary latency is now a chain, not a vote window.** A committed
  intent trails the audio it describes by the feature latency plus the
  committer's lag -- around 13.7 s at the corpus median bar, and proportional to
  bar length. The playback delay is what makes that invisible to the room, and on
  a slow enough track it does not fully cover it.
- **Stability is bought with accuracy, and the cost lands on fast-alternating
  tracks.** Committing few, long, confident runs is what produces the flicker
  win; on a track whose structure alternates faster than the fitted duration
  prior expects, the same property lets one run swallow several real sections. A
  twitchy classifier collects partial credit for passing through the right state;
  a committed one does not. This is priced rather than accidental -- the decoder
  can trade the over-commitment back and pays macro-F1 for it -- so removing the
  cost without paying elsewhere is acoustic-model work, not decoder work.
- **One sound-stop instant cannot be reproduced by any RMS reading**, as
  described above. It is late by an inaudible margin on one fixture track, and
  reverting it is a one-line change if the ruling ever goes the other way.
- **The cold start has no evidence and is covered by a floor rather than by a
  measurement.** Nothing can be committed until the grid has bar lines and the
  committer's lag has filled. The engine's floor margin is chosen, not measured:
  bounded on one side by not firing on any fixture track, and on the other by the
  fact that being wrong costs one extra effect change against a dark stage.
- **Every runtime number on this page is one rig.** The CPU, GPU and soak figures
  were measured on a single machine, and the rhythm stage alone measures 2.5x
  apart between days on identical code -- so they are anchored within a session
  and must not be subtracted across files.

## Future Work

- **Continuous bar tracking.** The single largest lever on the show's crispness,
  and the one thing the integration shipped a priced fallback for. 57 of 215
  validation tracks already need nothing; the rest need a tracker that
  re-anchors after a slip, which nothing on the measured estimator surface does.
- **Export MERT to ONNX**, dropping `torch` and `transformers` from the live
  path. Removes the last training-shaped dependency from the show and a large
  amount of install; unmeasured, so it follows the model rather than leading it.
- **Read the boundary head per bar with a tie-free tolerance.** It beats chance
  per bar with strongly track-correlated errors, and the decoder's tolerance is
  the wrong window at the corpus's beat period -- it ties a third of all bars.
  That is a better use of the head than the track-level vote that was refuted.
- **Re-measure the corpus-wide label-aligned baseline against the neural show.**
  `baseline_eval.json` still describes the rule engine.
