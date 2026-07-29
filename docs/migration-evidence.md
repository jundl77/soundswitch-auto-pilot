# What the madmom migration did, and why it was made

The reports that drove this decision — the beat-source investigation and the
decoded sweep — live under `.superpowers/`, which is **gitignored**. Nothing in
them is in the repository. This document and `training/migration_deltas.json`
are the committed record, so a future reader does not have to take a commit
message's word for either the basis or the effect.

## Why madmom

aubio's weakness on this material is **beat phase**, not tempo. The decoded
sweep measured downbeat F1 on the 215-track validation split, at the operating
point a light show actually needs — stability gated at a median of one phase
flip per track:

| condition | gate-faithful F1 | max F1 |
|---|---|---|
| aubio (shipped before) | 0.2701 | 0.5277 |
| **madmom, online** | **0.5013** | **0.5855** |
| expert beats (diagnostic bound) | 0.7066 | 0.7066 |

At the gate-faithful operating point madmom nearly doubles downbeat F1 and
closes 71 % of the distance to what annotator-grade beats achieve on the same
chain. Two caveats travel with those numbers: they are **validation-split only**
(no test read was taken), and only madmom's **online** tracker was admitted —
its offline path scores better and cannot run live, so a number from it is one
the runtime could never reproduce.

The "roughly doubled F1" shorthand used elsewhere refers to the gate-faithful
row. At max-F1 the gain is +0.058, which is real but far less dramatic; quoting
the larger number without saying which row it comes from would be misleading.

## What it changed in the show

Full per-track numbers, before and after, are in
`training/migration_deltas.json`. The shape of it:

- **Beat counts rose sharply on two of three tracks** (+56 %, +38 %), and did
  not move on the third. aubio was missing beats.
- **The clearest single result is a self-consistency check.** On `hzIFjGcOKbg`,
  aubio emitted 1.774 beats/s while its own reported tempo implied 1.428/s — a
  24 % disagreement between a tracker's beat stream and its own tempo estimate.
  madmom emits 2.774/s against an implied 2.857/s: 2.9 %. That is visible in our
  own report format, needs no annotation, and is the most direct evidence that
  the old stream was dropping beats it had already counted.
- **Onset density rose on every track**, by +15.0 %, +7.8 % and +37.1 %
  respectively — stated per track because the spread is the point; a range
  hides which track sits where.
- **Intent timelines moved a lot**, and not uniformly: two tracks became
  markedly more stable (41 → 19 and 23 → 16 intent changes), one apparently
  less so (28 → 38).

**That last number is an artefact of the denominator, not a stability
regression.** Intent changes are counted per track, but madmom finds ~38 % more
beats on `PNpXKsge4xM`, so the same show is being divided by a larger number.
Per *beat* — the honest denominator, since the classifier runs once per beat —
stability is **flat: 0.0531 before, 0.0521 after**. Three interacting non-bugs
account for the raw count: beat-absence handling at the outro, the beat-count
denominator itself, and a double-tempo lock on that track.

**Do not read the intent-timeline changes as an accuracy improvement.** Nothing
here is scored against labels. What is established is that the beat stream is
better (measured, above) and that the show consequently differs. Whether each
difference is an improvement is a question for the labelled evaluation, which
lives on another branch.

## What did not change

The **aubio mel filterbank is byte-identical**, pinned two ways: a golden hash
of the mel path generated from the pre-migration commit's own code, and a
whole-track fixed-time-grid fingerprint that no beat grid can perturb. Every
trained model depends on that bank; this migration does not touch it.

The **report schema is unchanged** on all three tracks.

## Cost

| | before | after |
|---|---|---|
| front-end per buffer (5.805 ms budget) | 0.079 ms | 1.493 ms |
| front-end share of one core | 1.4 % | 25.7 % |
| fast-sim throughput | 46.3× real time | 3.8× |
| corpus report-cache regeneration | ~3.4 CPU-h | ~41 CPU-h |

The corpus figure is derived from the 1387-track manifest's decoded durations
(156.6 audio-hours), not estimated. At 12 workers it is roughly 3.5 hours of
wall clock — track parallelism recovers the wall time; per-core throughput is
what fell.

Under a strict reading of the campaign's ≥ 5× / ≤ 20 %-of-a-core realtime bar,
25.7 % **fails**. That bar was written for neural posterior generation and was
ruled not to bind a DSP front-end; the front-end's gates are sustained 1×
whole-pipeline with headroom, the backpressure machinery, and a 30-minute soak,
all of which pass. Both readings are recorded because the miss is real under the
original wording.
