"""Tests for the frozen eval-set selector (training/select_eval_set.py).

The eval set is the benchmark the simulation is judged against and the set the
neural classifier is forbidden to train on, so the selector's two load-bearing
promises are worth pinning: it is **deterministic** (same corpus -> same ten
tracks, or the benchmark silently moves under the baseline file) and it
**spans the tempo range** rather than the corpus histogram, which is massed so
hard at 124-130 BPM that a naive sampler would certify four-to-the-floor techno
and nothing else.
"""
import json
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import select_eval_set  # noqa: E402
from select_eval_set import (  # noqa: E402  (needs the path insert above)
    EVAL_SET_FILE,
    GAP_BUCKET_BPM,
    MAX_DURATION_SEC,
    MIN_BOUNDARIES,
    MIN_DURATION_SEC,
    Candidate,
    artist_of,
    beat_grid_bpm,
    build_candidates,
    equal_width_bins,
    family_of,
    gap_bucket,
    is_eligible,
    load_eval_set,
    rationale_line,
    select,
    tiebreak,
    verify_inputs,
)
from lib.label_space import SECTION_LABELS  # noqa: E402


def candidate(track_id="0001.aaaaaaaaaaa", youtube="aaaaaaaaaaa", bpm=126.0,
              duration=300.0, boundaries=9, classes=("intro", "buildup",
                                                     "breakdown", "drop", "outro"),
              artist="artist", genre="Techno") -> Candidate:
    return Candidate(
        track_id=track_id, youtube_id=youtube, duration_sec=duration, bpm=bpm,
        boundaries=boundaries, classes=frozenset(classes), genre=genre,
        title=f"{artist} - Track", artist=artist, family=family_of(track_id),
    )


def spread(count, bpms, **kwargs) -> list:
    """``count`` candidates with the given tempos, all otherwise identical."""
    return [
        candidate(track_id=f"{index:04d}.id{index:09d}", youtube=f"id{index:09d}",
                  bpm=bpms[index % len(bpms)], artist=f"artist{index}", **kwargs)
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
# structural facts: the raw vocabulary, no fold of the selector's own
# --------------------------------------------------------------------------- #


def test_the_selector_owns_no_fold_of_its_own_any_more():
    """The second merge existed only to undo the retired v1 fold."""
    for name in ("v1_runs", "V1_CLASSES", "V1_ORDER", "label_v1", "canonical_runs"):
        assert not hasattr(select_eval_set, name), name


def corpus_with(tmp_path: Path, sections: list, track_id="0001.aaaaaaaaaaa",
                bpm=120.0, duration=300.0):
    beats = tmp_path / "annotations" / "beats"
    beats.mkdir(parents=True, exist_ok=True)
    step = 60.0 / bpm
    lines = ["time"] + [f"{index * step:.6f}" for index in range(32)]
    (beats / f"{track_id}.beat.csv").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    rows = [{"track_id": track_id, "youtube_id": track_id.split(".", 1)[1],
             "decoded_duration_sec": str(duration)}]
    tracks = [{"key": track_id, "title": "Someone - A Track", "genre": "Techno",
               "sections": [{"start": start, "end": end, "name": name}
                            for start, end, name in sections]}]
    return build_candidates(tmp_path, rows, tracks)


def test_boundaries_are_counted_in_the_raw_vocabulary_without_folding(tmp_path):
    """breakdown|cooldown was one v1 run; in the raw space it is two sections
    and the boundary between them is one the evaluator can now see."""
    picked = corpus_with(tmp_path, [(0.0, 10.0, "breakdown"),
                                    (10.0, 20.0, "cooldown"),
                                    (20.0, 30.0, "drop")])
    assert [c.boundaries for c in picked] == [2]
    assert picked[0].classes == frozenset({"breakdown", "cooldown", "drop"})


def test_the_alt_classes_are_kept_apart_from_the_ones_they_used_to_fold_into(tmp_path):
    picked = corpus_with(tmp_path, [(0.0, 5.0, "outro"), (5.0, 9.0, "altoutro")])
    assert picked[0].classes == frozenset({"outro", "altoutro"})
    assert picked[0].boundaries == 1


def test_the_end_sentinel_is_still_dropped_rather_than_counted(tmp_path):
    picked = corpus_with(tmp_path, [(0.0, 5.0, "drop"), (5.0, 9.0, "end")])
    assert picked[0].classes == frozenset({"drop"})
    assert picked[0].boundaries == 0


def test_adjacent_sections_of_one_label_are_still_one_run(tmp_path):
    picked = corpus_with(tmp_path, [(0.0, 5.0, "drop"), (5.0, 9.0, "drop"),
                                    (9.0, 14.0, "outro")])
    assert picked[0].boundaries == 1


# --------------------------------------------------------------------------- #
# beat_grid_bpm
# --------------------------------------------------------------------------- #


def test_beat_grid_bpm_is_the_median_inter_beat_interval():
    assert beat_grid_bpm([index * 0.5 for index in range(16)]) == 120.0


def test_beat_grid_bpm_ignores_a_single_outlier_interval():
    # One dropped beat must not drag the tempo: the median is immune, a mean
    # would not be.
    times = [index * 0.5 for index in range(16)]
    times[8] = times[7] + 4.0
    assert round(beat_grid_bpm(times), 6) == 120.0


def test_beat_grid_bpm_rejects_a_grid_too_short_to_trust():
    assert beat_grid_bpm([0.0, 0.5, 1.0]) is None


def test_beat_grid_bpm_rejects_a_grid_with_no_forward_motion():
    assert beat_grid_bpm([1.0] * 20) is None


# --------------------------------------------------------------------------- #
# artist_of / family_of
# --------------------------------------------------------------------------- #


def test_artist_of_a_plain_title():
    assert artist_of("Jack Master - Bang The Box (Slam Remix) [Soma]") == "jack master"


def test_artist_of_a_chart_tagged_title():
    assert artist_of("[069] Taras Van De Voorde - Chasing Winters [Suara]") == \
        "taras van de voorde"


def test_artist_of_an_editorially_prefixed_title():
    assert artist_of("Record Of The Week: Pryda - Lycka [Pryda]") == "pryda"


def test_artist_of_a_glyph_prefixed_title_matches_the_plain_form():
    # The corpus carries both shapes for Konflict; if these disagreed the
    # same-artist guard would let both into the eval set.
    assert artist_of("+ Konflict - Beckoning [Renegade Hardware]") == \
        artist_of("Konflict - Messiah [Renegade Hardware]")


def test_family_of_is_the_track_id_index_block():
    assert family_of("0834.NyEKXA7_6z0") == "08"


# --------------------------------------------------------------------------- #
# equal_width_bins
# --------------------------------------------------------------------------- #


def test_equal_width_bins_are_equal_width_and_cover_the_range():
    bands = equal_width_bins([100.0, 200.0], 4)
    assert bands == [(100.0, 125.0), (125.0, 150.0), (150.0, 175.0), (175.0, 200.0)]


def test_equal_width_bins_collapse_when_every_value_is_identical():
    assert equal_width_bins([126.0, 126.0], 5) == [(126.0, 126.0)]


# --------------------------------------------------------------------------- #
# is_eligible
# --------------------------------------------------------------------------- #


def test_is_eligible_accepts_the_duration_bounds_inclusively():
    assert is_eligible(candidate(duration=MIN_DURATION_SEC))
    assert is_eligible(candidate(duration=MAX_DURATION_SEC))


def test_is_eligible_rejects_a_track_outside_the_duration_window():
    assert not is_eligible(candidate(duration=MIN_DURATION_SEC - 0.1))
    assert not is_eligible(candidate(duration=MAX_DURATION_SEC + 0.1))


def test_is_eligible_rejects_a_structurally_thin_track():
    assert not is_eligible(candidate(boundaries=MIN_BOUNDARIES - 1))


def test_is_eligible_does_not_require_full_class_coverage():
    # Criterion 1 is "where possible" -- a rank, not a gate, so a sparse tempo
    # band still contributes a track.
    assert is_eligible(candidate(classes=("intro", "drop", "outro")))


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #


def test_select_is_deterministic():
    pool = spread(60, [118.0, 124.0, 126.0, 131.0, 140.0, 174.0])
    first = [pick.candidate.youtube_id for pick in select(pool, size=10, seed=7)]
    second = [pick.candidate.youtube_id for pick in select(list(reversed(pool)),
                                                           size=10, seed=7)]
    assert first == second


def test_select_depends_on_the_seed():
    pool = spread(60, [118.0, 124.0, 126.0, 131.0, 140.0, 174.0])
    assert [pick.candidate.youtube_id for pick in select(pool, size=10, seed=7)] != \
        [pick.candidate.youtube_id for pick in select(pool, size=10, seed=8)]


def test_select_returns_the_requested_size():
    assert len(select(spread(60, [118.0, 124.0, 126.0, 131.0, 140.0, 174.0]),
                      size=10, seed=1)) == 10


def test_select_never_takes_two_tracks_by_one_artist():
    pool = [
        candidate(track_id=f"{index:04d}.id{index:09d}", youtube=f"id{index:09d}",
                  bpm=120.0 + index, artist="the same guy")
        for index in range(30)
    ]
    assert len(select(pool, size=10, seed=1)) == 1


def test_select_spans_the_tempo_range_rather_than_its_mass():
    # 50 tracks at 126 BPM and one each at the extremes: an equal-count sampler
    # would return 126 ten times over.
    pool = spread(50, [126.0]) + [
        candidate(track_id="0900.slowaaaaaaa", youtube="slowaaaaaaa", bpm=90.0,
                  artist="slow"),
        candidate(track_id="0901.fastaaaaaaa", youtube="fastaaaaaaa", bpm=175.0,
                  artist="fast"),
    ]
    bpms = [pick.candidate.bpm for pick in select(pool, size=10, seed=1)]
    assert min(bpms) == 90.0 and max(bpms) == 175.0


def test_select_prefers_full_class_coverage_within_a_band():
    poor = candidate(track_id="0010.poorbbbbbbb", youtube="poorbbbbbbb", bpm=126.0,
                     classes=("intro", "drop"), artist="poor", boundaries=20)
    rich = candidate(track_id="0011.richbbbbbbb", youtube="richbbbbbbb", bpm=126.0,
                     artist="rich", boundaries=9)
    picked = [pick.candidate.youtube_id for pick in select([poor, rich], size=1, seed=1)]
    assert picked == ["richbbbbbbb"]


def test_select_skips_ineligible_tracks():
    assert select([candidate(duration=60.0), candidate(boundaries=1)],
                  size=5, seed=1) == []


def test_select_reuses_a_corpus_block_rather_than_losing_a_tempo_band():
    # Both fast tracks sit in block 04, which the slow pick already occupies.
    # Criterion 2 outranks criterion 5, so the band must still be represented.
    pool = [
        candidate(track_id="0400.slowaaaaaaa", youtube="slowaaaaaaa", bpm=120.0,
                  artist="slow"),
        candidate(track_id="0401.fastaaaaaaa", youtube="fastaaaaaaa", bpm=170.0,
                  artist="fast"),
    ]
    assert len(select(pool, size=2, seed=1)) == 2


def test_select_output_is_ordered_by_track_id():
    picks = select(spread(60, [118.0, 124.0, 126.0, 131.0, 140.0, 174.0]),
                   size=10, seed=1)
    assert [pick.candidate.track_id for pick in picks] == \
        sorted(pick.candidate.track_id for pick in picks)


# --------------------------------------------------------------------------- #
# tiebreak / rationale
# --------------------------------------------------------------------------- #


def test_tiebreak_is_stable_across_processes():
    # A literal, not a re-computation: this is the whole determinism guarantee,
    # and PYTHONHASHSEED would break a str.__hash__ based version of it.
    assert tiebreak(1, "abcdefghijk") == tiebreak(1, "abcdefghijk")
    assert tiebreak(1, "abcdefghijk") != tiebreak(2, "abcdefghijk")


def test_rationale_line_names_the_track_and_the_reason():
    from select_eval_set import Pick

    line = rationale_line(Pick(candidate(bpm=128.0, duration=420.0), "BPM band"))
    assert "0001.aaaaaaaaaaa" in line and "128.0 BPM" in line and "BPM band" in line


# --------------------------------------------------------------------------- #
# Fill pass: the block rule must not outrank tempo diversity
# --------------------------------------------------------------------------- #


def test_gap_bucket_treats_near_equal_tempo_gaps_as_tied():
    assert gap_bucket(2.0) == gap_bucket(2.0 + GAP_BUCKET_BPM / 2)
    assert gap_bucket(2.0) != gap_bucket(2.0 + GAP_BUCKET_BPM)


# Bands over [120, 170] at size=3 are [120, 136.7), [136.7, 153.3), [153.3, 170].
# Band 1 takes `near` (closest to its centre) and band 3 takes `far`, which
# occupies block 01.  Band 2 is empty, so exactly one tempo-gap fill runs, and
# it must choose between `anchor` (gap 4.0, block 01 ALREADY USED) and `mid`
# (gap 3.0, block 05 free) -- the shape the reviewer used to expose the bug.
_FILL_POOL = [
    ("0100.anchoraaaaa", "anchoraaaaa", 120.0, "anchor"),
    ("0100.faraaaaaaaa", "faraaaaaaaa", 170.0, "far"),
    ("0900.nearaaaaaaa", "nearaaaaaaa", 124.0, "near"),
    ("0500.midaaaaaaaa", "midaaaaaaaa", 121.0, "mid"),
]


def fill_pool() -> list:
    return [candidate(track_id=track_id, youtube=youtube, bpm=bpm, artist=artist)
            for track_id, youtube, bpm, artist in _FILL_POOL]


def test_fill_pass_prefers_the_wider_tempo_gap_over_a_free_corpus_block():
    # The regression the reviewer caught: applying the block preference to the
    # whole remaining pool makes it a HARD rule in the fill pass, because the
    # pool practically always holds some unused-block track.  Criterion 2
    # outranks criterion 5, so the wider gap must win even though its block is
    # already occupied.  The buggy version returned `mid` here.
    picked = {pick.candidate.youtube_id for pick in select(fill_pool(), size=3, seed=1)}
    assert picked == {"nearaaaaaaa", "faraaaaaaaa", "anchoraaaaa"}


def test_a_fill_pass_block_reuse_is_labelled_in_the_reason():
    reasons = {pick.candidate.youtube_id: pick.reason
               for pick in select(fill_pool(), size=3, seed=1)}
    assert "block reused" in reasons["anchoraaaaa"]
    assert "block reused" not in reasons["nearaaaaaaa"]


def test_class_coverage_breaks_a_near_tie_on_tempo_gap():
    # The band-2 pick is `rich` (144.3).  `poor` then sits 0.6 BPM from it and
    # `rich2` 0.5 -- WIDER and narrower, but inside one GAP_BUCKET_BPM of each
    # other, which is the whole point of the bucket: without it the fill pass
    # ranks on the raw float, `poor` wins outright on 0.1 BPM nobody can hear,
    # and criterion 1 (class coverage) never gets a vote.  With it the two are
    # tied on tempo and the 5-class track wins.  The gaps must stay unequal for
    # this to discriminate -- equal raw gaps pass either way.
    pool = [
        candidate(track_id="0100.anchoraaaaa", youtube="anchoraaaaa", bpm=120.0,
                  artist="anchor"),
        candidate(track_id="0200.faraaaaaaaa", youtube="faraaaaaaaa", bpm=170.0,
                  artist="far"),
        candidate(track_id="0300.pooraaaaaaa", youtube="pooraaaaaaa", bpm=144.9,
                  artist="poor", classes=("intro", "drop")),
        candidate(track_id="0400.richaaaaaaa", youtube="richaaaaaaa", bpm=144.3,
                  artist="rich"),
        candidate(track_id="0500.rich2aaaaaa", youtube="rich2aaaaaa", bpm=144.8,
                  artist="rich2"),
    ]
    picked = {pick.candidate.youtube_id for pick in select(pool, size=4, seed=1)}
    assert "rich2aaaaaa" in picked and "pooraaaaaaa" not in picked


# --------------------------------------------------------------------------- #
# The committed artifact
# --------------------------------------------------------------------------- #


def test_the_committed_eval_set_is_a_well_formed_frozen_set():
    document = load_eval_set(EVAL_SET_FILE)
    ids = document["youtube_ids"]
    assert len(ids) == len(set(ids)) == 10
    assert sorted(document["rationale"]) == sorted(ids)
    assert [track["youtube_id"] for track in document["tracks"]] == ids


def test_a_fresh_document_records_the_raw_vocabulary_it_was_selected_under(tmp_path):
    from select_eval_set import Pick, build_document

    (tmp_path / "annotations").mkdir()
    (tmp_path / "clean_manifest.csv").write_text("x", encoding="utf-8")
    (tmp_path / "annotations" / "segments.json").write_text("[]", encoding="utf-8")
    picks = [Pick(candidate(classes=("intro", "cooldown", "drop")), "BPM band")]
    document = build_document(picks, tmp_path, 1, 1, 1, seed=1)

    assert document["selected_from"]["criteria"]["classes"] == list(SECTION_LABELS)
    assert document["tracks"][0]["classes"] == ["intro", "drop", "cooldown"]
    assert "v1_classes" not in document["tracks"][0]
    assert "v1_boundaries" not in document["tracks"][0]


def test_load_eval_set_rejects_a_document_that_is_not_an_eval_set(tmp_path):
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps({"tracks": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="youtube_ids"):
        load_eval_set(path)


def test_load_eval_set_names_the_file_it_could_not_find(tmp_path):
    with pytest.raises(RuntimeError, match="select_eval_set"):
        load_eval_set(tmp_path / "absent.json")


def test_verify_inputs_reports_a_changed_selection_input(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "clean_manifest.csv").write_text("x", encoding="utf-8")
    (tmp_path / "annotations" / "segments.json").write_text("[]", encoding="utf-8")
    document = {"selected_from": {"inputs": {"clean_manifest.csv": "0" * 64,
                                             "segments.json": "0" * 64}}}
    drift = verify_inputs(document, tmp_path)
    assert len(drift) == 2 and all("changed since the freeze" in entry for entry in drift)


def test_verify_inputs_is_silent_when_nothing_moved(tmp_path):
    from select_eval_set import sha256_of

    (tmp_path / "annotations").mkdir()
    (tmp_path / "clean_manifest.csv").write_text("x", encoding="utf-8")
    (tmp_path / "annotations" / "segments.json").write_text("[]", encoding="utf-8")
    document = {"selected_from": {"inputs": {
        "clean_manifest.csv": sha256_of(tmp_path / "clean_manifest.csv"),
        "segments.json": sha256_of(tmp_path / "annotations" / "segments.json"),
    }}}
    assert verify_inputs(document, tmp_path) == []
