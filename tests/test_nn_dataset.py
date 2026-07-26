"""Tests for the NN dataset builder (``training/nn/dataset.py``).

Two things are pinned here, because both are silent when they break.

**Splits.**  The frozen eval set is the benchmark the whole plan is judged
against; a single eval track leaking into ``train`` invalidates every number
that follows, and nothing downstream would notice.  The same is true one step
out: a producer's other tracks teach the net that producer's sound, so an
artist match against an eval track is contamination too.  And because the
corpus is still downloading, the assignment has to be a pure function of the
id -- adding tracks must never move a track that is already placed.

**Targets and masks.**  Everything the net learns comes from these arrays.  A
mask that is off by a frame trains the model on unannotated audio; a boundary
target at the wrong frame teaches the decoder to fire early.  Neither shows up
as an error -- only as a slightly worse model -- so they are asserted directly
against hand-built section lists.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from build_training_table import (  # noqa: E402  (needs the path insert above)
    V1_ORDER,
    write_feature_sidecar,
)
from nn.dataset import (  # noqa: E402
    BOUNDARY_MASK_RADIUS_SEC,
    BOUNDARY_SIGMA_SEC,
    FRAME_SEC,
    GAIN_JITTER_DB,
    IGNORE_INDEX,
    LABEL_FRAMES,
    LABEL_POOL,
    SPLITS_FILE,
    WINDOW_FRAMES,
    WINDOW_SEC,
    TrackRef,
    WindowDataset,
    artist_participants,
    assign_split,
    excluded_artist_names,
    make_splits,
    partition,
    track_targets,
)

CLASS_INDEX = {label: index for index, label in enumerate(V1_ORDER)}


# --------------------------------------------------------------------------- #
# Geometry constants
# --------------------------------------------------------------------------- #


def test_window_geometry_matches_the_spec():
    """~16 s of ~46 ms frames, poolable to the ~10 Hz label head."""
    assert FRAME_SEC == pytest.approx(8 * 256 / 44100)
    assert WINDOW_FRAMES == 348                       # 16 s / 46.4 ms, aligned up
    assert WINDOW_FRAMES * FRAME_SEC >= WINDOW_SEC    # covers the window, never short
    assert WINDOW_FRAMES % LABEL_POOL == 0
    assert LABEL_FRAMES == WINDOW_FRAMES // LABEL_POOL
    label_rate = 1.0 / (FRAME_SEC * LABEL_POOL)
    assert 9.0 <= label_rate <= 12.0                  # "~10 Hz" label head


# --------------------------------------------------------------------------- #
# Artist matching (collaboration-aware)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("title, expected", [
    ("Greg Downey Feat. Bo Bruce - Come To Me [Kearnage]",
     {"greg downey", "bo bruce"}),
    ("[79:26] DJ Misjah & DJ Tim - Access [X-Trax]", {"dj misjah", "dj tim"}),
    ("+ Konflict - Beckoning [Renegade Hardware]", {"konflict"}),
    ("[78] Oxia - Domino [Kompakt Extra]", {"oxia"}),
    ("Noisia vs. The Upbeats - Dead Limit [Vision]", {"noisia", "the upbeats"}),
    ("Sub Focus ft Culture Shock - Vibration [Ram]", {"sub focus", "culture shock"}),
    ("Andy C - Roll On [Ram]", {"andy c"}),
])
def test_artist_participants_splits_collaborations(title, expected):
    assert artist_participants(title) == expected


def test_solo_artist_matches_their_collaboration():
    """The contamination the plan names explicitly: a Feat. credit must catch the solo."""
    collaboration = artist_participants("Greg Downey Feat. Bo Bruce - Come To Me")
    solo = artist_participants("Greg Downey - Rewired [Discover]")
    assert collaboration & solo


def test_excluded_artist_names_unions_every_participant():
    names = excluded_artist_names([
        {"title": "Greg Downey Feat. Bo Bruce - Come To Me"},
        {"title": "Andy C - Roll On"},
    ])
    assert names == {"greg downey", "bo bruce", "andy c"}


# --------------------------------------------------------------------------- #
# Split assignment
# --------------------------------------------------------------------------- #


def test_assign_split_is_deterministic_and_seed_dependent():
    assert assign_split("hzIFjGcOKbg", 1337) == assign_split("hzIFjGcOKbg", 1337)
    ids = [f"id{index:08d}" for index in range(500)]
    with_seed_a = [assign_split(i, 1337) for i in ids]
    with_seed_b = [assign_split(i, 7331) for i in ids]
    assert with_seed_a != with_seed_b            # the seed actually moves tracks


def test_assign_split_hits_the_70_15_15_ratios():
    ids = [f"id{index:08d}" for index in range(4000)]
    counts = {"train": 0, "val": 0, "test": 0}
    for youtube_id in ids:
        counts[assign_split(youtube_id, 1337)] += 1
    assert counts["train"] / len(ids) == pytest.approx(0.70, abs=0.03)
    assert counts["val"] / len(ids) == pytest.approx(0.15, abs=0.03)
    assert counts["test"] / len(ids) == pytest.approx(0.15, abs=0.03)


def refs(count, start=0, title="Someone - Track"):
    return [
        TrackRef(track_id=f"{index:04d}.id{index:08d}",
                 youtube_id=f"id{index:08d}", title=title)
        for index in range(start, start + count)
    ]


def test_partition_extension_never_reshuffles_existing_assignments():
    """The 1,423-track retrain must extend this file, not regenerate it."""
    first = partition(refs(200), eval_ids=frozenset(), artist_names=frozenset(), seed=1337)
    grown = partition(refs(400), eval_ids=frozenset(), artist_names=frozenset(),
                      seed=1337, existing=first)

    for split in ("train", "val", "test"):
        assert set(first[split]) <= set(grown[split])
    placed_before = {i for split in ("train", "val", "test") for i in first[split]}
    placed_after = {i for split in ("train", "val", "test") for i in grown[split]}
    assert len(placed_after) == 400
    assert placed_before < placed_after


def test_partition_places_every_candidate_exactly_once():
    result = partition(refs(300), eval_ids=frozenset(), artist_names=frozenset(), seed=1337)
    everything = result["train"] + result["val"] + result["test"]
    assert len(everything) == len(set(everything)) == 300


def test_partition_output_is_sorted_for_a_stable_file():
    result = partition(refs(120), eval_ids=frozenset(), artist_names=frozenset(), seed=1337)
    for split in ("train", "val", "test"):
        assert result[split] == sorted(result[split])


def test_partition_excludes_eval_ids_and_artist_matches():
    candidates = [
        TrackRef("0001.evalaaaaaaa", "evalaaaaaaa", "Greg Downey Feat. Bo Bruce - Come To Me"),
        TrackRef("0002.soloaaaaaaa", "soloaaaaaaa", "Greg Downey - Rewired"),
        TrackRef("0003.featbbbbbbb", "featbbbbbbb", "Bo Bruce & Someone Else - Other"),
        TrackRef("0004.cleanaaaaaa", "cleanaaaaaa", "Unrelated Artist - Track"),
    ]
    eval_ids = frozenset({"evalaaaaaaa"})
    artist_names = excluded_artist_names([
        {"title": "Greg Downey Feat. Bo Bruce - Come To Me"},
    ])

    result = partition(candidates, eval_ids=eval_ids, artist_names=artist_names, seed=1337)

    placed = set(result["train"]) | set(result["val"]) | set(result["test"])
    assert placed == {"cleanaaaaaa"}
    assert result["excluded_eval_set"] == ["evalaaaaaaa"]
    assert set(result["excluded_artist"]) == {"soloaaaaaaa", "featbbbbbbb"}


def test_partition_prefers_the_frozen_file_over_the_hash():
    """The written file is the record: recomputation must not quietly override it."""
    ref = TrackRef("0001.aaaaaaaaaaa", "aaaaaaaaaaa", "A - B")
    natural = assign_split(ref.youtube_id, 1337)
    other = next(name for name in ("train", "val", "test") if name != natural)

    result = partition([ref], eval_ids=frozenset(), artist_names=frozenset(),
                       seed=1337, existing={other: [ref.youtube_id]})

    assert result[other] == [ref.youtube_id]
    assert result[natural] == []


def test_partition_exclusions_win_over_a_stale_assignment():
    """An id that becomes excluded is pulled out of the frozen file, not left in."""
    stale = {"train": ["evalaaaaaaa"], "val": [], "test": []}
    result = partition(
        [TrackRef("0001.evalaaaaaaa", "evalaaaaaaa", "A - B")],
        eval_ids=frozenset({"evalaaaaaaa"}), artist_names=frozenset(),
        seed=1337, existing=stale,
    )
    assert result["train"] == []
    assert result["excluded_eval_set"] == ["evalaaaaaaa"]


# --------------------------------------------------------------------------- #
# make_splits over a miniature corpus on disk
# --------------------------------------------------------------------------- #


SECTIONS = [
    {"start": 0.0, "end": 30.0, "name": "intro"},
    {"start": 30.0, "end": 60.0, "name": "buildup"},
    {"start": 60.0, "end": 120.0, "name": "drop"},
    {"start": 120.0, "end": 150.0, "name": "outro"},
]


def fake_corpus(tmp_path, tracks, frames=400):
    """A data dir with just what the dataset builder reads."""
    data_dir = tmp_path / "raveform"
    (data_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (data_dir / "features").mkdir(parents=True, exist_ok=True)

    with open(data_dir / "clean_manifest.csv", "w", encoding="utf-8", newline="") as handle:
        handle.write("track_id,youtube_id,mp3_path,ffprobe_duration_sec,"
                     "decoded_duration_sec,annotation_duration_sec,status,detail\n")
        for track_id, youtube, _title in tracks:
            handle.write(f"{track_id},{youtube},x.mp3,150.0,150.0,150.0,ok,\n")

    segments = [
        {"key": track_id, "id": youtube, "title": title, "sections": SECTIONS}
        for track_id, youtube, title in tracks
    ]
    with open(data_dir / "annotations" / "segments.json", "w", encoding="utf-8") as handle:
        json.dump(segments, handle)

    rng = np.random.default_rng(0)
    for _track_id, youtube, _title in tracks:
        mel = rng.random((frames, 40), dtype=np.float32)
        write_feature_sidecar(data_dir / "features" / f"{youtube}.npz",
                              mel, FRAME_SEC, FRAME_SEC)

    eval_set = tmp_path / "eval_set.json"
    with open(eval_set, "w", encoding="utf-8") as handle:
        json.dump({"youtube_ids": [tracks[0][1]],
                   "tracks": [{"youtube_id": tracks[0][1], "title": tracks[0][2]}]}, handle)
    return data_dir, eval_set


def corpus_tracks(count, titles=None):
    titles = titles or {}
    return [
        (f"{index:04d}.id{index:08d}", f"id{index:08d}",
         titles.get(index, f"Artist {index} - Track"))
        for index in range(count)
    ]


def test_make_splits_writes_and_reuses_the_frozen_file(tmp_path):
    tracks = corpus_tracks(60)
    data_dir, eval_set = fake_corpus(tmp_path, tracks)

    first = make_splits(data_dir, eval_set_path=eval_set)
    assert (data_dir / SPLITS_FILE).exists()

    again = make_splits(data_dir, eval_set_path=eval_set)
    for split in ("train", "val", "test"):
        assert first[split] == again[split]


def test_make_splits_excludes_the_eval_set_and_its_artists(tmp_path):
    tracks = corpus_tracks(40, titles={
        0: "Greg Downey Feat. Bo Bruce - Come To Me",   # the eval track
        1: "Greg Downey - Rewired",                     # solo, must be dropped
        2: "Bo Bruce - Alone",                          # featured, must be dropped
    })
    data_dir, eval_set = fake_corpus(tmp_path, tracks)

    result = make_splits(data_dir, eval_set_path=eval_set)

    placed = set(result["train"]) | set(result["val"]) | set(result["test"])
    assert "id00000000" not in placed
    assert "id00000001" not in placed
    assert "id00000002" not in placed
    assert result["excluded_eval_set"] == ["id00000000"]
    assert set(result["excluded_artist"]) == {"id00000001", "id00000002"}
    assert len(placed) == 37


def test_make_splits_extends_when_the_corpus_grows(tmp_path):
    tracks = corpus_tracks(80)
    data_dir, eval_set = fake_corpus(tmp_path, tracks)
    before = make_splits(data_dir, eval_set_path=eval_set)

    grown, _eval_set = fake_corpus(tmp_path / "grown", corpus_tracks(160))
    # Re-point the grown corpus at the same frozen splits file.
    (grown / SPLITS_FILE).write_bytes((data_dir / SPLITS_FILE).read_bytes())
    after = make_splits(grown, eval_set_path=eval_set)

    for split in ("train", "val", "test"):
        assert set(before[split]) <= set(after[split])


def test_make_splits_skips_tracks_without_a_sidecar(tmp_path):
    tracks = corpus_tracks(20)
    data_dir, eval_set = fake_corpus(tmp_path, tracks)
    (data_dir / "features" / f"{tracks[5][1]}.npz").unlink()

    result = make_splits(data_dir, eval_set_path=eval_set)

    placed = set(result["train"]) | set(result["val"]) | set(result["test"])
    assert tracks[5][1] not in placed
    assert result["skipped_no_sidecar"] == [tracks[5][1]]


# --------------------------------------------------------------------------- #
# Targets and masks
# --------------------------------------------------------------------------- #


def sections(*spans):
    return [(float(start), float(end), label) for start, end, label in spans]


def frame_of(t):
    """Index of the frame carrying song time ``t`` (frame k is stamped t0+k*dt)."""
    return int(round(t / FRAME_SEC)) - 1


def targets_for(spans, duration=200.0):
    n_frames = int(duration / FRAME_SEC)
    return track_targets(sections(*spans), n_frames, FRAME_SEC, FRAME_SEC)


def test_leading_offset_is_masked_for_both_heads():
    """[0, first section start) is unannotated audio -- never a training target."""
    targets = targets_for([(20.0, 60.0, "intro"), (60.0, 120.0, "drop")])

    last_leading = frame_of(19.0)
    assert not targets.label_mask[:last_leading + 1].any()
    assert not targets.boundary_mask[:last_leading + 1].any()
    inside = frame_of(40.0)
    assert targets.label_mask[inside]
    assert targets.boundary_mask[inside]


def test_trailing_audio_past_the_last_section_is_masked():
    targets = targets_for([(0.0, 30.0, "intro"), (30.0, 90.0, "drop")], duration=150.0)

    assert not targets.label_mask[frame_of(100.0)]
    assert not targets.boundary_mask[frame_of(100.0)]
    assert targets.label_mask[frame_of(80.0)]


def test_labels_are_folded_into_the_v1_space():
    targets = targets_for([
        (0.0, 30.0, "altintro"), (30.0, 60.0, "bridge"),
        (60.0, 90.0, "cooldown"), (90.0, 120.0, "altoutro"),
    ])
    at = lambda t: targets.label_frame[frame_of(t)]
    assert at(15.0) == CLASS_INDEX["intro"]
    assert at(45.0) == CLASS_INDEX["breakdown"]      # bridge -> breakdown
    assert at(75.0) == CLASS_INDEX["breakdown"]      # cooldown -> breakdown
    assert at(105.0) == CLASS_INDEX["outro"]         # altoutro -> outro


def test_boundary_target_peaks_at_the_boundary_frame():
    targets = targets_for([(0.0, 60.0, "buildup"), (60.0, 120.0, "drop")])
    peak = int(np.argmax(targets.boundary))

    assert targets.boundary[peak] == pytest.approx(1.0, abs=0.01)
    assert abs(peak - frame_of(60.0)) <= 1
    # Gaussian shape: exactly one sigma away the target is exp(-1/2).
    one_sigma = frame_of(60.0 + BOUNDARY_SIGMA_SEC)
    assert targets.boundary[one_sigma] == pytest.approx(math.exp(-0.5), abs=0.05)
    # ...and far from any boundary it is flat zero.
    assert targets.boundary[frame_of(20.0)] == pytest.approx(0.0, abs=1e-6)


def test_boundary_targets_sit_at_every_label_change():
    targets = targets_for([
        (0.0, 30.0, "intro"), (30.0, 60.0, "buildup"),
        (60.0, 120.0, "drop"), (120.0, 150.0, "outro"),
    ])
    for boundary in (30.0, 60.0, 120.0):
        assert targets.boundary[frame_of(boundary)] == pytest.approx(1.0, abs=0.01)


def test_merged_run_joins_are_deleted_not_taught_as_negatives():
    """breakdown|cooldown folds to one v1 run -- the join is unknowable, so it is masked."""
    targets = targets_for([
        (0.0, 30.0, "intro"), (30.0, 60.0, "breakdown"),
        (60.0, 90.0, "cooldown"), (90.0, 150.0, "drop"),
    ])
    join = frame_of(60.0)

    assert not targets.boundary_mask[join]
    radius = int(BOUNDARY_MASK_RADIUS_SEC / FRAME_SEC)
    assert not targets.boundary_mask[join - radius + 1:join + radius - 1].any()
    # The real boundaries either side keep their supervision.
    assert targets.boundary_mask[frame_of(30.0)]
    assert targets.boundary[frame_of(30.0)] == pytest.approx(1.0, abs=0.01)
    assert targets.boundary_mask[frame_of(90.0)]
    assert targets.boundary[frame_of(90.0)] == pytest.approx(1.0, abs=0.01)
    # The label head is unaffected: the whole merged run is one class.
    assert targets.label_frame[frame_of(45.0)] == CLASS_INDEX["breakdown"]
    assert targets.label_frame[frame_of(75.0)] == CLASS_INDEX["breakdown"]


def test_a_real_boundary_beside_a_merged_join_keeps_its_target():
    """Deletion must not swallow a genuine transition that happens to sit close."""
    targets = targets_for([
        (0.0, 30.0, "breakdown"), (30.0, 30.5, "cooldown"), (30.5, 90.0, "drop"),
    ])
    assert targets.boundary_mask[frame_of(30.5)]
    assert targets.boundary[frame_of(30.5)] == pytest.approx(1.0, abs=0.01)


def test_dropped_end_sentinel_leaves_an_unlabeled_gap():
    """``end`` is not a section; its time must not be re-attributed to a neighbour."""
    targets = targets_for([
        (0.0, 30.0, "intro"), (30.0, 40.0, "end"), (40.0, 90.0, "drop"),
    ])
    assert not targets.label_mask[frame_of(35.0)]
    assert not targets.boundary_mask[frame_of(35.0)]
    assert targets.label_frame[frame_of(20.0)] == CLASS_INDEX["intro"]
    assert targets.label_frame[frame_of(60.0)] == CLASS_INDEX["drop"]


def test_negative_length_sections_are_clamped_not_crashed():
    targets = targets_for([(0.0, 30.0, "intro"), (60.0, 45.0, "buildup"),
                           (60.0, 120.0, "drop")])
    assert targets.label_frame[frame_of(90.0)] == CLASS_INDEX["drop"]


def test_label_pooling_takes_the_majority_of_each_group():
    targets = targets_for([(0.0, 60.0, "intro"), (60.0, 120.0, "drop")])
    pooled_at = lambda t: targets.label_pooled[frame_of(t) // LABEL_POOL]

    assert pooled_at(30.0) == CLASS_INDEX["intro"]
    assert pooled_at(90.0) == CLASS_INDEX["drop"]
    assert len(targets.label_pooled) == len(targets.label_frame) // LABEL_POOL


def test_label_pooling_masks_a_group_only_when_every_frame_is_masked():
    targets = targets_for([(20.0, 60.0, "intro")], duration=100.0)
    assert not targets.label_pooled_mask[frame_of(5.0) // LABEL_POOL]
    assert targets.label_pooled_mask[frame_of(40.0) // LABEL_POOL]
    assert not targets.label_pooled_mask[frame_of(80.0) // LABEL_POOL]
    # A masked pooled frame still carries the loss-ignore sentinel, not class 0.
    masked = ~targets.label_pooled_mask
    assert (targets.label_pooled[masked] == IGNORE_INDEX).all()


def test_a_track_with_no_labeled_sections_is_entirely_masked():
    targets = targets_for([(0.0, 30.0, "end")])
    assert not targets.label_mask.any()
    assert not targets.boundary_mask.any()
    assert targets.boundary.max() == 0.0


# --------------------------------------------------------------------------- #
# WindowDataset
# --------------------------------------------------------------------------- #


def dataset(tmp_path, count=8, frames=3000, **kwargs):
    tracks = corpus_tracks(count)
    data_dir, _eval_set = fake_corpus(tmp_path, tracks, frames=frames)
    ids = [youtube for _t, youtube, _title in tracks]
    return WindowDataset(data_dir, ids, **kwargs)


def test_window_dataset_yields_the_documented_shapes_and_dtypes(tmp_path):
    data = dataset(tmp_path)
    mel, labels, label_mask, boundary, boundary_mask = data[0]

    assert mel.shape == (WINDOW_FRAMES, 40) and mel.dtype == np.float32
    assert labels.shape == (LABEL_FRAMES,) and labels.dtype == np.int64
    assert label_mask.shape == (LABEL_FRAMES,) and label_mask.dtype == np.bool_
    assert boundary.shape == (WINDOW_FRAMES,) and boundary.dtype == np.float32
    assert boundary_mask.shape == (WINDOW_FRAMES,) and boundary_mask.dtype == np.bool_


def test_window_dataset_covers_every_track(tmp_path):
    data = dataset(tmp_path, count=8, frames=3000)
    assert len(data) == 8 * (3000 // WINDOW_FRAMES)


def test_eval_mode_is_deterministic_and_untouched_by_the_epoch(tmp_path):
    data = dataset(tmp_path, augment=False)
    first = data[3][0].copy()
    data.set_epoch(9)
    assert np.array_equal(data[3][0], first)


def test_augmentation_moves_the_window_and_the_gain(tmp_path):
    plain = dataset(tmp_path, augment=False)
    jittered = dataset(tmp_path, augment=True)
    windows = {jittered.window_offset(index) for index in range(len(jittered))}
    assert len(windows) > 1                                   # offsets really vary

    jittered.set_epoch(1)
    epoch_one = jittered.window_offset(0)
    jittered.set_epoch(2)
    assert isinstance(epoch_one, int)
    assert not np.array_equal(plain[0][0], jittered[0][0])     # gain and/or offset moved


def test_gain_jitter_stays_within_the_documented_range_and_is_non_negative(tmp_path):
    data = dataset(tmp_path, augment=True, gain_jitter_db=GAIN_JITTER_DB)
    plain = dataset(tmp_path, augment=False)
    limit = GAIN_JITTER_DB * math.log(10.0) / 20.0

    for index in range(len(data)):
        offset = data.window_offset(index)
        assert offset % LABEL_POOL == 0                        # pooled slices stay aligned
        mel = data[index][0]
        assert (mel >= 0.0).all()                              # log1p output is non-negative
        reference = plain.window(index, offset)[0]
        delta = mel[reference > limit] - reference[reference > limit]
        if delta.size:
            assert abs(float(delta.max()) - float(delta.min())) < 1e-4   # one gain per window
            assert abs(float(delta[0])) <= limit + 1e-6


def test_augmentation_does_not_disturb_the_targets(tmp_path):
    plain = dataset(tmp_path, augment=False)
    jittered = dataset(tmp_path, augment=True)
    offset = jittered.window_offset(0)
    assert np.array_equal(jittered[0][1], plain.window(0, offset)[1])


def test_short_track_is_padded_and_the_padding_is_masked(tmp_path):
    data = dataset(tmp_path, count=1, frames=WINDOW_FRAMES // 2)
    mel, _labels, label_mask, _boundary, boundary_mask = data[0]

    assert mel.shape == (WINDOW_FRAMES, 40)
    assert (mel[WINDOW_FRAMES // 2:] == 0.0).all()
    assert not boundary_mask[WINDOW_FRAMES // 2:].any()
    assert not label_mask[WINDOW_FRAMES // 2 // LABEL_POOL:].any()


def test_dataset_refuses_a_track_with_nothing_to_supervise(tmp_path):
    """A fully masked track is hours of zero gradient that looks like it works."""
    tracks = corpus_tracks(1)
    data_dir, _eval_set = fake_corpus(tmp_path, tracks, frames=1000)
    with pytest.raises(RuntimeError, match="label_v1"):
        WindowDataset(data_dir, [tracks[0][1]],
                      sections_by_youtube_id={tracks[0][1]: [(0.0, 30.0, "end")]})


def test_dataset_refuses_a_sidecar_with_a_foreign_frame_rate(tmp_path):
    tracks = corpus_tracks(1)
    data_dir, _eval_set = fake_corpus(tmp_path, tracks, frames=1000)
    write_feature_sidecar(data_dir / "features" / f"{tracks[0][1]}.npz",
                          np.zeros((1000, 40), dtype=np.float32), 0.01, 0.01)
    data = WindowDataset(data_dir, [tracks[0][1]])
    with pytest.raises(RuntimeError, match="frame_sec"):
        _ = data[0]
