"""Tests for the eval-set benchmark runner (training/run_eval_set.py)."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from eval_assets import (  # noqa: E402
    ACCEPTED,
    EVAL_LABELS_FILE,
    OVERRIDDEN,
    OWNER,
    PUBLISHED,
    UNREVIEWED,
    committed_audio_path,
    owner_rulings,
)
from evaluate_against_labels import LEGACY_V1, RAW9  # noqa: E402
from run_eval_set import (  # noqa: E402
    AUDIO_MISSING_HINT,
    BASELINE_FILE,
    BOUNDARY_TOLERANCE_SEC,
    COUNT_FACTS,
    DATA_DIR_ENV,
    DEFAULT_FLICKER_TOLERANCE,
    DEFAULT_SCORE_TOLERANCE,
    EVAL_SET_FILE,
    GATED_SPACE,
    GATED_TRUTH,
    GUARDED_METRICS,
    REPORTED_SPACES,
    REPORTED_TRUTHS,
    audio_path,
    build_document,
    build_jobs,
    build_parser,
    compare,
    corpus_audio_path,
    corpus_dir,
    default_data_dir,
    file_sha256,
    labels_source,
    load_baseline,
    load_eval_set,
    load_sections_by_truth,
    missing_inputs,
    partial_baseline_refusal,
    score_report,
    select_tracks,
    shortest_track_ids,
    space_facts,
    track_entry,
    track_metrics,
    unreviewed_truth_refusal,
    verify_ground_truth,
    verify_owner_ground_truth,
)

LOOK_AHEAD = 2.5

NO_SLICE = "absent-labels.json"


def eval_document(*specs) -> dict:
    return {
        "youtube_ids": [youtube for _track, youtube, _duration in specs],
        "tracks": [
            {"track_id": track, "youtube_id": youtube, "duration_sec": duration}
            for track, youtube, duration in specs
        ],
    }


def metrics(macro_f1=0.5, accuracy=0.5, boundary_f1=0.5, crispness=0.5,
            flicker_per_min=1.0) -> dict:
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "boundary_f1": boundary_f1,
        "crispness": crispness,
        "flicker_per_min": flicker_per_min,
    }


def result_document(tracks: dict, aggregate: dict | None = None,
                    eval_sha: str = "abc123") -> dict:
    return {
        "eval_set": {"sha256": eval_sha, "tracks": len(tracks),
                     "youtube_ids": sorted(tracks)},
        "pipeline_sha": "deadbeef",
        "space": GATED_SPACE,
        "truth": GATED_TRUTH,
        "aggregate": aggregate if aggregate is not None else metrics(),
        "tracks": tracks,
    }


def entry(checksum="cafe", **overrides) -> dict:
    record = {"youtube_id": "yt", "checksum": checksum, "beats": 100, "rows": 90,
              "song_sec": 300.0, "exposure_sec": 280.0,
              "changes_intent": 20, "changes_class": 15, "label_boundaries": 12,
              "late": 1, "blocks_measurable": 14,
              "silence_leading": 0, "silence_interior": 0, "silence_trailing": 1}
    record.update(metrics())
    record.update(overrides)
    return record


def corpus(tmp_path: Path, segments: str = "[]") -> Path:
    (tmp_path / "annotations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "annotations" / "segments.json").write_text(segments, encoding="utf-8")
    (tmp_path / "clean_manifest.csv").write_text("track_id\n", encoding="utf-8")
    return tmp_path


def frozen_against(tmp_path: Path) -> dict:
    from select_eval_set import input_paths, sha256_of

    return {"selected_from": {"inputs": {name: sha256_of(path) for name, path
                                         in input_paths(tmp_path).items()}}}


def test_shortest_track_ids_takes_the_shortest_by_duration():
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0),
                             ("c.3", "3", 300.0), ("d.4", "4", 470.0))
    assert shortest_track_ids(document, 2) == ["b.2", "c.3"]


def test_shortest_track_ids_keeps_eval_set_order():
    document = eval_document(("a.1", "1", 300.0), ("b.2", "2", 250.0),
                             ("c.3", "3", 400.0))
    assert shortest_track_ids(document, 2) == ["a.1", "b.2"]


def test_shortest_track_ids_breaks_duration_ties_by_id():
    document = eval_document(("z.9", "9", 300.0), ("a.1", "1", 300.0),
                             ("m.5", "5", 400.0))
    assert shortest_track_ids(document, 1) == ["a.1"]


def test_shortest_track_ids_clamps_to_the_set_size():
    document = eval_document(("a.1", "1", 300.0))
    assert shortest_track_ids(document, 5) == ["a.1"]


def test_select_tracks_defaults_to_the_whole_set_in_order():
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    assert [track["track_id"] for track in select_tracks(document)] == ["a.1", "b.2"]


def test_select_tracks_accepts_track_ids_and_youtube_ids():
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    picked = select_tracks(document, ["2", "a.1"])
    assert [track["track_id"] for track in picked] == ["a.1", "b.2"]


def test_select_tracks_rejects_an_id_outside_the_frozen_set():
    document = eval_document(("a.1", "1", 400.0))
    with pytest.raises(RuntimeError, match="not in the eval set"):
        select_tracks(document, ["nope"])


def test_missing_inputs_names_every_absent_mp3_and_both_places_to_get_it(tmp_path):
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    (tmp_path / "audio").mkdir()
    corpus_audio_path(tmp_path, "1").write_bytes(b"not really an mp3")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "segments.json").write_text("[]", encoding="utf-8")

    problems = missing_inputs(tmp_path, select_tracks(document))
    assert len(problems) == 1
    assert "b.2" in problems[0] and AUDIO_MISSING_HINT in problems[0]
    assert "eval_audio" in problems[0] and "raveform_download" in problems[0]


def test_missing_inputs_is_empty_when_everything_is_there(tmp_path):
    document = eval_document(("a.1", "1", 400.0))
    (tmp_path / "audio").mkdir()
    corpus_audio_path(tmp_path, "1").write_bytes(b"x")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "segments.json").write_text("[]", encoding="utf-8")
    assert missing_inputs(tmp_path, select_tracks(document)) == []


def test_build_jobs_refuses_a_track_with_no_annotation(tmp_path):
    document = eval_document(("a.1", "1", 400.0))
    with pytest.raises(RuntimeError, match="no annotation"):
        build_jobs(tmp_path, select_tracks(document), {})


def by_truth(published: dict, owner: dict | None = None) -> dict:
    return {PUBLISHED: published, OWNER: owner if owner is not None else published}


def test_build_jobs_gives_each_worker_only_its_own_sections(tmp_path):
    document = eval_document(("a.1", "1", 400.0))
    sections = {"a.1": [(0.0, 10.0, "intro")], "other.2": [(0.0, 5.0, "drop")]}
    jobs = build_jobs(tmp_path, select_tracks(document), by_truth(sections))
    assert [job.sections for job in jobs] == [
        {truth: sections["a.1"] for truth in REPORTED_TRUTHS}]


def test_build_jobs_carries_every_truths_reading_of_the_same_track(tmp_path):
    """One simulation, both readings -- the sections travel with the job so a
    worker never re-reads a ground truth the parent already resolved."""
    document = eval_document(("a.1", "1", 400.0))
    published = {"a.1": [(0.0, 10.0, "drop")]}
    owner = {"a.1": [(0.0, 10.0, "bridge")]}
    jobs = build_jobs(tmp_path, select_tracks(document),
                      by_truth(published, owner))

    assert jobs[0].sections[PUBLISHED] == published["a.1"]
    assert jobs[0].sections[OWNER] == owner["a.1"]


def test_build_jobs_refuses_a_track_missing_from_one_truth_only(tmp_path):
    document = eval_document(("a.1", "1", 400.0))
    with pytest.raises(RuntimeError, match="no annotation"):
        build_jobs(tmp_path, select_tracks(document),
                   {PUBLISHED: {"a.1": [(0.0, 10.0, "intro")]}, OWNER: {}})


def test_missing_inputs_reports_a_missing_annotation_file(tmp_path):
    document = eval_document(("a.1", "1", 400.0))
    (tmp_path / "audio").mkdir()
    corpus_audio_path(tmp_path, "1").write_bytes(b"x")
    problems = missing_inputs(tmp_path, select_tracks(document),
                              labels=tmp_path / NO_SLICE)
    assert any("segments.json" in problem for problem in problems)


def test_missing_inputs_does_not_want_the_corpus_when_the_slice_is_committed(tmp_path):
    document = eval_document(("a.1", "1", 400.0))
    (tmp_path / "audio").mkdir()
    corpus_audio_path(tmp_path, "1").write_bytes(b"x")
    assert missing_inputs(tmp_path, select_tracks(document)) == []


def test_audio_resolution_prefers_the_committed_copy(tmp_path, monkeypatch):
    youtube_id = load_eval_set(EVAL_SET_FILE)["youtube_ids"][0]
    (tmp_path / "audio").mkdir()
    corpus_audio_path(tmp_path, youtube_id).write_bytes(b"a decoy corpus mp3")

    assert audio_path(tmp_path, youtube_id) == committed_audio_path(youtube_id)


def test_audio_resolution_falls_back_to_the_corpus(tmp_path):
    assert audio_path(tmp_path, "not-an-eval-track") == corpus_audio_path(
        tmp_path, "not-an-eval-track")


def test_missing_audio_names_the_corpus_path_a_human_can_act_on(tmp_path):
    assert audio_path(tmp_path, "nope") == corpus_audio_path(tmp_path, "nope")


def test_labels_resolution_prefers_the_committed_slice(tmp_path):
    path, committed = labels_source(tmp_path)
    assert committed is True
    assert path == EVAL_LABELS_FILE


def test_labels_resolution_falls_back_to_the_corpus_annotation(tmp_path):
    path, committed = labels_source(tmp_path, labels=tmp_path / NO_SLICE)
    assert committed is False
    assert path == tmp_path / "annotations" / "segments.json"


def test_the_committed_slice_covers_exactly_the_frozen_eval_set(tmp_path):
    truths = load_sections_by_truth(tmp_path)
    frozen = {track["track_id"] for track in load_eval_set(EVAL_SET_FILE)["tracks"]}
    assert set(truths) == set(REPORTED_TRUTHS)
    for truth, sections in truths.items():
        assert set(sections) == frozen, truth
        assert all(len(sections[track_id]) > 1 for track_id in frozen), truth


def test_every_committed_mp3_the_eval_set_names_is_there():
    missing = [youtube_id
               for youtube_id in load_eval_set(EVAL_SET_FILE)["youtube_ids"]
               if not committed_audio_path(youtube_id).exists()]
    assert not missing, f"{len(missing)} eval-set mp3s are not committed: {missing}"


def slice_cut_from(tmp_path: Path, sha: str, rulings: dict | None = None) -> Path:
    path = tmp_path / "eval_labels.json"
    path.write_text(json.dumps({
        "schema": 2,
        "truths": {
            "published": {"file": "annotations/segments.json", "sha256": sha},
            "owner": {"rulings": rulings or {"a.1": {"ruling": UNREVIEWED}}},
        },
        "tracks": [{"key": "a.1", "sections": []}],
        "owner_tracks": [],
    }), encoding="utf-8")
    return path


def test_a_slice_cut_from_the_frozen_labels_passes(tmp_path):
    document = {"selected_from": {"inputs": {"segments.json": "abc123"}}}
    verify_ground_truth(document, tmp_path,
                        labels=slice_cut_from(tmp_path, "abc123"))


def test_a_slice_cut_from_other_labels_is_a_hard_failure(tmp_path):
    document = {"selected_from": {"inputs": {"segments.json": "abc123"}}}
    with pytest.raises(RuntimeError) as excinfo:
        verify_ground_truth(document, tmp_path,
                            labels=slice_cut_from(tmp_path, "d1ffe4e47"))
    message = str(excinfo.value)
    assert "GROUND TRUTH" in message and "eval_assets.py --cut" in message


def test_a_slice_with_no_recorded_source_is_a_failure(tmp_path):
    path = tmp_path / "eval_labels.json"
    path.write_text(json.dumps({"schema": 2, "tracks": []}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="GROUND TRUTH"):
        verify_ground_truth({"selected_from": {"inputs": {"segments.json": "a"}}},
                            tmp_path, labels=path)


def test_the_committed_slice_was_cut_from_the_frozen_ground_truth():
    verify_ground_truth(load_eval_set(EVAL_SET_FILE), Path("does-not-exist"))


# --------------------------------------------------------------------------- #
# The owner half of the provenance guard
# --------------------------------------------------------------------------- #


def hand_label(tmp_path: Path, body: str = '{"sections": []}') -> tuple:
    """A stand-in hand label named RELATIVE TO THE REPO, as a ruling records it.

    The path is deliberately built from the repo root rather than a corpus dir:
    a linked worktree's corpus_dir() is the MAIN checkout's, so hashing through
    it would make this guard's verdict a fact about which worktree ran it.
    """
    from run_eval_set import REPO_ROOT

    path = tmp_path / "a-hand-label.json"
    path.write_bytes(body.encode("utf-8"))
    try:
        name = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        name = path.as_posix()
    return name, hashlib.sha256(body.encode("utf-8")).hexdigest()


def owner_slice(rulings: dict) -> dict:
    return {"truths": {"owner": {"rulings": rulings}}}


def test_an_unmoved_hand_label_passes_the_owner_ground_truth_check(tmp_path):
    name, sha = hand_label(tmp_path)
    verify_owner_ground_truth(owner_slice(
        {"a.1": {"ruling": OVERRIDDEN, "file": name, "sha256": sha}}))


def test_a_moved_hand_label_is_a_hard_failure_that_names_the_track(tmp_path):
    name, sha = hand_label(tmp_path)
    (tmp_path / "a-hand-label.json").write_bytes(b'{"sections": ["moved"]}')

    with pytest.raises(RuntimeError) as excinfo:
        verify_owner_ground_truth(owner_slice(
            {"a.1": {"ruling": OVERRIDDEN, "file": name, "sha256": sha}}))

    message = str(excinfo.value)
    assert "OWNER GROUND TRUTH" in message
    assert "a.1" in message and "eval_assets.py --cut" in message


def test_a_missing_hand_label_is_a_hard_failure_rather_than_a_skip(tmp_path):
    name, sha = hand_label(tmp_path)
    (tmp_path / "a-hand-label.json").unlink()

    with pytest.raises(RuntimeError, match="not on this checkout"):
        verify_owner_ground_truth(owner_slice(
            {"a.1": {"ruling": OVERRIDDEN, "file": name, "sha256": sha}}))


def test_an_overridden_ruling_with_no_sha_is_a_failure():
    with pytest.raises(RuntimeError, match="OWNER GROUND TRUTH"):
        verify_owner_ground_truth(owner_slice({"a.1": {"ruling": OVERRIDDEN}}))


def test_a_track_the_owner_has_not_relabelled_pins_nothing(tmp_path):
    """A freeze that does not follow a NEW hand label is the freeze working;
    only what the slice adopted is pinned."""
    verify_owner_ground_truth(owner_slice(
        {"a.1": {"ruling": UNREVIEWED},
         "b.2": {"ruling": ACCEPTED, "dated": "2026-09-09"}}))


def test_every_overridden_ruling_has_its_hand_label_on_disk_at_the_recorded_sha():
    from eval_assets import load_labels
    from run_eval_set import REPO_ROOT

    verify_owner_ground_truth(load_labels(EVAL_LABELS_FILE))

    for track_id, ruling in owner_rulings(load_labels(EVAL_LABELS_FILE)).items():
        if ruling.get("ruling") != OVERRIDDEN:
            continue
        path = REPO_ROOT / ruling["file"]
        assert path.exists(), track_id
        assert file_sha256(path) == ruling["sha256"], track_id


def test_the_owner_rulings_name_every_frozen_track():
    """A newly committed hand label the slice does not carry reds the FAST suite
    in milliseconds rather than hiding until someone runs the benchmark."""
    from eval_assets import hand_records, load_labels

    frozen = load_eval_set(EVAL_SET_FILE)["tracks"]
    rulings = owner_rulings(load_labels(EVAL_LABELS_FILE))
    assert set(rulings) == {track["track_id"] for track in frozen}

    on_disk = hand_records()
    overridden = {track_id for track_id, ruling in rulings.items()
                  if ruling.get("ruling") == OVERRIDDEN}
    expected = {str(track["track_id"]) for track in frozen
                if str(track["youtube_id"]) in on_disk}
    assert overridden == expected, (
        "a hand label for a frozen track is not adopted by the slice -- re-cut "
        "it with training/eval_assets.py --cut and re-cut the baseline")


def test_unmoved_labels_pass_the_ground_truth_check(tmp_path):
    data_dir = corpus(tmp_path, segments='[{"key": "a.1"}]')
    verify_ground_truth(frozen_against(data_dir), data_dir,
                        labels=tmp_path / NO_SLICE)


def test_moved_labels_are_a_hard_failure_that_names_the_cause(tmp_path):
    data_dir = corpus(tmp_path, segments='[{"key": "a.1"}]')
    document = frozen_against(data_dir)
    (data_dir / "annotations" / "segments.json").write_text(
        '[{"key": "a.1", "sections": "moved"}]', encoding="utf-8")

    with pytest.raises(RuntimeError) as excinfo:
        verify_ground_truth(document, data_dir, labels=tmp_path / NO_SLICE)
    message = str(excinfo.value)
    assert "GROUND TRUTH" in message
    assert "segments.json" in message
    assert "baseline" in message


def test_a_grown_corpus_does_not_fail_the_ground_truth_check(tmp_path):
    data_dir = corpus(tmp_path, segments='[{"key": "a.1"}]')
    document = frozen_against(data_dir)
    (data_dir / "clean_manifest.csv").write_text("track_id\nb.2\n", encoding="utf-8")

    verify_ground_truth(document, data_dir, labels=tmp_path / NO_SLICE)


def test_an_eval_set_with_no_recorded_label_checksum_is_a_failure(tmp_path):
    data_dir = corpus(tmp_path)
    with pytest.raises(RuntimeError, match="GROUND TRUTH"):
        verify_ground_truth({"selected_from": {"inputs": {}}}, data_dir,
                            labels=tmp_path / NO_SLICE)


def test_an_identical_run_compares_clean():
    document = result_document({"a.1": entry()})
    assert compare(document, document).failed is False


def test_a_changed_checksum_fails_and_names_the_track():
    baseline = result_document({"a.1": entry(checksum="cafe")})
    current = result_document({"a.1": entry(checksum="f00d")})
    outcome = compare(baseline, current)
    assert outcome.failed
    assert outcome.checksum_drift and "a.1" in outcome.checksum_drift[0]
    assert not outcome.regressions


def test_a_score_drop_beyond_tolerance_is_a_regression():
    baseline = result_document({"a.1": entry(macro_f1=0.50)})
    current = result_document({"a.1": entry(macro_f1=0.50 - DEFAULT_SCORE_TOLERANCE - 0.01)})
    outcome = compare(baseline, current)
    assert outcome.failed
    assert any("macro_f1" in line for line in outcome.regressions)


def test_a_score_drop_inside_tolerance_is_not_a_regression():
    baseline = result_document({"a.1": entry(macro_f1=0.50)})
    current = result_document({"a.1": entry(macro_f1=0.50 - DEFAULT_SCORE_TOLERANCE / 2)})
    assert compare(baseline, current).regressions == []


def test_a_score_improvement_is_never_a_regression():
    baseline = result_document({"a.1": entry(macro_f1=0.50)})
    current = result_document({"a.1": entry(macro_f1=0.95)})
    assert compare(baseline, current).regressions == []


def test_flicker_regresses_upward_not_downward():
    baseline = result_document({"a.1": entry(flicker_per_min=1.0)})
    worse = result_document({"a.1": entry(flicker_per_min=1.0 + DEFAULT_FLICKER_TOLERANCE + 0.1)})
    better = result_document({"a.1": entry(flicker_per_min=0.1)})
    assert any("flicker" in line for line in compare(baseline, worse).regressions)
    assert compare(baseline, better).regressions == []


def test_the_aggregate_row_is_gated_too():
    baseline = result_document({"a.1": entry()}, aggregate=metrics(macro_f1=0.6))
    current = result_document({"a.1": entry()}, aggregate=metrics(macro_f1=0.2))
    outcome = compare(baseline, current)
    assert any("(aggregate)" in line for line in outcome.regressions)


@pytest.mark.parametrize("fact,moved", [
    ("rows", 89),
    ("label_boundaries", 13),
    ("exposure_sec", 279.5),
    ("late", 2),
    ("beats", 101),
    ("changes_intent", 21),
    ("silence_interior", 1),
    ("silence_trailing", 2),
    ("silence_leading", 1),
])
def test_a_moved_count_fact_fails_even_when_every_score_holds(fact, moved):
    baseline = result_document({"a.1": entry()})
    current = result_document({"a.1": entry(**{fact: moved})})

    outcome = compare(baseline, current)

    assert outcome.failed
    assert any(fact in line and "a.1" in line for line in outcome.fact_drift)
    assert outcome.regressions == [] and outcome.checksum_drift == []


def test_count_facts_are_exact_with_no_tolerance():
    baseline = result_document({"a.1": entry(rows=90)})
    current = result_document({"a.1": entry(rows=90 + 1)})

    assert compare(baseline, current).failed


def test_the_aggregate_row_is_not_faulted_for_facts_it_never_carries():
    baseline = result_document({"a.1": entry()}, aggregate=metrics())
    assert compare(baseline, baseline).fact_drift == []


def test_every_guarded_metric_is_actually_checked():
    for metric, direction in GUARDED_METRICS.items():
        worse = 0.5 - 5.0 if direction == "down" else 0.5 + 5.0
        baseline = result_document({"a.1": entry(**{metric: 0.5})})
        current = result_document({"a.1": entry(**{metric: worse})})
        outcome = compare(baseline, current)
        assert any(metric in line for line in outcome.regressions), metric


def test_a_subset_run_compares_only_its_own_tracks():
    baseline = result_document({"a.1": entry(), "b.2": entry(), "c.3": entry()},
                               aggregate=metrics(macro_f1=0.60))
    current = result_document({"a.1": entry()}, aggregate=metrics(macro_f1=0.10))
    outcome = compare(baseline, current)
    assert outcome.failed is False
    assert outcome.subset is True


def test_a_track_with_no_baseline_entry_fails():
    baseline = result_document({"a.1": entry()})
    current = result_document({"a.1": entry(), "new.9": entry()})
    outcome = compare(baseline, current)
    assert outcome.failed
    assert any("new.9" in line for line in outcome.unbaselined)


def test_a_baseline_cut_against_a_different_eval_set_is_a_hard_failure():
    baseline = result_document({"a.1": entry()}, eval_sha="OLD")
    current = result_document({"a.1": entry()}, eval_sha="NEW")
    outcome = compare(baseline, current)
    assert outcome.failed
    assert outcome.desync and "OLD"[:6] in outcome.desync[0]


def test_a_drift_with_improved_scores_still_fails():
    baseline = result_document({"a.1": entry(checksum="cafe", macro_f1=0.20,
                                             flicker_per_min=4.0)})
    current = result_document({"a.1": entry(checksum="f00d", macro_f1=0.60,
                                            flicker_per_min=1.0)})
    outcome = compare(baseline, current)
    assert outcome.failed
    assert outcome.checksum_drift
    assert outcome.regressions == []


def test_a_guarded_metric_missing_from_the_baseline_is_a_failure_not_a_skip():
    stale = entry()
    stale.pop("macro_f1")
    outcome = compare(result_document({"a.1": stale}),
                      result_document({"a.1": entry()}))
    assert outcome.failed
    assert any("macro_f1" in line and "NOT being gated" in line
               for line in outcome.ungated)


def test_load_baseline_explains_itself_when_absent(tmp_path):
    with pytest.raises(RuntimeError, match="--write-baseline"):
        load_baseline(tmp_path / "nope.json")


def test_load_baseline_rejects_a_document_without_tracks(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"eval_set": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a baseline"):
        load_baseline(path)


def committed_baseline() -> dict:
    return load_baseline(BASELINE_FILE)


def test_the_committed_baseline_covers_the_whole_frozen_eval_set():
    frozen = {track["track_id"] for track in load_eval_set(EVAL_SET_FILE)["tracks"]}
    assert set(committed_baseline()["tracks"]) == frozen


def test_the_frozen_artifacts_are_checked_out_with_canonical_line_endings():
    for path in (EVAL_SET_FILE, BASELINE_FILE):
        assert b"\r" not in Path(path).read_bytes(), (
            f"{Path(path).name} was checked out with CRLF, but its recorded "
            f"sha256 is over LF bytes -- check .gitattributes still pins "
            f"training/*.json to eol=lf, then `git add --renormalize` it"
        )


def test_the_committed_baseline_was_cut_against_the_current_eval_set():
    baseline = committed_baseline()
    assert baseline["eval_set"]["sha256"] == file_sha256(EVAL_SET_FILE)
    assert baseline["eval_set"]["tracks"] == len(baseline["tracks"])


def test_every_committed_row_carries_every_guarded_metric_and_a_checksum():
    baseline = committed_baseline()
    rows = {"(aggregate)": baseline["aggregate"], **baseline["tracks"]}
    for name, row in rows.items():
        missing = [metric for metric in GUARDED_METRICS if metric not in row]
        assert not missing, f"{name} is missing {missing} -- it would not be gated"
    for track_id, row in baseline["tracks"].items():
        assert len(row.get("checksum", "")) == 64, track_id


def test_every_committed_row_reads_the_run_through_both_annotators():
    baseline = committed_baseline()
    rows = {"(aggregate)": baseline["aggregate"], **baseline["tracks"]}
    for name, row in rows.items():
        assert set(row.get("truths") or {}) == set(REPORTED_TRUTHS), name
        for truth, block in row["truths"].items():
            missing = [metric for metric in GUARDED_METRICS if metric not in block]
            assert not missing, f"{name}/{truth} is missing {missing}"


def test_every_committed_row_carries_the_count_facts_the_tripwire_reads():
    for track_id, row in committed_baseline()["tracks"].items():
        missing = [fact for fact in COUNT_FACTS if fact not in row]
        assert not missing, f"{track_id} is missing {missing} -- not a tripwire"
        for truth, block in row["truths"].items():
            missing = [fact for fact in COUNT_FACTS
                       if fact != "beats" and fact not in block]
            assert not missing, f"{track_id}/{truth} is missing {missing}"


def test_the_committed_baseline_top_level_is_its_gated_truths_reading():
    """The gate reads the top level, so it must be one annotator's numbers --
    which is also what makes 'no gated number moved' checkable after the flip."""
    for track_id, row in committed_baseline()["tracks"].items():
        gated = row["truths"][GATED_TRUTH]
        for metric in GUARDED_METRICS:
            assert row[metric] == gated[metric], (track_id, metric)
        for fact in COUNT_FACTS:
            if fact == "beats":
                continue
            assert row[fact] == gated[fact], (track_id, fact)


def test_the_committed_baseline_records_the_configuration_it_was_cut_under():
    baseline = committed_baseline()
    assert baseline["gate"]["metrics"] == dict(GUARDED_METRICS)
    assert baseline["space"] and baseline["stream"]
    assert baseline["truth"] == GATED_TRUTH
    assert list(baseline["reported_truths"]) == list(REPORTED_TRUTHS)


def test_the_committed_baseline_compares_clean_against_itself():
    baseline = committed_baseline()
    assert compare(baseline, baseline).failed is False


def test_a_subset_may_not_overwrite_the_committed_baseline():
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    refusal = partial_baseline_refusal(
        select_tracks(document, ["a.1"]), document, BASELINE_FILE)
    assert refusal and "REFUSING" in refusal
    assert "--allow-partial-baseline" in refusal


def test_a_full_run_may_overwrite_the_committed_baseline():
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    assert partial_baseline_refusal(
        select_tracks(document), document, BASELINE_FILE) is None


def test_a_subset_may_overwrite_an_explicit_alternative_path(tmp_path):
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    assert partial_baseline_refusal(
        select_tracks(document, ["a.1"]), document,
        tmp_path / "scratch.json") is None


def test_the_subset_refusal_can_be_overridden_deliberately():
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    assert partial_baseline_refusal(
        select_tracks(document, ["a.1"]), document, BASELINE_FILE,
        allowed=True) is None


# --------------------------------------------------------------------------- #
# All-or-nothing on REVIEW
# --------------------------------------------------------------------------- #


def ruled(**rulings) -> dict:
    return {"truths": {"owner": {"rulings": rulings}}}


def gates_owner_truth(monkeypatch) -> None:
    import run_eval_set

    monkeypatch.setattr(run_eval_set, "GATED_TRUTH", OWNER)


def test_an_unreviewed_track_refuses_an_owner_truth_baseline(monkeypatch):
    gates_owner_truth(monkeypatch)
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    refusal = unreviewed_truth_refusal(
        document, ruled(**{"a.1": {"ruling": OVERRIDDEN}}), BASELINE_FILE)

    assert refusal and "REFUSING" in refusal and "b.2" in refusal
    assert "--allow-unreviewed-truth" in refusal
    assert "1 of 2" in refusal


def test_a_fully_ruled_set_may_cut_an_owner_truth_baseline(monkeypatch):
    gates_owner_truth(monkeypatch)
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    assert unreviewed_truth_refusal(document, ruled(**{
        "a.1": {"ruling": OVERRIDDEN},
        "b.2": {"ruling": ACCEPTED, "dated": "2026-09-09"},
    }), BASELINE_FILE) is None


def test_accepting_a_published_label_counts_as_ruling_on_it(monkeypatch):
    """Coverage is 'how many has he ruled on', never 'how many did he relabel':
    pressuring him to relabel eight tracks he has no complaint about would
    inject transcription noise into eight ground truths to fix two."""
    gates_owner_truth(monkeypatch)
    document = eval_document(("a.1", "1", 400.0))
    assert unreviewed_truth_refusal(
        document, ruled(**{"a.1": {"ruling": ACCEPTED, "dated": "2026-09-09"}}),
        BASELINE_FILE) is None


def test_the_review_refusal_can_be_overridden_deliberately(monkeypatch):
    gates_owner_truth(monkeypatch)
    document = eval_document(("a.1", "1", 400.0))
    assert unreviewed_truth_refusal(document, ruled(), BASELINE_FILE,
                                    allowed=True) is None


def test_an_unreviewed_set_may_write_an_explicit_alternative_path(tmp_path, monkeypatch):
    gates_owner_truth(monkeypatch)
    document = eval_document(("a.1", "1", 400.0))
    assert unreviewed_truth_refusal(document, ruled(),
                                    tmp_path / "scratch.json") is None


def test_a_published_truth_baseline_does_not_wait_on_owner_rulings():
    """The gate is on the published annotation today, so unreviewed tracks say
    nothing about whether the baseline may be re-cut."""
    document = eval_document(("a.1", "1", 400.0))
    assert unreviewed_truth_refusal(document, ruled(), BASELINE_FILE) is None


def beat(t: float) -> dict:
    return {"t": t, "bpm": 128.0, "onset_density": 4.0, "kick_strength": 2.0,
            "centroid_trend": 1.0, "sub_bass_ratio": 0.3, "rms": 0.1}


def perfect_report(beat_times, blocks, duration_sec: float) -> dict:
    return {
        "duration_sec": duration_sec,
        "beats": [beat(t) for t in beat_times],
        "intents": [{"t": start + LOOK_AHEAD, "intent": intent,
                     "end": end + LOOK_AHEAD}
                    for start, end, intent in blocks],
        "metrics": {"look_ahead_sec": LOOK_AHEAD},
    }


def test_a_report_matching_the_annotation_scores_one():
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    blocks = [(0.0, 20.0, "atmospheric"), (20.0, 40.0, "drop"),
              (40.0, 60.5, "atmospheric")]
    scores, rows, _stats = score_report(
        "t.1", "1", perfect_report(beat_times, blocks, 63.0), sections)
    assert rows == len(beat_times)
    for space in REPORTED_SPACES:
        result = track_metrics(scores[space])
        assert result["macro_f1"] == pytest.approx(1.0), space
        assert result["accuracy"] == pytest.approx(1.0), space
        assert result["boundary_f1"] == pytest.approx(1.0), space
        assert result["flicker_per_min"] == pytest.approx(0.0), space


def test_a_constant_intent_scores_far_below_one():
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    blocks = [(0.0, 60.5, "drop")]
    scores, _rows, _stats = score_report(
        "t.1", "1", perfect_report(beat_times, blocks, 63.0), sections)
    assert track_metrics(scores[GATED_SPACE])["macro_f1"] < 0.5


def test_score_report_scores_every_reported_granularity():
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    blocks = [(0.0, 60.5, "drop")]
    scores, _rows, _stats = score_report(
        "t.1", "1", perfect_report(beat_times, blocks, 63.0), sections)
    assert set(scores) == set(REPORTED_SPACES)
    assert {RAW9, LEGACY_V1} == set(REPORTED_SPACES)


def test_the_gated_space_is_the_raw_nine_the_decoder_commits():
    """Phase 2's flip: the gate reads the vocabulary the show decodes.  The
    legacy fold rides along as a reported view for comparability with the
    banked record."""
    assert GATED_SPACE == RAW9
    assert GATED_SPACE in REPORTED_SPACES
    assert LEGACY_V1 in REPORTED_SPACES


def test_a_baseline_cut_in_another_space_is_a_desync_not_a_comparison():
    baseline = {"eval_set": {"sha256": "abc"}, "space": LEGACY_V1,
                "tracks": {}, "aggregate": {}}
    current = {"eval_set": {"sha256": "abc"}, "space": GATED_SPACE,
               "tracks": {}, "aggregate": {}}
    outcome = compare(baseline, current)
    assert outcome.failed
    assert any("space" in line for line in outcome.desync)


def test_no_single_hardcoded_space_survives_in_the_runner():
    import run_eval_set

    assert not hasattr(run_eval_set, "SPACE")


def test_the_gated_truth_is_still_the_published_annotation():
    """The two-truth machinery landed as a pure refactor: the owner reading is
    measured and printed, and the gate has not moved.  Flipping this line is a
    re-freeze of the ground truth and re-cuts the baseline in the same commit."""
    assert GATED_TRUTH == PUBLISHED
    assert set(REPORTED_TRUTHS) == {OWNER, PUBLISHED}


def test_no_single_hardcoded_truth_survives_in_the_runner():
    import run_eval_set

    assert not hasattr(run_eval_set, "TRUTH")


def test_a_baseline_cut_against_another_truth_is_a_desync_not_a_comparison():
    baseline = {"eval_set": {"sha256": "abc"}, "space": GATED_SPACE,
                "truth": OWNER, "tracks": {}, "aggregate": {}}
    current = {"eval_set": {"sha256": "abc"}, "space": GATED_SPACE,
               "truth": PUBLISHED, "tracks": {}, "aggregate": {}}
    outcome = compare(baseline, current)
    assert outcome.failed
    assert any("ground truth" in line for line in outcome.desync)


def test_a_baseline_that_records_no_truth_at_all_is_a_desync():
    """Cut before the owner reading existed, so nothing says which annotator its
    numbers were scored against."""
    baseline = {"eval_set": {"sha256": "abc"}, "space": GATED_SPACE,
                "tracks": {}, "aggregate": {}}
    current = result_document({})
    current["eval_set"]["sha256"] = "abc"
    outcome = compare(baseline, current)
    assert outcome.failed
    assert any("ground truth" in line for line in outcome.desync)


def test_the_ungated_truth_cannot_fail_the_gate_on_its_own():
    """The owner column is REPORTED, not gated -- moving it must not stop a
    commit until the flip."""
    baseline = result_document({"a.1": entry()})
    current = result_document({"a.1": entry(
        truths={truth: metrics(macro_f1=0.01) for truth in REPORTED_TRUTHS})})

    assert compare(baseline, current).failed is False


def built_entry(report, sections, owner_sections=None) -> dict:
    """One simulation read by both annotators, exactly as ``run_job`` does it."""
    reads = {PUBLISHED: score_report("t.1", "1", report, sections),
             OWNER: score_report("t.1", "1", report,
                                 owner_sections if owner_sections else sections)}
    return track_entry(report, reads, "1", 63.0)


def test_a_track_entry_reports_both_granularities_with_every_guarded_metric():
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    report = perfect_report(beat_times, [(0.0, 60.5, "drop")], 63.0)

    built = built_entry(report, sections)

    assert set(built["spaces"]) == set(REPORTED_SPACES)
    for space, block in built["spaces"].items():
        assert not [metric for metric in GUARDED_METRICS if metric not in block], space


def test_a_track_entry_reports_both_annotators_with_every_guarded_metric():
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    owner = [(0.0, 20.0, "intro"), (20.0, 40.0, "bridge"), (40.0, 60.5, "outro")]
    report = perfect_report(beat_times, [(0.0, 60.5, "drop")], 63.0)

    built = built_entry(report, sections, owner)

    assert set(built["truths"]) == set(REPORTED_TRUTHS)
    for truth, block in built["truths"].items():
        assert not [metric for metric in GUARDED_METRICS
                    if metric not in block], truth
        assert not [fact for fact in COUNT_FACTS
                    if fact not in block and fact != "beats"], truth
        assert set(block["spaces"]) == set(REPORTED_SPACES), truth


def test_the_gated_numbers_at_the_top_of_an_entry_are_the_gated_spaces_numbers():
    """The baseline gate reads the top level, so it must name one space, not a mix."""
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "breakdown"), (20.0, 40.0, "cooldown"),
                (40.0, 60.5, "drop")]
    report = perfect_report(beat_times, [(0.0, 60.5, "drop")], 63.0)

    built = built_entry(report, sections)
    gated = built["spaces"][GATED_SPACE]

    for metric in GUARDED_METRICS:
        assert built[metric] == gated[metric], metric
    for fact in ("label_boundaries", "changes_class"):
        assert built[fact] == gated[fact], fact


def test_the_gated_numbers_at_the_top_of_an_entry_are_the_gated_truths_numbers():
    """Same rule one axis over: the top level names one ANNOTATOR, not a mix,
    and the other reading sits beside it where no gate reads it."""
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    owner = [(0.0, 20.0, "intro"), (20.0, 40.0, "bridge"), (40.0, 60.5, "outro")]
    report = perfect_report(beat_times, [(0.0, 40.0, "drop"),
                                         (40.0, 60.5, "atmospheric")], 63.0)

    built = built_entry(report, sections, owner)
    gated = built["truths"][GATED_TRUTH]

    for metric in GUARDED_METRICS:
        assert built[metric] == gated[metric], metric
    for fact in COUNT_FACTS:
        if fact == "beats":
            continue
        assert built[fact] == gated[fact], fact
    assert built["spaces"] == gated["spaces"]
    assert built["truths"][OWNER]["macro_f1"] != built["macro_f1"]


def test_beats_and_the_checksum_are_facts_about_the_run_not_the_reading():
    """A ground truth cannot move the beat stream or the report bytes.  If one
    of these ever sits inside a truth block, a relabelling would look like a
    pipeline change."""
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    owner = [(0.0, 60.5, "bridge")]
    report = perfect_report(beat_times, [(0.0, 60.5, "drop")], 63.0)

    built = built_entry(report, sections, owner)

    assert built["beats"] == len(beat_times)
    for truth, block in built["truths"].items():
        assert "beats" not in block and "checksum" not in block, truth
        assert "song_sec" not in block, truth


def test_a_track_with_no_owner_override_scores_identically_under_both_truths():
    """Eight of the ten are the same annotation read twice, so every number and
    every count fact must match -- a difference means the owner half picked up a
    track it should not have."""
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    report = perfect_report(beat_times, [(0.0, 40.0, "drop")], 63.0)

    built = built_entry(report, sections)

    assert built["truths"][OWNER] == built["truths"][PUBLISHED]


def test_the_raw_space_never_sees_fewer_label_boundaries_than_the_fold():
    """The legacy view can only merge classes, so it can only lose boundaries."""
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "breakdown"), (20.0, 40.0, "cooldown"),
                (40.0, 60.5, "drop")]
    scores, _rows, _stats = score_report(
        "t.1", "1", perfect_report(beat_times, [(0.0, 60.5, "drop")], 63.0), sections)

    assert space_facts(scores[RAW9])["label_boundaries"] == 2
    assert space_facts(scores[LEGACY_V1])["label_boundaries"] == 1


def test_the_ungated_granularity_cannot_fail_the_gate_on_its_own():
    """raw9 is REPORTED, not gated -- moving it must not stop a commit yet."""
    baseline = result_document({"a.1": entry()})
    current = result_document({"a.1": entry(
        spaces={space: metrics(macro_f1=0.01) for space in REPORTED_SPACES})})

    assert compare(baseline, current).failed is False


def test_build_document_round_trips_through_json(tmp_path):
    document = result_document({"a.1": entry()})
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    assert compare(load_baseline(path), document).failed is False


def test_build_document_carries_the_gate_settings():
    built = build_document(
        eval_set={"sha256": "abc", "tracks": 1, "youtube_ids": ["1"]},
        pipeline_sha_="x", entries={"a.1": entry()}, aggregate_metrics=metrics(),
    )
    assert built["gate"]["score_tolerance"] == DEFAULT_SCORE_TOLERANCE
    assert built["gate"]["flicker_tolerance"] == DEFAULT_FLICKER_TOLERANCE
    assert built["gate"]["metrics"] == dict(GUARDED_METRICS)
    assert built["space"] and built["stream"] and built["boundary_tolerance_sec"]


def test_corpus_dir_prefers_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert corpus_dir() == tmp_path.resolve()


def test_corpus_dir_falls_back_to_the_repo_or_the_main_worktree(monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    resolved = corpus_dir()
    assert resolved.name == "raveform"
    assert resolved.exists() or resolved == default_data_dir()


def test_crispness_is_gated_downward_like_the_other_scores():
    from run_eval_set import CRISPNESS_TOLERANCE_SEC

    assert CRISPNESS_TOLERANCE_SEC < BOUNDARY_TOLERANCE_SEC
    baseline = result_document({"a.1": entry(crispness=0.60)})
    current = result_document({"a.1": entry(crispness=0.30)})
    outcome = compare(baseline, current)
    assert outcome.failed
    assert any("crispness" in line for line in outcome.regressions)


def test_a_baseline_without_crispness_is_not_gated_rather_than_skipped():
    stale = entry()
    stale.pop("crispness")
    outcome = compare(result_document({"a.1": stale}),
                      result_document({"a.1": entry()}))
    assert outcome.failed
    assert any("crispness" in line and "NOT being gated" in line
               for line in outcome.ungated)


def test_the_committed_baseline_records_the_crispness_tolerance():
    from run_eval_set import CRISPNESS_TOLERANCE_SEC

    assert committed_baseline()["crispness_tolerance_sec"] == CRISPNESS_TOLERANCE_SEC


def test_lateness_is_a_count_fact_rather_than_a_score_with_a_tolerance():
    from run_eval_set import COUNT_FACTS, GUARDED_METRICS

    assert "late" not in GUARDED_METRICS and "late" in COUNT_FACTS
    for track_id, row in committed_baseline()["tracks"].items():
        assert "late" in row and "blocks_measurable" in row, track_id
        assert row["late"] <= row["blocks_measurable"], track_id


def test_a_machine_without_the_model_is_refused_rather_than_scored(tmp_path, monkeypatch):
    import run_eval_set
    from lib import section_chain

    monkeypatch.setattr(section_chain, 'artifacts_present', lambda *a, **k: False)
    refusal = run_eval_set.missing_model(tmp_path)
    assert refusal is not None
    assert 'do not' in refusal and 're-cut' in refusal

    monkeypatch.setattr(section_chain, 'artifacts_present', lambda *a, **k: True)
    assert run_eval_set.missing_model(tmp_path) is None


def test_the_refusal_happens_before_anything_is_simulated(tmp_path, monkeypatch, capsys):
    import run_eval_set
    from lib import section_chain

    monkeypatch.setattr(section_chain, 'artifacts_present', lambda *a, **k: False)

    def never(*args, **kwargs):
        raise AssertionError('a track was simulated on a machine with no model')

    monkeypatch.setattr(run_eval_set, 'run', never)
    code = run_eval_set.main(['--data-dir', str(tmp_path), '--quiet'])
    assert code == 2
    assert 'shipped model is not on this machine' in capsys.readouterr().err


def test_a_blackout_is_counted_exactly_rather_than_scored_with_a_tolerance():
    from run_eval_set import COUNT_FACTS, GUARDED_METRICS

    for fact in ("silence_leading", "silence_interior", "silence_trailing"):
        assert fact in COUNT_FACTS and fact not in GUARDED_METRICS


def test_a_blackout_that_moved_into_labeled_time_fails_even_with_every_score_held():
    baseline = result_document({"a.1": entry(silence_interior=0)})
    current = result_document({"a.1": entry(silence_interior=1)})

    outcome = compare(baseline, current)

    assert outcome.failed and outcome.regressions == []
    assert any("silence_interior" in line for line in outcome.fact_drift)


def test_a_count_fact_missing_from_the_baseline_is_a_failure_not_a_skip():
    stale = entry()
    del stale["silence_interior"]
    baseline = result_document({"a.1": stale})
    current = result_document({"a.1": entry()})

    outcome = compare(baseline, current)

    assert outcome.failed
    assert any("silence_interior" in line for line in outcome.fact_drift)


def test_the_printed_table_shows_the_ungated_granularity_beside_the_gated_one():
    from run_eval_set import TrackRun, render_table

    spaces = {space: dict(metrics(), label_boundaries=7, changes_class=5)
              for space in REPORTED_SPACES}
    run = TrackRun("a.1", "1", entry(spaces=spaces), {}, 3.0)
    text = render_table([run], dict(metrics(), spaces=spaces), 300.0, 3.0)

    for space in REPORTED_SPACES:
        assert space in text
    assert "(aggregate)" in text
    assert text.isascii()


def test_the_printed_table_shows_the_owner_reading_beside_the_gated_one():
    """The owner column is measured and PRINTED from the first commit -- his
    correction is visible before the gate flips, so nothing waits."""
    from run_eval_set import TrackRun, render_table

    truths = {truth: dict(metrics(), label_boundaries=7, rows=90)
              for truth in REPORTED_TRUTHS}
    run = TrackRun("a.1", "1", entry(truths=truths), {}, 3.0)
    text = render_table([run], dict(metrics(), truths=truths), 300.0, 3.0)

    for truth in REPORTED_TRUTHS:
        assert truth in text
    assert GATED_TRUTH in text and "gated one" in text
    assert text.isascii()


def test_the_table_renders_a_row_that_carries_only_the_guarded_metrics():
    """The aggregate's blocks are thinner than a track's, and the table is built
    exactly once, at the END of a sixteen-minute run -- a KeyError there costs
    the whole pass.  Nothing on the display path may require a field."""
    from run_eval_set import TrackRun, render_table

    thin = {truth: metrics() for truth in REPORTED_TRUTHS}
    run = TrackRun("a.1", "1", entry(truths=thin), {}, 3.0)

    text = render_table([run], dict(metrics(), truths=thin), 300.0, 3.0)

    assert "a.1" in text and "(aggregate)" in text


def test_the_benchmark_runs_serial_unless_a_human_asks_otherwise():
    assert build_parser().parse_args([]).workers == 1


def test_parallelism_is_still_reachable_for_a_machine_that_can_afford_it():
    assert build_parser().parse_args(["--workers", "4"]).workers == 4
