"""Tests for the eval-set benchmark runner (training/run_eval_set.py).

The runner is the successor of the old plumbing-only PASS gate: it is the thing
that says "the pipeline still behaves, and it still behaves *well*".  Two
promises carry that weight and both are pinned here without touching audio:

* **the gate actually gates** -- a changed report checksum, a dropped score, or
  a baseline cut against a different eval set must each fail loudly and by
  name, and a run inside tolerance must not;
* **the score wiring is the corpus wiring** -- the runner reuses
  ``build_training_table.join_track`` (and therefore ``realign_intents``) and
  ``evaluate_against_labels.score_track``, so a report whose intents match the
  annotation exactly scores 1.0.  If the look-ahead realignment were dropped or
  duplicated here, that number would not be 1.0.

Track selection is pinned too: the integration test asks for the *shortest*
tracks by name, so "shortest" has to be a deterministic function of the frozen
document rather than of whatever order the corpus happens to be in.
"""
import json
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from run_eval_set import (  # noqa: E402  (needs the path insert above)
    AUDIO_MISSING_HINT,
    BASELINE_FILE,
    DATA_DIR_ENV,
    DEFAULT_FLICKER_TOLERANCE,
    DEFAULT_SCORE_TOLERANCE,
    EVAL_SET_FILE,
    GUARDED_METRICS,
    audio_path,
    build_document,
    build_jobs,
    compare,
    corpus_dir,
    default_data_dir,
    file_sha256,
    load_baseline,
    load_eval_set,
    missing_inputs,
    partial_baseline_refusal,
    score_report,
    select_tracks,
    shortest_track_ids,
    track_metrics,
)

LOOK_AHEAD = 2.5


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def eval_document(*specs) -> dict:
    """A minimal frozen-eval-set document: ``(track_id, youtube_id, duration)``."""
    return {
        "youtube_ids": [youtube for _track, youtube, _duration in specs],
        "tracks": [
            {"track_id": track, "youtube_id": youtube, "duration_sec": duration}
            for track, youtube, duration in specs
        ],
    }


def metrics(macro_f1=0.5, accuracy=0.5, boundary_f1=0.5, flicker_per_min=1.0) -> dict:
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "boundary_f1": boundary_f1,
        "flicker_per_min": flicker_per_min,
    }


def result_document(tracks: dict, aggregate: dict | None = None,
                    eval_sha: str = "abc123") -> dict:
    """A baseline/result document in the shape ``build_document`` emits."""
    return {
        "eval_set": {"sha256": eval_sha, "tracks": len(tracks),
                     "youtube_ids": sorted(tracks)},
        "pipeline_sha": "deadbeef",
        "aggregate": aggregate if aggregate is not None else metrics(),
        "tracks": tracks,
    }


def entry(checksum="cafe", **overrides) -> dict:
    record = {"youtube_id": "yt", "checksum": checksum, "beats": 100, "rows": 90,
              "song_sec": 300.0, "exposure_sec": 280.0,
              "changes_intent": 20, "changes_class": 15}
    record.update(metrics())
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- #
# Track selection
# --------------------------------------------------------------------------- #


def test_shortest_track_ids_takes_the_shortest_by_duration():
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0),
                             ("c.3", "3", 300.0), ("d.4", "4", 470.0))
    assert shortest_track_ids(document, 2) == ["b.2", "c.3"]


def test_shortest_track_ids_keeps_eval_set_order():
    """The selection is by duration; the ORDER it is returned in is the frozen
    document's, so a subset run reads like a prefix of the full run."""
    document = eval_document(("a.1", "1", 300.0), ("b.2", "2", 250.0),
                             ("c.3", "3", 400.0))
    assert shortest_track_ids(document, 2) == ["a.1", "b.2"]


def test_shortest_track_ids_breaks_duration_ties_by_id():
    """Two tracks of identical length must not depend on dict/file order --
    the integration test names the result, so it has to be a function."""
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


# --------------------------------------------------------------------------- #
# Missing inputs
# --------------------------------------------------------------------------- #


def test_missing_inputs_names_every_absent_mp3_and_the_download_hint(tmp_path):
    document = eval_document(("a.1", "1", 400.0), ("b.2", "2", 250.0))
    (tmp_path / "audio").mkdir()
    audio_path(tmp_path, "1").write_bytes(b"not really an mp3")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "segments.json").write_text("[]", encoding="utf-8")

    problems = missing_inputs(tmp_path, select_tracks(document))
    assert len(problems) == 1
    assert "b.2" in problems[0] and AUDIO_MISSING_HINT in problems[0]


def test_missing_inputs_is_empty_when_everything_is_there(tmp_path):
    document = eval_document(("a.1", "1", 400.0))
    (tmp_path / "audio").mkdir()
    audio_path(tmp_path, "1").write_bytes(b"x")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "segments.json").write_text("[]", encoding="utf-8")
    assert missing_inputs(tmp_path, select_tracks(document)) == []


def test_build_jobs_refuses_a_track_with_no_annotation(tmp_path):
    """The audio can be there and the annotation missing; scoring an unlabeled
    track would produce a number with nothing behind it."""
    document = eval_document(("a.1", "1", 400.0))
    with pytest.raises(RuntimeError, match="no annotation"):
        build_jobs(tmp_path, select_tracks(document), {})


def test_build_jobs_gives_each_worker_only_its_own_sections(tmp_path):
    """The corpus map is every annotated track; shipping it per job would cost
    more than the simulation it parallelises."""
    document = eval_document(("a.1", "1", 400.0))
    sections = {"a.1": [(0.0, 10.0, "intro")], "other.2": [(0.0, 5.0, "drop")]}
    jobs = build_jobs(tmp_path, select_tracks(document), sections)
    assert [job.sections for job in jobs] == [sections["a.1"]]


def test_missing_inputs_reports_a_missing_annotation_file(tmp_path):
    document = eval_document(("a.1", "1", 400.0))
    (tmp_path / "audio").mkdir()
    audio_path(tmp_path, "1").write_bytes(b"x")
    problems = missing_inputs(tmp_path, select_tracks(document))
    assert any("segments.json" in problem for problem in problems)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


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
    """Every other guarded metric is better when larger; flicker is changes the
    audience had no reason for, so MORE of it is the regression."""
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


def test_every_guarded_metric_is_actually_checked():
    """A metric added to the table but forgotten in the gate would be silent."""
    for metric, direction in GUARDED_METRICS.items():
        worse = 0.5 - 5.0 if direction == "down" else 0.5 + 5.0
        baseline = result_document({"a.1": entry(**{metric: 0.5})})
        current = result_document({"a.1": entry(**{metric: worse})})
        outcome = compare(baseline, current)
        assert any(metric in line for line in outcome.regressions), metric


# --------------------------------------------------------------------------- #
# Subsets and desync
# --------------------------------------------------------------------------- #


def test_a_subset_run_compares_only_its_own_tracks():
    """The integration test runs three tracks against the ten-track baseline:
    the missing seven are not failures, and the subset's aggregate is NOT the
    baseline's, so it must not be compared."""
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
    """select_eval_set --force re-freezes the benchmark; the baseline then
    describes tracks that may no longer be in it, and the gate would be lying."""
    baseline = result_document({"a.1": entry()}, eval_sha="OLD")
    current = result_document({"a.1": entry()}, eval_sha="NEW")
    outcome = compare(baseline, current)
    assert outcome.failed
    assert outcome.desync and "OLD"[:6] in outcome.desync[0]


def test_a_drift_with_improved_scores_still_fails():
    """The gate is not "did it get worse", it is "did it change".  A behaviour
    change that happens to score better is still a change the operator has to
    look at and accept -- silently passing it would mean the committed
    checksums stop describing the pipeline that produced them."""
    baseline = result_document({"a.1": entry(checksum="cafe", macro_f1=0.20,
                                             flicker_per_min=4.0)})
    current = result_document({"a.1": entry(checksum="f00d", macro_f1=0.60,
                                            flicker_per_min=1.0)})
    outcome = compare(baseline, current)
    assert outcome.failed
    assert outcome.checksum_drift
    assert outcome.regressions == []


def test_a_guarded_metric_missing_from_the_baseline_is_a_failure_not_a_skip():
    """Skipping it would un-gate that number silently: an old-schema baseline
    would keep passing while the metric it protects drifted anywhere."""
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


# --------------------------------------------------------------------------- #
# The committed baseline itself -- the gate guarding the gate
# --------------------------------------------------------------------------- #
#
# Everything above tests the comparison against a SYNTHETIC baseline.  These
# read the real committed file, because every one of these properties can be
# broken without any test noticing: the benchmark would keep printing a table
# and keep exiting 0 while covering fewer tracks, describing a set that has
# since been re-frozen, or leaving a metric un-gated.  They need no audio and
# run in the fast suite.


def committed_baseline() -> dict:
    return load_baseline(BASELINE_FILE)


def test_the_committed_baseline_covers_the_whole_frozen_eval_set():
    """A baseline cut from a subset run passes by construction: the gate
    compares the tracks it ran against the tracks in the file, so the missing
    ones simply stop being checked.  This is the tripwire on that."""
    frozen = {track["track_id"] for track in load_eval_set(EVAL_SET_FILE)["tracks"]}
    assert set(committed_baseline()["tracks"]) == frozen


def test_the_committed_baseline_was_cut_against_the_current_eval_set():
    """`select_eval_set --force` re-freezes the benchmark and desynchronizes the
    baseline.  The runner detects it at run time; this detects it in the fast
    suite, without the corpus."""
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


def test_the_committed_baseline_records_the_configuration_it_was_cut_under():
    baseline = committed_baseline()
    assert baseline["gate"]["metrics"] == dict(GUARDED_METRICS)
    assert baseline["space"] and baseline["stream"]


def test_the_committed_baseline_compares_clean_against_itself():
    """A whole-file smoke test of the comparison against real data: whatever
    else changes, the committed file must at least be self-consistent."""
    baseline = committed_baseline()
    assert compare(baseline, baseline).failed is False


# --------------------------------------------------------------------------- #
# Refusing to shrink the benchmark
# --------------------------------------------------------------------------- #


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
    """Experiments are fine -- no gate reads them."""
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
# Score wiring (no audio: a synthetic report against a synthetic annotation)
# --------------------------------------------------------------------------- #


def beat(t: float) -> dict:
    return {"t": t, "bpm": 128.0, "onset_density": 4.0, "kick_strength": 2.0,
            "centroid_trend": 1.0, "sub_bass_ratio": 0.3, "rms": 0.1}


def perfect_report(beat_times, blocks, duration_sec: float) -> dict:
    """A report whose committed intents match the annotation exactly.

    Intent blocks are stamped in AUDIENCE time (``t = beat + look-ahead``), the
    way the delayed command queue stamps them, so the runner has to realign
    them before joining or the score collapses.
    """
    return {
        "duration_sec": duration_sec,
        "beats": [beat(t) for t in beat_times],
        "intents": [{"t": start + LOOK_AHEAD, "intent": intent,
                     "end": end + LOOK_AHEAD}
                    for start, end, intent in blocks],
        "metrics": {"look_ahead_sec": LOOK_AHEAD},
    }


def test_a_report_matching_the_annotation_scores_one():
    beat_times = [0.5 * index for index in range(1, 121)]      # 0.5 .. 60.0
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    blocks = [(0.0, 20.0, "atmospheric"), (20.0, 40.0, "drop"),
              (40.0, 60.5, "atmospheric")]
    score, rows = score_report("t.1", "1", perfect_report(beat_times, blocks, 63.0),
                               sections)
    assert rows == len(beat_times)
    result = track_metrics(score)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["boundary_f1"] == pytest.approx(1.0)
    assert result["flicker_per_min"] == pytest.approx(0.0)


def test_a_constant_intent_scores_far_below_one():
    """The 1.0 above must come from the alignment, not from a metric that
    cannot fail."""
    beat_times = [0.5 * index for index in range(1, 121)]
    sections = [(0.0, 20.0, "intro"), (20.0, 40.0, "drop"), (40.0, 60.5, "outro")]
    blocks = [(0.0, 60.5, "drop")]
    score, _rows = score_report("t.1", "1", perfect_report(beat_times, blocks, 63.0),
                                sections)
    assert track_metrics(score)["macro_f1"] < 0.5


def test_build_document_round_trips_through_json(tmp_path):
    """The baseline is committed: it must be plain JSON with no float that
    changes shape between write and read."""
    document = result_document({"a.1": entry()})
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    assert compare(load_baseline(path), document).failed is False


def test_build_document_carries_the_gate_settings():
    """The baseline records the configuration it was cut under: a file read
    under different tolerances or a different label space is a file lying."""
    built = build_document(
        eval_set={"sha256": "abc", "tracks": 1, "youtube_ids": ["1"]},
        pipeline_sha_="x", entries={"a.1": entry()}, aggregate_metrics=metrics(),
    )
    assert built["gate"]["score_tolerance"] == DEFAULT_SCORE_TOLERANCE
    assert built["gate"]["flicker_tolerance"] == DEFAULT_FLICKER_TOLERANCE
    assert built["gate"]["metrics"] == dict(GUARDED_METRICS)
    assert built["space"] and built["stream"] and built["boundary_tolerance_sec"]


def test_corpus_dir_prefers_the_environment(tmp_path, monkeypatch):
    """A linked git worktree has no corpus of its own; the override is how it
    reaches the single gitignored copy."""
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert corpus_dir() == tmp_path.resolve()


def test_corpus_dir_falls_back_to_the_repo_or_the_main_worktree(monkeypatch):
    """Without the override it must resolve to a corpus that is actually there,
    or -- on a machine that has none -- to the repo-local path, so the error
    message names somewhere a human recognises."""
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    resolved = corpus_dir()
    assert resolved.name == "raveform"
    assert resolved.exists() or resolved == default_data_dir()
