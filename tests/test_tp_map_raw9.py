"""Tests for the raw-9 draft mapper (``training/third_party/tp_map_raw9.py``).

The pure mapping logic is what is pinned: the label rule table, the measured
chorus threshold's application, masked handling (verse and friends never get a
raw-9 label), the end-sentinel rule, and the spot-check seed's fidelity to the
labelling tool's working format.  The ``.mapped.json`` suffix must stay as
invisible to the corpus loaders as the ``.<source>.json`` quarantine it sits
beside.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
for _path in (str(REPO_ROOT), str(TRAINING_DIR), str(TRAINING_DIR / "raveform"),
              str(TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tp_map_raw9 as mapper  # noqa: E402
import tp_record  # noqa: E402
from build_training_table import load_sections_by_track  # noqa: E402
from raveform_fetch_annotations import load_all_tracks, load_hand_tracks  # noqa: E402

from tests.test_third_party_admission import (  # noqa: E402
    hand_record,
    make_corpus,
    published_record,
)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_normalise_label_strips_variant_letters_digits_and_compounds():
    assert mapper.normalise_label("chorus A") == "chorus"
    assert mapper.normalise_label("Chorus2") == "chorus"
    assert mapper.normalise_label("chorus, fade-out") == "chorus"
    assert mapper.normalise_label("pre-chorus") == "prechorus"
    assert mapper.normalise_label("PreChorus5") == "prechorus"
    assert mapper.normalise_label("fade-out, out") == "fadeout"
    assert mapper.normalise_label("main_theme, theme") == "maintheme"
    assert mapper.normalise_label("intropt2") == "intropt"
    assert mapper.normalise_label("verse_slow") == "verseslow"
    assert mapper.normalise_label("bridge b") == "bridge"


# --------------------------------------------------------------------------- #
# The rule table
# --------------------------------------------------------------------------- #


def test_honest_classes_map_direct():
    assert mapper.rule_for("intro", final=False) == (mapper.RULE_DIRECT, "intro")
    assert mapper.rule_for("outro", final=False) == (mapper.RULE_DIRECT, "outro")
    assert mapper.rule_for("ending", final=True) == (mapper.RULE_DIRECT, "outro")
    assert mapper.rule_for("bridge", final=False) == (mapper.RULE_DIRECT, "bridge")
    assert mapper.rule_for("breakdown", final=False) == (
        mapper.RULE_DIRECT, "breakdown")


def test_build_flavoured_labels_map_to_buildup():
    for canonical in ("prechorus", "build", "buildup", "transition"):
        assert mapper.rule_for(canonical, final=False) == (
            mapper.RULE_DIRECT, "buildup")


def test_chorus_family_routes_to_the_evidence_split():
    for canonical in ("chorus", "quietchorus", "instchorus", "refrain"):
        rule, target = mapper.rule_for(canonical, final=False)
        assert target == "chorus"


def test_dishonest_labels_are_masked_never_guessed():
    for canonical in ("verse", "solo", "inst", "instrumental", "silence",
                      "postchorus", "interlude", "head", "maintheme",
                      "nothing", "gtr", "unheard_of_label"):
        assert mapper.rule_for(canonical, final=False) == (
            mapper.RULE_NO_TARGET, mapper.MASKED)


def test_the_end_sentinel_is_masked_not_a_class():
    assert mapper.rule_for("end", final=True) == (mapper.RULE_END, mapper.MASKED)


def test_tail_functions_are_outro_only_at_the_tail():
    assert mapper.rule_for("fadeout", final=True) == (
        mapper.RULE_TAIL_OUTRO, "outro")
    assert mapper.rule_for("fadeout", final=False) == (
        mapper.RULE_TAIL_MASKED, mapper.MASKED)
    assert mapper.rule_for("coda", final=True) == (
        mapper.RULE_TAIL_OUTRO, "outro")


# --------------------------------------------------------------------------- #
# The chorus threshold
# --------------------------------------------------------------------------- #


def test_chorus_at_or_above_threshold_is_drop_below_is_breakdown():
    assert mapper.decide_chorus(5.0, 4.0) == (mapper.RULE_CHORUS_DROP, "drop")
    assert mapper.decide_chorus(4.0, 4.0) == (mapper.RULE_CHORUS_DROP, "drop")
    assert mapper.decide_chorus(3.9, 4.0) == (
        mapper.RULE_CHORUS_BREAKDOWN, "breakdown")


def test_chorus_with_no_entry_evidence_defaults_to_breakdown():
    assert mapper.decide_chorus(None, 4.0) == (
        mapper.RULE_CHORUS_DEFAULT, "breakdown")


def test_derive_thresholds_registers_the_pooled_q3():
    thresholds = mapper.derive_thresholds(
        {"rwc": [0.0, 1.0, 2.0, 3.0], "harmonix": [4.0, 5.0, 6.0, 7.0]})
    assert thresholds["chorus_drop_threshold_db"] == 5.25
    assert thresholds["chorus_entry_contrast_db"]["pooled"]["n"] == 8
    assert thresholds["chorus_entry_contrast_db"]["rwc"]["n"] == 4


# --------------------------------------------------------------------------- #
# map_track
# --------------------------------------------------------------------------- #


def _record(sections, masked=()):
    return {
        "schema": 1, "source": "harmonix", "id": "hx-t", "native_id": "t",
        "title": "T", "artist": "A", "genre": "", "audio": "hx-t.mp3",
        "duration": 100.0, "label_vocabulary": "harmonix_function",
        "sections": sections, "masked": list(masked),
        "provenance": {"converter": "x", "converted_utc": "y"},
    }


def _features(energies):
    return {"sections": [
        {"start": e[0], "end": e[1], "rms_db": e[2], "head_db": e[3],
         "pre_db": e[4]} for e in energies]}


def test_map_track_applies_the_rules_and_logs_the_evidence():
    record = _record([
        {"name": "intro", "start": 0.0, "end": 10.0},
        {"name": "verse", "start": 10.0, "end": 30.0},
        {"name": "chorus", "start": 30.0, "end": 50.0},
        {"name": "chorus", "start": 50.0, "end": 70.0},
    ])
    features = _features([
        (0.0, 10.0, -30.0, -30.0, None),
        (10.0, 30.0, -20.0, -21.0, -29.0),
        (30.0, 50.0, -12.0, -13.0, -19.0),   # +6 contrast -> drop
        (50.0, 70.0, -18.0, -17.0, -18.5),   # +1.5 contrast -> breakdown
    ])
    mapped = mapper.map_track(record, features, threshold_db=4.0)

    labels = [s["label"] for s in mapped["sections"]]
    assert labels == ["intro", "masked", "drop", "breakdown"]
    drop = mapped["sections"][2]["evidence"]
    assert drop["rule"] == mapper.RULE_CHORUS_DROP
    assert drop["entry_contrast_db"] == 6.0
    assert drop["threshold_db"] == 4.0
    assert drop["margin_db"] == 2.0
    assert drop["source_label"] == "chorus"
    verse = mapped["sections"][1]["evidence"]
    assert verse["rule"] == mapper.RULE_NO_TARGET
    assert verse["rms_db"] == -20.0
    assert verse["entry_step_db"] == 10.0


def test_map_track_carries_source_masked_spans_in_time_order():
    record = _record(
        [{"name": "chorus", "start": 10.0, "end": 30.0}],
        masked=[{"start": 0.0, "end": 10.0, "reason": "non_musical",
                 "function": "Silence"}])
    features = _features([(10.0, 30.0, -12.0, -12.0, -40.0)])
    mapped = mapper.map_track(record, features, threshold_db=4.0)

    assert [s["label"] for s in mapped["sections"]] == ["masked", "drop"]
    carried = mapped["sections"][0]["evidence"]
    assert carried["rule"] == mapper.RULE_CARRYOVER
    assert carried["source_label"] == "__masked__(Silence)"


def test_map_track_records_the_registered_threshold():
    record = _record([{"name": "intro", "start": 0.0, "end": 10.0}])
    features = _features([(0.0, 10.0, -30.0, -30.0, None)])
    mapped = mapper.map_track(record, features, threshold_db=3.25)
    assert mapped["thresholds"]["chorus_drop_entry_contrast_db"] == 3.25
    assert mapped["label_vocabulary"] == mapper.MAPPED_VOCABULARY


# --------------------------------------------------------------------------- #
# Spot-check seeding
# --------------------------------------------------------------------------- #


def _mapped(sections):
    return {"sections": sections, "duration": 100.0}


def test_seed_lines_skip_masked_spans_and_keep_the_tool_format():
    mapped = _mapped([
        {"start": 0.04, "end": 10.0, "label": "intro"},
        {"start": 10.0, "end": 30.0, "label": "masked"},
        {"start": 30.0, "end": 50.0, "label": "drop"},
    ])
    assert mapper.seed_lines(mapped) == ["0.000,intro", "30.000,drop"]


def test_seed_lines_open_a_masked_start_with_an_intro_placeholder():
    mapped = _mapped([
        {"start": 0.0, "end": 20.0, "label": "masked"},
        {"start": 20.0, "end": 40.0, "label": "breakdown"},
    ])
    assert mapper.seed_lines(mapped) == ["0.000,intro", "20.000,breakdown"]


def test_seed_lines_are_empty_for_a_fully_masked_track():
    assert mapper.seed_lines(
        _mapped([{"start": 0.0, "end": 50.0, "label": "masked"}])) == []


def test_stage_spot_check_never_overwrites_existing_work(tmp_path):
    corpus = tmp_path / "raveform"
    (corpus / "tmp_labels").mkdir(parents=True)
    existing = corpus / "tmp_labels" / "a.mp3.labels.csv"
    existing.write_text("0.000,drop\n", encoding="utf-8")
    picks = [
        {"id": "hx-a", "audio_path": str(tmp_path / "a.mp3"), "bucket": "x",
         "mapped": _mapped([{"start": 0.0, "end": 10.0, "label": "intro"}])},
        {"id": "hx-b", "audio_path": str(tmp_path / "b.mp3"), "bucket": "x",
         "mapped": _mapped([{"start": 0.0, "end": 10.0, "label": "intro"}])},
    ]

    seeded, skipped = mapper.stage_spot_check(picks, corpus)

    assert [p["id"] for p in seeded] == ["hx-b"]
    assert skipped[0][0] == "hx-a"
    assert existing.read_text(encoding="utf-8") == "0.000,drop\n"
    assert (corpus / "tmp_labels" / "b.mp3.labels.csv").read_text(
        encoding="utf-8") == "0.000,intro\n"


# --------------------------------------------------------------------------- #
# The quarantine holds for the mapped tier too
# --------------------------------------------------------------------------- #


def test_the_corpus_loaders_never_see_a_mapped_draft(tmp_path):
    data_dir = make_corpus(tmp_path, [published_record()], [hand_record()])
    draft = mapper.map_track(
        _record([{"name": "intro", "start": 0.0, "end": 10.0}]),
        _features([(0.0, 10.0, -30.0, -30.0, None)]), threshold_db=4.0)
    path = data_dir / "annotations" / f"hx-t{mapper.MAPPED_SUFFIX}"
    path.write_text(json.dumps(draft), encoding="utf-8")

    assert [t["key"] for t in load_all_tracks(data_dir)] == [
        "0001.native00001", "hand-ab12cd34ef56"]
    assert [r["key"] for r in load_hand_tracks(data_dir)] == [
        "hand-ab12cd34ef56"]
    assert set(load_sections_by_track(data_dir)) == {
        "0001.native00001", "hand-ab12cd34ef56"}


def test_the_mapped_suffix_is_not_a_third_party_record_suffix():
    for source in tp_record.THIRD_PARTY_SOURCES:
        assert tp_record.record_suffix(source) != mapper.MAPPED_SUFFIX


def test_choose_sample_never_picks_a_track_with_nothing_to_seed():
    def stat(track_id, masked_frac, sections):
        return {"id": track_id, "source": "salami", "audio_path": "x",
                "drops": 0, "breakdowns": 0, "masked_frac": masked_frac,
                "mapped": _mapped(sections)}
    fully_masked = stat("salami-1", 1.0,
                        [{"start": 0.0, "end": 50.0, "label": "masked"}])
    seedable = stat("salami-2", 0.8,
                    [{"start": 0.0, "end": 40.0, "label": "masked"},
                     {"start": 40.0, "end": 50.0, "label": "intro"}])

    picks = mapper.choose_sample([fully_masked, seedable])

    assert {p["id"] for p in picks} == {"salami-2"}
