"""Tests for the committed eval-set artifacts (training/eval_assets.py).

The ten mp3s live in the repository under names DERIVED from their YouTube ids,
so there is no lookup table anywhere -- which means the derivation itself is the
contract.  If ``opaque_name`` ever returns something else, every committed file
is orphaned in silence: the resolver falls through to a corpus nobody has, the
integration suite fails with a download hint, and nothing says the cause was a
one-character change here.  Hence the golden literal below.

The rest pins what ``--cut`` promises: the committed bytes ARE the corpus bytes,
and the label slice is a verbatim subset of the annotation it records.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from eval_assets import (  # noqa: E402  (needs the path insert above)
    AUDIO_NAME_SCHEME,
    EVAL_AUDIO_DIR,
    EVAL_LABELS_FILE,
    NAME_CHARS,
    artifacts_block,
    build_labels_document,
    committed_audio_path,
    copy_audio,
    cut_labels,
    labels_source_sha,
    load_labels,
    opaque_name,
    sections_by_track,
    verify,
)
from select_eval_set import EVAL_SET_FILE, load_eval_set  # noqa: E402

EVAL_SET = load_eval_set(EVAL_SET_FILE)


# --------------------------------------------------------------------------- #
# The derivation
# --------------------------------------------------------------------------- #


def test_opaque_name_is_stable():
    """The golden literal.  Recomputing the expression here instead would test
    that Python still works, not that the committed filenames still resolve."""
    assert opaque_name("PNpXKsge4xM") == "pTxPTSeC8F"


def test_opaque_name_is_the_documented_expression():
    """The scheme is recorded in eval_set.json's provenance block, where a human
    reproduces it by hand.  This is the check that the record is true."""
    import base64

    for youtube_id in EVAL_SET["youtube_ids"]:
        digest = hashlib.sha256(youtube_id.encode("utf-8")).digest()
        assert opaque_name(youtube_id) == (
            base64.urlsafe_b64encode(digest).decode("ascii")[:NAME_CHARS])
    assert f"[:{NAME_CHARS}]" in AUDIO_NAME_SCHEME


def test_derived_names_are_filename_safe_and_unique():
    names = [committed_audio_path(youtube_id).name
             for youtube_id in EVAL_SET["youtube_ids"]]
    assert len(set(names)) == len(names)
    for name in names:
        assert name.endswith(".mp3") and len(name) == NAME_CHARS + 4
        assert not set(name) & set('/\\:*?"<>| ')


def test_derived_names_leak_nothing_about_the_track():
    """Opacity is the other half of the point: a directory listing must not tell
    a reader which YouTube ids -- or which songs -- are in the repository."""
    for track in EVAL_SET["tracks"]:
        name = committed_audio_path(track["youtube_id"]).name
        assert track["youtube_id"] not in name
        for word in track["title"].split():
            if len(word) > 3:
                assert word.lower() not in name.lower()


def test_the_provenance_block_records_the_scheme_not_a_mapping():
    block = artifacts_block()
    assert block["audio_name_scheme"] == AUDIO_NAME_SCHEME
    assert block["audio_dir"] == "training/eval_audio"
    assert block["labels"] == "training/eval_labels.json"
    for youtube_id in EVAL_SET["youtube_ids"]:
        assert youtube_id not in json.dumps(block)


def test_the_eval_set_records_the_scheme_this_code_implements():
    """A provenance block that drifts from the code is worse than none: it is a
    documented way to compute the wrong filename."""
    assert EVAL_SET["artifacts"] == artifacts_block()


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #


def fake_corpus(tmp_path: Path, records: list) -> Path:
    (tmp_path / "audio").mkdir(parents=True, exist_ok=True)
    (tmp_path / "annotations").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "annotations" / "segments.json", "w",
              encoding="utf-8") as handle:
        json.dump(records, handle)
    return tmp_path


def test_copy_audio_writes_derived_names_and_verifies_the_bytes(tmp_path):
    data_dir = fake_corpus(tmp_path, [])
    (data_dir / "audio" / "abc.mp3").write_bytes(b"pretend mp3")
    destination = tmp_path / "committed"

    copied = copy_audio(data_dir, ["abc"], audio_dir=destination)

    (youtube_id, path, sha) = copied[0]
    assert youtube_id == "abc"
    assert path == destination / f"{opaque_name('abc')}.mp3"
    assert path.read_bytes() == b"pretend mp3"
    assert sha == hashlib.sha256(b"pretend mp3").hexdigest()


def test_copy_audio_refuses_a_track_the_corpus_does_not_have(tmp_path):
    data_dir = fake_corpus(tmp_path, [])
    with pytest.raises(RuntimeError, match="missing corpus audio"):
        copy_audio(data_dir, ["gone"], audio_dir=tmp_path / "committed")


def test_cut_labels_takes_the_records_verbatim(tmp_path):
    """Verbatim, so the slice is provably a SUBSET of the annotation rather than
    a re-derivation nobody can check against the source."""
    wanted = {"key": "a.1", "id": "1", "sections": [
        {"name": "intro", "start": 0.0, "end": 10.0},
        {"name": "drop", "start": 10.0, "end": 20.0}]}
    other = {"key": "z.9", "id": "9", "sections": []}
    data_dir = fake_corpus(tmp_path, [wanted, other])
    eval_set = {"tracks": [{"track_id": "a.1", "youtube_id": "1"}]}

    document = cut_labels(data_dir, eval_set, path=tmp_path / "labels.json")

    assert document["tracks"] == [wanted]
    assert document["source"]["tracks_in_source"] == 2
    assert sections_by_track(document) == {
        "a.1": [(0.0, 10.0, "intro"), (10.0, 20.0, "drop")]}


def test_cut_labels_records_the_sha_of_the_annotation_it_cut(tmp_path):
    data_dir = fake_corpus(tmp_path, [{"key": "a.1", "id": "1", "sections": []}])
    eval_set = {"tracks": [{"track_id": "a.1", "youtube_id": "1"}]}

    document = cut_labels(data_dir, eval_set, path=tmp_path / "labels.json")

    on_disk = (data_dir / "annotations" / "segments.json").read_bytes()
    assert labels_source_sha(document) == hashlib.sha256(on_disk).hexdigest()


def test_cut_labels_refuses_an_eval_track_with_no_annotation(tmp_path):
    data_dir = fake_corpus(tmp_path, [{"key": "a.1", "id": "1", "sections": []}])
    eval_set = {"tracks": [{"track_id": "a.1", "youtube_id": "1"},
                           {"track_id": "b.2", "youtube_id": "2"}]}
    with pytest.raises(RuntimeError, match="b.2"):
        cut_labels(data_dir, eval_set, path=tmp_path / "labels.json")


def test_load_labels_rejects_a_document_that_is_not_a_slice(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"schema": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a label slice"):
        load_labels(path)


def test_load_labels_explains_itself_when_absent(tmp_path):
    with pytest.raises(RuntimeError, match=r"--cut"):
        load_labels(tmp_path / "nope.json")


def test_build_labels_document_is_json_round_trippable():
    document = build_labels_document([{"key": "a.1", "sections": []}], "sha", 1)
    assert json.loads(json.dumps(document)) == document


# --------------------------------------------------------------------------- #
# What is actually committed
# --------------------------------------------------------------------------- #


def test_verify_passes_on_the_committed_artifacts():
    assert verify(EVAL_SET) == []


def test_verify_names_a_pruned_mp3(tmp_path):
    problems = verify(EVAL_SET, audio_dir=tmp_path)
    assert len(problems) == len(EVAL_SET["youtube_ids"])
    assert all("missing committed audio" in problem for problem in problems)


def test_the_committed_slice_is_readable_and_covers_the_frozen_set():
    sections = sections_by_track(load_labels(EVAL_LABELS_FILE))
    assert set(sections) == {track["track_id"] for track in EVAL_SET["tracks"]}
    for track_id, spans in sections.items():
        assert spans, track_id
        assert all(end > start for start, end, _label in spans), track_id


def test_the_committed_slice_is_checked_out_with_canonical_line_endings():
    """Its sha256 is not recorded anywhere, but the sha it RECORDS is compared
    to the eval set's, and a CRLF rewrite of a JSON document is exactly the kind
    of platform accident that has already broken the frozen artifacts once
    (see .gitattributes)."""
    assert b"\r" not in EVAL_LABELS_FILE.read_bytes()


def test_the_committed_audio_dir_holds_only_the_eval_set():
    """Precisely ten tracks, and nothing else: the owner authorised committing
    the eval set, not a habit of committing audio."""
    expected = {committed_audio_path(youtube_id).name
                for youtube_id in EVAL_SET["youtube_ids"]}
    assert {path.name for path in EVAL_AUDIO_DIR.glob("*.mp3")} == expected
