#!/usr/bin/env python
"""The eval-set artifacts that live IN the repository: ten mp3s and their labels.

The benchmark used to need the corpus.  ``training/eval_set.json`` froze WHICH
ten tracks it runs, but both things a run actually reads -- the audio and the
section labels -- sat in the gitignored ``training/data/raveform`` tree, so a
fresh clone could not run the benchmark at all: ``uv run pytest`` failed its
integration half with a line naming the downloader, and the gate that is
supposed to catch a pipeline behaviour change caught nothing until somebody had
fetched ~90 GB of corpus.  A benchmark that only runs on one laptop is not a
benchmark.  The owner authorised committing the ten tracks (and only those ten)
so that validation runs anywhere from a clone with no downloads.

Two artifacts, both committed:

``training/eval_audio/``
    The ten mp3s, byte-identical to the corpus copies, under DERIVED names --
    ``base64url(sha256(youtube_id))[:10] + ".mp3"``.  Derived rather than
    arbitrary so that code maps an id to its file with no lookup table (a table
    is a thing that goes stale silently), and opaque so the directory listing
    says nothing about what the tracks are.  Ten characters of base64 is 60 bits
    of the digest: collisions are not a concern at ten files, and ``cut``
    refuses one anyway.

``training/eval_labels.json``
    The ten tracks' records, VERBATIM out of ``annotations/segments.json``, plus
    the sha256 of the file they were cut from.  Verbatim so the slice is
    provably a subset rather than a re-derivation, and sha-pinned so it can be
    proved to be a slice of the labels the eval set was frozen against -- the
    check ``run_eval_set.verify_ground_truth`` makes, which is what the corpus
    file's own sha256 used to be for.

    Schema 2 carries a SECOND reading of the same ten tracks beside the first.
    The owner has ruled published labels wrong on benchmark tracks and
    relabelled them, so the slice records what he has said about each of the ten
    (``overridden`` / ``accepted`` / ``unreviewed``) and, for the overridden
    ones, the merge of the published record with his hand label.  ``tracks``
    stays byte-for-byte the published half; the owner half is a separate block
    with its own per-file sha provenance, so neither reading can be mistaken for
    the other and the published pin keeps meaning what it meant.

The naming scheme is recorded in the eval set's ``artifacts`` provenance block
(see ``artifacts_block``) -- the scheme, not a mapping, because the mapping is
computable and a committed one would be a second source of truth.

Re-cutting (only after the eval set is re-frozen)::

    uv run python training/eval_assets.py --cut
    uv run python training/eval_assets.py            # verify what is committed
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    str(Path(__file__).resolve().parent),
    str(REPO_ROOT / "training" / "raveform"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from build_clean_manifest import AUDIO_DIR  # noqa: E402  (needs the path inserts)
from raveform_fetch_annotations import (  # noqa: E402
    HAND_LABEL_SUFFIX,
    SEGMENTS_FILE,
    annotations_dir,
    load_hand_tracks,
    load_tracks,
    merge_hand_tracks,
    parse_sections,
    youtube_id as track_youtube_id,
)

EVAL_AUDIO_DIR = REPO_ROOT / "training" / "eval_audio"
EVAL_LABELS_FILE = REPO_ROOT / "training" / "eval_labels.json"

# Hand labels are the one thing under training/data/ that git tracks, so every
# checkout and every linked worktree materialises its own copy here.
# corpus_root.corpus_dir() resolves a LINKED worktree to the MAIN checkout's
# corpus, which holds whatever that checkout has -- including uncommitted
# labels.  Hashing through it would make the provenance verdict a fact about
# which worktree ran it, so the tracked path is the one the slice pins.
REPO_CORPUS_DIR = REPO_ROOT / "training" / "data" / "raveform"

# How many characters of the base64url digest name a file.  Changing this
# renames every committed mp3, so it is a re-cut, not a tweak.
NAME_CHARS = 10

# The one-line spelling of the derivation, for the provenance block and for
# every error message that has to tell a human where a file should have been.
AUDIO_NAME_SCHEME = f"base64url(sha256(youtube_id))[:{NAME_CHARS}] + '.mp3'"

LABELS_SCHEMA = 2

# The two annotators the same ten tracks can be read against.  PUBLISHED is the
# Raveform annotation the eval set was frozen against; OWNER is that reading
# with the owner's hand labels substituted wherever he has ruled one wrong.
PUBLISHED = "published"
OWNER = "owner"
TRUTHS = (OWNER, PUBLISHED)

# What the owner has said about one frozen track.  Coverage is "how many has he
# ruled on", not "how many did he relabel": for a track he listens to and
# agrees with, ACCEPTED is the decision, and it is worth recording as one.
OVERRIDDEN = "overridden"     # he relabelled it -- a committed *.hand.json
ACCEPTED = "accepted"         # he ruled the published label correct
UNREVIEWED = "unreviewed"     # he has not ruled
RULINGS = (OVERRIDDEN, ACCEPTED, UNREVIEWED)


# --------------------------------------------------------------------------- #
# Line endings
# --------------------------------------------------------------------------- #


def crlf_drift(path: Path, recorded: str, name: str | None = None) -> str | None:
    """Why a sha over a tracked file can fail while ``git status`` reports clean.

    ``core.autocrlf=true`` smudges LF to CRLF on checkout, and a working copy
    materialised before its path gained an ``eol=lf`` rung in ``.gitattributes``
    stays CRLF for ever -- git normalises on read, so nothing shows it.  Proving
    that is what makes the remedy safe to state: LF-normalising the bytes has to
    reproduce the pin exactly, so a genuine edit is never blamed on a platform.

    Plain ``git checkout -- <path>`` is a no-op here -- git believes the file
    matches and skips it -- which is why the remedy names two commands rather
    than the obvious one.
    """
    path = Path(path)
    if not path.is_file():
        return None
    data = path.read_bytes()
    if b"\r\n" not in data:
        return None
    if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != recorded:
        return None
    name = name or path.as_posix()
    return (f"{name} is CRLF on disk and LF in git, which git status cannot "
            f"show you -- repair the working copy with "
            f"`rm {name} && git checkout -- {name}`")


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def opaque_name(youtube_id: str) -> str:
    """The committed stem for one eval track: ``base64url(sha256(id))[:10]``.

    A pure function of the id, so nothing anywhere stores a mapping.  base64url
    rather than hex because ten hex characters is only 40 bits and reads like a
    truncated hash a human might try to look up; base64url is filename-safe on
    every platform this runs on.
    """
    digest = hashlib.sha256(youtube_id.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")[:NAME_CHARS]


def committed_audio_path(youtube_id: str, audio_dir: Path | None = None) -> Path:
    """Where the committed copy of one eval track's mp3 is."""
    return Path(audio_dir or EVAL_AUDIO_DIR) / f"{opaque_name(youtube_id)}.mp3"


def artifacts_block() -> dict:
    """The eval set's record of what is committed alongside it.

    The SCHEME, never the mapping: ten ``id -> filename`` lines in a committed
    document would be a second source of truth that nothing recomputes, and the
    first re-freeze would leave them describing files that are no longer there.
    """
    return {
        "audio_dir": str(EVAL_AUDIO_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "audio_name_scheme": AUDIO_NAME_SCHEME,
        "labels": str(EVAL_LABELS_FILE.relative_to(REPO_ROOT)).replace("\\", "/"),
    }


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def load_labels(path: Path | None = None) -> dict:
    """Read the committed label slice, refusing anything that is not one."""
    path = Path(path or EVAL_LABELS_FILE)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        raise RuntimeError(
            f"missing {path} -- cut it with training/eval_assets.py --cut"
        ) from None
    except ValueError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(document, dict) or not isinstance(document.get("tracks"), list):
        raise RuntimeError(f"{path} has no 'tracks' list -- it is not a label slice")
    schema = document.get("schema")
    if schema != LABELS_SCHEMA:
        raise RuntimeError(
            f"{path} is schema {schema!r}, not {LABELS_SCHEMA} -- a slice now "
            f"carries both annotators' readings of the frozen ten; re-cut it "
            f"with training/eval_assets.py --cut")
    return document


def sections_by_track(document: dict) -> dict:
    """``track_id -> [(start, end, label)]``, exactly as the corpus loader gives.

    Uses the corpus's own ``parse_sections`` rather than a copy of it: the
    committed slice must join to a report the same way the gitignored file does,
    or the benchmark would score two different things depending on which one a
    machine happened to have.
    """
    return {str(track["key"]): parse_sections(track) for track in document["tracks"]}


def sections_by_truth(document: dict) -> dict:
    """``{truth: {track_id -> spans}}`` -- the same ten tracks, both annotators.

    The owner reading is the published one with the overridden tracks replaced.
    Only overridden records are stored, so a track the owner has not relabelled
    is the SAME object under both truths rather than a second copy that can
    drift from the first.
    """
    published = sections_by_track(document)
    owner = dict(published)
    owner.update({str(track["key"]): parse_sections(track)
                  for track in document.get("owner_tracks") or []})
    return {PUBLISHED: published, OWNER: owner}


def published_source(document: dict) -> dict:
    """What the slice records about the annotation its published half came from."""
    return ((document.get("truths") or {}).get(PUBLISHED)) or {}


def labels_source_sha(document: dict) -> str | None:
    """The sha256 of the ``segments.json`` the slice was cut from."""
    return published_source(document).get("sha256")


def owner_rulings(document: dict) -> dict:
    """``track_id -> ruling`` for every frozen track, always all ten of them."""
    return (((document.get("truths") or {}).get(OWNER)) or {}).get("rulings") or {}


def build_labels_document(records: list, source_sha: str, source_tracks: int,
                          rulings: dict | None = None,
                          owner_records: list | None = None) -> dict:
    """The committed label slice: both annotators, then the records.

    One file rather than two because the pair has to be re-cut atomically -- two
    files drift, and there is then no single artifact whose sha pins the pair.
    """
    return {
        "schema": LABELS_SCHEMA,
        "truths": {
            PUBLISHED: {
                "file": f"annotations/{SEGMENTS_FILE}",
                "sha256": source_sha,
                "tracks_in_source": source_tracks,
            },
            OWNER: {"rulings": dict(rulings or {})},
        },
        "tracks": records,
        "owner_tracks": list(owner_records or []),
    }


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hand_label_path(youtube_id: str, root: Path | None = None) -> Path:
    """The TRACKED copy of one track's hand label -- see ``REPO_CORPUS_DIR``."""
    return annotations_dir(Path(root or REPO_CORPUS_DIR)) / f"{youtube_id}{HAND_LABEL_SUFFIX}"


def hand_label_name(youtube_id: str, root: Path | None = None) -> str:
    """The path the slice records: repo-relative, so nothing resolves it against
    a corpus dir that may not be this checkout's."""
    path = hand_label_path(youtube_id, root)
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return path.as_posix()


def hand_records(root: Path | None = None) -> dict:
    """``youtube_id -> record`` for every hand label tracked in this repo."""
    root = Path(root or REPO_CORPUS_DIR)
    if not annotations_dir(root).exists():
        return {}
    return {track_youtube_id(record): record for record in load_hand_tracks(root)}


def carried_ruling(previous: dict | None) -> dict:
    """What a track with no hand label is ruled, given what the slice said before.

    An ``accepted`` ruling is the owner listening to a track and deciding the
    published label is right.  A re-cut must not wipe that -- it is the only
    place the decision is recorded, and losing it silently re-opens a track he
    has already closed.
    """
    if previous and str(previous.get("ruling")) == ACCEPTED:
        return dict(previous)
    return {"ruling": UNREVIEWED}


def corpus_audio_path(data_dir: Path, youtube_id: str) -> Path:
    """Where the downloader parks one track's mp3."""
    return Path(data_dir) / AUDIO_DIR / f"{youtube_id}.mp3"


def copy_audio(data_dir: Path, youtube_ids: list,
               audio_dir: Path | None = None) -> list:
    """Copy each track's mp3 into the committed dir under its derived name.

    Returns ``[(youtube_id, destination, sha256)]``.  Every copy is re-hashed
    from disk and compared to the source: the committed bytes have to BE the
    corpus bytes, or the two paths would decode to different audio and the
    checksum gate would fire on whichever machine used the other one.
    """
    audio_dir = Path(audio_dir or EVAL_AUDIO_DIR)
    audio_dir.mkdir(parents=True, exist_ok=True)

    names = {opaque_name(youtube_id) for youtube_id in youtube_ids}
    if len(names) != len(set(youtube_ids)):
        raise RuntimeError(
            f"two eval tracks derive the same name at {NAME_CHARS} characters "
            f"-- widen NAME_CHARS and re-cut")

    copied = []
    for youtube_id in youtube_ids:
        source = corpus_audio_path(data_dir, youtube_id)
        if not source.exists():
            raise RuntimeError(f"missing corpus audio: {source}")
        destination = committed_audio_path(youtube_id, audio_dir)
        shutil.copyfile(source, destination)
        source_sha, destination_sha = sha256_of(source), sha256_of(destination)
        if source_sha != destination_sha:
            raise RuntimeError(
                f"{destination.name}: copy does not match {source} "
                f"({destination_sha[:12]}... vs {source_sha[:12]}...)")
        copied.append((youtube_id, destination, destination_sha))
    return copied


def cut_labels(data_dir: Path, eval_set: dict, path: Path | None = None,
               hand_root: Path | None = None) -> dict:
    """Write the label slice for the eval set out of the corpus annotations.

    The published records go in VERBATIM.  Re-deriving them into a tidier shape
    would make the slice a transformation nobody can check against the source,
    and the check that matters here is exactly "these are the corpus's own bytes
    for these ten tracks".

    The owner half is derived by the corpus's OWN ``merge_hand_tracks`` rather
    than a re-derivation, so an owner record is provably "published record plus
    hand sections" and nothing invented.
    """
    path = Path(path or EVAL_LABELS_FILE)
    wanted = {track["track_id"] for track in eval_set["tracks"]}
    tracks = load_tracks(Path(data_dir))
    records = [track for track in tracks if str(track.get("key")) in wanted]
    found = {str(record["key"]) for record in records}
    if found != wanted:
        raise RuntimeError(
            f"{SEGMENTS_FILE} has no annotation for: "
            f"{', '.join(sorted(wanted - found))}")

    records = sorted(records, key=lambda record: str(record["key"]))
    by_key = {str(record["key"]): record for record in records}
    hand = hand_records(hand_root)
    try:
        previous = owner_rulings(load_labels(path))
    except RuntimeError:
        previous = {}

    rulings, owner_records = {}, []
    for track in sorted(eval_set["tracks"], key=lambda item: str(item["track_id"])):
        track_id, youtube = str(track["track_id"]), str(track["youtube_id"])
        record = hand.get(youtube)
        if record is None:
            rulings[track_id] = carried_ruling(previous.get(track_id))
            continue
        rulings[track_id] = {
            "ruling": OVERRIDDEN,
            "file": hand_label_name(youtube, hand_root),
            "sha256": sha256_of(hand_label_path(youtube, hand_root)),
        }
        owner_records.append(merge_hand_tracks([by_key[track_id]], [record])[0])

    document = build_labels_document(
        records,
        sha256_of(annotations_dir(Path(data_dir)) / SEGMENTS_FILE),
        len(tracks),
        rulings=rulings,
        owner_records=owner_records,
    )
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    return document


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def verify(eval_set: dict, audio_dir: Path | None = None,
           labels_path: Path | None = None) -> list:
    """Everything about the committed artifacts that is wrong, one line each."""
    problems = []
    for youtube_id in eval_set["youtube_ids"]:
        path = committed_audio_path(youtube_id, audio_dir)
        if not path.exists():
            problems.append(f"missing committed audio: {path}")
    try:
        document = load_labels(labels_path)
    except RuntimeError as exc:
        return problems + [str(exc)]
    have = set(sections_by_track(document))
    want = {track["track_id"] for track in eval_set["tracks"]}
    if have != want:
        problems.append(
            f"the label slice covers {len(have)} of {len(want)} eval tracks; "
            f"missing {', '.join(sorted(want - have)) or '(none)'}")
    if not labels_source_sha(document):
        problems.append("the label slice records no source sha256")
    ruled = set(owner_rulings(document))
    if ruled != want:
        problems.append(
            f"the owner rulings name {len(ruled)} of {len(want)} eval tracks; "
            f"missing {', '.join(sorted(want - ruled)) or '(none)'} -- every "
            f"frozen track carries a ruling, even an unreviewed one")
    return problems


def main(argv: list | None = None) -> int:
    from select_eval_set import EVAL_SET_FILE, load_eval_set

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="corpus root to cut from (default: the resolved one)")
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET_FILE)
    parser.add_argument("--cut", action="store_true",
                        help="re-copy the audio and re-cut the label slice")
    args = parser.parse_args(argv)

    eval_set = load_eval_set(Path(args.eval_set))

    if args.cut:
        from run_eval_set import corpus_dir

        data_dir = Path(args.data_dir) if args.data_dir else corpus_dir()
        print(f"corpus: {data_dir}")
        for youtube_id, path, sha in copy_audio(data_dir, eval_set["youtube_ids"]):
            print(f"  {youtube_id:<14} -> {path.name}  {sha[:16]}...")
        document = cut_labels(data_dir, eval_set)
        print(f"labels: {EVAL_LABELS_FILE.name} "
              f"({len(document['tracks'])} tracks, source "
              f"{labels_source_sha(document)[:16]}...)")
        tally = collections.Counter(str(ruling.get("ruling"))
                                    for ruling in owner_rulings(document).values())
        print(f"owner : {', '.join(f'{tally[name]} {name}' for name in RULINGS)}")

    problems = verify(eval_set)
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"committed artifacts OK: {len(eval_set['youtube_ids'])} mp3s + labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
