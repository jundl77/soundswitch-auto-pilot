"""The pinned artifacts' BYTES are their identity, and git status cannot see them drift.

``core.autocrlf=true`` -- the Windows installer's default -- smudges LF to CRLF
on checkout for anything git reads as text.  ``.gitattributes`` pins the frozen
benchmark's artifacts to ``eol=lf`` so that stops happening, but a working copy
materialised BEFORE its path gained a rung stays CRLF for ever: git normalises
on read, so ``git status`` reports the tree clean while the bytes on disk are not
the bytes the sha pins were taken over.  The benchmark then refuses to score for
a reason nothing on screen names.

So the working copy is under test, not just the committed blob.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import run_eval_set  # noqa: E402  (needs the path insert above)
from eval_assets import OVERRIDDEN, OWNER, crlf_drift  # noqa: E402


def ls_files_eol() -> list:
    """``index eol, working-tree eol, attributes, path`` for every tracked file."""
    out = subprocess.run(["git", "ls-files", "--eol"], cwd=REPO_ROOT,
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        fields, _, path = line.partition("\t")
        index, worktree, attrs = fields.split(None, 2)
        rows.append((index, worktree, attrs.strip(), path))
    return rows


def test_no_pinned_file_is_stale_crlf_in_this_working_copy():
    stale = [path for _, worktree, attrs, path in ls_files_eol()
             if "eol=lf" in attrs and worktree == "w/crlf"]
    paths = " ".join(stale)
    assert not stale, (
        f"{len(stale)} tracked files are LF in git and CRLF on disk -- a working "
        f"copy checked out before .gitattributes pinned them, which git status "
        f"cannot show you.  Every sha taken over their bytes fails here and "
        f"nowhere else.  Repair the working copy (no commit, no index change):"
        f"\n\n  rm {paths} && git checkout -- {paths}\n"
    )


def test_every_tracked_json_under_training_is_pinned_to_lf():
    """The rungs do not recurse, so one per directory is one short of the next one.

    Every committed json under training/ is a frozen artifact, a recorded
    measurement or a shipped config -- the kinds of file whose bytes become
    load-bearing without anyone revisiting .gitattributes.
    """
    unpinned = [path for _, _, attrs, path in ls_files_eol()
                if path.startswith("training/") and path.endswith(".json")
                and "eol=lf" not in attrs]
    assert not unpinned, (
        f".gitattributes pins no eol for {unpinned} -- add the rung that matches "
        f"it, `training/*.json` does not match a subdirectory"
    )


def test_a_crlf_working_copy_is_diagnosed_rather_than_reported_as_a_mismatch(tmp_path):
    label = tmp_path / "x.hand.json"
    payload = b'{"sections": []}\n'
    label.write_bytes(payload.replace(b"\n", b"\r\n"))

    drift = crlf_drift(label, hashlib.sha256(payload).hexdigest())

    assert drift and "CRLF" in drift
    assert "git status" in drift
    assert f"rm {label.as_posix()} && git checkout -- {label.as_posix()}" in drift


def test_a_genuinely_edited_file_is_not_blamed_on_line_endings(tmp_path):
    label = tmp_path / "x.hand.json"
    label.write_bytes(b'{"sections": [1]}\r\n')

    assert crlf_drift(label, hashlib.sha256(b'{"sections": []}\n').hexdigest()) is None


def test_the_ground_truth_check_names_the_cause_and_the_remedy(tmp_path, monkeypatch):
    """The real shape: committed blob LF, working copy CRLF, git status clean."""
    name = "training/data/raveform/annotations/x.hand.json"
    label = tmp_path / name
    label.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"sections": []}).encode("utf-8") + b"\n"
    label.write_bytes(payload.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(run_eval_set, "REPO_ROOT", tmp_path)

    with pytest.raises(RuntimeError) as excinfo:
        run_eval_set.verify_owner_ground_truth({"truths": {OWNER: {"rulings": {
            "x": {"ruling": OVERRIDDEN, "file": name,
                  "sha256": hashlib.sha256(payload).hexdigest()},
        }}}})

    message = str(excinfo.value)
    assert "CRLF" in message and "git checkout --" in message
    assert "eval_assets.py --cut" not in message


def test_the_eval_set_desync_does_not_advise_re_cutting_a_crlf_working_copy(
        tmp_path, monkeypatch):
    """--write-baseline against a stale working copy would bake the wrong bytes in."""
    payload = json.dumps({"tracks": []}).encode("utf-8") + b"\n"
    frozen = tmp_path / "training" / "eval_set.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_bytes(payload.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(run_eval_set, "REPO_ROOT", tmp_path)

    outcome = run_eval_set.compare(
        {"eval_set": {"sha256": hashlib.sha256(payload).hexdigest()},
         "space": run_eval_set.GATED_SPACE, "truth": run_eval_set.GATED_TRUTH,
         "tracks": {}},
        {"eval_set": {"sha256": "b" * 64, "path": "training/eval_set.json"},
         "tracks": {}},
    )

    assert len(outcome.desync) == 1
    assert "CRLF" in outcome.desync[0]
    assert "--write-baseline" not in outcome.desync[0]
