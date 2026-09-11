"""Where the corpus is, when the answer is not the obvious one."""
import json

import pytest

import corpus_root


def _worktree(tmp_path, name):
    """A checkout as ``git worktree add`` leaves it: hand labels and nothing else.

    Those labels are the one thing under training/data git tracks, so the
    directory exists on a fresh checkout while holding none of the corpus.
    """
    root = tmp_path / name
    annotations = root / "training" / "data" / "raveform" / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "hand-abc.hand.json").write_text(json.dumps({}), encoding="utf-8")
    return root


def _corpus(root):
    raveform = root / "training" / "data" / "raveform"
    (raveform / "models" / "l9c").mkdir(parents=True)
    (raveform / "manifest.csv").write_text("youtube_id\n", encoding="utf-8")
    return raveform


@pytest.fixture
def linked(tmp_path, monkeypatch):
    main = _worktree(tmp_path, "main")
    corpus = _corpus(main)
    linked_root = _worktree(tmp_path, "linked")
    monkeypatch.delenv(corpus_root.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(corpus_root, "REPO_ROOT", linked_root)
    monkeypatch.setattr(corpus_root, "_git", lambda *a: str(main / ".git"))
    return main, corpus, linked_root


def test_a_linked_worktree_reads_the_main_checkouts_corpus(linked):
    _main, corpus, _linked_root = linked
    assert corpus_root.corpus_dir() == corpus


def test_the_main_checkouts_own_corpus_wins(tmp_path, monkeypatch):
    main = _worktree(tmp_path, "main")
    corpus = _corpus(main)
    monkeypatch.delenv(corpus_root.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(corpus_root, "REPO_ROOT", main)
    monkeypatch.setattr(corpus_root, "_git", lambda *a: str(main / ".git"))
    assert corpus_root.corpus_dir() == corpus


def test_the_environment_override_outranks_both(linked, monkeypatch, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv(corpus_root.DATA_DIR_ENV, str(elsewhere))
    assert corpus_root.corpus_dir() == elsewhere


def test_no_corpus_anywhere_still_names_this_checkout(tmp_path, monkeypatch):
    linked_root = _worktree(tmp_path, "linked")
    monkeypatch.delenv(corpus_root.DATA_DIR_ENV, raising=False)
    monkeypatch.setattr(corpus_root, "REPO_ROOT", linked_root)
    monkeypatch.setattr(corpus_root, "_git", lambda *a: None)
    assert corpus_root.corpus_dir() == linked_root / "training" / "data" / "raveform"
