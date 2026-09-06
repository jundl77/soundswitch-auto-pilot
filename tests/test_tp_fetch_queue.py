"""Tests for the merged third-party fetch queue (``tp_fetch_queue``).

The queue is what a single sequential downloader sweeps, so its two failure
modes are both silent: a row that should have been filtered out costs a
download of the wrong recording, and a row that should have been kept is a
track that simply never arrives.  The Harmonix key is the load-bearing one --
28 videos are shared by two annotated tracks each, so anything keyed on the
video id drops half of every pair with nothing to say so.
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
for _path in (str(REPO_ROOT), str(TRAINING_DIR), str(TRAINING_DIR / "raveform"),
              str(TRAINING_DIR / "third_party")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tp_fetch_queue as queue  # noqa: E402

SALAMI_METADATA_HEADER = ("SONG_ID", "SOURCE", "SONG_TITLE", "ARTIST", "CLASS")
PAIRINGS_HEADER = ("salami_id", "salami_length", "youtube_id", "youtube_length",
                   "coverage", "coverage_percent", "onset_in_youtube",
                   "onset_in_salami")


def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def make_salami_dir(tmp_path: Path, ids=("2", "3"), klass="popular") -> Path:
    root = tmp_path / "salami-data-public"
    write_csv(root / "metadata" / "metadata.csv", SALAMI_METADATA_HEADER,
              [(track, "Codaich", f"Title {track}", f"Artist {track}", klass)
               for track in ids])
    for track in ids:
        (root / "annotations" / track).mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    return root


def make_pairings(tmp_path: Path, rows) -> Path:
    path = tmp_path / "matching-salami" / queue.SALAMI_PAIRINGS_FILE
    write_csv(path, PAIRINGS_HEADER, rows)
    return path


def pairing(salami_id="2", coverage="0.98", video="VID2", length="264.8"):
    return (salami_id, length, video, "265", "261.4", coverage, "0.6", "0.5")


def make_harmonixset(tmp_path: Path, rows) -> Path:
    root = tmp_path / "harmonixset"
    dataset = root / "dataset"
    write_csv(dataset / queue.HARMONIX_URLS_FILE, ("File", "URL"),
              [(name, url) for name, url, _score in rows])
    write_csv(dataset / queue.HARMONIX_SCORES_FILE, ("File", "score"),
              [(name, score) for name, _url, score in rows])
    write_csv(dataset / queue.HARMONIX_METADATA_FILE,
              ("File", "Title", "Artist", "Release", "Duration", "BPM"),
              [(name, f"Title {name}", f"Artist {name}", "Album", "142.47", "113")
               for name, _url, _score in rows])
    return root


def make_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "raveform"
    (corpus / "third_party" / "audio").mkdir(parents=True, exist_ok=True)
    return corpus


def salami_ids(tally):
    return [row.track_id for row in tally.rows]


def test_a_salami_pairing_below_the_coverage_floor_is_rejected(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path)
    pairings = make_pairings(tmp_path, [pairing("2", coverage="0.89")])
    tally = queue.salami_rows(corpus, salami, pairings)
    assert tally.rows == []
    assert tally.counts["salami_low_coverage"] == 1


def test_a_salami_pairing_at_the_coverage_floor_is_kept(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path)
    pairings = make_pairings(tmp_path, [pairing("2", coverage="0.9")])
    tally = queue.salami_rows(corpus, salami, pairings)
    assert salami_ids(tally) == ["salami-2"]


def test_a_salami_pairing_with_no_annotation_directory_is_rejected(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path, ids=("2",))
    pairings = make_pairings(tmp_path, [pairing("2"), pairing("99", video="VID99")])
    tally = queue.salami_rows(corpus, salami, pairings)
    assert salami_ids(tally) == ["salami-2"]
    assert tally.counts["salami_no_annotation"] == 1


def test_a_live_music_archive_salami_track_is_rejected(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path, klass="Live_Music_Archive")
    pairings = make_pairings(tmp_path, [pairing("2")])
    tally = queue.salami_rows(corpus, salami, pairings)
    assert tally.rows == []
    assert tally.counts["salami_live_music_archive"] == 1


def test_a_salami_track_whose_audio_is_already_in_the_dataset_is_rejected(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path)
    (salami / "audio" / "2.mp3").write_bytes(b"already here")
    pairings = make_pairings(tmp_path, [pairing("2")])
    tally = queue.salami_rows(corpus, salami, pairings)
    assert tally.rows == []
    assert tally.counts["salami_already_on_disk"] == 1


def test_a_salami_track_already_fetched_into_the_quarantine_is_rejected(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path)
    fetched = queue.audio_path(corpus, "salami", "salami-2")
    fetched.parent.mkdir(parents=True, exist_ok=True)
    fetched.write_bytes(b"fetched")
    pairings = make_pairings(tmp_path, [pairing("2")])
    tally = queue.salami_rows(corpus, salami, pairings)
    assert tally.rows == []
    assert tally.counts["salami_already_fetched"] == 1


def test_each_salami_filter_rejects_on_its_own(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path, ids=("2", "3", "4"))
    write_csv(salami / "metadata" / "metadata.csv", SALAMI_METADATA_HEADER,
              [("2", "Codaich", "T2", "A2", "popular"),
               ("3", "Codaich", "T3", "A3", "Live_Music_Archive"),
               ("4", "Codaich", "T4", "A4", "popular"),
               ("5", "Codaich", "T5", "A5", "popular")])
    (salami / "audio" / "4.mp3").write_bytes(b"on disk")
    pairings = make_pairings(tmp_path, [
        pairing("2"),
        pairing("3"),
        pairing("4"),
        pairing("5", coverage="0.10"),
        pairing("6"),
    ])
    tally = queue.salami_rows(corpus, salami, pairings)
    assert salami_ids(tally) == ["salami-2"]
    assert tally.counts["salami_live_music_archive"] == 1
    assert tally.counts["salami_already_on_disk"] == 1
    assert tally.counts["salami_low_coverage"] == 1
    assert tally.counts["salami_no_annotation"] == 1


def test_the_salami_row_carries_the_annotation_duration_and_metadata(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path)
    pairings = make_pairings(tmp_path, [pairing("2", length="264.8")])
    row = queue.salami_rows(corpus, salami, pairings).rows[0]
    assert row.source == "salami"
    assert row.youtube_id == "VID2"
    assert row.title == "Title 2"
    assert row.artist == "Artist 2"
    assert row.annotation_duration_sec == "264.8"


def test_the_harmonix_threshold_is_inclusive_at_its_boundary(tmp_path):
    corpus = make_corpus(tmp_path)
    harmonixset = make_harmonixset(tmp_path, [
        ("0001_a", "http://www.youtube.com/watch?v=AAA", "0.95"),
        ("0002_b", "http://www.youtube.com/watch?v=BBB", "0.9499999"),
    ])
    tally = queue.harmonix_rows(corpus, harmonixset, 0.95)
    assert [row.track_id for row in tally.rows] == ["hx-0001_a"]
    assert tally.counts["harmonix_below_threshold"] == 1


def test_the_harmonix_threshold_is_a_parameter_not_a_constant(tmp_path):
    corpus = make_corpus(tmp_path)
    harmonixset = make_harmonixset(tmp_path, [
        ("0001_a", "http://www.youtube.com/watch?v=AAA", "0.91"),
    ])
    assert queue.harmonix_rows(corpus, harmonixset, 0.95).rows == []
    assert len(queue.harmonix_rows(corpus, harmonixset, 0.90).rows) == 1


def test_the_nine_unaligned_harmonix_tracks_are_excluded_whatever_their_score(tmp_path):
    corpus = make_corpus(tmp_path)
    rows = [(name, f"https://www.youtube.com/watch?v=V{index} (needs to be sped-up)",
             "1.0")
            for index, name in enumerate(sorted(queue.HARMONIX_UNALIGNED))]
    harmonixset = make_harmonixset(tmp_path, rows)
    tally = queue.harmonix_rows(corpus, harmonixset, 0.95)
    assert tally.rows == []
    assert tally.counts["harmonix_unaligned_by_name"] == 9


def test_the_hard_exclusion_list_is_exactly_the_nine_named_tracks():
    assert len(queue.HARMONIX_UNALIGNED) == 9
    assert "0653_dynamite" in queue.HARMONIX_UNALIGNED
    assert "0046_castlesmadeofsand" in queue.HARMONIX_UNALIGNED


def test_the_video_id_survives_https_and_trailing_free_text():
    assert queue.parse_video_id(
        "http://www.youtube.com/watch?v=iBHNgV6_znU") == "iBHNgV6_znU"
    assert queue.parse_video_id(
        "https://www.youtube.com/watch?v=03lWKQtl9BU") == "03lWKQtl9BU"
    assert queue.parse_video_id(
        "https://www.youtube.com/watch?v=qETzuNQ-OSg (needs to be 2.45% sped-up)"
    ) == "qETzuNQ-OSg"
    assert queue.parse_video_id(
        "https://www.youtube.com/watch?v=AAA&t=30s") == "AAA"
    assert queue.parse_video_id("") == ""
    assert queue.parse_video_id("https://www.youtube.com/watch") == ""


def test_two_harmonix_tracks_sharing_one_video_each_get_their_own_row(tmp_path):
    corpus = make_corpus(tmp_path)
    harmonixset = make_harmonixset(tmp_path, [
        ("0001_a", "http://www.youtube.com/watch?v=SHARED", "0.99"),
        ("0002_b", "http://www.youtube.com/watch?v=SHARED", "0.98"),
    ])
    tally = queue.harmonix_rows(corpus, harmonixset, 0.95)
    assert [row.track_id for row in tally.rows] == ["hx-0001_a", "hx-0002_b"]
    assert {row.youtube_id for row in tally.rows} == {"SHARED"}


def test_a_harmonix_track_already_fetched_is_skipped(tmp_path):
    corpus = make_corpus(tmp_path)
    harmonixset = make_harmonixset(tmp_path, [
        ("0001_a", "http://www.youtube.com/watch?v=AAA", "0.99"),
        ("0002_b", "http://www.youtube.com/watch?v=BBB", "0.99"),
    ])
    fetched = queue.audio_path(corpus, "harmonix", "hx-0001_a")
    fetched.parent.mkdir(parents=True, exist_ok=True)
    fetched.write_bytes(b"fetched")
    tally = queue.harmonix_rows(corpus, harmonixset, 0.95)
    assert [row.track_id for row in tally.rows] == ["hx-0002_b"]
    assert tally.counts["harmonix_already_fetched"] == 1


def test_the_merged_queue_puts_salami_first_and_round_trips(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path)
    pairings = make_pairings(tmp_path, [pairing("2"), pairing("3", video="VID3")])
    harmonixset = make_harmonixset(tmp_path, [
        ("0001_a", "http://www.youtube.com/watch?v=AAA", "0.99"),
    ])
    tally = queue.build_queue(corpus, salami, pairings, harmonixset, 0.95)
    assert [row.track_id for row in tally.rows] == ["salami-2", "salami-3", "hx-0001_a"]

    target = queue.queue_path(corpus)
    queue.write_queue(target, tally.rows)
    assert not target.with_suffix(target.suffix + ".part").exists()
    assert queue.read_queue(target) == tally.rows
    with open(target, encoding="utf-8", newline="") as handle:
        assert next(csv.reader(handle)) == list(queue.QUEUE_HEADER)


def test_the_queue_is_keyed_on_track_id_so_a_shared_video_keeps_both_rows(tmp_path):
    corpus = make_corpus(tmp_path)
    salami = make_salami_dir(tmp_path, ids=())
    pairings = make_pairings(tmp_path, [])
    harmonixset = make_harmonixset(tmp_path, [
        ("0001_a", "http://www.youtube.com/watch?v=SHARED", "0.99"),
        ("0002_b", "http://www.youtube.com/watch?v=SHARED", "0.99"),
    ])
    tally = queue.build_queue(corpus, salami, pairings, harmonixset, 0.95)
    target = queue.queue_path(corpus)
    queue.write_queue(target, tally.rows)
    written = queue.read_queue(target)
    assert len({row.track_id for row in written}) == 2
