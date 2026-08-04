"""Tests for the hand-labelling tool (training/label_tool.py)."""
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from label_tool import (  # noqa: E402
    DEFAULT_STRENGTH,
    LABEL_COLORS,
    LABELS,
    STRENGTHS,
    add_boundary,
    format_csv,
    labels_path,
    load_labels,
    normalise,
    parse_csv,
    save_labels,
)

SECTIONS = [
    {'start': 0.0, 'label': 'intro', 'strength': 'major'},
    {'start': 31.25, 'label': 'buildup', 'strength': 'major'},
    {'start': 48.0, 'label': 'drop', 'strength': 'major'},
    {'start': 120.001, 'label': 'breakdown', 'strength': 'major'},
]
MIXED = [dict(entry, strength='minor' if entry['label'] == 'buildup' else 'major')
         for entry in SECTIONS]


def test_all_major_stays_a_two_column_file():
    assert format_csv(SECTIONS) == (
        '0.000,intro\n31.250,buildup\n48.000,drop\n120.001,breakdown\n')


def test_any_minor_writes_the_strength_column_on_every_row():
    assert format_csv(MIXED) == (
        '0.000,intro,major\n31.250,buildup,minor\n'
        '48.000,drop,major\n120.001,breakdown,major\n')


def test_two_column_rows_load_as_major():
    assert parse_csv('0.000,intro\n48.000,drop\n') == [
        {'start': 0.0, 'label': 'intro', 'strength': DEFAULT_STRENGTH},
        {'start': 48.0, 'label': 'drop', 'strength': DEFAULT_STRENGTH},
    ]


def test_round_trip_is_byte_identical_in_both_shapes():
    for sections in (SECTIONS, MIXED):
        once = format_csv(sections)
        assert format_csv(parse_csv(once)) == once


def test_saved_file_round_trips_on_disk_with_lf_endings(tmp_path):
    for sections in (SECTIONS, MIXED):
        audio = tmp_path / f'{sections[1]["strength"]}.mp3'
        audio.write_bytes(b'')

        path = save_labels(str(audio), sections)
        assert path == labels_path(str(audio))

        first = path.read_bytes()
        assert b'\r' not in first
        save_labels(str(audio), load_labels(str(audio)))
        assert path.read_bytes() == first


def test_load_seeds_an_opening_intro_when_no_file_exists(tmp_path):
    audio = tmp_path / 'song.mp3'
    audio.write_bytes(b'')
    assert load_labels(str(audio)) == [
        {'start': 0.0, 'label': LABELS[0], 'strength': DEFAULT_STRENGTH}]


def test_every_label_in_the_vocabulary_round_trips():
    sections = [{'start': index * 10.0, 'label': label,
                 'strength': STRENGTHS[index % len(STRENGTHS)]}
                for index, label in enumerate(LABELS)]
    once = format_csv(sections)
    assert parse_csv(once) == normalise(sections)
    assert format_csv(parse_csv(once)) == once
    assert all(label in LABEL_COLORS for label in LABELS)


def test_an_unknown_strength_falls_back_to_major():
    assert parse_csv('0.000,intro,sideways\n')[0]['strength'] == DEFAULT_STRENGTH


def test_sections_are_sorted_and_starts_are_unique():
    scrambled = [
        {'start': 48.0, 'label': 'drop'},
        {'start': 0.0, 'label': 'intro'},
        {'start': 48.0004, 'label': 'outro'},
        {'start': -3.0, 'label': 'cooldown'},
    ]
    assert normalise(scrambled) == [
        {'start': 0.0, 'label': 'intro', 'strength': DEFAULT_STRENGTH},
        {'start': 48.0, 'label': 'drop', 'strength': DEFAULT_STRENGTH},
    ]


def test_adding_at_an_existing_start_replaces_rather_than_duplicates():
    updated = add_boundary(SECTIONS, 48.0002, 'cooldown', 'minor')
    assert [s['start'] for s in updated] == [0.0, 31.25, 48.0, 120.001]
    assert updated[2] == {'start': 48.0, 'label': 'cooldown', 'strength': 'minor'}


def test_adding_keeps_the_list_sorted():
    updated = add_boundary(SECTIONS, 40.0, 'buildup')
    starts = [s['start'] for s in updated]
    assert starts == sorted(starts)
    assert 40.0 in starts
