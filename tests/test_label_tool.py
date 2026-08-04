"""Tests for the hand-labelling tool (training/label_tool.py)."""
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from label_tool import (  # noqa: E402
    DEFAULT_STRENGTH,
    LABEL_COLORS,
    LABELS,
    STRENGTHS,
    add_boundary,
    apply_edit,
    format_csv,
    labels_path,
    load_labels,
    normalise,
    output_device_name,
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


def test_a_save_leaves_no_temp_file_behind(tmp_path):
    audio = tmp_path / 'song.mp3'
    audio.write_bytes(b'')
    save_labels(str(audio), SECTIONS)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        'song.mp3', 'song.mp3.labels.csv']


def test_every_mutation_reaches_disk_before_the_next_one(tmp_path):
    audio = tmp_path / 'song.mp3'
    audio.write_bytes(b'')
    sections = load_labels(str(audio))
    assert not labels_path(str(audio)).exists()

    def step(trigger, **kwargs):
        nonlocal sections
        updated, status = apply_edit(str(audio), trigger, sections, **kwargs)
        assert updated is not None and 'saved' in status
        sections = updated
        assert _load_in_a_new_process(audio) == sections

    step('mark', cursor=31.25, new_label='buildup', new_strength='minor')
    step('mark', cursor=48.0, new_label='drop')
    step('mark', cursor=120.0, new_label='breakdown')
    step({'type': 'row-nudge', 'index': 2, 'delta': -0.5})
    step({'type': 'row-label', 'index': 1},
         row_labels=['intro', 'cooldown', 'drop', 'breakdown'])
    step({'type': 'row-strength', 'index': 3},
         row_strengths=['major', 'major', 'major', 'minor'])
    step({'type': 'row-del', 'index': 2})

    assert [(s['start'], s['label'], s['strength']) for s in sections] == [
        (0.0, 'intro', 'major'),
        (31.25, 'cooldown', 'minor'),
        (120.0, 'breakdown', 'minor'),
    ]


def test_a_no_op_edit_neither_rewrites_nor_reports(tmp_path):
    audio = tmp_path / 'song.mp3'
    audio.write_bytes(b'')
    sections = load_labels(str(audio))
    apply_edit(str(audio), 'mark', sections, cursor=0.0, new_label='intro')
    assert apply_edit(str(audio), 'mark', sections, cursor=0.0,
                      new_label='intro') == (None, None)
    assert apply_edit(str(audio), {'type': 'row-del', 'index': 9},
                      sections) == (None, None)


def _load_in_a_new_process(audio: Path) -> list:
    source = (
        'import json, sys; sys.path.insert(0, sys.argv[1]); '
        'import label_tool; '
        'print(json.dumps(label_tool.load_labels(sys.argv[2])))'
    )
    finished = subprocess.run(
        [sys.executable, '-c', source, str(TRAINING_DIR), str(audio)],
        capture_output=True, text=True, check=True)
    return json.loads(finished.stdout)


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


async def test_auto_pilot_label_parses_and_dispatches(monkeypatch, tmp_path):
    from lib.main import build_parser, label_cmd

    song = tmp_path / 'song.mp3'
    args = build_parser().parse_args(
        ['label', str(song), '--port', '9001', '-o', '7'])
    assert args.func is label_cmd
    assert (args.audio, args.port, args.output_device) == (str(song), 9001, '7')

    launched = {}
    monkeypatch.setitem(sys.modules, 'label_tool', types.SimpleNamespace(
        launch=lambda audio, port, output_device: launched.update(
            audio=audio, port=port, output_device=output_device)))
    await label_cmd(args)
    assert launched == {'audio': str(song), 'port': 9001, 'output_device': 7}


async def test_label_without_an_output_device_asks_for_the_system_default(
        monkeypatch, tmp_path):
    from lib.main import build_parser, label_cmd

    args = build_parser().parse_args(['label', str(tmp_path / 'song.mp3')])
    assert (args.port, args.output_device) == (8070, None)

    launched = {}
    monkeypatch.setitem(sys.modules, 'label_tool', types.SimpleNamespace(
        launch=lambda audio, port, output_device: launched.update(
            output_device=output_device)))
    await label_cmd(args)
    assert launched == {'output_device': None}


class FakePyAudio:
    def __init__(self, devices):
        self.devices = devices
        self.terminated = False

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, index):
        return self.devices[index]

    def terminate(self):
        self.terminated = True


def _fake():
    return FakePyAudio([
        {'name': 'Microphone (USB Audio)', 'maxInputChannels': 2,
         'maxOutputChannels': 0},
        {'name': 'Speakers (Realtek(R) Audio)', 'maxInputChannels': 0,
         'maxOutputChannels': 2},
        {'name': 'Headphones (WH-1000XM4 Stereo)', 'maxInputChannels': 0,
         'maxOutputChannels': 2},
    ])


def test_an_output_index_resolves_to_the_name_the_browser_will_match_on():
    assert output_device_name(2, _fake()) == 'Headphones (WH-1000XM4 Stereo)'


def test_an_index_outside_the_device_list_names_the_range():
    with pytest.raises(SystemExit, match=r'index 7 \(0\.\.2\)'):
        output_device_name(7, _fake())
    with pytest.raises(SystemExit):
        output_device_name(-1, _fake())


def test_an_input_only_device_is_refused_by_name():
    with pytest.raises(SystemExit, match='Microphone'):
        output_device_name(0, _fake())


def test_an_injected_handle_is_left_open_for_its_owner():
    audio = _fake()
    output_device_name(1, audio)
    assert not audio.terminated
