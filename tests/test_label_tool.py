"""Tests for the hand-labelling tool (training/label_tool.py)."""
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import label_tool  # noqa: E402
from label_tool import (  # noqa: E402
    DEFAULT_STRENGTH,
    LABEL_COLORS,
    LABELS,
    STRENGTHS,
    _YOUTUBE_ID,
    add_boundary,
    apply_edit,
    audio_digest,
    beat_grid,
    commit_labels,
    default_title,
    format_csv,
    from_annotation,
    labels_path,
    load_labels,
    normalise,
    parse_csv,
    save_labels,
    resolve_identity,
    to_annotation,
)

SECTIONS = [
    {'start': 0.0, 'label': 'intro', 'strength': 'major'},
    {'start': 31.25, 'label': 'buildup', 'strength': 'major'},
    {'start': 48.0, 'label': 'drop', 'strength': 'major'},
    {'start': 120.001, 'label': 'breakdown', 'strength': 'major'},
]
MIXED = [dict(entry, strength='minor' if entry['label'] == 'buildup' else 'major')
         for entry in SECTIONS]


_REAL_TMP_LABELS_DIR = label_tool.tmp_labels_dir


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    """No test may write into the real corpus's working-label workspace."""
    workspace = tmp_path / 'scratch_corpus' / label_tool.TMP_LABELS_DIR_NAME
    monkeypatch.setattr(label_tool, 'tmp_labels_dir', lambda: workspace)
    return workspace


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


def test_a_save_leaves_no_temp_file_behind(tmp_path, scratch):
    audio = tmp_path / 'song.mp3'
    audio.write_bytes(b'')
    save_labels(str(audio), SECTIONS)
    assert sorted(p.name for p in scratch.iterdir()) == ['song.mp3.labels.csv']


def test_every_mutation_reaches_disk_before_the_next_one(tmp_path, scratch):
    audio = tmp_path / 'song.mp3'
    audio.write_bytes(b'')
    sections = load_labels(str(audio))
    assert not labels_path(str(audio)).exists()

    def step(trigger, **kwargs):
        nonlocal sections
        updated, status = apply_edit(str(audio), trigger, sections, **kwargs)
        assert updated is not None and 'saved' in status
        sections = updated
        assert _load_in_a_new_process(audio, scratch) == sections

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


def _load_in_a_new_process(audio: Path, scratch: Path) -> list:
    source = (
        'import json, pathlib, sys; sys.path.insert(0, sys.argv[1]); '
        'import label_tool; '
        'label_tool.tmp_labels_dir = lambda: pathlib.Path(sys.argv[3]); '
        'print(json.dumps(label_tool.load_labels(sys.argv[2])))'
    )
    finished = subprocess.run(
        [sys.executable, '-c', source, str(TRAINING_DIR), str(audio),
         str(scratch)],
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
    args = build_parser().parse_args(['label', str(song), '--port', '9001'])
    assert args.func is label_cmd
    assert (args.audio, args.port) == (str(song), 9001)

    launched = {}
    monkeypatch.setitem(sys.modules, 'label_tool', types.SimpleNamespace(
        launch=lambda audio, port: launched.update(audio=audio, port=port)))
    await label_cmd(args)
    assert launched == {'audio': str(song), 'port': 9001}


async def test_the_labeller_takes_no_audio_device_flag(tmp_path):
    """Where the sound comes out is a browser question, answered in the page."""
    from lib.main import build_parser

    args = build_parser().parse_args(['label', str(tmp_path / 'song.mp3')])
    assert args.port == 8070
    assert not hasattr(args, 'output_device')
    with pytest.raises(SystemExit):
        build_parser().parse_args(['label', 'x.mp3', '-o', '7'])


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path / 'corpus'
    (root / 'annotations').mkdir(parents=True)
    (root / 'annotations' / 'segments.json').write_text('[]', encoding='utf-8')
    monkeypatch.setattr(label_tool, 'corpus_dir', lambda: root)
    return root


@pytest.fixture
def song(tmp_path):
    audio = tmp_path / 'is_it_beautiful.kUP_iJuoq9g.mp3'
    audio.write_bytes(b'ID3 not really an mp3, but bytes are bytes')
    return audio


@pytest.fixture
def song_id(song):
    return resolve_identity(str(song))[0]


TITLE = 'Ferry Corsten - Is It Beautiful'


def test_the_working_file_lives_in_the_scratch_workspace_not_beside_the_audio(
        song, scratch):
    path = labels_path(str(song))
    assert path.parent == scratch
    assert path.name == 'is_it_beautiful.kUP_iJuoq9g.mp3.labels.csv'
    save_labels(str(song), SECTIONS)
    assert path.exists()
    assert not song.with_name(song.name + '.labels.csv').exists()


def test_working_state_follows_the_corpus_rather_than_the_checkout(
        tmp_path, monkeypatch):
    elsewhere = tmp_path / 'main-checkout' / 'training' / 'data' / 'raveform'
    monkeypatch.setattr(label_tool, 'corpus_dir', lambda: elsewhere)
    monkeypatch.setattr(label_tool, 'tmp_labels_dir', _REAL_TMP_LABELS_DIR)
    scratch = label_tool.tmp_labels_dir()
    assert scratch == elsewhere / label_tool.TMP_LABELS_DIR_NAME
    assert scratch.parent == label_tool.hand_label_path('x').parent.parent, (
        'the scratch file and the committed label must resolve to one corpus')


def test_new_audio_is_content_addressed_and_never_looks_like_a_download(
        tmp_path, corpus):
    one = tmp_path / 'is_it_beautiful.kUP_iJuoq9g.mp3'
    one.write_bytes(b'the same bytes')
    renamed = tmp_path / 'totally different name.mp3'
    renamed.write_bytes(b'the same bytes')
    other = tmp_path / 'other.mp3'
    other.write_bytes(b'different bytes')

    identity = lambda path: resolve_identity(str(path))[0]
    assert identity(one) == identity(renamed)
    assert identity(one) != identity(other)
    assert identity(one).startswith('hand-')
    assert not _YOUTUBE_ID.match(identity(one)[len('hand-'):])
    assert identity(one) == 'hand-' + hashlib.sha256(
        b'the same bytes').hexdigest()[:12]
    assert resolve_identity(str(one))[1] is None


def _publish(corpus, youtube_id, payload, record_checksum=True):
    """A track the corpus already holds, exactly as the downloader left it."""
    audio = corpus / 'audio' / f'{youtube_id}.mp3'
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(payload)
    if record_checksum:
        (corpus / label_tool.CHECKSUMS_FILE).write_text(
            f'{hashlib.sha256(payload).hexdigest()}  audio/{youtube_id}.mp3\n',
            encoding='utf-8')
    return audio


def test_audio_the_corpus_already_holds_keeps_its_native_id(song, corpus):
    published = _publish(corpus, 'kUP_iJuoq9g', song.read_bytes())

    identifier, native = resolve_identity(str(song))
    assert identifier == 'kUP_iJuoq9g'
    assert native == published

    result = commit_labels(str(song), load_labels(str(song)), 214.842, TITLE)
    assert result['copied'] is False
    assert result['labels'] == corpus / 'annotations' / 'kUP_iJuoq9g.hand.json'
    assert result['audio'] == published
    assert sorted(p.name for p in (corpus / 'audio').iterdir()) == [
        'kUP_iJuoq9g.mp3'], 'nothing was copied'
    record = json.loads(result['labels'].read_text(encoding='utf-8'))
    assert record['id'] == 'kUP_iJuoq9g'
    assert record['audio'] == 'kUP_iJuoq9g.mp3'


def test_a_renamed_copy_of_a_corpus_track_is_still_that_track(tmp_path, corpus):
    payload = b'the exact bytes the downloader fetched'
    _publish(corpus, 'hzIFjGcOKbg', payload)
    renamed = tmp_path / 'some track I saved off a mix.mp3'
    renamed.write_bytes(payload)

    assert resolve_identity(str(renamed))[0] == 'hzIFjGcOKbg'


def test_dedupe_falls_back_to_hashing_when_the_baseline_is_absent(song, corpus):
    published = _publish(corpus, 'kUP_iJuoq9g', song.read_bytes(),
                         record_checksum=False)
    assert not (corpus / label_tool.CHECKSUMS_FILE).exists()

    identifier, native = resolve_identity(str(song))
    assert identifier == 'kUP_iJuoq9g'
    assert native == published


def test_a_stale_baseline_does_not_hide_a_track_it_never_recorded(song, corpus):
    published = _publish(corpus, 'kUP_iJuoq9g', song.read_bytes(),
                         record_checksum=False)
    (corpus / label_tool.CHECKSUMS_FILE).write_text(
        f'{"0" * 64}  audio/somethingelse.mp3\n', encoding='utf-8')

    assert resolve_identity(str(song)) == ('kUP_iJuoq9g', published)


def test_audio_of_the_same_size_but_different_bytes_is_not_the_same_track(
        song, corpus):
    same_size = bytes(len(song.read_bytes()))
    _publish(corpus, 'kUP_iJuoq9g', same_size, record_checksum=False)

    identifier, native = resolve_identity(str(song))
    assert identifier.startswith('hand-')
    assert native is None


def test_the_digest_is_the_whole_file():
    assert len(audio_digest(__file__)) == 64


def test_the_title_prefill_drops_a_trailing_youtube_id():
    assert default_title('/x/is_it_beautiful.kUP_iJuoq9g.mp3') == 'is it beautiful'
    assert default_title('/x/Konflict - Beckoning.mp3') == 'Konflict - Beckoning'


def test_the_committed_shape_is_the_corpus_section_shape_and_round_trips():
    record = to_annotation(MIXED, 214.842, 'hand-abc123', 'hand-abc123.mp3',
                           TITLE)
    assert record['sections'] == [
        {'name': 'intro', 'start': 0.0, 'end': 31.25, 'strength': 'major'},
        {'name': 'buildup', 'start': 31.25, 'end': 48.0, 'strength': 'minor'},
        {'name': 'drop', 'start': 48.0, 'end': 120.001, 'strength': 'major'},
        {'name': 'breakdown', 'start': 120.001, 'end': 214.842,
         'strength': 'major'},
    ]
    assert record['title'] == TITLE
    assert from_annotation(record) == normalise(MIXED)

    # The published reader is the contract, so it has to accept a hand label.
    sys.path.insert(0, str(TRAINING_DIR / 'raveform'))
    from raveform_fetch_annotations import parse_sections
    assert parse_sections(record)[1] == (31.25, 48.0, 'buildup')


def test_commit_promotes_new_audio_into_the_dataset(song, corpus, song_id):
    sections = load_labels(str(song))
    sections = add_boundary(sections, 61.9, 'buildup', 'minor')
    save_labels(str(song), sections)

    result = commit_labels(str(song), sections, 214.842, TITLE)
    assert result['labels'] == corpus / 'annotations' / f'{song_id}.hand.json'
    assert result['copied'] is True
    assert result['audio'] == corpus / 'audio' / f'{song_id}.mp3'
    assert result['audio'].read_bytes() == song.read_bytes()

    record = json.loads(result['labels'].read_text(encoding='utf-8'))
    assert record['id'] == song_id
    assert record['title'] == TITLE
    assert record['audio'] == f'{song_id}.mp3'
    assert from_annotation(record) == sections
    assert labels_path(str(song)).exists(), 'the scratch file is kept, not moved'


def test_commit_never_touches_the_published_annotation(song, corpus, song_id):
    segments = corpus / 'annotations' / 'segments.json'
    before = segments.read_bytes()
    commit_labels(str(song), load_labels(str(song)), 214.842, TITLE)
    assert segments.read_bytes() == before
    assert sorted(p.name for p in (corpus / 'annotations').iterdir()) == [
        f'{song_id}.hand.json', 'segments.json']


def test_committing_twice_overwrites_rather_than_accumulates(song, corpus):
    sections = load_labels(str(song))
    first = commit_labels(str(song), sections, 214.842, TITLE)
    once = first['labels'].read_bytes()

    again = commit_labels(str(song), sections, 214.842, TITLE)
    assert again['labels'] == first['labels']
    assert again['labels'].read_bytes() == once
    assert not again['labels'].with_name(again['labels'].name + '.tmp').exists()

    moved = commit_labels(str(song), add_boundary(sections, 12.0, 'drop'),
                          214.842, TITLE)
    assert moved['labels'].read_bytes() != once
    assert len(list((corpus / 'annotations').glob('*.hand.json'))) == 1


def test_a_second_commit_of_new_audio_copies_nothing_further(song, corpus):
    first = commit_labels(str(song), load_labels(str(song)), 214.842, TITLE)
    assert first['copied'] is True

    again = commit_labels(str(song), load_labels(str(song)), 214.842, TITLE)
    assert again['copied'] is False, 'the corpus now holds it'
    assert again['audio'] == first['audio']
    assert len(list((corpus / 'audio').iterdir())) == 1


def test_commit_reports_both_paths_through_the_dispatcher(song, corpus, song_id):
    sections = load_labels(str(song))
    updated, status = apply_edit(str(song), 'commit', sections, duration=214.842,
                                 title=TITLE)
    assert updated is None
    assert f'{song_id}.hand.json' in status and 'audio copied' in status
    _, status = apply_edit(str(song), 'commit', sections, duration=214.842,
                           title=TITLE)
    assert 'audio already at' in status


def test_commit_is_refused_without_a_title_and_writes_nothing(song, corpus):
    for blank in ('', '   ', None):
        updated, status = apply_edit(str(song), 'commit', load_labels(str(song)),
                                     duration=214.842, title=blank)
        assert updated is None
        assert status.startswith('commit refused')
        assert 'artist guard' in status
    assert list((corpus / 'annotations').glob('*.hand.json')) == []
    assert not (corpus / 'audio').exists()


def test_a_title_with_no_artist_dash_is_refused_and_writes_nothing(song, corpus):
    """The prefill for the owner's first real file was exactly this shape."""
    updated, status = apply_edit(str(song), 'commit', load_labels(str(song)),
                                 duration=214.842, title='is it beautiful')
    assert updated is None
    assert status.startswith('commit refused')
    assert 'no artist' in status and 'contamination' in status
    assert list((corpus / 'annotations').glob('*.hand.json')) == []
    assert not (corpus / 'audio').exists()


def test_a_release_with_genuinely_no_artist_goes_through_the_override(
        song, corpus, song_id):
    _, status = apply_edit(str(song), 'commit', load_labels(str(song)),
                           duration=214.842, title='is it beautiful',
                           no_artist=True)
    assert 'committed' in status
    assert 'NO artist' in status

    record = json.loads(
        (corpus / 'annotations' / f'{song_id}.hand.json').read_text('utf-8'))
    assert record['title'] == 'is it beautiful'
    assert record['artist'] is None, 'absent honestly, not absent by omission'


def test_a_dash_title_records_the_artist_the_guard_will_read(song, corpus,
                                                             song_id):
    apply_edit(str(song), 'commit', load_labels(str(song)), duration=214.842,
               title='Ferry Corsten - Is It Beautiful')
    record = json.loads(
        (corpus / 'annotations' / f'{song_id}.hand.json').read_text('utf-8'))
    assert record['artist'] == 'Ferry Corsten'

    sys.path.insert(0, str(TRAINING_DIR))
    from select_eval_set import artist_of
    assert artist_of(record['title']) == artist_of('Ferry Corsten - anything')


def test_the_override_is_not_needed_when_a_dash_is_present(song, corpus):
    _, status = apply_edit(str(song), 'commit', load_labels(str(song)),
                           duration=214.842, title=TITLE, no_artist=False)
    assert 'committed' in status


def test_commit_still_places_the_files_when_no_admission_step_exists(song, corpus):
    result = commit_labels(str(song), load_labels(str(song)), 214.842, TITLE)
    assert result['admitted'] is False
    assert 'admission pending' in result['admission']
    assert label_tool.ADMISSION_HINT in result['admission']
    assert result['labels'].exists() and result['audio'].exists()


def test_commit_calls_the_admission_step_when_the_branch_has_one(
        song, corpus, monkeypatch):
    seen = {}

    def admit(track_id, audio, labels, corpus):
        seen.update(track_id=track_id, audio=audio, labels=labels, corpus=corpus)
        return 'grid + manifest row added'

    monkeypatch.setitem(sys.modules, label_tool.ADMISSION_MODULE,
                        types.SimpleNamespace(admit=admit))
    result = commit_labels(str(song), load_labels(str(song)), 214.842, TITLE)
    assert result['admitted'] is True
    assert result['admission'] == 'grid + manifest row added'
    assert seen == {'track_id': result['audio'].stem, 'audio': result['audio'],
                    'labels': result['labels'], 'corpus': corpus}


def test_a_raising_admission_is_reported_and_never_swallowed(
        song, corpus, monkeypatch):
    def admit(**_):
        raise RuntimeError('madmom is not installed')

    monkeypatch.setitem(sys.modules, label_tool.ADMISSION_MODULE,
                        types.SimpleNamespace(admit=admit))
    result = commit_labels(str(song), load_labels(str(song)), 214.842, TITLE)
    assert result['admitted'] is False
    assert 'admission FAILED' in result['admission']
    assert 'madmom is not installed' in result['admission']
    assert result['labels'].exists(), 'the labels are placed either way'


def test_a_generated_beat_grid_is_read_when_present(song, corpus, song_id):
    assert beat_grid(str(song)) == []

    beats = corpus / 'annotations' / 'beats'
    beats.mkdir(parents=True)
    grid = beats / f'{song_id}.hand.beat.csv'
    grid.write_text('time,downbeat,section\n0.5,1,intro\n1.0,0,intro\n',
                    encoding='utf-8')
    assert beat_grid(str(song)) == [(0.5, 1), (1.0, 0)]

    grid.write_text('nonsense\n', encoding='utf-8')
    assert beat_grid(str(song)) == [], 'an unreadable grid is absent, not fatal'


REPO_ROOT = Path(__file__).resolve().parents[1]

# Re-including hand labels punches a hole through the rule that keeps ~13 GiB of
# corpus out of git. These are the paths that must never come through it.
MUST_STAY_IGNORED = (
    'training/data/raveform/annotations/segments.json',
    'training/data/raveform/audio/kUP_iJuoq9g.mp3',
    'training/data/raveform/manifest.csv',
    'training/data/raveform/training_table.csv.gz',
    'training/data/raveform/annotations/beats/0064.hzIFjGcOKbg.beat.csv',
    'training/data/raveform/models/v1/model.onnx',
)


def _is_ignored(path: str) -> bool:
    # Without -v: a path matching a NEGATED pattern is not reported, which is the
    # only reading of check-ignore that answers "is this ignored" for both.
    return subprocess.run(['git', 'check-ignore', '-q', path],
                          cwd=REPO_ROOT).returncode == 0


def test_the_corpus_stays_out_of_git_apart_from_hand_labels():
    escaped = [path for path in MUST_STAY_IGNORED if not _is_ignored(path)]
    assert not escaped, f'these would be committed: {escaped}'


def test_a_hand_label_is_the_one_thing_git_will_take():
    assert not _is_ignored(
        'training/data/raveform/annotations/hand-0123456789ab.hand.json')
    assert not _is_ignored(
        'training/data/raveform/annotations/anything.hand.json')


def test_the_hole_is_exactly_one_suffix_in_exactly_one_directory():
    for near_miss in ('training/data/raveform/hand-abc.hand.json',
                      'training/data/hand-abc.hand.json',
                      'training/data/raveform/annotations/beats/x.hand.json',
                      'training/data/raveform/annotations/x.hand.json.bak',
                      'training/data/tmp_labels/song.mp3.labels.csv'):
        assert _is_ignored(near_miss), f'{near_miss} escaped the corpus rule'


def test_an_interrupt_with_no_show_to_stop_is_not_swallowed():
    """The whole of the reported Ctrl-C bug, reachable without a console.

    `lib.main` installs this handler at import, so it is what receives Ctrl-C
    under every subcommand -- including `label`, which never builds a show. A
    handler that returns consumes the interrupt, and a server blocked in a serve
    loop never learns it was asked to stop.
    """
    from lib import main

    assert main.global_app is None
    with pytest.raises(KeyboardInterrupt):
        main.death_handler(signal.SIGINT, None)


def test_the_handler_still_stops_a_running_show(monkeypatch):
    from lib import main

    stopped = []
    monkeypatch.setattr(main, 'global_app',
                        types.SimpleNamespace(stop=lambda: stopped.append(True)))
    main.death_handler(signal.SIGINT, None)
    assert stopped == [True]


def _console_signals_are_deliverable() -> bool:
    """Whether this process can actually send one, rather than whether it may.

    `os.kill(pid, CTRL_C_EVENT)` is `GenerateConsoleCtrlEvent`, which does
    nothing when the caller has no console -- a test runner started from a pipe
    has none. Without this probe the interrupt test passes by never testing
    anything, which is worse than not having it.
    """
    if os.name != 'nt':
        return False
    control = subprocess.Popen(
        [sys.executable, '-c', "import sys,time\nprint('READY', flush=True)\n"
                               "time.sleep(30)\n"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        if not control.stdout.readline().startswith('READY'):
            return False
        os.kill(control.pid, signal.CTRL_C_EVENT)
        try:
            control.wait(timeout=8)
            return True
        except subprocess.TimeoutExpired:
            return False
    finally:
        if control.poll() is None:
            subprocess.run(['taskkill', '/PID', str(control.pid), '/T', '/F'],
                           capture_output=True)


_CTRL_C_HARNESS = """import sys,time
print('READY', flush=True)
time.sleep(120)
"""


def test_one_ctrl_c_ends_the_labeller(tmp_path):
    if not _console_signals_are_deliverable():
        pytest.skip('this runner has no console, so CTRL_C_EVENT reaches nothing '
                    '-- run the suite from a terminal to exercise the interrupt')

    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]

    tone = tmp_path / 'tone.wav'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=2', str(tone)],
                   check=True, stdin=subprocess.DEVNULL)

    process = subprocess.Popen(
        [sys.executable, 'auto_pilot', 'label', str(tone), '--port', str(port)],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', port)) == 0:
                    break
            assert process.poll() is None, 'the labeller died before serving'
            time.sleep(0.2)
        else:
            raise AssertionError('the labeller never came up')

        os.kill(process.pid, signal.CTRL_C_EVENT)
        try:
            process.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            raise AssertionError('one Ctrl-C did not end the labeller')

        with socket.socket() as probe:
            assert probe.connect_ex(('127.0.0.1', port)) != 0, 'port still bound'
    finally:
        if process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           capture_output=True)
            process.wait(timeout=10)
