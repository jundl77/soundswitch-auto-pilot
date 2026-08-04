"""Hand-label a song's sections in the browser.

Launched by `auto_pilot label <song.mp3> [--port 8070]`, which is the only entry
point -- this module has no CLI of its own. Which speakers the audio comes out of
is a browser question rather than a server one, so it is answered in the page: a
dropdown of the outputs the browser can see, applied with `setSinkId` and
remembered in `localStorage`, so the owner picks his headphones once. A chip
names the device currently in use. The track is decoded once with ffmpeg into a waveform envelope and a
spectral-flux curve, so the beat starts a boundary should land on are visible
rather than guessed; play the audio, click the timeline to seek, and press "mark
boundary" to cut a section at the playhead, with the major/minor toggle saying
how strong that transition is. Each section gets a row in the table with a label
picker, a strength picker, +/- nudge buttons and a delete, and shows as a
coloured span on the timeline.

The vocabulary is the raw Raveform one minus `end`, which is a tail sentinel
rather than a phase; folding is a downstream decision and a labelling that has
already folded cannot be unfolded again. **A working file this tool cannot read
is refused by name rather than opened**: a time that is not a number, or a label
outside the vocabulary, names its own line and stops the launch before anything
is decoded. Both are reachable by hand-editing the CSV -- a pasted header line
is the whole of the first -- and both used to be silent, the one a `ValueError`
traceback out of the pre-app path and the other an empty picker whose next edit
wrote the blank back. An unknown *strength* is different and still falls back:
it has a defined default that renders and round-trips, which a label has not.

A labelling has two homes, and the split is the whole lifecycle. **Working
state** is a scratch CSV under `<corpus>/tmp_labels/`, rewritten atomically
on every single edit, so the file is the state and the page is only a view of
it; closing or reloading the tab costs nothing, and relaunching loads it back.
It is gitignored scratch and nothing downstream should read it, and it
resolves through `corpus_root` exactly as a committed label does -- one
resolution, so a relaunch from a different worktree finds the same work.

**Commit** promotes that scratch into the dataset: it writes
`<corpus>/annotations/<track_id>.hand.json` and, if the audio is not already in
the corpus, copies it to `<corpus>/audio/<track_id><ext>` so the track is
locatable the same way a downloaded one is. Labels are committed to git, songs
are not -- the narrow `.gitignore` exception admits `*.hand.json` and nothing
else beside it. Commit places files and performs **no git actions**; staging is
the owner's. It is idempotent: re-committing overwrites the same path
atomically.

**`segments.json` is never written, appended to or rewritten.** The frozen
benchmark checks its provenance by hashing that file, so its bytes are
inviolable; a hand label is always a new sibling file.

The committed shape is the corpus's own section shape plus one key::

    {"schema": 1, "source": "hand_label", "id": "hand-<sha256(audio)[:12]>",
     "title": "Artist - Track", "artist": "Artist" | null,
     "audio": "<file in the corpus audio dir>", "duration": <seconds>,
     "sections": [{"name": "<label>", "start": <s>, "end": <s>,
                   "strength": "major" | "minor"}, ...]}

**A track is identified by its bytes, not by its filename, and that decides the
`id`.** Commit hashes the audio and looks it up in the corpus's own
`checksums.sha256` (falling back to hashing only same-size files in `audio/`,
since that record is written by a validation run and can lag the downloader). If
the corpus already holds this recording, the label is filed under the **native**
id and nothing is copied -- `<native>.hand.json` sits beside the published entry
for that track and takes precedence over it. Only genuinely new audio gets a
`hand-<sha256[:12]>` id and a copy into `audio/`. So the id says which case it
is: a `hand-` prefix means new audio, anything else means an override.

Being content-addressed also matters for the split assignment, which hashes the
id: renaming a file must not move a track between train and val.

The `title` is required, and **commit is blocked until it is in `Artist - Track`
form**, because the benchmark's artist-exclusion guard reads it:
`select_eval_set.artist_of` takes everything before the dash, so a track without
one is never checked for contamination and nothing anywhere records that it was
skipped. A release that genuinely has no artist credit is admitted by ticking
"no artist", which stores `"artist": null` -- the difference between "there is
none" and "nobody filled it in" is the whole reason the field is written down
rather than re-derived. This started as a warning and was not enough: the first
real hand label went in as `is it beautiful`.

`name`/`start`/`end` are exactly what `parse_sections` in
`raveform/raveform_fetch_annotations.py` reads, so a consumer can treat a hand
label as one more track record. `strength` is the extra: it describes the
transition *into* that section and has no counterpart in the published data.
Sections are contiguous and sorted -- each one ends where the next begins, and
the last ends at the track duration.

Placing those two files is all this tool does. Making the track
*dataset-complete* -- the beat grid, the manifest row -- belongs to a corpus
admission step, and Commit calls it if this branch has one::

    training/hand_label_admission.py
        def admit(track_id: str, audio: Path, labels: Path, corpus: Path) -> str

The return value is shown to the owner verbatim. A missing module, or one
without that function, is an ordinary state and not an error: Commit still
places the files and says `admission pending`, naming the command to run. An
admission that raises is reported rather than swallowed, because the files are
already placed and a half-admitted track must not look finished.

A beat grid is drawn under the waveform when one exists, so a generated grid can
be eyeballed against the audio at labelling time. It is read from
`<corpus>/annotations/beats/<track_id>.hand.beat.csv` -- the published grids are
named by a corpus key this tool never sees, so a generated one is the only kind
it can find. Columns are the published ones, `time` and `downbeat`.
"""
import csv
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import dash
import flask
import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, Patch, State, callback_context, dcc, html

LABELS = ['intro', 'altintro', 'buildup', 'breakdown', 'bridge', 'drop',
          'cooldown', 'outro', 'altoutro']
STRENGTHS = ['major', 'minor']
DEFAULT_STRENGTH = STRENGTHS[0]
LABEL_COLORS = {
    'intro':     '#1565c0',
    'altintro':  '#0d47a1',
    'buildup':   '#e65100',
    'breakdown': '#7b1fa2',
    'bridge':    '#4a148c',
    'drop':      '#b71c1c',
    'cooldown':  '#00838f',
    'outro':     '#283593',
    'altoutro':  '#4527a0',
}

DARK_BG = '#0d1117'
CARD_BG = '#111827'
BORDER  = '#1e2937'
MUTED   = '#6e7681'
TEXT    = '#c9d1d9'

DECODE_RATE = 22050
FRAME = 1024
HOP = 512
FFT_CHUNK = 512
TICK_MS = 250
NUDGES = (-0.5, -0.1, 0.1, 0.5)
TIME_DECIMALS = 3


TMP_LABELS_DIR_NAME = 'tmp_labels'
HAND_LABEL_SUFFIX = '.hand.json'
HAND_LABEL_SCHEMA = 1
HAND_ID_PREFIX = 'hand-'
HAND_ID_LENGTH = 12
CHECKSUMS_FILE = 'checksums.sha256'
ARTIST_SEPARATOR = ' - '
_YOUTUBE_ID = re.compile(r'^[A-Za-z0-9_-]{11}$')


def tmp_labels_dir() -> Path:
    """Working state lives with the corpus, for the reason the labels do.

    Anything resolved against this file's own directory is worktree-local, while
    a committed label resolves through `corpus_root` to wherever the corpus
    actually is. Those two answers differ in a linked worktree, and the failure
    is silent: relaunching from a different checkout presents an empty labelling
    while the real one sits in the other tree. One resolution, so there is one
    place the work can be.
    """
    return corpus_dir() / TMP_LABELS_DIR_NAME


def labels_path(audio_path: str) -> Path:
    return tmp_labels_dir() / f'{Path(audio_path).name}.labels.csv'


def audio_digest(audio_path: str) -> str:
    digest = hashlib.sha256()
    with open(audio_path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: Path) -> dict:
    """The corpus's recorded content baseline, as {digest: name}."""
    recorded = {}
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            digest, _, name = line.strip().partition(' ')
            name = name.strip().lstrip('*')
            if digest and name:
                recorded[digest] = name
    return recorded


def locate_in_corpus(digest: str, size: int) -> Path:
    """The corpus file holding exactly these bytes, or None.

    Identity is content, never a filename: the same recording arrives here under
    whatever the owner called it. `checksums.sha256` is the cheap path because it
    is the corpus's own record of what it already has. It can be stale -- it is
    written by a validation run, not by the downloader -- so a miss there still
    falls through to the size scan, which only has to hash files that could
    possibly match. A missed match would copy audio the corpus already holds and
    would file the labels under a new id instead of overriding the published
    track, which is the expensive mistake here; a redundant hash is not.
    """
    try:
        audio = corpus_dir() / 'audio'
        checksums = corpus_dir() / CHECKSUMS_FILE
        if checksums.exists():
            name = read_checksums(checksums).get(digest)
            if name:
                found = audio / Path(name).name
                if found.exists():
                    return found
        for candidate in sorted(audio.glob('*')):
            if (candidate.is_file() and candidate.stat().st_size == size
                    and audio_digest(str(candidate)) == digest):
                return candidate
    except OSError:
        return None
    return None


def resolve_identity(audio_path: str) -> tuple:
    """The track id these bytes belong under, and their home if the corpus has one.

    A track the corpus already holds keeps its native id, so the hand label lands
    beside the published one as `<native>.hand.json` and takes precedence over
    it. Only genuinely new audio gets a `hand-` id -- content-addressed because
    splits hash the id and a rename must not move a track between train and val,
    and prefixed because new audio must never be mistakable for a downloaded row.
    """
    digest = audio_digest(audio_path)
    native = locate_in_corpus(digest, Path(audio_path).stat().st_size)
    if native is not None:
        return native.stem, native
    return f'{HAND_ID_PREFIX}{digest[:HAND_ID_LENGTH]}', None


def default_title(audio_path: str) -> str:
    """A prefill, not an answer -- the owner is expected to correct it."""
    stem = Path(audio_path).stem
    head, _, tail = stem.rpartition('.')
    if head and _YOUTUBE_ID.match(tail):
        stem = head
    return re.sub(r'[_\s]+', ' ', stem).strip()


def corpus_dir() -> Path:
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import corpus_root

    return corpus_root.corpus_dir()


def annotations_dir() -> Path:
    return corpus_dir() / 'annotations'


def hand_label_path(identifier: str) -> Path:
    return annotations_dir() / f'{identifier}{HAND_LABEL_SUFFIX}'


def corpus_audio_path(identifier: str, audio_path: str) -> Path:
    return corpus_dir() / 'audio' / f'{identifier}{Path(audio_path).suffix}'


def beat_grid(audio_path: str) -> list:
    """Beat times and downbeat flags for this track, or [] if it has no grid.

    Only `<track_id>.hand.beat.csv` is read. The published grids are named by a
    corpus key this tool never sees, so a generated grid is the only kind it can
    find -- for a native track as much as for a new one.
    """
    try:
        found = sorted((annotations_dir() / 'beats').glob(
            f'{resolve_identity(audio_path)[0]}.hand.beat.csv'))
    except OSError:
        return []
    if not found:
        return []
    try:
        with open(found[0], 'r', encoding='utf-8', newline='') as handle:
            return [(float(row['time']), int(row['downbeat']))
                    for row in csv.DictReader(handle)]
    except (OSError, KeyError, ValueError):
        return []


def title_artist(title: str, no_artist: bool = False) -> str:
    """The artist the guard will read out of this title, or None if there is none.

    `None` is a statement, not a gap: it means the owner said this release has no
    artist credit, rather than that nobody filled the field in.
    """
    head, separator, _ = title.strip().partition(ARTIST_SEPARATOR)
    return None if (no_artist or not separator) else head.strip()


def to_annotation(sections: list, duration: float, identifier: str,
                  audio_name: str, title: str, no_artist: bool = False) -> dict:
    ordered = normalise(sections)
    spans = []
    for index, entry in enumerate(ordered):
        end = (ordered[index + 1]['start'] if index + 1 < len(ordered)
               else max(duration, entry['start']))
        spans.append({'name': entry['label'], 'start': entry['start'],
                      'end': round(end, TIME_DECIMALS),
                      'strength': entry['strength']})
    return {'schema': HAND_LABEL_SCHEMA, 'source': 'hand_label',
            'id': identifier, 'title': title.strip(),
            'artist': title_artist(title, no_artist), 'audio': audio_name,
            'duration': round(float(duration), TIME_DECIMALS),
            'sections': spans}


def from_annotation(record: dict) -> list:
    return normalise([{'start': span['start'], 'label': span['name'],
                       'strength': span.get('strength', DEFAULT_STRENGTH)}
                      for span in record['sections']])


ADMISSION_MODULE = 'hand_label_admission'
ADMISSION_ENTRY = 'admit'
ADMISSION_HINT = 'python training/hand_label_admission.py <track_id>'


def admit_track(identifier: str, audio: Path, labels: Path) -> tuple:
    """Hand the placed files to the corpus admission step, if this branch has one.

    The rest of dataset-completeness -- the beat grid and the manifest row --
    belongs to that module, not here. Its absence is an ordinary state rather
    than an error: this tool ships before it and must keep working after.
    """
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        module = __import__(ADMISSION_MODULE)
        entry = getattr(module, ADMISSION_ENTRY)
    except (ImportError, AttributeError):
        return False, f'admission pending — run `{ADMISSION_HINT}`'
    try:
        return True, str(entry(track_id=identifier, audio=audio, labels=labels,
                               corpus=corpus_dir()))
    except Exception as error:
        return False, f'admission FAILED ({type(error).__name__}: {error})'


def commit_labels(audio_path: str, sections: list, duration: float,
                  title: str, no_artist: bool = False) -> dict:
    identifier, native = resolve_identity(audio_path)
    copied = native is None
    if native is not None:
        home = native
    else:
        home = corpus_audio_path(identifier, audio_path)
        home.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, home)
    path = hand_label_path(identifier)
    record = to_annotation(sections, duration, identifier, home.name, title,
                           no_artist)
    write_atomically(path, json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    admitted, admission = admit_track(identifier, home, path)
    return {'labels': path, 'audio': home, 'copied': copied,
            'sections': len(record['sections']), 'title': record['title'],
            'artist': record['artist'],
            'admitted': admitted, 'admission': admission}


def commit_status(result: dict) -> str:
    audio = (f'audio copied → {result["audio"]}' if result['copied']
             else f'audio already at {result["audio"]}')
    artist = (f'artist {result["artist"]!r}' if result['artist']
              else 'recorded as having NO artist, so no eval-set artist can be '
                   'excluded on it')
    return (f'committed ✓ {result["sections"]} sections · {result["title"]!r} · '
            f'{artist} → {result["labels"]} · {audio} · {result["admission"]}')


def clock_text(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f'{minutes}min {secs}sec' if minutes else f'{secs}sec'


def normalise(sections: list) -> list:
    rounded = [{'start': max(0.0, round(float(s['start']), TIME_DECIMALS)),
                'label': s['label'],
                'strength': s.get('strength') or DEFAULT_STRENGTH}
               for s in sections]
    out, seen = [], set()
    for entry in sorted(rounded, key=lambda s: s['start']):
        if entry['start'] in seen:
            continue
        seen.add(entry['start'])
        out.append(entry)
    return out


def add_boundary(sections: list, start: float, label: str,
                 strength: str = DEFAULT_STRENGTH) -> list:
    start = max(0.0, round(float(start), TIME_DECIMALS))
    kept = [s for s in sections
            if round(float(s['start']), TIME_DECIMALS) != start]
    return normalise(kept + [{'start': start, 'label': label,
                              'strength': strength}])


def clamp_time(seconds: float, duration: float = 0.0) -> float:
    """Onto the track, at both ends -- a section past the end renders nowhere.

    A duration of zero means the caller does not know one, which is the offline
    case rather than a track of no length.
    """
    seconds = max(0.0, round(float(seconds), TIME_DECIMALS))
    if duration > 0:
        return min(seconds, round(float(duration), TIME_DECIMALS))
    return seconds


def boundary_at(sections: list, start: float) -> dict:
    for entry in sections:
        if round(float(entry['start']), TIME_DECIMALS) == start:
            return entry
    return None


def format_csv(sections: list) -> str:
    """The third column is written all-or-nothing.

    A labelling with no minor boundary stays the two-column file it was, and
    either shape round-trips.
    """
    rows = normalise(sections)
    if all(s['strength'] == DEFAULT_STRENGTH for s in rows):
        return ''.join(f'{s["start"]:.{TIME_DECIMALS}f},{s["label"]}\n'
                       for s in rows)
    return ''.join(f'{s["start"]:.{TIME_DECIMALS}f},{s["label"]},{s["strength"]}\n'
                   for s in rows)


class LabelFileError(ValueError):
    """A working file that cannot be read, named rather than raised blind."""


def parse_csv(text: str) -> list:
    sections = []
    for number, line in enumerate(text.splitlines(), start=1):
        fields = [field.strip() for field in line.split(',')]
        if len(fields) < 2 or not fields[0] or not fields[1]:
            continue
        try:
            start = float(fields[0])
        except ValueError:
            raise LabelFileError(
                f'line {number}: {fields[0]!r} is not a time in seconds') from None
        if fields[1] not in LABELS:
            raise LabelFileError(
                f'line {number}: {fields[1]!r} is not one of '
                f'{", ".join(LABELS)}')
        strength = fields[2] if len(fields) > 2 else ''
        sections.append({
            'start': start, 'label': fields[1],
            'strength': strength if strength in STRENGTHS else DEFAULT_STRENGTH})
    return normalise(sections)


def load_labels(audio_path: str) -> list:
    path = labels_path(audio_path)
    if not path.exists():
        return normalise([{'start': 0.0, 'label': LABELS[0]}])
    return parse_csv(path.read_text(encoding='utf-8'))


def disk_labels(audio_path: str) -> list:
    """What the file says right now, or None if there is no file yet."""
    return load_labels(audio_path) if labels_path(audio_path).exists() else None


def write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def save_labels(audio_path: str, sections: list) -> Path:
    """Write the working labelling -- the file is the state, the page is a view.

    Every mutation calls this, so a killed process or a reloaded tab can only
    ever lose the edit in flight, never the ones already made.
    """
    path = labels_path(audio_path)
    write_atomically(path, format_csv(sections))
    return path


def saved_status(path: Path, sections: list) -> str:
    return (f'saved ✓ {time.strftime("%H:%M:%S")} · '
            f'{len(sections)} sections → {path}')


def apply_edit(audio_path: str, trigger, sections: list, cursor: float = 0.0,
               new_label: str = None, new_strength: str = DEFAULT_STRENGTH,
               row_labels: list = (), row_strengths: list = (),
               duration: float = 0.0, title: str = '',
               no_artist: bool = False) -> tuple:
    """Apply one UI event and write the result. `None` means "leave it alone".

    The file is the state, so a page whose sections disagree with it has lost
    the race -- a second tab, or one that has been open across another tab's
    edits. Applying its event would rewrite the file as that page's stale list
    plus this one edit, silently dropping every boundary made in between, so a
    disagreement is refused with a message instead. A file that cannot be parsed
    at all is refused the same way, because overwriting it would destroy the
    only copy of whatever the owner was trying to repair.
    """
    try:
        on_disk = disk_labels(audio_path)
    except LabelFileError as error:
        return None, (f'edit refused: {labels_path(audio_path)} is unreadable '
                      f'-- {error}')
    if on_disk is not None and on_disk != normalise(sections):
        return None, ('edit refused: this page is older than the labels on '
                      'disk -- reload it to catch up, nothing was changed')
    if trigger == 'save':
        return None, saved_status(save_labels(audio_path, sections), sections)
    if trigger == 'commit':
        title = (title or '').strip()
        if not title:
            return None, ('commit refused: a title is required — the dataset\'s '
                          'artist guard reads it, and a track without one is '
                          'invisible to the benchmark contamination check')
        if ARTIST_SEPARATOR not in title and not no_artist:
            return None, (
                f'commit refused: title needs "Artist{ARTIST_SEPARATOR}Track", '
                f'or tick "no artist" if it genuinely has none. The benchmark '
                f'keeps eval-set artists out of training by parsing the artist '
                f'out of this title, so a track without one is never checked '
                f'for contamination and nothing anywhere says so')
        return None, commit_status(
            commit_labels(audio_path, sections, duration, title, no_artist))
    if isinstance(trigger, dict) and not 0 <= trigger['index'] < len(sections):
        return None, None
    if trigger == 'mark':
        updated = add_boundary(sections, clamp_time(cursor or 0.0, duration),
                               new_label or LABELS[0], new_strength)
    elif isinstance(trigger, dict) and trigger['type'] == 'row-nudge':
        updated = list(sections)
        moved = updated.pop(trigger['index'])
        target = clamp_time(moved['start'] + trigger['delta'], duration)
        blocking = boundary_at(updated, target)
        if blocking is not None:
            return None, (f'nudge refused: {blocking["label"]} already starts '
                          f'at {target:.{TIME_DECIMALS}f} s -- moving onto a '
                          f'boundary would delete it')
        updated = add_boundary(updated, target, moved['label'],
                               moved['strength'])
    elif isinstance(trigger, dict) and trigger['type'] == 'row-del':
        updated = [s for i, s in enumerate(sections) if i != trigger['index']]
    elif isinstance(trigger, dict) and trigger['type'] in ('row-label',
                                                           'row-strength'):
        values = (row_labels if trigger['type'] == 'row-label' else row_strengths)
        field = 'label' if trigger['type'] == 'row-label' else 'strength'
        if trigger['index'] >= len(values):
            return None, None
        updated = list(sections)
        updated[trigger['index']] = dict(updated[trigger['index']],
                                         **{field: values[trigger['index']]})
    else:
        return None, None
    updated = normalise(updated)
    if updated == sections:
        return None, None
    return updated, saved_status(save_labels(audio_path, updated), updated)


def decode_mono(audio_path: str, rate: int = DECODE_RATE) -> np.ndarray:
    try:
        proc = subprocess.run(
            ['ffmpeg', '-nostdin', '-v', 'error', '-i', audio_path,
             '-f', 'f32le', '-ac', '1', '-ar', str(rate), '-'],
            capture_output=True, stdin=subprocess.DEVNULL,
        )
    except OSError as error:
        raise SystemExit(f'ffmpeg could not run: {error}')
    if proc.returncode != 0:
        raise SystemExit(f'ffmpeg failed on {audio_path}: '
                         f'{proc.stderr.decode("utf-8", "replace").strip()}')
    return np.frombuffer(proc.stdout, dtype='<f4')


def _unit(values: np.ndarray) -> np.ndarray:
    ceiling = float(np.percentile(values, 99.5)) if len(values) else 0.0
    if ceiling <= 0.0:
        return np.zeros_like(values)
    return np.clip(values / ceiling, 0.0, 1.0)


def _spectral_flux(windows: np.ndarray) -> np.ndarray:
    taper = np.hanning(windows.shape[1]).astype(np.float32)
    flux = np.empty(len(windows), dtype=np.float32)
    previous = None
    for start in range(0, len(windows), FFT_CHUNK):
        block = windows[start:start + FFT_CHUNK] * taper
        magnitude = np.abs(np.fft.rfft(block, axis=1)).astype(np.float32)
        deltas = np.diff(magnitude, axis=0,
                         prepend=magnitude[:1] if previous is None else previous)
        flux[start:start + len(magnitude)] = np.maximum(deltas, 0.0).sum(axis=1)
        previous = magnitude[-1:]
    return flux


class Track:
    def __init__(self, audio_path: str):
        samples = decode_mono(audio_path)
        self.duration = len(samples) / DECODE_RATE
        if len(samples) < FRAME:
            samples = np.zeros(FRAME, dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP]
        self.times = np.arange(len(windows)) * HOP / DECODE_RATE
        self.envelope = _unit(np.abs(windows).max(axis=1))
        self.flux = _unit(_spectral_flux(windows))


WAVE_MID, WAVE_HALF = 0.22, 0.19
FLUX_BASE, FLUX_SPAN = 0.46, 0.38
RIBBON_LOW, RIBBON_HIGH = 0.88, 1.0


def section_shapes(track: Track, sections: list) -> tuple:
    """Everything an edit can move: the playhead, the spans and their labels.

    Split out from the figure because it is the only part that changes. The
    traces behind it are the whole decode and are the same on every edit.
    """
    shapes = [dict(type='line', xref='x', yref='paper', layer='above',
                   x0=0.0, x1=0.0, y0=0.0, y1=1.0,
                   line=dict(color='#ffffff', width=1.5))]
    annotations = []
    for index, entry in enumerate(sections):
        start = entry['start']
        end = (sections[index + 1]['start'] if index + 1 < len(sections)
               else max(track.duration, start))
        if end <= start:
            continue
        color = LABEL_COLORS.get(entry['label'], MUTED)
        shapes.append(dict(type='rect', xref='x', yref='paper', layer='below',
                           x0=start, x1=end, y0=0.0, y1=RIBBON_LOW,
                           fillcolor=color, opacity=0.14, line_width=0))
        shapes.append(dict(type='rect', xref='x', yref='paper', layer='below',
                           x0=start, x1=end, y0=RIBBON_LOW, y1=RIBBON_HIGH,
                           fillcolor=color, opacity=0.85, line_width=0))
        minor = entry['strength'] == 'minor'
        shapes.append(dict(type='line', xref='x', yref='paper', layer='below',
                           x0=start, x1=start, y0=0.0, y1=RIBBON_HIGH,
                           line=dict(color=color, width=1.0 if minor else 2.0,
                                     dash='dash' if minor else 'dot')))
        annotations.append(dict(
            x=(start + end) / 2, y=0.94, xref='x', yref='paper',
            text=f'{entry["label"]} ·minor' if minor else entry['label'],
            showarrow=False,
            font=dict(color='rgba(255,255,255,0.92)', size=10, family='monospace')))
    return shapes, annotations


def render_patch(track: Track, sections: list) -> Patch:
    """An edit ships the spans, never the traces.

    The three full-track traces are the expensive half of the figure and no edit
    can touch them, so a partial update leaves them where they are -- and leaves
    the browser's webgl contexts and pan position alone with them.
    """
    shapes, annotations = section_shapes(track, sections)
    patched = Patch()
    patched['layout']['shapes'] = shapes
    patched['layout']['annotations'] = annotations
    return patched


def build_figure(track: Track, sections: list, beats: list = ()) -> go.Figure:
    """The whole figure, which is built on a page load and never on an edit.

    The waveform is two mirrored outlines rather than a filled band because
    scattergl draws no fill, and svg traces at this point count are too slow
    to pan.
    """
    shapes, annotations = section_shapes(track, sections)
    figure = go.Figure()
    for sign in (1.0, -1.0):
        figure.add_trace(go.Scattergl(
            x=track.times, y=WAVE_MID + sign * WAVE_HALF * track.envelope,
            mode='lines', line=dict(color='rgba(88,166,255,0.75)', width=1),
            hoverinfo='skip'))
    figure.add_trace(go.Scattergl(
        x=track.times, y=FLUX_BASE + FLUX_SPAN * track.flux,
        mode='lines', line=dict(color='rgba(63,185,80,0.9)', width=1),
        hoverinfo='skip'))
    for downbeat, height, size, alpha in ((0, 0.035, 9, 0.40),
                                          (1, 0.055, 20, 0.85)):
        times = [t for t, flag in beats if bool(flag) == bool(downbeat)]
        if times:
            figure.add_trace(go.Scattergl(
                x=times, y=[height] * len(times), mode='markers',
                marker=dict(symbol='line-ns', size=size,
                            line=dict(color=f'rgba(168,218,220,{alpha})',
                                      width=1.4)),
                hoverinfo='skip'))
    figure.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=[0.0, track.duration], gridcolor='#1a2332',
                   color=MUTED, ticksuffix='s', showline=False),
        yaxis=dict(range=[0.0, 1.0], showticklabels=False, showgrid=False,
                   fixedrange=True),
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        height=330, margin=dict(l=8, r=8, t=8, b=34),
        dragmode='pan', uirevision='labels', showlegend=False,
    )
    return figure


BUTTON_STYLE = {
    'background': CARD_BG, 'color': TEXT, 'border': f'1px solid {BORDER}',
    'borderRadius': '6px', 'padding': '6px 12px', 'fontFamily': 'monospace',
    'cursor': 'pointer',
}
CELL_STYLE = {'padding': '4px 10px', 'borderBottom': f'1px solid {BORDER}'}


def _nudge_button(index: int, delta: float) -> html.Button:
    return html.Button(
        f'{delta:+.1f}', id={'type': 'row-nudge', 'index': index, 'delta': delta},
        style=dict(BUTTON_STYLE, padding='3px 8px', marginRight='4px'))


def build_rows(sections: list) -> list:
    header = html.Tr([html.Th(text, style=dict(CELL_STYLE, color=MUTED,
                                               textAlign='left'))
                      for text in ('#', 'start', '', 'label', 'strength',
                                   'nudge', '')])
    rows = [header]
    for index, entry in enumerate(sections):
        rows.append(html.Tr([
            html.Td(str(index), style=dict(CELL_STYLE, color=MUTED)),
            html.Td(f'{entry["start"]:.3f} s', style=dict(CELL_STYLE, color=TEXT)),
            html.Td(clock_text(entry['start']), style=dict(CELL_STYLE, color=MUTED)),
            html.Td(dcc.Dropdown(
                id={'type': 'row-label', 'index': index}, options=LABELS,
                value=entry['label'], clearable=False,
                style={'width': '150px', 'color': '#000000'}), style=CELL_STYLE),
            html.Td(dcc.Dropdown(
                id={'type': 'row-strength', 'index': index}, options=STRENGTHS,
                value=entry['strength'], clearable=False,
                style={'width': '110px', 'color': '#000000'}), style=CELL_STYLE),
            html.Td([_nudge_button(index, delta) for delta in NUDGES],
                    style=CELL_STYLE),
            html.Td(html.Button('delete', id={'type': 'row-del', 'index': index},
                                style=dict(BUTTON_STYLE, padding='3px 10px',
                                           color='#f85149')),
                    style=CELL_STYLE),
        ]))
    return [html.Table(rows, style={'borderCollapse': 'collapse',
                                    'fontFamily': 'monospace', 'fontSize': '13px'})]


SINK_STORAGE_KEY = 'labelToolAudioSink'

SINK_JS = """
function (tick, chosen) {
    const audio = document.getElementById('player');
    const HOLD = window.dash_clientside.no_update;
    if (!window.__sink) {
        const sink = window.__sink = {status: 'reading devices…', ok: true,
                                      devices: [], bound: false};

        window.__sinkList = async function () {
            let found = await navigator.mediaDevices.enumerateDevices();
            if (!found.some(d => d.kind === 'audiooutput' && d.label)) {
                // Labels are blank until the page holds a media permission, and
                // an unnamed device is not something anyone can pick from.
                await navigator.mediaDevices.getUserMedia({audio: true})
                    .then(stream => stream.getTracks().forEach(t => t.stop()))
                    .catch(() => {});
                found = await navigator.mediaDevices.enumerateDevices();
            }
            sink.devices = found.filter(d => d.kind === 'audiooutput')
                .map(d => ({deviceId: d.deviceId,
                            label: d.label || 'unnamed (' + d.deviceId.slice(0, 8) + ')'}));
            if (!found.some(d => d.kind === 'audiooutput' && d.label)) {
                sink.ok = false;
                sink.status = 'device names are blocked — allow audio access to name them';
            } else if (!sink.chosen) {
                sink.status = 'system default';
            }
        };

        window.__sinkApply = async function (deviceId) {
            // Re-queried rather than closed over: this function outlives the
            // callback that made it, and #player is a node dash may replace.
            const audio = document.getElementById('player');
            if (!deviceId || !audio) { return; }
            if (!audio.setSinkId) {
                sink.ok = false;
                sink.status = 'this browser cannot switch outputs';
                return;
            }
            try {
                await audio.setSinkId(deviceId);
                sink.chosen = deviceId;
                sink.ok = true;
                const found = sink.devices.find(d => d.deviceId === deviceId);
                sink.status = found ? found.label : deviceId;
                window.localStorage.setItem(SINK_KEY, deviceId);
            } catch (error) {
                sink.ok = false;
                sink.status = error.name + ' — pick another';
            }
        };
        window.__sinkList();
    }

    const sink = window.__sink;
    if (audio && !sink.bound) {
        sink.bound = true;
        // Chrome wants a user gesture before it will move an element's output,
        // and pressing play is the one gesture this page is guaranteed to get.
        audio.addEventListener('play', () => {
            if (sink.pending) { window.__sinkApply(sink.pending); }
        });
    }

    const chip = {display: 'inline-block', marginLeft: '16px', padding: '3px 10px',
                  borderRadius: '6px', fontSize: '13px',
                  border: '1px solid ' + (sink.ok ? '#1e2937' : '#f85149'),
                  color: sink.ok ? '#6e7681' : '#f85149'};
    const options = sink.devices.map(d => ({label: d.label, value: d.deviceId}));

    let value = HOLD;
    if (!chosen && sink.devices.length) {
        // One global preference, restored once the device is actually present:
        // a remembered id that is not plugged in today must not blank the picker.
        const remembered = window.localStorage.getItem(SINK_KEY);
        if (remembered && sink.devices.some(d => d.deviceId === remembered)) {
            sink.pending = remembered;
            value = remembered;
        }
    }
    return ['output: ' + sink.status, chip, options, value];
}
""".replace('SINK_KEY', repr(SINK_STORAGE_KEY))

SINK_PICK_JS = """
function (deviceId) {
    if (deviceId) {
        window.__sink.pending = deviceId;
        window.__sinkApply(deviceId);
    }
    return window.dash_clientside.no_update;
}
"""

SEEK_SLOP_PX = 4

CURSOR_JS = """
function (tick) {
    const audio = document.getElementById('player');
    if (!audio) { return [0, '0.00 s', '0sec']; }
    const t = audio.currentTime || 0;
    const plot = document.querySelector('#timeline .js-plotly-plot');
    if (plot && window.Plotly) {
        window.Plotly.relayout(plot, {'shapes[0].x0': t, 'shapes[0].x1': t});
        if (!plot._seekBound) {
            plot._seekBound = true;
            // Plotly only emits plotly_click for clicks that land on a point, so
            // click-anywhere-to-seek has to convert the pixel itself.
            plot.addEventListener('mousedown', function (event) {
                plot._seekFrom = [event.clientX, event.clientY];
            });
            plot.addEventListener('click', function (event) {
                // The plot pans on drag, and releasing a pan fires click too --
                // so a seek is a click that did not travel.
                const from = plot._seekFrom;
                if (!from || Math.hypot(event.clientX - from[0],
                                        event.clientY - from[1]) > SEEK_SLOP) {
                    return;
                }
                const axis = plot._fullLayout && plot._fullLayout.xaxis;
                if (!axis) { return; }
                const box = plot.getBoundingClientRect();
                const seconds = axis.p2d(event.clientX - box.left - axis._offset);
                if (isFinite(seconds)) { audio.currentTime = Math.max(0, seconds); }
            });
        }
    }
    const minutes = Math.floor(t / 60), seconds = Math.floor(t % 60);
    return [t, t.toFixed(2) + ' s',
            minutes ? minutes + 'min ' + seconds + 'sec' : seconds + 'sec'];
}
""".replace('SEEK_SLOP', str(SEEK_SLOP_PX))


def audio_mimetype(audio_path: str) -> str:
    """Whatever the file is -- anything ffmpeg can decode is accepted here."""
    return mimetypes.guess_type(audio_path)[0] or 'application/octet-stream'


def build_app(audio_path: str, track: Track, beats: list = ()) -> dash.Dash:
    """The page, served from the file rather than from a launch-time snapshot.

    `app.layout` is a function because Dash serves it on every page load: as a
    single component instance it hands each reload the sections the process
    started with, and that stale list is the State every edit is applied to. One
    edit after a reload then rewrote the file as launch-time state plus that
    edit, destroying every boundary made in between. The layout reads the file,
    so a reload is a re-read and the page is only ever a view of it.
    """
    name = Path(audio_path).name
    app = dash.Dash(__name__, title=f'label · {name}')

    @app.server.route('/audio')
    def serve_audio():
        """conditional=True answers Range requests, which is the whole of what
        makes seeking inside an <audio> element work."""
        return flask.send_file(audio_path, mimetype=audio_mimetype(audio_path),
                               conditional=True)

    def serve_layout():
        sections = load_labels(audio_path)
        return html.Div([
            html.Div([
                html.Span(name, style={'color': TEXT}),
                html.Span(f'  ·  {clock_text(track.duration)}  ·  {labels_path(audio_path)}',
                          style={'color': MUTED}),
            ], style={'padding': '12px 20px', 'borderBottom': f'1px solid {BORDER}',
                      'fontSize': '13px'}),
            html.Div(html.Audio(id='player', src='/audio', controls=True,
                                style={'width': '100%'}),
                     style={'padding': '12px 20px 4px'}),
            html.Div([
                html.Div('0.00 s', id='cursor-seconds',
                         style={'fontSize': '32px', 'color': '#ffffff'}),
                html.Div('0sec', id='cursor-clock',
                         style={'fontSize': '18px', 'color': MUTED, 'marginLeft': '16px'}),
                html.Div(id='sink-chip'),
                dcc.Dropdown(id='sink-pick', clearable=False,
                             placeholder='audio output',
                             style={'width': '300px', 'color': '#000000',
                                    'marginLeft': '10px', 'display': 'inline-block',
                                    'verticalAlign': 'middle'}),
                html.Div([
                    dcc.Dropdown(id='new-label', options=LABELS, value=LABELS[0],
                                 clearable=False,
                                 style={'width': '150px', 'color': '#000000',
                                        'display': 'inline-block',
                                        'verticalAlign': 'middle'}),
                    dcc.RadioItems(id='new-strength', options=STRENGTHS,
                                   value=DEFAULT_STRENGTH, inline=True,
                                   style={'marginLeft': '10px'},
                                   labelStyle={'color': TEXT, 'fontSize': '13px',
                                               'marginRight': '6px'},
                                   inputStyle={'marginRight': '4px',
                                               'marginLeft': '8px'}),
                    html.Button('mark boundary at current time', id='mark',
                                style=dict(BUTTON_STYLE, marginLeft='10px')),
                    html.Button('save now', id='save',
                                style=dict(BUTTON_STYLE, marginLeft='10px',
                                           color='#3fb950')),
                    dcc.Input(id='title', value=default_title(audio_path),
                              placeholder='Artist - Track', debounce=False,
                              style={'marginLeft': '10px', 'width': '260px',
                                     'padding': '6px 10px', 'borderRadius': '6px',
                                     'background': CARD_BG, 'color': TEXT,
                                     'border': f'1px solid {BORDER}',
                                     'fontFamily': 'monospace'}),
                    dcc.Checklist(id='no-artist', options=[{'label': 'no artist',
                                                            'value': 'yes'}],
                                  value=[], inline=True,
                                  style={'marginLeft': '8px'},
                                  labelStyle={'color': TEXT, 'fontSize': '13px'},
                                  inputStyle={'marginRight': '4px'}),
                    html.Button('commit to dataset', id='commit',
                                style=dict(BUTTON_STYLE, marginLeft='10px',
                                           color='#58a6ff')),
                ], style={'marginLeft': 'auto', 'display': 'flex',
                          'alignItems': 'center'}),
            ], style={'display': 'flex', 'alignItems': 'center',
                      'padding': '4px 20px 12px'}),
            dcc.Graph(id='timeline', figure=build_figure(track, sections, beats),
                      config={'displayModeBar': False, 'scrollZoom': True}),
            html.Div(id='status', style={'padding': '10px 20px', 'color': MUTED,
                                         'fontSize': '13px'}),
            html.Div(build_rows(sections), id='table', style={'padding': '0 20px 40px'}),
            dcc.Store(id='sections', data=sections),
            dcc.Store(id='cursor', data=0.0),
            dcc.Store(id='sink-echo'),
            dcc.Interval(id='tick', interval=TICK_MS),
        ], style={'background': DARK_BG, 'minHeight': '100vh',
                  'fontFamily': 'monospace'})

    app.layout = serve_layout

    app.clientside_callback(
        CURSOR_JS,
        Output('cursor', 'data'),
        Output('cursor-seconds', 'children'),
        Output('cursor-clock', 'children'),
        Input('tick', 'n_intervals'),
    )

    app.clientside_callback(
        SINK_JS,
        Output('sink-chip', 'children'),
        Output('sink-chip', 'style'),
        Output('sink-pick', 'options'),
        Output('sink-pick', 'value'),
        Input('tick', 'n_intervals'),
        State('sink-pick', 'value'),
    )

    app.clientside_callback(
        SINK_PICK_JS,
        Output('sink-echo', 'data'),
        Input('sink-pick', 'value'),
        prevent_initial_call=True,
    )

    @app.callback(
        Output('sections', 'data'),
        Output('status', 'children'),
        Input('mark', 'n_clicks'),
        Input('save', 'n_clicks'),
        Input('commit', 'n_clicks'),
        Input({'type': 'row-nudge', 'index': ALL, 'delta': ALL}, 'n_clicks'),
        Input({'type': 'row-del', 'index': ALL}, 'n_clicks'),
        Input({'type': 'row-label', 'index': ALL}, 'value'),
        Input({'type': 'row-strength', 'index': ALL}, 'value'),
        State('sections', 'data'),
        State('cursor', 'data'),
        State('new-label', 'value'),
        State('new-strength', 'value'),
        State('title', 'value'),
        State('no-artist', 'value'),
        prevent_initial_call=True,
    )
    def edit(mark_clicks, save_clicks, commit_clicks, nudges, deletes,
             row_labels, row_strengths, sections, cursor, new_label,
             new_strength, title, no_artist):
        updated, status = apply_edit(
            audio_path, callback_context.triggered_id, sections, cursor,
            new_label, new_strength, row_labels, row_strengths, track.duration,
            title, bool(no_artist))
        return (dash.no_update if updated is None else updated,
                dash.no_update if status is None else status)

    @app.callback(
        Output('table', 'children'),
        Output('timeline', 'figure'),
        Input('sections', 'data'),
        prevent_initial_call=True,
    )
    def render(sections):
        return build_rows(sections), render_patch(track, sections)

    return app


def launch(audio: str, port: int = 8070) -> None:
    """Read the labels first: a file this refuses is one only this can repair.

    Decoding a whole track before finding out costs the owner a minute for a
    message that was available immediately. The server runs with no reloader and
    no hot reload -- a restart would drop the page mid-labelling, and there is
    nothing here worth watching a filesystem for.
    """
    audio_path = str(Path(audio).resolve())
    if not Path(audio_path).exists():
        raise SystemExit(f'no such audio file: {audio_path}')

    try:
        sections = load_labels(audio_path)
    except LabelFileError as error:
        raise SystemExit(
            f'{labels_path(audio_path)} cannot be read: {error}\n'
            f'  fix or delete that line and relaunch — nothing was changed')

    print(f'  decoding {Path(audio_path).name} ...')
    track = Track(audio_path)
    print(f'  {track.duration:.1f} s, {len(sections)} sections loaded from '
          f'{labels_path(audio_path)}')
    beats = beat_grid(audio_path)
    print(f'  beat grid: {len(beats)} beats' if beats
          else '  beat grid: none for this track')
    print(f'\n  Labeler → http://localhost:{port}\n')
    build_app(audio_path, track, beats).run(
        port=port, debug=False, use_reloader=False, dev_tools_hot_reload=False)
