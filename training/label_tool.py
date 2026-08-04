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

A labelling has two homes, and the split is the whole lifecycle. **Working
state** is a scratch CSV under `training/data/tmp_labels/`, rewritten atomically
on every single edit, so the file is the state and the page is only a view of
it; closing or reloading the tab costs nothing, and relaunching loads it back.
It is gitignored scratch and nothing downstream should read it.

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
     "title": "Artist - Track", "audio": "<file in the corpus audio dir>",
     "duration": <seconds>,
     "sections": [{"name": "<label>", "start": <s>, "end": <s>,
                   "strength": "major" | "minor"}, ...]}

The `id` is content-addressed and prefixed, for two separate reasons: splits are
assigned by hashing the id, so a rename must not move a track between train and
val; and a hand track must never be mistakable for a downloaded corpus row. The
`title` is required rather than optional because the benchmark's artist-exclusion
guard reads it -- `select_eval_set.artist_of` takes everything before the dash --
and a track with no title is invisible to that guard, which is a contamination
hole rather than a cosmetic gap.

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
be eyeballed against the audio at labelling time. Two spellings are read from
`<corpus>/annotations/beats/`: the published `<key>.beat.csv`, and
`<track_id>.hand.beat.csv` for a generated one. Columns are the published ones,
`time` and `downbeat`.
"""
import csv
import hashlib
import json
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
from dash import ALL, Input, Output, State, callback_context, dcc, html

# The raw Raveform vocabulary, unfolded, minus `end` -- that one is a tail
# sentinel rather than a phase. Folding is a downstream decision; a labelling
# that has already folded cannot be unfolded again.
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


TMP_LABELS_DIR = Path(__file__).resolve().parent / 'data' / 'tmp_labels'
HAND_LABEL_SUFFIX = '.hand.json'
HAND_LABEL_SCHEMA = 1
HAND_ID_PREFIX = 'hand-'
HAND_ID_LENGTH = 12
_YOUTUBE_ID = re.compile(r'^[A-Za-z0-9_-]{11}$')


def labels_path(audio_path: str) -> Path:
    return TMP_LABELS_DIR / f'{Path(audio_path).name}.labels.csv'


def track_id(audio_path: str) -> str:
    """Content-addressed, and visibly not a YouTube id.

    Splits hash the track id, so it has to be stable -- a rename must not move a
    track between train and val. The audio's own digest is the only thing about
    a hand-labelled file that cannot drift. The `hand-` prefix is the other half:
    a hand track must never be mistakable for a downloaded corpus row, and that
    is worth more as a property of the data than as a rule written down.
    """
    digest = hashlib.sha256()
    with open(audio_path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return f'{HAND_ID_PREFIX}{digest.hexdigest()[:HAND_ID_LENGTH]}'


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
    corpus key this tool never sees, and a hand id can never be one -- so a
    generated grid is the only kind that can exist for a hand track.
    """
    try:
        found = sorted((annotations_dir() / 'beats').glob(
            f'{track_id(audio_path)}.hand.beat.csv'))
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


def to_annotation(sections: list, duration: float, identifier: str,
                  audio_name: str, title: str) -> dict:
    ordered = normalise(sections)
    spans = []
    for index, entry in enumerate(ordered):
        end = (ordered[index + 1]['start'] if index + 1 < len(ordered)
               else max(duration, entry['start']))
        spans.append({'name': entry['label'], 'start': entry['start'],
                      'end': round(end, TIME_DECIMALS),
                      'strength': entry['strength']})
    return {'schema': HAND_LABEL_SCHEMA, 'source': 'hand_label',
            'id': identifier, 'title': title.strip(), 'audio': audio_name,
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
                  title: str) -> dict:
    identifier = track_id(audio_path)
    home = corpus_audio_path(identifier, audio_path)
    copied = not home.exists()
    if copied:
        home.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, home)
    path = hand_label_path(identifier)
    record = to_annotation(sections, duration, identifier, home.name, title)
    write_atomically(path, json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    admitted, admission = admit_track(identifier, home, path)
    return {'labels': path, 'audio': home, 'copied': copied,
            'sections': len(record['sections']), 'title': record['title'],
            'admitted': admitted, 'admission': admission}


def commit_status(result: dict) -> str:
    audio = (f'audio copied → {result["audio"]}' if result['copied']
             else f'audio already at {result["audio"]}')
    # artist_of() reads everything before the dash; with no dash there is no
    # artist to exclude on, and the benchmark guard silently sees nothing.
    warning = ('' if ' - ' in result['title'] else
               ' · NOTE title has no "Artist - Track" dash, so the artist '
               'guard cannot read an artist from it')
    return (f'committed ✓ {result["sections"]} sections · {result["title"]!r} → '
            f'{result["labels"]} · {audio} · {result["admission"]}{warning}')


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


def format_csv(sections: list) -> str:
    rows = normalise(sections)
    # The third column is written all-or-nothing: a labelling with no minor
    # boundary stays the two-column file it was, and either shape round-trips.
    if all(s['strength'] == DEFAULT_STRENGTH for s in rows):
        return ''.join(f'{s["start"]:.{TIME_DECIMALS}f},{s["label"]}\n'
                       for s in rows)
    return ''.join(f'{s["start"]:.{TIME_DECIMALS}f},{s["label"]},{s["strength"]}\n'
                   for s in rows)


def parse_csv(text: str) -> list:
    sections = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) < 2 or not fields[0] or not fields[1]:
            continue
        strength = fields[2] if len(fields) > 2 else ''
        sections.append({
            'start': float(fields[0]), 'label': fields[1],
            'strength': strength if strength in STRENGTHS else DEFAULT_STRENGTH})
    return normalise(sections)


def load_labels(audio_path: str) -> list:
    path = labels_path(audio_path)
    if not path.exists():
        return normalise([{'start': 0.0, 'label': LABELS[0]}])
    return parse_csv(path.read_text(encoding='utf-8'))


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
               duration: float = 0.0, title: str = '') -> tuple:
    """Apply one UI event and write the result. `None` means "leave it alone"."""
    if trigger == 'save':
        return None, saved_status(save_labels(audio_path, sections), sections)
    if trigger == 'commit':
        if not (title or '').strip():
            return None, ('commit refused: a title is required — the dataset\'s '
                          'artist guard reads it, and a track without one is '
                          'invisible to the benchmark contamination check')
        return None, commit_status(
            commit_labels(audio_path, sections, duration, title))
    if isinstance(trigger, dict) and not 0 <= trigger['index'] < len(sections):
        return None, None
    if trigger == 'mark':
        updated = add_boundary(sections, cursor or 0.0,
                               new_label or LABELS[0], new_strength)
    elif isinstance(trigger, dict) and trigger['type'] == 'row-nudge':
        updated = list(sections)
        moved = updated.pop(trigger['index'])
        updated = add_boundary(updated, moved['start'] + trigger['delta'],
                               moved['label'], moved['strength'])
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


def build_figure(track: Track, sections: list, beats: list = ()) -> go.Figure:
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

    figure = go.Figure()
    # Two mirrored outlines rather than a filled band: scattergl draws no fill,
    # and svg traces at this point count are too slow to pan.
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
            plot.addEventListener('click', function (event) {
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
"""


def build_app(audio_path: str, track: Track, sections: list,
              beats: list = ()) -> dash.Dash:
    name = Path(audio_path).name
    app = dash.Dash(__name__, title=f'label · {name}')

    @app.server.route('/audio')
    def serve_audio():
        # conditional=True answers Range requests, which is the whole of what
        # makes seeking inside an <audio> element work.
        return flask.send_file(audio_path, mimetype='audio/mpeg', conditional=True)

    app.layout = html.Div([
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
        prevent_initial_call=True,
    )
    def edit(mark_clicks, save_clicks, commit_clicks, nudges, deletes,
             row_labels, row_strengths, sections, cursor, new_label,
             new_strength, title):
        updated, status = apply_edit(
            audio_path, callback_context.triggered_id, sections, cursor,
            new_label, new_strength, row_labels, row_strengths, track.duration,
            title)
        return (dash.no_update if updated is None else updated,
                dash.no_update if status is None else status)

    @app.callback(
        Output('table', 'children'),
        Output('timeline', 'figure'),
        Input('sections', 'data'),
        prevent_initial_call=True,
    )
    def render(sections):
        return build_rows(sections), build_figure(track, sections, beats)

    return app


def launch(audio: str, port: int = 8070) -> None:
    audio_path = str(Path(audio).resolve())
    if not Path(audio_path).exists():
        raise SystemExit(f'no such audio file: {audio_path}')

    print(f'  decoding {Path(audio_path).name} ...')
    track = Track(audio_path)
    sections = load_labels(audio_path)
    print(f'  {track.duration:.1f} s, {len(sections)} sections loaded from '
          f'{labels_path(audio_path)}')
    beats = beat_grid(audio_path)
    print(f'  beat grid: {len(beats)} beats' if beats
          else '  beat grid: none for this track')
    print(f'\n  Labeler → http://localhost:{port}\n')
    # No reloader and no hot reload: a restart would drop the page mid-labelling,
    # and there is nothing here worth watching a filesystem for.
    build_app(audio_path, track, sections, beats).run(
        port=port, debug=False, use_reloader=False, dev_tools_hot_reload=False)
