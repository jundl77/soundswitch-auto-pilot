"""Hand-label a song's sections in the browser.

Launched by `auto_pilot label <song.mp3> [--port 8070] [-o DEVICE]`, which is the
only entry point -- this module has no CLI of its own. `-o` takes an index from
`auto_pilot list`, resolves it to a device name here, and asks the browser to
route playback there on the first press of play; a chip says which device it
actually landed on, with a picker beside it when the match or the permission
fails. The track is decoded once with ffmpeg into a waveform envelope and a
spectral-flux curve, so the beat starts a boundary should land on are visible
rather than guessed; play the audio, click the timeline to seek, and press "mark
boundary" to cut a section at the playhead, with the major/minor toggle saying
how strong that transition is. Each section gets a row in the table with a label
picker, a strength picker, +/- nudge buttons and a delete, and shows as a
coloured span on the timeline. Every one of those edits writes
`<audio>.labels.csv` beside the file immediately and atomically -- one
`start_seconds,label[,strength]` line per section, sorted, LF-terminated, the
third column present only when some boundary is minor -- so the file is the
state and the page is only a view of it; closing or reloading the tab costs
nothing, and relaunching loads the file back.
"""
import os
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


def labels_path(audio_path: str) -> Path:
    return Path(f'{audio_path}.labels.csv')


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


def save_labels(audio_path: str, sections: list) -> Path:
    """Write the labelling, atomically -- the file is the state, the page is a view.

    Every mutation calls this, so a killed process or a reloaded tab can only
    ever lose the edit in flight, never the ones already made.
    """
    path = labels_path(audio_path)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(format_csv(sections))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return path


def saved_status(path: Path, sections: list) -> str:
    return (f'saved ✓ {time.strftime("%H:%M:%S")} · '
            f'{len(sections)} sections → {path}')


def apply_edit(audio_path: str, trigger, sections: list, cursor: float = 0.0,
               new_label: str = None, new_strength: str = DEFAULT_STRENGTH,
               row_labels: list = (), row_strengths: list = ()) -> tuple:
    """Apply one UI event and write the result. `None` means "leave it alone"."""
    if trigger == 'save':
        return None, saved_status(save_labels(audio_path, sections), sections)
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


def output_device_name(index: int, py_audio=None) -> str:
    """Resolve a pyaudio output index to its name, in `auto_pilot list`'s index space.

    The browser cannot consume a pyaudio index, so the name is all that crosses
    to the client -- it is the only identifier both APIs describe the device by.
    """
    owned = py_audio is None
    if owned:
        import pyaudio
        py_audio = pyaudio.PyAudio()
    try:
        count = py_audio.get_device_count()
        if not 0 <= index < count:
            raise SystemExit(f'no audio device at index {index} (0..{count - 1}) '
                             f'-- run `python auto_pilot list`')
        info = py_audio.get_device_info_by_index(index)
        if not info.get('maxOutputChannels'):
            raise SystemExit(f'device {index} ({info["name"]}) has no output '
                             f'channels -- run `python auto_pilot list`')
        return str(info['name'])
    finally:
        if owned:
            py_audio.terminate()


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


def build_figure(track: Track, sections: list) -> go.Figure:
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


SINK_JS = """
function (tick, wanted) {
    const HIDDEN = {display: 'none'};
    if (!wanted) { return ['', HIDDEN, [], HIDDEN]; }
    const audio = document.getElementById('player');
    if (!window.__sink) {
        window.__sink = {status: 'press play to route to ' + wanted,
                         ok: true, devices: [], bound: false};

        // MME truncates a device name at 31 characters and Chrome prefixes the
        // default with "Default - ", so the two APIs rarely spell it the same.
        window.__sinkMatch = function (devices, target) {
            const clean = s => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
            const want = clean(target);
            let best = null, bestScore = 0;
            for (const device of devices) {
                const have = clean(device.label);
                if (!have) { continue; }
                let score;
                if (have === want) { score = 1000; }
                else if (have.includes(want) || want.includes(have)) {
                    score = 500 + Math.min(have.length, want.length);
                } else {
                    const tokens = new Set(want.split(' '));
                    score = have.split(' ').filter(t => tokens.has(t)).length;
                }
                if (score > bestScore) { best = device; bestScore = score; }
            }
            return bestScore > 0 ? best : null;
        };

        window.__sinkOutputs = async function () {
            let found = await navigator.mediaDevices.enumerateDevices();
            if (!found.some(d => d.kind === 'audiooutput' && d.label)) {
                // Labels are blank until the page holds a media permission.
                await navigator.mediaDevices.getUserMedia({audio: true})
                    .then(s => s.getTracks().forEach(t => t.stop()))
                    .catch(() => {});
                found = await navigator.mediaDevices.enumerateDevices();
            }
            return found.filter(d => d.kind === 'audiooutput')
                        .map(d => ({deviceId: d.deviceId, label: d.label || d.deviceId}));
        };

        window.__sinkApply = async function (target) {
            const sink = window.__sink;
            if (!audio.setSinkId) {
                sink.ok = false;
                sink.status = 'this browser has no setSinkId';
                return;
            }
            try {
                sink.devices = await window.__sinkOutputs();
                if (!sink.devices.some(d => d.label && d.label !== d.deviceId)) {
                    sink.ok = false;
                    sink.status = 'device names blocked - allow audio permission, or pick below';
                    return;
                }
                const match = window.__sinkMatch(sink.devices, target);
                if (!match) {
                    sink.ok = false;
                    sink.status = 'no device matches "' + target + '" - pick below';
                    return;
                }
                await audio.setSinkId(match.deviceId);
                sink.ok = true;
                sink.status = match.label;
            } catch (error) {
                sink.ok = false;
                sink.status = error.name + ' - pick below';
            }
        };
    }
    const sink = window.__sink;
    if (audio && !sink.bound) {
        sink.bound = true;
        // setSinkId wants a user gesture, and pressing play is the one gesture
        // this page is guaranteed to get.
        audio.addEventListener('play', () => window.__sinkApply(wanted));
    }
    const chip = {display: 'inline-block', marginLeft: '16px', padding: '3px 10px',
                  borderRadius: '6px', fontSize: '13px',
                  border: '1px solid ' + (sink.ok ? '#1e2937' : '#f85149'),
                  color: sink.ok ? '#6e7681' : '#f85149'};
    const options = sink.devices.map(d => ({label: d.label, value: d.deviceId}));
    return ['output: ' + sink.status, chip, options,
            options.length ? {width: '320px', color: '#000000', marginLeft: '10px',
                              display: 'inline-block', verticalAlign: 'middle'}
                           : HIDDEN];
}
"""

SINK_PICK_JS = """
function (deviceId) {
    const audio = document.getElementById('player');
    if (deviceId && audio && audio.setSinkId) {
        audio.setSinkId(deviceId).then(() => {
            const picked = window.__sink.devices.find(d => d.deviceId === deviceId);
            window.__sink.ok = true;
            window.__sink.status = picked ? picked.label : deviceId;
        }).catch(error => {
            window.__sink.ok = false;
            window.__sink.status = error.name;
        });
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
              output_device: str = None) -> dash.Dash:
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
            dcc.Dropdown(id='sink-pick', clearable=False, placeholder='output device'),
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
            ], style={'marginLeft': 'auto', 'display': 'flex',
                      'alignItems': 'center'}),
        ], style={'display': 'flex', 'alignItems': 'center',
                  'padding': '4px 20px 12px'}),
        dcc.Graph(id='timeline', figure=build_figure(track, sections),
                  config={'displayModeBar': False, 'scrollZoom': True}),
        html.Div(id='status', style={'padding': '10px 20px', 'color': MUTED,
                                     'fontSize': '13px'}),
        html.Div(build_rows(sections), id='table', style={'padding': '0 20px 40px'}),
        dcc.Store(id='sections', data=sections),
        dcc.Store(id='cursor', data=0.0),
        dcc.Store(id='sink-want', data=output_device),
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
        Output('sink-pick', 'style'),
        Input('tick', 'n_intervals'),
        State('sink-want', 'data'),
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
        Input({'type': 'row-nudge', 'index': ALL, 'delta': ALL}, 'n_clicks'),
        Input({'type': 'row-del', 'index': ALL}, 'n_clicks'),
        Input({'type': 'row-label', 'index': ALL}, 'value'),
        Input({'type': 'row-strength', 'index': ALL}, 'value'),
        State('sections', 'data'),
        State('cursor', 'data'),
        State('new-label', 'value'),
        State('new-strength', 'value'),
        prevent_initial_call=True,
    )
    def edit(mark_clicks, save_clicks, nudges, deletes, row_labels,
             row_strengths, sections, cursor, new_label, new_strength):
        updated, status = apply_edit(
            audio_path, callback_context.triggered_id, sections, cursor,
            new_label, new_strength, row_labels, row_strengths)
        return (dash.no_update if updated is None else updated,
                dash.no_update if status is None else status)

    @app.callback(
        Output('table', 'children'),
        Output('timeline', 'figure'),
        Input('sections', 'data'),
        prevent_initial_call=True,
    )
    def render(sections):
        return build_rows(sections), build_figure(track, sections)

    return app


def launch(audio: str, port: int = 8070, output_device: int = None) -> None:
    audio_path = str(Path(audio).resolve())
    if not Path(audio_path).exists():
        raise SystemExit(f'no such audio file: {audio_path}')

    device = (None if output_device is None
              else output_device_name(output_device))
    if device is not None:
        print(f'  output device {output_device}: {device}')

    print(f'  decoding {Path(audio_path).name} ...')
    track = Track(audio_path)
    sections = load_labels(audio_path)
    print(f'  {track.duration:.1f} s, {len(sections)} sections loaded from '
          f'{labels_path(audio_path)}')
    print(f'\n  Labeler → http://localhost:{port}\n')
    # No reloader and no hot reload: a restart would drop the page mid-labelling,
    # and there is nothing here worth watching a filesystem for.
    build_app(audio_path, track, sections, device).run(
        port=port, debug=False, use_reloader=False, dev_tools_hot_reload=False)
