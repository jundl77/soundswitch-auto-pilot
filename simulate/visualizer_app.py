import argparse
import http.client
import json
import logging

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

from lib.ui_bridge import SNAPSHOT_HOST, SNAPSHOT_PATH, snapshot_port
from simulate.runner import TIMING_TOLERANCE_SEC

TITLE = 'SoundSwitch Visualizer'
SLOT_LABELS = list('ABCDEFGH')
TIMELINE_WINDOW_SEC = 30.0
TIMELINE_LEAD_SEC = 0.5
# Drawn past the lead so the strip the browser translates in from the right is
# always future the audience has not reached, never a gap in the show.
TIMELINE_PAD_SEC = 1.0
TIMELINE_FULL_SEC = TIMELINE_WINDOW_SEC + TIMELINE_LEAD_SEC + TIMELINE_PAD_SEC
NOW_CURSOR_X = TIMELINE_WINDOW_SEC / TIMELINE_FULL_SEC
GLOW_BASE_PX = 16
DARK_BG   = '#0d1117'
CARD_BG   = '#111827'
BORDER    = '#1e2937'
OK_COLOR   = '#3fb950'
WARN_COLOR = '#f0883e'
MUTED      = '#6e7681'
POSTERIOR_FILL = '#58a6ff'

BEAT_MARKER_SIZE = 16

INTENT_CONFIG = {
    'atmospheric': {
        'primary':    '#1565c0',
        'accent':     '#6a1b9a',
        'slots':      [3, 4],
        'decay':      0.80,
        'glow_mult':  1.4,
        'label':      'ATMOSPHERIC',
    },
    'breakdown': {
        'primary':    '#7b1fa2',
        'accent':     '#880e4f',
        'slots':      [2, 3, 4],
        'decay':      0.55,
        'glow_mult':  1.6,
        'label':      'BREAKDOWN',
    },
    'buildup': {
        'primary':    '#e65100',
        'accent':     '#f9a825',
        'slots':      [0, 1, 2, 3, 4, 5],
        'decay':      0.25,
        'glow_mult':  2.2,
        'label':      'BUILDUP',
    },
    'drop': {
        'primary':    '#b71c1c',
        'accent':     '#ad1457',
        'slots':      [0, 1, 2, 3, 4, 5, 6, 7],
        'decay':      0.12,
        'glow_mult':  3.0,
        'label':      'DROP',
    },
    'peak': {
        'primary':    '#c62828',
        'accent':     '#ffffff',
        'slots':      [0, 1, 2, 3, 4, 5, 6, 7],
        'decay':      0.20,
        'glow_mult':  2.8,
        'label':      'PEAK',
    },
}

_DEFAULT_CONFIG = {
    'primary': '#2d3f52', 'accent': '#2d3f52',
    'slots': [], 'decay': 0.5, 'glow_mult': 1.0, 'label': '—',
}


def _intent_config(intent_key):
    return INTENT_CONFIG.get(intent_key, _DEFAULT_CONFIG)


def _room_events(events: list, snapshot: dict) -> list:
    """Detection-stamped records moved onto the room clock; unheard ones dropped."""
    delay = snapshot.get('look_ahead_sec', 0.0)
    now = snapshot.get('now', 0.0)
    return [dict(event, t=event['t'] + delay) for event in events
            if event['t'] + delay <= now]


def _clock_text(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f'{minutes}min {secs}sec' if minutes else f'{secs}sec'


def _song_origin(snapshot: dict) -> float | None:
    """The timeline's zero: the instant the room heard the current song start.

    None while the room is between songs.  The display describes what the room
    hears, so a stop the detector saw a look-ahead ago has not happened yet, and
    the start of the next song has not either.
    """
    heard = _room_events(snapshot.get('sound_events', []), snapshot)
    if heard:
        return heard[-1]['t'] if heard[-1]['playing'] else None
    starts = [e['t'] for e in snapshot.get('sound_events', []) if e['playing']]
    if not starts:
        return 0.0  # nothing ever claimed a boundary, so the session is the song
    return starts[-1] + snapshot.get('look_ahead_sec', 0.0)


def _display_now(snapshot: dict, origin: float | None) -> float:
    """How far into the song the room is — the axis, the cursor and the window."""
    if origin is None:
        return 0.0
    return max(0.0, snapshot.get('now', 0.0) - origin)


def _song_and_room(snapshot: dict) -> tuple:
    origin = _song_origin(snapshot)
    if origin is None or not any(e['playing']
                                 for e in snapshot.get('sound_events', [])):
        return None, None
    delay = snapshot.get('look_ahead_sec', 0.0)
    return (max(0.0, snapshot.get('now', 0.0) - origin + delay),
            _display_now(snapshot, origin))


def _anchor(snapshot: dict) -> dict:
    origin = _song_origin(snapshot)
    beats = [] if origin is None else [
        b['t'] - origin for b in _room_events(snapshot.get('beats', []), snapshot)
        if b['t'] - origin >= 0.0]
    song, room = _song_and_room(snapshot)
    return {
        'now':  _display_now(snapshot, origin),
        'beat': beats[-1] if beats else None,
        'song': song,
        'room': room,
        'span': TIMELINE_WINDOW_SEC,
        'lead': TIMELINE_LEAD_SEC,
        'pad':  TIMELINE_PAD_SEC,
    }


def _build_timeline(snapshot: dict) -> go.Figure:
    origin = _song_origin(snapshot)
    now    = _display_now(snapshot, origin)
    x0     = now - TIMELINE_WINDOW_SEC
    x1     = now + TIMELINE_LEAD_SEC + TIMELINE_PAD_SEC
    # The window opens a span before a song that may be seconds old, and nothing
    # from the previous one may show up in that gap.
    left   = max(x0, 0.0)

    shapes, annotations, beat_x = [], [], []

    for entry in snapshot.get('intents', []) if origin is not None else ():
        start   = entry['t'] - origin
        end     = entry['end'] - origin if 'end' in entry else x1
        t_start = max(start, left)
        t_end   = min(end, x1)
        if t_end <= t_start:
            continue
        cfg   = _intent_config(entry['intent'])
        color = cfg['primary']
        shapes.append(dict(
            type='rect', xref='x', yref='paper',
            x0=t_start, x1=t_end, y0=0.52, y1=0.96,
            fillcolor=color, opacity=0.80, line_width=0,
        ))
        if t_end - t_start > 1.5:
            annotations.append(dict(
                x=(t_start + t_end) / 2, y=0.74, xref='x', yref='paper',
                text=cfg['label'], showarrow=False,
                font=dict(color='rgba(255,255,255,0.85)', size=10, family='monospace'),
            ))

    if origin is not None:
        for ev in _room_events(snapshot.get('sound_events', []), snapshot):
            t = ev['t'] - origin
            if t < left:
                continue
            is_start = ev['playing']
            color    = '#3fb950' if is_start else '#f85149'
            label    = '▶ START' if is_start else '■ STOP'
            shapes.append(dict(
                type='line', xref='x', yref='paper',
                x0=t, x1=t, y0=0, y1=1,
                line=dict(color=color, width=1.5, dash='dash'),
            ))
            annotations.append(dict(
                x=t, y=0.04, xref='x', yref='paper',
                text=label, showarrow=False,
                font=dict(color=color, size=9, family='monospace'),
                xanchor='left',
            ))

        beat_x = [b['t'] - origin
                  for b in _room_events(snapshot.get('beats', []), snapshot)
                  if b['t'] - origin >= left]

    beat_y = [0.25] * len(beat_x)
    beat_size = [BEAT_MARKER_SIZE] * len(beat_x)

    fig = go.Figure()
    if beat_x:
        fig.add_trace(go.Scatter(
            x=beat_x, y=beat_y, mode='markers',
            marker=dict(
                symbol='line-ns', size=beat_size,
                color='rgba(168,218,220,0.65)',
                line=dict(color='rgba(168,218,220,0.65)', width=1.5),
            ),
            hoverinfo='skip',
        ))

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(
            range=[x0, x1],
            dtick=5.0,
            tickformat='.0f',
            ticksuffix='s',
            gridcolor='#1a2332',
            color='#6e7681',
            showline=False,
            minor=dict(dtick=1.0, showgrid=True,
                       gridcolor='#151e2b', gridwidth=0.8),
        ),
        yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        height=175, margin=dict(l=8, r=8, t=6, b=36),
        uirevision='timeline', showlegend=False,
    )
    return fig


def _build_stage(snapshot: dict) -> list:
    cfg     = _intent_config(snapshot.get('intent'))
    active  = sorted(cfg['slots'])
    peak_px = int(GLOW_BASE_PX * (1 + cfg['glow_mult']))

    slots = []
    for i, label in enumerate(SLOT_LABELS):
        on = i in active
        color = (cfg['accent'] if active.index(i) % 2 else cfg['primary']) if on else None

        lamp = {
            'width': '38px', 'height': '38px', 'borderRadius': '50%',
            'background': color or '#161d27',
            'margin': '0 auto 8px',
        }
        if on:
            lamp.update({'--ss-lamp': color,
                         '--ss-base': f'{GLOW_BASE_PX}px',
                         '--ss-peak': f'{peak_px}px',
                         '--ss-decay': f'{cfg["decay"]}s'})

        slots.append(html.Div([
            html.Div(className='ss-lamp ss-on' if on else 'ss-lamp', style=lamp),
            html.Div(label, style={
                'color': '#ffffff' if on else '#2d3f52',
                'fontSize': '12px', 'textAlign': 'center',
                'fontFamily': 'monospace', 'letterSpacing': '1px',
            }),
        ], style={
            'padding': '16px 8px', 'background': CARD_BG,
            'borderRadius': '8px',
            'border': f'1px solid {color}' if on else f'1px solid {BORDER}',
            'transition': 'border-color 0.08s ease',
        }))
    return slots


def _build_decoder(snapshot: dict) -> list:
    state = snapshot.get('decoder') or {}
    classes = state.get('classes') or []
    if not classes:
        return [html.Span('decoder: no decoder on this run',
                          style={'color': MUTED})]

    posterior = state.get('posterior')
    bars = []
    for index, name in enumerate(classes):
        value = 0.0 if posterior is None else float(posterior[index])
        bars.append(html.Div([
            html.Div(name, style={
                'color': MUTED, 'fontSize': '11px', 'marginBottom': '4px'}),
            html.Div(html.Div(style={
                'width': f'{value * 100:.1f}%', 'height': '100%',
                'background': POSTERIOR_FILL, 'borderRadius': '3px',
                'transition': 'width 0.15s ease',
            }), style={'height': '10px', 'background': '#161d27',
                       'borderRadius': '3px', 'overflow': 'hidden'}),
        ], style={'flex': '1'}))

    committed = state.get('committed_bar')
    cursor = 'no bar committed yet' if committed is None else (
        f'bar {state.get("observed_bar")} observed  →  committed '
        f'{committed} {state.get("committed_label")}  ·  '
        f'lag {state.get("lag_bars")} bars')
    if posterior is None:
        cursor += '  ·  no evidence at this bar'
    latency = state.get('chain_latency_sec')
    if latency is not None:
        cursor += f'  ·  chain {latency:.1f}s'

    return [
        html.Div(bars, style={'display': 'flex', 'gap': '10px'}),
        html.Div(cursor, style={'color': MUTED, 'fontSize': '12px',
                                'marginTop': '8px'}),
    ]


def _build_legend() -> list:
    items = [html.Span(TITLE, style={'fontWeight': 'bold', 'color': '#e6edf3',
                                     'marginRight': '28px'})]
    for cfg in INTENT_CONFIG.values():
        items.append(html.Span(f'■ {cfg["label"]}',
                               style={'color': cfg['primary'],
                                      'marginRight': '14px', 'fontSize': '12px'}))
    return items


def _build_metrics(snapshot: dict) -> list:
    bpm         = snapshot.get('bpm', 0.0)
    beats       = snapshot.get('beats_detected', 0)
    song, room  = _song_and_room(snapshot)
    song_text   = '—' if song is None else _clock_text(song)
    room_text   = '—' if room is None else _clock_text(room)
    intent_key  = snapshot.get('intent')
    cfg         = _intent_config(intent_key)
    intent_lbl  = cfg['label']
    intent_col  = cfg['primary']
    is_playing  = snapshot.get('is_playing', False)
    status_col  = '#3fb950' if is_playing else '#6e7681'
    status_lbl  = '● PLAYING' if is_playing else '◌ PAUSED'

    timing_str, timing_col = _timing_health(snapshot.get('timing_stats', {}))

    items = [
        html.Span(status_lbl,   style={'color': status_col,  'marginRight': '20px', 'fontWeight': 'bold'}),
        html.Span(f'room {room_text}', id='room-clock', style={'color': '#e6edf3', 'fontWeight': 'bold', 'marginRight': '10px'}),
        html.Span(f'song {song_text}', id='song-clock', style={'color': MUTED, 'marginRight': '20px'}),
        html.Span(f'{bpm:.0f} BPM',  style={'color': '#58a6ff', 'marginRight': '20px'}),
        html.Span(f'{beats} beats',   style={'color': OK_COLOR, 'marginRight': '20px'}),
        html.Span(f'intent: {intent_lbl}', style={'color': intent_col, 'fontWeight': 'bold', 'marginRight': '20px'}),
        html.Span(timing_str, style={'color': timing_col}),
    ]
    return items


def _timing_health(stats: dict) -> tuple:
    by_label = stats.get('by_label') or {}
    if not by_label:
        return 'cmd timing: —', MUTED
    tolerance_ms = TIMING_TOLERANCE_SEC * 1000
    worst = max(by_label, key=lambda label: by_label[label]['mean_error_ms'])
    late = by_label[worst]['mean_error_ms'] > tolerance_ms
    streams = ' '.join(f'{label} {s["mean_delta_sec"]:.2f}s±{s["mean_error_ms"]:.0f}ms'
                       for label, s in by_label.items())
    if late:
        return (f'cmd timing: {worst} misses its target by '
                f'{by_label[worst]["mean_error_ms"]:.0f}ms  │  {streams}'), WARN_COLOR
    return f'cmd timing: on target  │  {streams}', OK_COLOR


STYLESHEET = '''
.ss-lamp { transition: background 0.08s ease; }
.ss-lamp.ss-on { box-shadow: 0 0 var(--ss-base) var(--ss-lamp); }
@keyframes ss-pulse {
    from { box-shadow: 0 0 var(--ss-peak) var(--ss-lamp); }
    to   { box-shadow: 0 0 var(--ss-base) var(--ss-lamp); }
}
.ss-lamp.ss-on.ss-pulse { animation: ss-pulse var(--ss-decay) linear both; }
'''

INDEX_TEMPLATE = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + STYLESHEET + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''

ANIMATION_JS = '''
function(sync) {
    const ds = window.dash_clientside;
    const a = ds.ss = ds.ss || {};
    if (!sync) return ds.no_update;

    // Equal: the show's clock has stalled.  Backward: a song boundary re-based
    // the axis, and the session clock never goes back.  Extrapolating from
    // either invents time the room has not had.
    a.frozen = a.now === sync.now || sync.now < a.now;
    Object.assign(a, sync);
    a.at = performance.now();
    if (a.running) return ds.no_update;
    a.running = true;

    const RESEAT_PX = 24;

    const clock = (s) => {
        const t = Math.max(0, Math.floor(s));
        const m = Math.floor(t / 60);
        return m ? m + 'min ' + (t % 60) + 'sec' : t + 'sec';
    };

    const tick = (id, prefix, base, drift) => {
        const el = document.getElementById(id);
        if (!el || base == null) return;
        const text = prefix + clock(base + drift);
        if (el.textContent !== text) el.textContent = text;
    };

    const plotWidth = (gd) => {
        const fl = gd._fullLayout;
        if (!fl) return 0;
        if (fl.xaxis && fl.xaxis._length) return fl.xaxis._length;
        return fl.width - fl.margin.l - fl.margin.r;
    };

    // Scrolling by relayout re-renders every beat marker on every frame, which
    // measured 3.2 of 4.7 ms and 40% of the main thread.  Translating a wrapper
    // moves the same pixels on the compositor, and the axis is only re-seated
    // once the offset would expose the padded strip.
    const scroll = (now) => {
        const gd = document.querySelector('#timeline .js-plotly-plot');
        const el = document.getElementById('timeline-scroll');
        if (!gd || !el || !window.Plotly || !gd.layout || !gd.layout.xaxis) return;
        const range = gd.layout.xaxis.range;
        const width = plotWidth(gd);
        if (!range || !width) return;
        const pps = width / (range[1] - range[0]);
        let dx = (now + a.lead + a.pad - range[1]) * pps;
        if (dx < 0 || dx > RESEAT_PX) {
            window.Plotly.relayout(
                gd, {'xaxis.range': [now - a.span, now + a.lead + a.pad]});
            dx = 0;
        }
        el.style.transform = 'translate3d(' + (-dx).toFixed(2) + 'px,0,0)';
    };

    const pulse = (now) => {
        // Song times repeat across a reset, so the stamp goes with its song.
        if (a.beat == null) { a.pulsed = null; return; }
        if (a.beat === a.pulsed) return;
        a.pulsed = a.beat;
        document.querySelectorAll('.ss-lamp.ss-on').forEach((el) => {
            el.classList.remove('ss-pulse');
            void el.offsetWidth;
            el.style.animationDelay = Math.min(0, a.beat - now) + 's';
            el.classList.add('ss-pulse');
        });
    };

    const frame = () => {
        const drift = a.frozen ? 0 : Math.min(1.5, (performance.now() - a.at) / 1000);
        const now = a.now + drift;
        scroll(now);
        tick('room-clock', 'room ', a.room, drift);
        tick('song-clock', 'song ', a.song, drift);
        pulse(now);
        requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
    return ds.no_update;
}
'''


BLANK_SNAPSHOT = {
    'now': 0.0, 'look_ahead_sec': 0.0, 'is_playing': False,
    'beats': [], 'effects': [], 'intents': [], 'sound_events': [],
    'current_effect': None, 'bpm': 0.0, 'beats_detected': 0, 'intent': None,
    'timing_stats': {}, 'decoder': {},
}

POLL_TIMEOUT_SEC = 1.0


class SnapshotPoller:
    """The show over a socket, in place of the EventBuffer this used to hold."""

    def __init__(self, host: str = SNAPSHOT_HOST, port: int = 8051,
                 timeout: float = POLL_TIMEOUT_SEC):
        self._host, self._port, self._timeout = host, port, timeout
        self._connection = None
        self._last = dict(BLANK_SNAPSHOT)
        self._answering = True

    @property
    def url(self) -> str:
        return f'http://{self._host}:{self._port}{SNAPSHOT_PATH}'

    def snapshot(self) -> dict:
        try:
            if self._connection is None:
                self._connection = http.client.HTTPConnection(
                    self._host, self._port, timeout=self._timeout)
            self._connection.request('GET', SNAPSHOT_PATH)
            self._last = json.loads(self._connection.getresponse().read())
            if not self._answering:
                logging.info('[viewer] the show is answering again')
                self._answering = True
        except (OSError, http.client.HTTPException, ValueError) as error:
            self._connection = None
            if self._answering:
                logging.warning(f'[viewer] the show stopped answering on '
                                f'{self.url} ({error!r}) — holding the last frame')
                self._answering = False
        return self._last


def build_app(snapshot_source) -> dash.Dash:
    app = dash.Dash(__name__, title=TITLE, eager_loading=True)
    app.index_string = INDEX_TEMPLATE
    app.layout = html.Div([
        html.Div(_build_legend(), style={
            'padding': '12px 20px', 'borderBottom': f'1px solid {BORDER}',
            'fontFamily': 'monospace', 'fontSize': '14px',
        }),
        html.Div([
            html.Div(dcc.Graph(id='timeline',
                               config={'displayModeBar': False}),
                     id='timeline-scroll',
                     style={'willChange': 'transform'}),
            html.Div(id='now-cursor', style={
                'position': 'absolute', 'top': '0', 'bottom': '0',
                'left': f'{NOW_CURSOR_X * 100:.4f}%', 'width': '1px',
                'background': 'rgba(255,255,255,0.25)',
                'pointerEvents': 'none',
            }),
        ], style={
            'position': 'relative', 'overflow': 'hidden',
            'background': DARK_BG,
            'borderBottom': f'1px solid {BORDER}',
        }),
        html.Div(id='stage', style={
            'display': 'grid', 'gridTemplateColumns': 'repeat(8, 1fr)',
            'gap': '10px', 'padding': '20px 20px 16px',
        }),
        html.Div(id='decoder', style={
            'padding': '12px 20px 16px', 'borderTop': f'1px solid {BORDER}',
            'fontFamily': 'monospace', 'fontSize': '13px',
        }),
        html.Div(id='metrics', style={
            'padding': '10px 20px', 'borderTop': f'1px solid {BORDER}',
            'fontFamily': 'monospace', 'fontSize': '13px',
        }),
        dcc.Interval(id='tick', interval=250),
        dcc.Store(id='sync'),
        dcc.Store(id='anim'),
    ], style={'background': DARK_BG, 'minHeight': '100vh'})

    @app.callback(
        [Output('timeline', 'figure'),
         Output('stage', 'children'),
         Output('decoder', 'children'),
         Output('metrics', 'children'),
         Output('sync', 'data')],
        Input('tick', 'n_intervals'),
    )
    def refresh(_):
        snap = snapshot_source.snapshot()
        return (_build_timeline(snap), _build_stage(snap),
                _build_decoder(snap), _build_metrics(snap), _anchor(snap))

    app.clientside_callback(ANIMATION_JS, Output('anim', 'data'),
                            Input('sync', 'data'))
    return app


def run_app(snapshot_source, port: int = 8050) -> None:
    app = build_app(snapshot_source)
    print(f'\n  Visualizer → http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description='The lighting visualizer, in its own process: it polls the '
                    'show for snapshots and owns every pixel and every callback.')
    parser.add_argument('--port', type=int, default=8050,
                        help='Dash server port (default: 8050)')
    parser.add_argument('--snapshot-port', type=int, default=None,
                        help='Port the show serves /snapshot on '
                             '(default: --port + 1)')
    args = parser.parse_args(argv)
    logging.basicConfig(format='%(asctime)s [%(levelname)s ] %(message)s',
                        level=logging.INFO)
    port = (args.snapshot_port if args.snapshot_port is not None
            else snapshot_port(args.port))
    run_app(SnapshotPoller(port=port), port=args.port)


if __name__ == '__main__':
    main()
