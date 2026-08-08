import argparse
import http.client
import json
import logging
import threading
from statistics import median

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

from lib.engine.event_buffer import STOP_PERSISTENCE_SEC
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
TIMELINE_MARGIN_L_PX = 8
TIMELINE_MARGIN_R_PX = 8
# The cursor's fraction is of the AXIS, not of the container: ignoring the
# figure margins parked it a constant ~7 px right of the axis' now, which at
# this scale is ~150 ms of claimed future -- measured as a -146 ms crossing
# bias on every marker, and worse the narrower the window.
NOW_CURSOR_LEFT = (
    f'calc({TIMELINE_MARGIN_L_PX}px + {NOW_CURSOR_X * 100:.4f}% - '
    f'{NOW_CURSOR_X:.6f} * {TIMELINE_MARGIN_L_PX + TIMELINE_MARGIN_R_PX}px)')
GLOW_BASE_PX = 16
DARK_BG   = '#0d1117'
CARD_BG   = '#111827'
BORDER    = '#1e2937'
OK_COLOR   = '#3fb950'
WARN_COLOR = '#f0883e'
MUTED      = '#6e7681'
POSTERIOR_FILL = '#58a6ff'

HEALTH_FONT_SIZE = '14px'
TIMING_FONT_SIZE = '12px'

BEAT_MARKER_SIZE = 16
BEATS_PER_BAR = 4
BAR_LABEL_EVERY = 4
BAR_SPAN_TOLERANCE = 1.5
DOWNBEAT_COLOR = 'rgba(88,166,255,0.55)'
BEAT_TICK_COLOR = 'rgba(88,166,255,0.22)'

REFRESH_MS = 250
VIEW_EVERY_TICKS = 2
STALL_RELEASE_MS = 4000
STALE_REFRESHES = 12

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
}

_DEFAULT_CONFIG = {
    'primary': '#2d3f52', 'accent': '#2d3f52',
    'slots': [], 'decay': 0.5, 'glow_mult': 1.0, 'label': '—',
}


def _intent_config(intent_key):
    return INTENT_CONFIG.get(intent_key, _DEFAULT_CONFIG)


def _bridged(stop_t: float, start_t: float) -> bool:
    return stop_t < start_t <= stop_t + STOP_PERSISTENCE_SEC


def _elided(event: dict, snapshot: dict) -> bool:
    others = snapshot.get('sound_events', [])
    if event['playing']:
        return any(_bridged(other['t'], event['t'])
                   for other in others if not other['playing'])
    return any(_bridged(event['t'], other['t'])
               for other in others if other['playing'])


def _room_stops(snapshot: dict) -> list:
    return [event['t'] + STOP_PERSISTENCE_SEC
            for event in snapshot.get('sound_events', [])
            if not event['playing'] and not _elided(event, snapshot)]


def _room_sound_events(snapshot: dict) -> list:
    delay = snapshot.get('look_ahead_sec', 0.0)
    now = snapshot.get('now', 0.0)
    stops = _room_stops(snapshot)
    heard = []
    for event in snapshot.get('sound_events', []):
        if _elided(event, snapshot):
            continue
        if event['playing']:
            reaches_room_at = event['t'] + delay
            if any(event['t'] < stop <= reaches_room_at for stop in stops):
                continue
        else:
            reaches_room_at = event['t'] + STOP_PERSISTENCE_SEC
        if reaches_room_at <= now:
            heard.append(dict(event, t=reaches_room_at))
    return heard


def _room_beats(snapshot: dict, horizon_sec: float = 0.0) -> list:
    delay = snapshot.get('look_ahead_sec', 0.0)
    limit = snapshot.get('now', 0.0) + horizon_sec
    beats = []
    stops = _room_stops(snapshot)
    for beat in snapshot.get('beats', []):
        reaches_room_at = beat['t'] + delay
        cut = any(beat['t'] < stop <= reaches_room_at for stop in stops)
        if reaches_room_at <= limit and not cut:
            beats.append(dict(beat, t=reaches_room_at))
    return beats


def _heard_beats(snapshot: dict) -> list:
    return _room_beats(snapshot)


def _clock_text(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f'{minutes}min {secs}sec' if minutes else f'{secs}sec'


def _last_heard(snapshot: dict) -> dict | None:
    heard = _room_sound_events(snapshot)
    if not heard:
        return None
    return max(reversed(heard), key=lambda event: event['t'])


def _song_origin(snapshot: dict) -> float | None:
    last = _last_heard(snapshot)
    if last is not None:
        return last['t'] if last['playing'] else None
    starts = [e['t'] for e in snapshot.get('sound_events', []) if e['playing']]
    if not starts:
        return 0.0
    return starts[-1] + snapshot.get('look_ahead_sec', 0.0)


def _room_is_playing(snapshot: dict) -> bool:
    now = snapshot.get('now', 0.0)
    live = (bool(snapshot.get('is_playing'))
            or any(stop > now for stop in _room_stops(snapshot)))
    last = _last_heard(snapshot)
    if last is not None:
        return last['playing'] and live
    return not snapshot.get('sound_events') and live


def _room_bpm(snapshot: dict) -> float:
    heard = _heard_beats(snapshot)
    return heard[-1]['bpm'] if heard else snapshot.get('bpm', 0.0)


def _beats_still_travelling(snapshot: dict) -> int:
    delay = snapshot.get('look_ahead_sec', 0.0)
    now = snapshot.get('now', 0.0)
    stops = _room_stops(snapshot)
    return sum(1 for beat in snapshot.get('beats', [])
               if beat['t'] + delay > now
               and not any(beat['t'] < stop <= beat['t'] + delay for stop in stops))


def _room_beat_count(snapshot: dict) -> int:
    return max(0, snapshot.get('beats_detected', 0)
               - _beats_still_travelling(snapshot)
               - snapshot.get('beats_cut', 0))


def _display_now(snapshot: dict, origin: float | None) -> float:
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
    # Every beat is stamped one look-ahead before the room hears it, so the
    # whole drawable window -- the lead included -- is known in advance and the
    # browser can land each marker at exactly its room instant.
    origin = _song_origin(snapshot)
    now = _display_now(snapshot, origin)
    beats = [] if origin is None else [
        round(b['t'] - origin, 3)
        for b in _room_beats(snapshot, TIMELINE_LEAD_SEC + TIMELINE_PAD_SEC)
        if b['t'] - origin >= max(0.0, now - TIMELINE_WINDOW_SEC)]
    song, room = _song_and_room(snapshot)
    return {
        'now':  now,
        'beats': beats,
        'song': song,
        'room': room,
        'span': TIMELINE_WINDOW_SEC,
        'lead': TIMELINE_LEAD_SEC,
        'pad':  TIMELINE_PAD_SEC,
    }


def _bar_grid(snapshot: dict, origin: float) -> tuple:
    state = snapshot.get('decoder') or {}
    edges = state.get('bar_edges') or []
    if len(edges) < 2:
        return [], []
    delay = snapshot.get('look_ahead_sec', 0.0)
    first_bar = state.get('first_bar') or 0
    spans = [b - a for a, b in zip(edges, edges[1:])]
    bar_sec = state.get('bar_sec') or median(spans)

    shapes, annotations = [], []
    for index, start in enumerate(edges[:-1]):
        at = start + delay - origin
        span = spans[index]
        shapes.append(dict(
            type='line', xref='x', yref='paper',
            x0=at, x1=at, y0=0.0, y1=0.5,
            line=dict(color=DOWNBEAT_COLOR, width=1.4),
        ))
        if span <= bar_sec * BAR_SPAN_TOLERANCE:
            for beat in range(1, BEATS_PER_BAR):
                tick = at + span * beat / BEATS_PER_BAR
                shapes.append(dict(
                    type='line', xref='x', yref='paper',
                    x0=tick, x1=tick, y0=0.06, y1=0.20,
                    line=dict(color=BEAT_TICK_COLOR, width=0.8),
                ))
        bar = first_bar + index
        if bar % BAR_LABEL_EVERY == 0:
            annotations.append(dict(
                x=at, y=0.50, xref='x', yref='paper',
                text=str(bar), showarrow=False, xanchor='left', yanchor='bottom',
                font=dict(color=DOWNBEAT_COLOR, size=9, family='monospace'),
            ))
    return shapes, annotations


def _build_timeline(snapshot: dict) -> go.Figure:
    origin = _song_origin(snapshot)
    now    = _display_now(snapshot, origin)
    x0     = now - TIMELINE_WINDOW_SEC
    x1     = now + TIMELINE_LEAD_SEC + TIMELINE_PAD_SEC
    left   = max(x0, 0.0)

    shapes, annotations = [], []

    if origin is not None:
        grid_shapes, grid_labels = _bar_grid(snapshot, origin)
        shapes.extend(grid_shapes)
        annotations.extend(grid_labels)

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
        for ev in _room_sound_events(snapshot):
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

    # Beats are deliberately NOT drawn here: a figure push arrives most of a
    # second after the beat it carries, so the browser owns the markers (see
    # ANIMATION_JS) and this figure carries everything that is known ahead.
    fig = go.Figure()
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
        height=175, margin=dict(l=TIMELINE_MARGIN_L_PX, r=TIMELINE_MARGIN_R_PX,
                                t=6, b=36),
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
    bpm         = _room_bpm(snapshot)
    beats       = _room_beat_count(snapshot)
    song, room  = _song_and_room(snapshot)
    song_text   = '—' if song is None else _clock_text(song)
    room_text   = '—' if room is None else _clock_text(room)
    intent_key  = snapshot.get('intent')
    cfg         = _intent_config(intent_key)
    intent_lbl  = cfg['label']
    intent_col  = cfg['primary']
    is_playing  = _room_is_playing(snapshot)
    status_col  = '#3fb950' if is_playing else '#6e7681'
    status_lbl  = '● PLAYING' if is_playing else '◌ PAUSED'

    verdict_str, stream_strs, timing_col = _timing_health(
        snapshot.get('timing_stats', {}))
    shed = snapshot.get('shed') or {}
    pill_str, pill_col = _shed_pill(shed)
    rate_str, rate_col = _shed_rate(shed)

    return [
        _metric_row([
            html.Span(status_lbl, style={'color': status_col,
                                         'fontWeight': 'bold',
                                         'minWidth': '11ch'}),
            html.Span(f'room {room_text}', id='room-clock',
                      style={'color': '#e6edf3', 'fontWeight': 'bold',
                             'minWidth': '17ch'}),
            html.Span(f'song {song_text}', id='song-clock',
                      style={'color': MUTED, 'minWidth': '17ch'}),
        ]),
        _metric_row([
            html.Span(f'{bpm:.0f} BPM', style={'color': '#58a6ff',
                                               'minWidth': '9ch'}),
            html.Span(f'{beats} beats', style={'color': OK_COLOR,
                                               'minWidth': '12ch'}),
            html.Span(f'intent: {intent_lbl}', style={'color': intent_col,
                                                      'fontWeight': 'bold'}),
        ]),
        _metric_row([
            html.Span(pill_str, style={'color': pill_col,
                                       'fontWeight': 'bold',
                                       'fontSize': HEALTH_FONT_SIZE,
                                       'border': f'1px solid {pill_col}',
                                       'borderRadius': '4px',
                                       'padding': '2px 10px'}),
            html.Span(rate_str, style={'color': rate_col}),
        ], alignItems='center'),
        _metric_row(
            [html.Span(verdict_str, style={'color': timing_col,
                                           'fontWeight': 'bold'})]
            + [html.Span(text, style={'color': MUTED, 'minWidth': '21ch'})
               for text in stream_strs],
            fontSize=TIMING_FONT_SIZE, columnGap='16px', marginBottom='0',
            marginTop='2px', paddingTop='7px',
            borderTop=f'1px solid {BORDER}'),
    ]


def _metric_row(items: list, **overrides) -> html.Div:
    style = {'display': 'flex', 'alignItems': 'baseline', 'flexWrap': 'wrap',
             'columnGap': '22px', 'rowGap': '4px', 'marginBottom': '7px',
             'fontVariantNumeric': 'tabular-nums'}
    style.update(overrides)
    return html.Div(items, style=style)


def _shed_pill(shed: dict) -> tuple:
    if not shed:
        return 'health: —', MUTED
    if shed.get('level', 'NONE') != 'NONE':
        fault = shed.get('fault')
        return (f'health: ◆ DEGRADED — holding intent'
                f'{f" ({fault})" if fault else ""}'), WARN_COLOR
    return 'health: ● LIVE', OK_COLOR


def _shed_rate(shed: dict) -> tuple:
    if not shed:
        return 'sheds: —', MUTED
    rate = shed.get('sheds_per_min', 0)
    total = shed.get('sheds', 0)
    text = f'sheds {rate}/min'
    if total:
        text += f'  ·  {total} this run'
    return text, (WARN_COLOR if rate else MUTED)


def _timing_health(stats: dict) -> tuple:
    by_label = stats.get('by_label') or {}
    if not by_label:
        return 'cmd timing: —', [], MUTED
    tolerance_ms = TIMING_TOLERANCE_SEC * 1000
    worst = max(by_label, key=lambda label: by_label[label]['mean_error_ms'])
    late = by_label[worst]['mean_error_ms'] > tolerance_ms
    streams = [f'{label} {s["mean_delta_sec"]:.2f}s ±{s["mean_error_ms"]:.0f}ms'
               for label, s in by_label.items()]
    if late:
        return (f'cmd timing: {worst} misses its target by '
                f'{by_label[worst]["mean_error_ms"]:.0f}ms'), streams, WARN_COLOR
    return 'cmd timing: on target', streams, OK_COLOR


STYLESHEET = '''
.ss-lamp { transition: background 0.08s ease; }
.ss-lamp.ss-on { box-shadow: 0 0 var(--ss-base) var(--ss-lamp); }
.ss-beat { position: absolute; width: 2px; margin-left: -1px;
           height: ''' + str(BEAT_MARKER_SIZE) + '''px;
           left: -9999px;  /* off-screen until seated against the live range */
           background: rgba(168,218,220,0.65); }
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
    if (ds.ss_gate) delete ds.ss_gate.inflight['anchor'];
    if (!sync) return ds.no_update;

    a.frozen = a.now === sync.now || sync.now < a.now;
    Object.assign(a, sync);
    a.at = performance.now();
    if (a.running) return ds.no_update;
    a.running = true;

    const RESEAT_PX = 24;
    const HALF_MARKER_PX = ''' + str(BEAT_MARKER_SIZE // 2) + ''';

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

    const plotHeight = (gd) => {
        const fl = gd._fullLayout;
        if (!fl) return 0;
        if (fl.yaxis && fl.yaxis._length) return fl.yaxis._length;
        return fl.height - fl.margin.t - fl.margin.b;
    };

    // The anchor ships every drawable beat instant (the show knows each beat
    // one look-ahead before the room hears it), so the markers are DOM nodes
    // on the translated layer: they scroll on the compositor and cross the
    // fixed cursor at exactly their room instant, instead of popping in most
    // of a second late on the next figure push.
    const syncBeats = () => {
        const layer = document.getElementById('beat-layer');
        if (!layer || a.markerList === a.beats) return;
        a.markerList = a.beats || [];
        a.markers = a.markers || {};
        const keep = new Set(a.markerList.map((t) => t.toFixed(3)));
        let changed = false;
        for (const key of Object.keys(a.markers)) {
            if (keep.has(key)) continue;
            a.markers[key].remove();
            delete a.markers[key];
            changed = true;
        }
        for (const t of a.markerList) {
            const key = t.toFixed(3);
            if (a.markers[key]) continue;
            const mark = document.createElement('div');
            mark.className = 'ss-beat';
            mark.dataset.t = key;
            layer.appendChild(mark);
            a.markers[key] = mark;
            changed = true;
        }
        if (changed) a.seatedAt = null;
    };

    // Seated in axis px only when the range or the marker set moved; between
    // seats the wrapper's transform is the only thing that moves them.
    const seatBeats = (gd) => {
        if (!gd || !gd.layout || !gd.layout.xaxis) return;
        const range = gd.layout.xaxis.range;
        const width = plotWidth(gd);
        if (!range || !width) return;
        const stamp = range[0] + ':' + range[1] + ':' + width;
        if (a.seatedAt === stamp) return;
        a.seatedAt = stamp;
        const fl = gd._fullLayout;
        const pps = width / (range[1] - range[0]);
        const top = fl.margin.t + plotHeight(gd) * 0.75 - HALF_MARKER_PX;
        for (const key of Object.keys(a.markers || {})) {
            const mark = a.markers[key];
            mark.style.left = (fl.margin.l
                + (parseFloat(key) - range[0]) * pps).toFixed(2) + 'px';
            mark.style.top = top.toFixed(2) + 'px';
        }
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

    // The glow fires when the room clock crosses the beat, not when an anchor
    // happens to land -- the anchor cadence cost a median 154 ms of lag here.
    const pulse = (now) => {
        const beats = a.markerList || [];
        let latest = null;
        for (let i = beats.length - 1; i >= 0; i--) {
            if (beats[i] <= now) { latest = beats[i]; break; }
        }
        if (latest == null) { a.pulsed = null; return; }
        if (latest === a.pulsed) return;
        a.pulsed = latest;
        document.querySelectorAll('.ss-lamp.ss-on').forEach((el) => {
            el.classList.remove('ss-pulse');
            void el.offsetWidth;
            el.style.animationDelay = Math.min(0, latest - now) + 's';
            el.classList.add('ss-pulse');
        });
    };

    const meter = () => {
        a.ticks = (a.ticks || 0) + 1;
        const at = performance.now();
        if (a.meteredAt == null) { a.meteredAt = at; return; }
        const span = at - a.meteredAt;
        if (span < 1000) return;
        const el = document.getElementById('fps');
        if (el) el.textContent = Math.round(a.ticks * 1000 / span) + ' fps';
        a.ticks = 0;
        a.meteredAt = at;
    };

    const frame = () => {
        const drift = a.frozen ? 0 : Math.min(1.5, (performance.now() - a.at) / 1000);
        const now = a.now + drift;
        syncBeats();
        scroll(now);
        seatBeats(document.querySelector('#timeline .js-plotly-plot'));
        tick('room-clock', 'room ', a.room, drift);
        tick('song-clock', 'song ', a.song, drift);
        pulse(now);
        meter();
        requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
    return ds.no_update;
}
'''


_GATE_STATE_JS = '''
    const ds = window.dash_clientside = window.dash_clientside || {};
    const gate = ds.ss_gate = ds.ss_gate ||
        {inflight: {}, sent: null, behind: 0};
    const streamOf = (outputs) => {
        const list = Array.isArray(outputs) ? outputs : [outputs];
        for (let i = 0; i < list.length; i++) {
            if (!list[i]) continue;
            if (list[i].id === 'sync') return 'anchor';
            if (list[i].id === 'timeline') return 'view';
        }
        return null;
    };
'''

REQUEST_PRE_JS = 'function(payload) {' + _GATE_STATE_JS + '''
    const stream = streamOf(payload.outputs);
    if (stream) gate.inflight[stream] = Date.now();
}'''

VIEW_LANDED_JS = 'function(drawn) {' + _GATE_STATE_JS + '''
    delete gate.inflight['view'];
    return drawn;
}'''

CALLBACK_RESOLVED_JS = 'function(callback, result) {' + _GATE_STATE_JS + '''
    const stream = streamOf(callback.outputs);
    if (!stream) return;
    if (result.error) {
        delete gate.inflight[stream];
        gate.sent = null;
        gate.behind = 0;
        return;
    }
    if (stream !== 'anchor') return;
    const fresh = result.data && result.data.sync && result.data.sync.data;
    if (!fresh) return;
    const landed = ds.ss || {};
    if (gate.sent !== null && landed.now !== gate.sent) {
        gate.behind += 1;
        if (gate.behind >= ''' + str(STALE_REFRESHES) + ''') {
            console.warn('[viewer] ' + gate.behind + ' refreshes answered with '
                + 'fresh data the page never took (server said ' + gate.sent
                + 's, page still on ' + landed.now + 's) — the anchor is not '
                + 'reaching the layout');
            gate.behind = 0;
        }
    } else {
        gate.behind = 0;
    }
    gate.sent = fresh.now;
}'''

GATE_JS = 'function(n) {' + _GATE_STATE_JS + '''
    const free = (stream) => {
        const since = gate.inflight[stream];
        if (since == null) return true;
        const waited = Date.now() - since;
        if (waited < ''' + str(STALL_RELEASE_MS) + ''') return false;
        console.warn('[viewer] the ' + stream + ' refresh has not answered in '
            + (waited / 1000).toFixed(1) + 's — re-arming it');
        delete gate.inflight[stream];
        return true;
    };
    return [free('anchor') ? n : ds.no_update,
            (n % ''' + str(VIEW_EVERY_TICKS) + ''' === 0 && free('view'))
                ? n : ds.no_update];
}'''


BLANK_SNAPSHOT = {
    'now': 0.0, 'look_ahead_sec': 0.0, 'is_playing': False,
    'beats': [], 'effects': [], 'intents': [], 'sound_events': [],
    'current_effect': None, 'bpm': 0.0, 'beats_detected': 0, 'intent': None,
    'timing_stats': {}, 'decoder': {}, 'shed': {},
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
        self._lock = threading.Lock()

    @property
    def url(self) -> str:
        return f'http://{self._host}:{self._port}{SNAPSHOT_PATH}'

    def snapshot(self) -> dict:
        if not self._lock.acquire(blocking=False):
            return self._last
        try:
            try:
                connection = self._connection
                if connection is None:
                    connection = self._connection = http.client.HTTPConnection(
                        self._host, self._port, timeout=self._timeout)
                connection.request('GET', SNAPSHOT_PATH)
                self._last = json.loads(connection.getresponse().read())
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
        finally:
            self._lock.release()


def build_app(snapshot_source) -> dash.Dash:
    app = dash.Dash(__name__, title=TITLE, eager_loading=True,
                    hooks={'request_pre': REQUEST_PRE_JS,
                           'callback_resolved': CALLBACK_RESOLVED_JS})
    app.index_string = INDEX_TEMPLATE
    app.layout = html.Div([
        html.Div(_build_legend(), style={
            'padding': '12px 20px', 'borderBottom': f'1px solid {BORDER}',
            'fontFamily': 'monospace', 'fontSize': '14px',
        }),
        html.Div([
            html.Div([dcc.Graph(id='timeline',
                                config={'displayModeBar': False}),
                      html.Div(id='beat-layer', style={
                          'position': 'absolute', 'top': '0', 'left': '0',
                          'right': '0', 'bottom': '0',
                          'pointerEvents': 'none',
                      })],
                     id='timeline-scroll',
                     style={'willChange': 'transform',
                            'position': 'relative'}),
            html.Div(id='now-cursor', style={
                'position': 'absolute', 'top': '0', 'bottom': '0',
                'left': NOW_CURSOR_LEFT, 'width': '1px',
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
        html.Div([
            html.Div(id='metrics', style={'flex': '1'}),
            html.Div('— fps', id='fps', style={'color': MUTED, 'marginLeft': '20px'}),
        ], style={
            'display': 'flex', 'alignItems': 'flex-start',
            'padding': '10px 20px', 'borderTop': f'1px solid {BORDER}',
            'fontFamily': 'monospace', 'fontSize': '13px',
        }),
        dcc.Interval(id='tick', interval=REFRESH_MS),
        dcc.Store(id='gate'),
        dcc.Store(id='view-gate'),
        dcc.Store(id='sync'),
        dcc.Store(id='anim'),
        dcc.Store(id='drawn'),
        dcc.Store(id='taken'),
    ], style={'background': DARK_BG, 'minHeight': '100vh'})

    app.clientside_callback(
        GATE_JS,
        [Output('gate', 'data'), Output('view-gate', 'data')],
        Input('tick', 'n_intervals'))

    latest: dict = {}

    @app.callback(
        [Output('sync', 'data'),
         Output('metrics', 'children'),
         Output('stage', 'children'),
         Output('decoder', 'children')],
        Input('gate', 'data'),
        prevent_initial_call=True,
    )
    def refresh(_):
        snap = latest['snapshot'] = snapshot_source.snapshot()
        return (_anchor(snap), _build_metrics(snap),
                _build_stage(snap), _build_decoder(snap))

    @app.callback(
        [Output('timeline', 'figure'), Output('drawn', 'data')],
        Input('view-gate', 'data'),
        prevent_initial_call=True,
    )
    def refresh_view(tick):
        snap = latest.get('snapshot') or snapshot_source.snapshot()
        if snap is latest.get('drawn'):
            return dash.no_update, tick
        latest['drawn'] = snap
        return _build_timeline(snap), tick

    app.clientside_callback(ANIMATION_JS, Output('anim', 'data'),
                            Input('sync', 'data'))
    app.clientside_callback(VIEW_LANDED_JS, Output('taken', 'data'),
                            Input('drawn', 'data'))
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
