import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

from simulate.runner import TIMING_TOLERANCE_SEC

TITLE = 'SoundSwitch Visualizer'
SLOT_LABELS = list('ABCDEFGH')
TIMELINE_WINDOW_SEC = 30.0
DARK_BG   = '#0d1117'
CARD_BG   = '#111827'
BORDER    = '#1e2937'
OK_COLOR   = '#3fb950'
WARN_COLOR = '#f0883e'
MUTED      = '#6e7681'
POSTERIOR_FILL = '#58a6ff'

# Onset density used to scale this; the density chain left with the rule engine
# and the freed channel is deliberately not re-purposed (D14).  Constant, not
# derived from a field that is now always zero.
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
    # No `groove`: the model's class space has none and `LightIntent.GROOVE` is
    # retired (D7).  A legend entry for an intent the show cannot enter is the
    # exact lie D7 removed the enum member over.
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



def _build_timeline(snapshot: dict) -> go.Figure:
    now   = snapshot['now']
    x0    = now - TIMELINE_WINDOW_SEC
    x1    = now + 0.5

    shapes, annotations = [], []

    # Drawn first so the intent bands render on top.
    t_grid = int(x0)
    while t_grid <= x1:
        shapes.append(dict(
            type='line', xref='x', yref='paper',
            x0=t_grid, x1=t_grid, y0=0, y1=1,
            line=dict(color='#151e2b', width=0.8),
        ))
        t_grid += 1

    for entry in snapshot.get('intents', []):
        t_start = max(entry['t'], x0)
        t_end   = min(entry.get('end', now), x1)
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

    for ev in snapshot.get('sound_events', []):
        if ev['t'] < x0:
            continue
        is_start = ev['playing']
        color    = '#3fb950' if is_start else '#f85149'
        label    = '▶ START' if is_start else '■ STOP'
        shapes.append(dict(
            type='line', xref='x', yref='paper',
            x0=ev['t'], x1=ev['t'], y0=0, y1=1,
            line=dict(color=color, width=1.5, dash='dash'),
        ))
        annotations.append(dict(
            x=ev['t'], y=0.04, xref='x', yref='paper',
            text=label, showarrow=False,
            font=dict(color=color, size=9, family='monospace'),
            xanchor='left',
        ))

    beat_x = [b['t'] for b in snapshot['beats'] if b['t'] >= x0]
    beat_y = [0.25] * len(beat_x)
    beat_size = [BEAT_MARKER_SIZE] * len(beat_x)

    shapes.append(dict(
        type='line', xref='x', yref='paper',
        x0=now, x1=now, y0=0, y1=1,
        line=dict(color='rgba(255,255,255,0.25)', width=1, dash='dot'),
    ))

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
        ),
        yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        height=175, margin=dict(l=8, r=8, t=6, b=36),
        uirevision='timeline', showlegend=False,
    )
    return fig


def _build_stage(snapshot: dict) -> list:
    cfg      = _intent_config(snapshot.get('intent'))
    active   = set(cfg['slots'])
    primary  = cfg['primary']
    accent   = cfg['accent']
    decay    = cfg['decay']
    glow_m   = cfg['glow_mult']

    now   = snapshot['now']
    beats = snapshot.get('beats', [])
    dt    = (now - beats[-1]['t']) if beats else 999.0
    pulse = max(0.0, 1.0 - dt / decay)

    base_glow  = 16
    pulse_glow = int(base_glow + pulse * base_glow * glow_m)

    slots = []
    for i, label in enumerate(SLOT_LABELS):
        on = i in active
        active_sorted = sorted(active)
        pos_in_active = active_sorted.index(i) if on else -1
        color = (accent if pos_in_active % 2 == 1 else primary) if on else None

        glow_px = pulse_glow if on else 0
        dim_bg  = '#161d27'

        slots.append(html.Div([
            html.Div(style={
                'width': '38px', 'height': '38px', 'borderRadius': '50%',
                'background': color or dim_bg,
                'margin': '0 auto 8px',
                'boxShadow': f'0 0 {glow_px}px {color}' if on and glow_px > 0 else 'none',
                'transition': 'background 0.08s ease, box-shadow 0.08s ease',
            }),
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
    """D14's one new panel: what the committer is looking at.

    The show is driven by the decoder now, and from the stage view alone a
    stuck one and a quiet passage are the same picture -- so the class
    posteriors and the commit cursor are the only things worth adding.
    """
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
    elapsed     = snapshot.get('now', 0.0)
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
        html.Span(f'{elapsed:.1f}s', style={'color': MUTED, 'marginRight': '20px'}),
        html.Span(f'{bpm:.0f} BPM',  style={'color': '#58a6ff', 'marginRight': '20px'}),
        html.Span(f'{beats} beats',   style={'color': OK_COLOR, 'marginRight': '20px'}),
        html.Span(f'intent: {intent_lbl}', style={'color': intent_col, 'fontWeight': 'bold', 'marginRight': '20px'}),
        html.Span(timing_str, style={'color': timing_col}),
    ]
    return items


def _timing_health(stats: dict) -> tuple:
    """Each stream against its OWN target, never against a written-down delay.

    The four streams wait four different amounts (B1) and the numbers move with
    the measured chain, so the only durable question is whether the queue is
    hitting what it aimed at.  A literal here read amber for entire shows.
    """
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


def build_app(event_buffer) -> dash.Dash:
    app = dash.Dash(__name__, title=TITLE)
    app.layout = html.Div([
        html.Div(_build_legend(), style={
            'padding': '12px 20px', 'borderBottom': f'1px solid {BORDER}',
            'fontFamily': 'monospace', 'fontSize': '14px',
        }),
        dcc.Graph(id='timeline', config={'displayModeBar': False},
                  style={'borderBottom': f'1px solid {BORDER}'}),
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
        dcc.Interval(id='tick', interval=100),
    ], style={'background': DARK_BG, 'minHeight': '100vh'})

    @app.callback(
        [Output('timeline', 'figure'),
         Output('stage', 'children'),
         Output('decoder', 'children'),
         Output('metrics', 'children')],
        Input('tick', 'n_intervals'),
    )
    def refresh(_):
        snap = event_buffer.snapshot()
        return (_build_timeline(snap), _build_stage(snap),
                _build_decoder(snap), _build_metrics(snap))

    return app


def run_app(event_buffer, port: int = 8050) -> None:
    app = build_app(event_buffer)
    print(f'\n  Visualizer → http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
