"""
Dash-based lighting visualizer.

Shows a rolling 30-second timeline of beats and effect changes, plus a
stage view that reacts to the current LightIntent. Updated every 100 ms.

Intent → stage mapping (designed for EDM structure):
  ATMOSPHERIC — 2 center fixtures, deep blue + violet, barely pulsing
  BREAKDOWN   — 3 center fixtures, warm purple + rose, gentle pulse
  GROOVE      — 5 spread fixtures, teal + sky, on-beat pulse
  BUILDUP     — 6 fixtures, amber + gold, intensifying pulse
  DROP        — all 8, crimson + magenta, hard flash on beat
  PEAK        — all 8, white-hot center + red outer, searing sustained glow

Beat pulse: glow radius peaks immediately on beat, decays exponentially.
Decay speed is intent-specific (DROP is the shortest = sharpest flash).
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLOT_LABELS = list('ABCDEFGH')
TIMELINE_WINDOW_SEC = 30.0
DARK_BG   = '#0d1117'
CARD_BG   = '#111827'
BORDER    = '#1e2937'

# ---------------------------------------------------------------------------
# Intent display config
# Each entry: (primary_hex, accent_hex, active_slots, beat_decay_sec, label)
#
# active_slots: list of slot indices (0=A … 7=H) that are "on" for this intent.
# The pattern matters: ATMOSPHERIC uses center slots, DROP uses all, etc.
# primary vs accent alternate across the active slots for visual variety.
#
# beat_decay_sec: how long (seconds) before the beat glow fades completely.
# Short = sharp strobe-like flash (DROP). Long = slow ambient pulse (ATMOSPHERIC).
# ---------------------------------------------------------------------------

INTENT_CONFIG = {
    'atmospheric': {
        'primary':    '#1565c0',   # deep sapphire blue
        'accent':     '#6a1b9a',   # midnight violet
        'slots':      [3, 4],      # center pair only
        'decay':      0.80,        # very slow — barely perceptible pulse
        'glow_mult':  1.4,
        'label':      'ATMOSPHERIC',
    },
    'breakdown': {
        'primary':    '#7b1fa2',   # warm purple
        'accent':     '#880e4f',   # deep rose
        'slots':      [2, 3, 4],   # tight center cluster
        'decay':      0.55,        # gentle heartbeat feel
        'glow_mult':  1.6,
        'label':      'BREAKDOWN',
    },
    'groove': {
        'primary':    '#00897b',   # electric teal
        'accent':     '#0277bd',   # sky blue
        'slots':      [1, 2, 3, 4, 5],  # evenly spread, room to breathe
        'decay':      0.35,        # crisp on-beat pulse
        'glow_mult':  1.9,
        'label':      'GROOVE',
    },
    'buildup': {
        'primary':    '#e65100',   # deep amber
        'accent':     '#f9a825',   # warm gold
        'slots':      [0, 1, 2, 3, 4, 5],  # spreading outward = tension building
        'decay':      0.25,        # tighter pulse — energy rising
        'glow_mult':  2.2,
        'label':      'BUILDUP',
    },
    'drop': {
        'primary':    '#b71c1c',   # blood crimson
        'accent':     '#ad1457',   # hot magenta
        'slots':      [0, 1, 2, 3, 4, 5, 6, 7],  # all fixtures
        'decay':      0.12,        # hard flash — cuts immediately after beat
        'glow_mult':  3.0,
        'label':      'DROP',
    },
    'peak': {
        'primary':    '#c62828',   # vivid red
        'accent':     '#ffffff',   # pure white center (center slots get accent)
        'slots':      [0, 1, 2, 3, 4, 5, 6, 7],  # all fixtures
        'decay':      0.20,        # sustained but snappy
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


def _effect_color(channel: str, effect_type: str) -> str:
    if effect_type == 'SPECIAL_EFFECT':
        return '#ff006e'
    if 'BANK_1' in channel:
        return '#f4a261'
    if 'BANK_2' in channel:
        return '#48cae4'
    return '#888888'


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_timeline(snapshot: dict) -> go.Figure:
    now   = snapshot['now']
    x0    = now - TIMELINE_WINDOW_SEC
    x1    = now + 0.5

    shapes, annotations = [], []

    for eff in snapshot['effects']:
        t_start = max(eff['t'], x0)
        t_end   = min(eff.get('end', now), x1)
        if t_end <= t_start:
            continue
        color = _effect_color(eff['channel'], eff['type'])
        shapes.append(dict(
            type='rect', xref='x', yref='paper',
            x0=t_start, x1=t_end, y0=0.52, y1=0.96,
            fillcolor=color, opacity=0.80, line_width=0,
        ))
        if t_end - t_start > 1.5:
            label = eff['channel'].split('_')[-1]
            annotations.append(dict(
                x=(t_start + t_end) / 2, y=0.74, xref='x', yref='paper',
                text=label, showarrow=False,
                font=dict(color='rgba(0,0,0,0.85)', size=11, family='monospace'),
            ))

    # Beat markers — fixed y=0.25, size scales with strength (onset density proxy)
    beat_x, beat_y, beat_size = [], [], []
    for b in snapshot['beats']:
        if b['t'] < x0:
            continue
        beat_x.append(b['t'])
        beat_y.append(0.25)
        beat_size.append(max(16, b['strength'] * 40))

    # "Now" cursor
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
        xaxis=dict(range=[x0, x1], gridcolor='#1a2332', color='#6e7681',
                   tickformat='.0f', ticksuffix='s', showline=False),
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

    # Beat pulse factor: 1.0 immediately after a beat → 0.0 after `decay` seconds
    now   = snapshot['now']
    beats = snapshot.get('beats', [])
    dt    = (now - beats[-1]['t']) if beats else 999.0
    pulse = max(0.0, 1.0 - dt / decay)  # linear decay; simple and smooth

    base_glow  = 16   # px, minimum glow radius when no beat
    pulse_glow = int(base_glow + pulse * base_glow * glow_m)

    slots = []
    for i, label in enumerate(SLOT_LABELS):
        on = i in active
        # Alternate primary/accent across active slots for visual variety.
        # Center fixtures (indices 3,4) get accent in PEAK for white-hot center effect.
        active_sorted = sorted(active)
        pos_in_active = active_sorted.index(i) if on else -1
        color = (accent if pos_in_active % 2 == 1 else primary) if on else None

        glow_px = pulse_glow if on else 0
        dim_bg  = '#161d27'  # not fully black — realistic stage bleed

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
    return [
        html.Span(status_lbl,   style={'color': status_col,  'marginRight': '20px', 'fontWeight': 'bold'}),
        html.Span(f'{int(elapsed)}s', style={'color': '#6e7681', 'marginRight': '20px'}),
        html.Span(f'{bpm:.0f} BPM',  style={'color': '#58a6ff', 'marginRight': '20px'}),
        html.Span(f'{beats} beats',   style={'color': '#3fb950', 'marginRight': '20px'}),
        html.Span(f'intent: {intent_lbl}', style={'color': intent_col, 'fontWeight': 'bold'}),
    ]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(event_buffer) -> dash.Dash:
    legend_items = [
        html.Span('SoundSwitch Visualizer',
                  style={'fontWeight': 'bold', 'color': '#e6edf3', 'marginRight': '28px'}),
    ]
    for key, cfg in INTENT_CONFIG.items():
        legend_items.append(
            html.Span(f'■ {cfg["label"]}',
                      style={'color': cfg['primary'], 'marginRight': '14px', 'fontSize': '12px'})
        )

    app = dash.Dash(__name__, title='SoundSwitch Visualizer')
    app.layout = html.Div([
        html.Div(legend_items, style={
            'padding': '12px 20px', 'borderBottom': f'1px solid {BORDER}',
            'fontFamily': 'monospace', 'fontSize': '14px',
        }),
        dcc.Graph(id='timeline', config={'displayModeBar': False},
                  style={'borderBottom': f'1px solid {BORDER}'}),
        html.Div(id='stage', style={
            'display': 'grid', 'gridTemplateColumns': 'repeat(8, 1fr)',
            'gap': '10px', 'padding': '20px 20px 16px',
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
         Output('metrics', 'children')],
        Input('tick', 'n_intervals'),
    )
    def refresh(_):
        snap = event_buffer.snapshot()
        return _build_timeline(snap), _build_stage(snap), _build_metrics(snap)

    return app


def run_app(event_buffer, port: int = 8050) -> None:
    app = build_app(event_buffer)
    print(f'\n  Visualizer → http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
