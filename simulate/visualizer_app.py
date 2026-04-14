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



# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_timeline(snapshot: dict, labels: list[dict] | None = None) -> go.Figure:
    now   = snapshot['now']
    x0    = now - TIMELINE_WINDOW_SEC
    x1    = now + 0.5

    has_labels = bool(labels)
    # When ground-truth labels are loaded, split the band into two rows.
    # Top row = predicted intent; bottom row = ground truth.
    if has_labels:
        pred_y0, pred_y1 = 0.54, 0.97
        gt_y0,   gt_y1   = 0.03, 0.46
        beat_y            = 0.50
    else:
        pred_y0, pred_y1 = 0.52, 0.96
        gt_y0,   gt_y1   = None, None
        beat_y            = 0.25

    shapes, annotations = [], []

    # Minor gridlines every 1 s (drawn first so intent bands render on top)
    t_grid = int(x0)
    while t_grid <= x1:
        shapes.append(dict(
            type='line', xref='x', yref='paper',
            x0=t_grid, x1=t_grid, y0=0, y1=1,
            line=dict(color='#151e2b', width=0.8),
        ))
        t_grid += 1

    # Predicted intent bands
    for entry in snapshot.get('intents', []):
        t_start = max(entry['t'], x0)
        t_end   = min(entry.get('end', now), x1)
        if t_end <= t_start:
            continue
        cfg   = _intent_config(entry['intent'])
        color = cfg['primary']
        shapes.append(dict(
            type='rect', xref='x', yref='paper',
            x0=t_start, x1=t_end, y0=pred_y0, y1=pred_y1,
            fillcolor=color, opacity=0.80, line_width=0,
        ))
        if t_end - t_start > 1.5:
            annotations.append(dict(
                x=(t_start + t_end) / 2, y=(pred_y0 + pred_y1) / 2,
                xref='x', yref='paper',
                text=cfg['label'], showarrow=False,
                font=dict(color='rgba(255,255,255,0.85)', size=10, family='monospace'),
            ))

    # Ground-truth bands (when a label file is loaded)
    if has_labels:
        for lbl in labels:
            t_start = max(lbl['start'], x0)
            t_end   = min(lbl['end'],   x1)
            if t_end <= t_start:
                continue
            cfg   = _intent_config(lbl['intent'])
            # Base fill: same color but lower opacity so it reads as "reference"
            shapes.append(dict(
                type='rect', xref='x', yref='paper',
                x0=t_start, x1=t_end, y0=gt_y0, y1=gt_y1,
                fillcolor=cfg['primary'], opacity=0.30, line_width=0,
            ))
            # Diagonal hatch lines (one every 1.5s across the band)
            stripe_x = t_start
            while stripe_x < t_end:
                shapes.append(dict(
                    type='line', xref='x', yref='paper',
                    x0=stripe_x, x1=min(stripe_x + 1.0, t_end),
                    y0=gt_y0,    y1=gt_y1,
                    line=dict(color=cfg['primary'], width=1.5),
                ))
                stripe_x += 1.5
            if t_end - t_start > 1.5:
                annotations.append(dict(
                    x=(t_start + t_end) / 2, y=(gt_y0 + gt_y1) / 2,
                    xref='x', yref='paper',
                    text=cfg['label'], showarrow=False,
                    font=dict(color='rgba(255,255,255,0.70)', size=10, family='monospace'),
                ))

        # Row labels pinned to left edge
        annotations.append(dict(
            x=0.005, y=(pred_y0 + pred_y1) / 2, xref='paper', yref='paper',
            text='PRED', showarrow=False, xanchor='left',
            font=dict(color='#6e7681', size=8, family='monospace'),
        ))
        annotations.append(dict(
            x=0.005, y=(gt_y0 + gt_y1) / 2, xref='paper', yref='paper',
            text='GT', showarrow=False, xanchor='left',
            font=dict(color='#6e7681', size=8, family='monospace'),
        ))

    # Sound start / stop markers
    for ev in snapshot.get('sound_events', []):
        if ev['t'] < x0:
            continue
        is_start = ev['playing']
        color    = '#3fb950' if is_start else '#f85149'   # green / red
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

    # Beat markers — y position depends on whether GT row is present
    beat_x, beat_y_list, beat_size = [], [], []
    for b in snapshot['beats']:
        if b['t'] < x0:
            continue
        beat_x.append(b['t'])
        beat_y_list.append(beat_y)
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
            x=beat_x, y=beat_y_list, mode='markers',
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
            dtick=5.0,              # major labelled tick every 5 s
            tickformat='.0f',
            ticksuffix='s',
            gridcolor='#1a2332',    # major gridlines (5 s)
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

    ts          = snapshot.get('timing_stats', {})
    mean_delta  = ts.get('mean_delta_sec')
    max_err     = ts.get('max_error_ms')
    n_samples   = ts.get('samples', 0)
    if mean_delta is not None and n_samples > 0:
        delay_str   = f'cmd delay: {mean_delta:.3f}s'
        delay_col   = '#3fb950' if abs(mean_delta - 2.5) < 0.05 else '#f0883e'
        err_str     = f'max err: {max_err:.1f}ms'
    else:
        delay_str  = 'cmd delay: —'
        delay_col  = '#6e7681'
        err_str    = ''

    items = [
        html.Span(status_lbl,   style={'color': status_col,  'marginRight': '20px', 'fontWeight': 'bold'}),
        html.Span(f'{elapsed:.1f}s', style={'color': '#6e7681', 'marginRight': '20px'}),
        html.Span(f'{bpm:.0f} BPM',  style={'color': '#58a6ff', 'marginRight': '20px'}),
        html.Span(f'{beats} beats',   style={'color': '#3fb950', 'marginRight': '20px'}),
        html.Span(f'intent: {intent_lbl}', style={'color': intent_col, 'fontWeight': 'bold', 'marginRight': '20px'}),
        html.Span(delay_str, style={'color': delay_col, 'marginRight': '12px'}),
    ]
    if err_str:
        items.append(html.Span(err_str, style={'color': '#6e7681'}))
    return items


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(event_buffer, labels: list[dict] | None = None) -> dash.Dash:
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
        return _build_timeline(snap, labels=labels), _build_stage(snap), _build_metrics(snap)

    return app


def run_app(event_buffer, port: int = 8050, labels: list[dict] | None = None) -> None:
    app = build_app(event_buffer, labels=labels)
    print(f'\n  Visualizer → http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
