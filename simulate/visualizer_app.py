"""
Dash-based lighting visualizer.

Shows a rolling 30-second timeline of beats and effect changes, plus an 8-slot
stage view that reflects the currently active effect. Updated every 100 ms via
dcc.Interval polling the shared EventBuffer.

Color coding:
  Amber  (#f4a261) — AUTOLOOP BANK 1 (high-intensity scenes)
  Teal   (#48cae4) — AUTOLOOP BANK 2 (low/medium-intensity scenes)
  Magenta(#ff006e) — SPECIAL_EFFECT (strobe, static looks)
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLOT_LABELS = list('ABCDEFGH')
TIMELINE_WINDOW_SEC = 30.0

BANK_1_COLOR = '#f4a261'   # amber  — high intensity
BANK_2_COLOR = '#48cae4'   # teal   — low / medium
SPECIAL_COLOR = '#ff006e'  # magenta — special effects
DARK_BG = '#0d1117'
CARD_BG = '#111827'
BORDER = '#1e2937'


def _channel_to_slot(channel: str) -> int | None:
    """AUTOLOOP_BANK_1C → 2  (0-indexed A-H)"""
    for i, letter in enumerate('ABCDEFGH'):
        if channel.endswith(f'_{letter}'):
            return i
    return None


def _effect_color(channel: str, effect_type: str) -> str:
    if effect_type == 'SPECIAL_EFFECT':
        return SPECIAL_COLOR
    if 'BANK_1' in channel:
        return BANK_1_COLOR
    if 'BANK_2' in channel:
        return BANK_2_COLOR
    return '#888888'


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_timeline(snapshot: dict) -> go.Figure:
    now = snapshot['now']
    x0 = now - TIMELINE_WINDOW_SEC
    x1 = now + 0.5  # small margin at right edge

    shapes = []
    annotations = []

    # Effect bands as colored rectangles spanning y=[0.55, 0.95]
    for eff in snapshot['effects']:
        t_start = max(eff['t'], x0)
        t_end = min(eff.get('end', now), x1)
        if t_end <= t_start:
            continue
        color = _effect_color(eff['channel'], eff['type'])
        shapes.append(dict(
            type='rect', xref='x', yref='paper',
            x0=t_start, x1=t_end, y0=0.52, y1=0.96,
            fillcolor=color, opacity=0.80, line_width=0,
        ))
        if t_end - t_start > 1.5:
            # Show just the letter (A-H) so it fits
            label = eff['channel'].split('_')[-1]
            annotations.append(dict(
                x=(t_start + t_end) / 2, y=0.74, xref='x', yref='paper',
                text=label, showarrow=False,
                font=dict(color='rgba(0,0,0,0.85)', size=11, family='monospace'),
            ))

    # Beat markers as thin vertical lines from bottom, height = strength
    # Beat markers: fixed y=0.25 so they're always visible regardless of strength.
    # Size scales with Spotify-derived strength when available; minimum ensures
    # beats always show even without Spotify data (strength == 0).
    beat_x, beat_y, beat_size = [], [], []
    for b in snapshot['beats']:
        if b['t'] < x0:
            continue
        beat_x.append(b['t'])
        beat_y.append(0.25)
        beat_size.append(max(18, b['strength'] * 36))

    # "Now" cursor
    shapes.append(dict(
        type='line', xref='x', yref='paper',
        x0=now, x1=now, y0=0, y1=1,
        line=dict(color='rgba(255,255,255,0.25)', width=1, dash='dot'),
    ))

    fig = go.Figure()
    if beat_x:
        fig.add_trace(go.Scatter(
            x=beat_x, y=beat_y,
            mode='markers',
            marker=dict(
                symbol='line-ns',
                size=beat_size,
                color='rgba(168,218,220,0.65)',
                line=dict(color='rgba(168,218,220,0.65)', width=1.5),
            ),
            hoverinfo='skip',
        ))

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            range=[x0, x1],
            gridcolor='#1a2332', color='#6e7681',
            tickformat='.0f', ticksuffix='s',
            showline=False,
        ),
        yaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_BG,
        height=175,
        margin=dict(l=8, r=8, t=6, b=36),
        uirevision='timeline',
        showlegend=False,
    )
    return fig


def _build_stage(snapshot: dict) -> list:
    current = snapshot.get('current_effect')
    active_channel = current['channel'] if current else None
    active_slot = _channel_to_slot(active_channel) if active_channel else None
    active_color = _effect_color(active_channel, current['type']) if current else None

    slots = []
    for i, label in enumerate(SLOT_LABELS):
        on = (i == active_slot)
        color = active_color if on else None
        slots.append(html.Div([
            html.Div(style={
                'width': '38px', 'height': '38px', 'borderRadius': '50%',
                'background': color or '#1a2332',
                'margin': '0 auto 8px',
                'boxShadow': f'0 0 22px {color}' if on else 'none',
                'transition': 'all 0.12s ease',
            }),
            html.Div(label, style={
                'color': '#ffffff' if on else '#2d3f52',
                'fontSize': '12px', 'textAlign': 'center',
                'fontFamily': 'monospace', 'letterSpacing': '1px',
            }),
        ], style={
            'padding': '16px 8px',
            'background': CARD_BG,
            'borderRadius': '8px',
            'border': f'1px solid {color}' if on else f'1px solid {BORDER}',
            'transition': 'all 0.12s ease',
        }))
    return slots


def _build_metrics(snapshot: dict) -> list:
    bpm = snapshot.get('bpm', 0.0)
    beats = snapshot.get('beats_detected', 0)
    elapsed = snapshot.get('now', 0.0)
    current = snapshot.get('current_effect')
    effect_label = current['channel'] if current else '—'
    return [
        html.Span(f'{int(elapsed)}s', style={'color': '#6e7681', 'marginRight': '20px'}),
        html.Span(f'{bpm:.0f} BPM', style={'color': '#58a6ff', 'marginRight': '20px'}),
        html.Span(f'{beats} beats', style={'color': '#3fb950', 'marginRight': '20px'}),
        html.Span(f'active: {effect_label}', style={'color': '#d2a8ff'}),
    ]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def build_app(event_buffer) -> dash.Dash:
    app = dash.Dash(__name__, title='SoundSwitch Visualizer')
    app.layout = html.Div([
        # Header
        html.Div([
            html.Span('SoundSwitch Visualizer',
                      style={'fontWeight': 'bold', 'color': '#e6edf3', 'marginRight': '28px'}),
            html.Span('■ HIGH', style={'color': BANK_1_COLOR, 'marginRight': '14px', 'fontSize': '12px'}),
            html.Span('■ LOW / MED', style={'color': BANK_2_COLOR, 'marginRight': '14px', 'fontSize': '12px'}),
            html.Span('■ SPECIAL', style={'color': SPECIAL_COLOR, 'fontSize': '12px'}),
        ], style={
            'padding': '12px 20px', 'borderBottom': f'1px solid {BORDER}',
            'fontFamily': 'monospace', 'fontSize': '14px',
        }),

        # Timeline
        dcc.Graph(id='timeline', config={'displayModeBar': False},
                  style={'borderBottom': f'1px solid {BORDER}'}),

        # Stage: 8 fixture slots
        html.Div(id='stage', style={
            'display': 'grid', 'gridTemplateColumns': 'repeat(8, 1fr)',
            'gap': '10px', 'padding': '20px 20px 16px',
        }),

        # Metrics bar
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
