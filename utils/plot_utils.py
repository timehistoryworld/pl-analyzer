"""
Common plotting utilities for PL Analyzer
"""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = [
    '#4C9BE8', '#E85C4C', '#4CE87A', '#E8B84C',
    '#9B4CE8', '#4CE8D4', '#E84C9B', '#B8E84C',
    '#E8734C', '#4C6BE8'
]

LAYOUT_DEFAULTS = dict(
    template='plotly_dark',
    font=dict(family='monospace', size=13),
    paper_bgcolor='rgba(18,18,28,0.95)',
    plot_bgcolor='rgba(18,18,28,0.95)',
    margin=dict(l=60, r=30, t=50, b=60),
    legend=dict(
        bgcolor='rgba(30,30,45,0.8)',
        bordercolor='rgba(100,100,150,0.4)',
        borderwidth=1
    )
)

def make_figure(**kwargs):
    """Create a base figure with PL Analyzer styling."""
    fig = go.Figure()
    layout_kw = {**LAYOUT_DEFAULTS, **kwargs}
    fig.update_layout(**layout_kw)
    return fig

def style_axes(fig, xaxis_title='', yaxis_title='', grid=True):
    axis_style = dict(
        showgrid=grid,
        gridcolor='rgba(100,100,150,0.2)',
        linecolor='rgba(150,150,200,0.5)',
        tickcolor='rgba(150,150,200,0.5)',
        tickfont=dict(size=12),
    )
    fig.update_layout(
        xaxis=dict(title=xaxis_title, **axis_style),
        yaxis=dict(title=yaxis_title, **axis_style)
    )
    return fig

def add_spectrum(fig, x, y, name='', color=None, dash='solid', width=2, fill=False):
    """Add a spectrum trace."""
    trace_kwargs = dict(
        x=x, y=y, name=name, mode='lines',
        line=dict(color=color or COLORS[0], width=width, dash=dash)
    )
    if fill:
        trace_kwargs['fill'] = 'tozeroy'
        trace_kwargs['fillcolor'] = (color or COLORS[0]).replace(')', ',0.15)').replace('rgb', 'rgba').replace('#', '')
        # Fallback for hex colors
        if fill and '#' in (color or ''):
            trace_kwargs['fillcolor'] = hex_to_rgba(color or COLORS[0], 0.15)
    fig.add_trace(go.Scatter(**trace_kwargs))
    return fig

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

def spectral_colorscale():
    """Return a wavelength-like colorscale from violet to red."""
    return [
        [0.0,  '#8B00FF'],
        [0.15, '#4169E1'],
        [0.3,  '#00BFFF'],
        [0.45, '#00FF7F'],
        [0.6,  '#FFFF00'],
        [0.75, '#FFA500'],
        [1.0,  '#FF0000'],
    ]

def rainbow_colors(n):
    """Generate n colors evenly spaced across visible spectrum."""
    import colorsys
    colors = []
    for i in range(n):
        hue = 0.75 - 0.65 * (i / max(n-1, 1))  # violet → red
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    return colors

def add_vline(fig, x, label='', color='rgba(255,255,100,0.7)', dash='dash'):
    fig.add_vline(x=x, line=dict(color=color, dash=dash, width=1.5),
                  annotation_text=label,
                  annotation_font_color=color)
    return fig
