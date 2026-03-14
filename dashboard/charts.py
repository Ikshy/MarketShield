"""
dashboard/charts.py — Plotly chart builders for the MarketShield dashboard.

All charts use a unified dark industrial theme:
  - Background: #0A0A0F (near-black, not pure black — avoids harsh contrast)
  - Grid:       #1A1A2E (subtle blue-grey grid lines)
  - Accent:     #00D4FF (electric cyan — the "signal" colour)
  - Warning:    #FF9500 (amber)
  - Danger:     #FF2D55 (alarm red)
  - Safe:       #30D158 (system green)
  - Text:       #E5E5EA (off-white)
  - Muted:      #8E8E93 (secondary labels)

Font: 'JetBrains Mono' for data labels (monospace = precision),
      'IBM Plex Sans' for titles and annotations.
"""

from __future__ import annotations

import math
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
_BG       = "#0A0A0F"
_BG2      = "#111118"
_GRID     = "#1C1C2E"
_BORDER   = "#2C2C3E"
_CYAN     = "#00D4FF"
_AMBER    = "#FF9500"
_RED      = "#FF2D55"
_GREEN    = "#30D158"
_YELLOW   = "#FFD60A"
_TEXT     = "#E5E5EA"
_MUTED    = "#636366"
_FONT_MONO  = "JetBrains Mono, Courier New, monospace"
_FONT_SANS  = "IBM Plex Sans, system-ui, sans-serif"

_LEVEL_COLORS = {
    "low":      _GREEN,
    "medium":   _YELLOW,
    "high":     _AMBER,
    "critical": _RED,
}


def _base_layout(title: str = "", height: int = 380) -> dict:
    """Base Plotly layout applied to every chart."""
    return dict(
        title=dict(text=title, font=dict(family=_FONT_SANS, size=14, color=_TEXT), x=0.02),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG2,
        font=dict(family=_FONT_MONO, color=_TEXT, size=11),
        height=height,
        margin=dict(l=48, r=20, t=44, b=40),
        xaxis=dict(
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(size=10, color=_MUTED),
            linecolor=_BORDER,
        ),
        yaxis=dict(
            gridcolor=_GRID, zerolinecolor=_GRID,
            tickfont=dict(size=10, color=_MUTED),
            linecolor=_BORDER,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=_BORDER,
            font=dict(size=10, color=_TEXT),
        ),
    )


# ---------------------------------------------------------------------------
# 1. Price + sentiment overlay chart
# ---------------------------------------------------------------------------
def price_sentiment_chart(
    price_data: dict,
    sentiment_data: dict,
) -> go.Figure:
    """
    Dual-axis chart: price line (top) + sentiment bar (bottom).
    AI-burst windows are highlighted with a red background band.
    Volatility spikes get vertical dashed markers.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=["", "", ""],
    )

    times  = price_data["times"]
    prices = price_data["prices"]
    sents  = sentiment_data["sentiments"]
    ai_probs = sentiment_data["ai_probs"]
    counts   = sentiment_data["article_counts"]

    # Align lengths
    n = min(len(times), len(prices), len(sents))
    times_s  = times[:n]
    prices_s = prices[:n]
    sents_s  = sents[:n]
    ai_s     = ai_probs[:n]
    counts_s = counts[:n]

    # ── Row 1: Price ──────────────────────────────────────────────────────
    # Gradient fill under price line
    fig.add_trace(go.Scatter(
        x=times_s, y=prices_s,
        mode="lines",
        line=dict(color=_CYAN, width=1.8, shape="spline", smoothing=0.7),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
        name=f"{price_data['ticker']} Price",
        hovertemplate="%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # Highlight AI-burst windows (high ai_prob periods)
    burst_start = None
    for i, (t, ap) in enumerate(zip(times_s, ai_s)):
        if ap >= 0.65 and burst_start is None:
            burst_start = t
        elif ap < 0.65 and burst_start is not None:
            fig.add_vrect(
                x0=burst_start, x1=t,
                fillcolor="rgba(255,45,85,0.08)",
                line_width=0,
                layer="below", row=1, col=1,
            )
            burst_start = None

    # ── Row 2: Sentiment bars ─────────────────────────────────────────────
    bar_colors = [
        _RED if s < -0.2 else (_GREEN if s > 0.2 else _MUTED)
        for s in sents_s
    ]
    fig.add_trace(go.Bar(
        x=times_s, y=sents_s,
        marker_color=bar_colors,
        marker_line_width=0,
        name="Sentiment",
        hovertemplate="Sentiment: %{y:.3f}<extra></extra>",
        opacity=0.85,
    ), row=2, col=1)

    fig.add_hline(y=0, line_color=_MUTED, line_width=0.8, row=2, col=1)

    # ── Row 3: AI probability + article count ─────────────────────────────
    fig.add_trace(go.Scatter(
        x=times_s, y=ai_s,
        mode="lines",
        line=dict(color=_AMBER, width=1.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(255,149,0,0.12)",
        name="AI Probability",
        hovertemplate="AI prob: %{y:.2f}<extra></extra>",
    ), row=3, col=1)

    # Threshold line
    fig.add_hline(y=0.70, line_color=_RED, line_dash="dash",
                  line_width=0.8, row=3, col=1)

    # Layout
    layout = _base_layout(
        title=f"📈 {price_data['ticker']}  "
              f"${price_data['current']:,.2f}  "
              f"({price_data['change_pct']:+.2f}%)",
        height=520,
    )
    layout.update(
        showlegend=True,
        xaxis3=dict(tickformat="%H:%M", nticks=8, gridcolor=_GRID),
        yaxis=dict(title="Price", tickprefix="$"),
        yaxis2=dict(title="Sentiment", range=[-1.1, 1.1]),
        yaxis3=dict(title="AI Prob", range=[0, 1.05]),
    )
    fig.update_layout(**layout)

    # Subplot title annotations
    for i, (label, row_y) in enumerate([
        ("PRICE", 0.97), ("SENTIMENT", 0.42), ("AI DETECTION", 0.17)
    ]):
        fig.add_annotation(
            text=label, x=0.01, y=row_y, xref="paper", yref="paper",
            font=dict(size=9, color=_MUTED, family=_FONT_MONO),
            showarrow=False,
        )

    return fig


# ---------------------------------------------------------------------------
# 2. Risk score gauge
# ---------------------------------------------------------------------------
def risk_gauge(score: float, label: str = "") -> go.Figure:
    """
    Semicircular gauge showing the composite risk score.
    Colour transitions from green → yellow → amber → red.
    """
    level_color = (
        _RED    if score >= 0.85 else
        _AMBER  if score >= 0.65 else
        _YELLOW if score >= 0.40 else
        _GREEN
    )
    level_name = (
        "CRITICAL" if score >= 0.85 else
        "HIGH"     if score >= 0.65 else
        "MEDIUM"   if score >= 0.40 else
        "LOW"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        number=dict(
            suffix="%",
            font=dict(size=36, family=_FONT_MONO, color=level_color),
        ),
        title=dict(
            text=f"<b>{label}</b><br><span style='font-size:11px;color:{level_color}'>"
                 f"{level_name}</span>",
            font=dict(family=_FONT_SANS, size=13, color=_TEXT),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1, tickcolor=_MUTED,
                tickfont=dict(size=9, color=_MUTED),
            ),
            bar=dict(color=level_color, thickness=0.28),
            bgcolor=_BG2,
            borderwidth=1, bordercolor=_BORDER,
            steps=[
                dict(range=[0,   40], color="rgba(48,209,88,0.15)"),
                dict(range=[40,  65], color="rgba(255,214,10,0.12)"),
                dict(range=[65,  85], color="rgba(255,149,0,0.12)"),
                dict(range=[85, 100], color="rgba(255,45,85,0.15)"),
            ],
            threshold=dict(
                line=dict(color=level_color, width=3),
                thickness=0.85,
                value=score * 100,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=_BG,
        font=dict(family=_FONT_MONO, color=_TEXT),
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Component breakdown horizontal bar
# ---------------------------------------------------------------------------
def component_bars(
    ai_score: float,
    prop_score: float,
    mkt_score: float,
) -> go.Figure:
    """
    Three horizontal bars showing AI, Propagation, and Market Impact components.
    Each bar has a coloured fill proportional to its score.
    """
    labels  = ["AI Content", "Propagation", "Market Impact"]
    scores  = [ai_score, prop_score, mkt_score]
    weights = [0.35, 0.30, 0.35]
    colors  = [
        _RED if s >= 0.70 else _AMBER if s >= 0.45 else _YELLOW if s >= 0.25 else _GREEN
        for s in scores
    ]

    fig = go.Figure()

    for i, (label, score, weight, color) in enumerate(zip(labels, scores, weights, colors)):
        # Background bar (full width = grey)
        fig.add_trace(go.Bar(
            x=[1.0], y=[label],
            orientation="h",
            marker_color="rgba(44,44,62,0.8)",
            showlegend=False,
            hoverinfo="skip",
            width=0.55,
        ))
        # Score bar
        fig.add_trace(go.Bar(
            x=[score], y=[label],
            orientation="h",
            marker_color=color,
            marker_line_width=0,
            showlegend=False,
            name=label,
            hovertemplate=f"{label}: %{{x:.1%}}<extra></extra>",
            width=0.55,
        ))
        # Score annotation
        fig.add_annotation(
            x=score + 0.02, y=label,
            text=f"<b>{score:.0%}</b>  <span style='color:{_MUTED}'>×{weight:.0%}</span>",
            font=dict(size=11, color=color, family=_FONT_MONO),
            showarrow=False, xanchor="left",
        )

    layout = _base_layout("Risk Component Breakdown", height=200)
    layout.update(
        barmode="overlay",
        xaxis=dict(range=[0, 1.35], tickformat=".0%", gridcolor=_GRID),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11, color=_TEXT)),
        showlegend=False,
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 4. Propagation network graph
# ---------------------------------------------------------------------------
def network_graph(network_data: dict) -> go.Figure:
    """
    Force-directed-style network graph of article propagation.
    Seed (AI-origin) nodes are large red; amplifiers amber/yellow; legitimate green.
    """
    nodes = network_data["nodes"]
    edges = network_data["edges"]

    # Simple circular layout with cluster perturbation
    import math as _math
    n = len(nodes)
    positions = {}
    rng_local = __import__("random")
    rng_local.seed(42)

    # Seeds at centre, others on outer rings
    seed_ids = {nd["id"] for nd in nodes if nd.get("color") == _RED}
    for nd in nodes:
        nid = nd["id"]
        if nid in seed_ids:
            angle = rng_local.uniform(0, 2 * _math.pi)
            r = rng_local.uniform(0, 0.15)
        else:
            angle = rng_local.uniform(0, 2 * _math.pi)
            r = rng_local.uniform(0.3, 1.0)
        positions[nid] = (r * _math.cos(angle), r * _math.sin(angle))

    # Edge traces
    edge_x, edge_y = [], []
    for e in edges:
        x0, y0 = positions[e["source"]]
        x1, y1 = positions[e["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(100,100,150,0.3)"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Nodes by risk group
    for color_group, label in [
        (_RED,    "AI Origin"),
        (_AMBER,  "High Risk"),
        (_YELLOW, "Medium Risk"),
        (_GREEN,  "Legitimate"),
    ]:
        group_nodes = [nd for nd in nodes if nd["color"] == color_group]
        if not group_nodes:
            continue

        xs = [positions[nd["id"]][0] for nd in group_nodes]
        ys = [positions[nd["id"]][1] for nd in group_nodes]
        sizes = [nd["size"] for nd in group_nodes]
        texts = [
            f"<b>{nd['source']}</b><br>"
            f"AI: {nd['ai_prob']:.0%}<br>"
            f"{nd['title'][:40]}"
            for nd in group_nodes
        ]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                size=sizes,
                color=color_group,
                line=dict(width=1.5, color=_BG),
                opacity=0.90,
            ),
            name=label,
            hovertemplate="%{text}<extra></extra>",
            text=texts,
        ))

    layout = _base_layout("🕸️  Propagation Network", height=420)
    layout.update(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(10,10,15,0.8)"),
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 5. Score distribution donut
# ---------------------------------------------------------------------------
def score_distribution_donut(distribution: dict) -> go.Figure:
    """Donut chart of risk level distribution across all scored articles."""
    labels = ["Critical", "High", "Medium", "Low"]
    values = [
        distribution.get("critical", 0),
        distribution.get("high", 0),
        distribution.get("medium", 0),
        distribution.get("low", 0),
    ]
    colors = [_RED, _AMBER, _YELLOW, _GREEN]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=_BG, width=2)),
        textfont=dict(family=_FONT_MONO, size=10, color=_TEXT),
        hovertemplate="%{label}: %{value} articles<extra></extra>",
        pull=[0.05, 0.02, 0, 0],
    ))

    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:10px;color:{_MUTED}'>ARTICLES</span>",
        x=0.5, y=0.5,
        font=dict(family=_FONT_MONO, size=18, color=_TEXT),
        showarrow=False,
    )

    fig.update_layout(
        paper_bgcolor=_BG,
        font=dict(family=_FONT_MONO, color=_TEXT),
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(
            orientation="v", x=0.82, y=0.5,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text="Risk Distribution",
            font=dict(family=_FONT_SANS, size=13, color=_TEXT), x=0.05
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Volatility heatmap (ticker × time)
# ---------------------------------------------------------------------------
def volatility_heatmap(tickers: list[str], hours: int = 12) -> go.Figure:
    """
    Heatmap showing simulated volatility (absolute returns) per ticker over time.
    Red cells = anomalous, green = quiet.
    """
    import numpy as np
    n_tickers = len(tickers)
    n_bins = 24

    rng = np.random.default_rng(55)
    data = rng.uniform(0, 0.025, size=(n_tickers, n_bins))

    # Inject anomalies for meme stocks
    for i, t in enumerate(tickers):
        if t.upper() in ("TSLA", "GME", "AMC", "DOGE", "DOGECOIN"):
            spike_col = rng.integers(14, 22)
            data[i, spike_col:spike_col+2] *= rng.uniform(4, 8)

    now = datetime.now()
    time_labels = [
        (now - __import__("datetime").timedelta(hours=n_bins - j)).strftime("%H:%M")
        for j in range(n_bins)
    ]

    fig = go.Figure(go.Heatmap(
        z=data,
        x=time_labels,
        y=[t.upper() for t in tickers],
        colorscale=[
            [0.00, "#0A2010"],
            [0.30, "#30D158"],
            [0.60, "#FFD60A"],
            [0.80, "#FF9500"],
            [1.00, "#FF2D55"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="|Return|", font=dict(size=10, color=_MUTED)),
            tickformat=".1%",
            tickfont=dict(size=9, color=_MUTED),
            bgcolor=_BG,
            bordercolor=_BORDER,
        ),
        hovertemplate="%{y} at %{x}: %{z:.2%}<extra></extra>",
    ))

    layout = _base_layout("Volatility Heatmap", height=280)
    layout.update(
        xaxis=dict(tickangle=0, nticks=8),
        yaxis=dict(autorange="reversed"),
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# 7. AI probability histogram
# ---------------------------------------------------------------------------
def ai_probability_histogram(articles: list[dict]) -> go.Figure:
    """
    Histogram of AI probability scores across all analysed articles.
    Red region = AI-detected; blue = human-likely.
    """
    scores = [a["ai_component"] for a in articles]
    threshold = 0.70

    fig = go.Figure()

    # Human region
    human_scores = [s for s in scores if s < threshold]
    fig.add_trace(go.Histogram(
        x=human_scores, nbinsx=20,
        marker_color=_GREEN, opacity=0.75,
        name="Human-Likely",
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))

    # AI region
    ai_scores = [s for s in scores if s >= threshold]
    fig.add_trace(go.Histogram(
        x=ai_scores, nbinsx=20,
        marker_color=_RED, opacity=0.75,
        name="AI-Detected",
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))

    # Threshold line
    fig.add_vline(
        x=threshold,
        line_color=_AMBER, line_dash="dash", line_width=1.5,
        annotation_text=" threshold",
        annotation_font=dict(size=9, color=_AMBER),
    )

    layout = _base_layout("AI Probability Distribution", height=260)
    layout.update(
        barmode="overlay",
        xaxis=dict(title="AI Probability", tickformat=".0%", range=[0, 1]),
        yaxis=dict(title="Count"),
    )
    fig.update_layout(**layout)
    return fig
