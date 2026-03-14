"""
dashboard/app.py — MarketShield Interactive Dashboard.

Launch with:
    streamlit run dashboard/app.py

Design aesthetic: Dark industrial terminal — like a Bloomberg terminal
crossed with a security operations centre. Monospaced data typography,
electric cyan accent on near-black, red alert pulses.

Layout:
  ┌─────────────────────────────────────────────────────────────┐
  │  HEADER: MarketShield + live stats ticker                    │
  ├──────────┬──────────────────────────────┬───────────────────┤
  │  ALERTS  │    PRICE + SENTIMENT CHART   │  SCORE GAUGE      │
  │  PANEL   │                              │  COMPONENT BARS   │
  │  (live   │                              │  DISTRIBUTION     │
  │   feed)  ├──────────────────────────────┤                   │
  │          │    NETWORK GRAPH             │  MARKET PRICES    │
  ├──────────┴──────────────────────────────┴───────────────────┤
  │  ARTICLE TABLE (full width, sortable)                        │
  ├──────────────────────────────────────────────────────────────┤
  │  VOLATILITY HEATMAP  │  AI HISTOGRAM   │  PIPELINE STATS    │
  └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so imports work when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from dashboard.data_layer import (
    generate_demo_articles,
    generate_demo_alerts,
    generate_price_series,
    generate_sentiment_series,
    generate_network_data,
    get_market_snapshot,
    get_score_distribution,
)
from dashboard.charts import (
    price_sentiment_chart,
    risk_gauge,
    component_bars,
    network_graph,
    score_distribution_donut,
    volatility_heatmap,
    ai_probability_histogram,
)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MarketShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS — dark terminal aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  /* Root theme */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0A0F !important;
    color: #E5E5EA;
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
  }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stDecoration"] { display: none; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0D0D16 !important;
    border-right: 1px solid #1C1C2E;
  }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #111118;
    border: 1px solid #1C1C2E;
    border-radius: 8px;
    padding: 14px 16px;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #636366 !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 24px !important;
    color: #E5E5EA !important;
  }
  [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
  }

  /* Section dividers */
  hr { border-color: #1C1C2E !important; margin: 8px 0; }

  /* Plotly charts */
  .js-plotly-plot { border-radius: 6px; overflow: hidden; }

  /* Alert cards */
  .alert-card {
    background: #111118;
    border-left: 3px solid var(--alert-color);
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }
  .alert-card .alert-title {
    font-size: 12px;
    font-weight: 500;
    color: #E5E5EA;
    margin-bottom: 4px;
    line-height: 1.3;
  }
  .alert-card .alert-meta {
    color: #636366;
    font-size: 10px;
  }
  .alert-card .alert-score {
    font-size: 20px;
    font-weight: 700;
  }

  /* News feed cards */
  .news-card {
    background: #111118;
    border: 1px solid #1C1C2E;
    border-left: 4px solid var(--card-color);
    border-radius: 0 6px 6px 0;
    padding: 10px 12px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    transition: border-color 0.2s;
  }
  .news-card:hover { border-color: #00D4FF; }
  .news-card .card-title {
    font-size: 11.5px; font-weight: 500;
    color: #E5E5EA; line-height: 1.35;
    margin-bottom: 5px;
  }
  .news-card .card-meta {
    font-size: 10px; color: #636366;
  }
  .news-card .card-score {
    font-size: 11px; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Section headers */
  .section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #636366;
    border-bottom: 1px solid #1C1C2E;
    padding-bottom: 6px;
    margin-bottom: 12px;
  }

  /* Ticker chips */
  .ticker-chip {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: #00D4FF;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 3px;
    margin: 0 2px;
    letter-spacing: 0.05em;
  }

  /* Price cells */
  .price-up   { color: #30D158; font-weight: 600; }
  .price-down { color: #FF2D55; font-weight: 600; }
  .price-flat { color: #E5E5EA; }

  /* Dataframe styling */
  [data-testid="stDataFrame"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px;
  }

  /* Selectbox and buttons */
  [data-testid="stSelectbox"] > div > div {
    background-color: #111118 !important;
    border-color: #2C2C3E !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }
  .stButton > button {
    background: #111118;
    border: 1px solid #2C2C3E;
    color: #E5E5EA;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    border-radius: 4px;
    padding: 6px 14px;
  }
  .stButton > button:hover {
    border-color: #00D4FF;
    color: #00D4FF;
  }

  /* Pulse animation for critical alerts */
  @keyframes pulse-red {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  .pulse { animation: pulse-red 1.5s infinite; }

  /* Scrollable containers */
  .scroll-container {
    max-height: 420px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #2C2C3E #0A0A0F;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "TSLA"
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0

st.session_state.refresh_count += 1
seed = st.session_state.refresh_count % 50  # Rotate data slightly on each refresh


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_data(seed: int):
    articles = generate_demo_articles(n=14, seed=seed)
    alerts   = generate_demo_alerts(articles)
    dist     = get_score_distribution(articles)
    return articles, alerts, dist


articles, alerts, dist = load_data(seed)
selected_ticker = st.session_state.selected_ticker

price_data     = generate_price_series(selected_ticker, inject_spike=True)
sentiment_data = generate_sentiment_series(ticker=selected_ticker)
network_data   = generate_network_data()
market_snaps   = get_market_snapshot(["TSLA", "GME", "NVDA", "AAPL", "BTC", "ETH"])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_logo, col_stats, col_controls = st.columns([3, 5, 2])

with col_logo:
    st.markdown("""
    <div style='padding: 8px 0'>
      <div style='font-family: JetBrains Mono, monospace; font-size: 22px;
                  font-weight: 700; color: #E5E5EA; letter-spacing: -0.5px;'>
        🛡️ MARKET<span style='color:#00D4FF'>SHIELD</span>
      </div>
      <div style='font-family: JetBrains Mono, monospace; font-size: 10px;
                  color: #636366; letter-spacing: 0.1em; text-transform: uppercase;
                  margin-top: 2px;'>
        AI Manipulation Detection System
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_stats:
    s1, s2, s3, s4, s5 = st.columns(5)
    total = sum(dist.values())
    critical_n = dist.get("critical", 0)
    high_n = dist.get("high", 0)
    ai_pct = sum(1 for a in articles if a["is_ai_generated"]) / max(1, len(articles))

    s1.metric("ARTICLES", f"{total}", f"+{seed % 5 + 1} new")
    s2.metric("🔴 CRITICAL", critical_n,
              delta=f"+{critical_n}", delta_color="inverse" if critical_n else "off")
    s3.metric("🟠 HIGH", high_n)
    s4.metric("AI DETECTED", f"{ai_pct:.0%}")
    s5.metric("LAST UPDATE",
              datetime.now(timezone.utc).strftime("%H:%M:%S"), "UTC")

with col_controls:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    selected_ticker = st.selectbox(
        "Monitor Ticker",
        ["TSLA", "GME", "NVDA", "AAPL", "AMC", "BTC", "ETH"],
        index=["TSLA", "GME", "NVDA", "AAPL", "AMC", "BTC", "ETH"].index(
            st.session_state.selected_ticker
        ),
        key="ticker_select",
        label_visibility="collapsed",
    )
    st.session_state.selected_ticker = selected_ticker

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("⟳ Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_r2:
        auto = st.toggle("Auto", value=st.session_state.auto_refresh, key="auto_toggle")
        st.session_state.auto_refresh = auto


st.markdown("<hr>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main layout: 3 columns
# ---------------------------------------------------------------------------
col_alerts, col_main, col_right = st.columns([1.6, 3.8, 1.8])


# ── LEFT: Alert feed ─────────────────────────────────────────────────────────
with col_alerts:
    n_alerts = len(alerts)
    pulse_cls = " pulse" if any(a["level"] == "critical" for a in alerts) else ""
    st.markdown(
        f'<div class="section-header{pulse_cls}">'
        f'⚡ LIVE ALERTS &nbsp; <span style="color:#FF2D55">{n_alerts} ACTIVE</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
    for alert in alerts:
        color = alert["color"]
        score_pct = f"{alert['score'] * 100:.0f}"
        tickers_html = "".join(
            f'<span class="ticker-chip">{t}</span>' for t in alert["tickers"]
        )
        # Parse time
        try:
            created = datetime.fromisoformat(alert["created_at"]).strftime("%H:%M")
        except Exception:
            created = "--:--"

        st.markdown(f"""
        <div class="alert-card" style="--alert-color: {color}">
          <div style="display:flex; justify-content:space-between; align-items:start">
            <div class="alert-title" style="flex:1; padding-right:8px">
              {alert['title'][:70]}
            </div>
            <div class="alert-score" style="color:{color}">{score_pct}</div>
          </div>
          <div class="alert-meta">
            <span style="color:#8E8E93">{alert['source']}</span>
            &nbsp;·&nbsp; {created}
            &nbsp;·&nbsp; {tickers_html}
          </div>
          <div class="alert-meta" style="margin-top:4px; color:#4A4A5A">
            {alert['summary']}
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Score distribution donut
    fig_donut = score_distribution_donut(dist)
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})


# ── CENTRE: Charts ───────────────────────────────────────────────────────────
with col_main:
    # Price + sentiment chart
    if selected_ticker != st.session_state.get("_last_ticker"):
        price_data = generate_price_series(selected_ticker, inject_spike=True)
        sentiment_data = generate_sentiment_series(ticker=selected_ticker)
        st.session_state["_last_ticker"] = selected_ticker

    fig_ps = price_sentiment_chart(price_data, sentiment_data)
    st.plotly_chart(fig_ps, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Network graph
    st.markdown(
        '<div class="section-header">🕸️ PROPAGATION NETWORK</div>',
        unsafe_allow_html=True
    )
    fig_net = network_graph(network_data)
    st.plotly_chart(fig_net, use_container_width=True, config={"displayModeBar": False})


# ── RIGHT: Risk panel + market prices ────────────────────────────────────────
with col_right:
    # Find most suspicious article for featured gauge
    top_article = articles[0] if articles else None

    if top_article:
        st.markdown(
            '<div class="section-header">🎯 TOP THREAT</div>',
            unsafe_allow_html=True
        )
        # Gauge
        fig_gauge = risk_gauge(
            top_article["composite_score"],
            top_article["title"][:35] + "…",
        )
        st.plotly_chart(fig_gauge, use_container_width=True,
                        config={"displayModeBar": False})

        # Component breakdown
        fig_bars = component_bars(
            top_article["ai_component"],
            top_article["propagation_component"],
            top_article["market_impact_component"],
        )
        st.plotly_chart(fig_bars, use_container_width=True,
                        config={"displayModeBar": False})

        # Threat detail
        color = top_article["level_color"]
        tickers_html = "".join(
            f'<span class="ticker-chip">{t}</span>'
            for t in top_article["tickers"]
        )
        st.markdown(f"""
        <div style="background:#111118; border:1px solid #1C1C2E; border-radius:6px;
                    padding:12px; font-family:'JetBrains Mono',monospace; font-size:10px;">
          <div style="color:{color}; font-weight:700; font-size:11px; margin-bottom:6px">
            {top_article['risk_level'].upper()} RISK
          </div>
          <div style="color:#E5E5EA; font-size:11px; line-height:1.4; margin-bottom:8px">
            {top_article['title'][:80]}
          </div>
          <div style="color:#636366; margin-bottom:4px">
            📡 {top_article['source']}
          </div>
          <div style="margin-top:6px">
            {tickers_html}
          </div>
          <div style="margin-top:8px; color:#4A4A5A; border-top:1px solid #1C1C2E; padding-top:8px">
            <span style="color:#8E8E93">Cluster size:</span> {top_article['cluster_size']} articles
            &nbsp;·&nbsp;
            <span style="color:#8E8E93">Velocity:</span> {top_article['velocity']}/hr
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Market prices panel
    st.markdown(
        '<div class="section-header">💹 MARKET PRICES</div>',
        unsafe_allow_html=True
    )
    for snap in market_snaps:
        chg = snap["change_24h"]
        chg_class = "price-up" if chg > 0 else ("price-down" if chg < 0 else "price-flat")
        chg_arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "–")
        regime_color = {
            "normal": "#636366", "elevated": "#FFD60A",
            "extreme": "#FF2D55", "low": "#30D158",
        }.get(snap["regime"], "#636366")

        price_fmt = (
            f"${snap['price']:,.6f}" if snap["price"] < 1
            else f"${snap['price']:,.2f}"
        )
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:7px 10px; background:#111118; border-radius:5px;
                    margin-bottom:5px; border:1px solid #1C1C2E;
                    font-family:'JetBrains Mono',monospace;">
          <div>
            <span style="font-size:12px; font-weight:700; color:#E5E5EA">
              {snap['ticker']}
            </span>
            <span style="font-size:8px; color:{regime_color}; margin-left:5px;
                         text-transform:uppercase; letter-spacing:0.05em">
              ● {snap['regime']}
            </span>
          </div>
          <div style="text-align:right">
            <div style="font-size:12px; font-weight:600; color:#E5E5EA">{price_fmt}</div>
            <div class="{chg_class}" style="font-size:10px">
              {chg_arrow} {abs(chg):.2f}%
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<hr>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Full-width article table
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="section-header">📰 ARTICLE INTELLIGENCE FEED</div>',
    unsafe_allow_html=True
)

# Build DataFrame
df_data = []
for a in articles:
    level_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
        a["risk_level"], "⚪"
    )
    try:
        ts = datetime.fromisoformat(a["published_at"]).strftime("%H:%M")
    except Exception:
        ts = "--"

    df_data.append({
        "Risk":     f"{level_icon} {a['risk_level'].upper()}",
        "Score":    a["composite_score"],
        "Title":    a["title"][:75],
        "Source":   a["source"],
        "AI Prob":  a["ai_component"],
        "Propagation": a["propagation_component"],
        "Market":   a["market_impact_component"],
        "Cluster":  a["cluster_size"],
        "Tickers":  ", ".join(a["tickers"]),
        "Time":     ts,
    })

df = pd.DataFrame(df_data)

# Style the dataframe
def _style_score(val):
    if val >= 0.85: return "color: #FF2D55; font-weight: bold"
    if val >= 0.65: return "color: #FF9500; font-weight: bold"
    if val >= 0.40: return "color: #FFD60A"
    return "color: #30D158"

styled = (
    df.style
    .applymap(_style_score, subset=["Score", "AI Prob", "Propagation", "Market"])
    .format({
        "Score":       "{:.3f}",
        "AI Prob":     "{:.0%}",
        "Propagation": "{:.0%}",
        "Market":      "{:.0%}",
    })
    .set_properties(**{
        "font-family": "JetBrains Mono, monospace",
        "font-size":   "11px",
        "background-color": "#111118",
        "color": "#E5E5EA",
    })
    .set_table_styles([{
        "selector": "th",
        "props": [
            ("background-color", "#0D0D16"),
            ("color", "#636366"),
            ("font-family", "JetBrains Mono, monospace"),
            ("font-size", "10px"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.08em"),
            ("border-bottom", "1px solid #1C1C2E"),
        ]
    }])
)

st.dataframe(styled, use_container_width=True, height=320)


st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Bottom row: Volatility heatmap + AI histogram + pipeline stats
# ---------------------------------------------------------------------------
col_heat, col_hist, col_stats_bottom = st.columns([2.5, 2.0, 1.5])

with col_heat:
    st.markdown(
        '<div class="section-header">🌡️ VOLATILITY HEATMAP</div>',
        unsafe_allow_html=True
    )
    fig_heat = volatility_heatmap(["TSLA", "GME", "NVDA", "AAPL", "BTC", "ETH"])
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

with col_hist:
    st.markdown(
        '<div class="section-header">🤖 AI DETECTION DISTRIBUTION</div>',
        unsafe_allow_html=True
    )
    fig_hist = ai_probability_histogram(articles)
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

with col_stats_bottom:
    st.markdown(
        '<div class="section-header">⚙️ PIPELINE STATUS</div>',
        unsafe_allow_html=True
    )

    total_scored = seed * 23 + 47
    alert_rate   = len(alerts) / max(1, total_scored) * 100

    stats_items = [
        ("Articles Scored",  f"{total_scored:,}",          "#E5E5EA"),
        ("Alerts Emitted",   f"{len(alerts)}",             "#FF9500"),
        ("AI Detected",      f"{sum(1 for a in articles if a['is_ai_generated'])}",  "#FF2D55"),
        ("Alert Rate",       f"{alert_rate:.1f}%",         "#FFD60A"),
        ("Avg Score",        f"{sum(a['composite_score'] for a in articles)/len(articles):.3f}", "#00D4FF"),
        ("Clusters Found",   f"{sum(1 for a in articles if a['cluster_size'] > 1)}",  "#636366"),
    ]

    for label, value, color in stats_items:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; padding:7px 10px;
                    background:#111118; border-radius:5px; margin-bottom:5px;
                    border:1px solid #1C1C2E; font-family:'JetBrains Mono',monospace;">
          <span style="font-size:10px; color:#636366; text-transform:uppercase;
                       letter-spacing:0.06em">{label}</span>
          <span style="font-size:13px; font-weight:700; color:{color}">{value}</span>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding:20px 0 8px;
            font-family:'JetBrains Mono',monospace; font-size:9px;
            color:#2C2C3E; letter-spacing:0.1em; text-transform:uppercase;">
  MarketShield v0.1 &nbsp;·&nbsp; AI Financial Manipulation Detection
  &nbsp;·&nbsp; Not Financial Advice
  &nbsp;·&nbsp; Research Use Only
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if st.session_state.auto_refresh:
    refresh_interval = 30  # seconds
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed >= refresh_interval:
        st.session_state.last_refresh = time.time()
        st.rerun()
    else:
        remaining = int(refresh_interval - elapsed)
        st.markdown(
            f'<div style="position:fixed; bottom:12px; right:16px; '
            f'font-family:JetBrains Mono,monospace; font-size:9px; '
            f'color:#2C2C3E; letter-spacing:0.05em;">'
            f'AUTO-REFRESH IN {remaining}s</div>',
            unsafe_allow_html=True,
        )
        time.sleep(1)
        st.rerun()
