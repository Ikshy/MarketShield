"""
dashboard/data_layer.py — Dashboard data provider.

Abstracts data sourcing from the UI layer. In production, reads from
the live RiskPipeline and MarketMonitor. In demo/dev mode, generates
realistic synthetic data so the dashboard runs without the full pipeline.

The Streamlit app imports ONLY from this module — never directly from
pipeline stages. This makes the dashboard independently runnable.
"""

from __future__ import annotations

import math
import random
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data generators (demo mode)
# ---------------------------------------------------------------------------
_FAKE_HEADLINES = [
    ("BREAKING: Coordinated short attack detected on TSLA options chain",
     "r/wallstreetbets", "reddit", 0.91, 0.88, 0.82, ["TSLA"]),
    ("Tesla TSLA Q4 earnings beat: revenue $25.2B, EPS $2.27",
     "Reuters", "rss", 0.04, 0.00, 0.05, ["TSLA"]),
    ("URGENT: Elon Musk facing imminent SEC investigation sources say",
     "CryptoFakeNews.io", "rss", 0.94, 0.91, 0.79, ["TSLA", "SEC"]),
    ("Bitcoin BTC falls 8% as Fed signals higher for longer",
     "CoinDesk", "rss", 0.06, 0.02, 0.11, ["BTC", "BITCOIN"]),
    ("GME short squeeze incoming — hedge funds covering positions",
     "r/Superstonk", "reddit", 0.72, 0.65, 0.55, ["GME"]),
    ("NVDA reports blowout quarter: datacenter revenue up 206%",
     "Bloomberg", "rss", 0.03, 0.01, 0.08, ["NVDA"]),
    ("AMC stock will 10x by Friday — AI model predicts squeeze",
     "StockAlertBot.net", "rss", 0.88, 0.76, 0.43, ["AMC"]),
    ("Federal Reserve holds rates steady, signals 2 cuts in 2025",
     "Wall Street Journal", "rss", 0.05, 0.00, 0.06, ["SPY", "TLT"]),
    ("Ethereum ETH merge upgrade causes massive network outage SELL NOW",
     "FakeBlockchain.xyz", "rss", 0.93, 0.84, 0.67, ["ETH", "ETHEREUM"]),
    ("AAPL iPhone 17 sales disappoint in China, analyst downgrades",
     "CNBC", "rss", 0.07, 0.03, 0.18, ["AAPL"]),
    ("Dogecoin DOGE pumping — Musk tweet imminent say insiders",
     "r/dogecoin", "reddit", 0.85, 0.79, 0.51, ["DOGE", "DOGECOIN"]),
    ("TSLA stock will collapse to zero — leaked internal memo",
     "ManipulationCentral.io", "rss", 0.96, 0.95, 0.88, ["TSLA"]),
    ("Microsoft MSFT acquires OpenAI for $97 billion in landmark deal",
     "TechCrunch", "rss", 0.08, 0.01, 0.09, ["MSFT"]),
    ("Coordinated bot network spreading TSLA FUD across 47 platforms",
     "MarketShield Internal", "rss", 0.97, 0.96, 0.91, ["TSLA"]),
    ("GameStop reports first profitable quarter since 2021",
     "MarketWatch", "rss", 0.06, 0.02, 0.12, ["GME"]),
]

_TICKER_PRICES = {
    "TSLA": 187.40, "GME": 22.15, "NVDA": 487.80, "AAPL": 182.60,
    "AMC": 4.22, "MSFT": 378.90, "SPY": 521.40,
    "BITCOIN": 67_420, "ETHEREUM": 3_840, "DOGECOIN": 0.1820,
    "SOLANA": 178.50, "BTC": 67_420, "ETH": 3_840, "DOGE": 0.1820,
}


def _risk_level(score: float) -> str:
    if score >= 0.85: return "critical"
    if score >= 0.65: return "high"
    if score >= 0.40: return "medium"
    return "low"


def _level_color(level: str) -> str:
    return {
        "critical": "#FF2D55",
        "high":     "#FF9500",
        "medium":   "#FFD60A",
        "low":      "#30D158",
    }.get(level, "#8E8E93")


def generate_demo_articles(n: int = 14, seed: int = 42) -> list[dict]:
    """Generate a list of synthetic article dicts for dashboard display."""
    random.seed(seed)
    articles = []
    now = datetime.now(timezone.utc)

    for i, (title, source, src_type, ai_p, prop_p, mkt_p, tickers) in \
            enumerate(_FAKE_HEADLINES[:n]):
        composite = round(0.35 * ai_p + 0.30 * prop_p + 0.35 * mkt_p, 3)
        level = _risk_level(composite)
        minutes_ago = random.uniform(i * 4, i * 4 + 15)

        articles.append({
            "id":          hashlib.md5(title.encode()).hexdigest()[:8],
            "title":       title,
            "source":      source,
            "source_type": src_type,
            "published_at": (now - timedelta(minutes=minutes_ago)).isoformat(),
            "composite_score":       composite,
            "ai_component":          round(ai_p, 3),
            "propagation_component": round(prop_p, 3),
            "market_impact_component": round(mkt_p, 3),
            "risk_level":   level,
            "level_color":  _level_color(level),
            "tickers":      tickers,
            "is_ai_generated": ai_p >= 0.70,
            "perplexity":   round(15.0 + (1 - ai_p) * 85, 1),
            "burstiness":   round(0.05 + (1 - ai_p) * 0.55, 3),
            "cluster_size": max(1, int(prop_p * 20 + random.uniform(-2, 2))),
            "velocity":     round(prop_p * 15, 1),
        })

    # Sort by score descending
    return sorted(articles, key=lambda x: x["composite_score"], reverse=True)


def generate_demo_alerts(articles: list[dict]) -> list[dict]:
    """Extract high-risk articles as formatted alerts."""
    alerts = []
    for a in articles:
        if a["composite_score"] >= 0.65:
            alerts.append({
                "alert_id":   a["id"],
                "title":      f"{'🔴' if a['risk_level']=='critical' else '🟠'} "
                              f"{a['risk_level'].upper()}: {a['title'][:55]}",
                "source":     a["source"],
                "score":      a["composite_score"],
                "level":      a["risk_level"],
                "color":      a["level_color"],
                "tickers":    a["tickers"],
                "created_at": a["published_at"],
                "summary":    (
                    f"AI prob {a['ai_component']:.0%} | "
                    f"Propagation {a['propagation_component']:.0%} | "
                    f"Market impact {a['market_impact_component']:.0%}"
                ),
            })
    return alerts


def generate_price_series(
    ticker: str,
    hours: int = 24,
    n_points: int = 96,
    inject_spike: bool = False,
    spike_at: float = 0.75,
) -> dict:
    """
    Generate a realistic price time series using GBM.
    Optionally injects a manipulation-driven spike.
    """
    base = _TICKER_PRICES.get(ticker.upper(), 100.0)
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16) % 10000
    rng = np.random.default_rng(seed)

    mu    = 0.0001
    sigma = 0.008 if ticker.upper() not in ("BITCOIN", "ETHEREUM", "DOGECOIN") else 0.018

    prices, times, returns = [base], [], [0.0]
    now = datetime.now(timezone.utc)

    for i in range(1, n_points):
        dt = hours / n_points
        shock = rng.normal(0, 1)

        # Inject spike (manipulation-driven)
        if inject_spike and abs(i / n_points - spike_at) < 0.04:
            shock = -6.0  # Sharp drop triggered by FUD campaign

        price = prices[-1] * math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * shock)
        price = max(0.01, price)
        prices.append(price)
        log_r = math.log(price / prices[-2]) if prices[-2] > 0 else 0.0
        returns.append(round(log_r, 6))
        times.append((now - timedelta(hours=hours) + timedelta(hours=i * dt)).isoformat())

    times.insert(0, (now - timedelta(hours=hours)).isoformat())

    return {
        "ticker":    ticker.upper(),
        "times":     times,
        "prices":    [round(p, 4) for p in prices],
        "returns":   returns,
        "current":   round(prices[-1], 4),
        "change_pct": round((prices[-1] / prices[0] - 1) * 100, 2),
        "vol_ann":   round(float(np.std(returns[1:])) * math.sqrt(8760) * 100, 2),
    }


def generate_sentiment_series(
    n_points: int = 48,
    ticker: str = "TSLA",
    inject_ai_burst: bool = True,
) -> dict:
    """
    Generate aligned sentiment + article count time series.
    Optionally injects a burst of AI-generated negative sentiment.
    """
    rng = np.random.default_rng(7)
    now = datetime.now(timezone.utc)
    times, sentiments, ai_probs, counts = [], [], [], []

    for i in range(n_points):
        t = now - timedelta(hours=n_points - i)
        times.append(t.isoformat())

        # Burst of AI-generated negative content at 75% through
        in_burst = inject_ai_burst and (0.70 <= i / n_points <= 0.85)

        sentiment = rng.normal(-0.75 if in_burst else 0.05, 0.15)
        sentiment = float(np.clip(sentiment, -1, 1))
        ai_prob   = float(np.clip(rng.normal(0.88 if in_burst else 0.12, 0.08), 0, 1))
        count     = int(rng.poisson(8 if in_burst else 2))

        sentiments.append(round(sentiment, 4))
        ai_probs.append(round(ai_prob, 4))
        counts.append(count)

    return {
        "ticker": ticker, "times": times,
        "sentiments": sentiments, "ai_probs": ai_probs,
        "article_counts": counts,
    }


def generate_network_data(n_nodes: int = 22) -> dict:
    """
    Generate a propagation network graph for the network visualisation.
    Creates a realistic hub-and-spoke + cluster topology.
    """
    rng = np.random.default_rng(99)
    nodes, edges = [], []

    # Seed nodes (AI-generated "origin" articles)
    seed_ids = list(range(3))
    for i in seed_ids:
        nodes.append({
            "id": i, "label": f"Seed_{i}",
            "ai_prob": round(float(rng.uniform(0.88, 0.97)), 2),
            "size": 28, "color": "#FF2D55",
            "source": random.choice(["FakeNews.io", "CryptoScam.net", "PumpBot.xyz"]),
            "title": random.choice([
                "BREAKING: TSLA collapse imminent",
                "Musk arrested FBI sources say",
                "GME squeeze guaranteed 1000%",
            ]),
        })

    # Propagation nodes
    source_names = [
        "r/wallstreetbets", "r/Superstonk", "StockAlert24", "CoinPump.io",
        "TelegramBot1", "TelegramBot2", "Twitter_Bot_A", "Twitter_Bot_B",
        "FakeNews2.net", "CryptoFUD.xyz", "Reuters", "Bloomberg",
        "MarketWatch", "CoinDesk", "CNBC", "r/investing",
        "SeekingAlpha", "ZeroHedge",
    ]

    for i in range(3, n_nodes):
        ai_prob = float(rng.uniform(0.05, 0.95))
        is_legit = ai_prob < 0.30
        nodes.append({
            "id": i,
            "label": f"Node_{i}",
            "ai_prob": round(ai_prob, 2),
            "size": 16 if not is_legit else 10,
            "color": "#FF9500" if ai_prob > 0.65 else ("#FFD60A" if ai_prob > 0.35 else "#30D158"),
            "source": source_names[(i - 3) % len(source_names)],
            "title": "Amplification node" if not is_legit else "Independent reporting",
        })

    # Edges: seeds → amplifiers
    for src in seed_ids:
        n_targets = rng.integers(3, 7)
        targets = rng.choice(range(3, n_nodes), size=int(n_targets), replace=False)
        for tgt in targets:
            edges.append({
                "source": src, "target": int(tgt),
                "weight": round(float(rng.uniform(0.6, 1.0)), 2),
                "type": "cluster",
            })

    # Secondary spread
    for _ in range(12):
        src = int(rng.integers(3, n_nodes))
        tgt = int(rng.integers(3, n_nodes))
        if src != tgt:
            edges.append({
                "source": src, "target": tgt,
                "weight": round(float(rng.uniform(0.3, 0.7)), 2),
                "type": "temporal",
            })

    return {"nodes": nodes, "edges": edges}


def get_market_snapshot(tickers: list[str]) -> list[dict]:
    """Return current simulated market snapshot for display."""
    rng = np.random.default_rng(int(datetime.now().timestamp()) % 10000)
    snaps = []
    for t in tickers:
        base = _TICKER_PRICES.get(t.upper(), 50.0)
        price = base * (1 + rng.normal(0, 0.012))
        chg_24h = float(rng.normal(0.3, 2.8))
        chg_1h  = float(rng.normal(0.0, 0.9))
        snaps.append({
            "ticker":   t.upper(),
            "price":    round(max(0.0001, price), 4 if base < 10 else 2),
            "change_24h": round(chg_24h, 2),
            "change_1h":  round(chg_1h, 2),
            "regime":   random.choice(["normal", "normal", "elevated", "extreme"])
                        if abs(chg_24h) > 3 else "normal",
        })
    return snaps


def get_score_distribution(articles: list[dict]) -> dict:
    dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for a in articles:
        dist[a["risk_level"]] += 1
    return dist
