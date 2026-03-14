"""
market_analysis/correlator.py — Sentiment ↔ price movement correlator.

Measures how closely news sentiment (and AI-manipulation scores) track
asset price movements in time. High correlation between suspicious
content and price spikes is the core signal of financial manipulation.

Three correlation analyses:

  1. Contemporaneous correlation
     Pearson r between sentiment polarity scores and % price returns
     in the same time window. Strong negative sentiment + negative
     price move = consistent (normal). Positive sentiment + negative
     price = contradictory (suspicious).

  2. Lead-lag analysis
     Granger-causality approximation: does sentiment at time T predict
     price at T+k? Detected manipulation often shows the narrative
     *preceding* the price move (news leads price).

  3. AI content → price impact
     Correlation between AI-probability scores and abnormal returns.
     If high-AI-probability articles consistently precede price moves,
     that's a manipulation signal independent of sentiment direction.

Outputs:
  - CorrelationResult per ticker (Pearson r, p-value, lag estimate)
  - ManipulationCorrelationScore: 0–1 combined score
  - TimeSeriesAlignment: visualisation data for the dashboard
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from logger import get_logger
from models import EnrichedArticle, MarketSnapshot, VolatilityAlert

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CorrelationResult:
    """Pearson correlation between sentiment/AI score and price returns."""
    ticker: str
    pearson_r: float                    # -1 to +1
    p_value: float                      # Statistical significance
    n_samples: int
    lag_hours: int                      # Best predictive lag (0 = contemporaneous)
    direction: str                      # "consistent" | "contradictory" | "neutral"
    sentiment_mean: float
    return_mean: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05 and self.n_samples >= 5

    @property
    def is_contradictory(self) -> bool:
        """Positive narrative + negative returns = possible manipulation signal."""
        return (
            self.is_significant
            and self.pearson_r < -0.3
            and self.sentiment_mean > 0.1   # Positive sentiment
            and self.return_mean < -0.005   # But negative returns
        )


@dataclass
class ManipulationCorrelationScore:
    """
    Combined 0–1 score measuring how strongly the content-market
    correlation pattern matches known manipulation profiles.
    """
    ticker: str
    score: float                        # 0–1
    ai_price_correlation: float         # r between AI prob and |returns|
    sentiment_contradiction_score: float
    lead_lag_score: float               # How much content leads price
    evidence_summary: str = ""

    @property
    def is_suspicious(self) -> bool:
        return self.score >= 0.50


@dataclass
class TimeSeriesPoint:
    """One point in a sentiment/price time series (for chart rendering)."""
    timestamp: datetime
    sentiment_score: float        # -1 to +1 (negative = bearish)
    ai_probability: float
    article_count: int
    price: Optional[float] = None
    price_return: Optional[float] = None
    volume: Optional[float] = None


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def _pearson_r(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Compute Pearson r and approximate p-value.
    Returns (r, p_value). Uses scipy if available, else manual calculation.
    """
    n = min(len(x), len(y))
    if n < 3:
        return 0.0, 1.0

    x_arr = np.array(x[:n], dtype=float)
    y_arr = np.array(y[:n], dtype=float)

    # Remove NaN
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_clean, y_clean = x_arr[mask], y_arr[mask]
    n_clean = len(x_clean)

    if n_clean < 3:
        return 0.0, 1.0

    # Pearson r
    x_mean, y_mean = x_clean.mean(), y_clean.mean()
    x_dev = x_clean - x_mean
    y_dev = y_clean - y_mean
    num = float(np.dot(x_dev, y_dev))
    denom = float(np.sqrt(np.dot(x_dev, x_dev) * np.dot(y_dev, y_dev)))

    if denom < 1e-10:
        return 0.0, 1.0

    r = float(np.clip(num / denom, -1.0, 1.0))

    # Approximate t-statistic → p-value
    try:
        t_stat = r * math.sqrt(n_clean - 2) / math.sqrt(max(1e-10, 1 - r**2))
        # Two-tailed p-value approximation (accurate for n > 10)
        # Using the relationship: p ≈ 2 * (1 - Φ(|t|))
        # Φ approximated via erfc
        p_value = float(math.erfc(abs(t_stat) / math.sqrt(2)))
    except Exception:
        p_value = 1.0

    return round(r, 4), round(max(0.0, min(1.0, p_value)), 6)


def _sentiment_to_score(article: EnrichedArticle) -> float:
    """
    Convert EnrichedArticle sentiment to a scalar in [-1, +1].
    Positive sentiment → +1, Negative → -1, Neutral → 0.
    """
    if not article.sentiment:
        return 0.0
    from models import Sentiment
    label = article.sentiment.label
    conf  = article.sentiment.score
    if label == Sentiment.POSITIVE:
        return conf
    elif label == Sentiment.NEGATIVE:
        return -conf
    return 0.0


def _compute_lag_correlation(
    sentiment_series: list[float],
    return_series: list[float],
    max_lag: int = 6,
) -> tuple[int, float]:
    """
    Find the lag (in time bins) at which sentiment best predicts returns.
    Returns (best_lag, best_r).

    A positive lag means sentiment at T predicts price at T+lag
    (sentiment leads price — manipulation signal).
    """
    if len(sentiment_series) < max_lag + 3:
        return 0, 0.0

    best_lag, best_r = 0, 0.0

    for lag in range(0, max_lag + 1):
        if lag == 0:
            s = sentiment_series
            r = return_series
        else:
            s = sentiment_series[:-lag]
            r = return_series[lag:]

        pearson_r, _ = _pearson_r(s, r)
        if abs(pearson_r) > abs(best_r):
            best_lag, best_r = lag, pearson_r

    return best_lag, best_r


# ---------------------------------------------------------------------------
# Time-series aggregation
# ---------------------------------------------------------------------------
def _bucket_articles(
    articles: list[EnrichedArticle],
    bucket_hours: float = 1.0,
) -> dict[datetime, list[EnrichedArticle]]:
    """
    Group articles into fixed-size time buckets.
    Returns a dict of bucket_start → list[article].
    """
    buckets: dict[datetime, list[EnrichedArticle]] = defaultdict(list)
    bucket_secs = int(bucket_hours * 3600)

    for art in articles:
        ts = art.article.published_at
        # Round down to bucket boundary
        ts_epoch = int(ts.timestamp())
        bucket_epoch = (ts_epoch // bucket_secs) * bucket_secs
        bucket_ts = datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)
        buckets[bucket_ts].append(art)

    return buckets


def _bucket_prices(
    snapshots: list[MarketSnapshot],
    bucket_hours: float = 1.0,
) -> dict[str, dict[datetime, float]]:
    """
    Group price snapshots into time buckets per ticker.
    Returns {ticker: {bucket_ts: price}}.
    """
    result: dict[str, dict[datetime, float]] = defaultdict(dict)
    bucket_secs = int(bucket_hours * 3600)

    for snap in snapshots:
        ts_epoch = int(snap.timestamp.timestamp())
        bucket_epoch = (ts_epoch // bucket_secs) * bucket_secs
        bucket_ts = datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)
        result[snap.ticker][bucket_ts] = snap.price

    return result


# ---------------------------------------------------------------------------
# Core correlator
# ---------------------------------------------------------------------------
class SentimentPriceCorrelator:
    """
    Correlates news sentiment and AI-detection scores with asset price movements.

    Maintains a rolling window of articles and price snapshots,
    computing correlations on each update cycle.

    Usage
    -----
    >>> correlator = SentimentPriceCorrelator()
    >>> correlator.add_articles(enriched_articles)
    >>> correlator.add_prices(market_snapshots)
    >>> results = correlator.compute_correlations(ticker="TSLA")
    >>> print(f"Pearson r: {results.pearson_r:.3f}")
    >>> print(f"Manipulation score: {correlator.manipulation_score('TSLA').score:.3f}")
    """

    def __init__(
        self,
        window_hours: float = 24.0,
        bucket_hours: float = 1.0,
    ):
        self.window_hours = window_hours
        self.bucket_hours = bucket_hours

        self._articles: list[EnrichedArticle] = []
        self._snapshots: list[MarketSnapshot] = []
        self._max_articles = 5000
        self._max_snapshots = 10000

    # ── Data ingestion ────────────────────────────────────────────────────
    def add_articles(self, articles: list[EnrichedArticle]) -> None:
        """Add enriched articles to the rolling window."""
        self._articles.extend(articles)
        # Keep only recent articles within window
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.window_hours)
        self._articles = [
            a for a in self._articles
            if a.article.published_at >= cutoff
        ][-self._max_articles:]

    def add_prices(self, snapshots: list[MarketSnapshot]) -> None:
        """Add market price snapshots to the rolling window."""
        self._snapshots.extend(snapshots)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.window_hours)
        self._snapshots = [
            s for s in self._snapshots
            if s.timestamp >= cutoff
        ][-self._max_snapshots:]

    # ── Correlation analysis ──────────────────────────────────────────────
    def compute_correlations(
        self,
        ticker: str,
        entity_filter: bool = True,
    ) -> CorrelationResult:
        """
        Compute sentiment ↔ price correlation for one ticker.

        If entity_filter=True, only uses articles that mention the ticker.
        """
        # Filter articles mentioning this ticker
        if entity_filter:
            ticker_upper = ticker.upper()
            relevant = [
                a for a in self._articles
                if ticker_upper in [e.upper() for e in a.named_entities]
                or ticker_upper in a.article.title.upper()
            ]
        else:
            relevant = self._articles

        # Filter price snapshots for this ticker
        ticker_prices = [
            s for s in self._snapshots
            if s.ticker.upper() == ticker.upper()
        ]

        if len(relevant) < 3 or len(ticker_prices) < 3:
            return CorrelationResult(
                ticker=ticker, pearson_r=0.0, p_value=1.0, n_samples=0,
                lag_hours=0, direction="neutral",
                sentiment_mean=0.0, return_mean=0.0,
            )

        # Bucket both series by the same time intervals
        art_buckets = _bucket_articles(relevant, self.bucket_hours)
        price_buckets = _bucket_prices(ticker_prices, self.bucket_hours)
        ticker_price_map = price_buckets.get(ticker.upper(), {})

        # Build aligned time series
        common_times = sorted(
            set(art_buckets.keys()) & set(ticker_price_map.keys())
        )

        if len(common_times) < 3:
            return CorrelationResult(
                ticker=ticker, pearson_r=0.0, p_value=1.0, n_samples=0,
                lag_hours=0, direction="neutral",
                sentiment_mean=0.0, return_mean=0.0,
            )

        sentiment_series: list[float] = []
        return_series: list[float] = []
        prices_sorted = sorted(ticker_price_map.items())

        for t in common_times:
            arts = art_buckets[t]
            # Mean sentiment score for this bucket
            sent_scores = [_sentiment_to_score(a) for a in arts]
            sentiment_series.append(float(np.mean(sent_scores)))

        # Compute log returns for the bucketed prices
        prices_list = [ticker_price_map[t] for t in sorted(ticker_price_map.keys())]
        if len(prices_list) < 2:
            return CorrelationResult(
                ticker=ticker, pearson_r=0.0, p_value=1.0, n_samples=0,
                lag_hours=0, direction="neutral",
                sentiment_mean=0.0, return_mean=0.0,
            )

        log_returns = [
            math.log(prices_list[i+1] / prices_list[i])
            for i in range(len(prices_list) - 1)
            if prices_list[i] > 0
        ]

        # Align lengths
        min_len = min(len(sentiment_series), len(log_returns))
        sentiment_series = sentiment_series[:min_len]
        log_returns      = log_returns[:min_len]

        r, p = _pearson_r(sentiment_series, log_returns)

        # Lead-lag
        best_lag, _ = _compute_lag_correlation(sentiment_series, log_returns)

        # Direction classification
        s_mean = float(np.mean(sentiment_series)) if sentiment_series else 0.0
        r_mean = float(np.mean(log_returns)) if log_returns else 0.0

        if s_mean > 0.1 and r_mean < -0.001:
            direction = "contradictory"   # Positive talk, falling price
        elif s_mean < -0.1 and r_mean > 0.001:
            direction = "contradictory"   # Negative talk, rising price
        elif abs(r) > 0.3:
            direction = "consistent"
        else:
            direction = "neutral"

        return CorrelationResult(
            ticker=ticker,
            pearson_r=r,
            p_value=p,
            n_samples=min_len,
            lag_hours=best_lag,
            direction=direction,
            sentiment_mean=round(s_mean, 4),
            return_mean=round(r_mean, 6),
        )

    def manipulation_score(
        self,
        ticker: str,
        volatility_alerts: list[VolatilityAlert] | None = None,
    ) -> ManipulationCorrelationScore:
        """
        Compute a 0–1 manipulation correlation score for one ticker.

        Combines:
          1. Sentiment contradiction (positive talk + falling price)
          2. AI probability → |return| correlation
          3. Lead-lag: sentiment leads price by 1+ hours
        """
        corr = self.compute_correlations(ticker)

        # ── Signal 1: Sentiment contradiction ─────────────────────────────
        if corr.is_contradictory:
            sentiment_contradiction = min(1.0, abs(corr.pearson_r) * 2)
        else:
            sentiment_contradiction = 0.0

        # ── Signal 2: AI probability vs |returns| correlation ──────────────
        ticker_upper = ticker.upper()
        relevant_arts = [
            a for a in self._articles
            if ticker_upper in [e.upper() for e in a.named_entities]
        ]
        ticker_prices = [
            s for s in self._snapshots
            if s.ticker.upper() == ticker_upper
        ]

        ai_price_r = 0.0
        if len(relevant_arts) >= 3 and len(ticker_prices) >= 3:
            art_buckets = _bucket_articles(relevant_arts, self.bucket_hours)
            price_buckets = _bucket_prices(ticker_prices, self.bucket_hours)
            ticker_map = price_buckets.get(ticker_upper, {})
            common = sorted(set(art_buckets.keys()) & set(ticker_map.keys()))

            if len(common) >= 3:
                ai_probs = [
                    float(np.mean([
                        a.ai_detection.ai_probability
                        for a in art_buckets[t]
                        if a.ai_detection
                    ]) if art_buckets[t] else 0.0)
                    for t in common
                ]
                prices_seq = [ticker_map[t] for t in sorted(ticker_map.keys())]
                abs_returns = [
                    abs(math.log(prices_seq[i+1] / prices_seq[i]))
                    for i in range(len(prices_seq) - 1)
                    if prices_seq[i] > 0
                ]
                min_len = min(len(ai_probs), len(abs_returns))
                if min_len >= 3:
                    ai_price_r, _ = _pearson_r(ai_probs[:min_len], abs_returns[:min_len])
                    ai_price_r = max(0.0, ai_price_r)  # Only positive correlation is suspicious

        # ── Signal 3: Lead-lag (content leads price) ───────────────────────
        lead_lag_score = min(1.0, corr.lag_hours / 3.0) if corr.lag_hours > 0 else 0.0

        # ── Ensemble ───────────────────────────────────────────────────────
        score = (
            0.40 * sentiment_contradiction
            + 0.35 * min(1.0, ai_price_r * 2)
            + 0.25 * lead_lag_score
        )
        score = round(float(np.clip(score, 0.0, 1.0)), 4)

        # ── Evidence summary ───────────────────────────────────────────────
        parts = []
        if sentiment_contradiction > 0.3:
            parts.append(
                f"positive sentiment ({corr.sentiment_mean:+.2f}) "
                f"with negative returns ({corr.return_mean:.4f})"
            )
        if ai_price_r > 0.3:
            parts.append(f"AI content correlated with price volatility (r={ai_price_r:.2f})")
        if corr.lag_hours > 0:
            parts.append(f"content leads price by {corr.lag_hours}h")

        evidence = "; ".join(parts) if parts else "No significant correlation detected"

        return ManipulationCorrelationScore(
            ticker=ticker,
            score=score,
            ai_price_correlation=round(ai_price_r, 4),
            sentiment_contradiction_score=round(sentiment_contradiction, 4),
            lead_lag_score=round(lead_lag_score, 4),
            evidence_summary=evidence,
        )

    # ── Time series for dashboard ─────────────────────────────────────────
    def build_time_series(
        self,
        ticker: str,
        hours: float = 24.0,
    ) -> list[TimeSeriesPoint]:
        """
        Build an aligned sentiment/price time series for dashboard charts.
        Returns list of TimeSeriesPoint sorted by timestamp.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_arts = [a for a in self._articles if a.article.published_at >= cutoff]
        recent_prices = [
            s for s in self._snapshots
            if s.ticker.upper() == ticker.upper() and s.timestamp >= cutoff
        ]

        art_buckets = _bucket_articles(recent_arts, self.bucket_hours)
        price_buckets = _bucket_prices(recent_prices, self.bucket_hours)
        ticker_map = price_buckets.get(ticker.upper(), {})

        all_times = sorted(set(art_buckets.keys()) | set(ticker_map.keys()))
        points: list[TimeSeriesPoint] = []
        prev_price: Optional[float] = None

        for t in all_times:
            arts = art_buckets.get(t, [])
            price = ticker_map.get(t)

            sent_score = float(np.mean([_sentiment_to_score(a) for a in arts])) if arts else 0.0
            ai_prob    = float(np.mean([
                a.ai_detection.ai_probability for a in arts if a.ai_detection
            ])) if arts else 0.0

            price_return = None
            if price and prev_price and prev_price > 0:
                price_return = math.log(price / prev_price)

            points.append(TimeSeriesPoint(
                timestamp=t,
                sentiment_score=round(sent_score, 4),
                ai_probability=round(ai_prob, 4),
                article_count=len(arts),
                price=price,
                price_return=round(price_return, 6) if price_return else None,
                volume=next(
                    (s.volume for s in recent_prices if
                     abs((s.timestamp - t).total_seconds()) < 3600), None
                ),
            ))
            if price:
                prev_price = price

        return points


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from datetime import timedelta
    from models import (
        ContentSource, RawArticle, EnrichedArticle, PropagationMetrics,
        SentimentResult, Sentiment, AIDetectionResult,
    )

    def _make_art(title, body, sentiment_label, ai_prob, hours_ago, entities):
        from models import Sentiment as S
        raw = RawArticle(
            source=ContentSource.RSS, source_name="Test",
            title=title, body=body,
            published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
            raw_metadata={},
        )
        sent_map = {S.POSITIVE: 0.85, S.NEGATIVE: 0.80, S.NEUTRAL: 0.6}
        score = sent_map[sentiment_label]
        return EnrichedArticle(
            article=raw,
            sentiment=SentimentResult(
                label=sentiment_label, score=score,
                positive=score if sentiment_label == S.POSITIVE else 0.1,
                negative=score if sentiment_label == S.NEGATIVE else 0.1,
                neutral=0.1,
            ),
            ai_detection=AIDetectionResult(
                ai_probability=ai_prob, is_ai_generated=ai_prob > 0.7,
                perplexity_score=22.0, burstiness_score=0.1),
            propagation=PropagationMetrics(),
            named_entities=entities,
        )

    ARTICLES = [
        _make_art("Tesla will DEFINITELY reach $1000!", "Buy TSLA now guaranteed returns!",
                  Sentiment.POSITIVE, 0.92, 5.0, ["TSLA", "Tesla"]),
        _make_art("TSLA to the moon, short sellers will be destroyed",
                  "Diamond hands, squeeze incoming on Tesla.", Sentiment.POSITIVE, 0.88, 4.0, ["TSLA"]),
        _make_art("Tesla stock surging 50% imminent", "AI model predicts TSLA moon.",
                  Sentiment.POSITIVE, 0.85, 3.0, ["TSLA", "Tesla"]),
    ]

    # Simulate falling TSLA prices during positive narrative → contradiction
    now = datetime.now(timezone.utc)
    PRICES = [
        MarketSnapshot(ticker="TSLA", asset_type="stock",
                       price=250 - i * 3.5, volume=1e7,
                       price_change_pct_24h=-8.0,
                       timestamp=now - timedelta(hours=6 - i))
        for i in range(6)
    ]

    correlator = SentimentPriceCorrelator()
    correlator.add_articles(ARTICLES)
    correlator.add_prices(PRICES)

    corr = correlator.compute_correlations("TSLA")
    manip = correlator.manipulation_score("TSLA")

    print("\n📈 Sentiment ↔ Price Correlation Demo\n")
    print(f"Ticker:                TSLA")
    print(f"Pearson r:             {corr.pearson_r:.3f}")
    print(f"Direction:             {corr.direction}")
    print(f"Sentiment mean:        {corr.sentiment_mean:+.3f}  (positive = bullish articles)")
    print(f"Return mean:           {corr.return_mean:.5f}   (negative = falling price)")
    print(f"Lead lag:              {corr.lag_hours}h")
    print(f"\nManipulation score:    {manip.score:.3f}")
    print(f"AI↔price correlation:  {manip.ai_price_correlation:.3f}")
    print(f"Contradiction score:   {manip.sentiment_contradiction_score:.3f}")
    print(f"Evidence: {manip.evidence_summary}")
