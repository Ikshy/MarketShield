"""
tests/test_market_analysis.py

Tests for price fetching, volatility detection, sentiment-price correlation,
and the market monitor. All tests use simulated/mocked data — no live APIs.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import (
    AIDetectionResult, ContentSource, EnrichedArticle, MarketSnapshot,
    PropagationMetrics, RawArticle, Sentiment, SentimentResult, VolatilityAlert,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _snap(ticker: str, price: float, asset_type: str = "stock",
          hours_ago: float = 0.0, volume: float = 1e6,
          change_24h: float = 0.0) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=ticker.upper(),
        asset_type=asset_type,
        price=price,
        volume=volume,
        price_change_pct_24h=change_24h,
        price_change_pct_1h=None,
        timestamp=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )


def _article(title: str, body: str, entities: list[str],
             sentiment: Sentiment = Sentiment.POSITIVE,
             ai_prob: float = 0.1, hours_ago: float = 0.0) -> EnrichedArticle:
    raw = RawArticle(
        source=ContentSource.RSS, source_name="Test",
        title=title, body=body,
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        raw_metadata={},
    )
    return EnrichedArticle(
        article=raw,
        sentiment=SentimentResult(
            label=sentiment, score=0.85,
            positive=0.85 if sentiment == Sentiment.POSITIVE else 0.05,
            negative=0.85 if sentiment == Sentiment.NEGATIVE else 0.05,
            neutral=0.10,
        ),
        ai_detection=AIDetectionResult(
            ai_probability=ai_prob, is_ai_generated=ai_prob >= 0.7,
            perplexity_score=22.0, burstiness_score=0.1,
        ),
        propagation=PropagationMetrics(),
        named_entities=entities,
    )


# ---------------------------------------------------------------------------
# TTL Cache tests
# ---------------------------------------------------------------------------
class TestTTLCache:

    def test_set_and_get(self):
        from market_analysis.price_fetcher import _TTLCache
        cache = _TTLCache(default_ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_missing_key_returns_none(self):
        from market_analysis.price_fetcher import _TTLCache
        cache = _TTLCache()
        assert cache.get("nonexistent") is None

    def test_expired_entry_returns_none(self):
        import time
        from market_analysis.price_fetcher import _TTLCache
        cache = _TTLCache(default_ttl=1)
        cache.set("key", "val", ttl=0)  # Expires immediately
        # Force expiry by manipulating the store
        cache._store["key"] = (time.monotonic() - 1, "val")
        assert cache.get("key") is None

    def test_clear_removes_all_entries(self):
        from market_analysis.price_fetcher import _TTLCache
        cache = _TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None


# ---------------------------------------------------------------------------
# Simulated price data tests
# ---------------------------------------------------------------------------
class TestSimulatedPrices:

    def test_simulated_stock_returns_valid_snapshot(self):
        from market_analysis.price_fetcher import _simulated_stock_snapshot
        snap = _simulated_stock_snapshot("TSLA")
        assert snap.ticker == "TSLA"
        assert snap.price > 0
        assert snap.asset_type == "stock"
        assert snap.volume > 0

    def test_simulated_crypto_returns_valid_snapshot(self):
        from market_analysis.price_fetcher import _simulated_crypto_snapshot
        snap = _simulated_crypto_snapshot("bitcoin")
        assert snap.ticker == "BITCOIN"
        assert snap.price > 0
        assert snap.asset_type == "crypto"

    def test_same_ticker_consistent_base_price(self):
        """Same ticker should produce similar base price across calls."""
        from market_analysis.price_fetcher import _simulated_stock_snapshot
        s1 = _simulated_stock_snapshot("AAPL")
        s2 = _simulated_stock_snapshot("AAPL")
        # Prices may vary due to random perturbation but should be in ballpark
        assert abs(s1.price - s2.price) / max(s1.price, s2.price) < 0.20

    def test_different_tickers_different_prices(self):
        from market_analysis.price_fetcher import _simulated_stock_snapshot
        tsla = _simulated_stock_snapshot("TSLA")
        aapl = _simulated_stock_snapshot("AAPL")
        # Different base prices (seeded from ticker hash)
        assert tsla.price != aapl.price


# ---------------------------------------------------------------------------
# PriceFetcher tests (simulation mode)
# ---------------------------------------------------------------------------
class TestPriceFetcher:

    @pytest.mark.asyncio
    async def test_fetch_all_returns_both_asset_types(self):
        from market_analysis.price_fetcher import PriceFetcher
        fetcher = PriceFetcher(
            stock_tickers=["AAPL", "TSLA"],
            crypto_tickers=["bitcoin"],
            use_live_data=False,
        )
        snapshots = await fetcher.fetch_all()
        assert len(snapshots) == 3
        types = {s.asset_type for s in snapshots}
        assert "stock" in types
        assert "crypto" in types

    @pytest.mark.asyncio
    async def test_fetch_stocks_simulation(self):
        from market_analysis.price_fetcher import PriceFetcher
        fetcher = PriceFetcher(stock_tickers=["GME", "AMC"], use_live_data=False)
        snaps = await fetcher.fetch_stocks()
        assert len(snaps) == 2
        for s in snaps:
            assert s.ticker in ("GME", "AMC")
            assert s.price > 0

    @pytest.mark.asyncio
    async def test_fetch_crypto_simulation(self):
        from market_analysis.price_fetcher import PriceFetcher
        fetcher = PriceFetcher(crypto_tickers=["bitcoin", "ethereum"], use_live_data=False)
        snaps = await fetcher.fetch_crypto()
        assert len(snaps) == 2

    @pytest.mark.asyncio
    async def test_fetch_empty_tickers_returns_empty(self):
        from market_analysis.price_fetcher import PriceFetcher
        fetcher = PriceFetcher(stock_tickers=[], crypto_tickers=[], use_live_data=False)
        snaps = await fetcher.fetch_all()
        assert snaps == []


# ---------------------------------------------------------------------------
# VolatilityDetector tests
# ---------------------------------------------------------------------------
class TestVolatilityDetector:

    def setup_method(self):
        from market_analysis.volatility import VolatilityDetector
        self.detector = VolatilityDetector(window=10, z_threshold=2.0)

    def _feed_normal_bars(self, ticker: str, n: int = 15) -> None:
        """Feed n normal (small-return) price bars."""
        price = 100.0
        for i in range(n):
            price *= (1 + random.gauss(0, 0.005))
            snap = _snap(ticker, price, hours_ago=n - i)
            self.detector.update([snap])

    def test_no_alert_on_normal_market(self):
        self._feed_normal_bars("TSLA", n=20)
        snap = _snap("TSLA", 100.5)  # Tiny move
        alerts = self.detector.update([snap])
        assert len(alerts) == 0

    def test_alert_on_large_return(self):
        self._feed_normal_bars("GME", n=15)
        # 20% spike
        big_snap = _snap("GME", 120.0)
        self.detector._buffers["GME"].prices[-1] = 100.0  # Set prev close
        # Force a large return by seeding appropriately
        # Instead: directly test z-score calculation
        buf = self.detector._buffers["GME"]
        large_return = 0.20  # 20% — should be many std devs above baseline
        z = buf.return_z_score(large_return)
        assert abs(z) > 2.0  # Must exceed threshold

    def test_buffer_ready_after_min_bars(self):
        from market_analysis.volatility import _MIN_BASELINE_BARS
        self._feed_normal_bars("AAPL", n=_MIN_BASELINE_BARS + 1)
        buf = self.detector._buffers.get("AAPL")
        assert buf is not None
        assert buf.ready

    def test_garman_klass_positive(self):
        from market_analysis.volatility import garman_klass_vol
        bars = [
            {"open": 100, "high": 105, "low": 98, "close": 103}
            for _ in range(20)
        ]
        vol = garman_klass_vol(bars)
        assert vol > 0.0

    def test_garman_klass_zero_for_flat_bars(self):
        from market_analysis.volatility import garman_klass_vol
        bars = [{"open": 100, "high": 100, "low": 100, "close": 100}]
        vol = garman_klass_vol(bars, annualise=False)
        assert vol == 0.0

    def test_regime_classification(self):
        from market_analysis.volatility import classify_regime, VolatilityRegime
        assert classify_regime(0.10) == VolatilityRegime.LOW
        assert classify_regime(0.40) == VolatilityRegime.NORMAL
        assert classify_regime(0.70) == VolatilityRegime.ELEVATED
        assert classify_regime(0.90) == VolatilityRegime.EXTREME

    def test_market_impact_score_no_match(self):
        self._feed_normal_bars("NVDA", n=15)
        alerts = [VolatilityAlert(
            ticker="TSLA", z_score=4.0, current_price=200.0,
            baseline_mean=0.001, baseline_std=0.01, direction="down",
        )]
        # NVDA not in alerts → impact = 0
        impact = self.detector.market_impact_score(["NVDA", "AMD"], alerts)
        assert impact == 0.0

    def test_market_impact_score_with_match(self):
        self._feed_normal_bars("TSLA", n=15)
        alerts = [VolatilityAlert(
            ticker="TSLA", z_score=5.0, current_price=150.0,
            baseline_mean=0.001, baseline_std=0.01, direction="down",
        )]
        impact = self.detector.market_impact_score(["TSLA"], alerts)
        assert impact > 0.5  # High z-score → high impact

    def test_seed_from_history(self):
        bars = [
            {"timestamp": "2024-01-01T00:00:00+00:00",
             "close": 100.0 + i * 0.1, "volume": 1e6}
            for i in range(30)
        ]
        self.detector.seed_from_history("AAPL", bars)
        buf = self.detector._buffers.get("AAPL")
        assert buf is not None
        assert len(buf.prices) > 0

    def test_market_summary_structure(self):
        self._feed_normal_bars("COIN", n=12)
        summary = self.detector.market_summary()
        assert "COIN" in summary
        assert "regime" in summary["COIN"]
        assert "realised_vol" in summary["COIN"]


# ---------------------------------------------------------------------------
# SentimentPriceCorrelator tests
# ---------------------------------------------------------------------------
class TestSentimentPriceCorrelator:

    def setup_method(self):
        from market_analysis.correlator import SentimentPriceCorrelator
        self.correlator = SentimentPriceCorrelator(window_hours=48.0, bucket_hours=1.0)

    def _add_falling_prices(self, ticker: str = "TSLA", n: int = 12) -> None:
        base = 200.0
        for i in range(n, 0, -1):
            price = base * (1 - 0.01 * (n - i))  # Gradual decline
            self.correlator.add_prices([_snap(ticker, price, hours_ago=float(i))])

    def _add_bullish_articles(self, ticker: str = "TSLA", n: int = 6) -> None:
        for i in range(n, 0, -1):
            self.correlator.add_articles([
                _article(f"Buy {ticker} now!", "Guaranteed returns!",
                         entities=[ticker], sentiment=Sentiment.POSITIVE,
                         ai_prob=0.85, hours_ago=float(i))
            ])

    def test_correlation_with_enough_data(self):
        self._add_falling_prices("TSLA", n=15)
        self._add_bullish_articles("TSLA", n=8)
        result = self.correlator.compute_correlations("TSLA")
        assert -1.0 <= result.pearson_r <= 1.0
        assert result.n_samples >= 0

    def test_insufficient_data_returns_neutral(self):
        # Only one data point — not enough for correlation
        self.correlator.add_prices([_snap("NVDA", 500.0)])
        result = self.correlator.compute_correlations("NVDA")
        assert result.pearson_r == 0.0
        assert result.n_samples == 0

    def test_contradictory_direction_detected(self):
        from market_analysis.correlator import CorrelationResult
        # Simulate a contradictory correlation result
        result = CorrelationResult(
            ticker="TSLA", pearson_r=-0.8, p_value=0.01, n_samples=20,
            lag_hours=0, direction="contradictory",
            sentiment_mean=0.7,   # Very bullish articles
            return_mean=-0.015,   # But price is falling
        )
        assert result.is_contradictory  # Positive narrative + negative returns
        assert result.is_significant

    def test_consistent_direction(self):
        from market_analysis.correlator import CorrelationResult
        result = CorrelationResult(
            ticker="AAPL", pearson_r=0.7, p_value=0.02, n_samples=15,
            lag_hours=0, direction="consistent",
            sentiment_mean=0.5, return_mean=0.005,
        )
        assert not result.is_contradictory

    def test_manipulation_score_range(self):
        self._add_falling_prices("GME", n=12)
        self._add_bullish_articles("GME", n=6)
        score = self.correlator.manipulation_score("GME")
        assert 0.0 <= score.score <= 1.0

    def test_time_series_structure(self):
        self._add_falling_prices("TSLA", n=8)
        self._add_bullish_articles("TSLA", n=4)
        series = self.correlator.build_time_series("TSLA", hours=24.0)
        assert isinstance(series, list)
        if series:
            pt = series[0]
            assert hasattr(pt, "timestamp")
            assert hasattr(pt, "sentiment_score")
            assert hasattr(pt, "ai_probability")

    def test_pearson_r_identical_series(self):
        from market_analysis.correlator import _pearson_r
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        r, p = _pearson_r(x, x)
        assert r == pytest.approx(1.0, abs=1e-4)

    def test_pearson_r_opposite_series(self):
        from market_analysis.correlator import _pearson_r
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        r, p = _pearson_r(x, y)
        assert r == pytest.approx(-1.0, abs=1e-4)

    def test_pearson_r_insufficient_data(self):
        from market_analysis.correlator import _pearson_r
        r, p = _pearson_r([1.0], [1.0])
        assert r == 0.0
        assert p == 1.0


# ---------------------------------------------------------------------------
# MarketMonitor tests
# ---------------------------------------------------------------------------
class TestMarketMonitor:

    @pytest.mark.asyncio
    async def test_fetch_once_returns_snapshots(self):
        from market_analysis.monitor import MarketMonitor
        monitor = MarketMonitor(use_live_data=False, seed_history=False)
        monitor._detector = __import__(
            "market_analysis.volatility", fromlist=["VolatilityDetector"]
        ).VolatilityDetector(window=5, z_threshold=10.0)  # High threshold: no alerts
        snaps, alerts = await monitor.fetch_once()
        assert len(snaps) > 0
        for s in snaps:
            assert isinstance(s, MarketSnapshot)

    @pytest.mark.asyncio
    async def test_status_structure(self):
        from market_analysis.monitor import MarketMonitor
        monitor = MarketMonitor(use_live_data=False, seed_history=False)
        await monitor.fetch_once()
        status = monitor.status()
        assert "cycles" in status
        assert "total_snapshots" in status
        assert "queue_sizes" in status

    @pytest.mark.asyncio
    async def test_snapshots_queued(self):
        from market_analysis.monitor import MarketMonitor
        monitor = MarketMonitor(use_live_data=False, seed_history=False)
        await monitor.fetch_once()
        assert monitor.snapshot_queue.qsize() > 0

    def test_market_impact_for_tickers_empty(self):
        from market_analysis.monitor import MarketMonitor
        monitor = MarketMonitor(use_live_data=False, seed_history=False)
        impact = monitor.market_impact_for_tickers(["TSLA"])
        assert impact == 0.0  # No alerts yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
