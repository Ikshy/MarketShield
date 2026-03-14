"""
market_analysis/monitor.py — Continuous market price monitoring loop.

Polls stock and crypto prices on a configurable interval, feeds data
to the VolatilityDetector and SentimentPriceCorrelator, and persists
snapshots to the database.

On startup:
  1. Seeds volatility baselines from stored historical data (DB)
  2. Optionally fetches 1-month OHLCV history to warm up the rolling window
  3. Starts the polling loop

On each cycle:
  1. Fetch all configured tickers
  2. Persist snapshots to DB
  3. Check for volatility alerts
  4. Update correlation models with new prices
  5. Publish alerts to the shared queue

The monitor is designed to run concurrently with the ingestion +
NLP pipelines, all feeding into the risk engine's event queue.
"""

from __future__ import annotations

import asyncio
import time
from asyncio import Queue
from datetime import datetime, timezone
from typing import Optional

from config import settings
from database import MarketRepository, init_db
from logger import get_logger, configure_logging
from models import MarketSnapshot, VolatilityAlert

from market_analysis.price_fetcher import PriceFetcher, fetch_historical_ohlcv
from market_analysis.volatility import VolatilityDetector
from market_analysis.correlator import SentimentPriceCorrelator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Market Monitor
# ---------------------------------------------------------------------------
class MarketMonitor:
    """
    Orchestrates continuous market data collection and analysis.

    Publishes to two queues:
      snapshot_queue  — all price snapshots (consumed by correlator)
      alert_queue     — volatility alerts (consumed by risk engine)

    Usage
    -----
    >>> monitor = MarketMonitor()
    >>> await monitor.run()          # Blocking loop
    >>> snapshots = await monitor.fetch_once()   # Single cycle
    """

    def __init__(
        self,
        snapshot_queue: Optional[Queue] = None,
        alert_queue: Optional[Queue] = None,
        use_live_data: bool = True,
        seed_history: bool = True,
    ):
        self.snapshot_queue: Queue[MarketSnapshot] = snapshot_queue or Queue(maxsize=10_000)
        self.alert_queue: Queue[VolatilityAlert]   = alert_queue     or Queue(maxsize=1_000)

        self._fetcher   = PriceFetcher(use_live_data=use_live_data)
        self._detector  = VolatilityDetector(
            window=60,
            z_threshold=settings.volatility_z_score_threshold,
        )
        self._correlator = SentimentPriceCorrelator(window_hours=24.0)
        self._repo       = MarketRepository()

        self._seed_history = seed_history
        self._cycle       = 0
        self._total_snaps = 0
        self._total_alerts = 0

    # ── Startup ───────────────────────────────────────────────────────────
    async def _warm_up_baselines(self) -> None:
        """
        Pre-seed the volatility detector with historical OHLCV data.
        Runs once at startup to avoid the cold-start period where
        no alerts can be generated (buffer not yet full).
        """
        all_tickers = [
            (t, "stock") for t in settings.stock_tickers
        ] + [
            (t, "crypto") for t in settings.crypto_tickers
        ]

        log.info(f"[Monitor] Warming up baselines for {len(all_tickers)} tickers…")
        start = time.monotonic()

        for ticker, asset_type in all_tickers:
            # Check DB first (fast path)
            stored = self._repo.get_ticker_history(ticker, limit=120)
            if len(stored) >= settings.volatility_z_score_threshold:
                # Convert DB rows to synthetic OHLCV-like dicts
                bars = [
                    {
                        "timestamp": r["timestamp"].isoformat()
                        if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"]),
                        "open":   r["price"], "high":  r["price"] * 1.002,
                        "low":    r["price"] * 0.998, "close": r["price"],
                        "volume": r.get("volume", 0),
                    }
                    for r in reversed(stored)
                ]
                self._detector.seed_from_history(ticker, bars)
            else:
                # Fetch from live source (async-in-thread for sync yfinance)
                loop = asyncio.get_event_loop()
                try:
                    bars = await loop.run_in_executor(
                        None,
                        fetch_historical_ohlcv, ticker, "1mo", "1h", asset_type,
                    )
                    if bars:
                        self._detector.seed_from_history(ticker, bars)
                except Exception as exc:
                    log.warning(f"[Monitor] Could not seed {ticker}: {exc}")

        elapsed = time.monotonic() - start
        log.info(f"[Monitor] Baseline warm-up complete in {elapsed:.1f}s")

    # ── Single fetch cycle ────────────────────────────────────────────────
    async def fetch_once(self) -> tuple[list[MarketSnapshot], list[VolatilityAlert]]:
        """
        Execute one complete price fetch cycle.

        Returns (snapshots, alerts).
        Persists snapshots to DB and publishes to queues.
        """
        self._cycle += 1
        start = time.monotonic()

        # Fetch all prices
        snapshots = await self._fetcher.fetch_all()
        if not snapshots:
            log.warning("[Monitor] Price fetch returned no data")
            return [], []

        # Feed to volatility detector
        alerts = self._detector.update(snapshots)

        # Feed prices to correlator
        self._correlator.add_prices(snapshots)

        # Persist to DB
        self._persist_snapshots(snapshots)

        # Publish to queues
        self._publish_to_queues(snapshots, alerts)

        elapsed = time.monotonic() - start
        self._total_snaps  += len(snapshots)
        self._total_alerts += len(alerts)

        log.info(
            f"[Monitor] Cycle {self._cycle}: "
            f"{len(snapshots)} snapshots, "
            f"{len(alerts)} alerts, "
            f"{elapsed:.2f}s"
        )
        return snapshots, alerts

    def _persist_snapshots(self, snapshots: list[MarketSnapshot]) -> None:
        """Write price snapshots to the database."""
        for snap in snapshots:
            try:
                self._repo.insert_snapshot({
                    "ticker":               snap.ticker,
                    "asset_type":           snap.asset_type,
                    "price":                snap.price,
                    "volume":               snap.volume,
                    "price_change_pct_1h":  snap.price_change_pct_1h,
                    "price_change_pct_24h": snap.price_change_pct_24h,
                    "timestamp":            snap.timestamp,
                })
            except Exception as exc:
                log.error(f"[Monitor] DB insert failed for {snap.ticker}: {exc}")

    def _publish_to_queues(
        self,
        snapshots: list[MarketSnapshot],
        alerts: list[VolatilityAlert],
    ) -> None:
        """Push snapshots and alerts to async queues for downstream consumers."""
        for snap in snapshots:
            try:
                self.snapshot_queue.put_nowait(snap)
            except asyncio.QueueFull:
                pass  # Non-critical; snapshots accumulate anyway

        for alert in alerts:
            try:
                self.alert_queue.put_nowait(alert)
                log.warning(
                    f"[Monitor] 🔔 Alert queued: {alert.ticker} "
                    f"z={alert.z_score:.2f} dir={alert.direction}"
                )
            except asyncio.QueueFull:
                log.warning("[Monitor] Alert queue full!")

    # ── Continuous loop ───────────────────────────────────────────────────
    async def run(self) -> None:
        """
        Main monitoring loop. Runs indefinitely until cancelled.
        Polls on settings.market_fetch_interval_seconds.
        """
        log.info("=" * 60)
        log.info("[Monitor] Market monitor starting")
        log.info(f"[Monitor] Poll interval: {settings.market_fetch_interval_seconds}s")
        log.info(f"[Monitor] Tickers: {settings.stock_tickers + settings.crypto_tickers}")
        log.info("=" * 60)

        if self._seed_history:
            await self._warm_up_baselines()

        try:
            while True:
                await self.fetch_once()
                await asyncio.sleep(settings.market_fetch_interval_seconds)
        except asyncio.CancelledError:
            log.info(
                f"[Monitor] Stopped after {self._cycle} cycles, "
                f"{self._total_snaps} snapshots, "
                f"{self._total_alerts} alerts"
            )

    # ── Status ────────────────────────────────────────────────────────────
    def status(self) -> dict:
        """Return monitoring status dict for logging / API."""
        return {
            "cycles": self._cycle,
            "total_snapshots": self._total_snaps,
            "total_alerts": self._total_alerts,
            "queue_sizes": {
                "snapshots": self.snapshot_queue.qsize(),
                "alerts":    self.alert_queue.qsize(),
            },
            "market_regimes": self._detector.market_summary(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public helper for risk engine ─────────────────────────────────────
    def get_recent_alerts(self, max_age_minutes: float = 30.0) -> list[VolatilityAlert]:
        """
        Drain the alert queue and return recent alerts.
        Called by the risk engine on each scoring cycle.
        """
        alerts: list[VolatilityAlert] = []
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                alerts.append(alert)
            except asyncio.QueueEmpty:
                break
        return alerts

    def market_impact_for_tickers(self, tickers: list[str]) -> float:
        """
        Convenience: return a 0-1 market impact score for a list of tickers.
        Uses the detector's method with recent alerts from the queue.
        """
        recent = self.get_recent_alerts()
        return self._detector.market_impact_score(tickers, recent)


# ---------------------------------------------------------------------------
# Sync wrapper for main.py
# ---------------------------------------------------------------------------
def run_market_monitor() -> None:
    """Blocking entry point for ``python main.py market``."""
    configure_logging(level=settings.log_level, log_to_file=True)
    init_db()
    monitor = MarketMonitor(use_live_data=True, seed_history=True)
    asyncio.run(monitor.run())


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    configure_logging(level="INFO", log_to_file=False)

    async def _demo():
        print("\n📡 MarketMonitor Demo (simulated data)\n")

        monitor = MarketMonitor(
            use_live_data=False,   # Use simulation — no API keys needed
            seed_history=False,    # Skip warm-up for quick demo
        )

        # Override detector with lower threshold for demo sensitivity
        monitor._detector = VolatilityDetector(window=5, z_threshold=1.5)

        # Run 10 cycles
        for i in range(10):
            snapshots, alerts = await monitor.fetch_once()

            if i == 0:
                print(f"{'Ticker':<12} {'Price':>10}  {'24h%':>7}  {'Regime'}")
                print("-" * 45)
                for s in snapshots:
                    regime = monitor._detector.get_regime(s.ticker)
                    chg = f"{s.price_change_pct_24h:+.2f}%" if s.price_change_pct_24h else "  N/A"
                    print(f"{s.ticker:<12} ${s.price:>9,.2f}  {chg:>7}  {regime.value}")

            if alerts:
                print(f"\n  🚨 Cycle {i+1}: {len(alerts)} alert(s):")
                for a in alerts:
                    print(f"     {a.ticker}: z={a.z_score:.2f}, ${a.current_price:.2f} {a.direction}")

            await asyncio.sleep(0.05)  # Fast demo

        print(f"\n📊 Final Status:")
        status = monitor.status()
        print(json.dumps({
            k: v for k, v in status.items()
            if k != "market_regimes"
        }, indent=2, default=str))

        print("\n🎯 Market regimes:")
        for ticker, info in status["market_regimes"].items():
            print(f"  {ticker:<12} {info['regime']:<10} vol={info['realised_vol']:.2%}")

    asyncio.run(_demo())
