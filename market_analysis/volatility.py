"""
market_analysis/volatility.py — Rolling volatility and anomaly detector.

Detects abnormal price movements that could be triggered by — or
correlated with — coordinated misinformation campaigns.

Methods implemented:

  1. Rolling Z-score
     z = (current_return - μ_baseline) / σ_baseline
     |z| > threshold → price spike / crash alert

  2. Garman-Klass volatility estimator (OHLCV-based)
     More efficient than close-to-close; uses full OHLCV bar.
     GK = 0.5 * ln(H/L)² - (2ln2-1) * ln(C/O)²

  3. Volume anomaly detection
     Unusual volume spikes often precede price manipulation.
     Same rolling z-score applied to volume.

  4. Volatility regime classifier
     LOW / NORMAL / ELEVATED / EXTREME based on rolling percentile
     of realised volatility vs the asset's own historical distribution.

All detectors return VolatilityAlert objects consumed by:
  - market_analysis/correlator.py  (for sentiment correlation)
  - risk_engine/scorer.py           (for market impact component)
  - dashboard/app.py                (for live alert display)
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

from config import settings
from logger import get_logger
from models import MarketSnapshot, VolatilityAlert

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------
class VolatilityRegime(str, Enum):
    LOW      = "low"
    NORMAL   = "normal"
    ELEVATED = "elevated"
    EXTREME  = "extreme"


# Minimum bars required before we trust the rolling statistics
_MIN_BASELINE_BARS = 10


# ---------------------------------------------------------------------------
# Rolling statistics buffer (per ticker)
# ---------------------------------------------------------------------------
@dataclass
class _RollingBuffer:
    """
    Circular buffer of price/volume observations for one ticker.
    Maintains enough history for baseline statistics.
    """
    ticker: str
    window: int = 60          # Number of bars in the baseline window
    prices: deque = field(default_factory=deque)
    volumes: deque = field(default_factory=deque)
    returns: deque = field(default_factory=deque)    # log returns
    timestamps: deque = field(default_factory=deque)

    def __post_init__(self):
        self.prices    = deque(maxlen=self.window)
        self.volumes   = deque(maxlen=self.window)
        self.returns   = deque(maxlen=self.window)
        self.timestamps = deque(maxlen=self.window)

    def push(self, snapshot: MarketSnapshot) -> Optional[float]:
        """
        Add a new snapshot. Returns the log return vs previous close,
        or None if this is the first observation.
        """
        log_return = None
        if self.prices:
            prev = self.prices[-1]
            if prev > 0:
                log_return = math.log(snapshot.price / prev)
                self.returns.append(log_return)

        self.prices.append(snapshot.price)
        self.volumes.append(snapshot.volume)
        self.timestamps.append(snapshot.timestamp)
        return log_return

    @property
    def ready(self) -> bool:
        return len(self.returns) >= _MIN_BASELINE_BARS

    # ── Return statistics ─────────────────────────────────────────────────
    @property
    def return_mean(self) -> float:
        return float(np.mean(self.returns)) if self.returns else 0.0

    @property
    def return_std(self) -> float:
        if len(self.returns) < 2:
            return 1e-8
        return float(np.std(self.returns)) + 1e-8  # Avoid zero division

    @property
    def realised_vol(self) -> float:
        """Annualised realised volatility (assuming hourly bars)."""
        if len(self.returns) < 2:
            return 0.0
        return float(np.std(self.returns)) * math.sqrt(8760)  # 8760 hours/year

    # ── Volume statistics ─────────────────────────────────────────────────
    @property
    def volume_mean(self) -> float:
        return float(np.mean(self.volumes)) if self.volumes else 0.0

    @property
    def volume_std(self) -> float:
        if len(self.volumes) < 2:
            return 1e-8
        return float(np.std(self.volumes)) + 1e-8

    # ── Return z-score ────────────────────────────────────────────────────
    def return_z_score(self, current_return: float) -> float:
        if not self.ready:
            return 0.0
        return (current_return - self.return_mean) / self.return_std

    def volume_z_score(self, current_volume: float) -> float:
        if not self.ready:
            return 0.0
        return (current_volume - self.volume_mean) / self.volume_std

    # ── Percentile rank of current vol vs history ─────────────────────────
    def vol_percentile(self, current_vol: float) -> float:
        """0–1: what fraction of historical windows had lower volatility."""
        if len(self.returns) < 5:
            return 0.5
        hist_vols = [
            abs(r) for r in list(self.returns)[:-1]
        ]
        if not hist_vols:
            return 0.5
        below = sum(1 for v in hist_vols if v < abs(current_vol))
        return below / len(hist_vols)


# ---------------------------------------------------------------------------
# Garman-Klass volatility estimator
# ---------------------------------------------------------------------------
def garman_klass_vol(ohlcv_bars: list[dict], annualise: bool = True) -> float:
    """
    Garman-Klass volatility estimator using OHLCV bars.

    More efficient than close-to-close (uses full bar range).
    σ² = 0.5 * [ln(H/L)]² - (2ln2 - 1) * [ln(C/O)]²

    Parameters
    ----------
    ohlcv_bars : List of dicts with keys open, high, low, close
    annualise  : If True, annualise assuming hourly bars

    Returns annualised volatility (or raw if annualise=False).
    """
    if len(ohlcv_bars) < 2:
        return 0.0

    gk_variances = []
    for bar in ohlcv_bars:
        o, h, l, c = (
            bar.get("open",  1), bar.get("high",  1),
            bar.get("low",   1), bar.get("close", 1),
        )
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            continue
        try:
            hl_term = 0.5 * (math.log(h / l)) ** 2
            co_term = (2 * math.log(2) - 1) * (math.log(c / o)) ** 2
            gk_variances.append(hl_term - co_term)
        except (ZeroDivisionError, ValueError):
            continue

    if not gk_variances:
        return 0.0

    gk_vol = math.sqrt(max(0.0, float(np.mean(gk_variances))))
    if annualise:
        gk_vol *= math.sqrt(8760)  # Assuming hourly bars
    return round(gk_vol, 6)


# ---------------------------------------------------------------------------
# Volatility regime classifier
# ---------------------------------------------------------------------------
def classify_regime(vol_percentile: float) -> VolatilityRegime:
    """
    Classify the current volatility regime based on historical percentile.

    Thresholds:
      0.00–0.25 → LOW      (quieter than 75% of history)
      0.25–0.60 → NORMAL
      0.60–0.85 → ELEVATED
      0.85–1.00 → EXTREME  (more volatile than 85% of history)
    """
    if vol_percentile >= 0.85:
        return VolatilityRegime.EXTREME
    elif vol_percentile >= 0.60:
        return VolatilityRegime.ELEVATED
    elif vol_percentile >= 0.25:
        return VolatilityRegime.NORMAL
    else:
        return VolatilityRegime.LOW


# ---------------------------------------------------------------------------
# Core volatility detector
# ---------------------------------------------------------------------------
class VolatilityDetector:
    """
    Real-time rolling volatility detector for multiple assets.

    Maintains a per-ticker rolling buffer and detects:
      - Return z-score spikes (price anomalies)
      - Volume z-score spikes
      - Volatility regime transitions

    Usage
    -----
    >>> detector = VolatilityDetector()
    >>> alerts = detector.update(snapshots)    # Feed live data
    >>> for alert in alerts:
    ...     print(f"{alert.ticker}: z={alert.z_score:.2f} {alert.direction}")

    The detector is stateful — call update() on each polling cycle.
    State persists across calls (rolling baseline accumulates over time).
    """

    def __init__(
        self,
        window: int = 60,
        z_threshold: float | None = None,
        volume_z_threshold: float = 3.0,
    ):
        """
        Parameters
        ----------
        window           : Number of historical bars for baseline (default: 60)
        z_threshold      : Return z-score threshold for alerts
        volume_z_threshold: Volume z-score threshold
        """
        self.window = window
        self.z_threshold = z_threshold or settings.volatility_z_score_threshold
        self.vol_z_threshold = volume_z_threshold

        # Per-ticker state
        self._buffers: dict[str, _RollingBuffer] = {}
        self._regime: dict[str, VolatilityRegime] = {}
        self._alert_count = 0

    def _get_buffer(self, ticker: str) -> _RollingBuffer:
        if ticker not in self._buffers:
            self._buffers[ticker] = _RollingBuffer(ticker=ticker, window=self.window)
        return self._buffers[ticker]

    def update(
        self,
        snapshots: list[MarketSnapshot],
    ) -> list[VolatilityAlert]:
        """
        Process a batch of market snapshots. Returns any volatility alerts.

        Call this on each price polling cycle (e.g., every 60 seconds).
        Alerts are only generated once the buffer has enough history.
        """
        alerts: list[VolatilityAlert] = []

        for snapshot in snapshots:
            alert = self._process_snapshot(snapshot)
            if alert:
                alerts.append(alert)

        if alerts:
            log.warning(
                f"[Volatility] {len(alerts)} alert(s) triggered: "
                + ", ".join(f"{a.ticker}(z={a.z_score:.1f})" for a in alerts)
            )

        return alerts

    def _process_snapshot(self, snapshot: MarketSnapshot) -> Optional[VolatilityAlert]:
        """Process one snapshot and return an alert if threshold exceeded."""
        buf = self._get_buffer(snapshot.ticker)
        log_return = buf.push(snapshot)

        if not buf.ready or log_return is None:
            return None  # Not enough history yet

        # ── Return z-score ─────────────────────────────────────────────────
        ret_z = buf.return_z_score(log_return)

        # ── Volume z-score ─────────────────────────────────────────────────
        vol_z = buf.volume_z_score(snapshot.volume) if snapshot.volume > 0 else 0.0

        # ── Regime classification ──────────────────────────────────────────
        vol_pct = buf.vol_percentile(log_return)
        regime = classify_regime(vol_pct)
        self._regime[snapshot.ticker] = regime

        # ── Alert decision ─────────────────────────────────────────────────
        # Trigger on extreme return OR extreme volume (volume often leads price)
        if abs(ret_z) < self.z_threshold and vol_z < self.vol_z_threshold:
            return None

        direction = "up" if log_return > 0 else "down"
        effective_z = ret_z if abs(ret_z) >= self.z_threshold else vol_z

        self._alert_count += 1
        alert = VolatilityAlert(
            ticker=snapshot.ticker,
            z_score=round(effective_z, 4),
            current_price=snapshot.price,
            baseline_mean=round(buf.return_mean, 6),
            baseline_std=round(buf.return_std, 6),
            direction=direction,
            triggered_at=snapshot.timestamp,
        )

        log.warning(
            f"[Volatility] 🔔 {snapshot.ticker} spike! "
            f"z={effective_z:.2f}, price=${snapshot.price:.4f}, "
            f"regime={regime.value}, dir={direction}"
        )
        return alert

    # ── Bulk load from historical data ────────────────────────────────────
    def seed_from_history(
        self,
        ticker: str,
        ohlcv_bars: list[dict],
    ) -> None:
        """
        Pre-fill the rolling buffer from historical OHLCV data.
        Call this on startup to avoid the cold-start period.

        After seeding, the first live snapshot can immediately trigger alerts.
        """
        buf = self._get_buffer(ticker)
        for bar in ohlcv_bars[-self.window:]:
            close = bar.get("close", 0.0)
            volume = bar.get("volume", 0.0)
            ts_str = bar.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)

            snapshot = MarketSnapshot(
                ticker=ticker, asset_type="stock",
                price=close, volume=volume, timestamp=ts,
            )
            buf.push(snapshot)

        log.debug(
            f"[Volatility] Seeded {ticker} buffer with "
            f"{len(ohlcv_bars[-self.window:])} bars "
            f"(baseline ready: {buf.ready})"
        )

    # ── State inspection ──────────────────────────────────────────────────
    def get_regime(self, ticker: str) -> VolatilityRegime:
        return self._regime.get(ticker, VolatilityRegime.NORMAL)

    def get_realised_vol(self, ticker: str) -> float:
        buf = self._buffers.get(ticker)
        return buf.realised_vol if buf else 0.0

    def market_summary(self) -> dict:
        """Summary of all tracked tickers for dashboard display."""
        return {
            ticker: {
                "regime": self._regime.get(ticker, VolatilityRegime.NORMAL).value,
                "realised_vol": round(self.get_realised_vol(ticker), 4),
                "buffer_size": len(self._buffers[ticker].returns)
                               if ticker in self._buffers else 0,
            }
            for ticker in self._buffers
        }

    # ── Market impact score (used by risk engine) ─────────────────────────
    def market_impact_score(
        self,
        related_tickers: list[str],
        recent_alerts: list[VolatilityAlert],
    ) -> float:
        """
        Compute a 0–1 market impact score for a set of tickers.

        Input:
          related_tickers : Tickers mentioned in a suspicious article
          recent_alerts   : VolatilityAlerts from the last N minutes

        Logic:
          - If any mentioned ticker has an active volatility alert → high score
          - Score scales with z-score magnitude
          - Multiple simultaneous alerts add up (capped at 1.0)
        """
        if not related_tickers or not recent_alerts:
            return 0.0

        ticker_set = {t.upper() for t in related_tickers}
        matching_alerts = [
            a for a in recent_alerts
            if a.ticker.upper() in ticker_set
        ]

        if not matching_alerts:
            return 0.0

        # Use the highest z-score among matching alerts
        max_z = max(abs(a.z_score) for a in matching_alerts)

        # Normalise: z=2.5 → 0.5, z=5.0 → ~0.9
        impact = 1.0 / (1.0 + math.exp(-0.5 * (max_z - 3.0)))

        # Boost for multiple simultaneous spikes across tickers
        if len(matching_alerts) > 1:
            impact = min(1.0, impact * 1.2)

        return round(impact, 4)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from market_analysis.price_fetcher import _simulated_stock_snapshot, _simulated_crypto_snapshot
    import random

    print("\n📊 VolatilityDetector Demo\n")

    detector = VolatilityDetector(window=20, z_threshold=2.0)

    # Simulate a quiet market, then inject a manipulation-driven spike
    tickers = ["TSLA", "GME", "BITCOIN"]
    all_alerts: list[VolatilityAlert] = []

    print("Feeding 25 normal market bars...")
    for bar in range(25):
        snapshots = []
        for t in tickers:
            if t == "BITCOIN":
                s = _simulated_crypto_snapshot("bitcoin")
            else:
                s = _simulated_stock_snapshot(t)
            # Normal: small perturbation
            s.price *= (1 + random.gauss(0, 0.005))
            s.volume *= random.uniform(0.8, 1.2)
            snapshots.append(s)
        alerts = detector.update(snapshots)
        all_alerts.extend(alerts)

    # Inject an artificial price spike on GME (simulating a pump)
    print("\n💥 Injecting GME price spike (+18% in one bar)...")
    spike_snap = _simulated_stock_snapshot("GME")
    spike_snap.price *= 1.18
    spike_snap.volume *= 8.0  # Volume spike too
    alerts = detector.update([spike_snap])
    all_alerts.extend(alerts)

    print("\n📋 Market Summary:")
    for ticker, info in detector.market_summary().items():
        print(f"  {ticker:<12} regime={info['regime']:<10} "
              f"ann_vol={info['realised_vol']:.2%}  "
              f"bars={info['buffer_size']}")

    print(f"\n🚨 Total alerts triggered: {len(all_alerts)}")
    for alert in all_alerts:
        print(f"  {alert.ticker}: z={alert.z_score:.2f}, "
              f"price=${alert.current_price:.2f}, dir={alert.direction}")

    # Market impact scoring
    impact = detector.market_impact_score(["GME", "AMC"], all_alerts)
    print(f"\n💥 Market impact score for GME/AMC: {impact:.3f}")
