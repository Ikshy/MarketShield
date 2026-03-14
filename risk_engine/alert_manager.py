"""
risk_engine/alert_manager.py — Alert lifecycle manager.

Responsibilities:
  1. Threshold gating:     Only emit alerts when composite_score ≥ threshold
  2. Deduplication:        Don't re-alert on the same article/cluster
  3. Cooldown:             Don't spam the same ticker more than once per window
  4. Alert construction:   Build ManipulationAlert from RiskScore + article
  5. Persistence:          Write alerts to the database
  6. Dispatch:             Push to async queue for dashboard + webhook consumers

Alert suppression rules
-----------------------
  - Same article_id:      Never re-alert (permanent)
  - Same cluster_id:      1 alert per cluster per cooldown window (default 30 min)
  - Same ticker:          Max 3 alerts per ticker per hour
  - LOW risk articles:    Always suppressed (below threshold)

Alert severity escalation
-------------------------
  If a cluster grows (more articles appear) after initial alert, re-score
  and potentially escalate from HIGH → CRITICAL.
"""

from __future__ import annotations

import json
from asyncio import Queue
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import settings
from database import RiskRepository, init_db
from logger import get_logger
from models import (
    EnrichedArticle,
    ManipulationAlert,
    RiskLevel,
    RiskScore,
    VolatilityAlert,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Alert deduplication state
# ---------------------------------------------------------------------------
class _AlertState:
    """
    Tracks alert history for deduplication and rate limiting.
    In-memory only — reset on restart (acceptable for a monitoring system).
    """

    def __init__(
        self,
        cooldown_minutes: float = 30.0,
        max_per_ticker_per_hour: int = 3,
    ):
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._max_per_ticker = max_per_ticker_per_hour

        # Permanent dedup: article_id → alert_id
        self._alerted_articles: dict[str, str] = {}

        # Cluster cooldown: cluster_id → last_alert_time
        self._cluster_last_alert: dict[str, datetime] = {}

        # Ticker rate limit: ticker → deque of alert timestamps (last hour)
        self._ticker_alerts: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._max_per_ticker)
        )

    def should_suppress(
        self,
        risk: RiskScore,
        article: EnrichedArticle,
    ) -> tuple[bool, str]:
        """
        Determine whether this alert should be suppressed.
        Returns (suppress: bool, reason: str).
        """
        # 1. Below global threshold
        if risk.composite_score < settings.risk_alert_threshold:
            return True, f"below_threshold({risk.composite_score:.3f}<{settings.risk_alert_threshold})"

        # 2. LOW risk level always suppressed
        if risk.risk_level == RiskLevel.LOW:
            return True, "risk_level_low"

        # 3. Article already alerted
        if risk.article_id in self._alerted_articles:
            return True, f"duplicate_article({risk.article_id[:8]})"

        # 4. Cluster cooldown
        cluster_id = (
            article.propagation.duplicate_cluster_id
            if article.propagation else None
        )
        if cluster_id and cluster_id in self._cluster_last_alert:
            last = self._cluster_last_alert[cluster_id]
            age = datetime.now(timezone.utc) - last
            if age < self._cooldown:
                remaining = (self._cooldown - age).total_seconds() / 60
                return True, f"cluster_cooldown({remaining:.0f}min remaining)"

        # 5. Ticker rate limit
        now = datetime.now(timezone.utc)
        for ticker in risk.related_tickers:
            q = self._ticker_alerts[ticker.upper()]
            # Remove timestamps older than 1 hour
            while q and (now - q[0]).total_seconds() > 3600:
                q.popleft()
            if len(q) >= self._max_per_ticker:
                return True, f"ticker_rate_limit({ticker}: {len(q)}/{self._max_per_ticker}/hr)"

        return False, ""

    def record_alert(
        self,
        alert: ManipulationAlert,
        article: EnrichedArticle,
    ) -> None:
        """Register an alert as emitted (updates all dedup state)."""
        risk = alert.risk_score

        # Mark article as alerted
        self._alerted_articles[risk.article_id] = alert.alert_id

        # Update cluster cooldown
        cluster_id = (
            article.propagation.duplicate_cluster_id
            if article.propagation else None
        )
        if cluster_id:
            self._cluster_last_alert[cluster_id] = datetime.now(timezone.utc)

        # Update ticker rate limit counters
        now = datetime.now(timezone.utc)
        for ticker in risk.related_tickers:
            self._ticker_alerts[ticker.upper()].append(now)

    def stats(self) -> dict:
        return {
            "alerted_articles": len(self._alerted_articles),
            "clusters_on_cooldown": len(self._cluster_last_alert),
            "ticker_alert_counts": {
                t: len(q) for t, q in self._ticker_alerts.items()
            },
        }


# ---------------------------------------------------------------------------
# Alert builder
# ---------------------------------------------------------------------------
def _build_alert(
    risk: RiskScore,
    article: EnrichedArticle,
    volatility_alerts: list[VolatilityAlert],
) -> ManipulationAlert:
    """
    Construct a ManipulationAlert from a RiskScore and its source article.

    The alert title and summary are written for a human operator reading
    the dashboard — they should be concise, factual, and actionable.
    """
    level_emoji = {
        RiskLevel.LOW:      "🟢",
        RiskLevel.MEDIUM:   "🟡",
        RiskLevel.HIGH:     "🟠",
        RiskLevel.CRITICAL: "🔴",
    }

    emoji = level_emoji.get(risk.risk_level, "⚪")
    title = (
        f"{emoji} {risk.risk_level.value.upper()} MANIPULATION RISK: "
        f"{article.article.title[:60]}"
    )

    # Build summary paragraphs
    summary_parts = [
        f"Source: {article.article.source_name}",
        f"Published: {article.article.published_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Risk score: {risk.composite_score:.3f}",
    ]

    if risk.ai_component >= 0.50:
        summary_parts.append(
            f"AI content: {risk.ai_component:.0%} probability"
        )
    if risk.propagation_component >= 0.30:
        prop = article.propagation
        if prop and prop.cluster_size > 1:
            summary_parts.append(
                f"Propagation: {prop.cluster_size} near-duplicate articles "
                f"spreading at {prop.spread_velocity:.1f}/hr"
            )
    if volatility_alerts:
        tickers_in_alert = {a.ticker for a in volatility_alerts}
        relevant = [
            a for a in volatility_alerts
            if a.ticker in {e.upper() for e in article.named_entities}
        ]
        if relevant:
            top = relevant[0]
            summary_parts.append(
                f"Market spike: {top.ticker} z={top.z_score:.1f} "
                f"(${top.current_price:.2f}, {top.direction})"
            )

    return ManipulationAlert(
        title=title,
        summary=" | ".join(summary_parts),
        risk_score=risk,
        enriched_article=article,
        volatility_alerts=volatility_alerts,
    )


# ---------------------------------------------------------------------------
# Core alert manager
# ---------------------------------------------------------------------------
class AlertManager:
    """
    Manages the full alert lifecycle: scoring → gating → dedup → dispatch.

    Usage
    -----
    >>> manager = AlertManager(alert_queue=my_queue)
    >>> alerts = manager.process(risk_scores, articles, volatility_alerts)
    >>> for alert in alerts:
    ...     print(alert.title)

    Alerts that pass all filters are:
      1. Persisted to the database (alerts table)
      2. Pushed to alert_queue (for dashboard + webhooks)
      3. Returned from process() for immediate use
    """

    def __init__(
        self,
        alert_queue: Optional[Queue] = None,
        cooldown_minutes: float = 30.0,
        max_per_ticker_per_hour: int = 3,
    ):
        self._queue: Optional[Queue] = alert_queue
        self._state = _AlertState(cooldown_minutes, max_per_ticker_per_hour)
        self._repo = RiskRepository()
        self._emitted = 0
        self._suppressed = 0
        init_db()

    def process(
        self,
        risk_scores: list[RiskScore],
        articles: list[EnrichedArticle],
        volatility_alerts: list[VolatilityAlert] | None = None,
    ) -> list[ManipulationAlert]:
        """
        Process a batch of risk scores and emit alerts for qualifying ones.

        Pairs each RiskScore with its source EnrichedArticle by article_id.
        """
        volatility_alerts = volatility_alerts or []
        art_lookup = {a.article.id: a for a in articles}
        emitted_alerts: list[ManipulationAlert] = []

        for risk in risk_scores:
            article = art_lookup.get(risk.article_id)
            if article is None:
                log.warning(f"[AlertManager] No article found for id {risk.article_id[:8]}")
                continue

            suppress, reason = self._state.should_suppress(risk, article)
            if suppress:
                self._suppressed += 1
                log.debug(f"[AlertManager] Suppressed ({reason}): {article.article.title[:50]}")
                continue

            # Build and persist alert
            alert = _build_alert(risk, article, volatility_alerts)
            self._persist_alert(alert, risk)
            self._persist_risk_event(risk)

            # Register in dedup state
            self._state.record_alert(alert, article)

            # Publish to queue
            if self._queue is not None:
                try:
                    self._queue.put_nowait(alert)
                except Exception:
                    log.warning("[AlertManager] Alert queue full — alert not queued")

            emitted_alerts.append(alert)
            self._emitted += 1

            log.warning(
                f"[AlertManager] 🚨 ALERT EMITTED: "
                f"[{risk.risk_level.value.upper()}] "
                f"score={risk.composite_score:.3f} | "
                f"{article.article.title[:60]}"
            )

        if emitted_alerts:
            log.info(
                f"[AlertManager] Batch: {len(emitted_alerts)} emitted, "
                f"{self._suppressed} suppressed total"
            )

        return emitted_alerts

    def process_single(
        self,
        risk: RiskScore,
        article: EnrichedArticle,
        volatility_alerts: list[VolatilityAlert] | None = None,
    ) -> Optional[ManipulationAlert]:
        """Convenience method: process a single risk score."""
        results = self.process([risk], [article], volatility_alerts)
        return results[0] if results else None

    # ── Persistence ───────────────────────────────────────────────────────
    def _persist_risk_event(self, risk: RiskScore) -> None:
        try:
            self._repo.insert_risk_event({
                "article_id":               risk.article_id,
                "composite_score":          risk.composite_score,
                "risk_level":               risk.risk_level.value,
                "ai_component":             risk.ai_component,
                "propagation_component":    risk.propagation_component,
                "market_impact_component":  risk.market_impact_component,
                "explanation":              risk.explanation,
                "related_tickers":          risk.related_tickers,
                "scored_at":                risk.scored_at,
            })
        except Exception as exc:
            log.error(f"[AlertManager] Risk event persist failed: {exc}")

    def _persist_alert(self, alert: ManipulationAlert, risk: RiskScore) -> None:
        try:
            self._repo.insert_alert({
                "alert_id":         alert.alert_id,
                "title":            alert.title,
                "summary":          alert.summary,
                "composite_score":  risk.composite_score,
                "risk_level":       risk.risk_level.value,
                "article_id":       risk.article_id,
                "related_tickers":  risk.related_tickers,
                "created_at":       alert.created_at,
                "full_json":        json.loads(alert.model_dump_json()),
            })
        except Exception as exc:
            log.error(f"[AlertManager] Alert persist failed: {exc}")

    # ── Recent alerts (for dashboard) ─────────────────────────────────────
    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        """Return the most recent alerts from the database."""
        return self._repo.get_recent_alerts(limit=limit)

    # ── Stats ─────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "emitted": self._emitted,
            "suppressed": self._suppressed,
            "suppression_rate": (
                self._suppressed / max(1, self._emitted + self._suppressed)
            ),
            "dedup_state": self._state.stats(),
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models import (
        ContentSource, RawArticle, EnrichedArticle, PropagationMetrics,
        SentimentResult, Sentiment, AIDetectionResult, RiskScore, RiskLevel,
    )
    import uuid

    def _mock_risk(article_id, score, level=None):
        return RiskScore(
            article_id=article_id,
            composite_score=score,
            risk_level=level or RiskLevel.LOW,
            ai_component=score * 0.9,
            propagation_component=score * 0.7,
            market_impact_component=score * 0.85,
            related_tickers=["TSLA", "GME"],
        )

    def _mock_article(article_id, title, coord=False):
        raw = RawArticle(
            id=article_id,
            source=ContentSource.RSS, source_name="TestFeed",
            title=title, body="Body text.",
            published_at=datetime.now(timezone.utc), raw_metadata={},
        )
        return EnrichedArticle(
            article=raw,
            propagation=PropagationMetrics(
                coordination_score=0.8 if coord else 0.0,
                is_coordinated=coord, cluster_size=8 if coord else 1,
                spread_velocity=6.0 if coord else 0.0,
            ),
            named_entities=["TSLA"],
        )

    print("\n🚨 AlertManager Demo\n")
    manager = AlertManager(cooldown_minutes=5, max_per_ticker_per_hour=3)

    test_cases = [
        (0.30, "Routine market update — Tesla Q3 summary"),            # Below threshold
        (0.68, "TSLA puts activity rises ahead of earnings"),          # HIGH
        (0.87, "BREAKING: Musk arrested — coordinated AI campaign"),   # CRITICAL
        (0.87, "BREAKING: Musk arrested — coordinated AI campaign"),   # Duplicate → suppressed
        (0.72, "TSLA insider selling leak — sources say collapse"),    # HIGH (different article)
        (0.75, "GME squeeze imminent — hedge funds to cover"),         # Rate limit test
    ]

    for score, title in test_cases:
        aid = str(uuid.uuid4())
        risk  = _mock_risk(aid, score)
        art   = _mock_article(aid, title, coord=score > 0.7)
        alert = manager.process_single(risk, art)
        status = "✅ EMITTED" if alert else "🔕 suppressed"
        print(f"  {status}  score={score:.2f}  {title[:55]}")
        if alert:
            print(f"           level={alert.risk_score.risk_level.value}")

    print(f"\nStats: {manager.stats()}")
