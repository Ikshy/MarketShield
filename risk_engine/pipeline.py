"""
risk_engine/pipeline.py — Real-time risk scoring pipeline orchestrator.

Sits at the end of the processing chain, consuming from all upstream stages:

    data_ingestion.runner.article_queue       → RawArticle
    ai_detection.runner.output_queue          → EnrichedArticle
    market_analysis.monitor.alert_queue       → VolatilityAlert

Processing loop (every N seconds or when queue drains):
  1. Drain EnrichedArticle queue
  2. Get latest VolatilityAlerts from MarketMonitor
  3. Get correlator scores for mentioned tickers
  4. Score each article → RiskScore
  5. Pass to AlertManager (threshold + dedup + persist + dispatch)
  6. Publish ManipulationAlerts to output_queue (dashboard consumer)

Also provides:
  - run_once(): score a fixed list of articles (for CLI/testing)
  - get_dashboard_state(): snapshot of all current risk data for dashboard
"""

from __future__ import annotations

import asyncio
import json
import time
from asyncio import Queue
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from config import settings
from database import init_db
from logger import get_logger, configure_logging
from models import (
    EnrichedArticle,
    ManipulationAlert,
    MarketSnapshot,
    RiskScore,
    VolatilityAlert,
)

from risk_engine.scorer import RiskScorer
from risk_engine.alert_manager import AlertManager

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
class RiskPipeline:
    """
    End-to-end risk scoring pipeline.

    Consumes EnrichedArticles + VolatilityAlerts, emits ManipulationAlerts.

    Usage (standalone)
    ------------------
    >>> pipeline = RiskPipeline()
    >>> alerts = await pipeline.run_once(enriched_articles, volatility_alerts)

    Usage (continuous, integrated with other runners)
    -------------------------------------------------
    >>> pipeline = RiskPipeline(
    ...     enriched_queue=nlp_runner.output_queue,
    ...     market_monitor=monitor,
    ...     alert_queue=dashboard_queue,
    ... )
    >>> await pipeline.run()
    """

    def __init__(
        self,
        enriched_queue: Optional[Queue] = None,
        alert_queue: Optional[Queue] = None,
        market_monitor=None,
        correlator=None,
    ):
        self._enriched_queue: Queue[EnrichedArticle] = enriched_queue or Queue()
        self._alert_queue: Queue[ManipulationAlert]  = alert_queue or Queue(maxsize=500)

        self._scorer        = RiskScorer()
        self._alert_manager = AlertManager(alert_queue=self._alert_queue)
        self._market_monitor = market_monitor    # Optional MarketMonitor instance
        self._correlator     = correlator        # Optional SentimentPriceCorrelator

        # Rolling recent state (for dashboard queries)
        self._recent_scores:  deque[RiskScore]          = deque(maxlen=200)
        self._recent_alerts:  deque[ManipulationAlert]  = deque(maxlen=50)
        self._recent_vol_alerts: deque[VolatilityAlert] = deque(maxlen=100)

        self._cycle   = 0
        self._scored  = 0
        self._alerted = 0

        init_db()

    # ── Single-cycle processing ───────────────────────────────────────────
    async def _process_cycle(self) -> int:
        """
        Drain the enriched queue and score all waiting articles.
        Returns number of articles processed in this cycle.
        """
        # Collect all waiting enriched articles (non-blocking drain)
        batch: list[EnrichedArticle] = []
        while not self._enriched_queue.empty():
            try:
                article = self._enriched_queue.get_nowait()
                batch.append(article)
            except asyncio.QueueEmpty:
                break

        if not batch:
            return 0

        self._cycle += 1
        start = time.monotonic()

        # Get latest volatility alerts
        vol_alerts = self._get_volatility_alerts()

        # Get correlator scores for mentioned tickers
        correlator_scores = self._get_correlator_scores(batch)

        # Score batch
        risk_scores = self._scorer.score_batch(
            batch,
            volatility_alerts=vol_alerts,
            correlator_scores=correlator_scores,
        )

        # Feed correlator with new articles (for future correlation)
        if self._correlator:
            self._correlator.add_articles(batch)

        # Alert manager processes scores
        alerts = self._alert_manager.process(
            risk_scores, batch, vol_alerts
        )

        # Update rolling state
        self._recent_scores.extend(risk_scores)
        self._recent_alerts.extend(alerts)
        self._recent_vol_alerts.extend(vol_alerts)

        self._scored  += len(batch)
        self._alerted += len(alerts)

        elapsed = time.monotonic() - start
        log.info(
            f"[RiskPipeline] Cycle {self._cycle}: "
            f"{len(batch)} articles scored, "
            f"{len(alerts)} alerts emitted, "
            f"{elapsed:.2f}s"
        )
        return len(batch)

    def _get_volatility_alerts(self) -> list[VolatilityAlert]:
        """Drain recent volatility alerts from the market monitor."""
        if self._market_monitor is None:
            return []
        try:
            return self._market_monitor.get_recent_alerts()
        except Exception as exc:
            log.warning(f"[RiskPipeline] Could not get vol alerts: {exc}")
            return []

    def _get_correlator_scores(
        self, articles: list[EnrichedArticle]
    ) -> dict[str, float]:
        """Get manipulation correlation scores for all mentioned tickers."""
        if self._correlator is None:
            return {}

        scores: dict[str, float] = {}
        mentioned_tickers = set()
        for art in articles:
            for e in art.named_entities:
                if e.isupper() and 1 <= len(e) <= 5:
                    mentioned_tickers.add(e)

        for ticker in mentioned_tickers:
            try:
                result = self._correlator.manipulation_score(ticker)
                scores[ticker] = result.score
            except Exception:
                pass

        return scores

    # ── Continuous loop ───────────────────────────────────────────────────
    async def run(self, poll_interval: float = 2.0) -> None:
        """
        Continuous risk scoring loop. Runs until cancelled.

        Polls the enriched queue every poll_interval seconds.
        When queue is empty, waits before polling again.
        """
        log.info("=" * 60)
        log.info("[RiskPipeline] Risk scoring pipeline starting")
        log.info(f"[RiskPipeline] Alert threshold: {settings.risk_alert_threshold}")
        log.info(f"[RiskPipeline] Weights: AI={settings.risk_weight_ai_probability}, "
                 f"Prop={settings.risk_weight_propagation_anomaly}, "
                 f"Mkt={settings.risk_weight_market_impact}")
        log.info("=" * 60)

        try:
            while True:
                processed = await self._process_cycle()
                if processed == 0:
                    await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            log.info(
                f"[RiskPipeline] Stopped — "
                f"{self._scored} scored, {self._alerted} alerts"
            )

    # ── Standalone / CLI mode ─────────────────────────────────────────────
    async def run_once(
        self,
        articles: list[EnrichedArticle],
        volatility_alerts: list[VolatilityAlert] | None = None,
        correlator_scores: dict[str, float] | None = None,
    ) -> list[ManipulationAlert]:
        """
        Score a fixed list of articles and return emitted alerts.
        No queues involved — pure function-style for testing/CLI.
        """
        vol_alerts = volatility_alerts or []
        corr_scores = correlator_scores or {}

        risk_scores = self._scorer.score_batch(
            articles,
            volatility_alerts=vol_alerts,
            correlator_scores=corr_scores,
        )

        alerts = self._alert_manager.process(risk_scores, articles, vol_alerts)

        self._recent_scores.extend(risk_scores)
        self._recent_alerts.extend(alerts)
        self._scored  += len(articles)
        self._alerted += len(alerts)

        return alerts

    # ── Dashboard data ────────────────────────────────────────────────────
    def get_dashboard_state(self) -> dict:
        """
        Snapshot of current pipeline state for the dashboard.
        Called by the Streamlit app on each refresh cycle.
        """
        recent_scores = list(self._recent_scores)
        recent_alerts = list(self._recent_alerts)

        # Score distribution
        score_bins = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for rs in recent_scores:
            score_bins[rs.risk_level.value] += 1

        # Top risk articles (for live feed)
        top_articles = sorted(
            recent_scores,
            key=lambda s: s.composite_score,
            reverse=True,
        )[:10]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_stats": {
                "total_scored": self._scored,
                "total_alerted": self._alerted,
                "cycles": self._cycle,
            },
            "score_distribution": score_bins,
            "recent_alerts": [
                {
                    "alert_id":    a.alert_id,
                    "title":       a.title,
                    "summary":     a.summary,
                    "score":       a.risk_score.composite_score,
                    "level":       a.risk_score.risk_level.value,
                    "tickers":     a.risk_score.related_tickers,
                    "created_at":  a.created_at.isoformat(),
                    "source":      a.enriched_article.article.source_name,
                }
                for a in sorted(recent_alerts, key=lambda a: a.created_at, reverse=True)[:20]
            ],
            "top_risk_articles": [
                {
                    "article_id":   rs.article_id,
                    "score":        rs.composite_score,
                    "level":        rs.risk_level.value,
                    "ai_component": rs.ai_component,
                    "prop_component": rs.propagation_component,
                    "mkt_component":  rs.market_impact_component,
                    "tickers":        rs.related_tickers,
                    "explanation":    rs.explanation[:100],
                }
                for rs in top_articles
            ],
            "scorer_stats":  self._scorer.stats(),
            "alert_stats":   self._alert_manager.stats(),
        }


# ---------------------------------------------------------------------------
# Full end-to-end integration runner
# ---------------------------------------------------------------------------
async def run_full_pipeline_demo() -> dict:
    """
    Run all pipeline stages end-to-end using simulated data.
    Returns a comprehensive demo result for testing and CI.
    """
    from datetime import timedelta
    from models import (
        ContentSource, RawArticle, EnrichedArticle, PropagationMetrics,
        SentimentResult, Sentiment, AIDetectionResult, VolatilityAlert,
    )

    # ── Build test articles covering the full risk spectrum ─────────────
    def _make(title, body, ai_prob, coord_score, cluster_sz, velocity,
              sentiment=Sentiment.NEGATIVE, hours_ago=0.0):
        raw = RawArticle(
            source=ContentSource.RSS, source_name="Test",
            title=title, body=body,
            published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
            raw_metadata={},
        )
        return EnrichedArticle(
            article=raw,
            sentiment=SentimentResult(
                label=sentiment, score=0.88,
                positive=0.06 if sentiment != Sentiment.POSITIVE else 0.88,
                negative=0.88 if sentiment == Sentiment.NEGATIVE else 0.06,
                neutral=0.06,
            ),
            ai_detection=AIDetectionResult(
                ai_probability=ai_prob, is_ai_generated=ai_prob >= 0.7,
                perplexity_score=16.0 if ai_prob > 0.7 else 72.0,
                burstiness_score=0.07 if ai_prob > 0.7 else 0.58,
            ),
            propagation=PropagationMetrics(
                coordination_score=coord_score,
                is_coordinated=coord_score >= 0.45,
                cluster_size=cluster_sz,
                spread_velocity=velocity,
            ),
            named_entities=["TSLA", "GME", "Tesla"],
        )

    articles = [
        _make("Tesla Q4 beats expectations",
              "Revenue exceeded analyst estimates by 8%.",
              0.05, 0.0,  1,  0.0, Sentiment.POSITIVE, 2.0),

        _make("GME options activity elevated ahead of earnings",
              "Unusual put volume detected in after-hours trading.",
              0.35, 0.15, 2,  0.8, Sentiment.NEUTRAL, 1.5),

        _make("BREAKING: TSLA insider selling $500M worth of shares",
              "Sources say executives dumping before Q1 earnings miss.",
              0.70, 0.55, 6,  3.5, Sentiment.NEGATIVE, 1.0),

        _make("URGENT: Elon Musk arrested FBI securities fraud — TSLA crashing",
              "Multiple sources confirm CEO in FBI custody. Sell TSLA immediately.",
              0.93, 0.88, 18, 12.0, Sentiment.NEGATIVE, 0.5),

        _make("Buy GME NOW guaranteed 1000% returns AI algorithm confirmed",
              "Our proprietary AI system has identified imminent short squeeze.",
              0.97, 0.92, 24, 22.0, Sentiment.POSITIVE, 0.2),
    ]

    vol_alerts = [
        VolatilityAlert(
            ticker="TSLA", z_score=4.8, current_price=155.0,
            baseline_mean=0.001, baseline_std=0.015, direction="down",
        ),
        VolatilityAlert(
            ticker="GME", z_score=6.2, current_price=22.0,
            baseline_mean=0.002, baseline_std=0.020, direction="up",
        ),
    ]

    pipeline = RiskPipeline()
    alerts = await pipeline.run_once(articles, vol_alerts)
    state  = pipeline.get_dashboard_state()

    return {
        "articles_scored": len(articles),
        "alerts_emitted":  len(alerts),
        "score_distribution": state["score_distribution"],
        "top_risks": [
            {"title": a.enriched_article.article.title[:55],
             "score": a.risk_score.composite_score,
             "level": a.risk_score.risk_level.value}
            for a in sorted(alerts, key=lambda x: x.risk_score.composite_score, reverse=True)
        ],
        "pipeline_stats": state["pipeline_stats"],
        "scorer_stats":   state["scorer_stats"],
        "alert_stats":    state["alert_stats"],
    }


# ---------------------------------------------------------------------------
# Sync wrapper for main.py
# ---------------------------------------------------------------------------
def run_risk_pipeline() -> None:
    """Entry point for ``python main.py pipeline`` (full pipeline mode)."""
    configure_logging(level=settings.log_level, log_to_file=True)
    init_db()
    pipeline = RiskPipeline()
    asyncio.run(pipeline.run())


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json as _json

    configure_logging(level="INFO", log_to_file=False)

    async def _demo():
        print("\n⚙️  Risk Pipeline End-to-End Demo\n")
        result = await run_full_pipeline_demo()

        print(f"Articles scored:   {result['articles_scored']}")
        print(f"Alerts emitted:    {result['alerts_emitted']}")
        print(f"\nScore distribution: {result['score_distribution']}")

        print("\n🏆 Top risk events:")
        for item in result["top_risks"]:
            bar = "█" * int(item["score"] * 20)
            print(f"  [{item['level'].upper():<8}] {item['score']:.3f}  {bar:<20}  {item['title']}")

        print(f"\n📊 Scorer stats: {result['scorer_stats']}")
        print(f"🚨 Alert stats:  {result['alert_stats']}")

    asyncio.run(_demo())
