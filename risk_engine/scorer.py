"""
risk_engine/scorer.py — Composite risk scoring engine.

The risk score is the single number that summarises how likely a piece
of content is part of a financial manipulation campaign.

Formula
-------
    risk = w_ai   * AI_probability
         + w_prop * propagation_anomaly
         + w_mkt  * market_impact

Default weights (configurable via .env):
    w_ai   = 0.35   (AI-generated text is the primary signal)
    w_prop = 0.30   (coordinated spreading amplifies the threat)
    w_mkt  = 0.35   (actual market movement is the ultimate evidence)

Score thresholds → RiskLevel:
    0.00–0.39 → LOW      (monitor passively)
    0.40–0.64 → MEDIUM   (flag for review)
    0.65–0.84 → HIGH     (likely manipulation, alert)
    0.85–1.00 → CRITICAL (active campaign, escalate)

Component score derivation
--------------------------
  AI component:
    - Base: EnrichedArticle.ai_detection.ai_probability
    - Bonus: +0.10 if perplexity < 20 (extremely uniform text)
    - Bonus: +0.05 if burstiness < 0.15 (mechanically uniform style)
    - Capped at 1.0

  Propagation component:
    - Base: PropagationMetrics.coordination_score
    - Bonus: spread_velocity scaled to 0.1 bonus at 10+ articles/hour
    - Bonus: cluster_size scaled to 0.1 bonus at 20+ articles
    - Capped at 1.0

  Market impact component:
    - From MarketMonitor.market_impact_for_tickers()
    - Boosted by SentimentPriceCorrelator.manipulation_score()
    - If no market data available, uses sentiment extremity as proxy
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from config import settings
from logger import get_logger
from models import (
    AIDetectionResult,
    EnrichedArticle,
    ManipulationAlert,
    PropagationMetrics,
    RiskLevel,
    RiskScore,
    VolatilityAlert,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------
def _ai_component(ai: Optional[AIDetectionResult]) -> float:
    """
    Derive the AI-content component score (0–1) from detection results.

    Starts from the raw ai_probability, then applies bonuses for
    corroborating signals: very low perplexity or very low burstiness
    both indicate mechanically generated content.
    """
    if ai is None:
        return 0.0

    score = float(ai.ai_probability)

    # Perplexity bonus: human news perplexity is typically 50–150.
    # Values below 20 indicate the model generated almost every token.
    if ai.perplexity_score < 20.0:
        score += 0.10
    elif ai.perplexity_score < 30.0:
        score += 0.05

    # Burstiness bonus: AI writes in uniform "sheets" of similar complexity.
    # Human writing is uneven — some sentences polished, others rough.
    if ai.burstiness_score < 0.15:
        score += 0.05
    elif ai.burstiness_score < 0.25:
        score += 0.02

    return round(min(1.0, score), 4)


def _propagation_component(prop: Optional[PropagationMetrics]) -> float:
    """
    Derive the propagation anomaly component (0–1).

    Combines the base coordination_score from the graph analysis with
    bonuses for high velocity (fast spreading) and large cluster size
    (many copies exist).
    """
    if prop is None:
        return 0.0

    score = float(prop.coordination_score)

    # Velocity bonus: each article/hour above baseline adds a small increment
    # Saturates at 10 articles/hour (max +0.10 bonus)
    velocity_bonus = min(0.10, prop.spread_velocity / 100.0)
    score += velocity_bonus

    # Cluster size bonus: large clusters = wider campaign
    # +0.10 at cluster_size = 20
    size_bonus = min(0.10, (prop.cluster_size - 1) / 190.0)
    score += size_bonus

    # Hard flag: if explicitly marked as coordinated, minimum score = 0.50
    if prop.is_coordinated:
        score = max(0.50, score)

    return round(min(1.0, score), 4)


def _market_impact_component(
    named_entities: list[str],
    volatility_alerts: list[VolatilityAlert],
    correlator_score: float = 0.0,
    sentiment_extremity: float = 0.0,
) -> float:
    """
    Derive the market impact component (0–1).

    Priority order:
    1. Active volatility alert matching article tickers → direct evidence
    2. Sentiment-price contradiction score from correlator
    3. Sentiment extremity proxy (fallback when no market data)
    """
    # ── Direct volatility alert match ─────────────────────────────────────
    ticker_set = {e.upper() for e in named_entities}
    matching_alerts = [
        a for a in volatility_alerts
        if a.ticker.upper() in ticker_set
    ]

    if matching_alerts:
        max_z = max(abs(a.z_score) for a in matching_alerts)
        # Logistic curve: z=2.5→0.5, z=5→0.88
        alert_score = 1.0 / (1.0 + math.exp(-0.5 * (max_z - 3.0)))
        # Multiple simultaneous alerts = coordinated market impact
        if len(matching_alerts) > 1:
            alert_score = min(1.0, alert_score * 1.20)
        return round(min(1.0, alert_score), 4)

    # ── Correlator manipulation score ─────────────────────────────────────
    if correlator_score > 0.0:
        return round(min(1.0, correlator_score), 4)

    # ── Sentiment extremity proxy (no market data) ─────────────────────────
    # Extreme negative sentiment in AI-generated content targeting
    # a specific ticker is itself a manipulation signal even without
    # confirmed price movement.
    return round(min(0.40, sentiment_extremity * 0.40), 4)


def _sentiment_extremity(article: EnrichedArticle) -> float:
    """
    How extreme is the sentiment? 0 = neutral, 1 = maximally one-sided.
    Used as a market impact proxy when live price data is unavailable.
    """
    if not article.sentiment:
        return 0.0
    from models import Sentiment
    s = article.sentiment
    if s.label == Sentiment.NEUTRAL:
        return 0.0
    # Distance from 0.5 (neutral confidence) → extremity
    return round(min(1.0, abs(s.score - 0.5) * 2), 4)


# ---------------------------------------------------------------------------
# Explanation generator
# ---------------------------------------------------------------------------
def _generate_explanation(
    article: EnrichedArticle,
    ai_comp: float,
    prop_comp: float,
    mkt_comp: float,
    composite: float,
    volatility_alerts: list[VolatilityAlert],
) -> str:
    """
    Generate a human-readable risk explanation for the dashboard and alerts.
    Focuses on the highest-contributing signals.
    """
    parts: list[str] = []

    # Headline
    level = RiskScore(
        article_id="x", composite_score=composite,
        risk_level=RiskLevel.LOW,
        ai_component=ai_comp, propagation_component=prop_comp,
        market_impact_component=mkt_comp,
    ).risk_level

    parts.append(f"[{level.value.upper()}] Composite risk: {composite:.2f}")

    # AI signal
    if ai_comp >= 0.70:
        ppl = article.ai_detection.perplexity_score if article.ai_detection else "?"
        parts.append(
            f"AI-generated content detected "
            f"(probability={ai_comp:.0%}, perplexity={ppl:.1f})"
            if isinstance(ppl, float) else
            f"AI-generated content detected (probability={ai_comp:.0%})"
        )
    elif ai_comp >= 0.40:
        parts.append(f"Possible AI content (probability={ai_comp:.0%})")

    # Propagation signal
    if prop_comp >= 0.50:
        prop = article.propagation
        cluster_sz = prop.cluster_size if prop else 1
        velocity = prop.spread_velocity if prop else 0
        parts.append(
            f"Coordinated spreading: {cluster_sz} near-duplicate articles, "
            f"{velocity:.1f} articles/hour"
        )
    elif prop_comp >= 0.30:
        parts.append(f"Elevated propagation activity (score={prop_comp:.2f})")

    # Market signal
    matching = [
        a for a in volatility_alerts
        if a.ticker.upper() in {e.upper() for e in article.named_entities}
    ]
    if matching:
        for alert in matching[:2]:  # Show max 2 alerts in explanation
            parts.append(
                f"Volatility spike on {alert.ticker}: "
                f"z={alert.z_score:.1f}, "
                f"price ${alert.current_price:.2f} ({alert.direction})"
            )
    elif mkt_comp >= 0.40:
        parts.append(f"Elevated market impact (score={mkt_comp:.2f})")

    # Entities
    if article.named_entities:
        tickers = [e for e in article.named_entities
                   if e.isupper() and len(e) <= 5][:3]
        if tickers:
            parts.append(f"Mentions: {', '.join(tickers)}")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------
class RiskScorer:
    """
    Computes composite manipulation risk scores for enriched articles.

    Usage
    -----
    >>> scorer = RiskScorer()
    >>> risk = scorer.score(enriched_article, volatility_alerts=[...])
    >>> print(f"{risk.risk_level.value}: {risk.composite_score:.2f}")
    >>> print(risk.explanation)

    The scorer is stateless — each call is independent.
    Weights are read from config.settings on each call, so they
    can be changed at runtime without restarting the service.
    """

    def __init__(self):
        self._scored = 0
        self._high_risk_count = 0

    def score(
        self,
        article: EnrichedArticle,
        volatility_alerts: list[VolatilityAlert] | None = None,
        correlator_score: float = 0.0,
    ) -> RiskScore:
        """
        Compute the composite risk score for one enriched article.

        Parameters
        ----------
        article           : Fully enriched article (all NLP + propagation stages run)
        volatility_alerts : Recent VolatilityAlerts (from MarketMonitor)
        correlator_score  : Pre-computed manipulation score from SentimentPriceCorrelator

        Returns
        -------
        RiskScore with composite_score, component scores, and explanation.
        """
        volatility_alerts = volatility_alerts or []

        # ── Component scores ───────────────────────────────────────────────
        ai_comp   = _ai_component(article.ai_detection)
        prop_comp = _propagation_component(article.propagation)
        mkt_comp  = _market_impact_component(
            named_entities    = article.named_entities,
            volatility_alerts = volatility_alerts,
            correlator_score  = correlator_score,
            sentiment_extremity = _sentiment_extremity(article),
        )

        # ── Weighted composite ─────────────────────────────────────────────
        composite = (
            settings.risk_weight_ai_probability    * ai_comp
            + settings.risk_weight_propagation_anomaly * prop_comp
            + settings.risk_weight_market_impact       * mkt_comp
        )

        # Normalise (in case weights don't sum to exactly 1.0)
        weight_sum = settings.risk_weights_sum
        if weight_sum > 0:
            composite /= weight_sum

        composite = round(float(np.clip(composite, 0.0, 1.0)), 4)

        # ── Explanation ────────────────────────────────────────────────────
        explanation = _generate_explanation(
            article, ai_comp, prop_comp, mkt_comp, composite, volatility_alerts
        )

        # ── Build result ───────────────────────────────────────────────────
        risk = RiskScore(
            article_id              = article.article.id,
            composite_score         = composite,
            risk_level              = RiskLevel.LOW,   # Overwritten by model validator
            ai_component            = ai_comp,
            propagation_component   = prop_comp,
            market_impact_component = mkt_comp,
            explanation             = explanation,
            related_tickers         = [
                e for e in article.named_entities
                if e.isupper() and 1 <= len(e) <= 5
            ][:5],
        )

        # Update stats
        self._scored += 1
        if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            self._high_risk_count += 1
            log.warning(
                f"[Scorer] ⚠️  {risk.risk_level.value.upper()}: "
                f"score={composite:.3f}  "
                f"article='{article.article.title[:55]}'"
            )

        return risk

    def score_batch(
        self,
        articles: list[EnrichedArticle],
        volatility_alerts: list[VolatilityAlert] | None = None,
        correlator_scores: dict[str, float] | None = None,
    ) -> list[RiskScore]:
        """
        Score a batch of articles.

        correlator_scores: optional dict of {ticker → manipulation_score}
        """
        correlator_scores = correlator_scores or {}
        results: list[RiskScore] = []

        for art in articles:
            # Look up correlator score for any ticker this article mentions
            c_score = max(
                (correlator_scores.get(e.upper(), 0.0)
                 for e in art.named_entities),
                default=0.0,
            )
            results.append(
                self.score(art, volatility_alerts=volatility_alerts, correlator_score=c_score)
            )

        return results

    def stats(self) -> dict:
        return {
            "total_scored": self._scored,
            "high_risk_count": self._high_risk_count,
            "high_risk_rate": (
                self._high_risk_count / self._scored if self._scored else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta
    from models import (
        ContentSource, RawArticle, EnrichedArticle, PropagationMetrics,
        SentimentResult, Sentiment, AIDetectionResult, VolatilityAlert,
    )

    def _art(title, body, ai_prob, coord_score, cluster_sz, velocity, sentiment=Sentiment.NEGATIVE):
        raw = RawArticle(
            source=ContentSource.RSS, source_name="Test",
            title=title, body=body,
            published_at=datetime.now(timezone.utc),
            raw_metadata={},
        )
        return EnrichedArticle(
            article=raw,
            sentiment=SentimentResult(label=sentiment, score=0.88,
                                      positive=0.06 if sentiment==Sentiment.NEGATIVE else 0.88,
                                      negative=0.88 if sentiment==Sentiment.NEGATIVE else 0.06,
                                      neutral=0.06),
            ai_detection=AIDetectionResult(
                ai_probability=ai_prob, is_ai_generated=ai_prob>=0.7,
                perplexity_score=18.0 if ai_prob>0.7 else 75.0,
                burstiness_score=0.08 if ai_prob>0.7 else 0.55,
            ),
            propagation=PropagationMetrics(
                coordination_score=coord_score,
                is_coordinated=coord_score>=0.45,
                cluster_size=cluster_sz,
                spread_velocity=velocity,
            ),
            named_entities=["TSLA", "GME"],
        )

    SAMPLES = [
        ("✅ Legitimate news",
         _art("Tesla Q4 earnings beat analyst estimates",
              "Revenue $25.2B, EPS $2.27 vs $2.10 expected.",
              ai_prob=0.08, coord_score=0.0, cluster_sz=1, velocity=0.0,
              sentiment=Sentiment.POSITIVE)),
        ("⚠️  Suspicious blog post",
         _art("TSLA short squeeze incoming, buy NOW",
              "Sources say massive short position about to unwind.",
              ai_prob=0.55, coord_score=0.20, cluster_sz=3, velocity=1.5)),
        ("🚨 Coordinated AI campaign",
         _art("BREAKING: Elon Musk arrested for securities fraud by FBI",
              "Multiple sources confirm CEO detained. TSLA to crash immediately.",
              ai_prob=0.92, coord_score=0.78, cluster_sz=12, velocity=8.5)),
        ("💥 Critical: All signals firing",
         _art("URGENT: TSLA insider selling, company bankrupt by Friday",
              "Leaked documents show Tesla has zero cash. Sell everything.",
              ai_prob=0.96, coord_score=0.91, cluster_sz=25, velocity=18.0)),
    ]

    FAKE_ALERTS = [
        VolatilityAlert(
            ticker="TSLA", z_score=4.8, current_price=155.0,
            baseline_mean=0.001, baseline_std=0.015, direction="down",
        ),
    ]

    scorer = RiskScorer()

    print("\n🎯 RiskScorer Demo\n")
    print(f"{'Sample':<35} {'AI':>6} {'Prop':>6} {'Mkt':>6} {'Score':>7}  {'Level'}")
    print("-" * 75)

    for label, article in SAMPLES:
        risk = scorer.score(article, volatility_alerts=FAKE_ALERTS)
        print(
            f"{label:<35} "
            f"{risk.ai_component:>6.3f} "
            f"{risk.propagation_component:>6.3f} "
            f"{risk.market_impact_component:>6.3f} "
            f"{risk.composite_score:>7.3f}  "
            f"{risk.risk_level.value.upper()}"
        )
        if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            print(f"  └─ {risk.explanation[:90]}")

    print(f"\nStats: {scorer.stats()}")
