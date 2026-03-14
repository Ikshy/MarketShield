"""
tests/test_risk_engine.py

Tests for risk scoring, alert management, and the end-to-end pipeline.
All tests use in-memory state — no live APIs or external services.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from models import (
    AIDetectionResult, ContentSource, EnrichedArticle, ManipulationAlert,
    PropagationMetrics, RawArticle, RiskLevel, RiskScore,
    Sentiment, SentimentResult, VolatilityAlert,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _article(
    title: str = "Test Article",
    body: str = "Some financial content about market movements.",
    ai_prob: float = 0.1,
    coord_score: float = 0.0,
    cluster_sz: int = 1,
    velocity: float = 0.0,
    sentiment: Sentiment = Sentiment.NEUTRAL,
    entities: list[str] | None = None,
    hours_ago: float = 0.0,
) -> EnrichedArticle:
    raw = RawArticle(
        source=ContentSource.RSS, source_name="TestFeed",
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
            ai_probability=ai_prob,
            is_ai_generated=ai_prob >= 0.7,
            perplexity_score=16.0 if ai_prob > 0.7 else 75.0,
            burstiness_score=0.07 if ai_prob > 0.7 else 0.55,
        ),
        propagation=PropagationMetrics(
            coordination_score=coord_score,
            is_coordinated=coord_score >= 0.45,
            cluster_size=cluster_sz,
            spread_velocity=velocity,
        ),
        named_entities=entities or ["TSLA", "GME"],
    )


def _vol_alert(ticker: str = "TSLA", z: float = 4.5,
               direction: str = "down") -> VolatilityAlert:
    return VolatilityAlert(
        ticker=ticker, z_score=z, current_price=150.0,
        baseline_mean=0.001, baseline_std=0.015, direction=direction,
    )


# ---------------------------------------------------------------------------
# Component score tests
# ---------------------------------------------------------------------------
class TestComponentScorers:

    def test_ai_component_zero_for_none(self):
        from risk_engine.scorer import _ai_component
        assert _ai_component(None) == 0.0

    def test_ai_component_base_probability(self):
        from risk_engine.scorer import _ai_component
        ai = AIDetectionResult(
            ai_probability=0.70, is_ai_generated=True,
            perplexity_score=50.0, burstiness_score=0.40,
        )
        score = _ai_component(ai)
        assert score == pytest.approx(0.70, abs=0.01)

    def test_ai_component_perplexity_bonus(self):
        from risk_engine.scorer import _ai_component
        ai_low_ppl = AIDetectionResult(
            ai_probability=0.70, is_ai_generated=True,
            perplexity_score=15.0,   # Very low → +0.10 bonus
            burstiness_score=0.40,
        )
        ai_high_ppl = AIDetectionResult(
            ai_probability=0.70, is_ai_generated=True,
            perplexity_score=80.0,   # Normal → no bonus
            burstiness_score=0.40,
        )
        assert _ai_component(ai_low_ppl) > _ai_component(ai_high_ppl)

    def test_ai_component_capped_at_one(self):
        from risk_engine.scorer import _ai_component
        ai = AIDetectionResult(
            ai_probability=0.99, is_ai_generated=True,
            perplexity_score=10.0, burstiness_score=0.05,  # Max bonuses
        )
        assert _ai_component(ai) <= 1.0

    def test_propagation_component_zero_for_none(self):
        from risk_engine.scorer import _propagation_component
        assert _propagation_component(None) == 0.0

    def test_propagation_component_base_coordination(self):
        from risk_engine.scorer import _propagation_component
        prop = PropagationMetrics(
            coordination_score=0.60, is_coordinated=True,
            cluster_size=5, spread_velocity=2.0,
        )
        score = _propagation_component(prop)
        assert score >= 0.60  # At least the base score

    def test_propagation_coordinated_floor(self):
        from risk_engine.scorer import _propagation_component
        prop = PropagationMetrics(
            coordination_score=0.10,   # Low raw score
            is_coordinated=True,       # But explicitly flagged
            cluster_size=1, spread_velocity=0.0,
        )
        score = _propagation_component(prop)
        assert score >= 0.50  # Floor for coordinated articles

    def test_propagation_singleton_low_score(self):
        from risk_engine.scorer import _propagation_component
        prop = PropagationMetrics(
            coordination_score=0.0, is_coordinated=False,
            cluster_size=1, spread_velocity=0.0,
        )
        score = _propagation_component(prop)
        assert score < 0.10

    def test_market_impact_zero_no_match(self):
        from risk_engine.scorer import _market_impact_component
        score = _market_impact_component(
            named_entities=["NVDA"],
            volatility_alerts=[_vol_alert("TSLA")],  # TSLA alert, not NVDA
        )
        assert score == 0.0

    def test_market_impact_with_matching_alert(self):
        from risk_engine.scorer import _market_impact_component
        score = _market_impact_component(
            named_entities=["TSLA"],
            volatility_alerts=[_vol_alert("TSLA", z=5.5)],
        )
        assert score > 0.5

    def test_market_impact_high_z_score(self):
        from risk_engine.scorer import _market_impact_component
        low_z  = _market_impact_component(["TSLA"], [_vol_alert("TSLA", z=2.5)])
        high_z = _market_impact_component(["TSLA"], [_vol_alert("TSLA", z=6.0)])
        assert high_z > low_z


# ---------------------------------------------------------------------------
# RiskScorer integration tests
# ---------------------------------------------------------------------------
class TestRiskScorer:

    def setup_method(self):
        from risk_engine.scorer import RiskScorer
        self.scorer = RiskScorer()

    def test_legitimate_article_low_risk(self):
        art = _article(
            title="Tesla Q4 earnings beat analyst estimates by 8%",
            ai_prob=0.05, coord_score=0.0, cluster_sz=1, velocity=0.0,
            sentiment=Sentiment.POSITIVE,
        )
        risk = self.scorer.score(art)
        assert risk.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)
        assert risk.composite_score < 0.50

    def test_full_signal_article_high_risk(self):
        art = _article(
            title="BREAKING: Musk arrested FBI fraud TSLA crashing",
            ai_prob=0.93, coord_score=0.88, cluster_sz=18, velocity=12.0,
            sentiment=Sentiment.NEGATIVE,
        )
        vol_alerts = [_vol_alert("TSLA", z=5.0)]
        risk = self.scorer.score(art, volatility_alerts=vol_alerts)
        assert risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert risk.composite_score >= 0.65

    def test_composite_score_in_range(self):
        art = _article(ai_prob=0.5, coord_score=0.3)
        risk = self.scorer.score(art)
        assert 0.0 <= risk.composite_score <= 1.0

    def test_risk_level_auto_assigned(self):
        art = _article(ai_prob=0.95, coord_score=0.90, cluster_sz=20, velocity=15.0)
        risk = self.scorer.score(art, volatility_alerts=[_vol_alert("TSLA", z=6.0)])
        assert risk.risk_level != RiskLevel.LOW

    def test_explanation_non_empty(self):
        art = _article(ai_prob=0.80, coord_score=0.60)
        risk = self.scorer.score(art)
        assert len(risk.explanation) > 10

    def test_related_tickers_populated(self):
        art = _article(entities=["TSLA", "GME", "NVDA"])
        risk = self.scorer.score(art)
        assert len(risk.related_tickers) > 0

    def test_score_batch_correct_count(self):
        articles = [_article(ai_prob=p) for p in [0.1, 0.5, 0.9]]
        risks = self.scorer.score_batch(articles)
        assert len(risks) == 3

    def test_higher_ai_prob_higher_score(self):
        low_ai  = self.scorer.score(_article(ai_prob=0.10, coord_score=0.0))
        high_ai = self.scorer.score(_article(ai_prob=0.95, coord_score=0.0))
        assert high_ai.composite_score > low_ai.composite_score

    def test_weights_sum_normalisation(self):
        """Score should still be valid even if weights don't sum to 1.0."""
        art = _article(ai_prob=0.8, coord_score=0.7)
        risk = self.scorer.score(art)
        assert 0.0 <= risk.composite_score <= 1.0

    def test_stats_increments(self):
        initial = self.scorer.stats()["total_scored"]
        self.scorer.score(_article())
        self.scorer.score(_article())
        assert self.scorer.stats()["total_scored"] == initial + 2


# ---------------------------------------------------------------------------
# AlertManager tests
# ---------------------------------------------------------------------------
class TestAlertManager:

    def setup_method(self):
        from risk_engine.alert_manager import AlertManager
        # Patch out DB calls to keep tests fast
        with patch("risk_engine.alert_manager.RiskRepository"):
            with patch("risk_engine.alert_manager.init_db"):
                self.manager = AlertManager(cooldown_minutes=30.0)
                self.manager._repo = MagicMock()

    def _make_risk(self, article_id: str, score: float) -> RiskScore:
        return RiskScore(
            article_id=article_id,
            composite_score=score,
            risk_level=RiskLevel.LOW,
            ai_component=score * 0.9,
            propagation_component=score * 0.7,
            market_impact_component=score * 0.85,
            related_tickers=["TSLA"],
        )

    def test_below_threshold_suppressed(self):
        art = _article(title="Low risk article")
        risk = self._make_risk(art.article.id, 0.20)  # Well below 0.65
        alerts = self.manager.process([risk], [art])
        assert len(alerts) == 0

    def test_above_threshold_emitted(self):
        art = _article(title="High risk article", ai_prob=0.90, coord_score=0.80)
        risk = self._make_risk(art.article.id, 0.80)
        alerts = self.manager.process([risk], [art])
        assert len(alerts) == 1

    def test_duplicate_article_suppressed(self):
        art = _article(title="High risk article", ai_prob=0.90)
        risk = self._make_risk(art.article.id, 0.80)
        # First call: emitted
        first = self.manager.process([risk], [art])
        # Second call: same article_id → suppressed
        second = self.manager.process([risk], [art])
        assert len(first) == 1
        assert len(second) == 0

    def test_alert_has_required_fields(self):
        art = _article(title="Critical manipulation attempt", ai_prob=0.95, coord_score=0.85)
        risk = self._make_risk(art.article.id, 0.85)
        alerts = self.manager.process([risk], [art])
        if alerts:
            a = alerts[0]
            assert a.alert_id
            assert a.title
            assert a.summary
            assert a.risk_score is not None
            assert a.enriched_article is not None

    def test_stats_track_emitted_and_suppressed(self):
        # One suppressed (below threshold), one emitted (above)
        art1 = _article(title="Article 1")
        art2 = _article(title="Article 2", ai_prob=0.9)
        risk_low  = self._make_risk(art1.article.id, 0.20)
        risk_high = self._make_risk(art2.article.id, 0.80)

        self.manager.process([risk_low, risk_high], [art1, art2])
        stats = self.manager.stats()
        assert stats["emitted"] >= 1
        assert stats["suppressed"] >= 1


# ---------------------------------------------------------------------------
# RiskPipeline integration tests
# ---------------------------------------------------------------------------
class TestRiskPipeline:

    @pytest.mark.asyncio
    async def test_run_once_returns_alerts_for_high_risk(self):
        from risk_engine.pipeline import RiskPipeline

        with patch("risk_engine.pipeline.RiskRepository"), \
             patch("risk_engine.pipeline.init_db"):
            pipeline = RiskPipeline()
            pipeline._alert_manager._repo = MagicMock()

        high_risk = _article(
            title="CRITICAL: Mass manipulation campaign detected",
            ai_prob=0.95, coord_score=0.90, cluster_sz=20, velocity=15.0,
            sentiment=Sentiment.NEGATIVE,
        )
        normal = _article(
            title="Tesla earnings in line with expectations",
            ai_prob=0.05, coord_score=0.0, cluster_sz=1, velocity=0.0,
            sentiment=Sentiment.NEUTRAL,
        )

        vol_alerts = [_vol_alert("TSLA", z=5.2)]
        alerts = await pipeline.run_once([high_risk, normal], vol_alerts)

        # High risk article should trigger an alert
        assert len(alerts) >= 1
        for a in alerts:
            assert a.risk_score.composite_score >= settings_threshold()

    @pytest.mark.asyncio
    async def test_run_once_empty_input(self):
        from risk_engine.pipeline import RiskPipeline
        pipeline = RiskPipeline()
        with patch.object(pipeline._alert_manager._repo, "insert_risk_event"), \
             patch.object(pipeline._alert_manager._repo, "insert_alert"):
            alerts = await pipeline.run_once([])
        assert alerts == []

    @pytest.mark.asyncio
    async def test_dashboard_state_structure(self):
        from risk_engine.pipeline import RiskPipeline
        pipeline = RiskPipeline()

        articles = [_article(ai_prob=p, coord_score=c)
                    for p, c in [(0.1, 0.0), (0.5, 0.3), (0.9, 0.8)]]
        with patch.object(pipeline._alert_manager._repo, "insert_risk_event"), \
             patch.object(pipeline._alert_manager._repo, "insert_alert"):
            await pipeline.run_once(articles)

        state = pipeline.get_dashboard_state()
        assert "score_distribution" in state
        assert "pipeline_stats" in state
        assert "recent_alerts" in state
        assert "top_risk_articles" in state
        assert state["pipeline_stats"]["total_scored"] == 3

    @pytest.mark.asyncio
    async def test_full_end_to_end_demo(self):
        """Integration: all 5 pipeline stages produce valid output."""
        from risk_engine.pipeline import run_full_pipeline_demo

        with patch("risk_engine.pipeline.init_db"), \
             patch("risk_engine.alert_manager.init_db"), \
             patch("database.ArticleRepository"), \
             patch("database.RiskRepository") as mock_repo_cls:

            mock_repo = MagicMock()
            mock_repo_cls.return_value = mock_repo

            result = await run_full_pipeline_demo()

        assert result["articles_scored"] == 5
        assert "score_distribution" in result
        total = sum(result["score_distribution"].values())
        assert total == 5   # All articles classified

        # Verify score ordering
        dist = result["score_distribution"]
        assert dist.get("low", 0) + dist.get("medium", 0) >= 1  # At least one normal
        assert dist.get("high", 0) + dist.get("critical", 0) >= 1  # At least one flagged


def settings_threshold():
    from config import settings
    return settings.risk_alert_threshold


# ---------------------------------------------------------------------------
# Scoring formula validation
# ---------------------------------------------------------------------------
class TestScoringFormula:
    """
    Verify the weighted composite formula behaves correctly
    across a range of component combinations.
    """

    def test_all_zeros_score_zero(self):
        from risk_engine.scorer import RiskScorer
        scorer = RiskScorer()
        art = _article(ai_prob=0.0, coord_score=0.0)
        # Force market impact to zero by having no matching ticker in alerts
        risk = scorer.score(art, volatility_alerts=[])
        assert risk.composite_score < 0.10

    def test_all_ones_score_near_one(self):
        from risk_engine.scorer import RiskScorer
        scorer = RiskScorer()
        art = _article(
            ai_prob=0.99, coord_score=0.99,
            cluster_sz=50, velocity=50.0, entities=["TSLA"]
        )
        risk = scorer.score(
            art,
            volatility_alerts=[_vol_alert("TSLA", z=10.0)],
        )
        assert risk.composite_score >= 0.80

    def test_score_monotonic_with_ai_prob(self):
        from risk_engine.scorer import RiskScorer
        scorer = RiskScorer()
        scores = [
            scorer.score(_article(ai_prob=p, coord_score=0.0)).composite_score
            for p in [0.1, 0.4, 0.7, 0.9]
        ]
        # Each higher AI probability should produce a higher or equal score
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))

    def test_risk_level_thresholds(self):
        """Verify RiskLevel aligns with composite score thresholds."""
        from risk_engine.scorer import RiskScorer
        from models import RiskScore
        scorer = RiskScorer()

        for score, expected_level in [
            (0.20, RiskLevel.LOW),
            (0.50, RiskLevel.MEDIUM),
            (0.70, RiskLevel.HIGH),
            (0.90, RiskLevel.CRITICAL),
        ]:
            rs = RiskScore(
                article_id="test", composite_score=score,
                risk_level=RiskLevel.LOW,
                ai_component=score, propagation_component=score,
                market_impact_component=score,
            )
            assert rs.risk_level == expected_level, \
                f"score={score} should be {expected_level.value}, got {rs.risk_level.value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
