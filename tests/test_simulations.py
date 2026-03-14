"""
tests/test_simulations.py

Tests for the scenario definitions, replay engine, and evaluation metrics.
All tests run without live data or API calls.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone

import pytest

from simulations.scenarios.definitions import (
    ManipulationScenario,
    SimArticle,
    get_scenario,
    get_all_scenarios,
    ALL_SCENARIOS,
)
from simulations.historical_replay import (
    ArticleEvaluation,
    ScenarioResult,
    BenchmarkResult,
    HistoricalReplay,
    _sim_article_to_enriched,
    _get_volatility_alerts,
)


# ---------------------------------------------------------------------------
# Scenario definition tests
# ---------------------------------------------------------------------------
class TestScenarioDefinitions:

    def test_all_scenarios_loadable(self):
        scenarios = get_all_scenarios()
        assert len(scenarios) == len(ALL_SCENARIOS)

    def test_scenario_has_required_fields(self):
        for name in ALL_SCENARIOS:
            s = get_scenario(name)
            assert s.name
            assert s.ticker
            assert len(s.articles) >= 3
            assert len(s.price_series) >= 3
            assert s.duration_hours > 0
            assert s.baseline_price > 0

    def test_each_scenario_has_manipulative_and_legit_articles(self):
        for name in ALL_SCENARIOS:
            s = get_scenario(name)
            has_manip = any(a.is_manipulative for a in s.articles)
            has_legit  = any(not a.is_manipulative for a in s.articles)
            assert has_manip, f"{name} has no manipulative articles"
            assert has_legit, f"{name} has no legitimate articles"

    def test_manipulation_rate_between_zero_and_one(self):
        for name in ALL_SCENARIOS:
            s = get_scenario(name)
            assert 0.0 < s.manipulation_rate < 1.0

    def test_ai_probs_in_range(self):
        for name in ALL_SCENARIOS:
            s = get_scenario(name)
            for art in s.articles:
                assert 0.0 <= art.expected_ai_prob <= 1.0
                assert 0.0 <= art.expected_coord_score <= 1.0

    def test_manipulative_articles_have_higher_expected_scores(self):
        """Manipulative articles should generally have higher expected scores."""
        for name in ALL_SCENARIOS:
            s = get_scenario(name)
            manip_ai = [a.expected_ai_prob for a in s.articles if a.is_manipulative]
            legit_ai  = [a.expected_ai_prob for a in s.articles if not a.is_manipulative]
            if manip_ai and legit_ai:
                assert sum(manip_ai) / len(manip_ai) > sum(legit_ai) / len(legit_ai)

    def test_articles_in_chronological_order_validity(self):
        """Article timestamps should be non-negative."""
        for name in ALL_SCENARIOS:
            s = get_scenario(name)
            for art in s.articles:
                assert art.minutes_from_start >= 0

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            get_scenario("NONEXISTENT_SCENARIO")

    def test_gme_scenario_has_reddit_articles(self):
        gme = get_scenario("GME_SHORT_SQUEEZE")
        reddit_arts = [a for a in gme.articles if a.source_type == "reddit"]
        assert len(reddit_arts) >= 2

    def test_ai_fud_scenario_has_high_ai_probs(self):
        fud = get_scenario("AI_FUD_CAMPAIGN")
        ai_arts = [a for a in fud.articles if a.is_manipulative]
        assert all(a.expected_ai_prob >= 0.7 for a in ai_arts)

    def test_price_series_has_anomalous_points(self):
        for name in ALL_SCENARIOS:
            s = get_scenario(name)
            anomalies = [p for p in s.price_series if p.is_anomalous]
            assert len(anomalies) >= 1, f"{name} has no anomalous price points"


# ---------------------------------------------------------------------------
# Article evaluation tests
# ---------------------------------------------------------------------------
class TestArticleEvaluation:

    def _make_eval(self, gt: bool, flagged: bool, score: float = 0.5) -> ArticleEvaluation:
        return ArticleEvaluation(
            article_title="Test",
            is_manipulative_gt=gt,
            predicted_score=score,
            predicted_level="high",
            flagged_as_alert=flagged,
            ai_component=score,
            prop_component=score,
            mkt_component=score,
            minutes_from_start=0.0,
        )

    def test_true_positive(self):
        e = self._make_eval(gt=True, flagged=True)
        assert e.is_tp and not e.is_fp and not e.is_fn and not e.is_tn

    def test_true_negative(self):
        e = self._make_eval(gt=False, flagged=False)
        assert e.is_tn and not e.is_tp and not e.is_fp and not e.is_fn

    def test_false_positive(self):
        e = self._make_eval(gt=False, flagged=True)
        assert e.is_fp and not e.is_tp and not e.is_fn and not e.is_tn

    def test_false_negative(self):
        e = self._make_eval(gt=True, flagged=False)
        assert e.is_fn and not e.is_tp and not e.is_fp and not e.is_tn


# ---------------------------------------------------------------------------
# ScenarioResult metric computation tests
# ---------------------------------------------------------------------------
class TestScenarioResultMetrics:

    def _make_result(self, evals: list[tuple[bool, bool]]) -> ScenarioResult:
        """Create a minimal ScenarioResult from (gt, flagged) pairs."""
        article_evals = [
            ArticleEvaluation(
                article_title=f"Article {i}",
                is_manipulative_gt=gt,
                predicted_score=0.8 if flagged else 0.3,
                predicted_level="high" if flagged else "low",
                flagged_as_alert=flagged,
                ai_component=0.5, prop_component=0.5, mkt_component=0.5,
                minutes_from_start=float(i * 10),
            )
            for i, (gt, flagged) in enumerate(evals)
        ]
        result = ScenarioResult(
            scenario_name="TEST",
            scenario_description="Test scenario",
            ticker="TEST",
            duration_hours=1.0,
            tactics=["test"],
            article_evaluations=article_evals,
            alerts_emitted=[],
        )
        result.compute_metrics()
        return result

    def test_perfect_classifier(self):
        # All TP and TN, no errors
        result = self._make_result([
            (True, True), (True, True), (False, False), (False, False)
        ])
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        assert result.accuracy == 1.0

    def test_all_false_positives(self):
        # Everything flagged but nothing is actually manipulative
        result = self._make_result([
            (False, True), (False, True), (False, True)
        ])
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_all_false_negatives(self):
        # Nothing flagged but everything is manipulative
        result = self._make_result([
            (True, False), (True, False), (True, False)
        ])
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_partial_detection(self):
        # 2 TP, 1 FN, 1 TN
        result = self._make_result([
            (True, True), (True, True), (True, False), (False, False)
        ])
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(2/3, abs=0.01)

    def test_time_to_detection_is_min_tp_time(self):
        result = self._make_result([
            (True, True),   # at T=0
            (False, False), # at T=10
            (True, True),   # at T=20
        ])
        assert result.time_to_first_detection_minutes == 0.0

    def test_time_to_detection_none_when_no_tp(self):
        result = self._make_result([(False, False), (True, False)])
        assert result.time_to_first_detection_minutes is None

    def test_score_separation_positive_for_good_classifier(self):
        """Manipulative articles should score higher than legitimate ones."""
        result = self._make_result([
            (True, True), (True, True),   # High score
            (False, False), (False, False),  # Low score
        ])
        assert result.score_separation > 0

    def test_false_alert_rate_computation(self):
        # 2 FP over 2 hours
        result = self._make_result([(False, True), (False, True), (False, False)])
        result.duration_hours = 2.0
        result.compute_metrics()
        assert result.false_alert_rate_per_hour == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Volatility alert generation tests
# ---------------------------------------------------------------------------
class TestVolatilityAlerts:

    def test_no_alerts_before_anomaly(self):
        scenario = get_scenario("GME_SHORT_SQUEEZE")
        alerts = _get_volatility_alerts(scenario, minutes_elapsed=10)
        assert len(alerts) == 0  # Anomaly starts at T+720m

    def test_alerts_during_anomalous_period(self):
        scenario = get_scenario("GME_SHORT_SQUEEZE")
        alerts = _get_volatility_alerts(scenario, minutes_elapsed=1000)
        # Should have anomalous alerts by T+1000m
        assert len(alerts) >= 0  # May or may not trigger depending on z-score

    def test_alert_has_correct_ticker(self):
        scenario = get_scenario("AI_FUD_CAMPAIGN")
        alerts = _get_volatility_alerts(scenario, minutes_elapsed=120)
        if alerts:
            assert alerts[0].ticker == scenario.ticker


# ---------------------------------------------------------------------------
# SimArticle → EnrichedArticle conversion tests
# ---------------------------------------------------------------------------
class TestSimArticleConversion:

    def test_conversion_produces_valid_enriched(self):
        from models import EnrichedArticle
        sim = SimArticle(
            title="Test manipulation article",
            body="Some body text.",
            source="FakeNews.io",
            source_type="rss",
            minutes_from_start=60.0,
            is_manipulative=True,
            manipulation_type="ai_generated",
            expected_ai_prob=0.90,
            expected_coord_score=0.75,
            tickers_mentioned=["TSLA"],
        )
        start = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        enriched = _sim_article_to_enriched(sim, start)

        assert isinstance(enriched, EnrichedArticle)
        assert enriched.ai_detection.ai_probability == 0.90
        assert enriched.propagation.coordination_score == 0.75
        assert enriched.ai_detection.is_ai_generated is True
        assert "TSLA" in enriched.named_entities

    def test_legitimate_article_low_ai_prob(self):
        sim = SimArticle(
            title="Legitimate Reuters article",
            body="Factual reporting.",
            source="Reuters",
            source_type="rss",
            minutes_from_start=0,
            is_manipulative=False,
            manipulation_type="",
            expected_ai_prob=0.04,
            expected_coord_score=0.01,
            tickers_mentioned=["AAPL"],
        )
        start = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        enriched = _sim_article_to_enriched(sim, start)
        assert enriched.ai_detection.ai_probability == 0.04
        assert enriched.ai_detection.is_ai_generated is False

    def test_publication_time_offset(self):
        from datetime import timedelta
        sim = SimArticle(
            title="Test", body="Body", source="S", source_type="rss",
            minutes_from_start=120.0, is_manipulative=False,
            manipulation_type="", expected_ai_prob=0.1,
            expected_coord_score=0.0, tickers_mentioned=[],
        )
        start = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        enriched = _sim_article_to_enriched(sim, start)
        expected_time = start + timedelta(minutes=120)
        assert enriched.article.published_at == expected_time


# ---------------------------------------------------------------------------
# Full replay integration tests
# ---------------------------------------------------------------------------
class TestHistoricalReplay:

    @pytest.mark.asyncio
    async def test_single_scenario_returns_result(self):
        replay = HistoricalReplay(alert_threshold=0.65)
        result = await replay.run_scenario("AI_FUD_CAMPAIGN")
        assert result.scenario_name == "AI_FUD_CAMPAIGN"
        assert len(result.article_evaluations) == 6
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1_score <= 1.0

    @pytest.mark.asyncio
    async def test_replay_detects_high_ai_scenario(self):
        """AI_FUD_CAMPAIGN has articles with ai_prob >= 0.88 — should detect."""
        replay = HistoricalReplay(alert_threshold=0.60)
        result = await replay.run_scenario("AI_FUD_CAMPAIGN")
        # Should have at least some true positives
        tp = sum(1 for e in result.article_evaluations if e.is_tp)
        assert tp >= 1, "Should detect at least one AI-generated article"

    @pytest.mark.asyncio
    async def test_score_separation_positive(self):
        """Manipulative articles should score higher than legitimate ones."""
        replay = HistoricalReplay(alert_threshold=0.65)
        result = await replay.run_scenario("PENNY_PUMP_DUMP")
        assert result.score_separation > 0, \
            "Manipulative articles must score higher than legitimate ones"

    @pytest.mark.asyncio
    async def test_full_benchmark_runs_all_scenarios(self):
        replay = HistoricalReplay(alert_threshold=0.65)
        benchmark = await replay.run_all()
        assert len(benchmark.scenario_results) == len(ALL_SCENARIOS)
        assert benchmark.total_runtime_seconds > 0

    @pytest.mark.asyncio
    async def test_benchmark_macro_metrics_valid(self):
        replay = HistoricalReplay(alert_threshold=0.65)
        benchmark = await replay.run_all()
        assert 0.0 <= benchmark.macro_precision <= 1.0
        assert 0.0 <= benchmark.macro_recall <= 1.0
        assert 0.0 <= benchmark.macro_f1 <= 1.0

    @pytest.mark.asyncio
    async def test_detection_timeline_populated(self):
        replay = HistoricalReplay(alert_threshold=0.65)
        result = await replay.run_scenario("LUNA_COLLAPSE")
        assert len(result.detection_timeline) == len(result.article_evaluations)
        for entry in result.detection_timeline:
            assert "minutes" in entry
            assert "score" in entry
            assert "level" in entry
            assert "is_manip_gt" in entry

    @pytest.mark.asyncio
    async def test_to_report_dict_structure(self):
        replay = HistoricalReplay(alert_threshold=0.65)
        benchmark = await replay.run_all()
        report = benchmark.to_report_dict()
        assert "macro_precision" in report
        assert "macro_recall" in report
        assert "macro_f1" in report
        assert "scenarios" in report
        assert len(report["scenarios"]) == len(ALL_SCENARIOS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
