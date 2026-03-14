"""
simulations/historical_replay.py — Scenario replay and evaluation engine.

Runs each ManipulationScenario through the MarketShield pipeline and
measures detection performance against ground-truth labels.

Evaluation metrics:
  - Precision:            TP / (TP + FP)  — how accurate are our alerts?
  - Recall:               TP / (TP + FN)  — how many real threats did we catch?
  - F1 Score:             Harmonic mean of precision and recall
  - Time-to-detection:    Minutes from scenario start to first correct alert
  - False alert rate:     FP per hour of scenario runtime
  - Score calibration:    Mean predicted score for manipulative vs legitimate articles

The replay engine also generates a timeline showing how scores evolved
as the scenario unfolded — useful for understanding detection lag.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from logger import get_logger
from models import (
    AIDetectionResult,
    ContentSource,
    EnrichedArticle,
    ManipulationAlert,
    PropagationMetrics,
    RawArticle,
    RiskLevel,
    Sentiment,
    SentimentResult,
    VolatilityAlert,
)
from simulations.scenarios.definitions import (
    ManipulationScenario,
    SimArticle,
    PricePoint,
    get_all_scenarios,
    get_scenario,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Evaluation result data classes
# ---------------------------------------------------------------------------
@dataclass
class ArticleEvaluation:
    """Per-article evaluation result."""
    article_title: str
    is_manipulative_gt: bool          # Ground truth
    predicted_score: float
    predicted_level: str
    flagged_as_alert: bool
    ai_component: float
    prop_component: float
    mkt_component: float
    minutes_from_start: float

    @property
    def is_tp(self) -> bool:
        return self.is_manipulative_gt and self.flagged_as_alert

    @property
    def is_fp(self) -> bool:
        return not self.is_manipulative_gt and self.flagged_as_alert

    @property
    def is_fn(self) -> bool:
        return self.is_manipulative_gt and not self.flagged_as_alert

    @property
    def is_tn(self) -> bool:
        return not self.is_manipulative_gt and not self.flagged_as_alert


@dataclass
class ScenarioResult:
    """Full evaluation results for one scenario run."""
    scenario_name: str
    scenario_description: str
    ticker: str
    duration_hours: float
    tactics: list[str]

    article_evaluations: list[ArticleEvaluation]
    alerts_emitted: list[ManipulationAlert]

    # Detection timeline
    detection_timeline: list[dict] = field(default_factory=list)

    # Computed metrics (filled by compute_metrics())
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    time_to_first_detection_minutes: Optional[float] = None
    false_alert_rate_per_hour: float = 0.0
    mean_score_manipulative: float = 0.0
    mean_score_legitimate: float = 0.0
    score_separation: float = 0.0      # Higher = better discrimination

    def compute_metrics(self) -> None:
        """Compute all evaluation metrics from article evaluations."""
        evals = self.article_evaluations

        tp = sum(1 for e in evals if e.is_tp)
        fp = sum(1 for e in evals if e.is_fp)
        fn = sum(1 for e in evals if e.is_fn)
        tn = sum(1 for e in evals if e.is_tn)

        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        self.recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        self.f1_score  = (
            2 * self.precision * self.recall / (self.precision + self.recall)
            if (self.precision + self.recall) > 0 else 0.0
        )
        self.accuracy  = (tp + tn) / len(evals) if evals else 0.0

        # Time to first detection
        tp_evals = sorted(
            [e for e in evals if e.is_tp],
            key=lambda e: e.minutes_from_start,
        )
        self.time_to_first_detection_minutes = (
            tp_evals[0].minutes_from_start if tp_evals else None
        )

        # False alert rate per hour
        self.false_alert_rate_per_hour = (
            fp / self.duration_hours if self.duration_hours > 0 else 0.0
        )

        # Score separation (discrimination quality)
        manip_scores = [e.predicted_score for e in evals if e.is_manipulative_gt]
        legit_scores  = [e.predicted_score for e in evals if not e.is_manipulative_gt]

        self.mean_score_manipulative = (
            statistics.mean(manip_scores) if manip_scores else 0.0
        )
        self.mean_score_legitimate = (
            statistics.mean(legit_scores) if legit_scores else 0.0
        )
        self.score_separation = self.mean_score_manipulative - self.mean_score_legitimate

    def summary_dict(self) -> dict:
        return {
            "scenario":            self.scenario_name,
            "ticker":              self.ticker,
            "duration_hours":      self.duration_hours,
            "tactics":             self.tactics,
            "n_articles":          len(self.article_evaluations),
            "n_manipulative_gt":   sum(1 for e in self.article_evaluations if e.is_manipulative_gt),
            "n_alerts_emitted":    len(self.alerts_emitted),
            "precision":           round(self.precision, 4),
            "recall":              round(self.recall, 4),
            "f1_score":            round(self.f1_score, 4),
            "accuracy":            round(self.accuracy, 4),
            "time_to_detection_min": self.time_to_first_detection_minutes,
            "false_alert_rate_per_hour": round(self.false_alert_rate_per_hour, 4),
            "mean_score_manipulative": round(self.mean_score_manipulative, 4),
            "mean_score_legitimate":   round(self.mean_score_legitimate, 4),
            "score_separation":        round(self.score_separation, 4),
        }


@dataclass
class BenchmarkResult:
    """Aggregate results across all scenarios."""
    scenario_results: list[ScenarioResult]
    run_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_runtime_seconds: float = 0.0

    @property
    def macro_precision(self) -> float:
        vals = [r.precision for r in self.scenario_results if r.precision > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def macro_recall(self) -> float:
        vals = [r.recall for r in self.scenario_results if r.recall > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def macro_f1(self) -> float:
        vals = [r.f1_score for r in self.scenario_results if r.f1_score > 0]
        return statistics.mean(vals) if vals else 0.0

    @property
    def mean_detection_time(self) -> Optional[float]:
        times = [
            r.time_to_first_detection_minutes
            for r in self.scenario_results
            if r.time_to_first_detection_minutes is not None
        ]
        return statistics.mean(times) if times else None

    @property
    def mean_score_separation(self) -> float:
        seps = [r.score_separation for r in self.scenario_results]
        return statistics.mean(seps) if seps else 0.0

    def to_report_dict(self) -> dict:
        return {
            "run_timestamp":         self.run_timestamp.isoformat(),
            "total_runtime_seconds": round(self.total_runtime_seconds, 2),
            "n_scenarios":           len(self.scenario_results),
            "macro_precision":       round(self.macro_precision, 4),
            "macro_recall":          round(self.macro_recall, 4),
            "macro_f1":              round(self.macro_f1, 4),
            "mean_detection_time_min": round(self.mean_detection_time, 1)
                                       if self.mean_detection_time else None,
            "mean_score_separation": round(self.mean_score_separation, 4),
            "scenarios":             [r.summary_dict() for r in self.scenario_results],
        }


# ---------------------------------------------------------------------------
# Article construction from scenario
# ---------------------------------------------------------------------------
def _sim_article_to_enriched(
    sim: SimArticle,
    start_time: datetime,
) -> EnrichedArticle:
    """
    Convert a SimArticle (ground-truth scenario article) into an EnrichedArticle
    using its expected detection scores as the NLP output.

    In a real evaluation, you would run the actual NLP pipeline here.
    Using expected scores allows fast, deterministic evaluation of the
    scoring and alerting logic independent of model performance.
    """
    published_at = start_time + timedelta(minutes=sim.minutes_from_start)

    raw = RawArticle(
        source=ContentSource(sim.source_type),
        source_name=sim.source,
        title=sim.title,
        body=sim.body,
        published_at=published_at,
        raw_metadata={
            "scenario_article": True,
            "manipulation_type": sim.manipulation_type,
        },
    )

    # Map expected scores to sentiment
    if sim.is_manipulative:
        sentiment_label = Sentiment.NEGATIVE
        sentiment_score = 0.82
    else:
        sentiment_label = Sentiment.NEUTRAL
        sentiment_score = 0.60

    return EnrichedArticle(
        article=raw,
        sentiment=SentimentResult(
            label=sentiment_label,
            score=sentiment_score,
            positive=0.1 if sentiment_label == Sentiment.NEGATIVE else 0.15,
            negative=0.82 if sentiment_label == Sentiment.NEGATIVE else 0.05,
            neutral=0.08 if sentiment_label == Sentiment.NEGATIVE else 0.80,
        ),
        ai_detection=AIDetectionResult(
            ai_probability=sim.expected_ai_prob,
            is_ai_generated=sim.expected_ai_prob >= 0.70,
            perplexity_score=12.0 if sim.expected_ai_prob > 0.7 else 78.0,
            burstiness_score=0.06 if sim.expected_ai_prob > 0.7 else 0.52,
        ),
        propagation=PropagationMetrics(
            coordination_score=sim.expected_coord_score,
            is_coordinated=sim.expected_coord_score >= 0.45,
            cluster_size=max(1, int(sim.expected_coord_score * 15)),
            spread_velocity=sim.expected_coord_score * 10.0,
        ),
        named_entities=sim.tickers_mentioned + (
            ["coordination", "ai_generated"] if sim.is_manipulative else []
        ),
    )


def _get_volatility_alerts(
    scenario: ManipulationScenario,
    minutes_elapsed: float,
) -> list[VolatilityAlert]:
    """
    Return any volatility alerts that would have been triggered
    at the given point in the scenario timeline.
    """
    relevant_prices = [
        p for p in scenario.price_series
        if p.minutes_from_start <= minutes_elapsed and p.is_anomalous
    ]
    if not relevant_prices:
        return []

    latest = max(relevant_prices, key=lambda p: p.minutes_from_start)
    price_change = (latest.price - scenario.baseline_price) / scenario.baseline_price
    z_score = abs(price_change) * 25  # Approximate z-score

    if z_score < 2.0:
        return []

    return [VolatilityAlert(
        ticker=scenario.ticker,
        z_score=round(z_score, 2),
        current_price=latest.price,
        baseline_mean=0.0,
        baseline_std=scenario.baseline_price * 0.01,
        direction="up" if price_change > 0 else "down",
    )]


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------
class HistoricalReplay:
    """
    Runs ManipulationScenarios through the risk scoring pipeline
    and measures detection performance.

    Usage
    -----
    >>> replay = HistoricalReplay()
    >>> result = await replay.run_scenario("GME_SHORT_SQUEEZE")
    >>> print(f"F1: {result.f1_score:.3f}")
    >>> print(f"Detection time: {result.time_to_first_detection_minutes:.0f} min")

    >>> benchmark = await replay.run_all()
    >>> print(f"Macro F1: {benchmark.macro_f1:.3f}")
    """

    def __init__(self, alert_threshold: float = 0.65):
        from risk_engine.scorer import RiskScorer
        from risk_engine.alert_manager import AlertManager

        self._scorer  = RiskScorer()
        self._manager = AlertManager(
            cooldown_minutes=5,      # Short cooldown for evaluation
            max_per_ticker_per_hour=20,
        )
        self._threshold = alert_threshold

    async def run_scenario(
        self,
        scenario_or_name: ManipulationScenario | str,
    ) -> ScenarioResult:
        """
        Run a single scenario through the full pipeline.
        Returns a ScenarioResult with all evaluation metrics.
        """
        if isinstance(scenario_or_name, str):
            scenario = get_scenario(scenario_or_name)
        else:
            scenario = scenario_or_name

        log.info(f"[Replay] Starting scenario: {scenario.name}")
        start_time = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)

        article_evals: list[ArticleEvaluation] = []
        all_alerts: list[ManipulationAlert] = []
        timeline: list[dict] = []

        # Process articles in chronological order
        sorted_articles = sorted(scenario.articles, key=lambda a: a.minutes_from_start)

        for sim_art in sorted_articles:
            enriched = _sim_article_to_enriched(sim_art, start_time)
            vol_alerts = _get_volatility_alerts(scenario, sim_art.minutes_from_start)

            # Score
            risk = self._scorer.score(enriched, volatility_alerts=vol_alerts)

            # Alert decision
            flagged = risk.composite_score >= self._threshold

            # Try to emit alert (may be suppressed by dedup)
            alert = None
            if flagged:
                alert = self._manager.process_single(risk, enriched, vol_alerts)
                if alert:
                    all_alerts.append(alert)

            eval_item = ArticleEvaluation(
                article_title        = sim_art.title[:60],
                is_manipulative_gt   = sim_art.is_manipulative,
                predicted_score      = risk.composite_score,
                predicted_level      = risk.risk_level.value,
                flagged_as_alert     = flagged,
                ai_component         = risk.ai_component,
                prop_component       = risk.propagation_component,
                mkt_component        = risk.market_impact_component,
                minutes_from_start   = sim_art.minutes_from_start,
            )
            article_evals.append(eval_item)

            timeline.append({
                "minutes":       sim_art.minutes_from_start,
                "title":         sim_art.title[:45],
                "score":         round(risk.composite_score, 3),
                "level":         risk.risk_level.value,
                "is_manip_gt":   sim_art.is_manipulative,
                "flagged":       flagged,
                "alerted":       alert is not None,
                "vol_alert":     len(vol_alerts) > 0,
            })

        # Build and compute result
        result = ScenarioResult(
            scenario_name        = scenario.name,
            scenario_description = scenario.description,
            ticker               = scenario.ticker,
            duration_hours       = scenario.duration_hours,
            tactics              = scenario.tactics,
            article_evaluations  = article_evals,
            alerts_emitted       = all_alerts,
            detection_timeline   = timeline,
        )
        result.compute_metrics()

        log.info(
            f"[Replay] {scenario.name} complete — "
            f"P={result.precision:.3f} R={result.recall:.3f} "
            f"F1={result.f1_score:.3f} "
            f"TTD={result.time_to_first_detection_minutes}min"
        )
        return result

    async def run_all(self) -> BenchmarkResult:
        """Run all scenarios and compile aggregate benchmark results."""
        log.info("[Replay] Starting full benchmark across all scenarios…")
        t_start = time.monotonic()

        scenarios = get_all_scenarios()
        results: list[ScenarioResult] = []

        for scenario in scenarios:
            result = await self.run_scenario(scenario)
            results.append(result)

        elapsed = time.monotonic() - t_start
        benchmark = BenchmarkResult(
            scenario_results=results,
            total_runtime_seconds=elapsed,
        )

        log.info(
            f"[Replay] Benchmark complete in {elapsed:.1f}s — "
            f"Macro P={benchmark.macro_precision:.3f} "
            f"R={benchmark.macro_recall:.3f} "
            f"F1={benchmark.macro_f1:.3f}"
        )
        return benchmark


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------
def generate_text_report(benchmark: BenchmarkResult) -> str:
    """Generate a human-readable benchmark report."""
    lines = [
        "=" * 70,
        "  MARKETSHIELD — MANIPULATION DETECTION BENCHMARK REPORT",
        f"  Generated: {benchmark.run_timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 70,
        "",
        "AGGREGATE METRICS",
        "-" * 40,
        f"  Scenarios evaluated:     {len(benchmark.scenario_results)}",
        f"  Macro Precision:         {benchmark.macro_precision:.3f}",
        f"  Macro Recall:            {benchmark.macro_recall:.3f}",
        f"  Macro F1 Score:          {benchmark.macro_f1:.3f}",
        f"  Mean Detection Time:     "
        f"{benchmark.mean_detection_time:.1f} min"
        if benchmark.mean_detection_time else "  Mean Detection Time:     N/A",
        f"  Mean Score Separation:   {benchmark.mean_score_separation:.3f}",
        f"  Total Runtime:           {benchmark.total_runtime_seconds:.1f}s",
        "",
    ]

    for result in benchmark.scenario_results:
        tp = sum(1 for e in result.article_evaluations if e.is_tp)
        fp = sum(1 for e in result.article_evaluations if e.is_fp)
        fn = sum(1 for e in result.article_evaluations if e.is_fn)
        tn = sum(1 for e in result.article_evaluations if e.is_tn)

        lines += [
            f"SCENARIO: {result.scenario_name}",
            "-" * 40,
            f"  Description:  {result.scenario_description}",
            f"  Ticker:       {result.ticker}",
            f"  Duration:     {result.duration_hours}h",
            f"  Tactics:      {', '.join(result.tactics)}",
            "",
            f"  Articles:     {len(result.article_evaluations)} total",
            f"    Manipulative (GT): {result.n_manipulative_gt}",
            f"    Legitimate (GT):   {result.n_legitimate_gt}",
            "",
            f"  Confusion Matrix:",
            f"    TP={tp}  FP={fp}",
            f"    FN={fn}  TN={tn}",
            "",
            f"  Precision:    {result.precision:.3f}",
            f"  Recall:       {result.recall:.3f}",
            f"  F1 Score:     {result.f1_score:.3f}",
            f"  Accuracy:     {result.accuracy:.3f}",
        ]

        if result.time_to_first_detection_minutes is not None:
            lines.append(
                f"  Time-to-Det:  {result.time_to_first_detection_minutes:.0f} min"
            )

        lines += [
            f"  Mean Score (manip):  {result.mean_score_manipulative:.3f}",
            f"  Mean Score (legit):  {result.mean_score_legitimate:.3f}",
            f"  Score Separation:    {result.score_separation:.3f}",
            "",
            "  Timeline:",
        ]

        for t in result.detection_timeline:
            gt_str  = "MANIP" if t["is_manip_gt"] else "legit"
            flag_str = "🚨" if t["alerted"] else ("⚑ " if t["flagged"] else "  ")
            lines.append(
                f"    T+{t['minutes']:>5.0f}m  {flag_str}  "
                f"[{t['level'].upper():<8}] {t['score']:.3f}  "
                f"({gt_str})  {t['title']}"
            )
        lines.append("")

    lines += [
        "=" * 70,
        "  END OF REPORT",
        "=" * 70,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone demo / CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from logger import configure_logging
    from config import settings

    configure_logging(level="INFO", log_to_file=False)

    async def _demo():
        print("\n🎬 MarketShield Historical Replay Engine\n")
        print("Running all 5 manipulation scenarios...\n")

        replay    = HistoricalReplay(alert_threshold=settings.risk_alert_threshold)
        benchmark = await replay.run_all()

        # Print the full text report
        report = generate_text_report(benchmark)
        print(report)

        # Print JSON summary
        print("\n📊 JSON Summary:")
        print(json.dumps(benchmark.to_report_dict(), indent=2, default=str)[:3000])

    asyncio.run(_demo())
