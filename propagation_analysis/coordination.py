"""
propagation_analysis/coordination.py — Coordinated inauthentic behaviour detector.

Coordinated inauthentic behaviour (CIB) is when multiple accounts/sources
act together to amplify a narrative while concealing their coordination.

Classic patterns in financial manipulation:
  1. Burst posting: dozens of near-identical posts within minutes
  2. Cross-platform amplification: same narrative on Reddit + news sites simultaneously
  3. New account surge: accounts created shortly before a campaign
  4. Unusual timing correlation: posts cluster around specific market hours
  5. Network hubs: a few "seed" sources that are disproportionately copied

Detection methods implemented:

  A. Temporal burst detection
     - Sliding window over publication timestamps
     - Flag windows where article density is >> baseline
     - Metric: z-score of articles-per-minute vs rolling baseline

  B. Source correlation matrix
     - Measure how often source_A and source_B publish similar content
       within the same time window
     - High correlation between sources that shouldn't be related = suspicious

  C. Velocity anomaly
     - Track how fast a cluster grows: velocity spike = coordination signal

  D. Content-time fingerprint
     - Measure edit distance between (content, timestamp) pairs
     - Very similar content posted within seconds of each other
       across different platforms = bot behaviour

Output: CoordinationReport per cluster, with 0–1 coordination_score
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from config import settings
from logger import get_logger
from models import EnrichedArticle, PropagationMetrics
from propagation_analysis.deduplicator import ArticleCluster

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BurstWindow:
    """A time window with anomalously high article volume."""
    start: datetime
    end: datetime
    article_count: int
    baseline_rate: float          # Expected articles/minute at this time
    actual_rate: float            # Observed articles/minute
    z_score: float
    article_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60


@dataclass
class CoordinationReport:
    """
    Full coordination analysis for a cluster of articles.
    coordination_score is the primary output consumed by the risk engine.
    """
    cluster_id: str
    coordination_score: float           # 0–1 (higher = more coordinated)
    is_coordinated: bool

    # Contributing signals
    burst_score: float                  # Temporal burst signal
    source_diversity_score: float       # Low diversity = suspicious
    cross_platform_score: float         # High cross-platform = suspicious
    velocity_score: float               # Velocity anomaly signal

    # Evidence
    burst_windows: list[BurstWindow] = field(default_factory=list)
    suspicious_source_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    posting_timeline: list[dict] = field(default_factory=list)

    explanation: str = ""


# ---------------------------------------------------------------------------
# Temporal burst detection
# ---------------------------------------------------------------------------
def _compute_burst_score(
    timestamps: list[datetime],
    window_minutes: float = 10.0,
    z_threshold: float = 2.5,
) -> tuple[float, list[BurstWindow]]:
    """
    Detect abnormal posting bursts using a sliding window z-score.

    Parameters
    ----------
    timestamps     : Sorted list of article publication times
    window_minutes : Size of the sliding window
    z_threshold    : Z-score threshold for flagging a burst

    Returns
    -------
    (burst_score, burst_windows)
    burst_score: 0–1 (fraction of articles falling in flagged windows)
    """
    if len(timestamps) < 3:
        return 0.0, []

    timestamps = sorted(timestamps)
    window = timedelta(minutes=window_minutes)

    # Count articles in each window
    rates: list[float] = []
    windows: list[tuple[datetime, datetime, list[datetime]]] = []

    for i, ts in enumerate(timestamps):
        window_end = ts + window
        in_window = [t for t in timestamps if ts <= t <= window_end]
        rate = len(in_window) / window_minutes  # articles/minute
        rates.append(rate)
        windows.append((ts, window_end, in_window))

    rates_arr = np.array(rates)
    mean_rate = float(np.mean(rates_arr))
    std_rate = float(np.std(rates_arr))

    if std_rate < 1e-8:
        return 0.0, []

    burst_windows: list[BurstWindow] = []
    flagged_ids: set[int] = set()

    for i, (start, end, in_window) in enumerate(windows):
        z = (rates[i] - mean_rate) / std_rate
        if z >= z_threshold:
            burst_windows.append(BurstWindow(
                start=start,
                end=end,
                article_count=len(in_window),
                baseline_rate=mean_rate,
                actual_rate=rates[i],
                z_score=round(z, 2),
            ))
            flagged_ids.add(i)

    if not burst_windows:
        return 0.0, []

    burst_score = len(flagged_ids) / len(timestamps)
    return round(min(1.0, burst_score * 2), 4), burst_windows


# ---------------------------------------------------------------------------
# Source diversity analysis
# ---------------------------------------------------------------------------
def _source_diversity_score(articles: list[EnrichedArticle]) -> float:
    """
    Measure how diverse the sources are.

    Low diversity = same few sources posting the same content = suspicious.

    Score: 0 = all one source (maximally suspicious)
           1 = each article from a unique source (normal)
    """
    if not articles:
        return 1.0

    sources = [a.article.source_name for a in articles]
    n = len(sources)
    unique = len(set(sources))

    if n == 1:
        return 1.0

    # Simpson's diversity index
    counts = Counter(sources)
    D = sum(c * (c - 1) for c in counts.values())
    N = n * (n - 1)
    diversity = 1.0 - D / N if N > 0 else 1.0

    return round(diversity, 4)


# ---------------------------------------------------------------------------
# Cross-platform analysis
# ---------------------------------------------------------------------------
def _cross_platform_score(articles: list[EnrichedArticle]) -> float:
    """
    Measure how many different platforms (RSS, Reddit, Twitter) are
    pushing the same narrative simultaneously.

    High cross-platform activity within a short window is a CIB signal —
    organic news typically propagates with delays between platforms.
    Coordinated campaigns appear on all platforms at once.

    Score: 0 = single platform (low suspicion)
           1 = all platforms within minutes (high suspicion)
    """
    if not articles:
        return 0.0

    platforms = set(a.article.source.value for a in articles)
    num_platforms = len(platforms)

    if num_platforms <= 1:
        return 0.0

    # Time span of cross-platform activity
    times = [a.article.published_at for a in articles]
    span_minutes = (max(times) - min(times)).total_seconds() / 60

    # Multi-platform within 30 minutes = very suspicious
    if span_minutes <= 30:
        platform_score = (num_platforms - 1) / 2  # max 3 platforms → 1.0
    elif span_minutes <= 120:
        platform_score = (num_platforms - 1) / 4
    else:
        platform_score = 0.0

    return round(min(1.0, platform_score), 4)


# ---------------------------------------------------------------------------
# Velocity anomaly scoring
# ---------------------------------------------------------------------------
def _velocity_anomaly_score(cluster: ArticleCluster) -> float:
    """
    Convert cluster spread velocity to a 0–1 anomaly score.

    Calibration (based on typical financial news spread patterns):
      < 1.0 articles/hour : normal organic spread → score 0.0–0.2
      1–5 articles/hour   : elevated activity    → score 0.2–0.5
      5–20 articles/hour  : high coordination    → score 0.5–0.8
      > 20 articles/hour  : extreme coordination → score 0.8–1.0
    """
    v = cluster.spread_velocity
    if v <= 0:
        return 0.0
    # Logistic curve calibrated to the ranges above
    return round(1.0 / (1.0 + math.exp(-0.3 * (v - 5))), 4)


# ---------------------------------------------------------------------------
# Content-time fingerprinting
# ---------------------------------------------------------------------------
def _simple_edit_distance_ratio(a: str, b: str) -> float:
    """
    Fast normalised edit distance approximation using character bigrams.
    Full Levenshtein is O(mn) — too slow for large batches.
    """
    def bigrams(s: str) -> Counter:
        s = s.lower()[:200]
        return Counter(s[i:i+2] for i in range(len(s) - 1))

    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a or not bg_b:
        return 0.0

    intersection = sum((bg_a & bg_b).values())
    union = sum((bg_a | bg_b).values())
    return intersection / union if union > 0 else 0.0


def _detect_identical_timing(
    articles: list[EnrichedArticle],
    time_threshold_seconds: float = 60.0,
    similarity_threshold: float = 0.6,
) -> list[tuple[str, str, float]]:
    """
    Find pairs of articles that are both:
      1. Published within ``time_threshold_seconds`` of each other
      2. Have text similarity ≥ ``similarity_threshold``

    These are the strongest signal of bot/coordinated behaviour.
    Returns list of (article_id_a, article_id_b, similarity_score).
    """
    suspicious_pairs: list[tuple[str, str, float]] = []
    n = len(articles)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = articles[i], articles[j]
            time_diff = abs(
                (a.article.published_at - b.article.published_at).total_seconds()
            )
            if time_diff > time_threshold_seconds:
                continue

            # Same source → skip (not suspicious, just republished)
            if a.article.source_name == b.article.source_name:
                continue

            sim = _simple_edit_distance_ratio(
                a.article.title + " " + a.article.body[:100],
                b.article.title + " " + b.article.body[:100],
            )
            if sim >= similarity_threshold:
                suspicious_pairs.append((a.article.id, b.article.id, sim))

    return suspicious_pairs


# ---------------------------------------------------------------------------
# Core coordinator
# ---------------------------------------------------------------------------
class CoordinationDetector:
    """
    Detects coordinated inauthentic behaviour in article clusters.

    Usage
    -----
    >>> detector = CoordinationDetector()
    >>> reports = detector.analyse_clusters(clusters, articles)
    >>> for report in reports:
    ...     if report.is_coordinated:
    ...         print(f"Coordinated cluster: {report.coordination_score:.2f}")
    ...         print(report.explanation)

    Also provides a global sweep of all articles:
    >>> global_report = detector.analyse_all(articles)
    """

    # Score threshold for classifying as coordinated
    COORDINATION_THRESHOLD = 0.45

    def __init__(
        self,
        burst_window_minutes: float = 10.0,
        burst_z_threshold: float = 2.5,
    ):
        self.burst_window = burst_window_minutes
        self.burst_z = burst_z_threshold

    def analyse_cluster(
        self,
        cluster: ArticleCluster,
        all_articles: list[EnrichedArticle],
    ) -> CoordinationReport:
        """
        Analyse one cluster for coordinated behaviour.

        Combines 4 independent signals into a weighted coordination_score.
        """
        # Get article objects for this cluster
        art_lookup = {a.article.id: a for a in all_articles}
        cluster_articles = [
            art_lookup[aid]
            for aid in cluster.article_ids
            if aid in art_lookup
        ]

        if not cluster_articles:
            return CoordinationReport(
                cluster_id=cluster.cluster_id,
                coordination_score=0.0,
                is_coordinated=False,
                burst_score=0.0,
                source_diversity_score=1.0,
                cross_platform_score=0.0,
                velocity_score=0.0,
            )

        # ── Signal A: Temporal burst ───────────────────────────────────────
        timestamps = [a.article.published_at for a in cluster_articles]
        burst_score, burst_windows = _compute_burst_score(
            timestamps, self.burst_window, self.burst_z
        )

        # ── Signal B: Source diversity ─────────────────────────────────────
        diversity = _source_diversity_score(cluster_articles)
        # Invert: low diversity = high suspicion
        diversity_signal = 1.0 - diversity

        # ── Signal C: Cross-platform ───────────────────────────────────────
        cross_platform = _cross_platform_score(cluster_articles)

        # ── Signal D: Velocity anomaly ─────────────────────────────────────
        velocity = _velocity_anomaly_score(cluster)

        # ── Suspicious timing pairs ────────────────────────────────────────
        suspicious_pairs = _detect_identical_timing(cluster_articles)

        # ── Ensemble score ─────────────────────────────────────────────────
        # Weights calibrated for financial manipulation detection
        coordination_score = (
            0.30 * burst_score
            + 0.25 * diversity_signal
            + 0.20 * cross_platform
            + 0.25 * velocity
        )

        # Bonus for suspicious timing pairs (strong evidence)
        if suspicious_pairs:
            coordination_score = min(1.0, coordination_score + 0.15 * len(suspicious_pairs))

        coordination_score = round(float(np.clip(coordination_score, 0.0, 1.0)), 4)
        is_coordinated = coordination_score >= self.COORDINATION_THRESHOLD

        # ── Generate explanation ───────────────────────────────────────────
        explanation = self._explain(
            cluster, burst_score, diversity_signal,
            cross_platform, velocity, suspicious_pairs, coordination_score,
        )

        # ── Build timeline ─────────────────────────────────────────────────
        timeline = sorted([
            {
                "time": a.article.published_at.isoformat(),
                "source": a.article.source_name,
                "title": a.article.title[:60],
                "ai_prob": a.ai_detection.ai_probability if a.ai_detection else 0.0,
            }
            for a in cluster_articles
        ], key=lambda x: x["time"])

        return CoordinationReport(
            cluster_id=cluster.cluster_id,
            coordination_score=coordination_score,
            is_coordinated=is_coordinated,
            burst_score=burst_score,
            source_diversity_score=diversity,
            cross_platform_score=cross_platform,
            velocity_score=velocity,
            burst_windows=burst_windows,
            suspicious_source_pairs=suspicious_pairs,
            posting_timeline=timeline,
            explanation=explanation,
        )

    def analyse_clusters(
        self,
        clusters: list[ArticleCluster],
        all_articles: list[EnrichedArticle],
    ) -> list[CoordinationReport]:
        """Analyse all clusters (skips singletons)."""
        reports = []
        for cluster in clusters:
            if cluster.size < 2:
                continue
            report = self.analyse_cluster(cluster, all_articles)
            reports.append(report)

        flagged = sum(1 for r in reports if r.is_coordinated)
        log.info(
            f"[Coordination] Analysed {len(reports)} clusters → "
            f"{flagged} flagged as coordinated"
        )
        return reports

    def analyse_all(
        self,
        articles: list[EnrichedArticle],
    ) -> dict:
        """
        Global coordination sweep — doesn't require pre-clustered data.
        Looks for burst patterns and suspicious timing across all articles.

        Useful as a fast first-pass scan before full cluster analysis.
        """
        if not articles:
            return {"coordination_score": 0.0, "burst_windows": []}

        timestamps = [a.article.published_at for a in articles]
        burst_score, burst_windows = _compute_burst_score(
            timestamps, self.burst_window, self.burst_z
        )

        suspicious_pairs = _detect_identical_timing(articles)
        cross_platform = _cross_platform_score(articles)

        global_score = (
            0.40 * burst_score
            + 0.30 * cross_platform
            + 0.30 * min(1.0, len(suspicious_pairs) / max(1, len(articles)) * 10)
        )

        return {
            "coordination_score": round(float(global_score), 4),
            "burst_score": burst_score,
            "cross_platform_score": cross_platform,
            "suspicious_pair_count": len(suspicious_pairs),
            "burst_windows": len(burst_windows),
            "article_count": len(articles),
        }

    def update_article_metrics(
        self,
        articles: list[EnrichedArticle],
        reports: list[CoordinationReport],
    ) -> None:
        """
        Write coordination scores back into each article's PropagationMetrics.
        Call this after analyse_clusters() to update the enriched articles.
        """
        cluster_scores: dict[str, float] = {
            r.cluster_id: r.coordination_score for r in reports
        }
        cluster_flags: dict[str, bool] = {
            r.cluster_id: r.is_coordinated for r in reports
        }

        for art in articles:
            if art.propagation and art.propagation.duplicate_cluster_id:
                cid = art.propagation.duplicate_cluster_id
                art.propagation.coordination_score = cluster_scores.get(cid, 0.0)
                art.propagation.is_coordinated = cluster_flags.get(cid, False)

    @staticmethod
    def _explain(
        cluster: ArticleCluster,
        burst_score: float,
        diversity_signal: float,
        cross_platform: float,
        velocity: float,
        suspicious_pairs: list,
        final_score: float,
    ) -> str:
        """Generate a human-readable explanation of coordination signals."""
        parts = []

        if burst_score > 0.3:
            parts.append(
                f"Temporal burst detected (score={burst_score:.2f}): "
                f"{cluster.size} articles within a tight time window"
            )
        if diversity_signal > 0.5:
            parts.append(
                f"Low source diversity (signal={diversity_signal:.2f}): "
                f"narrative originates from {len(set(cluster.sources))} sources"
            )
        if cross_platform > 0.3:
            parts.append(
                f"Cross-platform amplification (score={cross_platform:.2f}): "
                f"simultaneous spread across multiple platforms"
            )
        if velocity > 0.5:
            parts.append(
                f"High spread velocity (score={velocity:.2f}): "
                f"{cluster.spread_velocity:.1f} articles/hour"
            )
        if suspicious_pairs:
            parts.append(
                f"{len(suspicious_pairs)} pairs of near-identical articles "
                f"published within 60 seconds from different sources"
            )

        if not parts:
            return "No coordination signals detected."

        return (
            f"Coordination score: {final_score:.2f} — "
            + "; ".join(parts)
        )


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta
    from models import ContentSource, RawArticle, EnrichedArticle, PropagationMetrics
    from models import SentimentResult, Sentiment, AIDetectionResult
    from propagation_analysis.deduplicator import SemanticDeduplicator

    def _make(title, body, source, minutes_ago, ai_prob=0.1):
        raw = RawArticle(
            source=ContentSource.RSS, source_name=source,
            title=title, body=body,
            published_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
            raw_metadata={},
        )
        return EnrichedArticle(
            article=raw,
            sentiment=SentimentResult(label=Sentiment.NEGATIVE, score=0.9,
                                      positive=0.05, negative=0.9, neutral=0.05),
            ai_detection=AIDetectionResult(
                ai_probability=ai_prob, is_ai_generated=ai_prob > 0.7,
                perplexity_score=22.0, burstiness_score=0.08),
            propagation=PropagationMetrics(),
            named_entities=["TSLA", "Elon Musk"],
        )

    # Simulate coordinated campaign: 5 near-identical articles in 8 minutes
    ARTICLES = [
        _make("BREAKING: Elon Musk arrested for fraud by FBI",
              "Tesla CEO taken into custody by federal agents early this morning.",
              "FakeCryptoNews.io", 8, 0.92),
        _make("BREAKING NEWS: Musk detained federal fraud charges",
              "FBI agents have detained Tesla founder Elon Musk this morning.",
              "StockManipulator.net", 6, 0.89),
        _make("URGENT: Elon Musk faces arrest securities fraud",
              "Federal authorities arrest Tesla chief Musk for fraud.",
              "PumpAndDump.xyz", 5, 0.87),
        _make("FBI arrests Elon Musk for securities violations",
              "Musk taken into custody on charges of securities fraud.",
              "FakeNews4.io", 3, 0.91),
        _make("Musk ARRESTED federal agents fraud allegations",
              "Tesla CEO under arrest following federal investigation into fraud.",
              "CoordiBot5.net", 1, 0.88),
        # Legitimate article
        _make("Tesla (TSLA) stock volatility increases amid rumours",
              "TSLA shares moved sharply on unverified social media claims.",
              "Reuters", 0, 0.04),
    ]

    print("\n🕵️  Coordination Detector Demo\n")

    dedup = SemanticDeduplicator(threshold=0.10, use_embeddings=False)
    clusters = dedup.cluster(ARTICLES)
    for art in ARTICLES:
        art.propagation = dedup.get_propagation_metrics(art)

    detector = CoordinationDetector(burst_window_minutes=10, burst_z_threshold=1.5)
    reports = detector.analyse_clusters(clusters, ARTICLES)
    detector.update_article_metrics(ARTICLES, reports)

    for report in reports:
        flag = "🚨 COORDINATED" if report.is_coordinated else "✓  Normal"
        print(f"\n{flag}  (score={report.coordination_score:.3f})")
        print(f"  Cluster size:    {len([r for r in [report] if True])} reports")
        print(f"  Burst score:     {report.burst_score:.3f}")
        print(f"  Diversity loss:  {1 - report.source_diversity_score:.3f}")
        print(f"  Cross-platform:  {report.cross_platform_score:.3f}")
        print(f"  Velocity score:  {report.velocity_score:.3f}")
        print(f"  Explanation: {report.explanation[:120]}")

    print("\n📡 Global sweep:")
    global_result = detector.analyse_all(ARTICLES)
    for k, v in global_result.items():
        print(f"  {k}: {v}")
