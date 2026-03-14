"""
propagation_analysis/runner.py — Propagation analysis pipeline orchestrator.

Connects deduplication → graph construction → coordination detection
into a single callable pipeline stage.

Input:  list[EnrichedArticle]  (from ai_detection.runner output queue)
Output: list[EnrichedArticle]  (with PropagationMetrics filled in)
        PropagationReport       (cluster/graph/coordination summary)

The runner is designed to process a sliding window of articles:
  - On each cycle it processes the last N hours of articles
  - This allows late-arriving articles to be matched against existing clusters
  - Clusters and graph state persist across cycles (incremental)

Integration in the full pipeline:
    data_ingestion → NLP runner → [this] Propagation runner → risk engine
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import settings
from logger import get_logger
from models import EnrichedArticle

from propagation_analysis.deduplicator import SemanticDeduplicator, ArticleCluster
from propagation_analysis.graph_builder import PropagationGraphBuilder, build_and_analyse
from propagation_analysis.coordination import CoordinationDetector, CoordinationReport

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report data class
# ---------------------------------------------------------------------------
@dataclass
class PropagationReport:
    """Full output of one propagation analysis cycle."""
    cycle_id: str
    processed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Article counts
    total_articles: int = 0
    clustered_articles: int = 0
    suspicious_cluster_count: int = 0

    # Graph summary
    graph_nodes: int = 0
    graph_edges: int = 0
    num_communities: int = 0
    cascade_depth: int = 0
    propagation_anomaly_score: float = 0.0

    # Coordination
    coordinated_cluster_count: int = 0
    max_coordination_score: float = 0.0

    # Detailed sub-reports
    clusters: list[dict] = field(default_factory=list)
    coordination_reports: list[CoordinationReport] = field(default_factory=list)
    top_seeds: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "processed_at": self.processed_at.isoformat(),
            "total_articles": self.total_articles,
            "clustered_articles": self.clustered_articles,
            "suspicious_cluster_count": self.suspicious_cluster_count,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "num_communities": self.num_communities,
            "cascade_depth": self.cascade_depth,
            "propagation_anomaly_score": self.propagation_anomaly_score,
            "coordinated_cluster_count": self.coordinated_cluster_count,
            "max_coordination_score": self.max_coordination_score,
            "clusters": self.clusters,
            "top_seeds": self.top_seeds,
        }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
class PropagationRunner:
    """
    Runs the full propagation analysis pipeline on a batch of articles.

    Maintains state across calls for incremental processing:
      - SemanticDeduplicator accumulates cluster history
      - PropagationGraphBuilder adds to the graph incrementally

    Usage
    -----
    >>> runner = PropagationRunner()
    >>> enriched, report = runner.process(articles)
    >>> print(f"Communities: {report.num_communities}")
    >>> print(f"Coordination score: {report.max_coordination_score:.2f}")
    """

    def __init__(self, use_embeddings: bool = False):
        self._dedup = SemanticDeduplicator(use_embeddings=use_embeddings)
        self._graph_builder = PropagationGraphBuilder()
        self._coordination_detector = CoordinationDetector()
        self._cycle = 0
        log.info(
            f"[PropagationRunner] Initialised "
            f"(embeddings={'on' if use_embeddings else 'off/heuristic'})"
        )

    def process(
        self,
        articles: list[EnrichedArticle],
    ) -> tuple[list[EnrichedArticle], PropagationReport]:
        """
        Run full propagation analysis on a batch of articles.

        Modifies article.propagation in-place and returns the updated list.
        Returns (articles, report).
        """
        self._cycle += 1
        cycle_id = f"cycle_{self._cycle:04d}"

        if not articles:
            return articles, PropagationReport(cycle_id=cycle_id)

        log.info(
            f"[PropagationRunner] Cycle {self._cycle}: "
            f"processing {len(articles)} articles…"
        )

        # ── Stage 1: Semantic deduplication ───────────────────────────────
        clusters = self._dedup.cluster(articles)

        # Write cluster metrics back to articles
        for art in articles:
            metrics = self._dedup.get_propagation_metrics(art)
            art.propagation = metrics

        # ── Stage 2: Graph construction + analysis ─────────────────────────
        G, graph_analysis = build_and_analyse(articles, self._dedup)

        # ── Stage 3: Coordination detection ────────────────────────────────
        coord_reports = self._coordination_detector.analyse_clusters(
            clusters, articles
        )
        self._coordination_detector.update_article_metrics(articles, coord_reports)

        # ── Compile report ─────────────────────────────────────────────────
        suspicious_clusters = self._dedup.suspicious_clusters()
        coordinated = [r for r in coord_reports if r.is_coordinated]

        report = PropagationReport(
            cycle_id=cycle_id,
            total_articles=len(articles),
            clustered_articles=sum(
                1 for a in articles
                if a.propagation and a.propagation.cluster_size > 1
            ),
            suspicious_cluster_count=len(suspicious_clusters),
            graph_nodes=graph_analysis.get("num_nodes", 0),
            graph_edges=graph_analysis.get("num_edges", 0),
            num_communities=graph_analysis.get("num_communities", 0),
            cascade_depth=graph_analysis.get("cascade_depth", 0),
            propagation_anomaly_score=graph_analysis.get(
                "propagation_anomaly_score", 0.0
            ),
            coordinated_cluster_count=len(coordinated),
            max_coordination_score=max(
                (r.coordination_score for r in coord_reports), default=0.0
            ),
            clusters=graph_analysis.get("suspicious_clusters", []),
            coordination_reports=coord_reports,
            top_seeds=graph_analysis.get("top_seeds", []),
        )

        # Log high-risk findings
        if report.max_coordination_score >= settings.risk_alert_threshold:
            log.warning(
                f"[PropagationRunner] 🚨 HIGH COORDINATION DETECTED "
                f"(score={report.max_coordination_score:.2f}) — "
                f"{len(coordinated)} coordinated cluster(s)"
            )
        if report.propagation_anomaly_score >= 0.5:
            log.warning(
                f"[PropagationRunner] ⚠️  Propagation anomaly "
                f"(score={report.propagation_anomaly_score:.2f})"
            )

        log.info(
            f"[PropagationRunner] Cycle {self._cycle} complete — "
            f"clusters={len(clusters)}, "
            f"coordinated={len(coordinated)}, "
            f"anomaly={report.propagation_anomaly_score:.3f}"
        )
        return articles, report

    def reset(self) -> None:
        """Clear all accumulated state (start fresh)."""
        self._dedup.reset()
        self._cycle = 0
        log.info("[PropagationRunner] State reset")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import timedelta
    from models import (
        ContentSource, RawArticle, EnrichedArticle, PropagationMetrics,
        SentimentResult, Sentiment, AIDetectionResult,
    )
    from uuid import uuid4

    def _art(title, body, source, mins_ago, ai_prob=0.1):
        raw = RawArticle(
            source=ContentSource.RSS, source_name=source,
            title=title, body=body,
            published_at=datetime.now(timezone.utc) - timedelta(minutes=mins_ago),
            raw_metadata={},
        )
        return EnrichedArticle(
            article=raw,
            sentiment=SentimentResult(label=Sentiment.NEGATIVE, score=0.85,
                                      positive=0.05, negative=0.85, neutral=0.1),
            ai_detection=AIDetectionResult(
                ai_probability=ai_prob, is_ai_generated=ai_prob > 0.7,
                perplexity_score=22.0, burstiness_score=0.09),
            propagation=PropagationMetrics(),
            named_entities=["TSLA", "Elon Musk", "Tesla", "short squeeze"],
        )

    BATCH = [
        # Coordinated fake news campaign
        _art("BREAKING: Elon Musk arrested by FBI for fraud",
             "FBI agents detained Tesla CEO this morning on fraud charges.",
             "FakeSource1.io", 9, 0.93),
        _art("BREAKING NEWS: Musk detained by federal authorities",
             "Tesla founder Elon Musk was taken into custody this morning.",
             "FakeSource2.net", 7, 0.91),
        _art("URGENT: Musk faces imminent arrest sources say",
             "Multiple sources confirm FBI is moving on Musk arrest warrant.",
             "FakeSource3.org", 5, 0.89),
        _art("FBI raids Tesla HQ, Musk under investigation",
             "Federal agents searching Tesla headquarters amid Musk probe.",
             "FakeSource4.xyz", 3, 0.87),
        # Legitimate articles
        _art("Tesla TSLA stock drops on social media rumours",
             "Shares fell 12% amid unverified claims spreading online.",
             "Reuters", 2, 0.06),
        _art("TSLA unusual options activity: record put volume",
             "Traders bought massive put options before today's TSLA drop.",
             "Bloomberg", 1, 0.04),
        _art("Bitcoin falls below 40k on macro uncertainty",
             "BTC declined as investors await Federal Reserve decision.",
             "CoinDesk", 0, 0.08),
    ]

    print("\n🔗 Propagation Runner Demo\n")
    runner = PropagationRunner(use_embeddings=False)
    enriched, report = runner.process(BATCH)

    print(json.dumps(report.to_dict(), indent=2, default=str)[:1500])
    print("\n--- Article propagation metrics ---")
    for art in enriched:
        p = art.propagation
        coord = "🚨 COORD" if p.is_coordinated else "✓"
        print(
            f"  {coord}  cluster_sz={p.cluster_size}  "
            f"coord={p.coordination_score:.2f}  "
            f"velocity={p.spread_velocity:.1f}/hr  "
            f"{art.article.title[:55]}"
        )
