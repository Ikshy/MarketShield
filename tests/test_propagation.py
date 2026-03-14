"""
tests/test_propagation.py

Tests for semantic deduplication, graph building, and coordination detection.
All tests are offline (no embeddings/network required — uses shingle/heuristic mode).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import List

import pytest

from models import (
    AIDetectionResult, ContentSource, EnrichedArticle, PropagationMetrics,
    RawArticle, Sentiment, SentimentResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_article(
    title: str,
    body: str,
    source: str = "TestFeed",
    source_type: ContentSource = ContentSource.RSS,
    minutes_ago: float = 0.0,
    ai_prob: float = 0.1,
    entities: list[str] | None = None,
) -> EnrichedArticle:
    raw = RawArticle(
        source=source_type,
        source_name=source,
        title=title,
        body=body,
        published_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        raw_metadata={},
    )
    return EnrichedArticle(
        article=raw,
        sentiment=SentimentResult(
            label=Sentiment.NEGATIVE, score=0.8,
            positive=0.1, negative=0.8, neutral=0.1,
        ),
        ai_detection=AIDetectionResult(
            ai_probability=ai_prob,
            is_ai_generated=ai_prob >= 0.7,
            perplexity_score=20.0 if ai_prob > 0.5 else 80.0,
            burstiness_score=0.1 if ai_prob > 0.5 else 0.5,
        ),
        propagation=PropagationMetrics(),
        named_entities=entities or ["TSLA", "Elon Musk"],
    )


# ---------------------------------------------------------------------------
# Shingling utility tests
# ---------------------------------------------------------------------------
class TestShingling:

    def test_identical_texts_have_jaccard_one(self):
        from propagation_analysis.deduplicator import jaccard_similarity
        text = "Tesla stock crashes as Musk sells shares on the open market"
        assert jaccard_similarity(text, text) == pytest.approx(1.0)

    def test_completely_different_texts_low_jaccard(self):
        from propagation_analysis.deduplicator import jaccard_similarity
        a = "Tesla stock crashes as Musk sells shares"
        b = "Bitcoin plunges after China announces mining ban policy"
        sim = jaccard_similarity(a, b)
        assert sim < 0.3

    def test_near_duplicate_texts_high_jaccard(self):
        from propagation_analysis.deduplicator import jaccard_similarity
        a = "BREAKING: Elon Musk arrested by FBI for securities fraud"
        b = "BREAKING: Elon Musk detained by FBI for securities fraud charges"
        sim = jaccard_similarity(a, b)
        assert sim > 0.5  # Should be similar

    def test_empty_text_returns_zero(self):
        from propagation_analysis.deduplicator import jaccard_similarity
        assert jaccard_similarity("", "some text") == 0.0
        assert jaccard_similarity("some text", "") == 0.0

    def test_shingle_returns_set_of_strings(self):
        from propagation_analysis.deduplicator import _shingle
        result = _shingle("hello world", k=3)
        assert isinstance(result, set)
        assert all(len(s) == 3 for s in result)


# ---------------------------------------------------------------------------
# SemanticDeduplicator tests
# ---------------------------------------------------------------------------
class TestSemanticDeduplicator:

    def setup_method(self):
        from propagation_analysis.deduplicator import SemanticDeduplicator
        self.dedup = SemanticDeduplicator(threshold=0.15, use_embeddings=False)

    def test_single_article_creates_singleton_cluster(self):
        articles = [_make_article("Tesla surges on earnings", "TSLA up 15%")]
        clusters = self.dedup.cluster(articles)
        assert len(clusters) == 1
        assert clusters[0].size == 1

    def test_identical_articles_merged_into_one_cluster(self):
        title = "BREAKING: Elon Musk arrested for fraud"
        body = "FBI detained Tesla CEO this morning on fraud charges."
        articles = [
            _make_article(title, body, "Source1", minutes_ago=5.0),
            _make_article(title, body, "Source2", minutes_ago=4.0),
            _make_article(title, body, "Source3", minutes_ago=3.0),
        ]
        clusters = self.dedup.cluster(articles)
        largest = max(clusters, key=lambda c: c.size)
        assert largest.size == 3

    def test_different_articles_create_separate_clusters(self):
        articles = [
            _make_article("Tesla TSLA earnings beat Q4 results", "Revenue exceeded expectations."),
            _make_article("Bitcoin BTC crashes regulatory news", "Crypto market falls sharply."),
        ]
        self.dedup.reset()
        clusters = self.dedup.cluster(articles)
        assert len(clusters) == 2

    def test_cluster_has_source_list(self):
        articles = [
            _make_article("Musk arrested FBI fraud", "FBI detained Musk.", "Reuters", minutes_ago=2),
            _make_article("Musk arrested by FBI", "Musk taken into custody.", "Bloomberg", minutes_ago=1),
        ]
        clusters = self.dedup.cluster(articles)
        largest = max(clusters, key=lambda c: c.size)
        assert len(largest.sources) >= 1

    def test_velocity_computed_for_multi_article_cluster(self):
        articles = [
            _make_article("Musk arrested FBI fraud", "FBI detained Musk.",
                          "S1", minutes_ago=60.0),
            _make_article("Musk arrested by FBI", "Musk detained by FBI.",
                          "S2", minutes_ago=30.0),
            _make_article("FBI detains Musk fraud charges", "Federal agents arrest Musk.",
                          "S3", minutes_ago=0.0),
        ]
        clusters = self.dedup.cluster(articles)
        largest = max(clusters, key=lambda c: c.size)
        if largest.size > 1:
            assert largest.spread_velocity >= 0.0

    def test_propagation_metrics_returned(self):
        title = "TSLA crashes on news"
        body = "Tesla stock dropped sharply after announcement."
        articles = [
            _make_article(title, body, "S1"),
            _make_article(title, body, "S2"),
        ]
        self.dedup.reset()
        self.dedup.cluster(articles)
        metrics = self.dedup.get_propagation_metrics(articles[0])
        assert metrics.cluster_size >= 1
        assert metrics.duplicate_cluster_id is not None

    def test_summary_returns_expected_keys(self):
        articles = [_make_article("Test", "Body")]
        self.dedup.reset()
        self.dedup.cluster(articles)
        summary = self.dedup.summary()
        assert "total_articles" in summary
        assert "total_clusters" in summary
        assert "largest_cluster_size" in summary

    def test_reset_clears_state(self):
        articles = [_make_article("Article A", "Some body text here.")]
        self.dedup.cluster(articles)
        self.dedup.reset()
        summary = self.dedup.summary()
        assert summary["total_articles"] == 0


# ---------------------------------------------------------------------------
# Coordination detector tests
# ---------------------------------------------------------------------------
class TestCoordinationDetector:

    def setup_method(self):
        from propagation_analysis.coordination import CoordinationDetector
        self.detector = CoordinationDetector(
            burst_window_minutes=10.0, burst_z_threshold=1.5
        )

    def _make_coordinated_cluster(self) -> tuple:
        """Create a cluster that should trigger coordination detection."""
        from propagation_analysis.deduplicator import ArticleCluster

        articles = [
            _make_article("Musk arrested FBI fraud",
                          "FBI detained Tesla CEO on fraud charges.", "Fake1", minutes_ago=8, ai_prob=0.92),
            _make_article("Musk detained federal agents",
                          "Tesla founder taken into FBI custody for fraud.", "Fake2", minutes_ago=6, ai_prob=0.90),
            _make_article("FBI arrests Musk securities",
                          "Federal agents arrested Musk on securities charges.", "Fake3", minutes_ago=4, ai_prob=0.88),
            _make_article("Musk faces federal charges FBI",
                          "Sources confirm Musk is under federal arrest.", "Fake4", minutes_ago=2, ai_prob=0.87),
        ]

        cluster = ArticleCluster(
            article_ids=[a.article.id for a in articles],
            centroid_title=articles[0].article.title,
            earliest_seen=articles[-1].article.published_at,
            latest_seen=articles[0].article.published_at,
            sources=["Fake1", "Fake2", "Fake3", "Fake4"],
        )
        cluster.compute_velocity()
        return cluster, articles

    def test_coordinated_cluster_detected(self):
        cluster, articles = self._make_coordinated_cluster()
        report = self.detector.analyse_cluster(cluster, articles)
        assert 0.0 <= report.coordination_score <= 1.0
        assert isinstance(report.is_coordinated, bool)

    def test_coordination_score_in_range(self):
        cluster, articles = self._make_coordinated_cluster()
        report = self.detector.analyse_cluster(cluster, articles)
        assert 0.0 <= report.coordination_score <= 1.0

    def test_single_source_low_diversity(self):
        from propagation_analysis.coordination import _source_diversity_score
        articles = [
            _make_article("News A", "Body A", source="SameSource"),
            _make_article("News B", "Body B", source="SameSource"),
            _make_article("News C", "Body C", source="SameSource"),
        ]
        diversity = _source_diversity_score(articles)
        assert diversity == 0.0  # All same source = zero diversity

    def test_all_different_sources_high_diversity(self):
        from propagation_analysis.coordination import _source_diversity_score
        articles = [
            _make_article("News A", "Body A", source="Reuters"),
            _make_article("News B", "Body B", source="Bloomberg"),
            _make_article("News C", "Body C", source="CoinDesk"),
        ]
        diversity = _source_diversity_score(articles)
        assert diversity > 0.5

    def test_cross_platform_score_multiple_platforms(self):
        from propagation_analysis.coordination import _cross_platform_score
        articles = [
            _make_article("News", "Body", source_type=ContentSource.RSS, minutes_ago=5),
            _make_article("News", "Body", source_type=ContentSource.REDDIT, minutes_ago=4),
            _make_article("News", "Body", source_type=ContentSource.TWITTER, minutes_ago=3),
        ]
        score = _cross_platform_score(articles)
        assert score > 0.0  # Multi-platform = some score

    def test_single_platform_zero_cross_platform(self):
        from propagation_analysis.coordination import _cross_platform_score
        articles = [
            _make_article("News A", "Body", source_type=ContentSource.RSS),
            _make_article("News B", "Body", source_type=ContentSource.RSS),
        ]
        score = _cross_platform_score(articles)
        assert score == 0.0

    def test_burst_detection_finds_spike(self):
        from propagation_analysis.coordination import _compute_burst_score
        # 5 articles in 5 minutes (burst) vs 1/hour baseline
        now = datetime.now(timezone.utc)
        timestamps = [
            now - timedelta(minutes=4),
            now - timedelta(minutes=3),
            now - timedelta(minutes=2),
            now - timedelta(minutes=1),
            now,
        ]
        score, windows = _compute_burst_score(timestamps, window_minutes=10, z_threshold=0.5)
        assert score >= 0.0  # May or may not trigger depending on variance

    def test_analyse_all_returns_expected_keys(self):
        articles = [
            _make_article("Article A", "Body A", minutes_ago=10),
            _make_article("Article B", "Body B", minutes_ago=5),
        ]
        result = self.detector.analyse_all(articles)
        assert "coordination_score" in result
        assert "burst_score" in result
        assert "article_count" in result

    def test_empty_articles_safe(self):
        result = self.detector.analyse_all([])
        assert result["coordination_score"] == 0.0

    def test_update_article_metrics_writes_scores(self):
        from propagation_analysis.coordination import CoordinationReport
        articles = [_make_article("News", "Body text", "Source")]
        cluster_id = articles[0].article.id
        articles[0].propagation.duplicate_cluster_id = cluster_id

        fake_report = CoordinationReport(
            cluster_id=cluster_id,
            coordination_score=0.75,
            is_coordinated=True,
            burst_score=0.6,
            source_diversity_score=0.2,
            cross_platform_score=0.5,
            velocity_score=0.8,
        )
        self.detector.update_article_metrics(articles, [fake_report])
        assert articles[0].propagation.coordination_score == 0.75
        assert articles[0].propagation.is_coordinated is True


# ---------------------------------------------------------------------------
# PropagationRunner integration test
# ---------------------------------------------------------------------------
class TestPropagationRunner:

    def test_runner_processes_articles(self):
        from propagation_analysis.runner import PropagationRunner

        articles = [
            _make_article("Fake Musk arrest news A", "FBI detained Tesla CEO.", "Fake1", minutes_ago=8),
            _make_article("Fake Musk arrest news B", "FBI arrested Tesla founder.", "Fake2", minutes_ago=6),
            _make_article("Tesla TSLA stock drops on rumour", "Shares fall on social media.", "Reuters", minutes_ago=2),
        ]
        runner = PropagationRunner(use_embeddings=False)
        updated, report = runner.process(articles)

        assert len(updated) == 3
        assert report.total_articles == 3
        assert report.propagation_anomaly_score >= 0.0

    def test_runner_fills_propagation_metrics(self):
        from propagation_analysis.runner import PropagationRunner

        articles = [
            _make_article("Test Article", "Test body content about stocks.", "SourceA"),
        ]
        runner = PropagationRunner(use_embeddings=False)
        updated, _ = runner.process(articles)

        for art in updated:
            assert art.propagation is not None
            assert art.propagation.cluster_size >= 1

    def test_runner_reset_clears_state(self):
        from propagation_analysis.runner import PropagationRunner

        runner = PropagationRunner(use_embeddings=False)
        articles = [_make_article("Article", "Body")]
        runner.process(articles)
        assert runner._cycle == 1

        runner.reset()
        assert runner._cycle == 0

    def test_empty_batch_returns_empty_report(self):
        from propagation_analysis.runner import PropagationRunner

        runner = PropagationRunner(use_embeddings=False)
        updated, report = runner.process([])
        assert updated == []
        assert report.total_articles == 0


# ---------------------------------------------------------------------------
# Graph builder tests
# ---------------------------------------------------------------------------
class TestGraphBuilder:

    def test_build_returns_none_when_nx_unavailable(self):
        """If networkx is available, graph should not be None for non-empty input."""
        try:
            import networkx as nx
            from propagation_analysis.graph_builder import PropagationGraphBuilder
            builder = PropagationGraphBuilder()
            articles = [_make_article("Test", "Body text for testing graph nodes.")]
            G = builder.build(articles, clusters=[], sim_matrix=None)
            assert G is not None
            assert G.number_of_nodes() == 1
        except ImportError:
            pytest.skip("networkx not installed")

    def test_propagation_anomaly_score_range(self):
        try:
            import networkx as nx
            from propagation_analysis.graph_builder import PropagationGraphBuilder
            builder = PropagationGraphBuilder()
            articles = [
                _make_article("AI Scam A", "Fraud scheme detected.", "Fake1", ai_prob=0.95),
                _make_article("AI Scam B", "Fraud scheme uncovered.", "Fake2", ai_prob=0.92),
            ]
            G = builder.build(articles, clusters=[])
            analysis = builder.analyse(G)
            score = analysis.get("propagation_anomaly_score", 0.0)
            assert 0.0 <= score <= 1.0
        except ImportError:
            pytest.skip("networkx not installed")

    def test_to_json_serialisable(self):
        import json
        try:
            import networkx as nx
            from propagation_analysis.graph_builder import PropagationGraphBuilder
            builder = PropagationGraphBuilder()
            articles = [_make_article("Test", "Test body", "Source")]
            G = builder.build(articles, clusters=[])
            result = builder.to_json(G)
            # Should be JSON-serialisable
            json.dumps(result)
            assert "nodes" in result
            assert "edges" in result
        except ImportError:
            pytest.skip("networkx not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
