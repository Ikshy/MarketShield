"""
propagation_analysis/deduplicator.py — Semantic similarity deduplication.

Detects articles that are functionally identical even when reworded,
translated, or slightly paraphrased. This is the primary signal for
coordinated information campaigns: the same narrative appearing across
dozens of sources within a short window.

Two strategies are combined:

  1. Exact hash dedup (O(1)) — catches byte-for-byte copies
     Already handled in data_ingestion. Not repeated here.

  2. Semantic similarity (O(n²) with early termination)
     Sentence-transformer embeddings → cosine similarity matrix.
     Articles with similarity ≥ threshold are merged into a cluster.

     Model: sentence-transformers/all-MiniLM-L6-v2
       - 384-dim embeddings
       - ~22 MB (tiny)
       - 80ms/article on CPU — suitable for real-time use
       - Performs well on short financial texts

  3. Shingling (fallback) — character n-gram Jaccard similarity.
     No model required. Less accurate on paraphrased content but
     fast and always available.

Cluster output feeds into:
  - PropagationMetrics.duplicate_cluster_id (per article)
  - PropagationMetrics.cluster_size (how many copies exist)
  - PropagationMetrics.spread_velocity (copies per hour)
  - CoordinationDetector (which accounts posted the cluster)
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional
from uuid import uuid4

import numpy as np

from config import settings
from logger import get_logger
from models import EnrichedArticle, PropagationMetrics

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ArticleCluster:
    """A group of semantically similar articles."""
    cluster_id: str = field(default_factory=lambda: str(uuid4()))
    article_ids: list[str] = field(default_factory=list)
    centroid_title: str = ""          # Title of the most-upvoted / earliest article
    earliest_seen: Optional[datetime] = None
    latest_seen: Optional[datetime] = None
    sources: list[str] = field(default_factory=list)   # Unique sources
    spread_velocity: float = 0.0      # Articles per hour

    @property
    def size(self) -> int:
        return len(self.article_ids)

    @property
    def time_span_hours(self) -> float:
        if self.earliest_seen and self.latest_seen:
            delta = (self.latest_seen - self.earliest_seen).total_seconds()
            return delta / 3600
        return 0.0

    def compute_velocity(self) -> float:
        """Articles per hour within the cluster's time window."""
        if self.time_span_hours > 0 and self.size > 1:
            self.spread_velocity = (self.size - 1) / self.time_span_hours
        else:
            self.spread_velocity = 0.0
        return self.spread_velocity


# ---------------------------------------------------------------------------
# Shingling fallback (no model required)
# ---------------------------------------------------------------------------
def _shingle(text: str, k: int = 5) -> set[str]:
    """
    Create character k-shingles from text.
    Fast approximation of document similarity.
    """
    text = re.sub(r"\s+", " ", text.lower().strip())
    return {text[i:i+k] for i in range(max(0, len(text) - k + 1))}


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def jaccard_similarity(text_a: str, text_b: str, k: int = 5) -> float:
    """Jaccard similarity between two texts using character shingles."""
    return _jaccard(_shingle(text_a, k), _shingle(text_b, k))


# ---------------------------------------------------------------------------
# Sentence transformer loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_embedder():
    """
    Load sentence-transformer model (cached after first call).
    Downloads ~22 MB on first use.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        log.info(f"[Dedup] Loading embedder: {settings.embedding_model}")
        model = SentenceTransformer(settings.embedding_model)
        log.info("[Dedup] Sentence transformer ready")
        return model
    except (ImportError, OSError) as exc:
        log.warning(f"[Dedup] sentence-transformers unavailable: {exc}")
        return None


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for a batch of embeddings.
    Normalises rows then uses dot product — O(n²) but vectorised.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)  # Avoid division by zero
    normalised = embeddings / norms
    return normalised @ normalised.T


# ---------------------------------------------------------------------------
# Core deduplicator
# ---------------------------------------------------------------------------
class SemanticDeduplicator:
    """
    Groups articles into clusters of semantically near-duplicate content.

    Usage
    -----
    >>> dedup = SemanticDeduplicator()
    >>> clusters = dedup.cluster(enriched_articles)
    >>> for cluster in clusters:
    ...     if cluster.size > 3:
    ...         print(f"Cluster of {cluster.size} near-duplicates: {cluster.centroid_title}")

    The deduplicator is stateful: calling cluster() multiple times
    accumulates history. Call reset() to start fresh.
    """

    def __init__(
        self,
        threshold: float | None = None,
        use_embeddings: bool = True,
    ):
        self.threshold = threshold or settings.similarity_dedup_threshold
        self._use_embeddings = use_embeddings
        self._embedder_available = False

        if use_embeddings:
            model = _load_embedder()
            self._embedder_available = model is not None

        # State: maps article_id → cluster_id
        self._article_to_cluster: dict[str, str] = {}
        # State: cluster_id → ArticleCluster
        self._clusters: dict[str, ArticleCluster] = {}

    def reset(self) -> None:
        """Clear all accumulated cluster state."""
        self._article_to_cluster.clear()
        self._clusters.clear()

    # ── Public API ────────────────────────────────────────────────────────
    def cluster(
        self,
        articles: list[EnrichedArticle],
    ) -> list[ArticleCluster]:
        """
        Cluster a batch of articles by semantic similarity.

        Articles already seen in previous calls are matched against
        existing clusters (incremental mode for the live pipeline).

        Returns all clusters (including singletons).
        Sorts by cluster size descending.
        """
        if not articles:
            return []

        texts = self._get_texts(articles)

        if self._embedder_available:
            sim_matrix = self._embedding_similarity(texts)
        else:
            sim_matrix = self._shingle_similarity(texts)

        self._assign_clusters(articles, sim_matrix)
        self._update_cluster_metadata(articles)

        clusters = sorted(
            self._clusters.values(),
            key=lambda c: c.size,
            reverse=True,
        )
        log.debug(
            f"[Dedup] {len(articles)} articles → "
            f"{len(clusters)} clusters "
            f"({sum(1 for c in clusters if c.size > 1)} with duplicates)"
        )
        return clusters

    def get_propagation_metrics(
        self, article: EnrichedArticle
    ) -> PropagationMetrics:
        """
        Return propagation metrics for one article.
        Requires that cluster() has already been called for this article.
        """
        cluster_id = self._article_to_cluster.get(article.article.id)
        if cluster_id is None:
            return PropagationMetrics()

        cluster = self._clusters.get(cluster_id)
        if cluster is None:
            return PropagationMetrics()

        return PropagationMetrics(
            duplicate_cluster_id=cluster_id,
            cluster_size=cluster.size,
            is_coordinated=False,  # Filled by CoordinationDetector
            coordination_score=0.0,
            spread_velocity=cluster.spread_velocity,
        )

    # ── Similarity computation ────────────────────────────────────────────
    def _get_texts(self, articles: list[EnrichedArticle]) -> list[str]:
        """Extract canonical text representations for similarity comparison."""
        return [
            f"{a.article.title}. {a.article.body[:300]}"
            for a in articles
        ]

    def _embedding_similarity(self, texts: list[str]) -> np.ndarray:
        """Compute pairwise cosine similarity using sentence embeddings."""
        model = _load_embedder()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return _cosine_similarity_matrix(embeddings)

    def _shingle_similarity(self, texts: list[str]) -> np.ndarray:
        """
        Compute pairwise Jaccard similarity using character shingles.
        O(n²) — avoid on batches > 500 articles.
        """
        n = len(texts)
        shingles = [_shingle(t, k=5) for t in texts]
        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            sim[i, i] = 1.0
            for j in range(i + 1, n):
                s = _jaccard(shingles[i], shingles[j])
                sim[i, j] = sim[j, i] = s
        return sim

    # ── Cluster assignment (greedy single-linkage) ────────────────────────
    def _assign_clusters(
        self,
        articles: list[EnrichedArticle],
        sim_matrix: np.ndarray,
    ) -> None:
        """
        Greedy single-linkage clustering.

        Each article is assigned to the cluster of its most similar
        already-clustered neighbour (if similarity ≥ threshold).
        If no neighbour qualifies, a new singleton cluster is created.

        Time: O(n²) — acceptable for batches up to ~1,000 articles.
        """
        n = len(articles)
        article_ids = [a.article.id for a in articles]

        # Temporary local cluster assignments for this batch
        local_clusters: list[int] = [-1] * n  # -1 = unassigned

        for i in range(n):
            art_id = article_ids[i]

            # Already processed in a previous call?
            if art_id in self._article_to_cluster:
                continue

            # Find the most similar already-assigned article in this batch
            best_j, best_sim = -1, self.threshold
            for j in range(i):
                if sim_matrix[i, j] >= best_sim:
                    best_j, best_sim = j, sim_matrix[i, j]

            if best_j >= 0 and local_clusters[best_j] >= 0:
                # Attach to existing local cluster
                local_clusters[i] = local_clusters[best_j]
            else:
                # New cluster
                local_clusters[i] = i

        # Merge local cluster assignments into global state
        local_to_global: dict[int, str] = {}

        for i, art_id in enumerate(article_ids):
            if art_id in self._article_to_cluster:
                continue

            local_id = local_clusters[i]
            if local_id not in local_to_global:
                # Check if article[local_id] is already in a global cluster
                leader_id = article_ids[local_id]
                if leader_id in self._article_to_cluster:
                    local_to_global[local_id] = self._article_to_cluster[leader_id]
                else:
                    # Create new global cluster
                    new_cluster = ArticleCluster(
                        centroid_title=articles[local_id].article.title
                    )
                    self._clusters[new_cluster.cluster_id] = new_cluster
                    local_to_global[local_id] = new_cluster.cluster_id

            global_id = local_to_global[local_id]
            self._article_to_cluster[art_id] = global_id
            self._clusters[global_id].article_ids.append(art_id)

    def _update_cluster_metadata(self, articles: list[EnrichedArticle]) -> None:
        """Refresh time windows, source lists, and velocities for all clusters."""
        # Build article lookup
        art_lookup = {a.article.id: a for a in articles}

        for cluster in self._clusters.values():
            times = []
            sources = set()
            for art_id in cluster.article_ids:
                art = art_lookup.get(art_id)
                if art:
                    times.append(art.article.published_at)
                    sources.add(art.article.source_name)

            if times:
                cluster.earliest_seen = min(times)
                cluster.latest_seen = max(times)
            cluster.sources = list(sources)
            cluster.compute_velocity()

    # ── Reporting ─────────────────────────────────────────────────────────
    def suspicious_clusters(
        self,
        min_size: int = 3,
        min_velocity: float = 2.0,
    ) -> list[ArticleCluster]:
        """
        Return clusters that look like coordinated spreading campaigns.

        Parameters
        ----------
        min_size : int
            Minimum number of articles to flag (default: 3)
        min_velocity : float
            Minimum spread velocity in articles/hour (default: 2.0)
        """
        return [
            c for c in self._clusters.values()
            if c.size >= min_size and c.spread_velocity >= min_velocity
        ]

    def summary(self) -> dict:
        """Return summary statistics for logging / dashboard."""
        clusters = list(self._clusters.values())
        return {
            "total_articles": len(self._article_to_cluster),
            "total_clusters": len(clusters),
            "singleton_clusters": sum(1 for c in clusters if c.size == 1),
            "multi_article_clusters": sum(1 for c in clusters if c.size > 1),
            "largest_cluster_size": max((c.size for c in clusters), default=0),
            "max_spread_velocity": max((c.spread_velocity for c in clusters), default=0.0),
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import timedelta
    from models import ContentSource, RawArticle, EnrichedArticle, PropagationMetrics

    def _make_enriched(title: str, body: str, source: str, hours_ago: float = 0) -> EnrichedArticle:
        raw = RawArticle(
            source=ContentSource.RSS,
            source_name=source,
            title=title,
            body=body,
            published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
            raw_metadata={},
        )
        return EnrichedArticle(article=raw, propagation=PropagationMetrics())

    # Simulated coordinated campaign: same narrative, different sources
    ARTICLES = [
        _make_enriched("BREAKING: Elon Musk arrested for fraud",
                       "Reports say Tesla CEO has been taken into custody by FBI agents.",
                       "CryptoNewsFast.io", hours_ago=2.0),
        _make_enriched("BREAKING NEWS: Elon Musk detained by federal authorities",
                       "Tesla CEO Elon Musk was reportedly arrested by the FBI earlier today.",
                       "StockAlert24.com", hours_ago=1.8),
        _make_enriched("URGENT: Elon Musk facing federal charges, sources say",
                       "Multiple sources confirm Tesla founder is under FBI investigation.",
                       "MarketBreaker.net", hours_ago=1.5),
        _make_enriched("Elon Musk steps down from Tesla CEO role voluntarily",
                       "In a surprise announcement, Musk ceded the CEO position to focus on X.",
                       "Reuters", hours_ago=1.0),
        _make_enriched("Bitcoin falls below $30k on macro concerns",
                       "Cryptocurrency markets declined amid rising interest rate expectations.",
                       "CoinDesk", hours_ago=0.5),
        _make_enriched("BTC drops to 30000 as Fed signals rate hike",
                       "Bitcoin slid to $30,000 after Federal Reserve members hinted at tightening.",
                       "CoinTelegraph", hours_ago=0.3),
    ]

    print("\n🔍 SemanticDeduplicator Demo (shingle mode — no model download)\n")
    dedup = SemanticDeduplicator(threshold=0.15, use_embeddings=False)
    clusters = dedup.cluster(ARTICLES)

    for c in clusters:
        flag = "🚨 SUSPICIOUS" if c.size >= 3 else ("📌 DUPLICATE" if c.size > 1 else "  ")
        print(f"  {flag} Cluster ({c.size} articles, {c.spread_velocity:.1f}/hr):")
        print(f"     Centroid: {c.centroid_title[:65]}")
        print(f"     Sources:  {', '.join(c.sources[:4])}")
        print()

    print("📊 Summary:", dedup.summary())
