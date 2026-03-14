"""
propagation_analysis/graph_builder.py — Information propagation graph.

Builds a directed NetworkX graph where:
  - Nodes  = articles (or clusters of articles)
  - Edges  = propagation relationships (temporal + semantic)

Edge types:
  SIMILAR   — two articles share ≥ threshold semantic similarity
  TEMPORAL  — one article cites/follows another within a time window
  CLUSTER   — both articles belong to the same dedup cluster

Node attributes:
  source_name, title, published_at, sentiment, ai_probability,
  cluster_id, is_ai_generated, source_type (rss/reddit/twitter)

Algorithms applied to the graph:
  1. Connected components — find isolated narratives vs widespread ones
  2. Louvain community detection — identify coordinated sub-networks
  3. Centrality measures — PageRank-like score to find "seed" articles
  4. Cascade depth — how many "hops" a narrative spread

The graph is serialisable to JSON (for dashboard) and exportable
to GEXF format (for Gephi visualisation).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

from config import settings
from logger import get_logger
from models import EnrichedArticle
from propagation_analysis.deduplicator import ArticleCluster, SemanticDeduplicator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Edge type constants
# ---------------------------------------------------------------------------
class EdgeType:
    SIMILAR  = "similar"     # Semantic similarity above threshold
    TEMPORAL = "temporal"    # Published within time window of similar article
    CLUSTER  = "cluster"     # Same dedup cluster


# ---------------------------------------------------------------------------
# Graph analyser
# ---------------------------------------------------------------------------
class PropagationGraphBuilder:
    """
    Constructs and analyses information propagation graphs.

    Usage
    -----
    >>> builder = PropagationGraphBuilder()
    >>> graph = builder.build(enriched_articles, clusters)
    >>> analysis = builder.analyse(graph)
    >>> print(f"Communities: {analysis['num_communities']}")
    >>> print(f"Most central: {analysis['top_seeds']}")
    """

    def __init__(
        self,
        temporal_window_hours: float = 6.0,
        similarity_threshold: float | None = None,
    ):
        if not _NX_AVAILABLE:
            log.warning("[GraphBuilder] networkx not installed — graph features disabled")
        self.temporal_window = timedelta(hours=temporal_window_hours)
        self.threshold = similarity_threshold or settings.similarity_dedup_threshold

    # ── Graph construction ────────────────────────────────────────────────
    def build(
        self,
        articles: list[EnrichedArticle],
        clusters: list[ArticleCluster] | None = None,
        sim_matrix: np.ndarray | None = None,
    ) -> Optional["nx.DiGraph"]:
        """
        Build the propagation graph from a list of enriched articles.

        Parameters
        ----------
        articles    : Enriched articles to add as nodes
        clusters    : Pre-computed clusters (from SemanticDeduplicator)
        sim_matrix  : Optional precomputed similarity matrix (n×n).
                      If None, temporal edges only are used.

        Returns a directed NetworkX graph, or None if networkx unavailable.
        """
        if not _NX_AVAILABLE:
            return None

        G = nx.DiGraph()

        # ── Add nodes ─────────────────────────────────────────────────────
        for art in articles:
            a = art.article
            G.add_node(
                a.id,
                title=a.title[:80],
                source_name=a.source_name,
                source_type=a.source.value,
                published_at=a.published_at.isoformat(),
                published_ts=a.published_at.timestamp(),
                ai_probability=art.ai_detection.ai_probability if art.ai_detection else 0.0,
                is_ai_generated=art.ai_detection.is_ai_generated if art.ai_detection else False,
                sentiment=art.sentiment.label.value if art.sentiment else "neutral",
                sentiment_score=art.sentiment.score if art.sentiment else 0.5,
                cluster_id=art.propagation.duplicate_cluster_id if art.propagation else None,
                named_entities=art.named_entities[:5],
            )

        # ── Add cluster edges ─────────────────────────────────────────────
        if clusters:
            cluster_id_map: dict[str, str] = {}
            for cluster in clusters:
                for art_id in cluster.article_ids:
                    cluster_id_map[art_id] = cluster.cluster_id

            for cluster in clusters:
                ids = cluster.article_ids
                if len(ids) < 2:
                    continue
                # Chain edges: earliest → ... → latest (temporal order)
                sorted_ids = sorted(
                    ids,
                    key=lambda i: G.nodes[i].get("published_ts", 0)
                    if i in G.nodes else 0,
                )
                for i in range(len(sorted_ids) - 1):
                    u, v = sorted_ids[i], sorted_ids[i + 1]
                    if u in G and v in G:
                        G.add_edge(u, v,
                                   edge_type=EdgeType.CLUSTER,
                                   weight=1.0,
                                   cluster_id=cluster.cluster_id)

        # ── Add temporal edges ────────────────────────────────────────────
        # Pairs of articles published close together and sharing named entities
        entity_to_articles: dict[str, list[str]] = defaultdict(list)
        for art in articles:
            for entity in art.named_entities[:5]:
                entity_to_articles[entity].append(art.article.id)

        added_temporal = 0
        for entity, art_ids in entity_to_articles.items():
            if len(art_ids) < 2:
                continue
            # Sort by publication time
            sorted_by_time = sorted(
                art_ids,
                key=lambda i: G.nodes[i].get("published_ts", 0) if i in G else 0,
            )
            for i in range(len(sorted_by_time) - 1):
                u, v = sorted_by_time[i], sorted_by_time[i + 1]
                if u not in G or v not in G:
                    continue
                if G.has_edge(u, v):
                    continue  # Already connected

                ts_u = G.nodes[u].get("published_ts", 0)
                ts_v = G.nodes[v].get("published_ts", 0)
                if abs(ts_v - ts_u) <= self.temporal_window.total_seconds():
                    G.add_edge(u, v,
                               edge_type=EdgeType.TEMPORAL,
                               weight=0.5,
                               shared_entity=entity)
                    added_temporal += 1

        # ── Add similarity edges (if matrix provided) ─────────────────────
        added_sim = 0
        if sim_matrix is not None:
            art_ids = [a.article.id for a in articles]
            n = len(art_ids)
            for i in range(n):
                for j in range(i + 1, n):
                    sim = float(sim_matrix[i, j])
                    if sim >= self.threshold:
                        u, v = art_ids[i], art_ids[j]
                        if not G.has_edge(u, v):
                            # Direction: earlier → later
                            ts_i = G.nodes[u].get("published_ts", 0)
                            ts_j = G.nodes[v].get("published_ts", 0)
                            src, dst = (u, v) if ts_i <= ts_j else (v, u)
                            G.add_edge(src, dst,
                                       edge_type=EdgeType.SIMILAR,
                                       weight=sim)
                            added_sim += 1

        log.info(
            f"[GraphBuilder] Graph built: "
            f"{G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges "
            f"({added_temporal} temporal, {added_sim} similarity)"
        )
        return G

    # ── Graph analysis ────────────────────────────────────────────────────
    def analyse(self, G: "nx.DiGraph") -> dict:
        """
        Run graph algorithms and return a structured analysis dict.

        Algorithms:
          - Weakly connected components (ignoring edge direction)
          - Louvain community detection (undirected projection)
          - PageRank (identifies narrative "seeds" — high-influence originators)
          - Cascade depth (longest path from any root node)
          - Anomaly score per component
        """
        if not _NX_AVAILABLE or G is None or G.number_of_nodes() == 0:
            return {"error": "Graph unavailable or empty"}

        result: dict = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
        }

        # ── Connected components ──────────────────────────────────────────
        undirected = G.to_undirected()
        components = list(nx.connected_components(undirected))
        result["num_components"] = len(components)
        result["largest_component_size"] = max(len(c) for c in components) if components else 0
        result["isolated_nodes"] = sum(1 for c in components if len(c) == 1)

        # ── Community detection (Louvain) ─────────────────────────────────
        try:
            import community as community_louvain  # type: ignore
            partition = community_louvain.best_partition(undirected)
            num_communities = len(set(partition.values()))
            result["num_communities"] = num_communities
            result["community_partition"] = partition

            # Add community label to graph nodes
            for node, comm_id in partition.items():
                if node in G:
                    G.nodes[node]["community"] = comm_id

            log.debug(f"[GraphBuilder] Louvain: {num_communities} communities")

        except ImportError:
            # Fall back to label propagation (built into networkx)
            try:
                communities = list(nx.community.label_propagation_communities(undirected))
                result["num_communities"] = len(communities)
                partition = {}
                for comm_id, nodes in enumerate(communities):
                    for node in nodes:
                        partition[node] = comm_id
                        if node in G:
                            G.nodes[node]["community"] = comm_id
                result["community_partition"] = partition
            except Exception:
                result["num_communities"] = result["num_components"]
                result["community_partition"] = {}

        # ── PageRank (narrative seeds) ────────────────────────────────────
        try:
            pagerank = nx.pagerank(G, weight="weight", max_iter=100)
            top_seeds = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            result["top_seeds"] = [
                {
                    "article_id": node_id,
                    "pagerank": round(score, 5),
                    "title": G.nodes[node_id].get("title", "")[:60],
                    "source": G.nodes[node_id].get("source_name", ""),
                    "ai_probability": G.nodes[node_id].get("ai_probability", 0.0),
                }
                for node_id, score in top_seeds
            ]
            for node_id, score in pagerank.items():
                if node_id in G:
                    G.nodes[node_id]["pagerank"] = score
        except Exception as exc:
            log.warning(f"[GraphBuilder] PageRank failed: {exc}")
            result["top_seeds"] = []

        # ── Cascade depth ─────────────────────────────────────────────────
        try:
            # Longest path in the DAG (topological order)
            if nx.is_directed_acyclic_graph(G):
                result["cascade_depth"] = nx.dag_longest_path_length(G)
            else:
                # Has cycles (can happen with imperfect temporal ordering)
                # Approximate with diameter of undirected graph
                if nx.is_connected(undirected) and undirected.number_of_nodes() < 500:
                    result["cascade_depth"] = nx.diameter(undirected)
                else:
                    result["cascade_depth"] = -1  # Too large / disconnected
        except Exception:
            result["cascade_depth"] = -1

        # ── AI content spread ─────────────────────────────────────────────
        ai_nodes = [n for n, d in G.nodes(data=True) if d.get("is_ai_generated")]
        result["ai_node_count"] = len(ai_nodes)
        result["ai_node_fraction"] = (
            len(ai_nodes) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0.0
        )

        # ── Propagation anomaly score ─────────────────────────────────────
        result["propagation_anomaly_score"] = self._propagation_anomaly_score(G, result)

        return result

    def _propagation_anomaly_score(self, G: "nx.DiGraph", analysis: dict) -> float:
        """
        0–1 score indicating how anomalous the propagation pattern is.

        Signals:
          - High AI fraction in spreading nodes: suspicious
          - Very fast cascade (deep + dense): suspicious
          - Many near-identical articles from different sources: suspicious
          - Hub-spoke topology (one source → many): suspicious
        """
        score = 0.0

        # AI content fraction
        ai_frac = analysis.get("ai_node_fraction", 0.0)
        score += 0.40 * min(1.0, ai_frac * 2)

        # Density anomaly (normal news: sparse; coordinated: dense)
        density = analysis.get("density", 0.0)
        score += 0.20 * min(1.0, density * 10)

        # Cascade depth (deeper = more suspicious)
        depth = analysis.get("cascade_depth", 0)
        if depth > 0:
            score += 0.20 * min(1.0, depth / 10)

        # Small number of communities relative to nodes (coordinated = few communities)
        n_nodes = analysis.get("num_nodes", 1)
        n_comms = analysis.get("num_communities", 1)
        if n_nodes > 5:
            comm_ratio = n_comms / n_nodes
            # Very few communities relative to nodes = more coordinated
            score += 0.20 * max(0.0, 1.0 - comm_ratio * 5)

        return round(min(1.0, score), 4)

    # ── Serialisation ─────────────────────────────────────────────────────
    def to_json(self, G: "nx.DiGraph") -> dict:
        """
        Serialise graph to a JSON-compatible dict for dashboard display.
        Format: {nodes: [...], edges: [...]}
        """
        if G is None:
            return {"nodes": [], "edges": []}

        nodes = []
        for node_id, attrs in G.nodes(data=True):
            nodes.append({
                "id": node_id,
                **{k: v for k, v in attrs.items()
                   if isinstance(v, (str, int, float, bool, list, type(None)))}
            })

        edges = []
        for u, v, attrs in G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                **{k: v2 for k, v2 in attrs.items()
                   if isinstance(v2, (str, int, float, bool, type(None)))}
            })

        return {"nodes": nodes, "edges": edges}

    def to_gexf(self, G: "nx.DiGraph", filepath: str) -> None:
        """Export graph to GEXF format for Gephi visualisation."""
        if G is None:
            return
        nx.write_gexf(G, filepath)
        log.info(f"[GraphBuilder] GEXF exported: {filepath}")


# ---------------------------------------------------------------------------
# Convenience function for pipeline integration
# ---------------------------------------------------------------------------
def build_and_analyse(
    articles: list[EnrichedArticle],
    deduplicator: SemanticDeduplicator | None = None,
) -> tuple[Optional["nx.DiGraph"], dict]:
    """
    Run the full propagation analysis pipeline on a batch of articles.

    Returns (graph, analysis_dict).
    """
    # Step 1: Cluster articles
    dedup = deduplicator or SemanticDeduplicator(use_embeddings=False)
    clusters = dedup.cluster(articles)

    # Step 2: Update propagation metrics on each article
    for art in articles:
        metrics = dedup.get_propagation_metrics(art)
        art.propagation = metrics

    # Step 3: Build graph
    builder = PropagationGraphBuilder()
    G = builder.build(articles, clusters)

    # Step 4: Analyse
    analysis = builder.analyse(G) if G is not None else {}
    analysis["dedup_summary"] = dedup.summary()
    analysis["suspicious_clusters"] = [
        {
            "cluster_id": c.cluster_id,
            "size": c.size,
            "velocity": round(c.spread_velocity, 2),
            "sources": c.sources,
            "centroid_title": c.centroid_title[:80],
        }
        for c in dedup.suspicious_clusters()
    ]

    return G, analysis


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json as _json
    from datetime import datetime, timezone, timedelta
    from models import ContentSource, RawArticle, EnrichedArticle, PropagationMetrics
    from models import SentimentResult, Sentiment, AIDetectionResult

    def _make_enriched(
        title: str, body: str, source: str,
        hours_ago: float = 0, ai_prob: float = 0.1
    ) -> EnrichedArticle:
        raw = RawArticle(
            source=ContentSource.RSS, source_name=source,
            title=title, body=body,
            published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
            raw_metadata={},
        )
        return EnrichedArticle(
            article=raw,
            sentiment=SentimentResult(
                label=Sentiment.NEGATIVE, score=0.8,
                positive=0.1, negative=0.8, neutral=0.1
            ),
            ai_detection=AIDetectionResult(
                ai_probability=ai_prob, is_ai_generated=ai_prob > 0.7,
                perplexity_score=25.0 if ai_prob > 0.5 else 80.0,
                burstiness_score=0.1 if ai_prob > 0.5 else 0.5,
            ),
            propagation=PropagationMetrics(),
            named_entities=["TSLA", "Elon Musk", "Tesla"],
        )

    ARTICLES = [
        _make_enriched("BREAKING: Elon Musk arrested for securities fraud",
                       "FBI has taken Tesla CEO into custody.", "FakeNews1.io", 3.0, 0.91),
        _make_enriched("BREAKING: Musk detained by FBI agents this morning",
                       "Tesla's founder faces federal fraud charges.", "FakeNews2.io", 2.8, 0.88),
        _make_enriched("Sources: Elon Musk facing imminent arrest",
                       "Federal authorities closing in on Musk.", "FakeNews3.net", 2.5, 0.85),
        _make_enriched("Tesla CEO Elon Musk steps down voluntarily",
                       "Musk cites need to focus on X platform.", "Reuters", 2.0, 0.05),
        _make_enriched("Tesla stock TSLA crashes 40% on Musk news",
                       "TSLA shares in freefall following reports.", "StockAlert.io", 1.5, 0.72),
        _make_enriched("TSLA options see record put volume ahead of announcement",
                       "Unusual options activity detected before news.", "MarketWatch", 1.0, 0.08),
    ]

    print("\n🕸️  Propagation Graph Analysis Demo\n")
    G, analysis = build_and_analyse(ARTICLES)

    print(f"Nodes:              {analysis.get('num_nodes', 0)}")
    print(f"Edges:              {analysis.get('num_edges', 0)}")
    print(f"Components:         {analysis.get('num_components', 0)}")
    print(f"Communities:        {analysis.get('num_communities', 0)}")
    print(f"Cascade depth:      {analysis.get('cascade_depth', 0)}")
    print(f"AI node fraction:   {analysis.get('ai_node_fraction', 0):.1%}")
    print(f"Anomaly score:      {analysis.get('propagation_anomaly_score', 0):.3f}")
    print(f"\nSuspicious clusters: {len(analysis.get('suspicious_clusters', []))}")
    for sc in analysis.get("suspicious_clusters", []):
        print(f"  [{sc['size']} articles, {sc['velocity']:.1f}/hr] {sc['centroid_title'][:60]}")
    print(f"\nTop seeds:")
    for seed in analysis.get("top_seeds", [])[:3]:
        print(f"  PageRank={seed['pagerank']:.5f}  AI={seed['ai_probability']:.2f}  {seed['title'][:55]}")
