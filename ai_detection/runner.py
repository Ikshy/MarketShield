"""
ai_detection/runner.py — NLP analysis pipeline orchestrator.

Consumes RawArticle objects (from data_ingestion queue or DB),
runs all NLP stages, and produces EnrichedArticle objects.

Pipeline per article:
  1. Clean + tokenise text
  2. Sentiment analysis (FinBERT or VADER)
  3. AI content detection (ensemble)
  4. Entity + ticker extraction
  5. Persist EnrichedArticle to DB
  6. Push to risk engine queue

Performance notes:
  - Transformers are loaded once and kept in memory (lru_cache)
  - Batch processing: articles are grouped in batches of 16 for sentiment
  - Typical throughput: ~30 articles/min on CPU with FinBERT
                        ~200 articles/min with heuristic fallbacks
"""

from __future__ import annotations

import asyncio
import json
import time
from asyncio import Queue
from datetime import datetime
from typing import Optional

from config import settings
from database import ArticleRepository, init_db
from logger import get_logger, configure_logging
from models import (
    AIDetectionResult,
    EnrichedArticle,
    PropagationMetrics,
    RawArticle,
    SentimentResult,
)
from ai_detection.sentiment import SentimentAnalyser
from ai_detection.ai_detector import AIContentDetector
from ai_detection.entity_extractor import EntityExtractor

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Article enrichment (single article)
# ---------------------------------------------------------------------------
def enrich_article(
    article: RawArticle,
    sentiment_analyser: SentimentAnalyser,
    ai_detector: AIContentDetector,
    entity_extractor: EntityExtractor,
) -> EnrichedArticle:
    """
    Run all NLP stages on a single article and return an EnrichedArticle.

    This function is pure (no I/O) — it can be unit-tested without DB/network.
    """
    full_text = f"{article.title}. {article.body}"

    # Stage 1: Sentiment
    sentiment: SentimentResult = sentiment_analyser.analyse_article(
        article.title, article.body
    )

    # Stage 2: AI detection
    ai_detection: AIDetectionResult = ai_detector.detect(full_text)

    # Stage 3: Entity extraction
    entities = entity_extractor.extract(article.title, article.body)

    # Stage 4: Propagation placeholder (filled by propagation_analysis module)
    propagation = PropagationMetrics()

    return EnrichedArticle(
        article=article,
        sentiment=sentiment,
        ai_detection=ai_detection,
        propagation=propagation,
        named_entities=entities["all_entities"],
        keywords=entities["keywords"],
    )


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------
def enrich_batch(
    articles: list[RawArticle],
    sentiment_analyser: SentimentAnalyser,
    ai_detector: AIContentDetector,
    entity_extractor: EntityExtractor,
) -> list[EnrichedArticle]:
    """
    Enrich a batch of articles with optimised batched inference.

    Sentiment uses batched transformer inference (faster than one-by-one).
    AI detection and entity extraction are sequential (per-article).
    """
    if not articles:
        return []

    log.info(f"[NLP Runner] Enriching batch of {len(articles)} articles…")
    start = time.monotonic()

    # Batched sentiment (most expensive — benefit most from batching)
    texts = [f"{a.title}. {a.title}. {a.body[:300]}" for a in articles]
    sentiments = sentiment_analyser.analyse_batch(texts)

    enriched: list[EnrichedArticle] = []
    for article, sentiment in zip(articles, sentiments):
        full_text = f"{article.title}. {article.body}"

        ai_detection = ai_detector.detect(full_text)
        entities = entity_extractor.extract(article.title, article.body)

        enriched.append(EnrichedArticle(
            article=article,
            sentiment=sentiment,
            ai_detection=ai_detection,
            propagation=PropagationMetrics(),
            named_entities=entities["all_entities"],
            keywords=entities["keywords"],
        ))

    elapsed = time.monotonic() - start
    rate = len(articles) / elapsed if elapsed > 0 else 0
    log.info(
        f"[NLP Runner] Batch complete — {len(enriched)} articles "
        f"in {elapsed:.1f}s ({rate:.1f} articles/sec)"
    )
    return enriched


# ---------------------------------------------------------------------------
# DB persistence for enriched articles
# ---------------------------------------------------------------------------
def _persist_enriched(enriched: EnrichedArticle, repo: ArticleRepository) -> None:
    """Serialise EnrichedArticle and upsert into the database."""
    row = {
        "article_id": enriched.article.id,
        "sentiment_label": enriched.sentiment.label.value if enriched.sentiment else None,
        "sentiment_score": enriched.sentiment.score if enriched.sentiment else None,
        "ai_probability": enriched.ai_detection.ai_probability if enriched.ai_detection else None,
        "is_ai_generated": enriched.ai_detection.is_ai_generated if enriched.ai_detection else False,
        "perplexity_score": enriched.ai_detection.perplexity_score if enriched.ai_detection else None,
        "coordination_score": enriched.propagation.coordination_score if enriched.propagation else 0.0,
        "duplicate_cluster_id": enriched.propagation.duplicate_cluster_id if enriched.propagation else None,
        "named_entities": enriched.named_entities,
        "keywords": enriched.keywords,
        "enriched_at": enriched.enriched_at,
        "full_json": json.loads(enriched.model_dump_json()),
    }
    repo.insert_enriched(row)


# ---------------------------------------------------------------------------
# Analysis runner (continuous loop)
# ---------------------------------------------------------------------------
class NLPAnalysisRunner:
    """
    Continuously reads RawArticles from a queue, enriches them,
    and pushes EnrichedArticles to the output queue.

    Connects the data_ingestion pipeline to the risk_engine.

    Usage
    -----
    >>> runner = NLPAnalysisRunner(input_queue, output_queue)
    >>> await runner.run()
    """

    def __init__(
        self,
        input_queue: Optional[Queue] = None,
        output_queue: Optional[Queue] = None,
        use_transformers: bool = True,
    ):
        self.input_queue: Queue[RawArticle] = input_queue or Queue()
        self.output_queue: Queue[EnrichedArticle] = output_queue or Queue(maxsize=500)

        # Initialise NLP components
        log.info("[NLP Runner] Initialising NLP components…")
        self._sentiment = SentimentAnalyser(use_transformer=use_transformers)
        self._detector = AIContentDetector(
            use_roberta=use_transformers,
            use_gpt2_perplexity=use_transformers,
        )
        self._extractor = EntityExtractor()
        self._repo = ArticleRepository()

        self._processed = 0
        self._ai_detected = 0

    async def run(self) -> None:
        """
        Main processing loop. Reads from input_queue, enriches,
        persists to DB, and writes to output_queue.

        Batches articles in groups of 16 (waits up to 5 seconds for a
        full batch before processing a partial one — keeps latency low).
        """
        log.info("[NLP Runner] Analysis pipeline started")
        batch_size = 16
        batch_timeout = 5.0  # seconds

        while True:
            batch: list[RawArticle] = []

            # Collect a batch (up to batch_size, up to batch_timeout seconds)
            deadline = time.monotonic() + batch_timeout
            while len(batch) < batch_size and time.monotonic() < deadline:
                try:
                    article = await asyncio.wait_for(
                        self.input_queue.get(), timeout=0.5
                    )
                    batch.append(article)
                except asyncio.TimeoutError:
                    break

            if not batch:
                await asyncio.sleep(1.0)
                continue

            enriched_batch = enrich_batch(
                batch, self._sentiment, self._detector, self._extractor
            )

            for enriched in enriched_batch:
                # Track stats
                self._processed += 1
                if enriched.ai_detection and enriched.ai_detection.is_ai_generated:
                    self._ai_detected += 1
                    log.warning(
                        f"[NLP Runner] 🚨 AI-generated content detected! "
                        f"Source: {enriched.article.source_name} | "
                        f"Title: {enriched.article.title[:60]} | "
                        f"AI prob: {enriched.ai_detection.ai_probability:.2%}"
                    )

                # Persist
                try:
                    _persist_enriched(enriched, self._repo)
                except Exception as exc:
                    log.error(f"[NLP Runner] DB persist error: {exc}")

                # Forward to risk engine
                try:
                    self.output_queue.put_nowait(enriched)
                except asyncio.QueueFull:
                    log.warning("[NLP Runner] Output queue full — risk engine may be lagging")

            # Log progress every 50 articles
            if self._processed % 50 == 0:
                ai_rate = self._ai_detected / self._processed if self._processed else 0
                log.info(
                    f"[NLP Runner] Progress — "
                    f"processed={self._processed}, "
                    f"ai_detected={self._ai_detected} ({ai_rate:.1%})"
                )

    async def analyse_once(self, articles: list[RawArticle]) -> list[EnrichedArticle]:
        """
        Analyse a fixed list of articles (non-loop mode).
        Useful for CLI usage and testing.
        """
        return enrich_batch(
            articles, self._sentiment, self._detector, self._extractor
        )


# ---------------------------------------------------------------------------
# Sync wrapper for main.py
# ---------------------------------------------------------------------------
def run_analysis_pipeline() -> None:
    """
    Pull all unprocessed articles from DB and enrich them.
    Called by ``python main.py analyse``.
    """
    from database import ArticleRepository
    configure_logging(level=settings.log_level, log_to_file=True)
    init_db()

    repo = ArticleRepository()
    raw_rows = repo.get_recent_raw(limit=200)

    if not raw_rows:
        log.info("[NLP Runner] No articles to analyse.")
        return

    # Reconstruct RawArticle objects from DB rows
    articles: list[RawArticle] = []
    for row in raw_rows:
        try:
            articles.append(RawArticle(
                id=row["id"],
                source=row["source"],
                source_name=row["source_name"],
                url=row.get("url"),
                title=row["title"],
                body=row.get("body", ""),
                author=row.get("author"),
                published_at=row["published_at"],
                collected_at=row.get("collected_at", datetime.utcnow()),
                raw_metadata=row.get("raw_metadata") or {},
            ))
        except Exception as exc:
            log.warning(f"[NLP Runner] Skipping malformed row: {exc}")

    log.info(f"[NLP Runner] Analysing {len(articles)} articles from DB…")

    runner = NLPAnalysisRunner(use_transformers=False)  # Heuristic mode for CLI
    enriched = asyncio.run(runner.analyse_once(articles))

    # Persist
    enriched_repo = ArticleRepository()
    for e in enriched:
        try:
            _persist_enriched(e, enriched_repo)
        except Exception as exc:
            log.error(f"[NLP Runner] Persist error: {exc}")

    ai_count = sum(1 for e in enriched if e.ai_detection and e.ai_detection.is_ai_generated)
    log.info(
        f"[NLP Runner] Analysis complete — "
        f"{len(enriched)} enriched, "
        f"{ai_count} flagged as AI-generated"
    )


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models import ContentSource
    import uuid

    configure_logging(level="INFO", log_to_file=False)

    DEMO_ARTICLES = [
        RawArticle(
            source=ContentSource.RSS,
            source_name="Reuters",
            title="Tesla Stock Surges 15% After Record Q4 Earnings Beat",
            body=(
                "Tesla Inc shares jumped 15 percent Wednesday after the electric vehicle "
                "maker reported fourth-quarter earnings that crushed analyst expectations. "
                "Revenue hit $25.2 billion, up 19 percent year-over-year. CEO Elon Musk "
                "said the company delivered a record 484,507 vehicles in the quarter."
            ),
            published_at=datetime.utcnow(),
        ),
        RawArticle(
            source=ContentSource.REDDIT,
            source_name="r/wallstreetbets",
            title="YOLO'd my life savings into GME calls. send help",
            body=(
                "ok so i literally put $50k into march GME calls at $25 strike. "
                "wife doesn't know yet. either im gonna be rich af or she's leaving me lol. "
                "diamond hands 💎🙌 anyone else all in on the squeeze?? "
                "wsb has never been wrong before (jk dont listen to me im an idiot)"
            ),
            published_at=datetime.utcnow(),
        ),
        RawArticle(
            source=ContentSource.RSS,
            source_name="Unknown Crypto Blog",
            title="Revolutionary DeFi Protocol Guarantees 1000% Annual Returns",
            body=(
                "The convergence of decentralised finance with advanced cryptographic "
                "algorithms has created an unprecedented wealth generation mechanism. "
                "Our proprietary protocol leverages sophisticated yield optimisation "
                "strategies to facilitate exceptional returns. Furthermore, the "
                "systematic accumulation of value through our tokenomics framework "
                "represents a transformative paradigm in asset management. "
                "It is important to note that early investors have realised substantial "
                "appreciation. This groundbreaking opportunity is available for a "
                "limited time to sophisticated market participants."
            ),
            published_at=datetime.utcnow(),
        ),
    ]

    print("\n🔬 NLP Analysis Pipeline Demo\n")
    print("=" * 70)

    runner = NLPAnalysisRunner(use_transformers=False)
    enriched_list = asyncio.run(runner.analyse_once(DEMO_ARTICLES))

    for enriched in enriched_list:
        a = enriched.article
        s = enriched.sentiment
        d = enriched.ai_detection

        ai_flag = "🚨 AI DETECTED" if (d and d.is_ai_generated) else "✓  Human-likely"
        print(f"\n📰 [{a.source_name}] {a.title[:60]}")
        print(f"   Sentiment:    {s.label.value.upper():10} (conf: {s.score:.2%})")
        print(f"   AI Detector:  {ai_flag}  (prob: {d.ai_probability:.2%})")
        print(f"   Perplexity:   {d.perplexity_score:.1f}  |  Burstiness: {d.burstiness_score:.4f}")
        print(f"   Tickers:      {enriched.named_entities[:5]}")
        print(f"   Keywords:     {enriched.keywords[:4]}")
