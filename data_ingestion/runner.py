"""
data_ingestion/runner.py — Orchestrates the continuous ingestion loop.

This is the top-level coordinator for data collection. It:
  1. Initialises all collectors (RSS + Reddit) on startup
  2. Pre-warms dedup hashes from the DB (so restarts don't re-process)
  3. Runs fetch cycles on configurable intervals
  4. Persists new articles via the storage adapter
  5. Publishes to an in-process queue for downstream pipeline stages
  6. Reports statistics to the logger and (optionally) a metrics endpoint

Run directly:
    python -m data_ingestion.runner
Or via main.py:
    python main.py ingest
"""

from __future__ import annotations

import asyncio
import time
from asyncio import Queue
from datetime import datetime

from config import settings
from logger import get_logger, configure_logging
from models import RawArticle
from data_ingestion.rss_collector import RSSCollector
from data_ingestion.reddit_collector import get_reddit_collector
from data_ingestion.storage_adapter import RawArticleStorage

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# IngestionRunner
# ---------------------------------------------------------------------------
class IngestionRunner:
    """
    Manages the full data ingestion lifecycle.

    Attributes
    ----------
    article_queue : asyncio.Queue
        Articles placed here are consumed by downstream pipeline stages
        (NLP analysis, risk scoring). Bounded at 1000 to apply backpressure.
    """

    def __init__(self, article_queue: Queue | None = None):
        self.article_queue: Queue[RawArticle] = article_queue or Queue(maxsize=1_000)
        self._storage = RawArticleStorage()
        self._rss = RSSCollector()
        self._reddit = get_reddit_collector()

        # Cycle counters for monitoring
        self._rss_cycles = 0
        self._reddit_cycles = 0
        self._total_articles_collected = 0

    async def _prewarm_dedup(self) -> None:
        """
        Load previously seen content hashes from the DB into all collectors.
        This prevents re-processing articles after a restart.
        """
        seen = self._storage.get_seen_hashes(limit=50_000)
        for h in seen:
            self._rss.mark_seen(h)
            self._reddit.mark_seen(h)
        log.info(f"[Runner] Pre-warmed {len(seen)} seen hashes from DB")

    async def _run_rss_cycle(self) -> int:
        """Execute one RSS fetch cycle. Returns number of new articles saved."""
        log.info("[Runner] Starting RSS fetch cycle…")
        start = time.monotonic()

        try:
            articles = await self._rss.fetch_all()
        except Exception as exc:
            log.error(f"[Runner] RSS cycle failed: {exc}")
            return 0

        stats = self._storage.save_batch(articles)
        elapsed = time.monotonic() - start
        self._rss_cycles += 1
        self._total_articles_collected += stats.inserted

        log.info(
            f"[Runner] RSS cycle #{self._rss_cycles} complete — "
            f"{stats.inserted} new articles in {elapsed:.1f}s"
        )

        # Push new articles to the queue for downstream processing
        for article in articles[: stats.inserted]:
            try:
                self.article_queue.put_nowait(article)
            except asyncio.QueueFull:
                log.warning("[Runner] Article queue full — downstream may be lagging")
                break

        return stats.inserted

    async def _run_reddit_cycle(self) -> int:
        """Execute one Reddit fetch cycle. Returns number of new articles saved."""
        log.info("[Runner] Starting Reddit fetch cycle…")
        start = time.monotonic()

        try:
            articles = await self._reddit.fetch_all()
        except Exception as exc:
            log.error(f"[Runner] Reddit cycle failed: {exc}")
            return 0

        stats = self._storage.save_batch(articles)
        elapsed = time.monotonic() - start
        self._reddit_cycles += 1
        self._total_articles_collected += stats.inserted

        log.info(
            f"[Runner] Reddit cycle #{self._reddit_cycles} complete — "
            f"{stats.inserted} new posts in {elapsed:.1f}s"
        )

        for article in articles[: stats.inserted]:
            try:
                self.article_queue.put_nowait(article)
            except asyncio.QueueFull:
                break

        return stats.inserted

    def _print_status(self) -> None:
        """Log a human-readable status summary."""
        log.info(
            f"[Runner] Status — "
            f"RSS cycles: {self._rss_cycles}, "
            f"Reddit cycles: {self._reddit_cycles}, "
            f"Total collected: {self._total_articles_collected}, "
            f"Queue size: {self.article_queue.qsize()}"
        )

    async def run(self) -> None:
        """
        Main ingestion loop. Runs indefinitely until cancelled.

        Schedule:
        - RSS:    every ``settings.rss_fetch_interval_seconds``    (default: 5 min)
        - Reddit: every ``settings.reddit_fetch_interval_seconds`` (default: 10 min)
        """
        log.info("=" * 60)
        log.info("[Runner] MarketShield ingestion pipeline starting")
        log.info(f"[Runner] RSS interval:    {settings.rss_fetch_interval_seconds}s")
        log.info(f"[Runner] Reddit interval: {settings.reddit_fetch_interval_seconds}s")
        log.info("=" * 60)

        await self._prewarm_dedup()

        rss_task = asyncio.create_task(self._rss_loop())
        reddit_task = asyncio.create_task(self._reddit_loop())
        status_task = asyncio.create_task(self._status_loop())

        try:
            await asyncio.gather(rss_task, reddit_task, status_task)
        except asyncio.CancelledError:
            log.info("[Runner] Ingestion pipeline stopped")
            rss_task.cancel()
            reddit_task.cancel()
            status_task.cancel()

    async def _rss_loop(self) -> None:
        """RSS fetch loop — runs immediately, then on interval."""
        while True:
            await self._run_rss_cycle()
            log.debug(f"[Runner] Next RSS cycle in {settings.rss_fetch_interval_seconds}s")
            await asyncio.sleep(settings.rss_fetch_interval_seconds)

    async def _reddit_loop(self) -> None:
        """Reddit fetch loop — waits one interval before first fetch."""
        # Stagger the first Reddit fetch so RSS runs first
        await asyncio.sleep(30)
        while True:
            await self._run_reddit_cycle()
            log.debug(
                f"[Runner] Next Reddit cycle in {settings.reddit_fetch_interval_seconds}s"
            )
            await asyncio.sleep(settings.reddit_fetch_interval_seconds)

    async def _status_loop(self) -> None:
        """Print status every 10 minutes."""
        while True:
            await asyncio.sleep(600)
            self._print_status()

    async def run_once(self) -> dict:
        """
        Run a single fetch cycle of both sources.
        Useful for testing and the CLI ``analyse`` command.

        Returns a summary dict.
        """
        await self._prewarm_dedup()
        rss_count = await self._run_rss_cycle()
        reddit_count = await self._run_reddit_cycle()
        return {
            "rss_new": rss_count,
            "reddit_new": reddit_count,
            "total_new": rss_count + reddit_count,
            "queue_size": self.article_queue.qsize(),
            "timestamp": datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# Sync wrapper (called from main.py which is synchronous)
# ---------------------------------------------------------------------------
def run_ingestion_loop() -> None:
    """Blocking entry point for ``python main.py ingest``."""
    configure_logging(level=settings.log_level, log_to_file=True)
    runner = IngestionRunner()
    asyncio.run(runner.run())


# ---------------------------------------------------------------------------
# Module-level test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    configure_logging(level="DEBUG", log_to_file=False)

    async def _demo():
        runner = IngestionRunner()
        print("\n🔍 Running single ingestion cycle (RSS + Reddit)…\n")
        summary = await runner.run_once()
        print("\n📊 Ingestion Summary:")
        print(json.dumps(summary, indent=2))

        print(f"\n📰 Sample articles in queue ({runner.article_queue.qsize()} total):")
        count = 0
        while not runner.article_queue.empty() and count < 5:
            article = runner.article_queue.get_nowait()
            print(f"\n  Source:  {article.source_name}")
            print(f"  Title:   {article.title[:80]}")
            print(f"  Posted:  {article.published_at.strftime('%Y-%m-%d %H:%M UTC')}")
            if article.source.value == "reddit":
                score = article.raw_metadata.get("score", "?")
                print(f"  Score:   ↑{score}")
            count += 1

    asyncio.run(_demo())
