"""
data_ingestion/storage_adapter.py — Persists raw articles to the database.

Sits between collectors and the database layer. Responsibilities:
  1. Deduplication check before every insert
  2. Serialise RawArticle → DB row dict
  3. Report insertion statistics
  4. Sync seen-hash sets back to collectors

This is the only module that imports from ``database``,
keeping collectors completely I/O-agnostic and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from database import ArticleRepository, init_db
from logger import get_logger
from models import RawArticle

log = get_logger(__name__)


@dataclass
class InsertStats:
    """Summary returned after a batch insert."""
    attempted: int = 0
    inserted: int = 0
    skipped_duplicate: int = 0
    errors: int = 0

    @property
    def insert_rate(self) -> float:
        if self.attempted == 0:
            return 0.0
        return self.inserted / self.attempted


class RawArticleStorage:
    """
    Persists RawArticle objects with deduplication.

    Usage
    -----
    >>> storage = RawArticleStorage()
    >>> stats = storage.save_batch(articles)
    >>> print(f"Saved {stats.inserted} / {stats.attempted}")
    """

    def __init__(self, repo: ArticleRepository | None = None):
        init_db()
        self._repo = repo or ArticleRepository()

    def save_one(self, article: RawArticle) -> bool:
        """
        Persist a single article. Returns True if inserted, False if duplicate.
        """
        content_hash: str = article.raw_metadata.get(
            "content_hash",
            _compute_hash(article),
        )

        if self._repo.exists(content_hash):
            return False

        row = _article_to_row(article, content_hash)
        try:
            self._repo.insert_raw(row, content_hash)
            return True
        except Exception as exc:
            log.error(f"[Storage] Failed to insert article '{article.title[:50]}': {exc}")
            return False

    def save_batch(self, articles: list[RawArticle]) -> InsertStats:
        """
        Persist a list of articles, skipping duplicates.
        Returns an InsertStats summary.
        """
        stats = InsertStats(attempted=len(articles))

        for article in articles:
            content_hash = article.raw_metadata.get(
                "content_hash", _compute_hash(article)
            )

            if self._repo.exists(content_hash):
                stats.skipped_duplicate += 1
                continue

            row = _article_to_row(article, content_hash)
            try:
                self._repo.insert_raw(row, content_hash)
                stats.inserted += 1
            except Exception as exc:
                log.error(
                    f"[Storage] Insert error for '{article.title[:40]}': {exc}"
                )
                stats.errors += 1

        log.info(
            f"[Storage] Batch complete — "
            f"inserted={stats.inserted}, "
            f"duplicates={stats.skipped_duplicate}, "
            f"errors={stats.errors} "
            f"(of {stats.attempted} attempted)"
        )
        return stats

    def get_seen_hashes(self, limit: int = 10_000) -> set[str]:
        """
        Load content hashes already in the DB.
        Used to pre-warm collector dedup sets on startup.
        """
        rows = self._repo.get_recent_raw(limit=limit)
        return {r["content_hash"] for r in rows if r.get("content_hash")}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import hashlib  # noqa: E402  (import after class definition is fine here)


def _compute_hash(article: RawArticle) -> str:
    return hashlib.sha256(
        f"{article.title}|{article.body[:500]}".encode()
    ).hexdigest()


def _article_to_row(article: RawArticle, content_hash: str) -> dict:
    """Convert RawArticle → DB row dict (matches raw_articles_table schema)."""
    return {
        "id": article.id,
        "source": article.source.value,
        "source_name": article.source_name,
        "url": str(article.url) if article.url else None,
        "title": article.title,
        "body": article.body,
        "author": article.author,
        "published_at": article.published_at,
        "collected_at": article.collected_at,
        "raw_metadata": article.raw_metadata,
        "content_hash": content_hash,
    }
