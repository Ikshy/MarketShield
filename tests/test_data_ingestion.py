"""
tests/test_data_ingestion.py

Unit tests for the data ingestion pipeline.
Tests use mocks so they run without network access or a real database.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import ContentSource, RawArticle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_article(
    title: str = "Test Article",
    body: str = "Test body content about stocks.",
    source: ContentSource = ContentSource.RSS,
    source_name: str = "Test Feed",
) -> RawArticle:
    content_hash = hashlib.sha256(f"{title}|{body}".encode()).hexdigest()
    return RawArticle(
        source=source,
        source_name=source_name,
        url="https://example.com/test",
        title=title,
        body=body,
        author="Test Author",
        published_at=datetime.now(timezone.utc),
        raw_metadata={"content_hash": content_hash},
    )


# ---------------------------------------------------------------------------
# RawArticle model tests
# ---------------------------------------------------------------------------
class TestRawArticleModel:
    def test_article_creation(self):
        article = make_article()
        assert article.source == ContentSource.RSS
        assert article.title == "Test Article"
        assert article.id is not None  # UUID auto-generated
        assert article.collected_at is not None

    def test_article_id_is_unique(self):
        a1 = make_article()
        a2 = make_article()
        assert a1.id != a2.id

    def test_article_serialises_to_json(self):
        article = make_article()
        data = article.model_dump()
        assert "id" in data
        assert "title" in data
        assert "source" in data


# ---------------------------------------------------------------------------
# RSS Collector tests
# ---------------------------------------------------------------------------
class TestRSSCollector:
    """Tests for RSSCollector using mocked HTTP responses."""

    def _make_feed_xml(self, title: str, link: str, summary: str) -> str:
        return f"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>{title}</title>
      <link>{link}</link>
      <description>{summary}</description>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""

    @pytest.mark.asyncio
    async def test_fetch_all_returns_articles(self):
        from data_ingestion.rss_collector import RSSCollector

        feed_xml = self._make_feed_xml(
            "TSLA Surges 20% on Musk Tweet",
            "https://example.com/tsla-surge",
            "Tesla stock jumped significantly today.",
        )

        mock_response = MagicMock()
        mock_response.text = feed_xml
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            collector = RSSCollector(
                feeds={"Test Feed": "https://example.com/rss"},
                max_per_source=10,
            )
            articles = await collector.fetch_all()

        assert len(articles) >= 1
        assert articles[0].title == "TSLA Surges 20% on Musk Tweet"
        assert articles[0].source == ContentSource.RSS

    @pytest.mark.asyncio
    async def test_deduplication_skips_seen_articles(self):
        from data_ingestion.rss_collector import RSSCollector

        feed_xml = self._make_feed_xml(
            "Duplicate Article",
            "https://example.com/dup",
            "This article will be seen twice.",
        )
        content_hash = hashlib.sha256(
            "Duplicate Article|This article will be seen twice.".encode()
        ).hexdigest()

        mock_response = MagicMock()
        mock_response.text = feed_xml
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            collector = RSSCollector(
                feeds={"Test Feed": "https://example.com/rss"},
                max_per_source=10,
            )
            collector.mark_seen(content_hash)
            articles = await collector.fetch_all()

        assert len(articles) == 0  # Duplicate skipped

    @pytest.mark.asyncio
    async def test_http_error_returns_empty_list(self):
        import httpx
        from data_ingestion.rss_collector import RSSCollector

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            collector = RSSCollector(
                feeds={"Bad Feed": "https://nonexistent.invalid/rss"},
                max_per_source=5,
            )
            articles = await collector.fetch_all()

        assert articles == []  # Graceful failure — no exception raised


# ---------------------------------------------------------------------------
# Reddit Collector tests
# ---------------------------------------------------------------------------
class TestPublicRedditCollector:

    def _make_reddit_response(self, posts: list[dict]) -> dict:
        return {
            "data": {
                "children": [{"data": p} for p in posts]
            }
        }

    @pytest.mark.asyncio
    async def test_fetch_converts_post_to_article(self):
        from data_ingestion.reddit_collector import PublicRedditCollector

        mock_posts = self._make_reddit_response([{
            "id": "abc123",
            "title": "GME to the moon 🚀",
            "selftext": "I bought 1000 shares of GME at open today.",
            "url": "https://reddit.com/r/wallstreetbets/abc123",
            "permalink": "/r/wallstreetbets/comments/abc123",
            "author": "ape_investor",
            "score": 15420,
            "upvote_ratio": 0.96,
            "num_comments": 842,
            "created_utc": 1704067200.0,
            "is_self": True,
            "link_flair_text": "YOLO",
            "gilded": 0,
            "total_awards_received": 12,
        }])

        mock_response = MagicMock()
        mock_response.json.return_value = mock_posts
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            collector = PublicRedditCollector(rate_limit_delay=0)
            articles = await collector.fetch_all(
                subreddits=[{"name": "wallstreetbets", "limit": 5, "priority": 1}]
            )

        assert len(articles) == 1
        assert articles[0].title == "GME to the moon 🚀"
        assert articles[0].source == ContentSource.REDDIT
        assert articles[0].source_name == "r/wallstreetbets"
        assert articles[0].raw_metadata["score"] == 15420

    @pytest.mark.asyncio
    async def test_empty_selftext_uses_url(self):
        from data_ingestion.reddit_collector import _post_to_article

        post = {
            "id": "xyz",
            "title": "Breaking: Fed raises rates",
            "selftext": "",
            "url": "https://reuters.com/fed-raises-rates",
            "permalink": "/r/investing/xyz",
            "author": "fin_news",
            "score": 500,
            "upvote_ratio": 0.89,
            "num_comments": 45,
            "created_utc": 1704067200.0,
            "is_self": False,
            "link_flair_text": "News",
            "gilded": 0,
            "total_awards_received": 0,
        }
        article = _post_to_article(post, "investing")
        assert article is not None
        assert article.body == "https://reuters.com/fed-raises-rates"

    @pytest.mark.asyncio
    async def test_deleted_posts_are_filtered(self):
        from data_ingestion.reddit_collector import _post_to_article

        post = {
            "id": "del1",
            "title": "Some post",
            "selftext": "[deleted]",
            "url": "",
            "permalink": "/r/stocks/del1",
            "author": "[deleted]",
            "score": 0,
            "upvote_ratio": 0.5,
            "num_comments": 0,
            "created_utc": 1704067200.0,
            "is_self": True,
            "link_flair_text": "",
            "gilded": 0,
            "total_awards_received": 0,
        }
        article = _post_to_article(post, "stocks")
        # Deleted + empty URL = no body = should be filtered
        assert article is None


# ---------------------------------------------------------------------------
# Storage Adapter tests
# ---------------------------------------------------------------------------
class TestRawArticleStorage:

    def test_save_one_inserts_new_article(self):
        from data_ingestion.storage_adapter import RawArticleStorage

        mock_repo = MagicMock()
        mock_repo.exists.return_value = False  # Not a duplicate

        storage = RawArticleStorage(repo=mock_repo)
        article = make_article(title="Unique Article About Crypto")
        result = storage.save_one(article)

        assert result is True
        mock_repo.insert_raw.assert_called_once()

    def test_save_one_skips_duplicate(self):
        from data_ingestion.storage_adapter import RawArticleStorage

        mock_repo = MagicMock()
        mock_repo.exists.return_value = True  # Already in DB

        storage = RawArticleStorage(repo=mock_repo)
        article = make_article()
        result = storage.save_one(article)

        assert result is False
        mock_repo.insert_raw.assert_not_called()

    def test_save_batch_returns_stats(self):
        from data_ingestion.storage_adapter import RawArticleStorage

        mock_repo = MagicMock()
        # First two are new, third is duplicate
        mock_repo.exists.side_effect = [False, False, True]

        storage = RawArticleStorage(repo=mock_repo)
        articles = [
            make_article(title=f"Article {i}", body=f"Body {i}")
            for i in range(3)
        ]
        stats = storage.save_batch(articles)

        assert stats.attempted == 3
        assert stats.inserted == 2
        assert stats.skipped_duplicate == 1
        assert stats.errors == 0

    def test_insert_rate_calculation(self):
        from data_ingestion.storage_adapter import InsertStats

        stats = InsertStats(attempted=10, inserted=7, skipped_duplicate=3)
        assert stats.insert_rate == pytest.approx(0.7)

    def test_zero_attempted_insert_rate(self):
        from data_ingestion.storage_adapter import InsertStats

        stats = InsertStats(attempted=0)
        assert stats.insert_rate == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
