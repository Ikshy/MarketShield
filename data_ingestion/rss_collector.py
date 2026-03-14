"""
data_ingestion/rss_collector.py — Async RSS feed scraper.

Fetches articles from 20+ financial news sources concurrently using
asyncio + httpx. Each article is normalised into a RawArticle model
before being handed to the storage layer.

Sources span:
  - Major financial news (Reuters, Bloomberg, WSJ, FT, MarketWatch)
  - Crypto-focused (CoinDesk, CoinTelegraph, Decrypt)
  - Analyst / research feeds (Seeking Alpha, Motley Fool)
  - Regulatory (SEC, CFTC press releases)
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import AsyncIterator

import feedparser
import httpx
from dateutil import parser as dateparser
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from logger import get_logger
from models import ContentSource, RawArticle

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feed registry — add/remove feeds here without touching any other code
# ---------------------------------------------------------------------------
RSS_FEEDS: dict[str, str] = {
    # ── Major financial news ─────────────────────────────────────────────
    "Reuters Business":
        "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Finance":
        "https://feeds.reuters.com/reuters/financialNews",
    "MarketWatch Top Stories":
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    "MarketWatch Market Pulse":
        "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "CNBC Finance":
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "CNBC Investing":
        "https://www.cnbc.com/id/15839069/device/rss/rss.html",
    "Seeking Alpha":
        "https://seekingalpha.com/feed.xml",
    "Investopedia News":
        "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline",
    "Yahoo Finance":
        "https://finance.yahoo.com/news/rssindex",
    "Motley Fool":
        "https://www.fool.com/feeds/index.aspx",
    "Barrons":
        "https://www.barrons.com/xml/rss/3_7531.xml",
    "Forbes Investing":
        "https://www.forbes.com/investing/feed/",
    "The Street":
        "https://www.thestreet.com/.rss/full/",

    # ── Crypto ────────────────────────────────────────────────────────────
    "CoinDesk":
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph":
        "https://cointelegraph.com/rss",
    "Decrypt":
        "https://decrypt.co/feed",
    "Bitcoin Magazine":
        "https://bitcoinmagazine.com/.rss/full/",
    "The Block":
        "https://www.theblock.co/rss.xml",

    # ── Regulatory / government ───────────────────────────────────────────
    "SEC Press Releases":
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&dateb=&owner=include&count=40&search_text=&output=atom",
    "CFTC News":
        "https://www.cftc.gov/rss/pressreleases.xml",
    "Federal Reserve":
        "https://www.federalreserve.gov/feeds/press_all.xml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _content_hash(title: str, body: str) -> str:
    """SHA-256 fingerprint used for deduplication."""
    return hashlib.sha256(f"{title}|{body}".encode()).hexdigest()


def _parse_date(entry: feedparser.FeedParserDict) -> datetime:
    """
    Extract publication datetime from a feedparser entry.
    Falls back to utcnow() if no date fields are present.
    """
    for attr in ("published", "updated", "created"):
        raw = getattr(entry, attr, None)
        if raw:
            try:
                return dateparser.parse(raw).replace(tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def _extract_body(entry: feedparser.FeedParserDict) -> str:
    """Pull the best available text body from a feed entry."""
    # Try content first (full text), then summary
    if hasattr(entry, "content") and entry.content:
        return entry.content[0].get("value", "")
    if hasattr(entry, "summary"):
        return entry.summary
    return ""


# ---------------------------------------------------------------------------
# Core collector
# ---------------------------------------------------------------------------
class RSSCollector:
    """
    Fetches articles from all registered RSS feeds concurrently.

    Usage
    -----
    >>> collector = RSSCollector()
    >>> articles = await collector.fetch_all()
    >>> print(f"Collected {len(articles)} articles")

    The collector respects ``settings.max_articles_per_source`` and skips
    items that lack a title or that have already been seen (tracked via the
    in-memory ``seen_hashes`` set, which is also persisted to the DB).
    """

    def __init__(
        self,
        feeds: dict[str, str] | None = None,
        max_per_source: int | None = None,
        timeout: float = 15.0,
    ):
        self.feeds = feeds or RSS_FEEDS
        self.max_per_source = max_per_source or settings.max_articles_per_source
        self.timeout = timeout
        self._seen_hashes: set[str] = set()

    def mark_seen(self, content_hash: str) -> None:
        """Register a hash as already processed (called by storage layer)."""
        self._seen_hashes.add(content_hash)

    # ── Single-feed fetch ─────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=False,
    )
    async def _fetch_feed(
        self,
        client: httpx.AsyncClient,
        name: str,
        url: str,
    ) -> list[RawArticle]:
        """
        Download and parse a single RSS feed URL.
        Returns a list of RawArticle objects (may be empty on error).
        """
        try:
            response = await client.get(url, timeout=self.timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning(f"[RSS] HTTP error fetching '{name}': {exc}")
            return []

        feed = feedparser.parse(response.text)

        if feed.bozo and not feed.entries:
            log.warning(f"[RSS] Malformed feed '{name}' — skipping")
            return []

        articles: list[RawArticle] = []
        for entry in feed.entries[: self.max_per_source]:
            title = getattr(entry, "title", "").strip()
            body = _extract_body(entry).strip()

            if not title:
                continue

            content_hash = _content_hash(title, body)
            if content_hash in self._seen_hashes:
                continue

            article = RawArticle(
                source=ContentSource.RSS,
                source_name=name,
                url=getattr(entry, "link", None),
                title=title,
                body=body,
                author=getattr(entry, "author", None),
                published_at=_parse_date(entry),
                raw_metadata={
                    "feed_url": url,
                    "tags": [t.get("term", "") for t in getattr(entry, "tags", [])],
                    "content_hash": content_hash,
                },
            )
            self._seen_hashes.add(content_hash)
            articles.append(article)

        log.debug(f"[RSS] '{name}' → {len(articles)} new articles")
        return articles

    # ── All feeds (concurrent) ────────────────────────────────────────────
    async def fetch_all(self) -> list[RawArticle]:
        """
        Fetch all registered feeds concurrently.

        Returns all new articles across all sources, sorted newest-first.
        """
        headers = {
            "User-Agent": (
                "MarketShield/0.1 (financial misinformation research; "
                "https://github.com/yourusername/MarketShield)"
            )
        }

        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            tasks = [
                self._fetch_feed(client, name, url)
                for name, url in self.feeds.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles: list[RawArticle] = []
        for result in results:
            if isinstance(result, Exception):
                log.error(f"[RSS] Unhandled error in feed task: {result}")
                continue
            all_articles.extend(result)

        # Sort newest-first
        all_articles.sort(key=lambda a: a.published_at, reverse=True)

        log.info(
            f"[RSS] Fetch complete — {len(all_articles)} new articles "
            f"from {len(self.feeds)} feeds"
        )
        return all_articles

    # ── Streaming variant (for pipeline integration) ──────────────────────
    async def stream_articles(self) -> AsyncIterator[RawArticle]:
        """
        Yield articles one by one as feeds respond.
        Useful for large fetch batches where you want to start
        processing before all feeds have returned.
        """
        headers = {"User-Agent": "MarketShield/0.1"}
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            tasks = {
                asyncio.create_task(self._fetch_feed(client, name, url)): name
                for name, url in self.feeds.items()
            }
            for coro in asyncio.as_completed(tasks):
                try:
                    articles = await coro
                    for article in articles:
                        yield article
                except Exception as exc:
                    log.error(f"[RSS] Stream error: {exc}")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    async def _demo():
        collector = RSSCollector(max_per_source=3)
        articles = await collector.fetch_all()
        print(f"\nFetched {len(articles)} articles\n")
        for a in articles[:5]:
            print(f"  [{a.source_name}] {a.title[:80]}")
            print(f"    URL: {a.url}")
            print(f"    Published: {a.published_at}")
            print()

    asyncio.run(_demo())
