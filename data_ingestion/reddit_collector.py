"""
data_ingestion/reddit_collector.py — Reddit financial community scraper.

Uses PRAW (Python Reddit API Wrapper) to collect posts from finance-focused
subreddits. Falls back to the public JSON API when no credentials are
configured, ensuring zero-credential operation.

Subreddits monitored:
  - r/wallstreetbets    (retail speculation, coordinated pumps)
  - r/investing         (long-term discussion)
  - r/stocks            (general equities)
  - r/CryptoCurrency    (crypto news and sentiment)
  - r/Bitcoin / r/ethereum
  - r/StockMarket
  - r/SecurityAnalysis  (fundamental research)
  - r/pennystocks       (manipulation hotspot)
  - r/Superstonk        (meme stock community)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import settings
from logger import get_logger
from models import ContentSource, RawArticle

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Subreddit registry
# ---------------------------------------------------------------------------
FINANCIAL_SUBREDDITS: list[dict] = [
    # name, fetch_limit, priority (1=highest)
    {"name": "wallstreetbets",  "limit": 25, "priority": 1},
    {"name": "investing",       "limit": 20, "priority": 1},
    {"name": "stocks",          "limit": 20, "priority": 1},
    {"name": "CryptoCurrency",  "limit": 20, "priority": 1},
    {"name": "Bitcoin",         "limit": 15, "priority": 2},
    {"name": "ethereum",        "limit": 15, "priority": 2},
    {"name": "StockMarket",     "limit": 15, "priority": 2},
    {"name": "SecurityAnalysis","limit": 10, "priority": 2},
    {"name": "pennystocks",     "limit": 20, "priority": 1},  # manipulation risk
    {"name": "Superstonk",      "limit": 15, "priority": 1},  # meme stock
    {"name": "options",         "limit": 10, "priority": 3},
    {"name": "valueinvesting",  "limit": 10, "priority": 3},
    {"name": "Economics",       "limit": 10, "priority": 3},
    {"name": "finance",         "limit": 10, "priority": 3},
]

# Public Reddit JSON API — no credentials needed
_PUBLIC_API_BASE = "https://www.reddit.com/r/{subreddit}/{sort}.json"
_PUBLIC_HEADERS = {
    "User-Agent": settings.reddit_user_agent,
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# Post → RawArticle conversion
# ---------------------------------------------------------------------------
def _post_to_article(post: dict, subreddit: str) -> Optional[RawArticle]:
    """
    Convert a Reddit post dict (from JSON API or PRAW .vars()) to RawArticle.
    Returns None if the post lacks meaningful text content.
    """
    title = post.get("title", "").strip()
    if not title:
        return None

    # Body: use selftext for text posts, URL for link posts
    body = post.get("selftext", "").strip()
    if body in ("[deleted]", "[removed]", ""):
        body = post.get("url", "")

    # Skip posts with no real content
    if not body or len(body) < 10:
        return None

    created_utc = post.get("created_utc", time.time())
    published_at = datetime.fromtimestamp(created_utc, tz=timezone.utc)

    content_hash = hashlib.sha256(f"{title}|{body}".encode()).hexdigest()

    return RawArticle(
        source=ContentSource.REDDIT,
        source_name=f"r/{subreddit}",
        url=f"https://reddit.com{post.get('permalink', '')}",
        title=title,
        body=body[:5000],  # Cap body at 5k chars
        author=post.get("author"),
        published_at=published_at,
        raw_metadata={
            "subreddit": subreddit,
            "score": post.get("score", 0),
            "upvote_ratio": post.get("upvote_ratio", 0.5),
            "num_comments": post.get("num_comments", 0),
            "is_self": post.get("is_self", True),
            "flair": post.get("link_flair_text", ""),
            "content_hash": content_hash,
            "post_id": post.get("id", ""),
            "gilded": post.get("gilded", 0),
            "awards": post.get("total_awards_received", 0),
        },
    )


# ---------------------------------------------------------------------------
# Public JSON API (no credentials required)
# ---------------------------------------------------------------------------
class PublicRedditCollector:
    """
    Fetches Reddit posts using the unauthenticated public JSON API.
    Rate-limited to ~1 req/sec per Reddit's guidelines.
    Activated automatically when no PRAW credentials are configured.
    """

    def __init__(self, rate_limit_delay: float = 1.1):
        self._delay = rate_limit_delay
        self._seen_hashes: set[str] = set()

    def mark_seen(self, content_hash: str) -> None:
        self._seen_hashes.add(content_hash)

    async def _fetch_subreddit(
        self,
        client: httpx.AsyncClient,
        subreddit: str,
        limit: int,
        sort: str = "new",
    ) -> list[RawArticle]:
        """Fetch posts from one subreddit via the JSON API."""
        url = _PUBLIC_API_BASE.format(subreddit=subreddit, sort=sort)
        params = {"limit": min(limit, 100), "t": "day"}

        try:
            resp = await client.get(url, params=params, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning(f"[Reddit] Failed to fetch r/{subreddit}: {exc}")
            return []

        posts = data.get("data", {}).get("children", [])
        articles: list[RawArticle] = []

        for child in posts:
            post = child.get("data", {})
            article = _post_to_article(post, subreddit)
            if article is None:
                continue

            content_hash = article.raw_metadata.get("content_hash", "")
            if content_hash in self._seen_hashes:
                continue

            self._seen_hashes.add(content_hash)
            articles.append(article)

        log.debug(f"[Reddit/public] r/{subreddit} → {len(articles)} posts")
        return articles

    async def fetch_all(
        self,
        subreddits: list[dict] | None = None,
        sort: str = "new",
    ) -> list[RawArticle]:
        """
        Fetch from all configured subreddits with polite rate limiting.

        Note: Sequential (not concurrent) to respect Reddit's public rate limits.
        """
        subs = subreddits or FINANCIAL_SUBREDDITS
        all_articles: list[RawArticle] = []

        async with httpx.AsyncClient(headers=_PUBLIC_HEADERS) as client:
            for sub_config in sorted(subs, key=lambda s: s["priority"]):
                articles = await self._fetch_subreddit(
                    client,
                    sub_config["name"],
                    sub_config["limit"],
                    sort,
                )
                all_articles.extend(articles)
                await asyncio.sleep(self._delay)  # Polite rate limiting

        all_articles.sort(key=lambda a: a.published_at, reverse=True)
        log.info(f"[Reddit/public] Fetched {len(all_articles)} posts total")
        return all_articles


# ---------------------------------------------------------------------------
# PRAW-based collector (authenticated — faster, higher limits)
# ---------------------------------------------------------------------------
class PRAWRedditCollector:
    """
    Authenticated Reddit collector using PRAW.
    Requires MARKETSHIELD_REDDIT_CLIENT_ID and CLIENT_SECRET in .env.

    Advantages over public API:
    - Higher rate limits (60 req/min)
    - Access to more post metadata
    - Stream mode for real-time posts
    """

    def __init__(self):
        try:
            import praw  # type: ignore
        except ImportError:
            raise ImportError("Install praw: pip install praw")

        self._reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
            check_for_async=False,
        )
        self._seen_hashes: set[str] = set()
        log.info("[Reddit/PRAW] Authenticated client ready")

    def mark_seen(self, content_hash: str) -> None:
        self._seen_hashes.add(content_hash)

    def _submission_to_dict(self, submission) -> dict:
        """Convert a PRAW Submission to a plain dict."""
        return {
            "id": submission.id,
            "title": submission.title,
            "selftext": submission.selftext,
            "url": submission.url,
            "author": str(submission.author) if submission.author else "[deleted]",
            "permalink": submission.permalink,
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "created_utc": submission.created_utc,
            "is_self": submission.is_self,
            "link_flair_text": submission.link_flair_text,
            "gilded": submission.gilded,
            "total_awards_received": submission.total_awards_received,
        }

    async def fetch_all(
        self,
        subreddits: list[dict] | None = None,
        sort: str = "new",
    ) -> list[RawArticle]:
        """Fetch posts from all configured subreddits using PRAW."""
        subs = subreddits or FINANCIAL_SUBREDDITS
        all_articles: list[RawArticle] = []

        # PRAW is synchronous — run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        for sub_config in subs:
            subreddit_name = sub_config["name"]
            limit = sub_config["limit"]

            def _fetch_sync(name=subreddit_name, lim=limit):
                sub = self._reddit.subreddit(name)
                fetcher = getattr(sub, sort, sub.new)
                return [self._submission_to_dict(s) for s in fetcher(limit=lim)]

            try:
                posts = await loop.run_in_executor(None, _fetch_sync)
            except Exception as exc:
                log.warning(f"[Reddit/PRAW] Error on r/{subreddit_name}: {exc}")
                continue

            for post in posts:
                article = _post_to_article(post, subreddit_name)
                if article is None:
                    continue
                content_hash = article.raw_metadata.get("content_hash", "")
                if content_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(content_hash)
                all_articles.append(article)

        all_articles.sort(key=lambda a: a.published_at, reverse=True)
        log.info(f"[Reddit/PRAW] Fetched {len(all_articles)} posts total")
        return all_articles


# ---------------------------------------------------------------------------
# Factory — picks the right collector based on available credentials
# ---------------------------------------------------------------------------
def get_reddit_collector() -> PublicRedditCollector | PRAWRedditCollector:
    """
    Return the best available Reddit collector.

    If Reddit API credentials are set in config, returns PRAWRedditCollector
    (higher limits). Otherwise falls back to the public JSON API.
    """
    if settings.reddit_client_id and settings.reddit_client_secret:
        try:
            collector = PRAWRedditCollector()
            log.info("[Reddit] Using authenticated PRAW collector")
            return collector
        except Exception as exc:
            log.warning(f"[Reddit] PRAW init failed ({exc}), falling back to public API")

    log.info("[Reddit] Using public JSON API (no credentials configured)")
    return PublicRedditCollector()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def _demo():
        collector = PublicRedditCollector()
        # Test with just two subreddits to keep demo fast
        demo_subs = [
            {"name": "investing", "limit": 5, "priority": 1},
            {"name": "CryptoCurrency", "limit": 5, "priority": 1},
        ]
        articles = await collector.fetch_all(subreddits=demo_subs)
        print(f"\nFetched {len(articles)} Reddit posts\n")
        for a in articles:
            score = a.raw_metadata.get("score", 0)
            print(f"  [{a.source_name}] ↑{score:>5}  {a.title[:70]}")
        print()

    asyncio.run(_demo())
