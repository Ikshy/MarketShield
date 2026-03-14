"""
MarketShield — Persistent storage layer.

Uses SQLite via SQLAlchemy Core (no ORM overhead) for:
- Raw article cache (avoids re-processing duplicates)
- Enriched article store
- Risk events log
- Market snapshots

Design choice: SQLite is zero-infrastructure and perfectly adequate for
a research system processing hundreds of articles per hour. Switch the
connection URL to PostgreSQL for production scale.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.engine import Engine

from config import settings
from logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine (singleton)
# ---------------------------------------------------------------------------
_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        db_url = f"sqlite:///{settings.db_path}"
        _engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            echo=settings.debug,
        )
        log.info(f"Database engine created: {db_url}")
    return _engine


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
metadata = MetaData()

# One row per raw article (dedup check by URL / content hash)
raw_articles_table = Table(
    "raw_articles",
    metadata,
    Column("id", String, primary_key=True),
    Column("source", String, nullable=False),
    Column("source_name", String),
    Column("url", String),
    Column("title", Text, nullable=False),
    Column("body", Text),
    Column("author", String),
    Column("published_at", DateTime),
    Column("collected_at", DateTime, default=datetime.utcnow),
    Column("raw_metadata", JSON, default={}),
    Column("content_hash", String, index=True),   # SHA256 of title+body
)

# Enriched articles with NLP scores
enriched_articles_table = Table(
    "enriched_articles",
    metadata,
    Column("article_id", String, primary_key=True),
    Column("sentiment_label", String),
    Column("sentiment_score", Float),
    Column("ai_probability", Float),
    Column("is_ai_generated", Boolean),
    Column("perplexity_score", Float),
    Column("coordination_score", Float),
    Column("duplicate_cluster_id", String),
    Column("named_entities", JSON, default=[]),
    Column("keywords", JSON, default=[]),
    Column("enriched_at", DateTime),
    Column("full_json", JSON),   # Full EnrichedArticle for reconstruction
)

# Risk scores
risk_events_table = Table(
    "risk_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("article_id", String, index=True),
    Column("composite_score", Float),
    Column("risk_level", String),
    Column("ai_component", Float),
    Column("propagation_component", Float),
    Column("market_impact_component", Float),
    Column("explanation", Text),
    Column("related_tickers", JSON, default=[]),
    Column("scored_at", DateTime),
)

# Market price snapshots
market_snapshots_table = Table(
    "market_snapshots",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String, index=True),
    Column("asset_type", String),
    Column("price", Float),
    Column("volume", Float),
    Column("price_change_pct_1h", Float),
    Column("price_change_pct_24h", Float),
    Column("timestamp", DateTime, index=True),
)

# Manipulation alerts (high-risk events)
alerts_table = Table(
    "alerts",
    metadata,
    Column("alert_id", String, primary_key=True),
    Column("title", Text),
    Column("summary", Text),
    Column("composite_score", Float),
    Column("risk_level", String),
    Column("article_id", String),
    Column("related_tickers", JSON, default=[]),
    Column("created_at", DateTime, index=True),
    Column("full_json", JSON),
)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    engine = get_engine()
    metadata.create_all(engine)
    log.info("Database schema initialised.")


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------
class ArticleRepository:
    """CRUD helpers for raw and enriched articles."""

    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or get_engine()

    def exists(self, content_hash: str) -> bool:
        """Return True if an article with this hash is already stored."""
        with self.engine.connect() as conn:
            row = conn.execute(
                select(raw_articles_table.c.id).where(
                    raw_articles_table.c.content_hash == content_hash
                )
            ).first()
        return row is not None

    def insert_raw(self, article_dict: dict, content_hash: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                raw_articles_table.insert().values(
                    **article_dict, content_hash=content_hash
                )
            )

    def insert_enriched(self, enriched_dict: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                enriched_articles_table.insert().prefix_with("OR REPLACE").values(
                    **enriched_dict
                )
            )

    def get_recent_raw(self, limit: int = 100) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(raw_articles_table)
                .order_by(raw_articles_table.c.collected_at.desc())
                .limit(limit)
            ).mappings().all()
        return [dict(r) for r in rows]

    def get_recent_enriched(self, limit: int = 50) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(enriched_articles_table)
                .order_by(enriched_articles_table.c.enriched_at.desc())
                .limit(limit)
            ).mappings().all()
        return [dict(r) for r in rows]


class RiskRepository:
    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or get_engine()

    def insert_risk_event(self, risk_dict: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(risk_events_table.insert().values(**risk_dict))

    def insert_alert(self, alert_dict: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                alerts_table.insert().prefix_with("OR REPLACE").values(**alert_dict)
            )

    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(alerts_table)
                .order_by(alerts_table.c.created_at.desc())
                .limit(limit)
            ).mappings().all()
        return [dict(r) for r in rows]


class MarketRepository:
    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine or get_engine()

    def insert_snapshot(self, snapshot_dict: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(market_snapshots_table.insert().values(**snapshot_dict))

    def get_ticker_history(self, ticker: str, limit: int = 200) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(market_snapshots_table)
                .where(market_snapshots_table.c.ticker == ticker)
                .order_by(market_snapshots_table.c.timestamp.desc())
                .limit(limit)
            ).mappings().all()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Convenience init on import
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    print("Database initialised successfully.")
