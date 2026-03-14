"""
MarketShield — Shared data models.

Pydantic models act as the "contracts" between pipeline stages.
Every module imports from here; no module defines its own ad-hoc dicts.

Pipeline flow:
    RawArticle → EnrichedArticle → RiskEvent
    MarketSnapshot → VolatilityAlert
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ContentSource(str, Enum):
    RSS = "rss"
    REDDIT = "reddit"
    TWITTER = "twitter"
    MANUAL = "manual"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Stage 1 — Raw ingestion
# ---------------------------------------------------------------------------
class RawArticle(BaseModel):
    """
    Represents one piece of content straight from a data source,
    before any enrichment or analysis.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: ContentSource
    source_name: str                        # e.g. "Reuters RSS", "r/wallstreetbets"
    url: Optional[str] = None
    title: str
    body: str
    author: Optional[str] = None
    published_at: datetime
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    raw_metadata: dict = Field(default_factory=dict)  # Source-specific extras

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ---------------------------------------------------------------------------
# Stage 2 — NLP enrichment
# ---------------------------------------------------------------------------
class SentimentResult(BaseModel):
    label: Sentiment
    score: float = Field(ge=0.0, le=1.0)   # Confidence in the label
    positive: float = Field(ge=0.0, le=1.0)
    negative: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)


class AIDetectionResult(BaseModel):
    ai_probability: float = Field(ge=0.0, le=1.0)
    is_ai_generated: bool
    perplexity_score: float
    burstiness_score: float
    method: str = "ensemble"                # Which detector(s) fired


class PropagationMetrics(BaseModel):
    duplicate_cluster_id: Optional[str] = None
    cluster_size: int = 1
    is_coordinated: bool = False
    coordination_score: float = Field(default=0.0, ge=0.0, le=1.0)
    spread_velocity: float = 0.0            # Articles/hour in the cluster


class EnrichedArticle(BaseModel):
    """
    An article after all NLP and graph analysis stages have run.
    """

    article: RawArticle
    sentiment: Optional[SentimentResult] = None
    ai_detection: Optional[AIDetectionResult] = None
    propagation: Optional[PropagationMetrics] = None
    named_entities: list[str] = Field(default_factory=list)  # Tickers, people, orgs
    keywords: list[str] = Field(default_factory=list)
    enriched_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Stage 3 — Market data
# ---------------------------------------------------------------------------
class MarketSnapshot(BaseModel):
    ticker: str
    asset_type: str                         # "stock" | "crypto"
    price: float
    volume: float
    price_change_pct_1h: Optional[float] = None
    price_change_pct_24h: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class VolatilityAlert(BaseModel):
    ticker: str
    z_score: float
    current_price: float
    baseline_mean: float
    baseline_std: float
    direction: str                          # "up" | "down"
    triggered_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Stage 4 — Risk scoring
# ---------------------------------------------------------------------------
class RiskScore(BaseModel):
    """
    Composite risk score for a piece of content + its market context.
    """

    article_id: str
    composite_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel

    # Component scores (each 0–1)
    ai_component: float = Field(ge=0.0, le=1.0)
    propagation_component: float = Field(ge=0.0, le=1.0)
    market_impact_component: float = Field(ge=0.0, le=1.0)

    # Human-readable explanation
    explanation: str = ""
    related_tickers: list[str] = Field(default_factory=list)
    scored_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def _set_risk_level(self) -> "RiskScore":
        s = self.composite_score
        if s >= 0.85:
            self.risk_level = RiskLevel.CRITICAL
        elif s >= 0.65:
            self.risk_level = RiskLevel.HIGH
        elif s >= 0.40:
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW
        return self


# ---------------------------------------------------------------------------
# Stage 5 — Alert / event (final output)
# ---------------------------------------------------------------------------
class ManipulationAlert(BaseModel):
    """
    Top-level alert emitted when risk exceeds the configured threshold.
    Consumed by the dashboard and (optionally) external webhooks.
    """

    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    summary: str
    risk_score: RiskScore
    enriched_article: EnrichedArticle
    volatility_alerts: list[VolatilityAlert] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
