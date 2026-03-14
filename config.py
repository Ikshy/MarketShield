"""
MarketShield — Central configuration system.

All environment variables and runtime settings are managed here.
Modules import `settings` rather than reading env vars directly,
keeping configuration changes isolated to one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Directory layout (resolved relative to this file)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist at import time
for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """
    Runtime configuration loaded from environment variables or a .env file.

    Override any value by setting the matching environment variable, e.g.:
        export MARKETSHIELD_LOG_LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix="MARKETSHIELD_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    app_name: str = "MarketShield"
    version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── Data ingestion ────────────────────────────────────────────────────
    rss_fetch_interval_seconds: int = Field(300, ge=60)
    reddit_fetch_interval_seconds: int = Field(600, ge=120)
    max_articles_per_source: int = Field(50, ge=1, le=500)

    # Optional API credentials — leave blank to skip that source
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "MarketShield/0.1"

    twitter_bearer_token: str = ""

    # ── NLP / AI detection ────────────────────────────────────────────────
    # Hugging Face model names (downloaded automatically on first use)
    sentiment_model: str = "ProsusAI/finbert"
    ai_detection_model: str = "roberta-base-openai-detector"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Thresholds
    ai_probability_threshold: float = Field(0.70, ge=0.0, le=1.0)
    similarity_dedup_threshold: float = Field(0.85, ge=0.0, le=1.0)

    # ── Market data ───────────────────────────────────────────────────────
    market_fetch_interval_seconds: int = Field(60, ge=10)
    volatility_z_score_threshold: float = Field(2.5, ge=1.0)

    # Tickers to monitor
    stock_tickers: list[str] = ["AAPL", "TSLA", "NVDA", "GME", "AMC"]
    crypto_tickers: list[str] = ["bitcoin", "ethereum", "dogecoin", "solana"]

    # ── Risk engine ───────────────────────────────────────────────────────
    # Weights must sum to 1.0
    risk_weight_ai_probability: float = Field(0.35, ge=0.0, le=1.0)
    risk_weight_propagation_anomaly: float = Field(0.30, ge=0.0, le=1.0)
    risk_weight_market_impact: float = Field(0.35, ge=0.0, le=1.0)

    risk_alert_threshold: float = Field(0.65, ge=0.0, le=1.0)

    # ── Storage ───────────────────────────────────────────────────────────
    db_path: Path = DATA_DIR / "marketshield.db"
    cache_ttl_seconds: int = Field(3600, ge=60)

    # ── API server ────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(8000, ge=1024, le=65535)

    # ── Dashboard ─────────────────────────────────────────────────────────
    dashboard_refresh_seconds: int = Field(30, ge=5)

    @field_validator("stock_tickers", "crypto_tickers", mode="before")
    @classmethod
    def _parse_comma_list(cls, v: str | list) -> list[str]:
        """Allow CSV strings from env vars: MARKETSHIELD_STOCK_TICKERS=AAPL,TSLA"""
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    @property
    def risk_weights_sum(self) -> float:
        return (
            self.risk_weight_ai_probability
            + self.risk_weight_propagation_anomaly
            + self.risk_weight_market_impact
        )


# ---------------------------------------------------------------------------
# Singleton — all modules do: from config import settings
# ---------------------------------------------------------------------------
settings = Settings()
