"""
ai_detection/entity_extractor.py — Named entity + ticker extraction.

Pulls structured signals from raw financial text:
  - Stock tickers (AAPL, TSLA, etc.) via regex + known ticker list
  - Crypto names (bitcoin, ethereum, etc.)
  - Company names via spaCy NER
  - Person names (executives, regulators)
  - Financial keywords (earnings, SEC, short squeeze, etc.)

These entities are attached to EnrichedArticle and used by the risk engine
to correlate news with specific market instruments.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from config import settings
from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Known ticker set (fast lookup)
# ---------------------------------------------------------------------------
# Top ~200 most-discussed tickers in financial media + configured tickers
_WELL_KNOWN_TICKERS = {
    # Mega-cap + meme stocks
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "NVDA", "META",
    "NFLX", "AMD", "INTC", "BABA", "ORCL", "CRM", "ADBE",
    "GME", "AMC", "BB", "NOK", "BBBY", "CLOV", "MVIS", "SPCE",
    # Finance
    "JPM", "GS", "MS", "BAC", "C", "WFC", "BRK.A", "BRK.B",
    # ETFs
    "SPY", "QQQ", "IWM", "VTI", "GLD", "SLV", "USO", "TLT",
    # Crypto-adjacent stocks
    "COIN", "MSTR", "RIOT", "MARA", "HUT",
    # Pharma / biotech
    "PFE", "MRNA", "BNTX", "JNJ", "ABBV",
    # Energy
    "XOM", "CVX", "OXY",
}

_CRYPTO_NAMES = {
    "bitcoin", "btc", "ethereum", "eth", "dogecoin", "doge",
    "solana", "sol", "cardano", "ada", "ripple", "xrp",
    "polygon", "matic", "avalanche", "avax", "chainlink", "link",
    "shiba inu", "shib", "pepe", "floki", "bonk",
    "tether", "usdc", "usdt", "dai",
    "binance coin", "bnb",
}

# High-signal financial keywords for risk scoring
_FINANCIAL_KEYWORDS = {
    # Manipulation signals
    "short squeeze", "gamma squeeze", "pump and dump", "coordinated buy",
    "short sellers", "hedge fund", "naked short",
    # Regulatory
    "sec investigation", "sec charges", "fraud", "insider trading",
    "market manipulation", "class action", "subpoena",
    # Sentiment-amplifying events
    "earnings beat", "earnings miss", "guidance raised", "guidance cut",
    "acquisition", "merger", "buyout", "takeover", "bankruptcy",
    "going private", "ipo", "spac",
    # Volatility signals
    "circuit breaker", "trading halt", "halted", "suspended",
    "all-time high", "all-time low", "crash", "collapse", "rally",
    "options expiration", "gamma exposure",
}


# ---------------------------------------------------------------------------
# spaCy loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_spacy():
    """Load spaCy en_core_web_sm model (cached after first call)."""
    try:
        import spacy  # type: ignore
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        log.info("[EntityExtractor] spaCy NER model loaded")
        return nlp
    except (ImportError, OSError) as exc:
        log.warning(f"[EntityExtractor] spaCy unavailable ({exc}). Using regex-only mode.")
        return None


# ---------------------------------------------------------------------------
# Ticker extraction
# ---------------------------------------------------------------------------
def _extract_tickers(text: str) -> list[str]:
    """
    Extract stock tickers from text using two strategies:

    1. Regex pattern: $AAPL or (AAPL) or "AAPL stock"
    2. Known ticker list: scan words for exact matches

    Returns deduplicated list of uppercase tickers.
    """
    found: set[str] = set()

    # Pattern 1: $TICKER format (Reddit / Twitter convention)
    dollar_tickers = re.findall(r"\$([A-Z]{1,5})\b", text)
    found.update(dollar_tickers)

    # Pattern 2: parenthesised tickers — "Apple Inc. (AAPL)"
    paren_tickers = re.findall(r"\(([A-Z]{1,5})\)", text)
    found.update(t for t in paren_tickers if len(t) >= 2)

    # Pattern 3: "TICKER stock/shares/calls/puts"
    inline_tickers = re.findall(
        r"\b([A-Z]{2,5})\s+(?:stock|shares?|calls?|puts?|options?|equity)\b",
        text,
    )
    found.update(inline_tickers)

    # Pattern 4: Match against known ticker set (whole-word, uppercase)
    words_upper = set(re.findall(r"\b[A-Z]{2,5}\b", text))
    found.update(words_upper & _WELL_KNOWN_TICKERS)

    # Add configured tickers if mentioned (case-insensitive)
    text_lower = text.lower()
    for ticker in settings.stock_tickers:
        if ticker.lower() in text_lower:
            found.add(ticker)

    # Filter obvious false positives (common words that look like tickers)
    _STOPWORDS = {"I", "A", "AN", "THE", "FOR", "OR", "AND", "TO",
                  "IN", "OF", "ON", "AT", "BY", "IT", "IS", "AS",
                  "US", "UK", "EU", "UN", "AI", "CEO", "CFO", "CTO",
                  "IPO", "ETF", "GDP", "CPI", "FED", "IMF"}
    found -= _STOPWORDS

    return sorted(found)


# ---------------------------------------------------------------------------
# Crypto extraction
# ---------------------------------------------------------------------------
def _extract_crypto(text: str) -> list[str]:
    """Extract cryptocurrency names and symbols from text."""
    text_lower = text.lower()
    found: set[str] = set()

    for name in _CRYPTO_NAMES:
        if name in text_lower:
            found.add(name)

    # Also check configured crypto tickers
    for ticker in settings.crypto_tickers:
        if ticker.lower() in text_lower:
            found.add(ticker)

    return sorted(found)


# ---------------------------------------------------------------------------
# Financial keyword extraction
# ---------------------------------------------------------------------------
def _extract_keywords(text: str) -> list[str]:
    """Extract high-signal financial keywords from text."""
    text_lower = text.lower()
    return sorted(kw for kw in _FINANCIAL_KEYWORDS if kw in text_lower)


# ---------------------------------------------------------------------------
# spaCy NER (organisations, people)
# ---------------------------------------------------------------------------
def _extract_spacy_entities(text: str) -> dict[str, list[str]]:
    """
    Use spaCy to extract named entities.
    Falls back to empty lists if spaCy is unavailable.

    Returns dict with keys: organisations, people, locations
    """
    nlp = _load_spacy()
    if nlp is None:
        return {"organisations": [], "people": [], "locations": []}

    # Process first 2000 chars (NER is O(n²) — cap for performance)
    doc = nlp(text[:2000])

    orgs: set[str] = set()
    people: set[str] = set()
    locs: set[str] = set()

    for ent in doc.ents:
        if ent.label_ == "ORG" and len(ent.text) > 2:
            orgs.add(ent.text.strip())
        elif ent.label_ == "PERSON" and len(ent.text) > 3:
            people.add(ent.text.strip())
        elif ent.label_ in ("GPE", "LOC") and len(ent.text) > 2:
            locs.add(ent.text.strip())

    return {
        "organisations": sorted(orgs)[:10],
        "people": sorted(people)[:10],
        "locations": sorted(locs)[:5],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class EntityExtractor:
    """
    Extracts tickers, crypto names, organisations, people and keywords
    from financial article text.

    Usage
    -----
    >>> extractor = EntityExtractor()
    >>> result = extractor.extract("TSLA stock crashes after Musk sells $5B in shares")
    >>> result["tickers"]  # ['TSLA']
    >>> result["people"]   # ['Musk']
    """

    def extract(self, title: str, body: str = "") -> dict:
        """
        Full extraction pipeline for one article.

        Returns a dict with:
          tickers, cryptos, organisations, people, keywords,
          all_entities (flat list for easy storage)
        """
        full_text = f"{title}. {body}"

        tickers  = _extract_tickers(full_text)
        cryptos  = _extract_crypto(full_text)
        keywords = _extract_keywords(full_text.lower())
        ner      = _extract_spacy_entities(full_text)

        # Combine tickers + crypto into a flat list of financial instruments
        instruments = list(set(tickers + [c.upper() for c in cryptos]))

        all_entities = list(set(
            tickers
            + cryptos
            + ner["organisations"]
            + ner["people"]
            + keywords
        ))

        return {
            "tickers": tickers,
            "cryptos": cryptos,
            "instruments": instruments,
            "organisations": ner["organisations"],
            "people": ner["people"],
            "locations": ner["locations"],
            "keywords": keywords,
            "all_entities": all_entities[:30],  # Cap for storage
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    SAMPLES = [
        (
            "TSLA Short Squeeze Alert",
            "Tesla (TSLA) shares surged 18% today as short sellers scrambled to cover positions. "
            "Elon Musk announced a $5 billion stock buyback program. Options traders reported "
            "massive gamma exposure near the $250 strike. SEC is reportedly investigating "
            "possible market manipulation following coordinated buying on Reddit."
        ),
        (
            "Bitcoin Crash on Regulation News",
            "$BTC collapsed 30% after Chinese regulators announced a complete ban on "
            "cryptocurrency trading. Ethereum and Dogecoin followed with similar losses. "
            "Coinbase (COIN) shares fell 20% in after-hours trading. The crypto market "
            "lost over $500 billion in market cap within 24 hours."
        ),
        (
            "GME Earnings Beat",
            "GameStop (GME) reported earnings that beat analyst expectations by 40%. "
            "Ryan Cohen announced a new strategic plan. WallStreetBets users celebrated "
            "with 'to the moon' posts. AMC also rallied in sympathy."
        ),
    ]

    extractor = EntityExtractor()

    for title, body in SAMPLES:
        print(f"\n📰 {title}")
        print(f"   Text: {body[:100]}…")
        result = extractor.extract(title, body)
        print(f"   Tickers:  {result['tickers']}")
        print(f"   Crypto:   {result['cryptos']}")
        print(f"   People:   {result['people']}")
        print(f"   Keywords: {result['keywords'][:5]}")
