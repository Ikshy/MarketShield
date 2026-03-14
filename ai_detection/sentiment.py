"""
ai_detection/sentiment.py — Financial sentiment analysis using FinBERT.

FinBERT is a BERT model fine-tuned on ~10,000 financial news sentences.
It classifies text into positive / negative / neutral with calibrated
confidence scores — far more accurate for financial text than general-
purpose models like VADER or TextBlob.

Model: ProsusAI/finbert (auto-downloaded from Hugging Face on first use)
Fallback: VADER lexicon-based scorer (zero dependencies, instant)

Architecture
------------
  SentimentAnalyser
    ├── _transformer_sentiment()   — FinBERT inference (GPU if available)
    ├── _vader_sentiment()         — Lexicon fallback
    └── analyse_batch()            — Batch inference with padding/chunking

Design choices:
  - Model is loaded lazily (first call), so import is always fast
  - Long texts are truncated to 512 tokens (BERT's limit) after extracting
    the first 400 + last 112 tokens — this captures both the lede and the
    conclusion, which carry the most sentiment signal
  - Batch size 16 is safe on CPU; set MARKETSHIELD_SENTIMENT_BATCH_SIZE
    in .env to increase for GPU inference
"""

from __future__ import annotations

import re
import warnings
from functools import lru_cache
from typing import Optional

import numpy as np

from config import settings
from logger import get_logger
from models import Sentiment, SentimentResult

log = get_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TOKENS = 512          # Hard BERT limit
BATCH_SIZE = 16           # Safe for CPU inference
VADER_FALLBACK = True     # Use VADER if transformers unavailable


# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """
    Normalise raw article text for model input.

    Steps:
      1. Collapse whitespace
      2. Remove URLs (they add noise, no sentiment)
      3. Remove HTML entities
      4. Strip Reddit-style flairs and vote counts
    """
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)           # HTML entities
    text = re.sub(r"\[deleted\]|\[removed\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]  # Cap input to save memory


def _truncate_for_bert(text: str, tokenizer) -> str:
    """
    Truncate text to fit within BERT's 512-token limit.

    Strategy: keep the first 400 tokens + last 100 tokens.
    The beginning and end of a financial article carry the most
    sentiment signal (headline + conclusion).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= MAX_TOKENS - 2:  # -2 for [CLS] and [SEP]
        return text

    # Take first 400 + last 100 tokens
    kept = tokens[:400] + tokens[-100:]
    return tokenizer.decode(kept, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# VADER fallback (pure lexicon, zero ML dependencies)
# ---------------------------------------------------------------------------
def _vader_sentiment(text: str) -> SentimentResult:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) fallback.
    Installed as part of nltk. Less accurate than FinBERT for financial
    text but always available and very fast.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        import nltk
        try:
            vader = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            vader = SentimentIntensityAnalyzer()

        scores = vader.polarity_scores(text)
        compound = scores["compound"]

        # Map compound score → label
        if compound >= 0.05:
            label = Sentiment.POSITIVE
        elif compound <= -0.05:
            label = Sentiment.NEGATIVE
        else:
            label = Sentiment.NEUTRAL

        # Normalise VADER's pos/neg/neu to sum to 1
        pos = scores["pos"]
        neg = scores["neg"]
        neu = scores["neu"]
        total = pos + neg + neu or 1.0

        return SentimentResult(
            label=label,
            score=abs(compound),
            positive=pos / total,
            negative=neg / total,
            neutral=neu / total,
        )

    except Exception as exc:
        log.warning(f"[Sentiment] VADER failed: {exc} — returning neutral")
        return SentimentResult(
            label=Sentiment.NEUTRAL, score=0.5,
            positive=0.33, negative=0.33, neutral=0.34,
        )


# ---------------------------------------------------------------------------
# FinBERT model loader (singleton)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_finbert():
    """
    Load FinBERT tokenizer + model (cached after first call).
    Downloads ~440 MB from Hugging Face on first use.
    Returns (tokenizer, model, device) or raises ImportError.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    import torch

    log.info(f"[Sentiment] Loading FinBERT model: {settings.sentiment_model}")
    log.info("[Sentiment] (First load downloads ~440 MB — subsequent loads are instant)")

    tokenizer = AutoTokenizer.from_pretrained(settings.sentiment_model)
    model = AutoModelForSequenceClassification.from_pretrained(settings.sentiment_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    log.info(f"[Sentiment] FinBERT ready on {device.upper()}")
    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------
class SentimentAnalyser:
    """
    Financial sentiment analyser using FinBERT with VADER fallback.

    Usage
    -----
    >>> analyser = SentimentAnalyser()
    >>> result = analyser.analyse("Tesla stock surges 15% on record earnings beat")
    >>> print(result.label, result.score)
    Sentiment.POSITIVE 0.97

    >>> results = analyser.analyse_batch(["TSLA up", "BTC crashes", "Mixed outlook"])
    """

    # FinBERT's label order (matches model config.json)
    _FINBERT_LABELS = ["positive", "negative", "neutral"]

    def __init__(self, use_transformer: bool = True):
        """
        Parameters
        ----------
        use_transformer : bool
            If True (default), attempts to load FinBERT.
            If False, uses VADER only (useful for testing / low-resource envs).
        """
        self._use_transformer = use_transformer
        self._model_available = False

        if use_transformer:
            try:
                _load_finbert()  # Trigger lazy load, confirm it works
                self._model_available = True
                log.info("[Sentiment] Using FinBERT transformer model")
            except (ImportError, OSError, Exception) as exc:
                log.warning(
                    f"[Sentiment] FinBERT unavailable ({exc}). "
                    "Falling back to VADER lexicon scorer."
                )

    # ── Single article ────────────────────────────────────────────────────
    def analyse(self, text: str) -> SentimentResult:
        """
        Analyse the sentiment of a single text string.

        Automatically falls back to VADER if FinBERT is unavailable or
        if the text is too short to be meaningful (< 5 words).
        """
        cleaned = _clean_text(text)

        if len(cleaned.split()) < 5:
            # Too short for FinBERT — use VADER
            return _vader_sentiment(cleaned)

        if self._model_available:
            try:
                return self._transformer_sentiment(cleaned)
            except Exception as exc:
                log.warning(f"[Sentiment] FinBERT inference failed: {exc} — using VADER")

        return _vader_sentiment(cleaned)

    def _transformer_sentiment(self, text: str) -> SentimentResult:
        """Run FinBERT inference on a single text."""
        import torch

        tokenizer, model, device = _load_finbert()
        text = _truncate_for_bert(text, tokenizer)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        # Softmax → probabilities
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # FinBERT label order: positive=0, negative=1, neutral=2
        pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])
        best_idx = int(np.argmax(probs))
        label_str = self._FINBERT_LABELS[best_idx]
        confidence = float(probs[best_idx])

        return SentimentResult(
            label=Sentiment(label_str),
            score=confidence,
            positive=pos,
            negative=neg,
            neutral=neu,
        )

    # ── Batch processing ─────────────────────────────────────────────────
    def analyse_batch(
        self,
        texts: list[str],
        batch_size: int = BATCH_SIZE,
    ) -> list[SentimentResult]:
        """
        Analyse a list of texts efficiently using batched inference.

        Batching is ~5x faster than calling analyse() in a loop because
        it amortises tokenisation and GPU kernel launch overhead.

        Parameters
        ----------
        texts : list[str]
            Input texts (any length; long texts are auto-truncated).
        batch_size : int
            Number of texts per forward pass. Reduce if OOM on GPU.
        """
        if not texts:
            return []

        if not self._model_available:
            return [_vader_sentiment(_clean_text(t)) for t in texts]

        results: list[SentimentResult] = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            batch_results = self._batch_transformer(chunk)
            results.extend(batch_results)

        log.debug(f"[Sentiment] Batch analysed {len(texts)} texts")
        return results

    def _batch_transformer(self, texts: list[str]) -> list[SentimentResult]:
        """Run FinBERT on a single batch of texts."""
        import torch

        tokenizer, model, device = _load_finbert()
        cleaned = [_clean_text(t) for t in texts]

        # Truncate each text individually before batching
        truncated = [_truncate_for_bert(t, tokenizer) for t in cleaned]

        inputs = tokenizer(
            truncated,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for probs in probs_batch:
            pos, neg, neu = float(probs[0]), float(probs[1]), float(probs[2])
            best_idx = int(np.argmax(probs))
            results.append(
                SentimentResult(
                    label=Sentiment(self._FINBERT_LABELS[best_idx]),
                    score=float(probs[best_idx]),
                    positive=pos,
                    negative=neg,
                    neutral=neu,
                )
            )
        return results

    # ── Convenience helpers ───────────────────────────────────────────────
    def analyse_article(self, title: str, body: str) -> SentimentResult:
        """
        Analyse a full article by combining title + first 300 chars of body.
        The title is weighted more heavily by prepending it twice.
        """
        combined = f"{title}. {title}. {body[:300]}"
        return self.analyse(combined)

    def sentiment_shift(
        self,
        results: list[SentimentResult],
    ) -> dict[str, float]:
        """
        Compute aggregate sentiment metrics over a batch of results.
        Useful for tracking how the overall tone of a news cluster changes.

        Returns
        -------
        dict with keys: mean_positive, mean_negative, mean_neutral,
                        bullish_ratio, bearish_ratio, mean_confidence
        """
        if not results:
            return {}

        pos_scores = [r.positive for r in results]
        neg_scores = [r.negative for r in results]
        neu_scores = [r.neutral for r in results]
        confidences = [r.score for r in results]

        bullish = sum(1 for r in results if r.label == Sentiment.POSITIVE)
        bearish = sum(1 for r in results if r.label == Sentiment.NEGATIVE)

        return {
            "mean_positive": float(np.mean(pos_scores)),
            "mean_negative": float(np.mean(neg_scores)),
            "mean_neutral": float(np.mean(neu_scores)),
            "bullish_ratio": bullish / len(results),
            "bearish_ratio": bearish / len(results),
            "mean_confidence": float(np.mean(confidences)),
            "sample_size": len(results),
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DEMO_TEXTS = [
        ("Bullish", "Tesla stock surges 15% on record-breaking quarterly earnings beat"),
        ("Bearish", "Crypto markets crash 40% as regulators announce sweeping new bans"),
        ("Neutral", "The Federal Reserve will meet next week to discuss monetary policy"),
        ("Suspicious", "BUY NOW before this stock EXPLODES 10000% guaranteed returns!!"),
        ("Reddit", "Just YOLO'd my life savings into GME calls. To the moon 🚀🚀🚀"),
        ("AI-like", "The convergence of macroeconomic variables suggests a recalibration of risk parameters is warranted"),
    ]

    print("\n🔍 SentimentAnalyser Demo (using VADER fallback)\n")
    print(f"{'Expected':<12} {'Label':<12} {'Conf':<8} {'Pos':<8} {'Neg':<8} {'Neu':<8}  Text")
    print("-" * 100)

    # Use VADER for demo (no model download required)
    analyser = SentimentAnalyser(use_transformer=False)
    for expected, text in DEMO_TEXTS:
        r = analyser.analyse(text)
        print(
            f"{expected:<12} {r.label.value:<12} {r.score:<8.3f} "
            f"{r.positive:<8.3f} {r.negative:<8.3f} {r.neutral:<8.3f}  {text[:60]}"
        )

    print("\n📊 Aggregate Sentiment Shift:")
    results = [analyser.analyse(t) for _, t in DEMO_TEXTS]
    shift = analyser.sentiment_shift(results)
    for k, v in shift.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
