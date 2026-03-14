"""
tests/test_nlp_pipeline.py

Tests for sentiment analysis, AI detection, perplexity scoring,
and entity extraction. Runs entirely without model downloads —
all transformer calls are mocked or use heuristic fallbacks.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from models import (
    ContentSource, RawArticle, Sentiment,
    SentimentResult, AIDetectionResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def make_article(title: str, body: str, source=ContentSource.RSS) -> RawArticle:
    return RawArticle(
        source=source,
        source_name="Test Source",
        title=title,
        body=body,
        published_at=datetime.now(timezone.utc),
        raw_metadata={},
    )


# ── Sample texts ─────────────────────────────────────────────────────────────
BULLISH_TEXT = "Tesla stock surges 20% on record earnings, analysts raise price targets to $400"
BEARISH_TEXT = "Bitcoin crashes 50% following regulatory crackdown and exchange collapse"
NEUTRAL_TEXT = "The Federal Reserve is scheduled to meet next week to discuss interest rates"
AI_LIKE_TEXT = (
    "The convergence of technological innovation and financial market dynamics "
    "presents significant opportunities for astute investors. Furthermore, the "
    "systematic implementation of comprehensive risk management protocols "
    "facilitates optimal portfolio allocation. It is important to note that "
    "sophisticated methodologies enable transformative value creation."
)
HUMAN_REDDIT_TEXT = (
    "lmao I just lost $4k on TSLA puts. wife is gonna kill me lol 💀 "
    "seriously tho why did I listen to WSB. buying the dip if it hits $200 "
    "anyone else bag holding these calls? this is brutal rn 😭"
)


# ---------------------------------------------------------------------------
# Sentiment Analyser tests
# ---------------------------------------------------------------------------
class TestSentimentAnalyser:
    """Tests using VADER fallback (no model download required)."""

    def setup_method(self):
        from ai_detection.sentiment import SentimentAnalyser
        # use_transformer=False → always uses VADER (fast, no download)
        self.analyser = SentimentAnalyser(use_transformer=False)

    def test_bullish_text_positive_sentiment(self):
        result = self.analyser.analyse(BULLISH_TEXT)
        assert result.label == Sentiment.POSITIVE
        assert result.score > 0.0

    def test_bearish_text_negative_sentiment(self):
        result = self.analyser.analyse(BEARISH_TEXT)
        assert result.label == Sentiment.NEGATIVE
        assert result.score > 0.0

    def test_neutral_text_neutral_sentiment(self):
        result = self.analyser.analyse(NEUTRAL_TEXT)
        assert result.label == Sentiment.NEUTRAL

    def test_probabilities_sum_to_one(self):
        result = self.analyser.analyse(BULLISH_TEXT)
        total = result.positive + result.negative + result.neutral
        assert abs(total - 1.0) < 0.01

    def test_short_text_returns_result(self):
        """Very short texts should not raise; return a valid result."""
        result = self.analyser.analyse("Up")
        assert result.label in Sentiment.__members__.values()

    def test_empty_text_returns_neutral(self):
        result = self.analyser.analyse("")
        assert result.label == Sentiment.NEUTRAL

    def test_batch_returns_correct_count(self):
        texts = [BULLISH_TEXT, BEARISH_TEXT, NEUTRAL_TEXT]
        results = self.analyser.analyse_batch(texts)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_batch_empty_returns_empty(self):
        assert self.analyser.analyse_batch([]) == []

    def test_sentiment_shift_structure(self):
        texts = [BULLISH_TEXT, BEARISH_TEXT, NEUTRAL_TEXT]
        results = self.analyser.analyse_batch(texts)
        shift = self.analyser.sentiment_shift(results)

        assert "bullish_ratio" in shift
        assert "bearish_ratio" in shift
        assert "mean_confidence" in shift
        assert shift["sample_size"] == 3
        assert 0.0 <= shift["bullish_ratio"] <= 1.0

    def test_analyse_article_combines_title_and_body(self):
        result = self.analyser.analyse_article(
            title="Stock crashes",
            body="Investors panic as prices plummet",
        )
        assert result.label == Sentiment.NEGATIVE

    def test_clean_text_removes_urls(self):
        from ai_detection.sentiment import _clean_text
        text = "Check out https://example.com for more info about stock gains"
        cleaned = _clean_text(text)
        assert "https://" not in cleaned
        assert "stock gains" in cleaned


# ---------------------------------------------------------------------------
# Perplexity Scorer tests
# ---------------------------------------------------------------------------
class TestPerplexityScorer:
    """Tests using heuristic mode (no GPT-2 download)."""

    def setup_method(self):
        from ai_detection.perplexity import PerplexityScorer
        self.scorer = PerplexityScorer(use_gpt2=False)

    def test_scores_return_valid_structure(self):
        result = self.scorer.score(AI_LIKE_TEXT)
        assert 0.0 < result.perplexity < 10_000
        assert 0.0 <= result.burstiness <= 1.0
        assert 0.0 <= result.ai_probability_from_perplexity <= 1.0
        assert result.token_count > 0

    def test_short_text_handled_gracefully(self):
        result = self.scorer.score("short")
        assert result.perplexity > 0
        assert result.method == "too_short"

    def test_ai_text_higher_ai_probability_than_human(self):
        ai_result = self.scorer.score(AI_LIKE_TEXT)
        human_result = self.scorer.score(HUMAN_REDDIT_TEXT)
        # AI text should have lower perplexity → higher ai_probability
        assert ai_result.ai_probability_from_perplexity >= human_result.ai_probability_from_perplexity

    def test_batch_returns_correct_count(self):
        texts = [AI_LIKE_TEXT, HUMAN_REDDIT_TEXT, NEUTRAL_TEXT]
        results = self.scorer.score_batch(texts)
        assert len(results) == 3

    def test_word_frequency_features_structure(self):
        from ai_detection.perplexity import _word_frequency_features
        features = _word_frequency_features(AI_LIKE_TEXT)

        assert "ttr" in features
        assert "avg_word_len" in features
        assert "contraction_rate" in features
        assert 0.0 < features["ttr"] <= 1.0

    def test_sentence_splitter(self):
        from ai_detection.perplexity import _split_sentences
        text = "First sentence here. Second one follows. And a third one."
        sentences = _split_sentences(text)
        assert len(sentences) >= 1  # At least some sentences found


# ---------------------------------------------------------------------------
# Linguistic feature extraction tests
# ---------------------------------------------------------------------------
class TestLinguisticFeatures:

    def test_feature_vector_shape(self):
        from ai_detection.ai_detector import _extract_linguistic_features
        import numpy as np
        features = _extract_linguistic_features(AI_LIKE_TEXT)
        assert features.shape == (15,)
        assert not np.any(np.isnan(features))

    def test_ai_text_has_more_ai_phrases(self):
        from ai_detection.ai_detector import _extract_linguistic_features
        ai_f = _extract_linguistic_features(AI_LIKE_TEXT)
        human_f = _extract_linguistic_features(HUMAN_REDDIT_TEXT)
        # Feature index 5 is ai_phrase_rate
        assert ai_f[5] >= human_f[5]

    def test_human_text_has_more_contractions(self):
        from ai_detection.ai_detector import _extract_linguistic_features
        ai_f = _extract_linguistic_features(AI_LIKE_TEXT)
        human_f = _extract_linguistic_features(HUMAN_REDDIT_TEXT)
        # Feature index 3 is contraction_rate
        assert human_f[3] >= ai_f[3]

    def test_linguistic_ai_score_range(self):
        from ai_detection.ai_detector import _linguistic_ai_score
        ai_score = _linguistic_ai_score(AI_LIKE_TEXT)
        human_score = _linguistic_ai_score(HUMAN_REDDIT_TEXT)
        assert 0.0 <= ai_score <= 1.0
        assert 0.0 <= human_score <= 1.0

    def test_ai_text_scores_higher_than_human(self):
        from ai_detection.ai_detector import _linguistic_ai_score
        ai_score = _linguistic_ai_score(AI_LIKE_TEXT)
        human_score = _linguistic_ai_score(HUMAN_REDDIT_TEXT)
        assert ai_score > human_score


# ---------------------------------------------------------------------------
# AI Content Detector ensemble tests
# ---------------------------------------------------------------------------
class TestAIContentDetector:
    """Tests using heuristic mode only (no RoBERTa, no GPT-2)."""

    def setup_method(self):
        from ai_detection.ai_detector import AIContentDetector
        self.detector = AIContentDetector(use_roberta=False, use_gpt2_perplexity=False)

    def test_detect_returns_valid_result(self):
        result = self.detector.detect(BULLISH_TEXT)
        assert isinstance(result, AIDetectionResult)
        assert 0.0 <= result.ai_probability <= 1.0
        assert isinstance(result.is_ai_generated, bool)

    def test_ai_text_flagged_with_higher_probability(self):
        ai_result = self.detector.detect(AI_LIKE_TEXT)
        human_result = self.detector.detect(HUMAN_REDDIT_TEXT)
        assert ai_result.ai_probability > human_result.ai_probability

    def test_short_text_not_flagged(self):
        result = self.detector.detect("Buy")
        assert result.method == "too_short"
        assert result.is_ai_generated is False

    def test_is_ai_generated_respects_threshold(self):
        from config import settings
        result = self.detector.detect(AI_LIKE_TEXT)
        expected = result.ai_probability >= settings.ai_probability_threshold
        assert result.is_ai_generated == expected

    def test_batch_detection(self):
        texts = [AI_LIKE_TEXT, HUMAN_REDDIT_TEXT, NEUTRAL_TEXT, BULLISH_TEXT]
        results = self.detector.detect_batch(texts)
        assert len(results) == 4
        for r in results:
            assert 0.0 <= r.ai_probability <= 1.0

    def test_explain_returns_full_breakdown(self):
        explanation = self.detector.explain(AI_LIKE_TEXT)
        assert "ensemble_ai_probability" in explanation
        assert "signals" in explanation
        assert "linguistic_features" in explanation
        assert "perplexity_details" in explanation
        assert "threshold" in explanation

    def test_method_string_without_roberta(self):
        result = self.detector.detect(AI_LIKE_TEXT)
        assert "linguistic" in result.method
        assert "roberta" not in result.method


# ---------------------------------------------------------------------------
# Entity Extractor tests
# ---------------------------------------------------------------------------
class TestEntityExtractor:

    def setup_method(self):
        from ai_detection.entity_extractor import EntityExtractor
        self.extractor = EntityExtractor()

    def test_extracts_dollar_sign_ticker(self):
        result = self.extractor.extract("$TSLA surges 20% on earnings", "")
        assert "TSLA" in result["tickers"]

    def test_extracts_parenthesised_ticker(self):
        result = self.extractor.extract("Tesla (TSLA) reports Q4 results", "")
        assert "TSLA" in result["tickers"]

    def test_extracts_crypto_name(self):
        result = self.extractor.extract("Bitcoin crashes after regulation news", "")
        assert "bitcoin" in result["cryptos"] or "bitcoin" in result["instruments"]

    def test_extracts_financial_keywords(self):
        result = self.extractor.extract(
            "SEC investigation reveals pump and dump scheme", ""
        )
        keywords = result["keywords"]
        assert any("pump" in kw or "sec" in kw.lower() for kw in keywords)

    def test_returns_instruments_list(self):
        result = self.extractor.extract("NVDA and $AMD rally on AI chip demand", "")
        assert isinstance(result["instruments"], list)

    def test_no_crash_on_empty_text(self):
        result = self.extractor.extract("", "")
        assert "tickers" in result
        assert isinstance(result["tickers"], list)

    def test_filters_common_word_false_positives(self):
        # Words like "I", "A", "FOR" should not appear as tickers
        result = self.extractor.extract("I bought a stock for my portfolio", "")
        assert "I" not in result["tickers"]
        assert "A" not in result["tickers"]
        assert "FOR" not in result["tickers"]

    def test_extract_from_body_text(self):
        result = self.extractor.extract(
            "Market update",
            "Shares of GameStop (GME) surged as short sellers faced a squeeze",
        )
        assert "GME" in result["tickers"]


# ---------------------------------------------------------------------------
# NLP Runner integration test
# ---------------------------------------------------------------------------
class TestNLPRunner:

    def test_enrich_article_returns_enriched_article(self):
        from ai_detection.runner import enrich_article
        from ai_detection.sentiment import SentimentAnalyser
        from ai_detection.ai_detector import AIContentDetector
        from ai_detection.entity_extractor import EntityExtractor
        from models import EnrichedArticle

        article = make_article(
            title="NVDA stock hits all-time high on AI demand",
            body="Nvidia shares rose 5% to a new record as data center demand surges.",
        )

        result = enrich_article(
            article,
            SentimentAnalyser(use_transformer=False),
            AIContentDetector(use_roberta=False, use_gpt2_perplexity=False),
            EntityExtractor(),
        )

        assert isinstance(result, EnrichedArticle)
        assert result.sentiment is not None
        assert result.ai_detection is not None
        assert result.article.id == article.id

    @pytest.mark.asyncio
    async def test_analyse_once_processes_all_articles(self):
        from ai_detection.runner import NLPAnalysisRunner

        articles = [
            make_article("Bull market rally continues", "Stocks gain for sixth consecutive day."),
            make_article("Crypto winter deepens", "Bitcoin down 15% this week amid sell-off."),
        ]

        runner = NLPAnalysisRunner(use_transformers=False)
        results = await runner.analyse_once(articles)

        assert len(results) == 2
        assert all(r.sentiment is not None for r in results)
        assert all(r.ai_detection is not None for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
