"""
ai_detection/ai_detector.py — Ensemble AI-generated text detector.

Combines three independent signals into a single ai_probability score:

  1. RoBERTa classifier (OpenAI's public detector, fine-tuned on GPT-2 outputs)
     Fast, high recall for GPT-2/3 generated text.

  2. Perplexity scorer (GPT-2 based)
     Low perplexity = model was "unsurprised" = text is likely AI output.

  3. Linguistic feature classifier (scikit-learn, trained on crafted features)
     Catches stylistic AI signatures that survive paraphrasing:
       - Low type-token ratio
       - Low contraction rate
       - High abstract noun density
       - Hedging phrase frequency
       - Sentence length uniformity

Ensemble fusion: weighted average, with weights tuned for financial text.
  - RoBERTa:   45% (most accurate on long-form content)
  - Perplexity: 30% (catches paraphrased AI)
  - Linguistic: 25% (fast, catches edge cases)

Fallback: if transformers unavailable, runs linguistic + perplexity only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

from config import settings
from logger import get_logger
from models import AIDetectionResult
from ai_detection.perplexity import PerplexityScorer, _word_frequency_features

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Linguistic feature extractor
# ---------------------------------------------------------------------------
# Phrases strongly over-represented in AI-generated financial content
_AI_HEDGING_PHRASES = [
    "it is important to note", "it is worth noting", "it should be noted",
    "in conclusion", "in summary", "to summarise", "in this context",
    "furthermore", "moreover", "additionally", "consequently",
    "it is crucial", "it is essential", "it is imperative",
    "significant implications", "multifaceted", "comprehensive approach",
    "robust framework", "paradigm shift", "synergistic",
    "leverage", "facilitate", "utilise", "optimize", "streamline",
    "unprecedented", "revolutionary", "transformative", "groundbreaking",
    "cutting-edge", "state-of-the-art", "best-in-class",
    "at the end of the day", "moving forward", "going forward",
    "in terms of", "with respect to", "with regard to",
]

# Phrases strongly associated with genuine human financial writing
_HUMAN_PHRASES = [
    "i think", "i believe", "in my opinion", "tbh", "imo", "lol", "wtf",
    "honestly", "actually", "basically", "literally", "seriously",
    "bought the dip", "bag holding", "to the moon", "diamond hands",
    "paper hands", "tendies", "apes together strong",
    "this is not financial advice", "not a financial advisor",
    "disclosure:", "i own shares", "full disclosure",
]


def _extract_linguistic_features(text: str) -> np.ndarray:
    """
    Extract a 15-dimension feature vector from text.
    Each feature is normalised to roughly 0–1.

    Features (higher = more AI-like for most):
      0: type-token ratio (lower = more AI)    → inverted
      1: punctuation density (lower = more AI) → inverted
      2: avg word length (higher = more AI)
      3: contraction rate (lower = more AI)    → inverted
      4: sentence length std (lower = more AI) → inverted
      5: AI hedging phrase count (per 100 words)
      6: human phrase count (per 100 words)    → inverted
      7: passive voice density (higher = more AI)
      8: abstract noun density
      9: first-person pronoun rate (lower = more AI) → inverted
     10: uppercase word ratio (lower = more AI) — humans shout in caps
     11: question mark rate (lower = more AI)
     12: exclamation rate (lower = more AI)    → inverted
     13: average sentence length
     14: paragraph uniformity (lower variance = more AI)
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return np.zeros(15)

    n = len(words)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 3]
    sent_lengths = [len(s.split()) for s in sentences] if sentences else [n]

    # F0: type-token ratio (lexical diversity — AI has less variety)
    ttr = len(set(words)) / n

    # F1: punctuation density
    punct = sum(1 for c in text if c in "!?;:—–-")
    punct_density = punct / n

    # F2: average word length
    avg_word_len = np.mean([len(w) for w in words])

    # F3: contraction rate
    contractions = len(re.findall(
        r"\b\w+n't\b|\bi'm\b|\bi've\b|\bi'll\b|\bit's\b|\bthey're\b",
        text, re.IGNORECASE
    ))
    contraction_rate = contractions / n

    # F4: sentence length standard deviation
    sent_len_std = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0

    # F5: AI hedging phrase rate
    ai_phrases = sum(
        1 for phrase in _AI_HEDGING_PHRASES
        if phrase in text.lower()
    )
    ai_phrase_rate = ai_phrases / (n / 100)

    # F6: human phrase rate
    human_phrases = sum(
        1 for phrase in _HUMAN_PHRASES
        if phrase in text.lower()
    )
    human_phrase_rate = human_phrases / (n / 100)

    # F7: passive voice density (simple heuristic: "is/are/was/were + past participle")
    passive = len(re.findall(
        r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", text, re.IGNORECASE
    ))
    passive_rate = passive / n

    # F8: abstract noun density (words ending in -tion, -ity, -ism, -ness, -ment)
    abstract = len(re.findall(
        r"\b\w+(tion|ity|ism|ness|ment|ance|ence)\b", text, re.IGNORECASE
    ))
    abstract_rate = abstract / n

    # F9: first-person pronoun rate (AI avoids first person)
    first_person = len(re.findall(r"\b(i|me|my|mine|myself|we|our|us)\b", text, re.IGNORECASE))
    first_person_rate = first_person / n

    # F10: all-caps word ratio (humans use caps for emphasis; AI doesn't)
    caps_words = sum(1 for w in re.findall(r"\b[A-Z]{2,}\b", text) if len(w) > 2)
    caps_rate = caps_words / n

    # F11: question mark rate
    q_rate = text.count("?") / (n / 100)

    # F12: exclamation rate
    excl_rate = text.count("!") / (n / 100)

    # F13: average sentence length
    avg_sent_len = float(np.mean(sent_lengths))

    # F14: paragraph length variance (split on double newlines)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_lengths = [len(p.split()) for p in paragraphs] if paragraphs else [n]
    para_len_std = float(np.std(para_lengths)) if len(para_lengths) > 1 else 0.0
    para_uniformity = 1.0 / (1.0 + para_len_std / 50)  # High = uniform

    features = np.array([
        ttr,                 # 0
        punct_density,       # 1
        avg_word_len / 10,   # 2 (normalise)
        contraction_rate,    # 3
        sent_len_std / 20,   # 4 (normalise)
        min(1.0, ai_phrase_rate / 5),    # 5
        min(1.0, human_phrase_rate / 3), # 6
        min(1.0, passive_rate * 10),     # 7
        min(1.0, abstract_rate * 5),     # 8
        min(1.0, first_person_rate * 20),# 9
        min(1.0, caps_rate * 10),        # 10
        min(1.0, q_rate / 2),            # 11
        min(1.0, excl_rate / 2),         # 12
        min(1.0, avg_sent_len / 40),     # 13
        para_uniformity,                  # 14
    ], dtype=np.float32)

    return features


def _linguistic_ai_score(text: str) -> float:
    """
    Compute a 0–1 AI probability from linguistic features alone.
    Uses a hand-crafted scoring formula (no training data required).

    Weights reflect empirical observation on financial text datasets.
    """
    f = _extract_linguistic_features(text)
    if f.sum() == 0:
        return 0.3  # Unknown

    # AI signals (higher value → more AI-like)
    ai_signals = [
        1.0 - f[0],           # Low TTR = AI
        1.0 - f[1] * 5,       # Low punctuation = AI
        min(1.0, (f[2] * 10 - 4.0) / 3.0),  # Long words = AI
        1.0 - min(1.0, f[3] * 20),  # No contractions = AI
        1.0 - f[4],           # Uniform sentence length = AI
        f[5],                 # AI hedging phrases
        1.0 - f[6],           # No human phrases
        f[7] * 2,             # Passive voice
        f[8] * 2,             # Abstract nouns
        1.0 - f[9] * 2,       # No first-person
        1.0 - f[10] * 2,      # No caps shouting
        1.0 - f[11] * 2,      # No questions
        1.0 - f[12] * 2,      # No exclamations
        f[14],                # Paragraph uniformity
    ]

    # Clip all to [0, 1]
    ai_signals = [max(0.0, min(1.0, s)) for s in ai_signals]

    # Weighted mean
    weights = [0.10, 0.08, 0.08, 0.08, 0.08, 0.10, 0.08,
               0.05, 0.05, 0.07, 0.05, 0.04, 0.04, 0.10]
    score = float(np.average(ai_signals, weights=weights))
    return round(score, 4)


# ---------------------------------------------------------------------------
# RoBERTa-based classifier (OpenAI detector)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_roberta_detector():
    """
    Load the RoBERTa OpenAI detector from Hugging Face.
    Model: roberta-base-openai-detector (~500 MB download)
    """
    from transformers import pipeline as hf_pipeline  # type: ignore

    log.info(f"[AIDetector] Loading RoBERTa detector: {settings.ai_detection_model}")
    detector = hf_pipeline(
        "text-classification",
        model=settings.ai_detection_model,
        truncation=True,
        max_length=512,
    )
    log.info("[AIDetector] RoBERTa detector ready")
    return detector


def _roberta_ai_score(text: str) -> float:
    """
    Run RoBERTa detector. Returns probability that text is AI-generated (0–1).
    Label "LABEL_1" = AI-generated in the OpenAI detector convention.
    """
    try:
        detector = _load_roberta_detector()
        result = detector(text[:1000])[0]  # Truncate to 512 tokens equivalent
        label = result["label"]
        score = result["score"]

        # LABEL_1 = fake (AI), LABEL_0 = real (human)
        if label == "LABEL_1":
            return float(score)
        else:
            return float(1.0 - score)
    except Exception as exc:
        log.warning(f"[AIDetector] RoBERTa inference failed: {exc}")
        return 0.5  # Uncertain


# ---------------------------------------------------------------------------
# Ensemble detector
# ---------------------------------------------------------------------------
class AIContentDetector:
    """
    Ensemble AI-generated text detector combining three independent signals.

    Usage
    -----
    >>> detector = AIContentDetector()
    >>> result = detector.detect("This article presents a comprehensive analysis...")
    >>> print(f"AI probability: {result.ai_probability:.2f}")
    >>> print(f"Is AI generated: {result.is_ai_generated}")

    The ensemble produces robust scores even when individual detectors fail —
    each signal degrades gracefully to a neutral 0.5 on error.
    """

    # Ensemble weights (must sum to 1.0)
    _W_ROBERTA    = 0.45
    _W_PERPLEXITY = 0.30
    _W_LINGUISTIC = 0.25

    def __init__(
        self,
        use_roberta: bool = True,
        use_gpt2_perplexity: bool = True,
    ):
        self._use_roberta = use_roberta
        self._roberta_available = False
        self._perplexity_scorer = PerplexityScorer(use_gpt2=use_gpt2_perplexity)

        if use_roberta:
            try:
                _load_roberta_detector()
                self._roberta_available = True
                log.info("[AIDetector] RoBERTa classifier available")
            except (ImportError, OSError, Exception) as exc:
                log.warning(
                    f"[AIDetector] RoBERTa unavailable ({exc}). "
                    "Running linguistic + perplexity only."
                )

    def detect(self, text: str) -> AIDetectionResult:
        """
        Run the full ensemble detection pipeline on one text.

        Returns an AIDetectionResult with:
          - ai_probability: ensemble score (0–1)
          - is_ai_generated: True if above settings threshold
          - perplexity_score: raw GPT-2 perplexity
          - burstiness_score: sentence-level perplexity variance
          - method: which detectors contributed
        """
        if not text or len(text.split()) < 8:
            return AIDetectionResult(
                ai_probability=0.0, is_ai_generated=False,
                perplexity_score=100.0, burstiness_score=1.0,
                method="too_short",
            )

        # ── Signal 1: RoBERTa ─────────────────────────────────────────────
        if self._roberta_available:
            roberta_score = _roberta_ai_score(text)
        else:
            roberta_score = None  # Will be excluded from ensemble

        # ── Signal 2: Perplexity ──────────────────────────────────────────
        ppl_result = self._perplexity_scorer.score(text)
        perplexity_score = ppl_result.ai_probability_from_perplexity

        # ── Signal 3: Linguistic features ────────────────────────────────
        linguistic_score = _linguistic_ai_score(text)

        # ── Ensemble fusion ───────────────────────────────────────────────
        if roberta_score is not None:
            # All three available
            ensemble = (
                self._W_ROBERTA * roberta_score
                + self._W_PERPLEXITY * perplexity_score
                + self._W_LINGUISTIC * linguistic_score
            )
            method = "roberta+perplexity+linguistic"
        else:
            # No RoBERTa — redistribute weights proportionally
            w_ppl = self._W_PERPLEXITY / (self._W_PERPLEXITY + self._W_LINGUISTIC)
            w_ling = self._W_LINGUISTIC / (self._W_PERPLEXITY + self._W_LINGUISTIC)
            ensemble = w_ppl * perplexity_score + w_ling * linguistic_score
            method = "perplexity+linguistic"

        ai_probability = round(float(np.clip(ensemble, 0.0, 1.0)), 4)
        is_ai = ai_probability >= settings.ai_probability_threshold

        return AIDetectionResult(
            ai_probability=ai_probability,
            is_ai_generated=is_ai,
            perplexity_score=ppl_result.perplexity,
            burstiness_score=ppl_result.burstiness,
            method=method,
        )

    def detect_batch(self, texts: list[str]) -> list[AIDetectionResult]:
        """Detect AI probability for a list of texts."""
        return [self.detect(t) for t in texts]

    def explain(self, text: str) -> dict:
        """
        Return a detailed breakdown of all detection signals.
        Useful for dashboard display and debugging.
        """
        ppl_result = self._perplexity_scorer.score(text)
        linguistic = _linguistic_ai_score(text)
        features = _extract_linguistic_features(text)

        result = self.detect(text)

        return {
            "ensemble_ai_probability": result.ai_probability,
            "is_ai_generated": result.is_ai_generated,
            "method": result.method,
            "signals": {
                "roberta": _roberta_ai_score(text) if self._roberta_available else None,
                "perplexity": ppl_result.ai_probability_from_perplexity,
                "linguistic": linguistic,
            },
            "perplexity_details": {
                "raw_perplexity": ppl_result.perplexity,
                "burstiness": ppl_result.burstiness,
                "sentence_count": len(ppl_result.sentence_perplexities),
                "method": ppl_result.method,
            },
            "linguistic_features": {
                "type_token_ratio": round(float(features[0]), 3),
                "avg_word_length": round(float(features[2] * 10), 2),
                "contraction_rate": round(float(features[3]), 4),
                "ai_phrases_found": round(float(features[5] * 5)),
                "human_phrases_found": round(float(features[6] * 3)),
                "passive_voice_density": round(float(features[7] / 2), 3),
                "abstract_noun_rate": round(float(features[8] / 2), 3),
                "first_person_rate": round(float(features[9] / 20), 4),
            },
            "threshold": settings.ai_probability_threshold,
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    SAMPLES = [
        ("Human — Reddit",
         "lmao just lost $4k on these NVDA puts. shoulda listened to you guys. "
         "at least I got my wife's blessing lol. gonna buy the dip if it hits 400 again. "
         "who else is bag holding rn??"),

        ("Human — News journalist",
         "GameStop shares jumped more than 100 percent Tuesday after amateur investors "
         "on Reddit coordinated a massive short squeeze against hedge funds that had bet "
         "heavily against the struggling video-game retailer. Short sellers lost billions."),

        ("AI-generated (GPT-like)",
         "The current macroeconomic environment presents significant opportunities for "
         "astute investors to leverage market inefficiencies. Furthermore, the convergence "
         "of technological innovation with traditional financial instruments facilitates "
         "comprehensive portfolio optimization. It is important to note that risk "
         "management protocols remain essential in this transformative landscape."),

        ("AI pump-and-dump",
         "Revolutionary blockchain technology has created an unprecedented investment "
         "opportunity that sophisticated investors are recognising as transformative. "
         "Our proprietary analysis indicates substantial upside potential in this "
         "groundbreaking asset class. The systematic accumulation phase presents an "
         "optimal entry point for strategic portfolio positioning."),

        ("Coordinated misinformation style",
         "Breaking: Major pharmaceutical company's CEO arrested for fraud. Insider sources "
         "confirm stock manipulation scheme has been exposed. Multiple executives expected "
         "to face charges. Company faces imminent bankruptcy. Share price expected to "
         "collapse to zero. Sell immediately before trading halt is announced."),
    ]

    print("\n🤖 AIContentDetector Demo (linguistic + heuristic perplexity)\n")
    print(f"{'Sample':<30} {'AI Prob':>8} {'Perplexity':>12} {'Burstiness':>12}  {'Flag'}")
    print("-" * 80)

    # Use heuristic modes for demo (no model download)
    detector = AIContentDetector(use_roberta=False, use_gpt2_perplexity=False)

    for label, text in SAMPLES:
        result = detector.detect(text)
        flag = "🚨 AI DETECTED" if result.is_ai_generated else "✓  Human"
        print(
            f"{label:<30} {result.ai_probability:>8.3f} "
            f"{result.perplexity_score:>12.1f} {result.burstiness_score:>12.4f}  {flag}"
        )

    print(f"\n  Threshold: {settings.ai_probability_threshold:.0%}")
    print("\n📋 Detailed explanation for AI pump-and-dump sample:")
    explanation = detector.explain(SAMPLES[3][1])
    print(json.dumps(explanation, indent=2, default=str))
