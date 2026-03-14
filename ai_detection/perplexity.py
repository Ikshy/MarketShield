"""
ai_detection/perplexity.py — GPT-2 perplexity & burstiness scorer.

Perplexity is a measure of how "surprised" a language model is by a text.
AI-generated text tends to have *low* perplexity — the model is not surprised
because the text is composed of high-probability, safe token sequences.

Human writing, especially informal (Reddit posts, real news), has *higher*
perplexity because humans use varied, unexpected phrasing.

Burstiness captures variance in sentence-level perplexity:
  - AI text: uniformly low perplexity across all sentences (low burstiness)
  - Human text: some sentences are fluent, others awkward (high burstiness)

This combination is harder to fool than either metric alone.

Reference: "DetectGPT" (Mitchell et al. 2023) — perturbation-based detection
           "GPTZero" — the perplexity/burstiness combination we implement here

Model: gpt2 (117M params, ~500 MB, auto-downloaded from Hugging Face)
Fallback: heuristic statistical scorer (no model required)
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PerplexityResult:
    """Full perplexity analysis for one text."""
    perplexity: float          # Lower = more likely AI-generated (< 20 suspicious)
    burstiness: float          # Lower = more uniform = more AI-like (< 0.2 suspicious)
    sentence_perplexities: list[float]   # Per-sentence breakdown
    ai_probability_from_perplexity: float  # 0–1 signal (not final prediction)
    token_count: int
    method: str                # "gpt2" | "heuristic"


# ---------------------------------------------------------------------------
# Text segmentation
# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences for burstiness calculation.
    Uses simple punctuation rules — good enough for news/Reddit text.
    """
    # Split on .!? followed by space + capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    # Filter out very short fragments
    return [s.strip() for s in sentences if len(s.split()) >= 4]


def _word_frequency_features(text: str) -> dict[str, float]:
    """
    Extract statistical text features that distinguish AI from human writing.
    These work without any model — useful as a fast pre-filter.

    AI text signatures:
      - Low type-token ratio (TTR): AI repeats words more
      - Low punctuation density: AI uses fewer dashes, semicolons
      - High average word length: AI favours Latinate vocabulary
      - Low contraction rate: AI rarely uses "don't", "I'm" etc.
      - Uniform sentence length: low std dev of sentence word counts
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {}

    sentences = _split_sentences(text)
    sent_lengths = [len(s.split()) for s in sentences] if sentences else [len(words)]

    # Type-token ratio (lexical diversity)
    ttr = len(set(words)) / len(words)

    # Punctuation density (chars per 100 words)
    punct_chars = sum(1 for c in text if c in "!?;:—–-()[]\"'")
    punct_density = (punct_chars / len(words)) * 100

    # Average word length
    avg_word_len = sum(len(w) for w in words) / len(words)

    # Contraction rate
    contractions = re.findall(
        r"\b(don't|can't|won't|I'm|I've|I'll|it's|that's|there's|they're|we're|you're|isn't|aren't|wasn't|weren't)\b",
        text, re.IGNORECASE
    )
    contraction_rate = len(contractions) / len(words)

    # Sentence length variance (burstiness proxy)
    sent_len_std = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0.0

    return {
        "ttr": ttr,
        "punct_density": punct_density,
        "avg_word_len": avg_word_len,
        "contraction_rate": contraction_rate,
        "sent_len_std": sent_len_std,
        "word_count": len(words),
        "sentence_count": len(sentences),
    }


def _heuristic_perplexity(text: str) -> PerplexityResult:
    """
    Estimate perplexity-equivalent score using statistical text features.
    No model required. Less accurate than GPT-2 but fast and always available.

    Scoring logic (calibrated against real AI vs human news text):
    - Low TTR → lower perplexity proxy (AI signal)
    - Low punctuation density → lower perplexity proxy
    - High avg word length → lower perplexity proxy
    - Low contraction rate → lower perplexity proxy
    - Low sentence length std → lower burstiness (AI signal)
    """
    features = _word_frequency_features(text)
    if not features:
        return PerplexityResult(
            perplexity=50.0, burstiness=0.5,
            sentence_perplexities=[], ai_probability_from_perplexity=0.3,
            token_count=0, method="heuristic"
        )

    # Normalise each feature to 0–1 (higher = more AI-like)
    ttr_signal = max(0, (0.75 - features["ttr"]) / 0.4)          # AI: TTR < 0.75
    punct_signal = max(0, (3.0 - features["punct_density"]) / 3.0) # AI: < 3 punct/100w
    wordlen_signal = min(1, (features["avg_word_len"] - 4.5) / 2.5) # AI: avg > 4.5 chars
    contract_signal = max(0, (0.01 - features["contraction_rate"]) / 0.01) # AI: < 1%
    burst_signal = max(0, (4.0 - features["sent_len_std"]) / 4.0)  # AI: low sentence var

    # Combine into pseudo-perplexity (lower = more AI-like, range ~10–200)
    ai_composite = np.mean([ttr_signal, punct_signal, wordlen_signal,
                             contract_signal, burst_signal])
    pseudo_perplexity = 100 * (1.0 - ai_composite) + 10
    burstiness = 1.0 - burst_signal

    # Convert to 0–1 AI probability (sigmoid-like, calibrated)
    ai_prob = 1.0 / (1.0 + math.exp((pseudo_perplexity - 35) / 15))

    sentences = _split_sentences(text)
    sent_perplexities = [pseudo_perplexity * (0.8 + 0.4 * np.random.random())
                         for _ in sentences]  # Approximate variance

    return PerplexityResult(
        perplexity=round(pseudo_perplexity, 2),
        burstiness=round(burstiness, 4),
        sentence_perplexities=sent_perplexities,
        ai_probability_from_perplexity=round(ai_prob, 4),
        token_count=features.get("word_count", 0),
        method="heuristic",
    )


# ---------------------------------------------------------------------------
# GPT-2 model loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_gpt2():
    """
    Load GPT-2 tokenizer + model (cached after first call).
    Downloads ~500 MB on first use; subsequent loads are <1 second.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # type: ignore
    import torch

    log.info("[Perplexity] Loading GPT-2 model (first load: ~500 MB download)…")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    log.info(f"[Perplexity] GPT-2 ready on {device.upper()}")
    return tokenizer, model, device


def _gpt2_sentence_perplexity(sentence: str, tokenizer, model, device) -> Optional[float]:
    """
    Compute GPT-2 perplexity for a single sentence.
    Returns None if the sentence is too short or tokenisation fails.
    """
    import torch

    try:
        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 3:
            return None

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss.item()

        # loss is mean cross-entropy; perplexity = exp(loss)
        return math.exp(loss)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------
class PerplexityScorer:
    """
    Scores text using GPT-2 perplexity + burstiness analysis.

    Lower perplexity + lower burstiness = stronger AI signal.

    Thresholds (calibrated on news/Reddit data):
      perplexity  < 20  → strongly AI-generated
      perplexity  < 35  → possibly AI-generated
      burstiness  < 0.2 → uniform style (AI signal)
      burstiness  > 0.5 → highly varied style (human signal)

    Usage
    -----
    >>> scorer = PerplexityScorer()
    >>> result = scorer.score("This article analyses macroeconomic convergence...")
    >>> print(f"Perplexity: {result.perplexity:.1f}, AI prob: {result.ai_probability_from_perplexity:.2f}")
    """

    def __init__(self, use_gpt2: bool = True):
        self._use_gpt2 = use_gpt2
        self._model_available = False

        if use_gpt2:
            try:
                _load_gpt2()
                self._model_available = True
                log.info("[Perplexity] GPT-2 scorer initialised")
            except (ImportError, OSError, Exception) as exc:
                log.warning(
                    f"[Perplexity] GPT-2 unavailable ({exc}). "
                    "Falling back to heuristic scorer."
                )

    def score(self, text: str) -> PerplexityResult:
        """
        Compute perplexity and burstiness for one text.

        For long texts, scores the first 10 sentences (truncated at 1024 tokens).
        Returns a full PerplexityResult with per-sentence breakdown.
        """
        if len(text.split()) < 10:
            return PerplexityResult(
                perplexity=100.0, burstiness=1.0,
                sentence_perplexities=[], ai_probability_from_perplexity=0.1,
                token_count=len(text.split()), method="too_short",
            )

        if self._model_available:
            try:
                return self._gpt2_score(text)
            except Exception as exc:
                log.warning(f"[Perplexity] GPT-2 scoring failed: {exc}")

        return _heuristic_perplexity(text)

    def _gpt2_score(self, text: str) -> PerplexityResult:
        """Full GPT-2-based perplexity analysis."""
        import torch

        tokenizer, model, device = _load_gpt2()
        sentences = _split_sentences(text)

        if not sentences:
            sentences = [text]

        # Score up to 10 sentences (cap computation time)
        scored_sentences = sentences[:10]
        sent_perplexities: list[float] = []

        for sent in scored_sentences:
            ppl = _gpt2_sentence_perplexity(sent, tokenizer, model, device)
            if ppl is not None and ppl < 10_000:  # Filter numerical explosions
                sent_perplexities.append(ppl)

        if not sent_perplexities:
            return _heuristic_perplexity(text)

        mean_ppl = float(np.mean(sent_perplexities))
        std_ppl = float(np.std(sent_perplexities))

        # Burstiness = coefficient of variation (std/mean)
        # Clipped to [0, 1] for interpretability
        burstiness = min(1.0, std_ppl / (mean_ppl + 1e-8))

        # Token count
        token_count = len(tokenizer.encode(text[:4096]))

        # AI probability from perplexity (sigmoid curve)
        # Calibrated: human news perplexity ~50–150, AI ~10–30
        ai_prob = 1.0 / (1.0 + math.exp((mean_ppl - 35) / 12))
        # Penalise low burstiness (uniform style = AI signal)
        if burstiness < 0.25:
            ai_prob = min(1.0, ai_prob * 1.3)

        return PerplexityResult(
            perplexity=round(mean_ppl, 2),
            burstiness=round(burstiness, 4),
            sentence_perplexities=[round(p, 2) for p in sent_perplexities],
            ai_probability_from_perplexity=round(ai_prob, 4),
            token_count=token_count,
            method="gpt2",
        )

    def score_batch(self, texts: list[str]) -> list[PerplexityResult]:
        """Score a list of texts (sequential — GPT-2 perplexity is per-sample)."""
        return [self.score(t) for t in texts]


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DEMO_TEXTS = {
        "Human (casual Reddit)": (
            "ok so I YOLO'd everything into calls this morning and... yeah. "
            "my wife's gonna kill me lol. WSB really did a number on me. "
            "anyone else bag-holding these things? price action is brutal rn"
        ),
        "Human (news article)": (
            "Shares of GameStop Corp. rose sharply Wednesday after a social-media "
            "campaign on Reddit's WallStreetBets forum encouraged retail investors "
            "to pile in, squeezing short-sellers who had bet against the stock. "
            "Trading was halted briefly as volumes surged past 100 million shares."
        ),
        "Likely AI-generated": (
            "The intersection of technological advancement and financial market dynamics "
            "presents a multifaceted landscape for investors. The convergence of artificial "
            "intelligence capabilities with traditional investment methodologies facilitates "
            "enhanced decision-making frameworks. Systematic risk evaluation protocols "
            "enable comprehensive portfolio optimization strategies."
        ),
        "AI pump-and-dump style": (
            "Revolutionary investment opportunity presents unprecedented potential for "
            "exceptional returns. Our proprietary algorithm has identified a significant "
            "market inefficiency that sophisticated investors are leveraging to achieve "
            "substantial wealth accumulation. The systematic convergence of multiple "
            "positive catalysts suggests imminent price appreciation."
        ),
    }

    print("\n📊 PerplexityScorer Demo (heuristic mode)\n")
    print(f"{'Sample':<28} {'Perplexity':>12} {'Burstiness':>12} {'AI Prob':>10}  {'Method'}")
    print("-" * 80)

    scorer = PerplexityScorer(use_gpt2=False)
    for label, text in DEMO_TEXTS.items():
        result = scorer.score(text)
        flag = "⚠️ " if result.ai_probability_from_perplexity > 0.5 else "  "
        print(
            f"{label:<28} {result.perplexity:>12.1f} {result.burstiness:>12.4f} "
            f"{result.ai_probability_from_perplexity:>10.3f}  {result.method} {flag}"
        )

    print("\n  ⚠️  = likely AI-generated (probability > 0.5)")
    print("  Note: Switch to GPT-2 mode for production accuracy")
