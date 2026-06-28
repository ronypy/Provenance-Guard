"""Signal 2 — stylometric heuristics (pure Python, no external libraries).

Measures *structural* properties of the text that tend to differ between human and AI writing:

* **Sentence-length variance (burstiness)** — humans vary sentence length a lot; AI is uniform.
* **Type-token ratio** — vocabulary diversity over the piece.
* **Punctuation density** — humans pepper in dashes, parentheses, ellipses; AI is plainer.
* **Mean sentence length** — AI trends toward steady medium-length sentences.

Each metric is turned into a partial "looks AI" score in ``[0, 1]`` via a documented rubric,
then averaged. This signal is meaning-blind — it never reads what the text *says*, only how it
is shaped — which is exactly what makes it independent of the LLM signal.

Blind spot: needs a few sentences of text, and formal human writing (academic/policy) is
naturally uniform, so it can read as AI-like. Documented as a known limitation.
"""

from __future__ import annotations

import re
import statistics

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"[A-Za-z']+")
_PUNCT_RE = re.compile(r"[,;:\-—()\"'…]")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _normalize(value: float, low: float, high: float) -> float:
    """Map ``value`` from the range [low, high] onto [0, 1], clamped."""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def stylometric_signal(text: str) -> dict:
    """Return ``{"score": P(AI), "metrics": {...}}`` from structural statistics."""
    words = _WORD_RE.findall(text.lower())
    sentences = _split_sentences(text)

    # Too little text to judge structurally -> sit on the fence.
    if len(words) < 15 or len(sentences) < 2:
        return {
            "score": 0.5,
            "metrics": {"note": "too short for reliable stylometry", "word_count": len(words)},
        }

    sentence_lengths = [len(_WORD_RE.findall(s)) for s in sentences]

    # 1) Burstiness: low variance -> AI-like. Use stdev of sentence length.
    stdev = statistics.pstdev(sentence_lengths)
    # Humans commonly land around stdev 6+; very uniform AI text near 1-2.
    burstiness_ai = 1.0 - _normalize(stdev, 1.0, 8.0)  # low stdev => high AI score

    # 2) Type-token ratio: very high OR very low diversity is less informative; AI tends to be
    #    moderately diverse and "safe". We treat mid-range TTR as slightly AI-leaning and
    #    extreme variety (very high TTR) as human-leaning.
    ttr = len(set(words)) / len(words)
    ttr_ai = 1.0 - _normalize(ttr, 0.35, 0.75)  # higher diversity => more human

    # 3) Punctuation density: low density (plain text) -> AI-like.
    punct_density = len(_PUNCT_RE.findall(text)) / max(len(words), 1)
    punct_ai = 1.0 - _normalize(punct_density, 0.02, 0.15)  # sparse punctuation => AI

    # 4) Mean sentence length: AI clusters around 18-25 words; extremes are more human.
    mean_len = statistics.mean(sentence_lengths)
    # Distance from the "AI sweet spot" (~22) -> closer means more AI-like.
    sweet_spot_ai = 1.0 - _normalize(abs(mean_len - 22.0), 0.0, 18.0)

    # Average the four partial scores into one signal score.
    score = statistics.mean([burstiness_ai, ttr_ai, punct_ai, sweet_spot_ai])

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "metrics": {
            "sentence_length_stdev": round(stdev, 2),
            "type_token_ratio": round(ttr, 3),
            "punctuation_density": round(punct_density, 3),
            "mean_sentence_length": round(mean_len, 1),
        },
    }
