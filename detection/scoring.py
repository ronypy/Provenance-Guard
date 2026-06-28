"""Confidence scoring — combine the signals into one calibrated verdict.

Each signal returns ``P(AI)`` in ``[0, 1]``. We combine them with documented weights, derive a
*confidence* (decisiveness) value, and map the combined score to one of three attribution
classes using deliberately asymmetric thresholds.

Design rationale (see planning.md):

* ``confidence = 2 * |combined_score - 0.5|`` -> 0.5 means maximum uncertainty (confidence ~0).
* The bar to call something AI (0.70) is HIGHER than the bar to call it human (0.40), because on
  a writing platform a false positive (labeling a human as AI) is worse than a false negative.
"""

from __future__ import annotations

from config import (
    THRESHOLD_AI,
    THRESHOLD_HUMAN,
    WEIGHT_LLM,
    WEIGHT_PHRASE,
    WEIGHT_STYLOMETRIC,
)

# Weights (0.5/0.3/0.2) and thresholds (AI 0.70, human 0.40) are defined in config.py.
# between the two thresholds -> uncertain.

LIKELY_AI = "likely_ai"
LIKELY_HUMAN = "likely_human"
UNCERTAIN = "uncertain"


def combine_scores(llm_score: float, stylometric_score: float, phrase_score: float) -> float:
    """Weighted average of the three signal scores -> combined P(AI)."""
    combined = (
        WEIGHT_LLM * llm_score
        + WEIGHT_STYLOMETRIC * stylometric_score
        + WEIGHT_PHRASE * phrase_score
    )
    return round(max(0.0, min(1.0, combined)), 4)


def confidence_from_score(combined_score: float) -> float:
    """Decisiveness in [0, 1]: 0 at score 0.5, 1 at either pole."""
    return round(2.0 * abs(combined_score - 0.5), 4)


def attribution_from_score(combined_score: float) -> str:
    """Map the combined score to one of the three attribution classes."""
    if combined_score >= THRESHOLD_AI:
        return LIKELY_AI
    if combined_score <= THRESHOLD_HUMAN:
        return LIKELY_HUMAN
    return UNCERTAIN


def score_text(llm_score: float, stylometric_score: float, phrase_score: float) -> dict:
    """Combine signal scores into ``combined_score``, ``confidence`` and ``attribution``."""
    combined = combine_scores(llm_score, stylometric_score, phrase_score)
    return {
        "combined_score": combined,
        "confidence": confidence_from_score(combined),
        "attribution": attribution_from_score(combined),
    }
