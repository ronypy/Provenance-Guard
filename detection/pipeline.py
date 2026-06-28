"""The detection pipeline — fan a text out to all three signals and combine the result.

``run_pipeline(text)`` is the single entry point the Flask app calls. It returns everything
needed to build a ``/submit`` response and an audit-log entry.

The LLM signal can be disabled (``use_llm=False``) so the stylometric + phrase signals can be
exercised offline without an API key — useful for tests and for graceful degradation.
"""

from __future__ import annotations

from .labels import make_label
from .phrase_signal import phrase_signal
from .scoring import score_text
from .stylometric import stylometric_signal


def run_pipeline(text: str, *, use_llm: bool = True) -> dict:
    """Run all detection signals on ``text`` and return the combined verdict.

    Returns a dict with: ``signals`` (per-signal scores + detail), ``combined_score``,
    ``confidence``, ``attribution``, and the reader-facing ``label``.
    """
    stylo = stylometric_signal(text)
    phrase = phrase_signal(text)

    if use_llm:
        from .llm_signal import llm_signal  # imported lazily so offline tests need no key

        llm = llm_signal(text)
    else:
        # Re-weight onto the two structural signals when the LLM is unavailable: average them
        # so the combined score still spans [0, 1] honestly instead of being dragged to 0.5.
        llm = {"score": (stylo["score"] + phrase["score"]) / 2, "reasoning": "LLM signal disabled"}

    scored = score_text(llm["score"], stylo["score"], phrase["score"])

    label = make_label(scored["attribution"], scored["confidence"])

    return {
        "signals": {
            "llm_score": round(llm["score"], 4),
            "llm_reasoning": llm.get("reasoning", ""),
            "stylometric_score": stylo["score"],
            "stylometric_metrics": stylo.get("metrics", {}),
            "phrase_score": phrase["score"],
            "phrase_hits": phrase.get("hits", []),
        },
        "combined_score": scored["combined_score"],
        "confidence": scored["confidence"],
        "attribution": scored["attribution"],
        "label": label,
    }
