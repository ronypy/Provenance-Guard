"""Transparency labels — the plain-language text a reader sees.

Three variants, one per attribution class. The text is intentionally non-technical: it leads
with an emoji and a short verdict, states the confidence as a percentage, and — for the AI
variant, the harmful-error case — explicitly invites an appeal.

These strings are the canonical label text and must match planning.md and the README verbatim.
"""

from __future__ import annotations

from .scoring import LIKELY_AI, LIKELY_HUMAN, UNCERTAIN


def _pct(confidence: float) -> int:
    return round(confidence * 100)


def make_label(attribution: str, confidence: float) -> str:
    """Return the reader-facing transparency label for an attribution + confidence."""
    pct = _pct(confidence)
    if attribution == LIKELY_AI:
        return (
            f"🤖 Likely AI-generated — our analysis strongly indicates this text was "
            f"produced by an AI system. Confidence: {pct}%. The creator can appeal this result."
        )
    if attribution == LIKELY_HUMAN:
        return (
            f"✍️ Likely human-written — our analysis strongly indicates a person wrote "
            f"this text. Confidence: {pct}%."
        )
    if attribution == UNCERTAIN:
        return (
            f"❓ Uncertain origin — we could not determine with confidence whether this text "
            f"was written by a person or an AI. Treat attribution as inconclusive. "
            f"Confidence: {pct}%."
        )
    raise ValueError(f"Unknown attribution: {attribution!r}")
