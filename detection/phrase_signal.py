"""Signal 3 (ensemble) — AI-tell phrase density (pure Python).

The third signal that makes Provenance Guard an *ensemble*. It looks for the formulaic
connective and hedging phrases that large language models reach for far more often than people
do in original creative writing — "Furthermore", "It is important to note", "plays a crucial
role", "In conclusion", and friends.

This is a *lexical fingerprint* signal: orthogonal to the LLM's holistic read (Signal 1) and to
the structural statistics of Signal 2. It measures specific word choices, not meaning or shape.

Output: ``P(AI)`` in ``[0, 1]`` based on how dense these tells are relative to text length.

Blind spot: formal human essays legitimately use these phrases, and the signal is trivially
evaded by an author who avoids them — which is why it carries the lowest weight in scoring.
"""

from __future__ import annotations

import re

# Phrases strongly associated with LLM output. Matched case-insensitively as substrings.
_AI_TELL_PHRASES = (
    "furthermore",
    "moreover",
    "in conclusion",
    "it is important to note",
    "it is worth noting",
    "plays a crucial role",
    "plays a vital role",
    "plays a significant role",
    "a testament to",
    "in today's world",
    "in the modern world",
    "delve into",
    "navigate the complexities",
    "it is essential to",
    "it is important to consider",
    "a wide range of",
    "stakeholders",
    "paradigm shift",
    "transformative",
    "on the other hand",
    "as a result",
    "in summary",
    "overall",
    "ultimately",
    "additionally",
)

_WORD_RE = re.compile(r"[A-Za-z']+")


def phrase_signal(text: str) -> dict:
    """Return ``{"score": P(AI), "hits": [...], "count": int}`` from AI-tell phrase density."""
    lowered = text.lower()
    word_count = max(len(_WORD_RE.findall(text)), 1)

    hits = [phrase for phrase in _AI_TELL_PHRASES if phrase in lowered]
    count = len(hits)

    # Density per 100 words, so long and short texts are compared fairly.
    density = count / word_count * 100.0

    # Rubric: 0 tells -> 0.15 (mild human lean); each unit of density per-100-words adds weight,
    # saturating near 0.9. A single tell in a short paragraph already pushes toward "AI-leaning".
    score = 0.15 + min(0.75, density * 0.45)

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "hits": hits,
        "count": count,
    }
