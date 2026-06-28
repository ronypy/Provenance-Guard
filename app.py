"""Provenance Guard — Flask backend.

Endpoints
---------
* ``POST /submit``  — classify a piece of text (rate-limited). Runs the multi-signal pipeline,
                      writes a content record + audit entry, returns the verdict and label.
* ``POST /appeal``  — a creator contests a classification. Flips status to ``under_review`` and
                      logs the appeal alongside the original decision. No re-classification.
* ``GET  /log``     — returns the most recent audit-log entries as JSON.
* ``GET  /health``  — liveness check.

Run with:  flask --app app run        (or:  python app.py)
"""

from __future__ import annotations

import uuid

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import RATE_LIMITS  # importing config also loads .env (GROQ_API_KEY)
from detection import run_pipeline
from storage import (
    append_audit_entry,
    get_content,
    get_log,
    save_content,
    update_content_status,
    utc_now_iso,
)

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
# 10/min absorbs a writer's legitimate bursts (re-editing and resubmitting their own work),
# while 100/day caps sustained scripted abuse. See README for the full reasoning.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

EXCERPT_LEN = 280  # how much of the submitted text to keep in the audit log


def _excerpt(text: str) -> str:
    return text if len(text) <= EXCERPT_LEN else text[:EXCERPT_LEN] + "…"


@app.errorhandler(429)
def rate_limit_exceeded(error):
    """Return a JSON body (not HTML) when a client exceeds the rate limit."""
    return (
        jsonify(
            {
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded: {error.description}. Please slow down.",
            }
        ),
        429,
    )


@app.route("/submit", methods=["POST"])
@limiter.limit(RATE_LIMITS)
def submit():
    """Classify a submitted piece of text."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = (data.get("creator_id") or "anonymous").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required and must be non-empty."}), 400

    result = run_pipeline(text)

    content_id = str(uuid.uuid4())
    timestamp = utc_now_iso()

    record = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": timestamp,
        "event": "classified",
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "combined_score": result["combined_score"],
        "signals": {
            "llm_score": result["signals"]["llm_score"],
            "stylometric_score": result["signals"]["stylometric_score"],
            "phrase_score": result["signals"]["phrase_score"],
        },
        "label": result["label"],
        "status": "classified",
        "appeal_reasoning": None,
        "text_excerpt": _excerpt(text),
    }

    save_content(record)
    append_audit_entry(record)

    return jsonify(
        {
            "content_id": content_id,
            "attribution": result["attribution"],
            "confidence": result["confidence"],
            "combined_score": result["combined_score"],
            "signals": result["signals"],
            "label": result["label"],
            "status": "classified",
        }
    )


@app.route("/appeal", methods=["POST"])
def appeal():
    """A creator contests a classification for a previously submitted piece of content."""
    data = request.get_json(silent=True) or {}
    content_id = (data.get("content_id") or "").strip()
    creator_reasoning = (data.get("creator_reasoning") or "").strip()

    if not content_id or not creator_reasoning:
        return (
            jsonify({"error": "Fields 'content_id' and 'creator_reasoning' are required."}),
            400,
        )

    original = get_content(content_id)
    if original is None:
        return jsonify({"error": f"No content found with id {content_id}."}), 404

    updated = update_content_status(
        content_id, "under_review", appeal_reasoning=creator_reasoning
    )

    # Log the appeal as its own event alongside the original decision.
    appeal_entry = {
        "content_id": content_id,
        "creator_id": original.get("creator_id"),
        "timestamp": utc_now_iso(),
        "event": "appealed",
        "attribution": original.get("attribution"),
        "confidence": original.get("confidence"),
        "combined_score": original.get("combined_score"),
        "signals": original.get("signals"),
        "label": original.get("label"),
        "status": "under_review",
        "appeal_reasoning": creator_reasoning,
        "text_excerpt": original.get("text_excerpt"),
    }
    append_audit_entry(appeal_entry)

    return jsonify(
        {
            "content_id": content_id,
            "status": "under_review",
            "message": "Appeal received. This content is now under review by a human moderator.",
            "original_attribution": original.get("attribution"),
            "original_confidence": updated.get("confidence") if updated else None,
        }
    )


@app.route("/log", methods=["GET"])
def log():
    """Return the most recent audit-log entries as JSON."""
    return jsonify({"entries": get_log()})


@app.route("/", methods=["GET"])
def index():
    """Landing page — describes the API so the base URL isn't a bare 404 in a browser."""
    return jsonify(
        {
            "service": "provenance-guard",
            "description": "Classifies whether submitted text is AI-generated or human-written.",
            "endpoints": {
                "POST /submit": "Classify text. Body: {text, creator_id}. Rate-limited 10/min;100/day.",
                "POST /appeal": "Contest a classification. Body: {content_id, creator_reasoning}.",
                "GET /log": "Most recent audit-log entries as JSON.",
                "GET /health": "Liveness check.",
            },
            "note": "This is a JSON API — use curl or a REST client, not a browser form.",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "provenance-guard"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
