"""Centralized configuration and constants for Provenance Guard.

Loads the environment once (so ``GROQ_API_KEY`` is available everywhere) and holds the
cross-cutting constants: the LLM model, storage paths, rate limits, and the scoring
weights/thresholds. Keeping these in one place mirrors the course convention and makes the
tunable knobs easy to find.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM (detection signal 1) ---------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# --- Storage --------------------------------------------------------------------------------
LOG_FILE = "logs/audit.jsonl"                    # append-only structured audit log (one JSON/line)
CONTENT_STORE_FILE = "logs/content_store.json"   # mutable map: content_id -> latest record

# --- Rate limiting (applied to POST /submit) ------------------------------------------------
RATE_LIMITS = "10 per minute;100 per day"

# --- Confidence scoring (see planning.md) ---------------------------------------------------
# Signal weights — the LLM is the strongest signal, phrase density the most evadable.
WEIGHT_LLM = 0.5
WEIGHT_STYLOMETRIC = 0.3
WEIGHT_PHRASE = 0.2

# Attribution thresholds on the combined score (asymmetric: the bar to call something AI is
# higher than the bar to call it human, because false positives are worse on a writing platform).
THRESHOLD_AI = 0.70       # >= this -> likely_ai
THRESHOLD_HUMAN = 0.40    # <= this -> likely_human
