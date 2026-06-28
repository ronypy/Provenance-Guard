"""Structured storage for Provenance Guard.

Two files live under ``logs/``:

* ``audit.jsonl``         — append-only audit log, one JSON object per line. This ``.jsonl``
                            format is the production-standard, log-aggregator-friendly approach
                            taught in the Lab 4 worked example: cheap to append, easy to parse.
* ``content_store.json``  — a mutable map of ``content_id`` -> the latest record, used to look
                            up and update a piece of content's status when an appeal arrives.

Paths come from ``config.py`` and are resolved relative to the project root so the service works
regardless of the current working directory.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CONTENT_STORE_FILE, LOG_FILE

BASE_DIR = Path(__file__).resolve().parent
AUDIT_LOG_PATH = BASE_DIR / LOG_FILE
CONTENT_STORE_PATH = BASE_DIR / CONTENT_STORE_FILE

# A single process-wide lock keeps concurrent requests from corrupting the files.
_lock = threading.Lock()


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Audit log (append-only JSONL)
# --------------------------------------------------------------------------- #
def append_audit_entry(entry: dict) -> None:
    """Append a single structured entry as one JSON line to the audit log."""
    with _lock:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # One-line console summary so logged decisions are visible in real time.
    print(
        f"[LOGGED] event={entry.get('event')} "
        f"attribution={entry.get('attribution')} "
        f"confidence={entry.get('confidence')} "
        f"content_id={entry.get('content_id')}"
    )


def get_log(limit: int = 50) -> list[dict]:
    """Return the most recent ``limit`` audit entries, newest last."""
    if not AUDIT_LOG_PATH.exists():
        return []
    entries: list[dict] = []
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a malformed line rather than failing the whole read
    return entries[-limit:]


# --------------------------------------------------------------------------- #
# Content store (mutable JSON map)
# --------------------------------------------------------------------------- #
def _read_store() -> dict:
    if not CONTENT_STORE_PATH.exists():
        return {}
    try:
        with CONTENT_STORE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_store(store: dict) -> None:
    CONTENT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONTENT_STORE_PATH.with_suffix(CONTENT_STORE_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, ensure_ascii=False)
    tmp.replace(CONTENT_STORE_PATH)  # atomic on POSIX


def save_content(record: dict) -> None:
    """Insert/replace a content record keyed by its ``content_id``."""
    with _lock:
        store = _read_store()
        store[record["content_id"]] = record
        _write_store(store)


def get_content(content_id: str) -> dict | None:
    """Return the content record for ``content_id`` or ``None`` if unknown."""
    return _read_store().get(content_id)


def update_content_status(content_id: str, status: str, **fields: Any) -> dict | None:
    """Update a content record's status (and any extra fields). Returns the record or ``None``."""
    with _lock:
        store = _read_store()
        record = store.get(content_id)
        if record is None:
            return None
        record["status"] = status
        record.update(fields)
        store[content_id] = record
        _write_store(store)
        return record
