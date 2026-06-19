"""
In-memory chat log store for tests and optional local feedback flows.

Persistent database logging is not used by this service.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "null"


def new_event_id() -> str:
    return str(uuid.uuid4())


@dataclass
class ChatEventRecord:
    """In-memory representation of a chat exchange ready for persistence."""

    event_id: str
    session_id: str
    corpus_id: str
    query: str
    answer: str
    source: str
    method: str
    retrieval_model: str
    used_hf: bool
    top_k: int
    min_score: float
    allow_fallback: bool
    latency_ms: int
    category: str = "chat"
    bucket: str = "answered"
    info: dict = field(default_factory=dict)


@dataclass
class ChatFeedbackRecord:
    """In-memory representation of a feedback entry for a chat event."""

    event_id: str
    session_id: str
    rating: int
    comment: str = ""
    category: str = "feedback"
    bucket: str = "neutral"
    info: dict = field(default_factory=dict)


class ChatLogStoreProtocol(Protocol):
    def insert_event(self, record: ChatEventRecord) -> None: ...
    def insert_feedback(self, record: ChatFeedbackRecord) -> None: ...
    def event_exists(self, event_id: str) -> bool: ...
    def fetch_recent_events(self, limit: int = 50) -> List[dict]: ...
    def fetch_recent_feedback(self, limit: int = 50) -> List[dict]: ...


class InMemoryChatLogStore:
    """Process-local store for unit tests and optional in-memory logging."""

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []
        self._feedback: List[Dict[str, Any]] = []
        self._counter = 0

    def insert_event(self, record: ChatEventRecord) -> None:
        self._counter += 1
        now = _utc_now_iso()
        self._events.append(
            {
                "id": self._counter,
                "event_id": record.event_id,
                "session_id": record.session_id,
                "corpus_id": record.corpus_id,
                "category": record.category,
                "bucket": record.bucket,
                "query": record.query,
                "answer": record.answer,
                "source": record.source,
                "method": record.method,
                "retrieval_model": record.retrieval_model,
                "used_hf": bool(record.used_hf),
                "top_k": int(record.top_k),
                "min_score": float(record.min_score),
                "allow_fallback": bool(record.allow_fallback),
                "latency_ms": int(record.latency_ms),
                "info": _to_json(record.info),
                "created_at": now,
                "updated_at": now,
            }
        )

    def insert_feedback(self, record: ChatFeedbackRecord) -> None:
        now = _utc_now_iso()
        self._feedback.append(
            {
                "id": len(self._feedback) + 1,
                "event_id": record.event_id,
                "session_id": record.session_id,
                "category": record.category,
                "bucket": record.bucket,
                "rating": int(record.rating),
                "comment": record.comment,
                "info": _to_json(record.info),
                "created_at": now,
                "updated_at": now,
            }
        )

    def event_exists(self, event_id: str) -> bool:
        return any(item["event_id"] == event_id for item in self._events)

    def fetch_recent_events(self, limit: int = 50) -> List[dict]:
        return list(reversed(self._events[-max(1, min(limit, 1000)) :]))

    def fetch_recent_feedback(self, limit: int = 50) -> List[dict]:
        return list(reversed(self._feedback[-max(1, min(limit, 1000)) :]))


_GLOBAL_STORE: Optional[ChatLogStoreProtocol] = None
_GLOBAL_STORE_LOCK = threading.Lock()


def _logging_enabled() -> bool:
    raw = os.getenv("CHAT_LOG_ENABLED", "false").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def get_store_diagnostics() -> Dict[str, Any]:
    store = _GLOBAL_STORE
    return {
        "enabled": _logging_enabled(),
        "store_ready": store is not None,
        "store_kind": type(store).__name__ if store is not None else "none",
    }


def get_store() -> Optional[ChatLogStoreProtocol]:
    """Return the in-memory store when enabled and initialized (tests call reset_store_for_tests)."""

    if _GLOBAL_STORE is not None:
        return _GLOBAL_STORE
    if not _logging_enabled():
        return None
    return None


def reset_store_for_tests() -> ChatLogStoreProtocol:
    """Reset the global store (intended for unit tests only)."""

    global _GLOBAL_STORE
    with _GLOBAL_STORE_LOCK:
        _GLOBAL_STORE = InMemoryChatLogStore()
        return _GLOBAL_STORE
