"""
Supabase Postgres-backed persistence for chat exchanges and feedback.

Schema follows project conventions: every table includes
`id`, `created_at`, `updated_at`, `category`, `bucket`, and `info` (JSONB).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

import psycopg
from psycopg.rows import dict_row

LOGGER = logging.getLogger(__name__)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "null"


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


class ChatLogStore:
    """Thread-safe Postgres chat log persistence layer."""

    _SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS chat_events (
            id BIGSERIAL PRIMARY KEY,
            event_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            corpus_id TEXT NOT NULL,
            category TEXT NOT NULL,
            bucket TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT '',
            method TEXT NOT NULL DEFAULT '',
            retrieval_model TEXT NOT NULL DEFAULT '',
            used_hf BOOLEAN NOT NULL DEFAULT FALSE,
            top_k INTEGER NOT NULL DEFAULT 0,
            min_score REAL NOT NULL DEFAULT 0.0,
            allow_fallback BOOLEAN NOT NULL DEFAULT TRUE,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            info JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_feedback (
            id BIGSERIAL PRIMARY KEY,
            event_id TEXT NOT NULL,
            session_id TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL,
            bucket TEXT NOT NULL,
            rating INTEGER NOT NULL DEFAULT 0,
            comment TEXT NOT NULL DEFAULT '',
            info JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            FOREIGN KEY (event_id) REFERENCES chat_events(event_id)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chat_events_session
            ON chat_events(session_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chat_events_created_at
            ON chat_events(created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chat_events_bucket
            ON chat_events(bucket)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chat_feedback_event_id
            ON chat_feedback(event_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chat_feedback_bucket
            ON chat_feedback(bucket)
        """,
    )

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._lock = threading.Lock()
        self._initialize_schema()

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._db_url, row_factory=dict_row)

    def _initialize_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    for ddl in self._SCHEMA:
                        cur.execute(ddl)
                conn.commit()

    @staticmethod
    def new_event_id() -> str:
        return str(uuid.uuid4())

    def insert_event(self, record: ChatEventRecord) -> None:
        now = _utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                INSERT INTO chat_events (
                    event_id, session_id, corpus_id, category, bucket,
                    query, answer, source, method, retrieval_model,
                    used_hf, top_k, min_score, allow_fallback, latency_ms,
                    info, created_at, updated_at
                ) VALUES (
                    %(event_id)s, %(session_id)s, %(corpus_id)s, %(category)s, %(bucket)s,
                    %(query)s, %(answer)s, %(source)s, %(method)s, %(retrieval_model)s,
                    %(used_hf)s, %(top_k)s, %(min_score)s, %(allow_fallback)s, %(latency_ms)s,
                    %(info)s::jsonb, %(created_at)s::timestamptz, %(updated_at)s::timestamptz
                )
                """,
                        {
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
                        },
                    )
                conn.commit()

    def insert_feedback(self, record: ChatFeedbackRecord) -> None:
        now = _utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                INSERT INTO chat_feedback (
                    event_id, session_id, category, bucket,
                    rating, comment, info, created_at, updated_at
                ) VALUES (
                    %(event_id)s, %(session_id)s, %(category)s, %(bucket)s,
                    %(rating)s, %(comment)s, %(info)s::jsonb,
                    %(created_at)s::timestamptz, %(updated_at)s::timestamptz
                )
                """,
                        {
                            "event_id": record.event_id,
                            "session_id": record.session_id,
                            "category": record.category,
                            "bucket": record.bucket,
                            "rating": int(record.rating),
                            "comment": record.comment,
                            "info": _to_json(record.info),
                            "created_at": now,
                            "updated_at": now,
                        },
                    )
                conn.commit()

    def event_exists(self, event_id: str) -> bool:
        if not event_id:
            return False
        with self._lock:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM chat_events WHERE event_id = %(event_id)s LIMIT 1",
                        {"event_id": event_id},
                    )
                    row = cur.fetchone()
        return row is not None

    def fetch_recent_events(self, limit: int = 50) -> List[dict]:
        with self._lock:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                SELECT * FROM chat_events
                ORDER BY id DESC
                LIMIT %(limit)s
                """,
                        {"limit": max(1, min(limit, 1000))},
                    )
                    rows = cur.fetchall()
        return [dict(row) for row in rows or []]

    def fetch_recent_feedback(self, limit: int = 50) -> List[dict]:
        with self._lock:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                SELECT * FROM chat_feedback
                ORDER BY id DESC
                LIMIT %(limit)s
                """,
                        {"limit": max(1, min(limit, 1000))},
                    )
                    rows = cur.fetchall()
        return [dict(row) for row in rows or []]


class InMemoryChatLogStore:
    """Used in tests to avoid requiring a running Postgres instance."""

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


def _resolve_db_url() -> str:
    raw = os.getenv("SUPABASE_DB_URL", "").strip()
    if raw:
        return raw
    return ""


def get_store() -> Optional[ChatLogStoreProtocol]:
    """Return process-wide store, or None if disabled via env.

    Set `CHAT_LOG_ENABLED=false` to fully disable persistence (useful for
    ephemeral test environments). Defaults to enabled.
    """

    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        return _GLOBAL_STORE
    raw = os.getenv("CHAT_LOG_ENABLED", "true").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return None
    db_url = _resolve_db_url()
    if not db_url:
        LOGGER.warning("CHAT_LOG_ENABLED=true but SUPABASE_DB_URL is missing; logging disabled.")
        return None
    with _GLOBAL_STORE_LOCK:
        if _GLOBAL_STORE is None:
            try:
                _GLOBAL_STORE = ChatLogStore(db_url=db_url)
            except Exception:
                LOGGER.exception("Failed to initialize chat log store")
                return None
    return _GLOBAL_STORE


def reset_store_for_tests() -> ChatLogStoreProtocol:
    """Reset the global store (intended for unit tests only)."""

    global _GLOBAL_STORE
    with _GLOBAL_STORE_LOCK:
        _GLOBAL_STORE = InMemoryChatLogStore()
        return _GLOBAL_STORE
