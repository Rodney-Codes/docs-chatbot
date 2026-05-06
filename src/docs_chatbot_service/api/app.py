from __future__ import annotations

import json
import logging
import os
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from urllib import error, request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from docs_chatbot_service.core.chat_log_store import (
    ChatEventRecord,
    ChatFeedbackRecord,
    ChatLogStore,
    get_store_diagnostics,
    get_store,
)
from docs_chatbot_service.core.query_nlp import (
    analyze_query,
    detect_intent_from_signals,
    detect_skill_entity,
    extract_best_sentences,
)
from docs_chatbot_service.core.service import RetrievalModel, RetrievalService, SearchParams
from docs_chatbot_service.core.text_util import tokenize

LOGGER = logging.getLogger(__name__)

load_dotenv()

INDEX_ROOT = Path("data/index")
app = FastAPI(title="Docs Chatbot Service")
service = RetrievalService(index_root=INDEX_ROOT)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    corpus_id: str = Field(default="default")
    doc_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0)
    retrieval_model: Optional[RetrievalModel] = Field(
        default=None,
        description=(
            "bm25 | hashed_vector | bm25_hashed_vector | rule_lexicon_tfidf "
            "(retrieval scoring pipeline)"
        ),
    )
    chunks_url: Optional[str] = Field(
        default=None,
        description="Optional artifact URL for chunks.json. If provided, service loads this corpus before search.",
    )
    vector_index_url: Optional[str] = Field(
        default=None,
        description="Optional artifact URL for vector_index.json. Used with chunks_url or alone.",
    )


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    source: str
    snippet: str
    score: float


class SearchResponse(BaseModel):
    query: str
    corpus_id: str
    total_results: int
    results: List[SearchResult]
    retrieval_model: str = Field(
        ...,
        description=(
            "bm25 | hashed_vector | bm25_hashed_vector | rule_lexicon_tfidf "
            "(which retrieval scorer was used)"
        ),
    )


class ChatAnswerMethod(str, Enum):
    """Composable answer pipelines built from answer functions."""

    hugging_face_lightweight_nlp = "hugging_face_lightweight_nlp"
    hugging_face = "hugging_face"
    lightweight_nlp = "lightweight_nlp"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2)
    corpus_id: str = Field(default="default")
    doc_ids: Optional[List[str]] = None
    top_k: int = Field(default=3, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0)
    allow_fallback: Optional[bool] = Field(default=None)
    answer_method: Optional[ChatAnswerMethod] = Field(
        default=None,
        description=(
            "hugging_face | lightweight_nlp | hugging_face_lightweight_nlp "
            "(answer pipeline)"
        ),
    )
    retrieval_model: Optional[RetrievalModel] = Field(
        default=None,
        description=(
            "bm25 | hashed_vector | bm25_hashed_vector | rule_lexicon_tfidf "
            "(retrieval scoring pipeline)"
        ),
    )
    chunks_url: Optional[str] = Field(
        default=None,
        description="Optional artifact URL for chunks.json. If provided, service loads this corpus before chat.",
    )
    vector_index_url: Optional[str] = Field(
        default=None,
        description="Optional artifact URL for vector_index.json. Used with chunks_url or alone.",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Stable client-supplied session identifier for grouping chat events.",
    )


class ChatResponse(BaseModel):
    query: str
    corpus_id: str
    answer: str
    source: str
    used_hf: bool
    method: str = Field(
        ...,
        description="hugging_face | lightweight_nlp | none (which path produced the answer)",
    )
    retrieval_model: str = Field(
        ...,
        description=(
            "bm25 | hashed_vector | bm25_hashed_vector | rule_lexicon_tfidf "
            "(which retrieval scorer was used)"
        ),
    )
    event_id: str = Field(
        default="",
        description="Stable identifier for this chat exchange. Used to attach feedback later.",
    )
    session_id: str = Field(
        default="",
        description="Echoed-back session identifier (server-generated if not supplied).",
    )


class ChatFeedbackRequest(BaseModel):
    event_id: str = Field(..., min_length=1, max_length=128)
    rating: int = Field(
        ...,
        ge=-1,
        le=1,
        description="-1 (negative), 0 (neutral/correction-only), or 1 (positive)",
    )
    comment: Optional[str] = Field(default="", max_length=2000)
    session_id: Optional[str] = Field(default=None, max_length=128)


class ChatFeedbackResponse(BaseModel):
    event_id: str
    accepted: bool
    rating: int
    bucket: str


class LoggingHealthResponse(BaseModel):
    enabled: bool
    db_url_present: bool
    project_url_present: bool
    service_role_key_present: bool
    store_ready: bool
    store_kind: str
    last_init_error: str


NO_ANSWER_MESSAGE = (
    "I'm unable to answer this query, please contact Rohit to get an answer for this query"
)


def _resolve_allow_fallback(request_body: ChatRequest) -> bool:
    if request_body.allow_fallback is not None:
        return request_body.allow_fallback
    return _env_bool("CHAT_ALLOW_FALLBACK", True)


def _resolve_retrieval_model(
    explicit: Optional[RetrievalModel],
) -> RetrievalModel:
    return explicit or RetrievalModel.bm25_hashed_vector


def _chat_answer_from_results(
    query: str,
    results: List[dict],
    answer_method: ChatAnswerMethod,
    allow_fallback: bool,
) -> Tuple[str, str, bool, str]:
    """
    Returns (answer, source, used_hf, method) where method is which generator produced the answer:
    hugging_face | lightweight_nlp | none
    """
    top = results[0]
    source = str(top.get("source", ""))

    if answer_method == ChatAnswerMethod.lightweight_nlp:
        if not allow_fallback:
            return NO_ANSWER_MESSAGE, source, False, "none"
        return _build_fallback_answer(query, results), source, False, "lightweight_nlp"

    if answer_method == ChatAnswerMethod.hugging_face:
        hf_answer = _generate_with_hf(query, results)
        if hf_answer:
            return hf_answer, source, True, "hugging_face"
        return "", source, False, "none"

    hf_answer = _generate_with_hf(query, results)
    if hf_answer:
        return hf_answer, source, True, "hugging_face"
    if not allow_fallback:
        return NO_ANSWER_MESSAGE, source, False, "none"
    return _build_fallback_answer(query, results), source, False, "lightweight_nlp"


class CorpusStatsResponse(BaseModel):
    corpus_id: str
    total_chunks: int
    total_docs: int


class CorpusExistsResponse(BaseModel):
    corpus_id: str
    exists: bool


class CorpusLoadRequest(BaseModel):
    corpus_id: str = Field(default="default")
    chunks_url: str = Field(..., min_length=8)
    vector_index_url: Optional[str] = Field(default=None)


class CorpusLoadResponse(BaseModel):
    corpus_id: str
    chunks_loaded: bool
    vector_loaded: bool
    chunks_path: str
    vector_path: str


def _fetch_json_from_url(url: str) -> object:
    req = request.Request(url, headers={"User-Agent": "docs-chatbot-service/1.0"}, method="GET")
    with request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_corpus_artifacts(corpus_id: str, chunks_url: str, vector_index_url: Optional[str]) -> CorpusLoadResponse:
    try:
        chunks_payload = _fetch_json_from_url(chunks_url)
    except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not load chunks_url: {exc}") from exc
    if not isinstance(chunks_payload, list):
        raise HTTPException(status_code=400, detail="chunks_url must resolve to a JSON list.")

    chunks_path = service._storage.save_chunks(corpus_id, chunks_payload)
    vector_loaded = False
    vector_path = ""
    if vector_index_url:
        try:
            vector_payload = _fetch_json_from_url(vector_index_url)
        except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Could not load vector_index_url: {exc}") from exc
        if not isinstance(vector_payload, dict):
            raise HTTPException(status_code=400, detail="vector_index_url must resolve to a JSON object.")
        vp = service._storage.save_vector_index(corpus_id, vector_payload)
        vector_loaded = True
        vector_path = str(vp)

    service.invalidate_cache()
    return CorpusLoadResponse(
        corpus_id=corpus_id,
        chunks_loaded=True,
        vector_loaded=vector_loaded,
        chunks_path=str(chunks_path),
        vector_path=vector_path,
    )


def _maybe_load_corpus_from_request(
    corpus_id: str, chunks_url: Optional[str], vector_index_url: Optional[str]
) -> None:
    if not chunks_url and not vector_index_url:
        return
    if not chunks_url:
        raise HTTPException(
            status_code=400,
            detail="chunks_url is required when vector_index_url is provided.",
        )
    _load_corpus_artifacts(corpus_id, chunks_url, vector_index_url)


def _maybe_autoload_default_corpus(corpus_id: str) -> None:
    if corpus_id != "default":
        return
    if service.corpus_exists(corpus_id):
        return
    chunks_url = os.getenv("CHATBOT_CHUNKS_URL", "").strip()
    vector_index_url = os.getenv("CHATBOT_VECTOR_INDEX_URL", "").strip() or None
    if not chunks_url:
        return
    try:
        _load_corpus_artifacts(corpus_id, chunks_url, vector_index_url)
        LOGGER.info("Auto-loaded default corpus from configured artifact URLs.")
    except Exception:  # pragma: no cover
        LOGGER.exception("Failed to auto-load default corpus from CHATBOT_* URLs")


def _resolve_session_id(supplied: Optional[str]) -> str:
    candidate = (supplied or "").strip()
    if candidate:
        return candidate[:128]
    return f"anon-{ChatLogStore.new_event_id()}"


def _bucket_for_method(method: str, results_count: int) -> str:
    if results_count == 0:
        return "no_results"
    if method == "hugging_face":
        return "hf"
    if method == "lightweight_nlp":
        return "nlp"
    return "none"


def _build_event_info(
    request_body: ChatRequest,
    results: List[dict],
) -> dict:
    top_results_summary = [
        {
            "chunk_id": str(r.get("chunk_id", "")),
            "doc_id": str(r.get("doc_id", "")),
            "title": str(r.get("title", "")),
            "section": str(r.get("section", "general")),
            "score": float(r.get("score", 0.0) or 0.0),
        }
        for r in results[: max(1, request_body.top_k)]
    ]
    scores = [item["score"] for item in top_results_summary]
    return {
        "doc_ids": list(request_body.doc_ids or []),
        "answer_method_requested": (
            request_body.answer_method.value
            if request_body.answer_method is not None
            else None
        ),
        "chunks_url_provided": bool(request_body.chunks_url),
        "vector_index_url_provided": bool(request_body.vector_index_url),
        "top_results": top_results_summary,
        "score_stats": {
            "max": max(scores) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
            "mean": (sum(scores) / len(scores)) if scores else 0.0,
            "count": len(scores),
        },
    }


def _log_chat_event_safe(record: ChatEventRecord) -> None:
    store = get_store()
    if store is None:
        return
    try:
        store.insert_event(record)
    except Exception:  # pragma: no cover
        LOGGER.exception("Failed to persist chat event %s", record.event_id)


def _extract_years_value(text: str) -> str:
    import re

    match = re.search(r"(\d+\+?)\s+years?", text, re.IGNORECASE)
    return match.group(1) if match else ""


def _build_fallback_answer(query: str, results: List[dict]) -> str:
    signals = analyze_query(query)
    intent = detect_intent_from_signals(signals)
    query_tokens = tokenize(query)
    asks_years = "year" in query_tokens or "years" in query_tokens
    skill_entity = detect_skill_entity(query)
    top = results[0]
    second = results[1] if len(results) > 1 else {}

    lines = extract_best_sentences(str(top.get("snippet", "")), signals, 2)
    if len(lines) < 2 and second:
        lines.extend(extract_best_sentences(str(second.get("snippet", "")), signals, 1))
    concise = " ".join(lines).strip()
    years_value = _extract_years_value(
        f"{top.get('snippet', '')} {second.get('snippet', '') if second else ''}"
    )

    if asks_years and skill_entity and years_value:
        tech_label = "AWS" if skill_entity.lower() == "aws" else skill_entity.upper()
        evidence = concise or "my portfolio highlights practical hands-on usage in project work."
        return f"I have about {years_value} years of experience working with {tech_label}. {evidence}"[:420]

    prefix_by_intent = {
        "skills": "Based on my profile,",
        "experience": "Based on my experience,",
        "projects": "Based on my projects,",
        "education": "From my education background,",
        "contact": "You can reach me via the details in my profile,",
        "general": "Here is what matches your question,",
    }
    prefix = prefix_by_intent.get(intent, prefix_by_intent["general"])
    if not concise:
        concise = str(top.get("snippet", ""))[:260]
    return f"{prefix} {concise}".strip()[:420]


def _parse_hf_text(payload: object) -> str:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            val = first.get("generated_text") or first.get("summary_text")
            return str(val).strip() if val else ""
    if isinstance(payload, dict):
        val = payload.get("generated_text") or payload.get("summary_text")
        return str(val).strip() if val else ""
    return ""


def _extract_chat_completion_text(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    # Some providers return an empty `content` while filling `reasoning_content`.
    # We intentionally prefer user-facing content and only fall back if needed.
    content = str(message.get("content", "")).strip()
    if content:
        return content
    reasoning = str(message.get("reasoning_content", "")).strip()
    return reasoning


def _generate_with_hf(query: str, results: List[dict]) -> str:
    hf_api_url = os.getenv(
        "HF_CHAT_API_URL", "https://router.huggingface.co/v1/chat/completions"
    ).strip()
    hf_api_token = os.getenv("HF_API_TOKEN", "").strip()
    hf_model = os.getenv("HF_MODEL", "").strip()
    hf_model_fallbacks = [
        model.strip()
        for model in os.getenv(
            "HF_MODEL_FALLBACKS", "meta-llama/Llama-3.1-8B-Instruct:novita"
        ).split(",")
        if model.strip()
    ]
    model_candidates = [model for model in [hf_model, *hf_model_fallbacks] if model]
    if not hf_api_url or not model_candidates:
        return ""

    query_tokens = tokenize(query)
    asks_years = "year" in query_tokens or "years" in query_tokens
    skill_entity = detect_skill_entity(query)
    signals = analyze_query(query)
    combined = " ".join(str(item.get("snippet", "")) for item in results[:2])
    years_value = _extract_years_value(combined)
    has_skill_evidence = bool(skill_entity and skill_entity.lower() in combined.lower())
    concise_hint = _build_fallback_answer(query, results)

    # Provide short, highly relevant evidence lines instead of full chunks.
    evidence_lines: List[str] = []
    for item in results[:3]:
        best = extract_best_sentences(str(item.get("snippet", "")), signals, 2)
        evidence_lines.extend(best)
    if not evidence_lines:
        evidence_lines = [str(item.get("snippet", ""))[:220] for item in results[:2]]
    context = "\n".join(f"- {line.strip()}" for line in evidence_lines[:6] if line.strip())

    rules: List[str] = [
        "You are an assistant for Rohit's portfolio chatbot.",
        "Answer in first person as Rohit and keep the answer to 1-2 concise sentences.",
        "Default to at most 2 short sentences unless the user explicitly asks for more detail.",
        "Use only the provided evidence. Do not invent facts.",
        "If the evidence clearly supports an answer, do not reply with uncertainty.",
    ]
    if asks_years and skill_entity and years_value and has_skill_evidence:
        tech_label = "AWS" if skill_entity.lower() == "aws" else skill_entity.upper()
        rules.append(
            f"If asked about years with {tech_label}, use the evidence and answer as approximately {years_value} years."
        )
    prompt = "\n".join(
        [
            "\n".join(rules),
            f"Question: {query}",
            "Evidence:",
            context,
            f"Hint: {concise_hint}",
            "Final answer:",
        ]
    )
    headers = {"Content-Type": "application/json"}
    if hf_api_token:
        headers["Authorization"] = f"Bearer {hf_api_token}"

    for model in model_candidates:
        payload = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 120,
            }
        ).encode("utf-8")
        req = request.Request(hf_api_url, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = _extract_chat_completion_text(body) or _parse_hf_text(body)
            cleaned = text.replace("Answer:", "").strip()
            if cleaned:
                return " ".join(cleaned.split())[:420]
        except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            continue
    return ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/logging", response_model=LoggingHealthResponse)
def logging_health() -> LoggingHealthResponse:
    diagnostics = get_store_diagnostics()
    return LoggingHealthResponse(**diagnostics)


@app.post("/corpora/load", response_model=CorpusLoadResponse)
def load_corpus(request_body: CorpusLoadRequest) -> CorpusLoadResponse:
    return _load_corpus_artifacts(
        corpus_id=request_body.corpus_id,
        chunks_url=request_body.chunks_url,
        vector_index_url=request_body.vector_index_url,
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    _maybe_load_corpus_from_request(
        corpus_id=request.corpus_id,
        chunks_url=request.chunks_url,
        vector_index_url=request.vector_index_url,
    )
    _maybe_autoload_default_corpus(request.corpus_id)
    if not service.corpus_exists(request.corpus_id):
        raise HTTPException(status_code=404, detail=f"Corpus not found: {request.corpus_id}")

    rm = _resolve_retrieval_model(request.retrieval_model)
    results = service.search(
        SearchParams(
            query=request.query,
            corpus_id=request.corpus_id,
            doc_ids=request.doc_ids,
            top_k=request.top_k,
            min_score=request.min_score,
            retrieval_model=rm,
        )
    )
    return SearchResponse(
        query=request.query,
        corpus_id=request.corpus_id,
        total_results=len(results),
        results=[SearchResult(**item) for item in results],
        retrieval_model=rm.value,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request_body: ChatRequest) -> ChatResponse:
    _maybe_load_corpus_from_request(
        corpus_id=request_body.corpus_id,
        chunks_url=request_body.chunks_url,
        vector_index_url=request_body.vector_index_url,
    )
    _maybe_autoload_default_corpus(request_body.corpus_id)
    allow_fallback = _resolve_allow_fallback(request_body)
    method_requested = (
        request_body.answer_method or ChatAnswerMethod.hugging_face_lightweight_nlp
    )
    rm = _resolve_retrieval_model(request_body.retrieval_model)
    if not service.corpus_exists(request_body.corpus_id):
        raise HTTPException(
            status_code=404, detail=f"Corpus not found: {request_body.corpus_id}"
        )
    event_id = ChatLogStore.new_event_id()
    session_id = _resolve_session_id(request_body.session_id)
    started = time.perf_counter()
    results = service.search(
        SearchParams(
            query=request_body.query,
            corpus_id=request_body.corpus_id,
            doc_ids=request_body.doc_ids,
            top_k=request_body.top_k,
            min_score=request_body.min_score,
            retrieval_model=rm,
        )
    )
    if not results:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        no_results_answer = (
            NO_ANSWER_MESSAGE
            if not allow_fallback
            else "I could not find a grounded answer in the current portfolio documents."
        )
        _log_chat_event_safe(
            ChatEventRecord(
                event_id=event_id,
                session_id=session_id,
                corpus_id=request_body.corpus_id,
                query=request_body.query,
                answer=no_results_answer,
                source="",
                method="none",
                retrieval_model=rm.value,
                used_hf=False,
                top_k=request_body.top_k,
                min_score=request_body.min_score,
                allow_fallback=allow_fallback,
                latency_ms=elapsed_ms,
                category="chat",
                bucket="no_results",
                info=_build_event_info(request_body, []),
            )
        )
        return ChatResponse(
            query=request_body.query,
            corpus_id=request_body.corpus_id,
            answer=no_results_answer,
            source="",
            used_hf=False,
            method="none",
            retrieval_model=rm.value,
            event_id=event_id,
            session_id=session_id,
        )

    answer, source, used_hf, method = _chat_answer_from_results(
        request_body.query,
        results,
        method_requested,
        allow_fallback,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    _log_chat_event_safe(
        ChatEventRecord(
            event_id=event_id,
            session_id=session_id,
            corpus_id=request_body.corpus_id,
            query=request_body.query,
            answer=answer,
            source=source,
            method=method,
            retrieval_model=rm.value,
            used_hf=used_hf,
            top_k=request_body.top_k,
            min_score=request_body.min_score,
            allow_fallback=allow_fallback,
            latency_ms=elapsed_ms,
            category="chat",
            bucket=_bucket_for_method(method, len(results)),
            info=_build_event_info(request_body, results),
        )
    )
    return ChatResponse(
        query=request_body.query,
        corpus_id=request_body.corpus_id,
        answer=answer,
        source=source,
        used_hf=used_hf,
        method=method,
        retrieval_model=rm.value,
        event_id=event_id,
        session_id=session_id,
    )


@app.post("/chat/feedback", response_model=ChatFeedbackResponse)
def chat_feedback(request_body: ChatFeedbackRequest) -> ChatFeedbackResponse:
    store = get_store()
    if store is None:
        return ChatFeedbackResponse(
            event_id=request_body.event_id,
            accepted=False,
            rating=request_body.rating,
            bucket="disabled",
        )

    if not store.event_exists(request_body.event_id):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown event_id: {request_body.event_id}",
        )

    bucket = "positive" if request_body.rating > 0 else "negative" if request_body.rating < 0 else "neutral"
    record = ChatFeedbackRecord(
        event_id=request_body.event_id,
        session_id=_resolve_session_id(request_body.session_id),
        rating=request_body.rating,
        comment=(request_body.comment or "").strip(),
        category="feedback",
        bucket=bucket,
        info={},
    )
    try:
        store.insert_feedback(record)
    except Exception:  # pragma: no cover
        LOGGER.exception("Failed to persist chat feedback for %s", request_body.event_id)
        raise HTTPException(status_code=500, detail="Could not persist feedback.") from None
    return ChatFeedbackResponse(
        event_id=request_body.event_id,
        accepted=True,
        rating=request_body.rating,
        bucket=bucket,
    )


@app.get("/corpora", response_model=List[CorpusStatsResponse])
def list_corpora() -> List[CorpusStatsResponse]:
    return [
        CorpusStatsResponse(
            corpus_id=stat.corpus_id,
            total_chunks=stat.total_chunks,
            total_docs=stat.total_docs,
        )
        for stat in service.list_corpora()
    ]


@app.get("/corpora/{corpus_id}", response_model=CorpusStatsResponse)
def get_corpus(corpus_id: str) -> CorpusStatsResponse:
    try:
        stat = service.get_corpus_stats(corpus_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CorpusStatsResponse(
        corpus_id=stat.corpus_id,
        total_chunks=stat.total_chunks,
        total_docs=stat.total_docs,
    )


@app.get("/corpora/{corpus_id}/exists", response_model=CorpusExistsResponse)
def corpus_exists(corpus_id: str) -> CorpusExistsResponse:
    return CorpusExistsResponse(corpus_id=corpus_id, exists=service.corpus_exists(corpus_id))

