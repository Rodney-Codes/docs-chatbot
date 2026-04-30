from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from docs_chatbot_service.core.service import RetrievalService
from docs_chatbot_service.core.storage import IndexStorage
from docs_chatbot_service.schemas.contracts import (
    CorpusExistsResponse,
    CorpusSummary,
    SearchRequest,
    SearchResponse,
)


def create_app(index_root: Path | None = None) -> FastAPI:
    app = FastAPI(title="Docs Chatbot Retrieval Service", version="0.1.0")
    resolved_index_root = index_root or Path("data/index")
    service = RetrievalService(index_root=resolved_index_root)
    storage = IndexStorage(index_root=resolved_index_root)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        try:
            return service.query(request=request)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/corpora", response_model=list[CorpusSummary])
    def list_corpora() -> list[CorpusSummary]:
        summaries: list[CorpusSummary] = []
        for corpus_id in storage.list_corpora():
            stats = storage.corpus_stats(corpus_id=corpus_id)
            summaries.append(CorpusSummary(**stats))
        return summaries

    @app.get("/corpora/{corpus_id}", response_model=CorpusSummary)
    def corpus_stats(corpus_id: str) -> CorpusSummary:
        if not storage.corpus_exists(corpus_id=corpus_id):
            raise HTTPException(status_code=404, detail=f"Corpus '{corpus_id}' not found.")
        stats = storage.corpus_stats(corpus_id=corpus_id)
        return CorpusSummary(**stats)

    @app.get("/corpora/{corpus_id}/exists", response_model=CorpusExistsResponse)
    def corpus_exists(corpus_id: str) -> CorpusExistsResponse:
        return CorpusExistsResponse(corpus_id=corpus_id, exists=storage.corpus_exists(corpus_id))

    return app
