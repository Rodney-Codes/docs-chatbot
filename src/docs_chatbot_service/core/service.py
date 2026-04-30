from __future__ import annotations

from pathlib import Path

from docs_chatbot_service.core.search import BM25Searcher
from docs_chatbot_service.core.storage import IndexStorage
from docs_chatbot_service.core.text import make_snippet
from docs_chatbot_service.schemas.contracts import SearchRequest, SearchResponse, SearchResult


class RetrievalService:
    def __init__(self, index_root: Path) -> None:
        self.storage = IndexStorage(index_root=index_root)
        self._cache: dict[str, BM25Searcher] = {}

    def _searcher_for_corpus(self, corpus_id: str) -> BM25Searcher:
        if corpus_id not in self._cache:
            payload = self.storage.load_index(corpus_id=corpus_id)
            self._cache[corpus_id] = BM25Searcher(index_payload=payload)
        return self._cache[corpus_id]

    def query(self, request: SearchRequest) -> SearchResponse:
        searcher = self._searcher_for_corpus(request.corpus_id)
        filtered_doc_ids = set(request.doc_ids) if request.doc_ids else None

        hits = searcher.search(
            query=request.query,
            top_k=request.top_k,
            doc_ids=filtered_doc_ids,
            min_score=request.min_score,
        )

        results = [
            SearchResult(
                chunk_id=hit.chunk.chunk_id,
                doc_id=hit.chunk.doc_id,
                title=hit.chunk.title,
                section=hit.chunk.section,
                source=hit.chunk.source,
                snippet=make_snippet(hit.chunk.text),
                score=round(hit.score, 6),
            )
            for hit in hits
        ]
        return SearchResponse(
            query=request.query,
            corpus_id=request.corpus_id,
            total_results=len(results),
            results=results,
        )
