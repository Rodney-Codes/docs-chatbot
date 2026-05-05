from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from docs_chatbot_service.core.rule_vector_retrieval import RuleVectorIndex
from docs_chatbot_service.core.search import BM25SearchEngine
from docs_chatbot_service.core.storage import CorpusStat, IndexStorage
from docs_chatbot_service.core.vector_search import HashedVectorIndex, build_hybrid_score


class RetrievalModel(str, Enum):
    """Composable retrieval pipelines built from scoring functions."""

    bm25 = "bm25"
    hashed_vector = "hashed_vector"
    bm25_hashed_vector = "bm25_hashed_vector"
    rule_lexicon_tfidf = "rule_lexicon_tfidf"


@dataclass(frozen=True)
class SearchParams:
    query: str
    corpus_id: str
    doc_ids: Optional[List[str]]
    top_k: int
    min_score: float
    retrieval_model: RetrievalModel = RetrievalModel.bm25_hashed_vector


class RetrievalService:
    def __init__(self, index_root: Path) -> None:
        self._storage = IndexStorage(index_root=index_root)

    def corpus_exists(self, corpus_id: str) -> bool:
        return self._storage.exists(corpus_id)

    def list_corpora(self) -> List[CorpusStat]:
        return self._storage.list_corpora()

    def get_corpus_stats(self, corpus_id: str) -> CorpusStat:
        for stat in self._storage.list_corpora():
            if stat.corpus_id == corpus_id:
                return stat
        raise FileNotFoundError(f"Corpus not found: {corpus_id}")

    @lru_cache(maxsize=32)
    def _load_corpus(
        self, corpus_id: str
    ) -> tuple[List[dict], BM25SearchEngine, Optional[HashedVectorIndex]]:
        chunks = self._storage.load_chunks(corpus_id)
        engine = BM25SearchEngine(chunks)
        vector_index: Optional[HashedVectorIndex] = None
        if self._storage.vector_index_exists(corpus_id):
            vector_index = HashedVectorIndex.load(self._storage.vector_index_path(corpus_id))
        return chunks, engine, vector_index

    @lru_cache(maxsize=32)
    def _rule_vector_index_for(self, corpus_id: str) -> RuleVectorIndex:
        chunks = self._storage.load_chunks(corpus_id)
        return RuleVectorIndex(chunks)

    def search(self, params: SearchParams) -> List[dict]:
        chunks, engine, vector_index = self._load_corpus(params.corpus_id)
        allowed_docs = set(params.doc_ids or [])
        has_filter = bool(params.doc_ids)
        use_rule_vector = params.retrieval_model == RetrievalModel.rule_lexicon_tfidf
        use_bm25 = params.retrieval_model in {
            RetrievalModel.bm25,
            RetrievalModel.bm25_hashed_vector,
        }
        use_hashed_vector = params.retrieval_model in {
            RetrievalModel.hashed_vector,
            RetrievalModel.bm25_hashed_vector,
        }
        rule_index = self._rule_vector_index_for(params.corpus_id) if use_rule_vector else None

        scored_results: List[dict] = []
        for chunk in chunks:
            if has_filter and chunk.get("doc_id") not in allowed_docs:
                continue
            if use_rule_vector:
                score = rule_index.score(params.query, chunk)
            else:
                bm25_score = engine.score(params.query, chunk) if use_bm25 else 0.0
                vector_score = (
                    vector_index.score(params.query, chunk["chunk_id"])
                    if (use_hashed_vector and vector_index)
                    else 0.0
                )
                if use_bm25 and use_hashed_vector:
                    score = (
                        build_hybrid_score(bm25_score=bm25_score, vector_score=vector_score)
                        if vector_index
                        else bm25_score
                    )
                elif use_bm25:
                    score = bm25_score
                else:
                    score = vector_score
            if score < params.min_score:
                continue

            scored_results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "title": chunk.get("title", ""),
                    "section": chunk.get("section", "general"),
                    "source": chunk.get("source", ""),
                    "snippet": chunk.get("text", ""),
                    "score": float(score),
                }
            )

        scored_results.sort(key=lambda item: item["score"], reverse=True)
        return scored_results[: params.top_k]

