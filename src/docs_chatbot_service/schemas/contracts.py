from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, description="End-user question or search input.")
    corpus_id: str = Field(
        default="default",
        description="Logical corpus identifier. Maps to one index artifact set.",
    )
    doc_ids: list[str] | None = Field(
        default=None,
        description="Optional allow-list to limit retrieval to specific documents.",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum score threshold for a result to be returned.",
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
    results: list[SearchResult]


class CorpusSummary(BaseModel):
    corpus_id: str
    total_chunks: int
    total_docs: int


class CorpusExistsResponse(BaseModel):
    corpus_id: str
    exists: bool
