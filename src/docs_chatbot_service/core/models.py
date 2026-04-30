from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    source: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
