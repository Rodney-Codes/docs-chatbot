from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from docs_chatbot_service.core.text import tokenize


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def build_index_from_chunks(chunks_path: Path, corpus_id: str) -> dict:
    chunks = _read_json(chunks_path)
    if not isinstance(chunks, list):
        raise ValueError("Chunk file must contain a JSON array.")

    postings: dict[str, list[dict]] = defaultdict(list)
    document_frequency: dict[str, int] = {}
    chunk_lengths: dict[str, int] = {}
    chunk_lookup: dict[str, dict] = {}

    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        text = str(chunk["text"])
        tokens = tokenize(text)
        term_counts = Counter(tokens)

        chunk_lookup[chunk_id] = {
            "chunk_id": chunk_id,
            "doc_id": str(chunk["doc_id"]),
            "title": str(chunk.get("title", "")),
            "section": str(chunk.get("section", "")),
            "source": str(chunk.get("source", "")),
            "text": text,
        }
        chunk_lengths[chunk_id] = len(tokens)

        for term, frequency in term_counts.items():
            postings[term].append({"chunk_id": chunk_id, "tf": frequency})

    total_chunks = len(chunk_lookup)
    for term, entries in postings.items():
        document_frequency[term] = len(entries)

    avg_chunk_length = (
        sum(chunk_lengths.values()) / total_chunks
        if total_chunks > 0
        else 0.0
    )

    idf: dict[str, float] = {}
    for term, df in document_frequency.items():
        idf[term] = math.log(((total_chunks - df + 0.5) / (df + 0.5)) + 1.0)

    return {
        "corpus_id": corpus_id,
        "bm25": {"k1": 1.2, "b": 0.75},
        "stats": {
            "total_chunks": total_chunks,
            "avg_chunk_length": avg_chunk_length,
        },
        "chunk_lengths": chunk_lengths,
        "idf": idf,
        "postings": postings,
        "chunks": chunk_lookup,
    }
