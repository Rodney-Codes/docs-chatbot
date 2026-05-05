from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from docs_chatbot_service.core.query_nlp import weighted_query_terms
from docs_chatbot_service.core.text_util import tokenize as split_tokens


def _char_ngrams(token: str, n: int = 3) -> List[str]:
    if len(token) <= n:
        return [token]
    return [token[idx : idx + n] for idx in range(0, len(token) - n + 1)]


def _stable_hash(term: str, dim: int) -> int:
    digest = hashlib.sha1(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dim


class HashedVectorIndex:
    def __init__(self, dim: int, idf: Dict[int, float], vectors: Dict[str, List[float]]) -> None:
        self.dim = dim
        self.idf = idf
        self.vectors = vectors

    @classmethod
    def from_chunks(cls, chunks: List[dict], dim: int = 512) -> "HashedVectorIndex":
        doc_term_freqs: Dict[str, Dict[int, float]] = {}
        doc_freq: Dict[int, int] = {}

        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", ""))
            text = str(chunk.get("text", ""))
            term_freq: Dict[int, float] = {}
            for token in split_tokens(text):
                for ngram in _char_ngrams(token, 3):
                    idx = _stable_hash(ngram, dim)
                    term_freq[idx] = term_freq.get(idx, 0.0) + 1.0
            if not chunk_id:
                continue
            doc_term_freqs[chunk_id] = term_freq
            for idx in term_freq:
                doc_freq[idx] = doc_freq.get(idx, 0) + 1

        total_docs = max(len(doc_term_freqs), 1)
        idf: Dict[int, float] = {}
        for idx, df in doc_freq.items():
            idf[idx] = math.log(1.0 + (total_docs / (1.0 + df)))

        vectors: Dict[str, List[float]] = {}
        for chunk_id, term_freq in doc_term_freqs.items():
            dense = [0.0] * dim
            for idx, tf in term_freq.items():
                dense[idx] += tf * idf.get(idx, 0.0)
            norm = math.sqrt(sum(value * value for value in dense))
            if norm > 0:
                dense = [value / norm for value in dense]
            vectors[chunk_id] = dense
        return cls(dim=dim, idf=idf, vectors=vectors)

    def _query_vector(self, query: str) -> List[float]:
        tf: Dict[int, float] = {}
        for token, weight in weighted_query_terms(query):
            for ngram in _char_ngrams(token, 3):
                idx = _stable_hash(ngram, self.dim)
                tf[idx] = tf.get(idx, 0.0) + weight
        dense = [0.0] * self.dim
        for idx, count in tf.items():
            dense[idx] += count * self.idf.get(idx, 0.0)
        norm = math.sqrt(sum(value * value for value in dense))
        if norm > 0:
            dense = [value / norm for value in dense]
        return dense

    def score(self, query: str, chunk_id: str) -> float:
        doc_vector = self.vectors.get(chunk_id)
        if not doc_vector:
            return 0.0
        qv = self._query_vector(query)
        return float(sum(a * b for a, b in zip(qv, doc_vector)))

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "idf": {str(k): v for k, v in self.idf.items()},
            "vectors": self.vectors,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "HashedVectorIndex":
        dim = int(payload.get("dim", 512))
        idf = {int(k): float(v) for k, v in dict(payload.get("idf", {})).items()}
        vectors = {
            str(chunk_id): [float(value) for value in values]
            for chunk_id, values in dict(payload.get("vectors", {})).items()
        }
        return cls(dim=dim, idf=idf, vectors=vectors)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HashedVectorIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def build_hybrid_score(bm25_score: float, vector_score: float, alpha: float = 0.6) -> float:
    # BM25 can exceed 1 while cosine is in [-1, 1]. Compress BM25 to [0, 1) first.
    bm25_scaled = 1.0 - math.exp(-max(bm25_score, 0.0))
    vector_scaled = max(vector_score, 0.0)
    return (alpha * vector_scaled) + ((1.0 - alpha) * bm25_scaled)

