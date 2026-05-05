from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from docs_chatbot_service.core.query_nlp import weighted_query_terms
from docs_chatbot_service.core.text_util import tokenize


class BM25SearchEngine:
    def __init__(self, chunks: List[dict]) -> None:
        self._chunks = chunks
        self._doc_tokens: Dict[str, Counter] = {}
        self._doc_len: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._inv_index: Dict[str, List[str]] = defaultdict(list)
        self._avg_doc_len = 0.0
        self._build()

    def _build(self) -> None:
        if not self._chunks:
            self._avg_doc_len = 0.0
            return

        for chunk in self._chunks:
            chunk_id = chunk["chunk_id"]
            token_counts = Counter(tokenize(chunk.get("text", "")))
            self._doc_tokens[chunk_id] = token_counts
            self._doc_len[chunk_id] = sum(token_counts.values())
            for term in token_counts:
                self._inv_index[term].append(chunk_id)

        self._avg_doc_len = sum(self._doc_len.values()) / len(self._doc_len)
        total_docs = len(self._doc_len)
        for term, postings in self._inv_index.items():
            df = len(postings)
            self._idf[term] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, chunk: dict) -> float:
        k1 = 1.5
        b = 0.75
        weighted_terms: List[Tuple[str, float]] = weighted_query_terms(query)
        if not weighted_terms:
            return 0.0

        chunk_id = chunk["chunk_id"]
        token_counts = self._doc_tokens.get(chunk_id, Counter())
        doc_len = self._doc_len.get(chunk_id, 0)
        if doc_len == 0:
            return 0.0

        denom_const = k1 * (1 - b + b * (doc_len / max(self._avg_doc_len, 1e-9)))
        score = 0.0
        for term, weight in weighted_terms:
            tf = token_counts.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            score += weight * (idf * ((tf * (k1 + 1)) / (tf + denom_const)))
        return float(score)

