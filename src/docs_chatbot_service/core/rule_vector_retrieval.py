"""
Rule-defined lexicon TF-IDF vectors and cosine similarity (no learned embeddings).

Dimensions are a fixed vocabulary built from query NLP rules (synonyms, skills,
phrase tokens, intent cues). Heavier than BM25 alone: per-corpus doc vectors plus
per-query vector and a dense dot product per chunk.
"""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple

from docs_chatbot_service.core.query_nlp import (
    PHRASE_BIGRAM_SOURCES,
    SKILL_PHRASES,
    STOPWORDS,
    SYNONYMS,
    weighted_query_terms,
)
from docs_chatbot_service.core.text_util import tokenize as search_tokenize

INTENT_LEXICON: Tuple[str, ...] = (
    "skills",
    "skill",
    "experience",
    "years",
    "project",
    "projects",
    "education",
    "degree",
    "university",
    "college",
    "contact",
    "email",
    "github",
    "linkedin",
    "career",
    "background",
    "work",
    "role",
    "job",
    "built",
    "build",
)


def _build_lexicon() -> Tuple[str, ...]:
    tokens: set[str] = set()
    for key, syns in SYNONYMS.items():
        if len(key) > 1 and key not in STOPWORDS:
            tokens.add(key)
        for s in syns:
            if len(s) > 1 and s not in STOPWORDS:
                tokens.add(s)
    for phrase in SKILL_PHRASES:
        for t in phrase.replace("-", " ").lower().split():
            t = t.strip()
            if len(t) > 1 and t not in STOPWORDS:
                tokens.add(t)
    for phrase in PHRASE_BIGRAM_SOURCES:
        for t in phrase.lower().split():
            if len(t) > 1 and t not in STOPWORDS:
                tokens.add(t)
    for t in INTENT_LEXICON:
        tokens.add(t)
    return tuple(sorted(tokens))


@lru_cache(maxsize=1)
def get_rule_lexicon() -> Tuple[str, ...]:
    return _build_lexicon()


class RuleVectorIndex:
    """TF-IDF over a fixed rule lexicon; L2-normalized vectors; cosine = dot product."""

    def __init__(self, chunks: List[dict]) -> None:
        self._lexicon = get_rule_lexicon()
        n_docs = max(len(chunks), 1)
        df: Counter[str] = Counter()
        for chunk in chunks:
            present = set(search_tokenize(str(chunk.get("text", ""))))
            for term in self._lexicon:
                if term in present:
                    df[term] += 1
        self._idf = {t: math.log(1.0 + (n_docs / (0.5 + float(df.get(t, 0))))) for t in self._lexicon}
        self._vecs: Dict[str, List[float]] = {}
        for chunk in chunks:
            cid = str(chunk.get("chunk_id", ""))
            if not cid:
                continue
            counts = Counter(search_tokenize(str(chunk.get("text", ""))))
            raw = [counts.get(t, 0) * self._idf[t] for t in self._lexicon]
            norm = math.sqrt(sum(v * v for v in raw)) or 1.0
            self._vecs[cid] = [v / norm for v in raw]

    def score(self, query: str, chunk: dict) -> float:
        qv = self._query_vector(query)
        cid = str(chunk.get("chunk_id", ""))
        dv = self._vecs.get(cid)
        if not dv:
            return 0.0
        return float(sum(a * b for a, b in zip(qv, dv)))

    def _query_vector(self, query: str) -> List[float]:
        wq = dict(weighted_query_terms(query))
        raw = [wq.get(t, 0.0) for t in self._lexicon]
        qtoks = set(search_tokenize(query))
        for i, t in enumerate(self._lexicon):
            if t in qtoks:
                raw[i] = max(raw[i], 1.0)
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]
