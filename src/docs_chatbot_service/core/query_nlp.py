"""
Lightweight, pre-LLM style query understanding: clause segmentation, stopword-aware
tokenization, synonym expansion, and bigram-style phrase signals for retrieval and
answer extraction. No external models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Tuple

from docs_chatbot_service.core.text_util import TOKEN_RE, tokenize as search_tokenize

STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "at",
        "from",
        "or",
        "and",
        "but",
        "if",
        "then",
        "so",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "my",
        "your",
        "yours",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "how",
        "why",
        "when",
        "where",
        "there",
        "here",
        "about",
        "into",
        "any",
        "some",
        "can",
        "could",
        "would",
        "should",
        "tell",
        "please",
        "just",
        "like",
        "want",
        "know",
        "did",
        "does",
        "get",
        "got",
        "give",
        "gave",
    }
)

# Lexical expansions (classic query expansion). Values are weakly weighted at retrieval time.
SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "cv": ("resume", "profile", "summary"),
    "resume": ("cv", "profile", "summary"),
    "profile": ("resume", "summary"),
    "stack": ("skills", "tools", "technologies", "tech"),
    "tech": ("technologies", "tools", "stack"),
    "tooling": ("tools", "stack"),
    "langs": ("languages", "programming"),
    "languages": ("python", "sql", "programming"),
    "etl": ("pipeline", "pipelines", "data"),
    "elt": ("pipeline", "data"),
    "viz": ("visualization", "dashboard", "tableau"),
    "visualization": ("dashboard", "tableau"),
    "ml": ("machine", "learning", "model", "models"),
    "ai": ("machine", "learning", "model"),
    "database": ("databases", "sql"),
    "databases": ("sql", "mongodb", "postgresql", "mysql"),
    "degree": ("education", "college", "university"),
    "school": ("university", "college", "education"),
    "role": ("position", "job", "work"),
    "position": ("role", "job"),
    "hire": ("contact", "reach", "email"),
    "hiring": ("contact", "work"),
    "reach": ("contact", "email"),
    "repo": ("github", "project", "projects"),
    "repos": ("github", "project"),
    "code": ("github", "python", "programming"),
    "background": ("experience", "career", "summary"),
    "worked": ("experience", "work"),
    "working": ("experience", "work"),
    "analytics": ("analysis", "data", "tableau"),
    "analyst": ("analysis", "data", "sql"),
    "engineer": ("engineering", "data", "software"),
    "engineering": ("engineer", "data", "systems"),
}

# Multi-word phrases in the raw query map to ordered bigrams for evidence matching.
PHRASE_BIGRAM_SOURCES: Tuple[str, ...] = (
    "power bi",
    "machine learning",
    "data science",
    "data engineering",
    "business intelligence",
    "deep learning",
    "large language",
)

# Multi-token skills / entities (longest first for substring detection).
SKILL_PHRASES: Tuple[str, ...] = (
    "power bi",
    "machine learning",
    "data engineering",
    "data science",
    "postgresql",
    "postgres",
    "mongodb",
    "mysql",
    "tableau",
    "fastapi",
    "kubernetes",
    "terraform",
    "snowflake",
    "databricks",
    "bigquery",
    "pytorch",
    "tensorflow",
    "javascript",
    "typescript",
    "react",
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "aws",
    "gcp",
    "azure",
    "sql",
    "python",
)


def segment_query(raw: str) -> List[str]:
    """Split compound questions on punctuation, newlines, and light conjunction boundaries."""
    text = (raw or "").strip()
    if not text:
        return []
    pieces: List[str] = []
    for block in re.split(r"[\n;|]+", text):
        block = block.strip()
        if not block:
            continue
        for part in re.split(r"\s+(?:and|&)\s+", block, flags=re.IGNORECASE):
            p = part.strip()
            if p:
                pieces.append(p)
    if not pieces:
        return [text]
    return pieces


def _filtered_scoring_tokens(raw: str) -> List[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(raw)]
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def weighted_query_terms(query: str) -> List[Tuple[str, float]]:
    """
    Terms with weights for BM25-style retrieval: primary tokens plus synonym expansions.
    """
    collected: Dict[str, float] = {}
    for segment in segment_query(query) or [query]:
        for tok in _filtered_scoring_tokens(segment):
            collected[tok] = max(collected.get(tok, 0.0), 1.0)
            for syn in SYNONYMS.get(tok, ()):
                if syn in STOPWORDS or len(syn) <= 1:
                    continue
                collected[syn] = max(collected.get(syn, 0.0), 0.38)
    if not collected:
        for tok in search_tokenize(query):
            if len(tok) > 0:
                collected[tok] = max(collected.get(tok, 0.0), 0.55)
    return sorted(collected.items(), key=lambda item: (-item[1], item[0]))


def _bigrams_from_tokens(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def phrase_anchor_bigrams(query: str) -> FrozenSet[Tuple[str, str]]:
    q = (query or "").lower()
    found: set[Tuple[str, str]] = set()
    for phrase in PHRASE_BIGRAM_SOURCES:
        if phrase in q:
            parts = phrase.split()
            if len(parts) == 2:
                found.add((parts[0], parts[1]))
    return frozenset(found)


@dataclass(frozen=True)
class QuerySignals:
    segments: Tuple[str, ...]
    weighted_terms: Tuple[Tuple[str, float], ...]
    bigrams: FrozenSet[Tuple[str, str]]
    intent_tokens: FrozenSet[str]


def analyze_query(query: str) -> QuerySignals:
    segments = tuple(segment_query(query) or ([query.strip()] if query.strip() else ()))
    weighted = tuple(weighted_query_terms(query))

    stream: List[str] = []
    for seg in segments:
        stream.extend(_filtered_scoring_tokens(seg))

    bigram_set = set(_bigrams_from_tokens(stream))
    bigram_set.update(phrase_anchor_bigrams(query))

    base_intent = {t.lower() for t in TOKEN_RE.findall(query)}
    for tok in list(base_intent):
        if tok in STOPWORDS:
            continue
        for syn in SYNONYMS.get(tok, ()):
            base_intent.add(syn)

    return QuerySignals(
        segments=segments,
        weighted_terms=weighted,
        bigrams=frozenset(bigram_set),
        intent_tokens=frozenset(base_intent),
    )


def detect_intent_from_signals(signals: QuerySignals) -> str:
    token_set = signals.intent_tokens
    if token_set.intersection({"sql", "python", "skills", "skill", "tools", "stack", "technologies", "tech"}):
        return "skills"
    if token_set.intersection(
        {"experience", "years", "worked", "background", "career", "work", "role", "position", "job"}
    ):
        return "experience"
    if token_set.intersection({"project", "projects", "built", "build", "repo", "repos"}):
        return "projects"
    if token_set.intersection({"education", "degree", "college", "university", "school"}):
        return "education"
    if token_set.intersection({"contact", "email", "linkedin", "github", "reach", "hire", "hiring"}):
        return "contact"
    return "general"


def detect_skill_entity(query: str) -> str:
    q = (query or "").lower()
    for skill in SKILL_PHRASES:
        if skill in q:
            return skill
    return ""


def lexical_match_score(text: str, signals: QuerySignals) -> float:
    """Bag-of-words plus ordered bigram overlap (lightweight semantic proxy)."""
    tokens = search_tokenize(text)
    if not tokens:
        return 0.0
    from collections import Counter

    counts = Counter(tokens)
    score = 0.0
    for term, weight in signals.weighted_terms:
        score += weight * float(counts.get(term, 0))
    for idx in range(len(tokens) - 1):
        pair = (tokens[idx], tokens[idx + 1])
        if pair in signals.bigrams:
            score += 2.35
    return score


def extract_best_sentences(text: str, signals: QuerySignals, max_sentences: int = 2) -> List[str]:
    clean = " ".join(text.split()).strip()
    if not clean:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean.replace("\n", " ")) if s.strip()]
    if not sentences:
        return []
    scored: List[Tuple[float, str]] = []
    for sentence in sentences:
        scored.append((lexical_match_score(sentence, signals), sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    picked = [s for value, s in scored if value > 0][:max_sentences]
    if picked:
        return picked
    return [s for _, s in scored[:max_sentences]]
