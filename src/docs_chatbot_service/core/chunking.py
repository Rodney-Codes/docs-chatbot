from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size_words: int = 450, overlap_words: int = 70) -> List[str]:
    words = text.split()
    if not words:
        return []

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be > 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be < chunk_size_words")

    chunks: List[str] = []
    step = chunk_size_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size_words >= len(words):
            break
    return chunks

