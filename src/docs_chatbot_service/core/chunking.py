from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_words: int = 450
    overlap_words: int = 70

    def __post_init__(self) -> None:
        if self.chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be greater than zero.")
        if self.overlap_words < 0:
            raise ValueError("overlap_words cannot be negative.")
        if self.overlap_words >= self.chunk_size_words:
            raise ValueError("overlap_words must be smaller than chunk_size_words.")


def split_text_into_word_chunks(text: str, config: ChunkingConfig) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = config.chunk_size_words - config.overlap_words
    for start in range(0, len(words), step):
        end = start + config.chunk_size_words
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks
