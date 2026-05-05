from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class CorpusStat:
    corpus_id: str
    total_chunks: int
    total_docs: int


class IndexStorage:
    def __init__(self, index_root: Path) -> None:
        self._index_root = index_root

    def corpus_path(self, corpus_id: str) -> Path:
        return self._index_root / corpus_id / "chunks.json"

    def vector_index_path(self, corpus_id: str) -> Path:
        return self._index_root / corpus_id / "vector_index.json"

    def exists(self, corpus_id: str) -> bool:
        return self.corpus_path(corpus_id).exists()

    def load_chunks(self, corpus_id: str) -> List[dict]:
        path = self.corpus_path(corpus_id)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def vector_index_exists(self, corpus_id: str) -> bool:
        return self.vector_index_path(corpus_id).exists()

    def list_corpora(self) -> List[CorpusStat]:
        if not self._index_root.exists():
            return []

        stats: List[CorpusStat] = []
        for corpus_dir in sorted([p for p in self._index_root.iterdir() if p.is_dir()]):
            chunks_path = corpus_dir / "chunks.json"
            if not chunks_path.exists():
                continue
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            doc_ids = {chunk.get("doc_id", "") for chunk in chunks if chunk.get("doc_id")}
            stats.append(
                CorpusStat(
                    corpus_id=corpus_dir.name,
                    total_chunks=len(chunks),
                    total_docs=len(doc_ids),
                )
            )
        return stats

