from __future__ import annotations

import json
from pathlib import Path


class IndexStorage:
    def __init__(self, index_root: Path) -> None:
        self.index_root = index_root

    def corpus_index_path(self, corpus_id: str) -> Path:
        return self.index_root / f"{corpus_id}.json"

    def save_index(self, corpus_id: str, payload: dict) -> Path:
        self.index_root.mkdir(parents=True, exist_ok=True)
        output_path = self.corpus_index_path(corpus_id)
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return output_path

    def load_index(self, corpus_id: str) -> dict:
        path = self.corpus_index_path(corpus_id)
        if not path.exists():
            raise FileNotFoundError(f"Index for corpus '{corpus_id}' was not found at '{path}'.")
        return json.loads(path.read_text(encoding="utf-8"))

    def corpus_exists(self, corpus_id: str) -> bool:
        return self.corpus_index_path(corpus_id).exists()

    def list_corpora(self) -> list[str]:
        if not self.index_root.exists():
            return []
        return sorted(path.stem for path in self.index_root.glob("*.json") if path.is_file())

    def corpus_stats(self, corpus_id: str) -> dict:
        payload = self.load_index(corpus_id=corpus_id)
        chunks = payload.get("chunks", {})
        doc_ids = {chunk.get("doc_id") for chunk in chunks.values() if isinstance(chunk, dict)}
        return {
            "corpus_id": corpus_id,
            "total_chunks": int(payload.get("stats", {}).get("total_chunks", len(chunks))),
            "total_docs": len({doc_id for doc_id in doc_ids if doc_id}),
        }
