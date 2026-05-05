from __future__ import annotations

import json
from pathlib import Path
from typing import List


def build_index(corpus_id: str, chunks: List[dict], index_root: Path) -> Path:
    corpus_dir = index_root / corpus_id
    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_path = corpus_dir / "chunks.json"
    output_path.write_text(json.dumps(chunks, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path

