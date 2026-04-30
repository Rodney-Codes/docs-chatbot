from __future__ import annotations

import argparse
from pathlib import Path

from docs_chatbot_service.core.indexer import build_index_from_chunks
from docs_chatbot_service.core.storage import IndexStorage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25 index from chunk JSON.")
    parser.add_argument("--corpus-id", required=True, help="Corpus identifier, e.g. portfolio-v1.")
    parser.add_argument(
        "--chunks-path",
        default="data/processed/chunks.json",
        help="Path to chunk JSON array.",
    )
    parser.add_argument(
        "--index-root",
        default="data/index",
        help="Directory where index artifacts are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks_path = Path(args.chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    payload = build_index_from_chunks(chunks_path=chunks_path, corpus_id=args.corpus_id)
    storage = IndexStorage(index_root=Path(args.index_root))
    output = storage.save_index(corpus_id=args.corpus_id, payload=payload)
    print(f"Index generated: {output}")


if __name__ == "__main__":
    main()
