from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest markdown/text docs into normalized JSON.")
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory that stores source docs.",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/docs.json",
        help="Output JSON path for normalized docs.",
    )
    parser.add_argument(
        "--source-prefix",
        default="/docs",
        help="Source URL prefix used to build source links.",
    )
    return parser.parse_args()


def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _load_doc(doc_path: Path, raw_dir: Path, source_prefix: str) -> dict:
    content = doc_path.read_text(encoding="utf-8")
    relative = doc_path.relative_to(raw_dir).as_posix()
    stem = doc_path.stem
    return {
        "doc_id": stem.lower().replace(" ", "-"),
        "title": _extract_title(content=content, fallback=stem),
        "section": "general",
        "source": f"{source_prefix.rstrip('/')}/{relative}",
        "text": content.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output_path)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw documents directory not found: {raw_dir}")

    files = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    docs = [_load_doc(path, raw_dir=raw_dir, source_prefix=args.source_prefix) for path in files]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(docs, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Ingested {len(docs)} docs into: {output_path}")


if __name__ == "__main__":
    main()
