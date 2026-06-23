"""
Build searchable corpus chunks from plain-text source files on disk.

Supported extensions: .md, .markdown, .txt
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from docs_chatbot_service.core.chunking import chunk_text
from docs_chatbot_service.core.indexer import build_index
from docs_chatbot_service.core.storage import IndexStorage
from docs_chatbot_service.core.vector_search import HashedVectorIndex

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}
HEADING_RE = re.compile(r"^#{1,3}\s+(.+?)\s*$")


@dataclass(frozen=True)
class IngestResult:
    corpus_id: str
    documents_found: int
    chunks_written: int
    vector_index_built: bool
    chunks_path: str
    vector_path: str


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "section"


def split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_title = "general"
    current_lines: List[str] = []
    saw_heading = False

    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            saw_heading = True
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_title, body))
            current_title = match.group(1).strip() or "general"
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_title, body))

    if not sections and text.strip():
        return [("general", text.strip())]
    if not saw_heading and sections:
        return [("general", sections[0][1])]
    return sections


def _iter_source_files(documents_root: Path) -> Iterable[Path]:
    if not documents_root.exists():
        return []
    files = [
        path
        for path in sorted(documents_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def _sections_for_file(path: Path) -> List[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".txt":
        return [("general", text)]
    return split_markdown_sections(text)


def build_chunks_from_directory(documents_root: Path) -> List[dict]:
    chunks: List[dict] = []
    for path in _iter_source_files(documents_root):
        doc_id = _slug(path.stem)
        rel_source = str(path.as_posix())
        sections = _sections_for_file(path)
        for section_title, section_text in sections:
            section_slug = _slug(section_title)
            parts = chunk_text(section_text)
            if not parts:
                continue
            for idx, part in enumerate(parts, start=1):
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}-{section_slug}-{idx}",
                        "doc_id": doc_id,
                        "title": path.stem.replace("-", " ").replace("_", " ").title(),
                        "section": section_title,
                        "source": rel_source,
                        "text": part,
                    }
                )
    return chunks


def ingest_documents_directory(
    documents_root: Path,
    corpus_id: str,
    index_root: Path,
    *,
    build_vector_index: bool = True,
) -> IngestResult:
    documents_root = documents_root.resolve()
    index_root = index_root.resolve()
    chunks = build_chunks_from_directory(documents_root)
    if not chunks:
        raise FileNotFoundError(
            f"No supported documents found under {documents_root}. "
            f"Add .md, .markdown, or .txt files."
        )

    chunks_path = build_index(corpus_id=corpus_id, chunks=chunks, index_root=index_root)
    vector_path = ""
    vector_built = False
    if build_vector_index:
        vector_index = HashedVectorIndex.from_chunks(chunks)
        vector_file = index_root / corpus_id / "vector_index.json"
        vector_index.save(vector_file)
        vector_path = str(vector_file)
        vector_built = True

    documents_found = len(list(_iter_source_files(documents_root)))
    return IngestResult(
        corpus_id=corpus_id,
        documents_found=documents_found,
        chunks_written=len(chunks),
        vector_index_built=vector_built,
        chunks_path=str(chunks_path),
        vector_path=vector_path,
    )


def documents_root_has_sources(documents_root: Path) -> bool:
    return any(True for _ in _iter_source_files(documents_root))


def ingest_into_storage(
    documents_root: Path,
    corpus_id: str,
    storage: IndexStorage,
    *,
    build_vector_index: bool = True,
) -> IngestResult:
    return ingest_documents_directory(
        documents_root=documents_root,
        corpus_id=corpus_id,
        index_root=storage.index_root,
        build_vector_index=build_vector_index,
    )
