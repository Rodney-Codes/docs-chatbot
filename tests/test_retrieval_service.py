from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from docs_chatbot_service.core.indexer import build_index_from_chunks
from docs_chatbot_service.core.service import RetrievalService
from docs_chatbot_service.core.storage import IndexStorage
from docs_chatbot_service.schemas.contracts import SearchRequest


class RetrievalServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.chunks_path = self.base_path / "chunks.json"
        self.index_root = self.base_path / "index"

        chunks = [
            {
                "chunk_id": "chunk_1",
                "doc_id": "projects",
                "title": "Projects",
                "section": "Chatbot",
                "source": "/projects/chatbot",
                "text": "Built zero-cost retrieval chatbot with BM25 ranking.",
            },
            {
                "chunk_id": "chunk_2",
                "doc_id": "experience",
                "title": "Experience",
                "section": "Data",
                "source": "/experience/data",
                "text": "Created data pipelines and monitoring alerts.",
            },
        ]
        self.chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

        payload = build_index_from_chunks(chunks_path=self.chunks_path, corpus_id="default")
        storage = IndexStorage(index_root=self.index_root)
        storage.save_index("default", payload)
        self.service = RetrievalService(index_root=self.index_root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_returns_relevant_hit(self) -> None:
        response = self.service.query(SearchRequest(query="chatbot bm25", corpus_id="default"))
        self.assertGreaterEqual(response.total_results, 1)
        self.assertEqual(response.results[0].doc_id, "projects")

    def test_doc_filter_works(self) -> None:
        response = self.service.query(
            SearchRequest(query="data pipelines", corpus_id="default", doc_ids=["projects"])
        )
        self.assertEqual(response.total_results, 0)


if __name__ == "__main__":
    unittest.main()
