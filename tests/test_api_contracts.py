from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

SERVICE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SERVICE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from docs_chatbot_service.api import app as app_module
from docs_chatbot_service.core import chat_log_store as cls_module


SAMPLE_CHUNKS = [
    {
        "chunk_id": "projects-1",
        "doc_id": "projects",
        "title": "Projects",
        "section": "general",
        "source": "/portfolio/projects.md",
        "text": "Built a retrieval first chatbot service with BM25 ranking.",
    },
    {
        "chunk_id": "skills-1",
        "doc_id": "skills",
        "title": "Skills",
        "section": "general",
        "source": "/portfolio/skills.md",
        "text": "Python, FastAPI, and search systems.",
    },
]


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.index_root = Path(self.tmp.name) / "index"
        corpus_dir = self.index_root / "portfolio-v1"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        (corpus_dir / "chunks.json").write_text(json.dumps(SAMPLE_CHUNKS), encoding="utf-8")
        cls_module.reset_store_for_tests()
        app_module.service = app_module.RetrievalService(index_root=self.index_root)
        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_search_contract_shape(self) -> None:
        response = self.client.post(
            "/search",
            json={
                "query": "chatbot projects",
                "corpus_id": "portfolio-v1",
                "doc_ids": ["projects", "skills"],
                "top_k": 5,
                "min_score": 0.0,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"], "chatbot projects")
        self.assertEqual(payload["corpus_id"], "portfolio-v1")
        self.assertIn("total_results", payload)
        self.assertEqual(payload["retrieval_model"], "bm25_hashed_vector")
        self.assertIsInstance(payload["results"], list)
        self.assertGreaterEqual(len(payload["results"]), 1)

        result = payload["results"][0]
        self.assertEqual(
            set(result.keys()),
            {"chunk_id", "doc_id", "title", "section", "source", "snippet", "score"},
        )

    def test_search_missing_corpus_is_404(self) -> None:
        response = self.client.post("/search", json={"query": "x1", "corpus_id": "missing"})
        self.assertEqual(response.status_code, 404)

    def test_search_validation_is_422(self) -> None:
        response = self.client.post("/search", json={"query": "a", "top_k": 100})
        self.assertEqual(response.status_code, 422)

    def test_corpora_endpoints_contract(self) -> None:
        list_response = self.client.get("/corpora")
        self.assertEqual(list_response.status_code, 200)
        item = list_response.json()[0]
        self.assertEqual(set(item.keys()), {"corpus_id", "total_chunks", "total_docs"})

        one_response = self.client.get("/corpora/portfolio-v1")
        self.assertEqual(one_response.status_code, 200)
        self.assertEqual(one_response.json()["corpus_id"], "portfolio-v1")

        missing_response = self.client.get("/corpora/missing")
        self.assertEqual(missing_response.status_code, 404)

        exists_response = self.client.get("/corpora/portfolio-v1/exists")
        self.assertEqual(exists_response.status_code, 200)
        self.assertEqual(exists_response.json(), {"corpus_id": "portfolio-v1", "exists": True})

    def test_corpora_load_from_url_contract(self) -> None:
        chunks_file = Path(self.tmp.name) / "external_chunks.json"
        vector_file = Path(self.tmp.name) / "external_vector.json"
        chunks_file.write_text(json.dumps(SAMPLE_CHUNKS), encoding="utf-8")
        vector_file.write_text(
            json.dumps({"dim": 8, "idf": {"1": 1.0}, "vectors": {"projects-1": [0.1] * 8}}),
            encoding="utf-8",
        )
        response = self.client.post(
            "/corpora/load",
            json={
                "corpus_id": "loaded-v1",
                "chunks_url": chunks_file.resolve().as_uri(),
                "vector_index_url": vector_file.resolve().as_uri(),
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["corpus_id"], "loaded-v1")
        self.assertTrue(payload["chunks_loaded"])
        self.assertTrue(payload["vector_loaded"])
        self.assertTrue((self.index_root / "loaded-v1" / "chunks.json").exists())

    def test_chat_endpoint_contract(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "query": "chatbot projects",
                "corpus_id": "portfolio-v1",
                "top_k": 3,
                "min_score": 0.0,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"], "chatbot projects")
        self.assertEqual(payload["corpus_id"], "portfolio-v1")
        self.assertIn("answer", payload)
        self.assertIn("source", payload)
        self.assertIn("used_hf", payload)
        self.assertIn("method", payload)
        self.assertEqual(payload["retrieval_model"], "bm25_hashed_vector")
        self.assertTrue(payload["event_id"])
        self.assertTrue(payload["session_id"])

    def test_chat_feedback_contract(self) -> None:
        chat_response = self.client.post(
            "/chat",
            json={
                "query": "chatbot projects",
                "corpus_id": "portfolio-v1",
                "top_k": 3,
                "min_score": 0.0,
            },
        )
        self.assertEqual(chat_response.status_code, 200)
        event_id = chat_response.json()["event_id"]

        feedback_response = self.client.post(
            "/chat/feedback",
            json={"event_id": event_id, "rating": 1, "comment": "useful answer"},
        )
        self.assertEqual(feedback_response.status_code, 200)
        payload = feedback_response.json()
        self.assertEqual(payload["event_id"], event_id)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["rating"], 1)
        self.assertEqual(payload["bucket"], "positive")

    def test_logging_health_contract(self) -> None:
        response = self.client.get("/health/logging")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload.keys()),
            {
                "enabled",
                "db_url_present",
                "project_url_present",
                "service_role_key_present",
                "store_ready",
                "store_kind",
                "last_init_error",
            },
        )
        self.assertIsInstance(payload["enabled"], bool)
        self.assertIsInstance(payload["db_url_present"], bool)
        self.assertIsInstance(payload["store_ready"], bool)

    def test_search_rule_lexicon_tfidf_model(self) -> None:
        response = self.client.post(
            "/search",
            json={
                "query": "python search",
                "corpus_id": "portfolio-v1",
                "top_k": 5,
                "min_score": 0.0,
                "retrieval_model": "rule_lexicon_tfidf",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["retrieval_model"], "rule_lexicon_tfidf")
        self.assertGreaterEqual(len(payload["results"]), 1)

    def test_search_invalid_retrieval_model_is_422(self) -> None:
        response = self.client.post(
            "/search",
            json={"query": "python", "corpus_id": "portfolio-v1", "retrieval_model": "neural_bert"},
        )
        self.assertEqual(response.status_code, 422)

    def test_search_loads_artifacts_from_urls(self) -> None:
        chunks_file = Path(self.tmp.name) / "search_load_chunks.json"
        chunks_file.write_text(json.dumps(SAMPLE_CHUNKS), encoding="utf-8")
        response = self.client.post(
            "/search",
            json={
                "query": "python",
                "corpus_id": "url-load-v1",
                "top_k": 5,
                "min_score": 0.0,
                "chunks_url": chunks_file.resolve().as_uri(),
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["corpus_id"], "url-load-v1")
        self.assertGreaterEqual(payload["total_results"], 1)

    def test_chat_lightweight_nlp_method(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "query": "chatbot projects",
                "corpus_id": "portfolio-v1",
                "top_k": 3,
                "min_score": 0.0,
                "allow_fallback": True,
                "answer_method": "lightweight_nlp",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["method"], "lightweight_nlp")
        self.assertFalse(payload["used_hf"])
        self.assertEqual(payload["retrieval_model"], "bm25_hashed_vector")
        self.assertIn("chatbot", payload["answer"].lower())

    def test_chat_with_rule_lexicon_tfidf_retrieval(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "query": "python",
                "corpus_id": "portfolio-v1",
                "top_k": 3,
                "min_score": 0.0,
                "allow_fallback": True,
                "answer_method": "lightweight_nlp",
                "retrieval_model": "rule_lexicon_tfidf",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["retrieval_model"], "rule_lexicon_tfidf")
        self.assertEqual(payload["method"], "lightweight_nlp")
        self.assertIn("python", payload["answer"].lower())

    def test_chat_hugging_face_method_smoke(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "query": "python skills",
                "corpus_id": "portfolio-v1",
                "answer_method": "hugging_face",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["method"], ("hugging_face", "none"))
        if payload["method"] == "hugging_face":
            self.assertTrue(payload["used_hf"])
            self.assertTrue(payload["answer"].strip())
        else:
            self.assertFalse(payload["used_hf"])
        self.assertEqual(payload["retrieval_model"], "bm25_hashed_vector")

    def test_chat_hugging_face_lightweight_nlp_method(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "query": "python skills",
                "corpus_id": "portfolio-v1",
                "allow_fallback": True,
                "answer_method": "hugging_face_lightweight_nlp",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["method"], ("hugging_face", "lightweight_nlp"))
        self.assertEqual(payload["retrieval_model"], "bm25_hashed_vector")

    def test_chat_invalid_answer_method_is_422(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "query": "python",
                "corpus_id": "portfolio-v1",
                "answer_method": "not_a_real_method",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_chat_invalid_retrieval_model_is_422(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "query": "python",
                "corpus_id": "portfolio-v1",
                "retrieval_model": "bert",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_chat_requires_chunks_url_when_vector_url_is_provided(self) -> None:
        vector_file = Path(self.tmp.name) / "vector_only.json"
        vector_file.write_text(
            json.dumps({"dim": 8, "idf": {"1": 1.0}, "vectors": {"projects-1": [0.1] * 8}}),
            encoding="utf-8",
        )
        response = self.client.post(
            "/chat",
            json={
                "query": "python",
                "corpus_id": "portfolio-v1",
                "vector_index_url": vector_file.resolve().as_uri(),
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_chat_autoloads_default_corpus_from_env_urls(self) -> None:
        chunks_file = Path(self.tmp.name) / "autoload_chunks.json"
        chunks_file.write_text(json.dumps(SAMPLE_CHUNKS), encoding="utf-8")
        previous = os.environ.get("CHATBOT_CHUNKS_URL")
        os.environ["CHATBOT_CHUNKS_URL"] = chunks_file.resolve().as_uri()
        try:
            app_module.service = app_module.RetrievalService(index_root=self.index_root)
            response = self.client.post(
                "/chat",
                json={"query": "python", "corpus_id": "default", "answer_method": "lightweight_nlp"},
            )
        finally:
            if previous is None:
                os.environ.pop("CHATBOT_CHUNKS_URL", None)
            else:
                os.environ["CHATBOT_CHUNKS_URL"] = previous
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()

