from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from docs_chatbot_service.api.app import create_app


class CorporaApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index_root = Path("data/index")
        app = create_app(index_root=self.index_root)
        self.client = TestClient(app)

    def test_list_corpora_contains_default(self) -> None:
        response = self.client.get("/corpora")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        corpus_ids = {entry["corpus_id"] for entry in payload}
        self.assertIn("default", corpus_ids)

    def test_corpus_exists_endpoint(self) -> None:
        response = self.client.get("/corpora/default/exists")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["exists"])

    def test_corpus_stats_endpoint(self) -> None:
        response = self.client.get("/corpora/default")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["corpus_id"], "default")
        self.assertGreaterEqual(payload["total_chunks"], 1)


if __name__ == "__main__":
    unittest.main()
