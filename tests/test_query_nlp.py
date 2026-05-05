from __future__ import annotations

import sys
import unittest
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SERVICE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from docs_chatbot_service.core.query_nlp import (
    analyze_query,
    detect_intent_from_signals,
    extract_best_sentences,
    phrase_anchor_bigrams,
    segment_query,
    weighted_query_terms,
)
from docs_chatbot_service.core.search import BM25SearchEngine


class QueryNlpTests(unittest.TestCase):
    def test_segment_query_splits_conjunctions(self) -> None:
        self.assertEqual(
            segment_query("Python experience and Tableau dashboards"),
            ["Python experience", "Tableau dashboards"],
        )

    def test_weighted_query_expands_synonyms(self) -> None:
        terms = dict(weighted_query_terms("What is your tech stack?"))
        self.assertGreaterEqual(terms.get("stack", 0), 1.0)
        self.assertGreater(terms.get("skills", 0), 0.0)

    def test_phrase_anchor_bigrams(self) -> None:
        anchors = phrase_anchor_bigrams("Do you use Power BI for dashboards?")
        self.assertIn(("power", "bi"), anchors)

    def test_intent_uses_expanded_tokens(self) -> None:
        signals = analyze_query("Tell me about your background")
        self.assertEqual(detect_intent_from_signals(signals), "experience")

    def test_extract_best_sentences_prefers_phrase_overlap(self) -> None:
        signals = analyze_query("machine learning projects")
        text = "I like hiking. I built models with machine learning pipelines. Other stuff."
        best = extract_best_sentences(text, signals, max_sentences=1)
        self.assertTrue(best)
        self.assertIn("machine learning", best[0].lower())

    def test_bm25_accepts_weighted_terms(self) -> None:
        chunks = [
            {
                "chunk_id": "a",
                "doc_id": "d",
                "title": "",
                "section": "general",
                "source": "",
                "text": "Strong Python and SQL for analytics engineering.",
            }
        ]
        engine = BM25SearchEngine(chunks)
        score = engine.score("analytics stack", chunks[0])
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
