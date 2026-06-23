from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from docs_chatbot_service.core.ingest import build_chunks_from_directory, split_markdown_sections


class IngestTests(unittest.TestCase):
    def test_split_markdown_sections(self) -> None:
        text = "# Title\n\nIntro.\n\n## Setup\n\nRun pip install.\n"
        sections = split_markdown_sections(text)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0][0], "Title")
        self.assertIn("Intro", sections[0][1])
        self.assertEqual(sections[1][0], "Setup")
        self.assertIn("pip install", sections[1][1])

    def test_build_chunks_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs_dir = Path(tmp)
            (docs_dir / "notes.txt").write_text("Plain text about analytics.", encoding="utf-8")
            chunks = build_chunks_from_directory(docs_dir)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["doc_id"], "notes")
            self.assertIn("analytics", chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
