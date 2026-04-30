from __future__ import annotations

import unittest

from docs_chatbot_service.core.chunking import ChunkingConfig, split_text_into_word_chunks


class ChunkingTests(unittest.TestCase):
    def test_chunking_with_overlap(self) -> None:
        text = " ".join([f"w{i}" for i in range(1, 21)])
        config = ChunkingConfig(chunk_size_words=8, overlap_words=2)
        chunks = split_text_into_word_chunks(text=text, config=config)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0].startswith("w1"))
        self.assertTrue(chunks[1].startswith("w7"))

    def test_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            ChunkingConfig(chunk_size_words=10, overlap_words=10)


if __name__ == "__main__":
    unittest.main()
