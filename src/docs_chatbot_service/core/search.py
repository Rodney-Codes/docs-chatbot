from __future__ import annotations

from collections import defaultdict

from docs_chatbot_service.core.models import Chunk, SearchHit
from docs_chatbot_service.core.text import tokenize


class BM25Searcher:
    def __init__(self, index_payload: dict) -> None:
        self.index_payload = index_payload
        self.idf = index_payload["idf"]
        self.postings = index_payload["postings"]
        self.chunk_lengths = index_payload["chunk_lengths"]
        self.chunks = index_payload["chunks"]
        self.avg_chunk_length = index_payload["stats"]["avg_chunk_length"] or 1.0
        self.k1 = float(index_payload["bm25"]["k1"])
        self.b = float(index_payload["bm25"]["b"])

    def search(
        self,
        query: str,
        top_k: int,
        doc_ids: set[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        scores: dict[str, float] = defaultdict(float)
        query_terms = tokenize(query)

        for term in query_terms:
            if term not in self.postings:
                continue
            idf = float(self.idf.get(term, 0.0))
            for posting in self.postings[term]:
                chunk_id = posting["chunk_id"]
                chunk = self.chunks[chunk_id]
                if doc_ids is not None and chunk["doc_id"] not in doc_ids:
                    continue

                tf = float(posting["tf"])
                dl = float(self.chunk_lengths[chunk_id])
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_chunk_length))
                scores[chunk_id] += idf * (numerator / denominator)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        hits: list[SearchHit] = []
        for chunk_id, score in ranked[:top_k]:
            if score < min_score:
                continue
            chunk_data = self.chunks[chunk_id]
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        chunk_id=chunk_data["chunk_id"],
                        doc_id=chunk_data["doc_id"],
                        title=chunk_data["title"],
                        section=chunk_data["section"],
                        source=chunk_data["source"],
                        text=chunk_data["text"],
                    ),
                    score=score,
                )
            )
        return hits
