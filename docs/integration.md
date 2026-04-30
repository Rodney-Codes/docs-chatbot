# Integration Guide (Service Mode)

This service is integration-first: your app sends a query and receives ranked, source-grounded snippets.

## Service contract

`POST /search`

Request:

```json
{
  "query": "Tell me about chatbot projects",
  "corpus_id": "portfolio-v1",
  "doc_ids": ["projects", "skills"],
  "top_k": 5,
  "min_score": 0.0
}
```

- `corpus_id`: choose which document corpus to search.
- `doc_ids`: optional document-level filter for context-sensitive integrations.

Response:

```json
{
  "query": "Tell me about chatbot projects",
  "corpus_id": "portfolio-v1",
  "total_results": 1,
  "results": [
    {
      "chunk_id": "c1",
      "doc_id": "projects",
      "title": "Projects Overview",
      "section": "AI Work",
      "source": "/portfolio/projects",
      "snippet": "Built a retrieval-first chatbot...",
      "score": 1.1234
    }
  ]
}
```

## Multi-app integration pattern

1. Each integrating app decides its own corpus build and version (`portfolio-v1`, `resume-v2`, `product-docs-v1`).
2. Your app can route users to specific corpora and pass selective `doc_ids` dynamically.
3. Retrieval remains deterministic and fully grounded; no cloud LLM dependency is required.

## Corpus management endpoints

- `GET /corpora`: list available corpora with summary stats.
- `GET /corpora/{corpus_id}`: get stats for one corpus.
- `GET /corpora/{corpus_id}/exists`: fast existence check for integration guards.

Example response (`GET /corpora`):

```json
[
  {
    "corpus_id": "portfolio-v1",
    "total_chunks": 120,
    "total_docs": 17
  }
]
```

## Corpus build pipeline

Use the same pipeline for every integration target. This keeps ingestion/index behavior consistent across apps.

1. Put `.md` or `.txt` files in `data/raw`.
2. Build a corpus artifact:
   - `python scripts/build_corpus.py --corpus-id portfolio-v1 --raw-dir data/raw --source-prefix /portfolio`
3. Query using that corpus:
   - `POST /search` with `"corpus_id": "portfolio-v1"`

The pipeline runs:

- `scripts/ingest.py` -> `data/processed/docs.json`
- `scripts/chunk.py` -> `data/processed/chunks.json`
- `scripts/build_index.py` -> `data/index/<corpus_id>.json`
