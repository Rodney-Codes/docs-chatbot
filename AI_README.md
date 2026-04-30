# AI README: Docs Chatbot Service

This file is the single-source context for any AI agent working in this repo.

## 1) What this repo is

- Retrieval-only chatbot backend (Phase 1), no cloud LLM calls.
- Purpose: return grounded snippets from owned documents at near-zero ongoing cost.
- Runtime model: caller app sends query + corpus selection (+ optional document filter), service returns ranked snippets with source metadata.

## 2) Architecture snapshot

- API layer: `src/docs_chatbot_service/api/app.py`
- Retrieval service: `src/docs_chatbot_service/core/service.py`
- Search engine (BM25): `src/docs_chatbot_service/core/search.py`
- Index build: `src/docs_chatbot_service/core/indexer.py`
- Index storage/discovery: `src/docs_chatbot_service/core/storage.py`
- Ingestion/chunking logic: `scripts/ingest.py`, `scripts/chunk.py`, `src/docs_chatbot_service/core/chunking.py`
- End-to-end corpus build: `scripts/build_corpus.py`

## 3) Setup and run (required sequence)

1. Create virtual environment.
2. Install package:
   - `pip install -e .`
3. (Optional for tests) install dev extras:
   - `pip install -e ".[dev]"`
4. Build corpus:
   - `python scripts/build_corpus.py --corpus-id default --raw-dir data/raw --source-prefix /portfolio`
5. Run API:
   - `uvicorn docs_chatbot_service.main:app --reload`

## 4) Integration contract (must-follow)

### Endpoint: `POST /search`

Request JSON:

```json
{
  "query": "Tell me about chatbot projects",
  "corpus_id": "portfolio-v1",
  "doc_ids": ["projects", "skills"],
  "top_k": 5,
  "min_score": 0.0
}
```

Request field rules:

- `query` (string, required): min length 2.
- `corpus_id` (string, optional): defaults to `"default"`.
- `doc_ids` (string array, optional): allow-list of docs to reference.
- `top_k` (int, optional): range `1..20`, default `5`.
- `min_score` (float, optional): `>=0`, default `0.0`.

Response JSON shape:

```json
{
  "query": "Tell me about chatbot projects",
  "corpus_id": "portfolio-v1",
  "total_results": 1,
  "results": [
    {
      "chunk_id": "projects-1",
      "doc_id": "projects",
      "title": "Projects",
      "section": "general",
      "source": "/portfolio/projects.md",
      "snippet": "Built a retrieval-first chatbot service...",
      "score": 1.234567
    }
  ]
}
```

Result semantics:

- `results` is score-descending.
- `snippet` is extractive text (not generated prose).
- `source` is integration-facing reference path/url.

## 5) Corpus management endpoints

- `GET /health` -> service health.
- `GET /corpora` -> list available corpora with stats.
- `GET /corpora/{corpus_id}` -> stats for one corpus.
- `GET /corpora/{corpus_id}/exists` -> boolean availability check.

`GET /corpora` response item format:

```json
{
  "corpus_id": "portfolio-v1",
  "total_chunks": 120,
  "total_docs": 17
}
```

## 6) How documents must be prepared

- Raw inputs: `.md` and `.txt` files in raw directory.
- Ingestion output (`docs.json`) record format:

```json
{
  "doc_id": "projects",
  "title": "Projects",
  "section": "general",
  "source": "/portfolio/projects.md",
  "text": "full normalized content",
  "updated_at": "2026-04-30T10:00:00+00:00"
}
```

- Chunk output (`chunks.json`) record format:

```json
{
  "chunk_id": "projects-1",
  "doc_id": "projects",
  "title": "Projects",
  "section": "general",
  "source": "/portfolio/projects.md",
  "text": "chunk text"
}
```

## 7) Standard pipeline (raw docs -> searchable corpus)

Single command:

- `python scripts/build_corpus.py --corpus-id <corpus_id> --raw-dir <raw_dir> --source-prefix <source_prefix>`

Equivalent step-by-step:

1. `python scripts/ingest.py --raw-dir ... --output-path data/processed/docs.json --source-prefix ...`
2. `python scripts/chunk.py --docs-path data/processed/docs.json --output-path data/processed/chunks.json --chunk-size-words 450 --overlap-words 70`
3. `python scripts/build_index.py --corpus-id ... --chunks-path data/processed/chunks.json --index-root data/index`

## 8) Error behavior to expect

- Missing corpus on `/search` -> HTTP `404`.
- Missing corpus on `/corpora/{corpus_id}` -> HTTP `404`.
- Validation errors (bad request fields) -> HTTP `422`.

## 9) Test command

- `python -m unittest discover -s tests -p "test_*.py"`

## 10) Agent operating rules for this repo

- Keep retrieval grounded; do not add mandatory LLM generation in Phase 1 paths.
- Preserve request/response schema backward compatibility unless explicitly versioning API.
- Keep corpus routing (`corpus_id`) and per-query doc filtering (`doc_ids`) first-class.
- Prefer deterministic behavior and explicit source metadata in all returned results.
