# Docs Chatbot Service (Phase 1)

Retrieval-first chatbot backend with zero cloud LLM calls.

For AI agents and automated integrations, read `AI_README.md` first.

This project provides a production-friendly service interface for integrating search-based chatbot responses into any app.

## Goals

- Zero ongoing AI token cost.
- Grounded responses from your own documents.
- Easy integration: send query + corpus + optional document filters.

## Project structure

- `src/docs_chatbot_service/core`: indexing, storage, search, service logic.
- `src/docs_chatbot_service/api`: FastAPI app and endpoints.
- `scripts/build_index.py`: index build pipeline.
- `docs/integration.md`: integration contract.
- `tests/test_retrieval_service.py`: retrieval behavior tests.

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -e .`
   - For running tests: `pip install -e ".[dev]"`
3. Build corpus end-to-end from raw docs:
   - `python scripts/build_corpus.py --corpus-id default --raw-dir data/raw --source-prefix /portfolio`
4. Run API:
   - `uvicorn docs_chatbot_service.main:app --reload`

## API

- `GET /health`
- `POST /search`
- `GET /corpora`
- `GET /corpora/{corpus_id}`
- `GET /corpora/{corpus_id}/exists`

Example request:

```json
{
  "query": "chatbot bm25",
  "corpus_id": "default",
  "doc_ids": ["projects"],
  "top_k": 5
}
```

`doc_ids` is optional and lets integrating apps choose which documents are reference-eligible for each query.

## Data pipeline scripts

- `scripts/ingest.py`: reads `data/raw` and generates normalized `docs.json`
- `scripts/chunk.py`: splits normalized docs into retrieval chunks
- `scripts/build_index.py`: builds BM25 index artifact for one corpus
- `scripts/build_corpus.py`: orchestrates full build in one command

## Tests

- `python -m unittest discover -s tests -p "test_*.py"`