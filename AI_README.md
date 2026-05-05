# AI README: Docs Chatbot Service

This file is the source-of-truth context for AI agents working in this repository.

## 1) Purpose

- Standalone chatbot backend extracted from `personal-info`.
- Serves grounded answers over indexed corpus artifacts.
- Supports composable retrieval and answer methods.

## 2) Runtime architecture

- ASGI entrypoint: `app/main.py`
- API: `src/docs_chatbot_service/api/app.py`
- Retrieval service: `src/docs_chatbot_service/core/service.py`
- Search engines:
  - BM25: `core/search.py`
  - Hashed vector: `core/vector_search.py`
  - Rule lexicon TF-IDF: `core/rule_vector_retrieval.py`
- NLP fallback shaping: `core/query_nlp.py`
- Index storage: `core/storage.py`

## 3) API contract

Primary endpoints:

- `GET /health`
- `POST /search`
- `POST /chat`
- `GET /corpora`
- `GET /corpora/{corpus_id}`
- `GET /corpora/{corpus_id}/exists`
- `POST /corpora/load` (artifact URL based corpus load)

### Retrieval models

- `bm25`
- `hashed_vector`
- `bm25_hashed_vector`
- `rule_lexicon_tfidf`

### Answer methods

- `hugging_face`
- `lightweight_nlp`
- `hugging_face_lightweight_nlp`

## 4) Corpus artifact loading

Service can load corpus artifacts by URL:

- `chunks_url` (required for load)
- `vector_index_url` (optional)

Load options:

1. `POST /corpora/load` to preload corpus.
2. Pass `chunks_url` (+ optional `vector_index_url`) directly in `/search` or `/chat`.

Loaded artifacts are persisted under `CHATBOT_INDEX_ROOT/<corpus_id>/`.

## 5) Environment variables

- `CHATBOT_INDEX_ROOT` (default `data/index`)
- `CHAT_ALLOW_FALLBACK`
- `CORS_ALLOW_ORIGINS`
- `HF_API_TOKEN`
- Optional HF tuning:
  - `HF_CHAT_API_URL`
  - `HF_MODEL`
  - `HF_MODEL_FALLBACKS`

## 6) Security guidance

- Never commit secrets (`HF_API_TOKEN`) to git.
- Restrict `CORS_ALLOW_ORIGINS` to known frontend origins.
- Prefer artifact URLs from controlled repos/storage.
- Validate external artifact JSON shape before use (already enforced in API).

## 7) Tests

- `python -m unittest tests.test_api_contracts tests.test_query_nlp -v`

When changing request/response shape, update tests in the same commit.
