# Docs Chatbot Service

Standalone chatbot backend with composable retrieval and answer pipelines.

For AI agents and automation context, read `AI_README.md` first.

## Features

- Search and chat APIs over indexed document corpora.
- Composable retrieval models:
  - `bm25`
  - `hashed_vector`
  - `bm25_hashed_vector`
  - `rule_lexicon_tfidf`
- Composable answer methods:
  - `hugging_face`
  - `lightweight_nlp`
  - `hugging_face_lightweight_nlp`

## Repository layout

- `app/main.py`: ASGI entrypoint.
- `src/docs_chatbot_service/`: core service and API modules.
- `tests/`: API contract and NLP unit tests.
- `.env.example`: runtime env template.

## Local run

1. Create and activate venv.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy env file:
   - `.env.example` -> `.env`
4. Run:
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## API endpoints

- `GET /health`
- `POST /corpora/load`
- `POST /search`
- `POST /chat`
- `GET /corpora`
- `GET /corpora/{corpus_id}`
- `GET /corpora/{corpus_id}/exists`

`/corpora/load` can load `chunks.json` and optional `vector_index.json` by URL,
so app-specific artifacts can stay in application repos while this service repo
remains generic.

## Test

- `python -m unittest tests.test_api_contracts tests.test_query_nlp -v`

## Deployment notes

- Set `CHATBOT_INDEX_ROOT` to persisted storage path (default `data/index`).
- Set `HF_API_TOKEN` if using `hugging_face*` answer methods.
- Configure `CORS_ALLOW_ORIGINS` for client apps.
