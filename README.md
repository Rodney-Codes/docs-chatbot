# Docs Chatbot Service

Standalone, zero-cost document chatbot backend for any application. Apps supply documents; this service retrieves relevant passages and returns grounded answers.

For AI agents and automation context, read `AI_README.md` first.

## Features

- Search and chat APIs over indexed document corpora.
- Ingest local `.md` / `.txt` files from a `documents/` folder (gitignored for local testing).
- Load pre-built `chunks.json` artifacts by URL for production integrations.
- Composable retrieval models:
  - `bm25`
  - `hashed_vector`
  - `bm25_hashed_vector` (default)
  - `rule_lexicon_tfidf`
- Default answer method: `lightweight_nlp` (no external AI API required).
- Optional Hugging Face answers when `HF_API_ENABLED=true` and `HF_API_TOKEN` is set.

## Repository layout

- `app/main.py`: ASGI entrypoint.
- `documents/`: local test documents (gitignored except `README.md`).
- `src/docs_chatbot_service/`: core service and API modules.
- `tests/`: API contract and NLP unit tests.
- `.env.example`: runtime env template.

## Local run

1. Create and activate venv.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Copy env file:
   - `.env.example` -> `.env`
4. Add test documents under `documents/` (see `documents/README.md`).
5. Run:
   - `uvicorn app.main:app --host 0.0.0.0 --port 8000`

On startup the service auto-ingests `documents/` into the `default` corpus when `CHATBOT_AUTO_INGEST=true`.

## API endpoints

- `GET /health`
- `POST /corpora/ingest` — build index from a documents directory
- `POST /corpora/load` — load `chunks.json` from a URL
- `POST /search`
- `POST /chat`
- `GET /corpora`
- `GET /corpora/{corpus_id}`
- `GET /corpora/{corpus_id}/exists`

### Integrating another application

1. Deploy this service (Docker or uvicorn).
2. Either:
   - publish `chunks.json` and call `POST /corpora/load`, or
   - mount a documents directory and call `POST /corpora/ingest`.
3. Call `POST /chat` with `corpus_id`, `query`, and optionally `answer_method: "lightweight_nlp"`.

## Test

- `python -m unittest tests.test_api_contracts tests.test_query_nlp tests.test_ingest -v`

## Deployment notes

When deploying (any provider):

- Set `CHATBOT_INDEX_ROOT` to a persisted storage path.
- Configure `CORS_ALLOW_ORIGINS` for client apps.
- For zero-cost operation, leave `HF_API_ENABLED=false`.
- Optional in-memory chat logging: `CHAT_LOG_ENABLED=true`.
