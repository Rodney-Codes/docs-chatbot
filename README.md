# Chatbot Service (Migration Workspace)

This folder is the migration workspace to run the chatbot as an independent service while still living in the same repository.

## Current approach

- Runtime entrypoint lives in this folder (`app/main.py`).
- Chatbot implementation is service-local at `services/chatbot/src/docs_chatbot_service`.
- Service import resolution uses only `services/chatbot/src` for chatbot package loading.

## Run locally

From repo root:

- `python -m venv .venv` (if not already created)
- `.\.venv\Scripts\python -m pip install -r services/chatbot/requirements.txt`
- `.\.venv\Scripts\python -m uvicorn services.chatbot.app.main:app --host 0.0.0.0 --port 8000`

## API

Primary endpoints:

- `GET /health`
- `POST /search`
- `POST /chat`
- `GET /corpora`
- `GET /corpora/{corpus_id}`
- `GET /corpora/{corpus_id}/exists`

## Migration phases

1. **Phase 1 (completed):** standalone service wrapper + deployment scaffolding.
2. **Phase 2 (completed):** moved `docs_chatbot_service` code into this folder and removed root `src` dependency.
3. **Phase 3:** split this folder to its own repository without changing API contract.
