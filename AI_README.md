# AI README: Docs Chatbot Service

**Source of truth for AI agents.** Read this file first before changing code, debugging, or integrating this service into another application.

---

## 1) What this repo is

A **standalone, generic document chatbot microservice** (FastAPI). Any application can:

1. Supply documents (local files or pre-built JSON artifacts).
2. Call HTTP APIs to **search** or **chat** over those documents.
3. Get **grounded answers** without paying for an external LLM (default path).

**Design goals:**

- **Retrieval-first RAG** — answers come from indexed document chunks, not free-form generation.
- **Zero-cost default** — `bm25_hashed_vector` retrieval + `lightweight_nlp` answers. No database. No Supabase. No Render config in repo.
- **Multi-corpus** — each integrating app uses its own `corpus_id`.
- **Composable** — retrieval model and answer method are selectable per request.

**Not in scope (today):**

- User authentication / API keys on this service
- Persistent chat history (optional in-memory logging only)
- Crawling websites or parsing PDFs (apps pre-process docs into text/markdown)
- Built-in frontend UI (apps build their own)

---

## 2) Repository layout

```
docs-chatbot/
  app/main.py              # ASGI entrypoint; rebinds index root; startup document ingest
  src/docs_chatbot_service/
    api/app.py             # FastAPI routes and request/response models
    core/
      ingest.py            # documents/ -> chunks.json + vector_index.json
      service.py           # RetrievalService orchestration
      search.py            # BM25
      vector_search.py     # Hashed char-ngram vectors (no ML model)
      rule_vector_retrieval.py  # Rule lexicon TF-IDF (optional)
      query_nlp.py         # Query understanding, sentence extraction
      storage.py           # Corpus file I/O under CHATBOT_INDEX_ROOT
      chunking.py          # Word-window text splitting
      chat_log_store.py    # In-memory chat events + feedback (optional)
  documents/               # Local test docs (gitignored except README.md)
  data/index/              # Generated indexes (gitignored)
  tests/                   # unittest contract + NLP + ingest tests
  Dockerfile               # Container image (uvicorn on port 8000)
  .env.example             # Runtime env template
```

---

## 3) RAG pipeline

```
[App documents]  -->  ingest or load  -->  chunks.json (+ optional vector_index.json)
                                              |
User question  -->  query NLP  -->  retrieve top-k chunks  -->  answer shaping  -->  JSON response
```

### Ingest path (`core/ingest.py`)

- Scans `CHATBOT_DOCUMENTS_ROOT` for `.md`, `.markdown`, `.txt` (recursive).
- Splits markdown on `#` / `##` / `###` headings into sections.
- Splits long sections with word-window chunking (`chunking.py`).
- Writes `CHATBOT_INDEX_ROOT/<corpus_id>/chunks.json`.
- Builds and saves `vector_index.json` (hashed vectors) for hybrid retrieval.

### Retrieval (`core/service.py`)

Default: `bm25_hashed_vector` — combines BM25 keyword score with hashed vector cosine similarity.

### Answer generation (`api/app.py`)

Default: `lightweight_nlp` — extracts best sentences from retrieved chunks and prefixes with a neutral template (e.g. "Based on the provided documents, ...").

Optional: Hugging Face chat completions when `HF_API_ENABLED=true` and `HF_API_TOKEN` is set.

---

## 4) API reference

Interactive docs when server is running:

- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/health/logging` | In-memory chat log store diagnostics |
| GET | `/corpora` | List loaded corpora + stats |
| GET | `/corpora/{corpus_id}` | Stats for one corpus |
| GET | `/corpora/{corpus_id}/exists` | Boolean existence check |
| POST | `/corpora/ingest` | Build index from a documents directory |
| POST | `/corpora/load` | Load `chunks.json` from URL |
| POST | `/search` | Retrieve ranked snippets (no answer synthesis) |
| POST | `/chat` | Retrieve + generate grounded answer |
| POST | `/chat/feedback` | Rate a prior chat `event_id` (requires `CHAT_LOG_ENABLED=true`) |

### Retrieval models (`retrieval_model`)

| Value | Description |
|-------|-------------|
| `bm25_hashed_vector` | **Default.** Hybrid BM25 + hashed vectors |
| `bm25` | Keyword only |
| `hashed_vector` | Hashed vectors only |
| `rule_lexicon_tfidf` | Domain rule lexicon + TF-IDF (portfolio-style lexicon baked in) |

### Answer methods (`answer_method`)

| Value | Cost | Description |
|-------|------|-------------|
| `lightweight_nlp` | **Default. Free.** | Sentence extraction + template |
| `hugging_face` | API cost if enabled | LLM answer from evidence |
| `hugging_face_lightweight_nlp` | API cost if enabled | HF first, then lightweight fallback |

---

## 5) Request / response examples

### POST `/corpora/ingest`

Build or refresh a corpus from files on disk.

**Request:**

```json
{
  "corpus_id": "my-app-docs",
  "documents_root": null,
  "build_vector_index": true
}
```

`documents_root` optional — defaults to `CHATBOT_DOCUMENTS_ROOT`. Relative paths resolve from repo/service root.

**Response:**

```json
{
  "corpus_id": "my-app-docs",
  "documents_found": 3,
  "chunks_written": 15,
  "vector_index_built": true,
  "chunks_path": "/app/data/index/my-app-docs/chunks.json",
  "vector_path": "/app/data/index/my-app-docs/vector_index.json"
}
```

### POST `/corpora/load`

Load pre-built artifacts (typical **production** path for integrating apps).

**Request:**

```json
{
  "corpus_id": "my-app-docs",
  "chunks_url": "https://example.com/artifacts/my-app/chunks.json",
  "vector_index_url": "https://example.com/artifacts/my-app/vector_index.json"
}
```

`vector_index_url` is optional. `chunks_url` is required.

### POST `/search`

**Request:**

```json
{
  "query": "how does authentication work",
  "corpus_id": "my-app-docs",
  "doc_ids": null,
  "top_k": 5,
  "min_score": 0.0,
  "retrieval_model": null,
  "chunks_url": null,
  "vector_index_url": null
}
```

**Response (shape):**

```json
{
  "query": "how does authentication work",
  "corpus_id": "my-app-docs",
  "total_results": 3,
  "retrieval_model": "bm25_hashed_vector",
  "results": [
    {
      "chunk_id": "auth-overview-1",
      "doc_id": "auth",
      "title": "Auth",
      "section": "Overview",
      "source": "docs/auth.md",
      "snippet": "API requests require a Bearer token...",
      "score": 0.74
    }
  ]
}
```

### POST `/chat`

**Request (typical integrating app):**

```json
{
  "query": "how does authentication work",
  "corpus_id": "my-app-docs",
  "top_k": 3,
  "answer_method": "lightweight_nlp",
  "session_id": "user-session-abc-123"
}
```

**Response (shape):**

```json
{
  "query": "how does authentication work",
  "corpus_id": "my-app-docs",
  "answer": "Based on the provided documents, API requests require a Bearer token...",
  "source": "docs/auth.md",
  "used_hf": false,
  "method": "lightweight_nlp",
  "retrieval_model": "bm25_hashed_vector",
  "event_id": "uuid",
  "session_id": "user-session-abc-123"
}
```

### POST `/chat/feedback`

Only persists when `CHAT_LOG_ENABLED=true` (in-memory; lost on restart).

```json
{
  "event_id": "uuid-from-chat-response",
  "rating": 1,
  "comment": "helpful",
  "session_id": "user-session-abc-123"
}
```

---

## 6) `chunks.json` contract

Integrating apps can publish this file from CI instead of using `/corpora/ingest`.

**Type:** JSON array of chunk objects.

**Required fields per chunk:**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | Unique within corpus (e.g. `auth-overview-1`) |
| `doc_id` | string | Logical document id (e.g. `auth`) |
| `text` | string | Chunk body used for retrieval and answers |

**Recommended fields:**

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Human document title |
| `section` | string | Section heading (default `general`) |
| `source` | string | Provenance path or URL |

**Example:**

```json
[
  {
    "chunk_id": "auth-overview-1",
    "doc_id": "auth",
    "title": "Authentication",
    "section": "Overview",
    "source": "docs/auth.md",
    "text": "API requests require a Bearer token in the Authorization header. Tokens expire after 24 hours."
  }
]
```

**Optional `vector_index.json`:** Precomputed hashed vectors. If omitted, hybrid retrieval falls back to BM25-only for vector component. Ingest builds this automatically.

**On-disk layout:**

```
CHATBOT_INDEX_ROOT/
  <corpus_id>/
    chunks.json
    vector_index.json   # optional but recommended
```

---

## 7) How integrating applications use this service

### Pattern A — Production (recommended)

Each app owns its document pipeline and hosts artifacts.

1. App CI converts docs (markdown, etc.) to `chunks.json` (+ optional `vector_index.json`).
2. App deploys this chatbot service (Docker or uvicorn).
3. On app deploy or startup hook, call `POST /corpora/load` with artifact URLs and a unique `corpus_id` (e.g. `acme-help-center`).
4. App backend or frontend calls `POST /chat` with that `corpus_id` and the user's `query`.
5. Display `answer`, optionally cite `source`.

### Pattern B — Documents directory (dev / simple deploy)

1. Mount or copy app docs into `documents/` (or a custom path).
2. `POST /corpora/ingest` with `corpus_id` (or rely on startup auto-ingest for `default` corpus).
3. Call `POST /chat`.

### Pattern C — Per-request artifact load

Pass `chunks_url` (and optional `vector_index_url`) directly on `/search` or `/chat`. Useful for one-off loads; service persists under `CHATBOT_INDEX_ROOT`.

### Multi-tenant isolation

- One service instance can host **multiple corpora** via distinct `corpus_id` values.
- Filter to specific docs with `doc_ids: ["auth", "billing"]` on `/search` and `/chat`.
- There is **no auth** between tenants today — use one deployment per app, or add API gateway auth in front.

### CORS

Browser clients must list their origin in `CORS_ALLOW_ORIGINS` (comma-separated). Server-side callers (app backend) do not need CORS.

### Example backend call (Python)

```python
import httpx

def ask_docs_chatbot(query: str, corpus_id: str = "my-app-docs") -> str:
    response = httpx.post(
        "http://docs-chatbot:8000/chat",
        json={"query": query, "corpus_id": corpus_id},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["answer"]
```

### Example from browser (only if CORS allows origin)

```javascript
const res = await fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "how does auth work", corpus_id: "my-app-docs" }),
});
const data = await res.json();
console.log(data.answer, data.source);
```

---

## 8) Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATBOT_INDEX_ROOT` | `data/index` | Persisted index directory |
| `CHATBOT_DOCUMENTS_ROOT` | `documents` | Source files for ingest |
| `CHATBOT_DEFAULT_CORPUS_ID` | `default` | Corpus for auto-ingest |
| `CHATBOT_AUTO_INGEST` | `true` | Ingest documents on startup |
| `CHAT_ALLOW_FALLBACK` | `true` | **Must be true** for `lightweight_nlp` answers. If `false`, `/chat` returns `method: none` even when chunks match. |
| `CORS_ALLOW_ORIGINS` | empty | Comma-separated browser origins |
| `CHAT_LOG_ENABLED` | `false` | In-memory chat event + feedback logging |
| `HF_API_ENABLED` | `false` | Gate Hugging Face answer path |
| `HF_API_TOKEN` | empty | HF API token when enabled |
| `HF_CHAT_API_URL` | HF router URL | Chat completions endpoint |
| `HF_MODEL` | empty | Primary model |
| `HF_MODEL_FALLBACKS` | llama fallback | Comma-separated fallbacks |
| `CHATBOT_CHUNKS_URL` | empty | Auto-load default corpus from URL if missing |
| `CHATBOT_VECTOR_INDEX_URL` | empty | Optional vector index for auto-load |

**Docker note:** Override paths for the container filesystem:

```
CHATBOT_INDEX_ROOT=/app/data/index
CHATBOT_DOCUMENTS_ROOT=/app/documents
```

Mount host folders: `-v ./documents:/app/documents -v ./data/index:/app/data/index`

**Restart required** after `.env` changes (loaded at process start).

---

## 9) Running and testing

### Local (venv)

```bash
pip install -r requirements.txt
cp .env.example .env
# Add .md/.txt files under documents/
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t docs-chatbot .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/documents:/app/documents" \
  -v "$(pwd)/data/index:/app/data/index" \
  -e CHATBOT_DOCUMENTS_ROOT=/app/documents \
  -e CHATBOT_INDEX_ROOT=/app/data/index \
  -e CHAT_ALLOW_FALLBACK=true \
  -e HF_API_ENABLED=false \
  docs-chatbot
```

### Unit tests

```bash
python -m unittest tests.test_api_contracts tests.test_query_nlp tests.test_ingest -v
```

### Manual smoke

1. `GET /health` -> `{"status":"ok"}`
2. `GET /corpora` -> corpus with `total_chunks > 0`
3. `POST /search` -> ranked `results`
4. `POST /chat` -> `method: lightweight_nlp`, grounded `answer`

---

## 10) Document authoring (for ingest)

- **No FAQ Q/A format required** — use clear headings and factual prose.
- **Prefer `##` sections** — one topic per section improves chunk quality.
- **Use vocabulary users will query** — "authentication", "pricing", "supported formats".
- **Avoid large code blocks in docs** used for Q&A — they pollute retrieved snippets.
- **Re-ingest after edits:** `POST /corpora/ingest` (no server restart needed).

---

## 11) Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `method: none`, generic "could not find answer" but `source` is set | `CHAT_ALLOW_FALLBACK=false` | Set `true` in `.env` and restart |
| `404 Corpus not found` | No ingest/load yet | `POST /corpora/ingest` or `/corpora/load` |
| Answers include curl/code noise | README-style docs rank high | Move how-to content out of Q&A docs |
| CORS error from browser | Origin not allowed | Add origin to `CORS_ALLOW_ORIGINS` |
| Env change has no effect | Server not restarted | Restart uvicorn/container |
| PowerShell `curl` fails | `curl` is `Invoke-WebRequest` alias | Use `Invoke-RestMethod` or `curl.exe` |

---

## 12) Security guidance

- Never commit `HF_API_TOKEN` or `.env` to git.
- Restrict `CORS_ALLOW_ORIGINS` to known frontends.
- Load artifact URLs only from trusted storage.
- This service has **no built-in API authentication** — place behind a reverse proxy or API gateway for production, or restrict network access.
- Chat logging is in-memory only; do not rely on it for compliance audit trails.

---

## 13) What is intentionally not in this repo

- Supabase / Postgres chat persistence (removed; in-memory optional)
- Render or other PaaS deploy manifests (deploy generically via Docker)
- Client SDK (apps use HTTP directly)
- PDF/HTML crawlers (apps preprocess content)
- API key middleware (add at gateway or in a future version)

---

## 14) Checklist for AI agents making changes

- [ ] Keep zero-cost defaults: `lightweight_nlp`, `bm25_hashed_vector`, `HF_API_ENABLED=false`
- [ ] Update `tests/test_api_contracts.py` if API shapes change
- [ ] Update this file if integration contract changes
- [ ] Do not reintroduce hardcoded portfolio-specific answer text
- [ ] Preserve `corpus_id` multi-tenant pattern
- [ ] Run full unittest suite before finishing
