# Zero-Cost Portfolio Chatbot Context (Handoff Document)

## Purpose of this document

This file captures the decisions and guidance from our discussion so it can be pasted into a new repository (or a new AI chat) as setup context.

Primary goal: build a chatbot for portfolio documents with **zero ongoing AI token cost** and website integration support.

---

## Conversation summary (decision log)

1. Initial idea was a RAG-powered chatbot over personal documents.
2. Concern raised: cloud RAG consumes tokens and costs money.
3. We evaluated alternatives (NLP-only, retrieval-only, local LLM, hybrid).
4. Final direction chosen for phase 1:
   - **Zero-cost path**
   - **No cloud LLM inference**
   - Use **retrieval/search-based chatbot** that returns relevant snippets and sources.

---

## Core constraints and requirements

- Must be deployable and integratable into a portfolio website.
- Prefer separate repo for chatbot component.
- Keep ongoing cost at or near zero.
- Avoid always-on paid server where possible.
- Answers should be grounded in personal documents.

---

## What was clarified about local LLMs

- A local LLM avoids token billing but still needs compute.
- For public availability, some machine must be running when users ask questions.
- Therefore, local LLM is not truly "free always-on" unless you accept:
  - your own machine being online continuously, or
  - paying for an always-on server.

This is why retrieval-only is a strong phase-1 solution for zero-cost deployment.

---

## Recommended phase-1 architecture (zero-cost)

## Approach

Build a **retrieval-first chatbot** that does not generate new text with an LLM.

Query flow:

1. User asks a question in chat widget.
2. App performs search over indexed portfolio documents.
3. App returns top relevant snippets with document/source metadata.
4. UI presents snippets in a chat-like format with links/references.

## Retrieval methods

Use one or both:

- **Keyword retrieval**: BM25 / TF-IDF / inverted index.
- **Semantic retrieval (optional local embedding)**:
  - precompute embeddings offline (local free model)
  - store vectors in static JSON or lightweight local DB at build time.

For strict zero-cost and easy deployment, start with keyword retrieval + simple fuzzy matching.

---

## Why this is the best phase-1 tradeoff

Pros:

- No API token spend.
- No dependency on cloud LLM provider.
- Can run as static app or serverless free tier.
- Reliable, grounded, and deterministic.

Limitations:

- Answers are extractive (less conversational polish).
- No deep synthesis unless added later.

Mitigation:

- Improve ranking quality.
- Return multiple snippets.
- Add optional "polish answer" mode later (paid or local).

---

## Suggested tech stack (free-friendly)

## Backend choice A: fully static/client-side

- Index generated at build time into JSON files.
- Frontend performs retrieval in browser.
- Deploy to GitHub Pages / Netlify / Vercel free tier.
- Best for zero infra cost.

## Backend choice B: lightweight API (free tier)

- Small API (FastAPI/Node) does retrieval only.
- Useful when index is larger or you want private processing.
- Still no LLM calls.

## Data/index options

- `lunr` / `minisearch` (JavaScript indexing/search)
- TF-IDF with custom scoring
- Optional FAISS/Qdrant local during build, export compact artifacts for runtime

---

## Repository blueprint for the new project

Suggested structure:

```text
portfolio-chatbot/
  data/
    raw/                 # source docs copied/synced from portfolio content
    processed/           # cleaned chunks
    index/               # generated search index artifacts
  scripts/
    ingest.py            # parse + normalize docs
    chunk.py             # chunk text with metadata
    build_index.py       # build BM25/TF-IDF/index files
    eval.py              # quality checks on test questions
  web/
    src/
      components/
        ChatWidget.tsx
        SourceCard.tsx
      lib/
        search.ts
        rank.ts
      data/              # runtime index artifacts if bundled
    public/
  tests/
    retrieval_cases.json
  docs/
    architecture.md
    deployment.md
  README.md
```

---

## Ingestion and indexing design

## Document parsing

- Parse markdown and text files first.
- Keep metadata:
  - `doc_id`
  - `title`
  - `section`
  - `url` or source path
  - `updated_at`

## Chunking strategy

- Chunk size: 300-800 words (or token-equivalent).
- Overlap: 10-20%.
- Preserve headings for context.

## Ranking

Baseline:

- BM25/TF-IDF score
- simple field boosting (title > headings > body)

Optional upgrades:

- fuzzy spell correction
- synonym expansion
- reranking by query-title cosine similarity

---

## Frontend chatbot behavior (retrieval-only UX)

- Present results as:
  - short extracted snippet
  - source title/link
  - relevance hint (optional)
- If confidence low:
  - state that exact answer not found
  - show closest related sections
- Provide quick suggestion chips:
  - "Projects"
  - "Skills"
  - "Experience"
  - "Contact"

---

## Evaluation plan (important)

Create a small benchmark set before launch:

- 30-50 realistic user questions
- expected document or section for each question

Track:

- Top-1 hit rate
- Top-3 hit rate
- failure patterns (missing synonyms, chunk too large/small, etc.)

This ensures retrieval quality improves systematically.

---

## Deployment options with zero cost

## Option 1: static hosting (recommended)

- GitHub Pages / Netlify / Vercel free tier
- build index during CI
- ship index artifacts with app

## Option 2: free serverless retrieval API

- Vercel/Netlify functions
- keep retrieval logic server-side if needed
- still no LLM API calls

---

## Integration with your portfolio website

- Embed as:
  - floating chat button + modal, or
  - dedicated `/chat` page.
- Expose config:
  - theme colors
  - greeting message
  - index version
  - source link behavior
- Keep this as a modular package/component for easy reuse across portfolio versions.

---

## Future upgrade path (phase 2 and beyond)

When/if you allow limited compute spend:

1. Add optional local LLM summary mode (self-hosted).
2. Or add paid API mode behind feature flag.
3. Keep retrieval as mandatory grounding layer.

Rule: retrieval stays first; generation is optional enhancement.

---

## Copy-paste context prompt for future AI setup

Use this in the next repo/chat:

```text
I am building a portfolio chatbot with zero ongoing AI token cost.
Phase 1 must avoid cloud LLM calls entirely.

Requirements:
- Chatbot answers should be grounded in my own documents.
- Use retrieval/search only (BM25/TF-IDF/fuzzy/optional local embeddings).
- Return top relevant snippets + source links in chat UI.
- Must be easy to embed into portfolio website.
- Prefer static or free-tier deployment with no always-on paid server.

Please generate:
1) repo scaffolding
2) ingestion + chunking + index scripts
3) retrieval API or client-side search module
4) chat widget integration
5) evaluation benchmark and tests
6) deployment instructions for free hosting
```

---

## Final recommendation

For your current constraints, the best route is:

1. Build a strong retrieval-only chatbot first (free, reliable).
2. Validate usefulness with real questions.
3. Add generation later only if clearly needed.

This gives maximum practical value now with minimal cost and complexity.
