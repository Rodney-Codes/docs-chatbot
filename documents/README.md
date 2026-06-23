# Local test documents

Add `.md`, `.markdown`, or `.txt` files here for local testing. This folder is gitignored except for this README.

On startup (when `CHATBOT_AUTO_INGEST=true`), the service ingests these files into the default corpus.

## Example

Create `documents/faq.md`:

```markdown
# FAQ

## What is this service?

A zero-cost retrieval chatbot that answers from your documents.

## Supported formats

Markdown and plain text files in this directory.
```

Then run the service and ask:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "what formats are supported", "corpus_id": "default"}'
```

## Re-ingest after changes

```bash
curl -X POST http://localhost:8000/corpora/ingest \
  -H "Content-Type: application/json" \
  -d '{"corpus_id": "default"}'
```
