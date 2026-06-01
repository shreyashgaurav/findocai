# Multimodal RAG System

A retrieval-augmented generation service that answers natural-language questions using evidence drawn from **both text documents and images**. Text is embedded with Sentence-BERT, images with CLIP, both are stored in ChromaDB, and **GPT-4o** generates answers grounded only in what was retrieved — with source citations for each modality.

---

## 1. Architecture

```
                          INGESTION (write path)
                            POST /upload
                                 │
              ┌──────────────────┴──────────────────┐
        Document pipeline                      Image pipeline
      (extract text + chunk)                 (load + validate)
              │                                      │
        SBERT embedder                         CLIP embedder
      (MiniLM · 384-d)                        (ViT-B/32 · 512-d)
              │                                      │
              ▼                                      ▼
        ┌──────────────────── ChromaDB (persistent) ────────────────────┐
        │   text_chunks  (384-d, cosine)        images (512-d, cosine)   │
        └───────────────────────────▲──────────────────────────────────┘
                                     │ top-k search (both collections)
                          embed + search ◄── POST /query
                                     │
                                  GPT-4o ──► answer + citations
                          RETRIEVAL (read path)
```

Two facts drive the whole design:

1. **SBERT (384-d) and CLIP (512-d) produce different-dimensional vectors**, so they live in **two separate ChromaDB collections**. A query searches both and the results are merged.
2. **Cross-modal retrieval uses CLIP's text encoder.** To find images relevant to a text query, the query is embedded with CLIP's text tower (shared image–text space) — not SBERT. SBERT only ever touches document text.

GPT-4o is natively multimodal, so retrieved images are passed to it directly (base64), not as filenames — the model reasons over the actual pixels alongside the text excerpts.

---

## 2. Project structure

```
multimodal-rag/
├── app/
│   ├── main.py                  # FastAPI app, routers, startup
│   ├── config.py                # Pydantic settings (env-driven)
│   ├── api/
│   │   ├── routes_upload.py      # POST /upload
│   │   └── routes_query.py       # POST /query, GET /stats
│   ├── core/
│   │   ├── embeddings.py         # SBERT + CLIP embedders (cached)
│   │   ├── vector_store.py       # ChromaDB wrapper (2 collections)
│   │   └── llm.py                # GPT-4o prompt + generation
│   ├── pipelines/
│   │   ├── ingestion.py          # route → process → embed → store
│   │   └── retrieval.py          # dual-embed → search both → flatten
│   ├── services/
│   │   ├── document_processor.py # PDF/txt extraction + chunking
│   │   └── image_processor.py    # image load/validate/persist
│   ├── models/
│   │   └── schemas.py            # Pydantic request/response models
│   └── utils/
│       └── helpers.py            # temp-file streaming
├── tests/test_basic.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

The layering is deliberate: **api** (HTTP) → **pipelines** (orchestration) → **services / core** (single-responsibility units). Each layer only depends downward, which keeps things testable and swappable (e.g. replace ChromaDB without touching the API).

---

## 3. Database & metadata design

Two collections, both cosine distance:

| Collection    | Vector source     | Dim | `document` field        |
|---------------|-------------------|-----|-------------------------|
| `text_chunks` | SBERT             | 384 | the chunk text          |
| `images`      | CLIP (image)      | 512 | original filename       |

Metadata is what powers **source tracking and citations**:

```jsonc
// text chunk
{ "source": "report.pdf", "doc_id": "ab12…", "type": "text",
  "page": 4, "chunk_index": 2 }

// image
{ "source": "chart.png", "doc_id": "cd34…", "type": "image",
  "stored_path": "data/storage/9f0e….png" }
```

`stored_path` matters: the original image is copied into `STORAGE_DIR` at ingest time so retrieval can re-open it to feed GPT-4o and to cite it. `doc_id` groups all chunks/images from one upload, enabling later document-level deletion.

---

## 4. Ingestion pipeline

`POST /upload` → `ingest_file()` routes by extension:

- **PDF / txt / md** → `document_processor` extracts text (per page for PDFs) → `chunk_text` splits it into overlapping 800-char windows → SBERT embeds each chunk → written to `text_chunks` with page + chunk-index metadata.
- **Images** → copied to persistent storage → loaded as RGB → CLIP embeds the image → written to `images` with `stored_path`.

Overlapping chunks (150 chars) ensure a fact straddling a boundary still appears intact in at least one chunk. Scanned PDFs with no extractable text return a note suggesting OCR (see improvements).

---

## 5. Retrieval pipeline

`retrieve(query, top_k)`:

1. SBERT-embed the query → top-k from `text_chunks`.
2. CLIP-text-embed the query → top-k from `images`.
3. Flatten Chroma's nested response and convert cosine distance to a `score = 1 - distance` for readable citations.

The two result sets are returned separately so the API can cite each modality distinctly.

---

## 6. GPT-4o integration

`llm.generate_answer()` builds one multimodal user message: text excerpts tagged `[T1], [T2]…`, then each retrieved image as a base64 `image_url` block tagged `[I1], [I2]…`. The system prompt enforces three rules — **answer only from context, cite the tags used, never invent citations** — and falls back to "not enough information" when the context doesn't support an answer. Temperature is low (0.2) for factual grounding.

---

## 7. API design

| Method | Path      | Body / Params                     | Returns                              |
|--------|-----------|-----------------------------------|--------------------------------------|
| POST   | `/upload` | multipart `files[]`               | per-file ingest summary              |
| POST   | `/query`  | `{ "query": str, "top_k"?: int }` | answer + text & image citations      |
| GET    | `/stats`  | —                                 | vector counts per collection         |
| GET    | `/health` | —                                 | `{ "status": "ok" }`                 |

Interactive docs are auto-generated at `/docs`. Example query response:

```jsonc
{
  "query": "How did Q3 revenue compare to Q2?",
  "answer": "Revenue rose 12% quarter-over-quarter [T1], which the bar chart confirms [I1].",
  "text_citations": [
    { "tag": "T1", "source": "report.pdf", "page": 4, "score": 0.83, "snippet": "Q3 revenue of $4.1M…" }
  ],
  "image_citations": [
    { "tag": "I1", "source": "revenue_chart.png", "stored_path": "data/storage/9f0e.png", "score": 0.71 }
  ]
}
```

---

## 8. Step-by-step setup (local)

```bash
# 1. Clone & enter
cd multimodal-rag

# 2. Virtual env + deps
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU build
pip install -r requirements.txt

# 3. Configure
cp .env.example .env          # then set OPENAI_API_KEY

# 4. Run (first start downloads the embedding models, ~1–2 min)
uvicorn app.main:app --reload

# 5. Try it
curl -F "files=@report.pdf" -F "files=@chart.png" http://localhost:8000/upload
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query":"What does the chart show about revenue?","top_k":5}'
```

Run tests with `pytest -q` (the included tests are pure-logic and need no API key or model download).

---

## 9. Production deployment

```bash
# One-command stack with persistent volumes for the DB, images, and model cache
export OPENAI_API_KEY=sk-...
docker compose up --build
```

The Dockerfile installs CPU-only PyTorch and **pre-downloads both embedding models at build time**, so containers start fast and need no model-hub access at runtime. Volumes (`rag_data`, `hf_cache`) persist ChromaDB, stored images, and model weights across restarts.

Production notes:
- Put the service behind a reverse proxy (nginx / Traefik) with TLS.
- Scale horizontally by running multiple API replicas against a **shared Chroma server** (`chromadb.HttpClient`) instead of the embedded persistent client used here.
- Run embedding on GPU by installing a CUDA PyTorch build and a GPU base image.
- Inject `OPENAI_API_KEY` via your platform's secret manager, never the image.
- Add request timeouts, structured logging, and a `/metrics` endpoint for observability.

---

## 10. Best practices & possible improvements

Implemented: modular layering, env-driven config, cached model loads, singleton vector store, overlapping chunks, normalized embeddings, citation-enforcing prompt, persistent volumes, pinned dependencies, pure-logic tests.

Worth adding for a more complete system:
- **OCR fallback** (Tesseract / `unstructured`) for scanned PDFs that yield no text.
- **PDF figure extraction** so embedded charts/photos become CLIP-indexed images, not just text.
- **Hybrid retrieval + reranking**: combine dense search with BM25 and a cross-encoder reranker before sending context to GPT-4o.
- **Image captioning at ingest** (BLIP) to also index a textual description of each image in `text_chunks`, improving recall for keyword-style queries.
- **Document management**: `DELETE /documents/{doc_id}`, dedup by content hash, and metadata filters in queries.
- **Evaluation harness**: a small Q/A set scored for retrieval hit-rate and answer faithfulness (e.g. RAGAS) to catch regressions.
- **Auth & rate limiting** (API keys / OAuth) and per-user collections for multi-tenant use.
- **Async ingestion** via a task queue (Celery/RQ) for large uploads so requests return immediately.
```
