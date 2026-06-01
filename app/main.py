"""FastAPI application entrypoint.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""
import os

from fastapi import FastAPI

from app.api.routes_query import router as query_router
from app.api.routes_upload import router as upload_router
from app.config import get_settings

app = FastAPI(
    title="Multimodal RAG API",
    description="Answer questions over text documents and images "
    "(SBERT + CLIP embeddings, ChromaDB retrieval, GPT-4o generation).",
    version="1.0.0",
)

app.include_router(upload_router)
app.include_router(query_router)


@app.on_event("startup")
def _ensure_dirs() -> None:
    s = get_settings()
    os.makedirs(s.chroma_persist_dir, exist_ok=True)
    os.makedirs(s.storage_dir, exist_ok=True)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}
