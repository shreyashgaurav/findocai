"""Ingestion pipeline.

Routes an uploaded file to the right processor, embeds the content with the
correct model, and writes vectors + metadata into the matching ChromaDB
collection. Returns a small summary used by the API response.

Metadata schema
---------------
Text chunk:  {source, doc_id, type="text", page, chunk_index}
Image:       {source, doc_id, type="image", stored_path}
"""
import os
import uuid
from typing import Any, Dict

from app.config import get_settings
from app.core.embeddings import get_clip_embedder, get_text_embedder
from app.core.vector_store import get_vector_store
from app.services.document_processor import chunk_text, extract_pages
from app.services.image_processor import load_image, persist_image


def ingest_text_document(tmp_path: str, filename: str) -> Dict[str, Any]:
    s = get_settings()
    ext = os.path.splitext(filename)[1].lower()
    doc_id = uuid.uuid4().hex

    pages = extract_pages(tmp_path, ext)
    ids, documents, metadatas = [], [], []
    for page in pages:
        chunks = chunk_text(page["text"], s.chunk_size, s.chunk_overlap)
        for idx, chunk in enumerate(chunks):
            ids.append(uuid.uuid4().hex)
            documents.append(chunk)
            metadatas.append(
                {
                    "source": filename,
                    "doc_id": doc_id,
                    "type": "text",
                    "page": page["page"] if page["page"] is not None else -1,
                    "chunk_index": idx,
                }
            )

    if not ids:
        return {"doc_id": doc_id, "type": "text", "chunks_indexed": 0,
                "note": "No extractable text (scanned PDF? consider OCR)."}

    embeddings = get_text_embedder().embed(documents)
    get_vector_store().add_text(ids, embeddings, documents, metadatas)
    return {"doc_id": doc_id, "type": "text", "chunks_indexed": len(ids)}


def ingest_image(tmp_path: str, filename: str) -> Dict[str, Any]:
    doc_id = uuid.uuid4().hex
    stored_path = persist_image(tmp_path, filename)

    image = load_image(stored_path)
    embedding = get_clip_embedder().embed_images([image])[0]

    record_id = uuid.uuid4().hex
    get_vector_store().add_images(
        ids=[record_id],
        embeddings=[embedding],
        documents=[filename],  # human-readable handle for the vector
        metadatas=[
            {
                "source": filename,
                "doc_id": doc_id,
                "type": "image",
                "stored_path": stored_path,
            }
        ],
    )
    return {"doc_id": doc_id, "type": "image", "images_indexed": 1,
            "stored_path": stored_path}


def ingest_file(tmp_path: str, filename: str) -> Dict[str, Any]:
    s = get_settings()
    ext = os.path.splitext(filename)[1].lower()
    if ext in s.allowed_text_ext:
        return ingest_text_document(tmp_path, filename)
    if ext in s.allowed_image_ext:
        return ingest_image(tmp_path, filename)
    raise ValueError(
        f"Unsupported file type '{ext}'. "
        f"Allowed: {s.allowed_text_ext + s.allowed_image_ext}"
    )
