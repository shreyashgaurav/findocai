"""ChromaDB vector store wrapper.

Two collections because the two modalities have different embedding
dimensions and can never share an index:

* ``text_chunks``  -> 384-dim SBERT vectors
* ``images``       -> 512-dim CLIP vectors

Both use cosine distance. A single persistent client backs both, so the whole
store survives restarts under ``chroma_persist_dir``.
"""
from typing import Any, Dict, List

import chromadb

from app.config import get_settings


class VectorStore:
    def __init__(self) -> None:
        s = get_settings()
        self.client = chromadb.PersistentClient(path=s.chroma_persist_dir)
        self.text = self.client.get_or_create_collection(
            name=s.text_collection, metadata={"hnsw:space": "cosine"}
        )
        self.images = self.client.get_or_create_collection(
            name=s.image_collection, metadata={"hnsw:space": "cosine"}
        )

    # --- writes ------------------------------------------------------------
    def add_text(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self.text.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def add_images(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self.images.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    # --- reads -------------------------------------------------------------
    def query_text(self, embedding: List[float], k: int) -> Dict[str, Any]:
        return self.text.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

    def query_images(self, embedding: List[float], k: int) -> Dict[str, Any]:
        return self.images.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

    # --- introspection -----------------------------------------------------
    def stats(self) -> Dict[str, int]:
        return {"text_chunks": self.text.count(), "images": self.images.count()}


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Process-wide singleton (Chroma's persistent client is not cheap)."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
