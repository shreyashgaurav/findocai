"""Embedding models.

Two embedders, two vector spaces:

* ``TextEmbedder`` (SBERT) -> document text chunks, 384-dim.
* ``ClipEmbedder``        -> images, AND text queries that must search the
  image collection. CLIP places images and text in a *shared* space, which is
  exactly what makes cross-modal ("find the image that matches this sentence")
  retrieval work, 512-dim.

The two spaces have different dimensions, so they must live in separate
ChromaDB collections (see ``vector_store.py``).

Models are loaded lazily and cached so the (expensive) load happens once.
"""
from functools import lru_cache
from typing import List

from PIL import Image
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class TextEmbedder:
    """SBERT embedder for document text."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # normalize_embeddings=True -> unit vectors, so cosine distance in
        # Chroma behaves well.
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class ClipEmbedder:
    """CLIP embedder. Encodes images and text into one shared space."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_images(self, images: List[Image.Image]) -> List[List[float]]:
        return self.model.encode(
            images, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_text_one(self, text: str) -> List[float]:
        return self.embed_text([text])[0]


@lru_cache
def get_text_embedder() -> TextEmbedder:
    return TextEmbedder(get_settings().text_embedding_model)


@lru_cache
def get_clip_embedder() -> ClipEmbedder:
    return ClipEmbedder(get_settings().clip_model)
