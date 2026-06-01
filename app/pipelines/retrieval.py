"""Retrieval pipeline.

A single natural-language query fans out to both collections:

* SBERT-embed the query  -> search ``text_chunks``
* CLIP-text-embed query  -> search ``images``  (cross-modal)

Chroma returns cosine *distance* (0 = identical). We convert to a similarity
score (1 - distance) for a more intuitive citation display and flatten the
nested result shape into plain dicts.
"""
from typing import Any, Dict, List, Tuple

from app.core.embeddings import get_clip_embedder, get_text_embedder
from app.core.vector_store import get_vector_store


def _flatten(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chroma returns lists-of-lists (one inner list per query)."""
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    out: List[Dict[str, Any]] = []
    for content, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "content": content,
                "metadata": meta,
                "distance": dist,
                "score": round(1.0 - dist, 4),
            }
        )
    return out


def retrieve(query: str, top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    vs = get_vector_store()

    text_embedding = get_text_embedder().embed_one(query)
    clip_embedding = get_clip_embedder().embed_text_one(query)

    text_results = _flatten(vs.query_text(text_embedding, top_k))
    image_results = _flatten(vs.query_images(clip_embedding, top_k))
    return text_results, image_results
