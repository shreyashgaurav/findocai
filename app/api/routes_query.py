"""Query endpoints."""
from fastapi import APIRouter

from app.config import get_settings
from app.core.llm import generate_answer
from app.core.vector_store import get_vector_store
from app.models.schemas import (
    ImageCitation,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    TextCitation,
)
from app.pipelines.retrieval import retrieve

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    s = get_settings()
    top_k = min(req.top_k or s.default_top_k, s.max_top_k)

    text_results, image_results = retrieve(req.query, top_k)
    answer = generate_answer(req.query, text_results, image_results)

    text_citations = [
        TextCitation(
            tag=f"T{i}",
            source=r["metadata"].get("source"),
            page=(None if r["metadata"].get("page") in (None, -1) else r["metadata"].get("page")),
            chunk_index=r["metadata"].get("chunk_index"),
            score=r["score"],
            snippet=(r["content"][:240] + ("..." if len(r["content"]) > 240 else "")),
        )
        for i, r in enumerate(text_results, 1)
    ]
    image_citations = [
        ImageCitation(
            tag=f"I{i}",
            source=r["metadata"].get("source"),
            stored_path=r["metadata"].get("stored_path"),
            score=r["score"],
        )
        for i, r in enumerate(image_results, 1)
    ]

    return QueryResponse(
        query=req.query,
        answer=answer,
        text_citations=text_citations,
        image_citations=image_citations,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    return StatsResponse(**get_vector_store().stats())
