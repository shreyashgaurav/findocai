import logging

import httpx
from fastapi import FastAPI, HTTPException
from google.genai import errors
from pydantic import BaseModel, Field

from findocai.generate import ask

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinDocAI",
    description="Answer questions about SEC fillings and earning decks, with page level citation",
)


class QueryRequest(BaseModel):
    # Filter out empty and oversized question
    question: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    tag: str
    doc_key: str
    page: int


class QueryResponse(BaseModel):
    answered: bool
    answer: str
    citations: list[Citation]
    model_version: str | None


@app.get("/health")
def health():
    # Live cjeck
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)  # ask() should return this shape
def query(request: QueryRequest):
    # Answer a ques. from the indexed doc, qith page citation
    try:
        return ask(request.question)
    except (errors.APIError, httpx.TransportError) as exc:
        logger.exception("Gemini call failed")
        raise HTTPException(
            status_code=502,
            detail="The language model is unavailable right now. Try agin later..",
        ) from exc
