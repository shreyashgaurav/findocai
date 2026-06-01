"""Lightweight unit tests that don't require model downloads or an API key.

Run with:  pytest -q
"""
from app.pipelines.retrieval import _flatten
from app.services.document_processor import chunk_text


def test_chunking_overlap():
    text = "a " * 1000  # 2000 chars before normalisation
    chunks = chunk_text(text, size=800, overlap=150)
    assert len(chunks) > 1
    # adjacent chunks must share the overlap region
    assert chunks[0][-150:] == chunks[1][:150]


def test_chunking_short_text_single_chunk():
    assert chunk_text("hello world", size=800, overlap=150) == ["hello world"]


def test_chunking_empty():
    assert chunk_text("   ", size=800, overlap=150) == []


def test_flatten_shapes_and_scores():
    raw = {
        "documents": [["chunk one", "chunk two"]],
        "metadatas": [[{"source": "a.pdf"}, {"source": "b.pdf"}]],
        "distances": [[0.1, 0.4]],
    }
    out = _flatten(raw)
    assert len(out) == 2
    assert out[0]["content"] == "chunk one"
    assert out[0]["score"] == 0.9   # 1 - 0.1
    assert out[1]["metadata"]["source"] == "b.pdf"


def test_flatten_empty():
    assert _flatten({"documents": [[]], "metadatas": [[]], "distances": [[]]}) == []
