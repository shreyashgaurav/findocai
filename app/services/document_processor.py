"""Document text extraction and chunking.

Supports PDF (per-page text via pypdf) and plain text / markdown. Chunking is
a simple, predictable character sliding window with overlap so that facts
spanning a boundary still appear whole in at least one chunk.
"""
from typing import Dict, List, Optional

from pypdf import PdfReader


def extract_text_from_pdf(path: str) -> List[Dict[str, Optional[str]]]:
    """Return a list of {page, text} dicts (one per non-empty page)."""
    reader = PdfReader(path)
    pages: List[Dict[str, Optional[str]]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def extract_text_from_plaintext(path: str) -> List[Dict[str, Optional[str]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()
    return [{"page": None, "text": content}] if content.strip() else []


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Whitespace-normalised character sliding window with overlap."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = max(size - overlap, 1)
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


def extract_pages(path: str, extension: str) -> List[Dict[str, Optional[str]]]:
    ext = extension.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in (".txt", ".md"):
        return extract_text_from_plaintext(path)
    raise ValueError(f"Unsupported text format: {extension}")
