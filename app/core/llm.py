"""GPT-4o answer generation.

The retrieved context (text chunks + the actual retrieved images, base64) is
assembled into a single multimodal user message. The system prompt forces the
model to answer *only* from that context and to cite sources with the tags we
hand it ([T1], [I2], ...). Because GPT-4o is natively multimodal we pass the
images themselves, not just filenames, which produces genuinely grounded
multimodal answers.
"""
import base64
import mimetypes
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from app.config import get_settings

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question using "
    "ONLY the provided context, which consists of text excerpts (tagged [T1], "
    "[T2], ...) and images (tagged [I1], [I2], ...). Rules:\n"
    "1. Do not use prior knowledge. If the answer is not supported by the "
    "context, reply that you don't have enough information.\n"
    "2. Cite the specific sources you used inline, e.g. 'The revenue grew "
    "[T2], as the chart shows [I1].'\n"
    "3. Be concise and factual. Never invent citations."
)


def _encode_image(path: str) -> Tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")
    return mime, b64


def build_user_content(
    query: str,
    text_results: List[Dict[str, Any]],
    image_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assemble the multimodal user message content blocks."""
    parts: List[Dict[str, Any]] = []

    header = [f"Question: {query}", "", "=== Text context ==="]
    if not text_results:
        header.append("(no relevant text found)")
    for i, r in enumerate(text_results, 1):
        meta = r["metadata"]
        page = meta.get("page")
        loc = f", page {page}" if page and page != -1 else ""
        header.append(f"[T{i}] (source: {meta.get('source')}{loc})\n{r['content']}")
    parts.append({"type": "text", "text": "\n\n".join(header)})

    if image_results:
        parts.append({"type": "text", "text": "=== Image context ==="})
        for i, r in enumerate(image_results, 1):
            meta = r["metadata"]
            parts.append(
                {"type": "text", "text": f"[I{i}] (source: {meta.get('source')})"}
            )
            mime, b64 = _encode_image(meta["stored_path"])
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
    return parts


def generate_answer(
    query: str,
    text_results: List[Dict[str, Any]],
    image_results: List[Dict[str, Any]],
) -> str:
    s = get_settings()
    # client = OpenAI(api_key=s.openai_api_key)
    client = OpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url or None)
    resp = client.chat.completions.create(
        model=s.openai_model,
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(query, text_results, image_results)},
        ],
    )
    return resp.choices[0].message.content or ""
