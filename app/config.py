"""Application configuration.

All tunables live here and are overridable via environment variables or a
local ``.env`` file. ``get_settings()`` is cached so the object is built once
per process.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---------------------------------------------------------------
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = ""
    llm_temperature: float = 0.2
    llm_max_tokens: int = 800

    # --- Embedding models --------------------------------------------------
    # SBERT handles document text (384-dim). CLIP handles images AND encodes
    # text queries into the same space for cross-modal retrieval (512-dim).
    text_embedding_model: str = "all-MiniLM-L6-v2"
    clip_model: str = "clip-ViT-B-32"

    # --- Vector store ------------------------------------------------------
    chroma_persist_dir: str = "./data/chroma"
    text_collection: str = "text_chunks"
    image_collection: str = "images"

    # --- Storage -----------------------------------------------------------
    # Original uploaded images are kept on disk so we can re-serve them to
    # GPT-4o (which is natively multimodal) and cite them in responses.
    storage_dir: str = "./data/storage"

    # --- Chunking ----------------------------------------------------------
    chunk_size: int = 800        # characters per chunk
    chunk_overlap: int = 150     # characters shared between adjacent chunks

    # --- Retrieval ---------------------------------------------------------
    default_top_k: int = 5
    max_top_k: int = 20

    # --- Upload limits -----------------------------------------------------
    allowed_text_ext: tuple[str, ...] = (".pdf", ".txt", ".md")
    allowed_image_ext: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


@lru_cache
def get_settings() -> Settings:
    return Settings()
