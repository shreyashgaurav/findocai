"""Image handling: validate, persist a canonical copy, and load for embedding.

Originals are copied into ``storage_dir`` under a content-stable name so the
retrieval layer can re-open them to (a) cite the source and (b) pass the bytes
to GPT-4o.
"""
import os
import shutil
import uuid

from PIL import Image

from app.config import get_settings


def is_supported_image(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in get_settings().allowed_image_ext


def persist_image(src_path: str, original_name: str) -> str:
    """Copy the uploaded image into storage and return the stored path."""
    s = get_settings()
    os.makedirs(s.storage_dir, exist_ok=True)
    ext = os.path.splitext(original_name)[1].lower() or ".png"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = os.path.join(s.storage_dir, stored_name)
    shutil.copyfile(src_path, stored_path)
    return stored_path


def load_image(path: str) -> Image.Image:
    """Open and normalise an image to RGB (CLIP expects 3 channels)."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
