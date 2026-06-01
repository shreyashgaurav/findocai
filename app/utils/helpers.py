"""Small filesystem helpers."""
import os
import tempfile
from typing import BinaryIO


def save_to_temp(file_obj: BinaryIO, suffix: str = "") -> str:
    """Stream an uploaded file object to a temp file and return its path."""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return tmp_path
