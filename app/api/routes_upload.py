"""Upload endpoints."""
import os
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import UploadResponse, UploadResult
from app.pipelines.ingestion import ingest_file
from app.utils.helpers import save_to_temp

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)) -> UploadResponse:
    """Upload PDFs, text files, and/or images. Each file is ingested into the
    appropriate modality and indexed in ChromaDB."""
    results: List[UploadResult] = []
    for f in files:
        suffix = os.path.splitext(f.filename or "")[1].lower()
        tmp_path = save_to_temp(f.file, suffix=suffix)
        try:
            summary = ingest_file(tmp_path, f.filename or f"upload{suffix}")
            results.append(UploadResult(filename=f.filename, **summary))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return UploadResponse(results=results)
