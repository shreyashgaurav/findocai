# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps: libgl/glib are needed by Pillow/torch image stacks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only torch first to keep the image lean, then the rest.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Pre-download the embedding models at build time so the first request is fast
# and the container can run without network access to model hubs.
RUN python -c "from sentence_transformers import SentenceTransformer as S; \
    S('all-MiniLM-L6-v2'); S('clip-ViT-B-32')"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
