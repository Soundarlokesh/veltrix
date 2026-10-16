FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/cache/tiktoken \
    HF_HOME=/opt/cache/huggingface

# OCR engine for scanned PDFs (PyMuPDF renders pages, Tesseract reads them).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small; sentence-transformers needs it for embeddings.
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Cache the tokenizer and the embedding model at build time so the container
# does not need to download them at runtime.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Runtime must not contact Hugging Face: the model is already cached above (verified offline).
# Set after the download step so the build itself can still fetch the model.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app ./app
COPY pytest.ini ./

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data \
    && chown appuser /data
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
