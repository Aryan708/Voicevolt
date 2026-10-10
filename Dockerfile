# VoiceVault - CPU image. Build:  docker build -t voicevault .
# Add LibreOffice for .pptx/.docx ingestion:  docker build --build-arg WITH_LIBREOFFICE=1 -t voicevault .
FROM python:3.11-slim

ARG WITH_LIBREOFFICE=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf_cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && if [ "$WITH_LIBREOFFICE" = "1" ]; then \
      apt-get install -y --no-install-recommends libreoffice-impress libreoffice-writer; fi \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch first: keeps the image ~2 GB smaller than the default CUDA wheels
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY voicevault ./voicevault
COPY scripts ./scripts
COPY finetune ./finetune

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "voicevault.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
