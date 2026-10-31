# syntax=docker/dockerfile:1

# ---- web UI ---------------------------------------------------------------
FROM node:20-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- API ------------------------------------------------------------------
FROM python:3.12-slim

# EMBEDDING_MODEL=none builds a smaller image that only uses BM25.
ARG EMBEDDING_MODEL=e5-small

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QANUN_ROOT=/app \
    QANUN_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app
RUN useradd --create-home --uid 1000 qanun

# CPU-only PyTorch keeps the image far smaller than the default CUDA build.
RUN if [ "$EMBEDDING_MODEL" != "none" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN if [ "$EMBEDDING_MODEL" != "none" ]; then pip install ".[embeddings]"; else pip install .; fi

COPY data/corpus ./data/corpus
COPY --from=web /web/dist ./web/dist

# Download the embedding model and build the vector index at build time so
# the container starts quickly and works offline.
RUN if [ "$EMBEDDING_MODEL" != "none" ]; then \
      qanun index --model "$EMBEDDING_MODEL" \
      && python -c "from qanun.retrieval.rerank import CrossEncoderScorer; CrossEncoderScorer('mmarco')"; \
    fi \
    && chown -R qanun:qanun /app

ENV HF_HUB_OFFLINE=1
USER qanun
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["qanun", "serve", "--host", "0.0.0.0", "--port", "8000"]
