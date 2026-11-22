# syntax=docker/dockerfile:1
# v2 image: the FastAPI support agent on CPU. MOCK_LLM=true by default (no keys, no network at runtime).
# Build: docker build -t nykaa-support .     Run everything: docker compose up   (see docker-compose.yml)

# --- build: dependencies from the Linux lock (CPU-only torch) + the pinned embedding model --------------------
FROM python:3.12-slim-bookworm AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_DEFAULT_TIMEOUT=120 PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv
WORKDIR /app
COPY requirements.lock.linux.txt .
# The cache mount stays out of the image; it only spares re-downloads on rebuilds (and survives flaky networks).
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 10 -r requirements.lock.linux.txt
# Only what the download imports, so code changes elsewhere in app/ don't invalidate the cached model layer.
COPY app/core/config.py app/core/
COPY app/rag/embeddings.py app/rag/
COPY scripts/download_model.py scripts/
# The only build step that needs the internet; the revision is pinned in the script.
RUN python scripts/download_model.py

# --- runtime ---------------------------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH MOCK_LLM=true
# A deployed image is a public demo: per-client question limits are ON (0 turns one off; see app/api/rate_limit.py).
ENV RATE_LIMIT_PER_MINUTE=3 RATE_LIMIT_PER_DAY=15 RATE_LIMIT_SITE_PER_DAY=70
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=build /app/models models
COPY alembic.ini .
COPY migrations migrations
COPY data data
COPY scripts scripts
COPY app app
# Runtime-writable paths only (request log, CrewAI storage); the code stays root-owned and read-only.
RUN mkdir logs .crewai_storage chroma_db && chown app:app logs .crewai_storage chroma_db
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
