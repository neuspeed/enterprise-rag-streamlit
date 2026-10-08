FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# poppler backs pdf2image, which renders PDF pages for the vision model.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so a source-only change does not invalidate the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY know_everything_ai ./know_everything_ai
# Shipped as code, not as a runtime artefact: the worker applies them itself at
# startup, so an image without them would come up healthy against an empty
# database and only fail when the first message arrives.
COPY migrations ./migrations
COPY pyproject.toml README.md ./

# Unprivileged runtime user: the worker fetches remote documents and writes
# scratch files, and does not need to write anywhere else.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/logs /app/.cache/fastembed \
    && chown -R appuser:appuser /app
USER appuser

# Sandbox dashboard. Streamlit is baked in here rather than installed at
# container start, so recreating this service after a .env change is instant
# instead of a ten-minute download. It is not in requirements.txt because the
# worker image has no use for a web framework.
FROM base AS ui
USER root
RUN pip install --no-cache-dir "streamlit>=1.40"
USER appuser
CMD ["python", "-m", "streamlit", "run", "know_everything_ai/ui/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]

# Read-only query API. FastAPI is baked in for the same reason Streamlit is
# above: a container restart after a .env change should not become a package
# download. Neither is in requirements.txt, because the worker never opens a
# socket and has no use for a web framework.
FROM base AS api
USER root
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.30"
USER appuser
EXPOSE 8000
CMD ["python", "-m", "know_everything_ai", "api"]

# Production worker. Declared last so a bare `docker build .` still yields it.
FROM base AS runtime
CMD ["python", "-m", "know_everything_ai"]