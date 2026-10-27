# ============================================================
# Multi-Stage Production Dockerfile — SummarizeAI
# ============================================================
# Build:  docker build -t summarize-ai:latest .
# Run:    docker run -p 8000:8000 --env-file .env summarize-ai:latest
# ============================================================

FROM python:3.11-slim AS base

# --- Security: Create non-root application user ---
RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid appgroup --shell /bin/false --create-home appuser

# --- System dependencies (minimal) ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# --- Application working directory ---
WORKDIR /app

# --- Layer caching: Install Python deps before copying app code ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Copy application code ---
COPY src/ src/
COPY config/ config/
COPY templates/ templates/
COPY static/ static/
COPY app.py main.py setup.py params.yaml ./

# --- Copy model artifacts if present (optional, can be volume-mounted) ---
COPY artifacts/ artifacts/

# --- Set ownership to non-root user ---
RUN chown -R appuser:appgroup /app

# --- Run as non-root user ---
USER appuser

# --- Environment defaults ---
ENV APP_ENV=production \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# --- Built-in liveness health check ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# --- Entrypoint ---
CMD ["python", "app.py"]
