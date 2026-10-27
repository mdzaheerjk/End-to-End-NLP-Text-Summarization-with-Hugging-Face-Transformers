# 🐳 Docker Deployment Guide — SummarizeAI

> **Production-Grade Container Deployment for Text Summarization System**

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Build the Docker Image](#build-the-docker-image)
3. [Run the Container](#run-the-container)
4. [Resource-Constrained Execution](#resource-constrained-execution)
5. [Container Inspection & Debugging](#container-inspection--debugging)
6. [Multi-Container Deployment with Docker Compose](#multi-container-deployment-with-docker-compose)
7. [Container Registry Push](#container-registry-push)
8. [Security Checklist](#security-checklist)

---

## Prerequisites

- Docker Engine ≥ 24.0 (or Docker Desktop)
- `.env` file copied from `.env.example` and configured
- Model weights: Loads from `artifacts/` if mounted, or streams `google/pegasus-cnn_dailymail` from Hugging Face Hub during container startup pre-warming

---

## Build the Docker Image

```bash
# Standard build
docker build -t summarize-ai:latest .

# Build with build-time metadata
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  -t summarize-ai:1.0.0 \
  -t summarize-ai:latest .

# Verify image
docker images summarize-ai
docker inspect summarize-ai:latest | grep -A5 "User"
```

---

## Run the Container

```bash
# Basic run
docker run -d \
  --name summarize-ai \
  -p 8000:8000 \
  --env-file .env \
  summarize-ai:latest

# Verify health probe
curl -i http://localhost:8000/health
# Response: {"status":"healthy","service":"SummarizeAI","model_loaded":true}

# Verify hardware & model diagnostics
curl -s http://localhost:8000/diagnostics | python -m json.tool

# Smoke test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hannah: Hey, do you have Betty'\''s number?\nAmanda: Lemme check... Yup, here it is: 555-019-2834"}'
```

---

## Resource-Constrained Execution

```bash
# Production run with memory and CPU limits
docker run -d \
  --name summarize-ai-prod \
  -p 8000:8000 \
  --env-file .env \
  --memory="4g" \
  --memory-swap="4g" \
  --cpus="2.0" \
  --restart=unless-stopped \
  --read-only \
  --tmpfs /tmp \
  summarize-ai:latest

# GPU-enabled run (requires NVIDIA Container Toolkit)
docker run -d \
  --name summarize-ai-gpu \
  -p 8000:8000 \
  --env-file .env \
  --gpus all \
  --memory="8g" \
  --cpus="4.0" \
  summarize-ai:latest
```

---

## Container Inspection & Debugging

```bash
# View real-time logs
docker logs -f summarize-ai

# Tail last 100 lines with timestamps
docker logs --tail 100 --timestamps summarize-ai

# Interactive shell (debugging)
docker exec -it summarize-ai /bin/bash

# Check container resource usage
docker stats summarize-ai --no-stream

# Inspect container networking
docker inspect --format='{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' summarize-ai

# Health check status
docker inspect --format='{{.State.Health.Status}}' summarize-ai
```

---

## Multi-Container Deployment with Docker Compose

Create `docker-compose.yml`:

```yaml
version: "3.9"

services:
  summarize-ai:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: summarize-ai
    ports:
      - "8000:8000"
    env_file:
      - .env
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
        reservations:
          memory: 2G
          cpus: "1.0"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    volumes:
      - model-cache:/app/artifacts
    networks:
      - app-network

  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./production/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      summarize-ai:
        condition: service_healthy
    networks:
      - app-network
    restart: unless-stopped

volumes:
  model-cache:

networks:
  app-network:
    driver: bridge
```

```bash
# Start all services
docker compose up -d

# Scale horizontally
docker compose up -d --scale summarize-ai=3

# View logs across all services
docker compose logs -f

# Tear down
docker compose down -v
```

---

## Container Registry Push

```bash
# Tag for registry
docker tag summarize-ai:latest your-registry.com/summarize-ai:1.0.0
docker tag summarize-ai:latest your-registry.com/summarize-ai:latest

# Push
docker push your-registry.com/summarize-ai:1.0.0
docker push your-registry.com/summarize-ai:latest
```

---

## Security Checklist

| Check | Status |
|---|---|
| Non-root user (`appuser:10001`) | ✅ |
| No hardcoded secrets in image | ✅ |
| Read-only filesystem supported | ✅ |
| HEALTHCHECK probe configured | ✅ |
| Minimal base image (`python:3.11-slim`) | ✅ |
| No unnecessary system packages | ✅ |
| Layer caching optimized | ✅ |
| `.dockerignore` excludes sensitive files | ✅ |
