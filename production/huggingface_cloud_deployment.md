# 🚀 Production Cloud Deployment Guide: Render & Hugging Face Spaces

> **Production PaaS Deployment Guide for SummarizeAI — Google Pegasus Dialogue Summarization**  
> Covers zero-downtime deployment to **Render** (Primary Web Service) and **Hugging Face Spaces** (Cloud Showcase).  
> *Zero Docker / AWS dependencies required — direct native GitOps cloud deployment.*

---

## 📑 Table of Contents
1. [Platform Comparison & Selection](#-platform-comparison--selection)
2. [Section 1: Render Production Deployment (Step-by-Step)](#-section-1-render-production-deployment-step-by-step)
   - [Architectural Overview](#architectural-overview)
   - [Option A: 1-Click Infrastructure-as-Code Blueprint (`render.yaml`)](#option-a-1-click-infrastructure-as-code-blueprint-renderyaml)
   - [Option B: Manual Dashboard Setup Guide](#option-b-manual-dashboard-setup-guide)
   - [Memory & Resource Sizing Guide](#memory--resource-sizing-guide)
   - [Configuring Environment Variables](#configuring-environment-variables)
   - [Health Check & Zero-Downtime Verification](#health-check--zero-downtime-verification)
   - [Custom Domain & Automatic SSL](#custom-domain--automatic-ssl)
   - [Render Troubleshooting & Incident Runbook](#render-troubleshooting--incident-runbook)
3. [Section 2: Hugging Face Spaces Deployment](#-section-2-hugging-face-spaces-deployment)
   - [Option A: Native Gradio Showcase Space (`hf_app.py`)](#option-a-native-gradio-showcase-space-hf_apppy)
   - [Option B: Automated GitHub-to-HuggingFace Sync (CI/CD)](#option-b-automated-github-to-huggingface-sync-cicd)
   - [Hardware Tier Recommendations](#hardware-tier-recommendations)
4. [Production Verification & Smoke Tests](#-production-verification--smoke-tests)

---

## ⚖️ Platform Comparison & Selection

| Feature | Render Web Service | Hugging Face Spaces |
|---|---|---|
| **Primary Use Case** | Full FastAPI backend + Modern Web UI + REST API | Public ML Model Portfolio Showcase & Interactive Demo |
| **Interface** | Custom Responsive Web App (`templates/index.html`) + `/docs` | Custom Gradio Dashboard (`hf_app.py`) with metric dials |
| **Health Checks** | Native active `/health` liveness probe with zero-downtime deploy | Automatic container restart on crash |
| **Free Tier Specs** | 512 MB RAM / 0.1 CPU (Spins down on inactivity) | **16 GB RAM / 2 vCPU** (Free indefinitely, no spin down) |
| **Recommended Paid Tier**| Starter ($7/mo: 2 GB RAM, 0.5 CPU — never sleeps) | Free Tier is sufficient (or T4 GPU $0.60/hr) |
| **Port Handling** | Dynamic via `$PORT` environment variable | Standard port `7860` |
| **Deployment Trigger** | Git push to `main` branch (Auto-deploy) | Git push / GitHub Actions automated mirror |

---

## 🌐 Section 1: Render Production Deployment (Step-by-Step)

### Architectural Overview

Render hosts SummarizeAI as a managed native Python web service.

```mermaid
flowchart LR
    A[Client Browser] -->|HTTPS :443| B[Render Global CDN / TLS]
    B -->|Port $PORT| C[FastAPI + Uvicorn]
    C -->|GET /health| D[Health Monitor]
    C -->|POST /predict| E[Pegasus Inference Engine]
    E -->|Torch Float32| F[Pre-warmed Model Weights]
```

---

### Option A: 1-Click Infrastructure-as-Code Blueprint (`render.yaml`)

This repository includes a production-ready `render.yaml` blueprint at the root:

```yaml
version: "1"

services:
  - type: web
    name: summarize-ai
    runtime: python
    buildCommand: pip install --upgrade pip && pip install -r requirements.txt
    startCommand: uvicorn app.py:app --host 0.0.0.0 --port $PORT
    plan: starter # 2 GB RAM recommended for Pegasus 568M model inference
    region: oregon
    healthCheckPath: /health
    autoDeploy: true
    envVars:
      - key: APP_ENV
        value: production
      - key: LOG_LEVEL
        value: INFO
      - key: PYTHON_VERSION
        value: 3.11.9
```

#### How to Deploy with the Blueprint:
1. Push your latest code to your GitHub repository.
2. Log in to [dashboard.render.com](https://dashboard.render.com).
3. Click **Blueprints** in the top navigation bar.
4. Click **New Blueprint Instance**.
5. Connect your GitHub repository: `End-to-End-NLP-Text-Summarization-with-Hugging-Face-Transformers`.
6. Render will automatically parse `render.yaml` and configure all settings, build commands, and health probes.
7. Click **Apply**. Render will automatically build the environment and launch your app!

---

### Option B: Manual Dashboard Setup Guide

If you prefer creating the service manually through Render's web interface:

#### Step 1: Create a New Web Service
1. Go to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** → **Web Service**.
3. Select **Build and deploy from a Git repository** and click **Next**.
4. Choose your repository and click **Connect**.

#### Step 2: Configure Service Settings

| Setting Field | Value to Enter | Notes |
|---|---|---|
| **Name** | `summarize-ai` | Your URL will be `https://summarize-ai.onrender.com` |
| **Region** | `Oregon (US West)` or `Frankfurt (EU)` | Choose closest to your target audience |
| **Branch** | `main` | Production branch |
| **Root Directory** | *(leave blank)* | Root of the repo |
| **Runtime** | `Python 3` | Native Python runtime |
| **Build Command** | `pip install --upgrade pip && pip install -r requirements.txt` | Installs PyTorch, Transformers, FastAPI |
| **Start Command** | `uvicorn app.py:app --host 0.0.0.0 --port $PORT` | Uses Render's dynamic `$PORT` variable |
| **Instance Type** | **Starter** (2 GB RAM, $7/mo) or **Free** | *See memory notes below* |

#### Step 3: Expand "Advanced" Settings

1. **Health Check Path:** Enter `/health`  
   *(Render will ping `https://your-app.onrender.com/health` every 10 seconds. New builds will only be swapped into production once this endpoint returns HTTP 200, guaranteeing zero downtime!)*
2. **Auto-Deploy:** Set to **Yes** *(Deploys automatically on every `git push origin main`)*.

---

### Memory & Resource Sizing Guide

The Google Pegasus architecture (`google/pegasus-cnn_dailymail`) is a 568M parameter sequence-to-sequence model.

| Metric | Free Tier (512 MB RAM) | Starter Tier (2 GB RAM) | Standard Tier (4 GB RAM) |
|---|---|---|---|
| **Model In-Memory Footprint** | ~1.1 GB (CPU float32) | ~1.1 GB | ~1.1 GB |
| **Peak Inference Memory** | ~1.4 GB | ~1.4 GB | ~1.4 GB |
| **Suitability** | ⚠️ High risk of OOM (`Out of Memory`) | ✅ **Recommended Production Tier** | ⚡ High-throughput concurrency |
| **Cold Start Behavior** | Spins down after 15 min idle; ~45s spin-up | Never sleeps; instant sub-second response | Never sleeps; multi-worker support |
| **Cost** | $0 / month | $7 / month | $25 / month |

> [!TIP]
> **Optimizing for the Free Tier:**  
> If using the Render Free Tier, set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` and ensure `torch.set_num_threads(2)` is configured to keep CPU memory overhead minimal. However, the **Starter (2 GB RAM)** plan is strongly recommended for seamless, uninterrupted production operation.

---

### Configuring Environment Variables

In your Render Dashboard, click **Environment** in the left sidebar and configure:

| Key | Value | Purpose |
|---|---|---|
| `APP_ENV` | `production` | Enables production optimizations and disables debug traces |
| `LOG_LEVEL` | `INFO` | Structured logging verbosity |
| `PYTHON_VERSION` | `3.11.9` | Locks the exact Python runtime version |
| `TOKENIZERS_PARALLELISM` | `false` | Avoids thread contention warnings in Hugging Face tokenizers |
| `MODEL_PATH` | `google/pegasus-cnn_dailymail` | Fallback HuggingFace repository ID |

---

### Health Check & Zero-Downtime Verification

Once deployed, Render automatically monitors the service via the `/health` endpoint:

```bash
# 1. Test basic liveness probe
curl -i https://summarize-ai.onrender.com/health

# Response:
# HTTP/2 200 OK
# content-type: application/json
# {"status":"healthy","service":"SummarizeAI","model_loaded":true}

# 2. Test detailed hardware diagnostics
curl -s https://summarize-ai.onrender.com/diagnostics | python -m json.tool

# Response:
# {
#     "status": "ok",
#     "device": "cpu",
#     "cuda_available": false,
#     "model_loaded": true
# }
```

---

### Custom Domain & Automatic SSL

To connect your own domain (e.g. `summarizer.yourdomain.com`):
1. In Render Dashboard, go to **Settings** → **Custom Domains**.
2. Click **Add Custom Domain** and enter your domain name.
3. In your DNS provider (Cloudflare, GoDaddy, Namecheap), add the CNAME record:
   - **Type:** `CNAME`
   - **Name:** `summarizer` (or `@`)
   - **Value:** `summarize-ai.onrender.com`
4. Render automatically provisions and renews a free **Let's Encrypt SSL/TLS Certificate**.

---

### Render Troubleshooting & Incident Runbook

#### Issue 1: `Out of Memory (OOM) killed`
- **Symptom:** Build succeeds, but service crashes immediately after `Pre-warming model into memory...`.
- **Root Cause:** Free tier has only 512 MB RAM, whereas Pegasus requires ~1.1 GB.
- **Resolution:** Upgrade to the **Starter** plan (2 GB RAM) under **Settings** → **Plan**.

#### Issue 2: `Port binding failed / Timed out waiting for port`
- **Symptom:** Logs show `Uvicorn running on http://127.0.0.1:8000`, followed by Render deployment timeout.
- **Root Cause:** Hardcoded localhost or incorrect port.
- **Resolution:** Render requires binding to `0.0.0.0` and reading `$PORT`. In this repo, `app.py` has already been updated to:
  ```python
  port = int(os.environ.get("PORT", os.environ.get("SERVER_PORT", 8000)))
  uvicorn.run(app, host="0.0.0.0", port=port)
  ```

#### Issue 3: `Pip build timeout during torch installation`
- **Symptom:** Build terminates during `pip install -r requirements.txt`.
- **Resolution:** Clear build cache in Render Dashboard: **Manual Deploy** → **Clear build cache & deploy**.

---

## 🤗 Section 2: Hugging Face Spaces Deployment

Hugging Face Spaces offers **16 GB RAM and 2 vCPUs completely FREE**, making it an ideal companion platform for ML models.

### Option A: Native Gradio Showcase Space (`hf_app.py`)

This repository includes a dedicated `hf_app.py` script featuring:
- Full conversation dialogue input area with 3 pre-loaded real-world presets.
- Interactive hyperparameter sliders: Beam search (`num_beams`), maximum length, length penalty.
- Real-time performance metrics: Input word count, summary word count, % compression ratio, and execution timer.

#### Deployment Steps:
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Fill in the Space settings:
   - **Space Name:** `summarize-ai`
   - **License:** `mit`
   - **SDK:** Select **Gradio**
   - **Hardware:** Select **CPU Basic (Free - 16 GB RAM, 2 vCPUs)**
3. In the new Space repository, upload:
   - `hf_app.py` (rename to `app.py` in the Space root, or configure `app_file: hf_app.py` in the space `README.md`)
   - `requirements.txt`
   - `src/` directory
   - `config/` directory
   - `params.yaml`
4. Space automatically builds dependencies and launches the live interactive app!

---

### Option B: Automated GitHub-to-HuggingFace Sync (CI/CD)

You can set up a GitHub Action to automatically synchronize every commit from your GitHub `main` branch directly to your Hugging Face Space.

Create `.github/workflows/hf_sync.yml`:

```yaml
name: Sync to Hugging Face Spaces

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true
      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config --global user.email "ci@summarize-ai.com"
          git config --global user.name "SummarizeAI Bot"
          git push --force https://YOUR_HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/YOUR_HF_USERNAME/summarize-ai main
```

> **To configure:**
> 1. Generate a write-access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
> 2. Add it as a secret named `HF_TOKEN` in GitHub repo: **Settings → Secrets and variables → Actions**.

---

### Hardware Tier Recommendations

| Hardware Option | Cost | Latency per 500 words | Concurrency Recommendation |
|---|---|---|---|
| **CPU Basic (2 vCPU, 16 GB RAM)** | **$0.00 (Free)** | ~4.2 - 6.5s | Single user / low traffic showcase |
| **CPU Upgrade (8 vCPU, 32 GB RAM)** | $0.03 / hour | ~2.5 - 3.8s | Small team testing |
| **Nvidia T4 GPU (16 GB VRAM)** | $0.60 / hour | ~0.6 - 1.1s | High-concurrency production demo |

---

## 🧪 Production Verification & Smoke Tests

After deploying to either Render or Hugging Face Spaces, run the automated smoke test script to verify end-to-end functionality:

```bash
# Test Dialogue Payload
curl -X POST "https://YOUR_APP_URL.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hannah: Hey, do you have Betty'\''s number?\nAmanda: Lemme check... Yup, here it is: 555-019-2834\nHannah: Thanks! Need to ask her about the budget report."}'

# Expected JSON Response:
# {
#   "summary": "Amanda gives Betty's phone number to Hannah.",
#   "text": "Amanda gives Betty's phone number to Hannah."
# }
```

### Swagger Documentation
Verify the interactive OpenAPI / Swagger UI by visiting:
`https://YOUR_APP_URL.onrender.com/docs`
