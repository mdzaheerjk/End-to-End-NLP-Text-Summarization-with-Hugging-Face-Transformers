# 🤗 Hugging Face Cloud Deployment Guide — SummarizeAI

> **Production ML Showcase & Cloud Serving Guide for Google Pegasus Dialogue Summarization**  
> Covers deployment to **Hugging Face Spaces** (Interactive Gradio Showcase), **Automated GitHub Actions Sync (GitOps)**, and **Hugging Face Inference Endpoints**.

---

## 📑 Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Method 1: Hugging Face Spaces via Gradio SDK (`hf_app.py`) — Recommended](#method-1-hugging-face-spaces-via-gradio-sdk-hf_apppy--recommended)
   - [Step 1: Create a Space](#step-1-create-a-space)
   - [Step 2: Space Configuration & Entry Point](#step-2-space-configuration--entry-point)
   - [Step 3: Features of the Gradio App](#step-3-features-of-the-gradio-app)
3. [Method 2: Automated GitHub-to-HuggingFace CI/CD (GitOps)](#method-2-automated-github-to-huggingface-cicd-gitops)
   - [Configuring GitHub Actions Mirroring](#configuring-github-actions-mirroring)
   - [Managing Hugging Face Access Tokens](#managing-hugging-face-access-tokens)
4. [Method 3: Hugging Face Dedicated Inference Endpoints](#method-3-hugging-face-dedicated-inference-endpoints)
5. [Hardware Tier Sizing & Latency Benchmarks](#hardware-tier-sizing--latency-benchmarks)
6. [Environment Variables & Secrets Management](#environment-variables--secrets-management)
7. [Space Monitoring, Reboot & Troubleshooting Runbooks](#space-monitoring-reboot--troubleshooting-runbooks)

---

## Overview & Architecture

Hugging Face Spaces provides an industry-standard platform to showcase transformer models with generous free hardware (**16 GB RAM, 2 vCPUs completely free**).

```mermaid
flowchart LR
    A[User / Browser] -->|HTTPS :443| B[Hugging Face Global Edge]
    B -->|Port 7860| C[Gradio Interface hf_app.py]
    C -->|Custom Generation Params| D[PredictionPipeline]
    D -->|Tokenizer + Model| E[Google Pegasus 568M]
    E -->|Inference Execution| F[Summary + Live Telemetry Cards]
    F -->|Render| A
```

---

## Method 1: Hugging Face Spaces via Gradio SDK (`hf_app.py`) — Recommended

This repository includes a dedicated Hugging Face Spaces application: [`hf_app.py`](../hf_app.py).

### Step 1: Create a Space
1. Navigate to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Configure the Space settings:
   - **Space Name:** `summarize-ai` (or your preferred name)
   - **License:** `mit`
   - **Space SDK:** Select **Gradio**
   - **Gradio Version:** Latest (4.x)
   - **Space Hardware:** Select **CPU Basic (2 vCPU, 16 GB RAM — Free)**
   - **Visibility:** Public (for portfolio visibility) or Private

### Step 2: Space Configuration & Entry Point

Hugging Face Spaces looks for an entry file at the root. You have two options:

#### Option A: Use `hf_app.py` via Space YAML Metadata
Add YAML frontmatter to the Space's `README.md`:

```yaml
---
title: SummarizeAI — Dialogue Text Summarization
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: hf_app.py
pinned: false
license: mit
---
```

#### Option B: Deploy `hf_app.py` as `app.py` in the Space
In your Hugging Face Space repository, simply copy or rename `hf_app.py` to `app.py`.

#### Required Files in the Space:
```
hf-space/
├── 📄 app.py                          # Renamed from hf_app.py
├── 📄 requirements.txt                # Production dependencies
├── 📄 params.yaml                     # Model hyperparameters
├── 📂 config/
│   └── 📄 config.yaml                 # Configuration paths
└── 📂 src/
    └── 📂 textSummarizer/             # Core ML pipeline logic
```

### Step 3: Features of the Gradio App

The included [`hf_app.py`](../hf_app.py) comes pre-configured with:
- **Interactive Input Area:** Paste arbitrary conversational dialogues or chat logs.
- **Generation Hyperparameter Sliders:**
  - **Beam Search (`num_beams`):** Control exploration vs. quality (range: 1–5, default: 2).
  - **Maximum Length (`max_length`):** Set summary ceiling (range: 32–256 tokens, default: 128).
  - **Length Penalty (`length_penalty`):** Penalize or encourage longer summaries (range: 0.2–2.0, default: 0.8).
- **Real-Time Quality Metrics:**
  - Input word count & summary word count.
  - Compression ratio percentage reduction.
  - End-to-end inference latency timer in seconds.
- **Curated 1-Click Dialogue Presets:**
  - Customer Service Escalation
  - Engineering Standup Meeting
  - Medical Consultation Transcript

---

## Method 2: Automated GitHub-to-HuggingFace CI/CD (GitOps)

You can automatically keep your Hugging Face Space synchronized with every push to your GitHub `main` branch.

### Step 1: Generate a Hugging Face Access Token
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Click **Create new token**.
3. Name: `github-actions-sync`, Type: **Write**.
4. Copy the generated token (`hf_...`).

### Step 2: Store Token in GitHub Secrets
1. In your GitHub repository, navigate to **Settings → Secrets and variables → Actions**.
2. Click **New repository secret**.
3. Name: `HF_TOKEN`, Secret: paste your Hugging Face write token.
4. Add another secret: Name: `HF_USERNAME`, Secret: your Hugging Face username.

### Step 3: Create GitHub Actions Workflow

Create `.github/workflows/hf_sync.yml`:

```yaml
name: Sync to Hugging Face Spaces

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  sync-to-hf:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true

      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          HF_USER: ${{ secrets.HF_USERNAME }}
        run: |
          git config --global user.email "ci@summarize-ai.com"
          git config --global user.name "SummarizeAI Sync Bot"
          git remote add hf https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/summarize-ai
          git push --force hf main
```

---

## Method 3: Hugging Face Dedicated Inference Endpoints

For production APIs with guaranteed uptime and auto-scaling:

1. Navigate to [ui.endpoints.huggingface.co](https://ui.endpoints.huggingface.co).
2. Click **New Endpoint**.
3. Model Repository: `google/pegasus-cnn_dailymail` (or your fine-tuned model path).
4. Task: `Text2Text Generation` / `Summarization`.
5. Cloud Provider: AWS (`us-east-1`) or GCP.
6. Instance Type:
   - **CPU (c6i.xlarge):** ~$0.06/hr
   - **GPU (Nvidia T4):** ~$0.60/hr
7. Click **Create Endpoint** — Hugging Face provides a production HTTPS API URL:
   ```bash
   curl -X POST https://your-endpoint.endpoints.huggingface.cloud \
     -H "Authorization: Bearer YOUR_HF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"inputs": "Hannah: Hey, do you have Betty'\''s number?\nAmanda: Yes, it is 555-019-2834"}'
   ```

---

## Hardware Tier Sizing & Latency Benchmarks

| Hardware Tier | Memory | Latency (500-word input) | Cost | Recommended Scenario |
|---|---|---|---|---|
| **CPU Basic** | 16 GB RAM, 2 vCPU | ~4.5 – 6.2s | **$0.00 (Free)** | Public portfolio showcase & interactive demo |
| **CPU Upgrade** | 32 GB RAM, 8 vCPU | ~2.5 – 3.8s | $0.03 / hr | Team testing & multi-user reviews |
| **Nvidia T4** | 16 GB VRAM, 4 vCPU | ~0.7 – 1.1s | $0.60 / hr | Low-latency client demonstrations |
| **Nvidia A10G** | 24 GB VRAM, 4 vCPU | ~0.3 – 0.6s | $1.05 / hr | High-throughput batch summarization |

---

## Environment Variables & Secrets Management

In your Hugging Face Space, configure environment variables under **Settings → Variables and secrets**:

| Type | Key | Value | Purpose |
|---|---|---|---|
| **Variable** | `APP_ENV` | `production` | Production environment flag |
| **Variable** | `LOG_LEVEL` | `INFO` | Structured logging verbosity |
| **Variable** | `SERVER_PORT` | `7860` | Hugging Face Spaces standard port |
| **Variable** | `TOKENIZERS_PARALLELISM` | `false` | Disables tokenizers multiprocessing warnings |
| **Secret** | `HF_TOKEN` | `hf_...` | Hugging Face API token for model downloads |

---

## Space Monitoring, Reboot & Troubleshooting Runbooks

### 1. View Logs & Debug Output
- In your Space header, click the **Logs** tab.
- Gradio displays real-time Python stdout/stderr traces.

### 2. Factory Reboot
- If the Space becomes unresponsive or dependencies are updated:
  1. Go to **Settings** in your Space.
  2. Scroll to the **Factory Reboot** section.
  3. Click **Restart this Space**.

### 3. Common Issues & Solutions

| Issue | Root Cause | Solution |
|---|---|---|
| **Build Timeout** | Large PyTorch download | Ensure `requirements.txt` specifies compatible pre-built wheels |
| **Port Collision** | App not listening on 7860 | Set `server_port=7860` in `demo.launch()` (handled automatically in `hf_app.py`) |
| **Model Download Timeout** | Slow Hugging Face Hub connection | Add retry logic or stream weights directly with `AutoModelForSeq2SeqLM.from_pretrained()` |
| **Git LFS File Size Limit** | Committing large model weights | Use `.gitignore` to exclude `artifacts/` and load pre-trained weights from Hugging Face Hub |
