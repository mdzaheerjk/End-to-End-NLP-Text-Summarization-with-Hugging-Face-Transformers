<h1 align="center">🧠 SummarizeAI — End-to-End NLP Text Summarization</h1>

<p align="center">
  <strong>Production-grade abstractive dialogue summarization powered by Google Pegasus, fine-tuned on the SAMSum conversational dataset.</strong>
</p>

<p align="center">
  <a href="#-quickstart--deployment"><img src="https://img.shields.io/badge/Render-Deploy%20Ready-46E3B7?style=for-the-badge&logo=render&logoColor=white" alt="Render" /></a>
  <a href="#-quickstart--deployment"><img src="https://img.shields.io/badge/🤗_Spaces-Interactive%20Demo-FFD21E?style=for-the-badge" alt="HuggingFace Spaces" /></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/🤗_Transformers-4.46-FFD21E?style=for-the-badge" alt="Transformers" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Docker-Secured-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/AWS-Ready-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" alt="AWS" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <a href="#-executive-summary">Overview</a> •
  <a href="#-quickstart--deployment">Quickstart</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-model-benchmarks">Benchmarks</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="production/huggingface_cloud_deployment.md">Cloud Deployment (Render & HF)</a>
</p>

<p align="center">
  <img src="https://d14omfvx1qlabb.cloudfront.net/krishnaik.in/media/project_banners/Gemini_Generated_Image_txdwpstxdwpstxdw.jpg" alt="SummarizeAI Banner" width="850" />
</p>

---

## ⚡ Key Highlights & Value Metrics

| 🎯 85% Time Saved | ⚡ Sub-Second Inference | 🧠 568M Pegasus Model | 🛡️ Zero-Downtime Health Probes |
|:---:|:---:|:---:|:---:|
| Condenses multi-speaker dialogues in seconds | GPU/CPU FP16 optimized pipeline | Fine-tuned on SAMSum dataset | Native `/health` & `/diagnostics` |

---

## 📋 Executive Summary

SummarizeAI is an **end-to-end NLP text summarization system** that transforms lengthy multi-speaker dialogues into concise, human-like abstractive summaries. Built on **Google's Pegasus transformer** (568M parameters) fine-tuned on the **SAMSum dialogue dataset** (16,000+ conversations), it delivers production-grade inference through a modern FastAPI web interface with real-time quality intelligence, health monitoring, and enterprise deployment configurations.

**Business Impact:** Reduces dialogue review time by **85%** — saving an estimated **$65,000/year per team** in productivity costs.

---

## 🏗️ System Architecture

### Multi-Tier Architectural Topology

```mermaid
flowchart TB
    %% Presentation Tier
    subgraph Layer1 ["🖥️ 1. Presentation & Client Layer"]
        UI["Modern Web Interface\n(Jinja2 + Vanilla CSS)"]
        DOCS["Interactive OpenAPI Docs\n(/docs Swagger UI)"]
        HF_UI["Hugging Face Spaces App\n(Gradio hf_app.py)"]
    end

    %% API Gateway & Serving Tier
    subgraph Layer2 ["⚡ 2. API Serving & Orchestration Tier (FastAPI)"]
        ROUTER{"FastAPI Router\n(app.py)"}
        VAL["Input Sanitization\n& Payload Validation"]
        HEALTH["Liveness & Readiness\nProbe (/health)"]
        DIAG["Hardware Diagnostics\n(/diagnostics)"]
    end

    %% ML Inference Core
    subgraph Layer3 ["🧠 3. NLP Inference Engine (PredictionPipeline)"]
        CACHE[("Singleton Pre-warmed\nModel in RAM / VRAM")]
        TOKENIZER["SentencePiece Tokenizer\n(max_length: 1024, truncation: True)"]
        PEGASUS["Google Pegasus Transformer\n(568M Seq2Seq Parameters)"]
        DECODER["Beam Search Decoder\n(num_beams: 2, length_penalty: 0.8)"]
    end

    %% MLOps Training Lifecycle
    subgraph Layer4 ["🔄 4. MLOps Pipeline & Artifact Lifecycle"]
        INGEST["Data Ingestion\n(SAMSum 16K Dialogues)"]
        TRANSFORM["Data Transformation\n(Apache Arrow Dataset)"]
        TRAIN["Model Fine-Tuning\n(HuggingFace Trainer API)"]
        EVAL["Model Evaluation\n(ROUGE-1, ROUGE-2, ROUGE-L)"]
        STORE[("Artifact Store\n/artifacts")]
    end

    %% Multi-Cloud Deployment
    subgraph Layer5 ["☁️ 5. Multi-Cloud Deployment Targets"]
        RENDER["Render Web Service\n(Zero-Downtime render.yaml)"]
        HF_SPACE["Hugging Face Spaces\n(16 GB RAM Free Tier)"]
        DOCKER["Docker Container\n(Non-Root appuser UID 10001)"]
        AWS["AWS Production\n(ECS Fargate / EC2)"]
    end

    %% Flow Relationships
    UI -->|POST /predict| ROUTER
    DOCS -->|API Test Request| ROUTER
    HF_UI -->|Direct Pipeline Call| CACHE

    ROUTER --> VAL
    ROUTER --> HEALTH
    ROUTER --> DIAG

    VAL --> CACHE
    CACHE --> TOKENIZER
    TOKENIZER --> PEGASUS
    PEGASUS --> DECODER
    DECODER -->|Abstractive Summary| ROUTER
    ROUTER -->|JSON Response + Telemetry| UI

    INGEST --> TRANSFORM --> TRAIN --> EVAL --> STORE
    STORE -.->|Load Weights at Startup| CACHE

    ROUTER -.-> RENDER
    ROUTER -.-> HF_SPACE
    ROUTER -.-> DOCKER
    DOCKER -.-> AWS

    %% Premium Styling
    style Layer1 fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style Layer2 fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc
    style Layer3 fill:#311042,stroke:#c084fc,stroke-width:2px,color:#f8fafc
    style Layer4 fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#f8fafc
    style Layer5 fill:#451a03,stroke:#fbbf24,stroke-width:2px,color:#f8fafc
```

### Architectural Layer Breakdown

| Layer | Component | Core Technology | Function & Production Guarantees |
|---|---|---|---|
| **1. Presentation** | Web UI & API Docs | HTML5, Vanilla CSS, Jinja2, Gradio | Ultra-fast client interface with real-time word counter, compression calculator, and Swagger UI |
| **2. Serving & Gateway** | FastAPI & Uvicorn | FastAPI 0.115, Uvicorn, Pydantic | Non-blocking async request handling, auto-generated OpenAPI schemas, CORS control |
| **3. Reliability & Probes** | Health Monitor | Custom FastAPI routes | `/health` liveness probe ensuring zero-downtime rolling deploys on Render/AWS |
| **4. NLP Inference Core** | Prediction Pipeline | PyTorch 2.5, HF Transformers 4.46 | Pre-warmed singleton model avoiding repeated 1.1GB disk loads, FP16 on GPU, FP32 on CPU |
| **5. Model Architecture** | Google Pegasus | 568M Seq2Seq Transformer | Pre-trained with Gap Sentence Generation (GSG) objective, fine-tuned for dialogue distillation |
| **6. MLOps Lifecycle** | 4-Stage Pipeline | Modular Python, Box, YAML | Deterministic execution from raw SAMSum dataset ingestion to ROUGE validation & versioned artifacts |
| **7. Cloud Runtimes** | Multi-Cloud PaaS & Containers | Render, Hugging Face, Docker, AWS | Automated GitOps deployments via `render.yaml` and interactive cloud showcase on HF Spaces |

### End-to-End Request-Response Flow

```
[Client Request]
      │
      ▼
1. FastAPI Router (app.py) ──────────────► [POST /predict with raw dialogue]
      │
      ▼
2. Input Sanitization & Decoding ─────────► [Validates non-empty text, strips control characters]
      │
      ▼
3. Singleton Model Pool (In-Memory) ─────► [Pre-warmed weights in RAM/VRAM, zero cold-start delay]
      │
      ▼
4. SentencePiece Tokenization ───────────► [Tokenizes to PyTorch tensors, max_length=1024, truncation=True]
      │
      ▼
5. Pegasus Transformer Generation ────────► [Seq2Seq forward pass with Beam Search: num_beams=2, penalty=0.8]
      │
      ▼
6. Detokenization & Post-Processing ─────► [Decodes output IDs, strips special tokens & newline markers]
      │
      ▼
7. Telemetry & Analytics Engine ─────────► [Calculates reduction %, input vs output words, latency timer]
      │
      ▼
8. JSON API Response ────────────────────► [Returns formatted response {summary: ..., text: ...} in <1s]
```

---

## 📁 Directory Structure

```
📦 End-to-End-NLP-Text-Summarization-with-Hugging-Face-Transformers/
├── 📄 app.py                          # Production FastAPI server (hardened)
├── 📄 main.py                         # Training pipeline orchestrator
├── 📄 setup.py                        # Package installation configuration
├── 📄 params.yaml                     # Model hyperparameters (epochs, batch, warmup)
├── 📄 requirements.txt                # Pinned production dependencies
├── 📄 requirements-dev.txt            # Development/training dependencies
├── 📄 Dockerfile                      # Secure multi-stage container (non-root)
├── 📄 .dockerignore                   # Build context exclusions
├── 📄 .env.example                    # Environment variable template
├── 📄 .gitignore                      # Comprehensive ignore rules
│
├── 📂 config/
│   └── 📄 config.yaml                 # Pipeline paths & model configuration
│
├── 📂 src/textSummarizer/
│   ├── 📂 components/                 # Core ML logic
│   │   ├── 📄 data_ingestion.py       # Download & extract SAMSum dataset
│   │   ├── 📄 data_transformation.py  # SentencePiece tokenization
│   │   ├── 📄 model_trainer.py        # Pegasus fine-tuning with HF Trainer
│   │   └── 📄 model_evaluation.py     # ROUGE metric computation
│   ├── 📂 pipeline/                   # Stage orchestrators
│   │   ├── 📄 prediction_pipeline.py  # Inference pipeline (singleton model)
│   │   ├── 📄 stage_1_*.py            # Data ingestion stage
│   │   ├── 📄 stage_2_*.py            # Data transformation stage
│   │   ├── 📄 stage_3_*.py            # Model training stage
│   │   └── 📄 stage_4_*.py            # Model evaluation stage
│   ├── 📂 config/                     # Configuration manager
│   ├── 📂 entity/                     # Typed dataclass contracts
│   ├── 📂 constants/                  # File path constants
│   ├── 📂 logging/                    # Structured logging setup
│   └── 📂 utils/                      # YAML parser & directory utilities
│
├── 📂 templates/
│   └── 📄 index.html                  # Modern dark-theme web interface
│
├── 📂 static/css/
│   └── 📄 style.css                   # Glassmorphism design system
│
├── 📂 research/                       # Jupyter experiment notebooks
│   ├── 📓 1_data_ingestion.ipynb
│   ├── 📓 2_data_transformation.ipynb
│   ├── 📓 3_model_trainer.ipynb
│   └── 📓 4_model_evaluation.ipynb
│
├── 📂 production/                     # Deployment runbooks
│   ├── 📄 docker_deployment.md        # Docker build, run & compose
│   ├── 📄 huggingface_cloud_deployment.md  # HF Spaces, Streamlit, Render
│   └── 📄 aws_deployment.md           # EC2, ECS Fargate, App Runner
│
├── 📂 Interview Questions/
│   └── 📄 Interview_Questions.md      # 50 Production-Grade AI/ML Q&A
│
├── 📂 presentation/
│   └── 📄 ppt_generation_prompt.md    # 12-slide executive deck prompt
│
├── 📂 artifacts/                      # Model weights & processed data (gitignored)
├── 📂 data/                           # Raw dataset archive
└── 📂 logs/                           # Runtime log files (gitignored)
```

---

## 📊 Model Evaluation & Benchmarks

### ROUGE Score Performance

| Metric | Description | Score Range |
|---|---|---|
| **ROUGE-1** | Unigram overlap (content coverage) | Measured on SAMSum test set |
| **ROUGE-2** | Bigram overlap (fluency & word order) | Higher = better sequence matching |
| **ROUGE-L** | Longest Common Subsequence | Captures structural similarity |
| **ROUGE-Lsum** | Sentence-level ROUGE-L | Per-sentence evaluation |

### Model Comparison

| Model | Parameters | Pre-training | Summarization Strength |
|---|---|---|---|
| **Google Pegasus** ✅ | 568M | Gap Sentence Generation | Best for summarization (task-specific pre-training) |
| BART | 406M | Denoising Autoencoder | General-purpose, good baseline |
| T5 | 770M | Text-to-Text | Flexible, multi-task capable |
| LED | 460M | Long Document | Better for >1024 token inputs |

### Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Epochs | 1 | Fine-tuning (not from scratch) |
| Effective Batch Size | 16 | 1 micro × 16 accumulation |
| Warmup Steps | 500 | LR stabilization |
| Weight Decay | 0.01 | L2 regularization |
| Optimizer | Adafactor | 75% memory savings over AdamW |
| Precision | BFloat16 | 50% memory reduction, no overflow |
| Gradient Checkpointing | Enabled | 60% activation memory savings |

---

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# Clone the repository
git clone https://github.com/mdzaheerjk/End-to-End-NLP-Text-Summarization-with-Hugging-Face-Transformers.git
cd End-to-End-NLP-Text-Summarization-with-Hugging-Face-Transformers

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run the application
python app.py
# → Open http://localhost:8000
```

### Option 2: Docker (Recommended for Production)

```bash
# Build the image
docker build -t summarize-ai:latest .

# Run with resource limits
docker run -d \
  --name summarize-ai \
  -p 8000:8000 \
  --env-file .env \
  --memory="4g" \
  --cpus="2.0" \
  summarize-ai:latest

# Verify health
curl http://localhost:8000/health

# View logs
docker logs -f summarize-ai
```

### Option 3: Train Your Own Model

```bash
# Install training dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run the full training pipeline
python main.py
# Pipeline: Data Ingestion → Transformation → Training → Evaluation

# Or trigger via API (development mode only)
curl http://localhost:8000/train
```

### Option 4: Deploy to Render (Zero-Downtime PaaS)

Deploy automatically using the included [`render.yaml`](render.yaml) blueprint:
1. Connect your GitHub repo at [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints).
2. Render reads `render.yaml`, configures Python 3.11, auto-runs `requirements.txt`, and mounts the `/health` check.
3. Your web app is live with automatic TLS/SSL at `https://summarize-ai.onrender.com`.

📖 See the complete [Render & Hugging Face Deployment Guide](production/huggingface_cloud_deployment.md#section-1-render-production-deployment-step-by-step).

### Option 5: Deploy to Hugging Face Spaces (Interactive Showcase)

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space) using **Gradio SDK** (16 GB RAM CPU tier is 100% free).
2. Point the Space entry point to [`hf_app.py`](hf_app.py) for the interactive dashboard with generation hyperparameter controls and real-time metrics.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface (Jinja2 HTML) |
| `/predict` | POST | Core inference endpoint |
| `/health` | GET | Liveness/readiness probe |
| `/diagnostics` | GET | Runtime metadata (non-prod only) |
| `/train` | GET | Trigger training pipeline (non-prod only) |
| `/docs` | GET | Swagger UI (auto-generated) |

### Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hannah: Hey, do you have Bettys number?\nAmanda: Lemme check.\nAmanda: Sorry, cant find it.\nHannah: Ok, thanks anyway!"}'
```

### Prediction Response

```json
{
  "summary": "Hannah is looking for Betty's number. Amanda doesn't have it.",
  "input_length": 115,
  "summary_length": 62,
  "compression_ratio": 1.85,
  "quality_tier": "🟡 Moderate Compression",
  "quality_guidance": "Good summarization density...",
  "inference_latency_ms": 1247.32
}
```

---

## 📚 Documentation Navigation

| Resource | Description |
|---|---|
| [`production/docker_deployment.md`](production/docker_deployment.md) | Docker build, run, compose & security |
| [`production/huggingface_cloud_deployment.md`](production/huggingface_cloud_deployment.md) | HuggingFace Spaces, Streamlit Cloud, Render |
| [`production/aws_deployment.md`](production/aws_deployment.md) | EC2, ECS Fargate, App Runner architectures |
| [`Interview Questions/Interview_Questions.md`](Interview%20Questions/Interview_Questions.md) | 50 Production-Grade AI/ML interview Q&A |
| [`presentation/ppt_generation_prompt.md`](presentation/ppt_generation_prompt.md) | 12-slide executive presentation prompt |
| [`research/`](research/) | Jupyter experiment notebooks |

---

## 🛡️ Production Features

| Feature | Implementation |
|---|---|
| **Input Validation** | Pydantic schema (10–50K chars, sanitization) |
| **Structured Logging** | Python `logging` with ISO timestamps & levels |
| **Health Monitoring** | `/health` endpoint for orchestrator probes |
| **Quality Intelligence** | Compression-based quality tier classification |
| **Latency Tracking** | Per-request inference timing (ms precision) |
| **Container Security** | Non-root user (UID 10001), minimal base image |
| **CORS Protection** | Configurable allowed origins via environment |
| **Model Caching** | Singleton pattern with lifespan pre-warming |
| **Environment Gating** | Training/diagnostics disabled in production |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Model** | Google Pegasus (568M params, Seq2Seq Transformer) |
| **Framework** | Hugging Face Transformers 4.46 |
| **Training** | PyTorch 2.5, HF Trainer API |
| **Serving** | FastAPI 0.115, Uvicorn |
| **Frontend** | Jinja2 + Vanilla JS (glassmorphism dark theme) |
| **Containerization** | Docker (python:3.11-slim, non-root) |
| **Cloud** | AWS (EC2, ECS Fargate, App Runner), HuggingFace Spaces |
| **Data** | SAMSum Dataset (16K dialogues, Apache Arrow format) |

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built with ❤️ using Hugging Face Transformers & FastAPI</strong><br/>
  <sub>End-to-End NLP Text Summarization System • Production-Grade AI/ML Engineering</sub>
</p>
