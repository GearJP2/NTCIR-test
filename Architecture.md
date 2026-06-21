# NTCIR-19 CSAT — Multimodal Semantic Search

Audio/multimodal semantic search for the NTCIR-19 Lifelog-7 CASTLE Semantic Access Task (CSAT).

## Architecture

```
CASTLE2024 (HuggingFace)
        │  streaming
        ▼
  train_and_index.py          ← offline ingestion (CLAP → MinIO → Milvus)
        │
        ▼
  POST /api/search/episodic   ← FastAPI episodic memory search
        │
        ▼
  frontend SearchPage         ← React UI
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (Milvus, MinIO, Redis)
- Node.js 18+ (frontend)
- HuggingFace token (`HF_TOKEN`) if CASTLE2024 is gated

## Quick Start

### 1. Environment

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN, DEVICE=cpu if no GPU
```

### 2. Install Python dependencies

```bash
pip install -e ".[dev]"
```

### 3. Start infrastructure

```bash
docker compose up -d milvus-standalone minio redis
# or: make up
```

### 4. Index dataset (dry-run first)

```bash
python scripts/train_and_index.py --max-items 5 --dry-run
python scripts/train_and_index.py --max-items 100 --split train
```

### 5. Start API

```bash
make dev
# or: docker compose up -d app
```

### 6. Test search

```bash
curl -X POST http://localhost:8000/api/search/episodic \
  -H "Content-Type: application/json" \
  -d '{"query": "eating lunch", "top_k": 5, "use_llm": false}'
```

### 7. Frontend

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

## Make targets

| Target | Description |
|---|---|
| `make up` | Start all Docker services |
| `make dev` | Run FastAPI with hot reload |
| `make test` | Run pytest with coverage |
| `make ingest` | Ingest local media corpus |
| `make build-index` | Index CASTLE2024 from HuggingFace |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/search/episodic` | Episodic memory search (primary) |
| `POST` | `/api/v1/search/episodic` | Same, versioned path |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/docs` | OpenAPI docs (dev only) |

## Project layout

```
app/           FastAPI application
services/      Business logic (query, ingestion, retrieval)
storage/       Milvus + MinIO clients
model_zoo/     ML model loaders (CLAP, Whisper, CLIP)
scripts/       CLI tools (train_and_index, ingest_corpus)
frontend/      React search UI
evaluation/    NTCIR evaluation scripts
tests/         Unit + integration tests
```

## Further reading

See [NTIR_CSAT_HANDOFF.md](NTIR_CSAT_HANDOFF.md) for full architecture notes, schema details, and remaining work items.
