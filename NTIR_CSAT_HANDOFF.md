# NTCIR-19 CSAT — AI Engineer Handoff Document

> **วัตถุประสงค์:** เอกสารนี้คือบริบทครบชุดสำหรับ AI ตัวถัดไปที่จะรับช่วงงาน  
> ครอบคลุมสถาปัตยกรรม, โครงสร้างโค้ดจริงใน repo, ไฟล์ที่เขียนเสร็จแล้ว,  
> และสิ่งที่ยังต้องทำต่อพร้อม context เพียงพอจะเริ่มทำได้ทันที

---

## 1. ภาพรวมโปรเจกต์

| รายการ | ค่า |
|---|---|
| ชื่อโปรเจกต์ | NTCIR-19 Lifelog-7 — CSAT (CASTLE Semantic Access Task) |
| GitHub repo | `https://github.com/GearJP2/NTCIR-test` |
| Dataset | `CASTLE-Dataset/CASTLE2024` (HuggingFace) |
| เป้าหมายหลัก | ระบบ Audio/Multimodal Semantic Search สำหรับค้นหา lifelog audio ด้วยภาษาธรรมชาติ |
| สถาปัตยกรรมแรงบันดาลใจ | WorldMM (Dynamic Multimodal Memory Agent) |

### แนวคิดระบบ (Pipeline)

```
CASTLE2024 (HuggingFace)
        │  streaming
        ▼
┌─────────────────────────────────────────┐  OFFLINE
│  ingest_castle.py                        │  (Milestone 1)
│  ├─ VAD chunking (webrtcvad, 30s/2s)    │
│  ├─ Whisper ASR  (faster-whisper)       │
│  ├─ CLAP embed   (laion/clap-htsat-fused│
│  ├─ MinIO upload (castle-audio bucket)  │
│  └─ Milvus upsert (csat_episodic_memory)│
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐  ONLINE
│  FastAPI: POST /api/search/episodic      │  (Milestone 2)
│  ├─ CLAP text encode (query → 512-dim)  │
│  ├─ Milvus HNSW/COSINE ANN (top_k×3)   │
│  ├─ BM25 rerank (rank-bm25, RRF)       │
│  └─ MinIO presigned URL refresh (1h)   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐  UI
│  SearchPage.jsx (React + Vite)           │  (Milestone 3)
│  ├─ Text query input                    │
│  ├─ Score / top-k controls              │
│  ├─ Result cards + <audio> player       │
│  └─ Pagination                          │
└─────────────────────────────────────────┘
```

---

## 2. Tech Stack

| Layer | Technology | Version/Detail |
|---|---|---|
| Frontend | React + Vite + TailwindCSS | JavaScript |
| Backend | FastAPI | Python 3.10+ |
| Task Queue | ARQ (Async Redis Queue) | async workers |
| Vector DB | Milvus Standalone | HNSW / COSINE, 512-dim |
| Object Storage | MinIO | bucket: `castle-audio` |
| Cache / Broker | Redis | สำหรับ ARQ |
| Embedding | `laion/clap-htsat-fused` | 512-dim, joint audio+text |
| ASR | `faster-whisper` | model size: base (ปรับได้) |
| Reranker | `rank-bm25` (BM25Okapi) + RRF | hybrid retrieval |
| Packaging | `pyproject.toml` | ไม่ใช่ `requirements.txt` |
| Container | Docker Compose | multi-service |

---

## 3. โครงสร้าง Repository จริง (ใน GitHub)

> ⚠️ โครงสร้างในรีโปจริงต่างจากที่เจ้าของโปรเจกต์อธิบายไว้เริ่มต้น  
> ได้รับการ refactor เป็น production-grade layout แล้ว

```
NTCIR-test/
│
├── app/                          ← FastAPI application (ไม่ใช่ backend/)
│   ├── main.py                   ✅ เขียนแล้ว (ดูหัวข้อ 4.1)
│   ├── core/
│   │   ├── config.py             ⚠️ ต้องตรวจสอบว่ามีหรือยัง
│   │   ├── exceptions.py         ⚠️ ต้องตรวจสอบ (ใช้ StorageError)
│   │   └── logging.py            ⚠️ ต้องตรวจสอบ
│   ├── api/
│   │   └── v1/
│   │       ├── router.py         ⚠️ ต้องตรวจสอบ (main.py import api_router)
│   │       └── endpoints/
│   │           └── search.py     ✅ เขียนใหม่แล้ว (Milestone 2)
│   └── schemas/
│       └── audio.py              ✅ เขียนใหม่แล้ว (AudioChunk, SearchResult)
│
├── model_zoo/                    ← โมเดล AI wrappers
│   ├── __init__.py               ⬜ ว่างเปล่า (stub เท่านั้น)
│   ├── registry.py               ✅ เขียนใหม่แล้ว
│   ├── clap/
│   │   ├── __init__.py           ⬜ ว่างเปล่า
│   │   └── loader.py             ✅ เขียนใหม่แล้ว (CLAPEmbedder)
│   ├── whisper/
│   │   ├── __init__.py           ⬜ ว่างเปล่า
│   │   └── loader.py             ✅ เขียนใหม่แล้ว (WhisperTranscriber)
│   ├── clip/
│   │   ├── __init__.py           ⬜ ว่างเปล่า
│   │   └── loader.py             ⬜ ว่างเปล่า (ยังไม่ implement)
│   ├── text_encoder/
│   │   ├── __init__.py           ⬜ ว่างเปล่า
│   │   └── loader.py             ⬜ ว่างเปล่า (ยังไม่ implement)
│   └── reranker/
│       ├── __init__.py           ⬜ ว่างเปล่า
│       └── loader.py             ⬜ ว่างเปล่า (ยังไม่ implement)
│
├── services/                     ← Service layer (business logic)
│   ├── milvus_service.py         ✅ เขียนแล้ว — สมบูรณ์มาก (ดูหัวข้อ 4.2)
│   ├── minio_service.py          ⚠️ ไม่ทราบ status — ต้องตรวจสอบ
│   └── audio_service.py          ⚠️ ไม่ทราบ status — ต้องตรวจสอบ
│
├── storage/
│   └── milvus/
│       ├── client.py             ⚠️ ต้องตรวจสอบ (MilvusService import จาก นี่)
│       └── collections.py        ⚠️ ต้องตรวจสอบ (ensure_all_collections)
│
├── workers/                      ← ARQ async task workers
│   └── ...                       ⚠️ ไม่ทราบ status ทั้งหมด
│
├── scripts/
│   └── ingest_castle.py          ✅ เขียนใหม่แล้ว (Milestone 1)
│
├── configs/                      ← YAML config files
├── docker/                       ← Dockerfiles (app + worker)
├── evaluation/                   ← NTCIR eval scripts
├── tests/
│
├── frontend/
│   └── src/
│       └── pages/
│           ├── SearchPage.jsx    ✅ เขียนใหม่แล้ว (Milestone 3)
│           └── UploadPage.jsx    ⚠️ partial / ไม่แน่ใจ status
│
├── .env.example                  ✅ มีในรีโป
├── Makefile                      ✅ มีในรีโป
├── docker-compose.yml            ✅ สมบูรณ์มาก (ดูหัวข้อ 4.3)
└── pyproject.toml                ✅ มีในรีโป (ดูหัวข้อ 4.4)
```

**สัญลักษณ์:**  
✅ เขียนเสร็จ / ยืนยันมีแล้ว  
⚠️ ต้องตรวจสอบ / ไม่ทราบ status  
⬜ ว่างเปล่า / stub เท่านั้น

---

## 4. ไฟล์สำคัญ — สรุป Content

### 4.1 `app/main.py` (ได้รับจากเจ้าของโปรเจกต์ — สมบูรณ์)

```python
# Import ที่สำคัญ (ต้องมีไฟล์เหล่านี้)
from app.api.v1.router import api_router          # ← ต้องตรวจสอบว่ามีหรือยัง
from app.api.v1.endpoints.search import router as search_router  # ✅ เขียนแล้ว
from app.core.config import settings              # ← ต้องตรวจสอบ
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from storage.milvus.client import get_milvus_client
from storage.milvus.collections import ensure_all_collections

# Routes ที่ mount:
# /api/v1/...          ← จาก api_router
# /api/search/episodic ← short alias ของ search_router
```

### 4.2 `services/milvus_service.py` (ได้รับจากเจ้าของโปรเจกต์ — สมบูรณ์)

**Milvus Collection Schema: `csat_episodic_memory`**

| Field | Type | Detail |
|---|---|---|
| `chunk_id` | VARCHAR(64) | Primary Key, UUID |
| `media_id` | VARCHAR(64) | Parent recording ID |
| `start_sec` | FLOAT | เริ่มต้น (วินาที) |
| `end_sec` | FLOAT | สิ้นสุด (วินาที) |
| `duration_sec` | FLOAT | ความยาว |
| `minio_url` | VARCHAR(1024) | Presigned URL |
| `object_key` | VARCHAR(512) | MinIO object key |
| `transcript` | VARCHAR(2048) | ASR text |
| `language` | VARCHAR(8) | ISO code (default: "th") |
| `embedding_model` | VARCHAR(64) | ชื่อโมเดล |
| `created_at` | INT64 | Unix timestamp |
| `embedding` | FLOAT_VECTOR(512) | CLAP vector |

**Index:** HNSW / COSINE (M=16, efConstruction=256)

**Methods ที่ implement แล้ว:**
- `upsert_chunks(chunks: list[AudioChunk]) → int`
- `search(query_vector, top_k, media_id_filter, score_threshold) → list[dict]`
- `get_by_chunk_id(chunk_id) → dict | None`
- `list_by_media_id(media_id, limit) → list[dict]`
- `delete_by_media_id(media_id) → int`
- `collection_stats() → dict`

**Import path ที่ใช้:**
```python
from storage.milvus.client import get_milvus_client
from app.core.exceptions import StorageError
```

### 4.3 `docker-compose.yml` (ยืนยันสมบูรณ์)

Services ที่ docker-compose ตั้งไว้:

| Service | Port | หมายเหตุ |
|---|---|---|
| `app` (FastAPI) | 8000 | Main API |
| `worker` (ARQ) | — | Async task worker |
| `milvus-standalone` | 19530 | Vector DB |
| `etcd` | — | Milvus dependency |
| `minio-milvus` | — | MinIO สำหรับ Milvus storage |
| `minio-media` | 9000, 9001 | MinIO สำหรับ audio files |
| `redis` | 6379 | ARQ broker |
| `attu` | 3000 | Milvus Web UI |

### 4.4 `pyproject.toml` — Dependencies ที่สำคัญ

```toml
# AI / ML
faster-whisper         # ASR
open-clip-torch        # CLIP embeddings
transformers           # CLAP (laion/clap-htsat-fused)
sentence-transformers
webrtcvad-wheels       # Voice Activity Detection
rank-bm25              # Hybrid retrieval
networkx               # Graph-based memory (future use)

# Storage
pymilvus               # Milvus client
minio                  # MinIO client

# API / Worker
fastapi
arq                    # Async Redis Queue
structlog              # Structured logging

# Data
datasets               # HuggingFace datasets
soundfile              # Audio I/O
```

---

## 5. ไฟล์ที่เขียนใหม่ในเซสชันนี้

ไฟล์ต่อไปนี้เขียนโดย Claude Sonnet 4.6 ในเซสชันนี้ และควร copy เข้า repo:

### Milestone 1 — Offline Ingestion

| ไฟล์ | สถานะ | คำอธิบาย |
|---|---|---|
| `app/schemas/audio.py` | ✅ สมบูรณ์ | `AudioChunk` dataclass + `SearchResult` dataclass |
| `model_zoo/registry.py` | ✅ สมบูรณ์ | Singleton factory (`get_embedder`, `get_transcriber`, `warm_up_models`) |
| `model_zoo/clap/loader.py` | ✅ สมบูรณ์ | `CLAPEmbedder` — `embed_text`, `embed_audio`, `embed_texts`, `embed_audios` |
| `model_zoo/whisper/loader.py` | ✅ สมบูรณ์ | `WhisperTranscriber` — `transcribe_file`, `transcribe_array`, `transcribe_segments` |
| `scripts/ingest_castle.py` | ✅ สมบูรณ์ | Pipeline ครบ: VAD → Whisper → CLAP → MinIO → Milvus |

### Milestone 2 — Search API

| ไฟล์ | สถานะ | คำอธิบาย |
|---|---|---|
| `app/api/v1/endpoints/search.py` | ✅ สมบูรณ์ | `POST /episodic`, `GET /episodic`, `/health` + BM25 hybrid rerank |

### Milestone 3 — Frontend UI

| ไฟล์ | สถานะ | คำอธิบาย |
|---|---|---|
| `frontend/src/pages/SearchPage.jsx` | ✅ สมบูรณ์ | Query input, result cards, `<audio>` player, pagination, skeleton loading |

---

## 6. สิ่งที่ยังต้องทำต่อ (Remaining Work)

### 🔴 Priority 1 — Blocking (ระบบทำงานไม่ได้ถ้าขาดสิ่งเหล่านี้)

#### 6.1 ตรวจสอบและเขียน `storage/milvus/client.py`

`MilvusService` และ `app/main.py` import จาก path นี้:
```python
from storage.milvus.client import get_milvus_client
```

ถ้าไฟล์นี้ว่างหรือไม่มี ให้เขียน:

```python
# storage/milvus/client.py
import os
from pymilvus import MilvusClient

_client: MilvusClient | None = None

def get_milvus_client() -> MilvusClient:
    global _client
    if _client is None:
        uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
        _client = MilvusClient(uri=uri)
    return _client
```

#### 6.2 ตรวจสอบและเขียน `storage/milvus/collections.py`

```python
# storage/milvus/collections.py
from services.milvus_service import MilvusService

def ensure_all_collections(client) -> None:
    """Called at startup to guarantee all schemas exist."""
    MilvusService(client=client)  # _ensure_collection() called in __init__
```

#### 6.3 ตรวจสอบ `app/core/config.py`

`app/main.py` ใช้ `settings.log_level` และ `settings.app_env`:

```python
# app/core/config.py (minimal working version)
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    log_level: str = "INFO"
    app_env: str = "development"
    milvus_uri: str = "http://milvus-standalone:19530"
    minio_endpoint: str = "minio-media:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_audio: str = "castle-audio"
    clap_model_id: str = "laion/clap-htsat-fused"
    whisper_model_size: str = "base"

    class Config:
        env_file = ".env"

settings = Settings()
```

#### 6.4 ตรวจสอบ `app/core/exceptions.py`

`MilvusService` raise `StorageError`:

```python
# app/core/exceptions.py (minimal)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class StorageError(Exception):
    pass

class NotFoundError(Exception):
    pass

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StorageError)
    async def storage_error_handler(request: Request, exc: StorageError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})
```

#### 6.5 ตรวจสอบ `app/api/v1/router.py`

```python
# app/api/v1/router.py (minimal)
from fastapi import APIRouter
from app.api.v1.endpoints import search

api_router = APIRouter()
api_router.include_router(search.router, prefix="/search", tags=["search"])
```

---

### 🟡 Priority 2 — Important (ทำหลัง P1 แก้แล้ว)

#### 6.6 เพิ่ม `model_zoo` ทุก `__init__.py`

ตอนนี้ทุกไฟล์ `__init__.py` ใน `model_zoo/` ว่างเปล่า  
ให้เพิ่มอย่างน้อย:

```python
# model_zoo/__init__.py
from model_zoo.registry import get_embedder, get_transcriber, warm_up_models
__all__ = ["get_embedder", "get_transcriber", "warm_up_models"]
```

#### 6.7 ตั้งค่า `.env` จาก `.env.example`

```env
# .env (copy จาก .env.example แล้วแก้ค่า)
MILVUS_URI=http://localhost:19530
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_AUDIO=castle-audio
MINIO_SECURE=false
HF_TOKEN=hf_xxxxxxxxxxxx        # ถ้า CASTLE2024 เป็น private dataset
CLAP_MODEL_ID=laion/clap-htsat-fused
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
MODEL_CACHE_DIR=/app/model_cache
```

#### 6.8 ทดสอบ ingestion แบบ dry-run ก่อน

```bash
# 1. Start infra
docker compose up -d milvus-standalone minio-media redis

# 2. Dry-run (ไม่เขียนข้อมูลจริง)
python scripts/ingest_castle.py \
  --max-rows 5 \
  --dry-run \
  --whisper-size tiny

# 3. ถ้า dry-run ผ่าน → ingest จริง
python scripts/ingest_castle.py \
  --max-rows 100 \
  --whisper-size base \
  --chunk-sec 30 \
  --overlap-sec 2
```

#### 6.9 ทดสอบ Search API

```bash
# Start app
docker compose up -d app

# Test search
curl -X POST http://localhost:8000/api/search/episodic \
  -H "Content-Type: application/json" \
  -d '{"query": "eating lunch", "top_k": 5}'

# Health check
curl http://localhost:8000/api/search/episodic/health
```

#### 6.10 เชื่อม Frontend กับ Backend

ตรวจสอบ `frontend/src/api/client.js`:

```javascript
// frontend/src/api/client.js
import axios from "axios";

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 30_000,
});

export default apiClient;
```

และตรวจสอบ `vite.config.js` ให้มี proxy:

```javascript
// vite.config.js
export default {
  server: {
    proxy: {
      "/api": "http://localhost:8000"
    }
  }
}
```

---

### 🟢 Priority 3 — Nice to Have (ทำหลังระบบ run แล้ว)

#### 6.11 ARQ Worker Integration

ปัจจุบัน ingestion ทำแบบ synchronous script  
ควร wrap เป็น ARQ task เพื่อให้ trigger ผ่าน API ได้:

```python
# workers/ingest_worker.py
from arq import cron
from scripts.ingest_castle import process_media_row

async def ingest_task(ctx, row: dict):
    """ARQ task for async ingestion triggered via API."""
    ...
```

#### 6.12 NTCIR Evaluation Integration

`evaluation/` directory มีอยู่แล้วในรีโป  
ต้อง implement:
- `evaluation/run_eval.py` — ส่ง search results ไปที่ NTCIR qrels format
- `evaluation/metrics.py` — คำนวณ MAP, NDCG@10, P@10

#### 6.13 `model_zoo/clip/loader.py` (Visual Modality)

ถ้าจะรองรับ image/video frames จาก lifelog:

```python
# ใช้ open-clip-torch (มีใน pyproject.toml แล้ว)
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
```

#### 6.14 Presigned URL Caching

ปัจจุบัน search endpoint regenerate presigned URL ทุก request  
ควรใช้ Redis cache ด้วย TTL 55 นาที (URL valid 1 ชั่วโมง):

```python
# ใช้ redis client เพื่อ cache url
cache_key = f"minio_url:{object_key}"
cached = await redis.get(cache_key)
if cached:
    return cached.decode()
```

---

## 7. Data Flow รายละเอียด

### 7.1 AudioChunk — Object ที่ไหลระหว่าง components

```python
@dataclass
class AudioChunk:
    chunk_id: str          # UUID4
    media_id: str          # HuggingFace row ID / session ID
    start_sec: float       # offset จากต้น recording
    end_sec: float
    transcript: str        # Whisper output
    language: str          # auto-detected ("th", "en", ...)
    embedding: np.ndarray  # shape=(512,), L2-normalized, float32
    embedding_model: str   # "clap"
    object_key: str        # MinIO key: "{media_id}/{chunk_id}.wav"
    minio_url: str         # presigned URL (7-day TTL จาก ingest)
    created_at: int        # unix timestamp
    # computed property:
    # duration_sec = end_sec - start_sec
```

### 7.2 Search Request/Response Schema

```
POST /api/search/episodic
Content-Type: application/json

Request:
{
  "query": "eating lunch at the cafeteria",  # required
  "top_k": 10,                               # default 10, max 100
  "score_threshold": 0.0,                    # cosine similarity floor
  "media_id_filter": null,                   # filter ด้วย recording ID
  "use_hybrid": true                         # BM25 reranking
}

Response:
{
  "query": "eating lunch at the cafeteria",
  "elapsed_ms": 142.3,
  "total": 8,
  "results": [
    {
      "chunk_id": "uuid...",
      "media_id": "session_001",
      "start_sec": 1234.5,
      "end_sec": 1264.5,
      "duration_sec": 30.0,
      "transcript": "กำลังทานข้าวกลางวันที่โรงอาหาร...",
      "language": "th",
      "score": 0.821,
      "audio_url": "http://minio:9000/castle-audio/...?X-Amz-Signature=...",
      "object_key": "session_001/uuid.wav"
    }
  ]
}
```

### 7.3 CLAP Embedding Logic

```
Text Query ──► ClapProcessor ──► ClapModel.get_text_features() ──► 512-dim ──► L2-norm
Audio Chunk ─► ClapProcessor ──► ClapModel.get_audio_features() ─► 512-dim ──► L2-norm

During ingest:
  embedding = L2_norm((audio_emb + text_emb) / 2)  # fused ถ้ามี transcript
  หรือ
  embedding = audio_emb  # ถ้าไม่มี transcript
```

---

## 8. Environment Variables ครบชุด

```env
# === Milvus ===
MILVUS_URI=http://milvus-standalone:19530

# === MinIO (Media) ===
MINIO_ENDPOINT=minio-media:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_AUDIO=castle-audio
MINIO_SECURE=false

# === Redis (ARQ) ===
REDIS_URL=redis://redis:6379

# === HuggingFace ===
HF_TOKEN=hf_xxxxxxxxxx     # ต้องการถ้า CASTLE2024 เป็น private

# === Models ===
CLAP_MODEL_ID=laion/clap-htsat-fused
WHISPER_MODEL_SIZE=base    # tiny/base/small/medium/large-v2/large-v3
WHISPER_DEVICE=cpu         # cpu หรือ cuda
WHISPER_COMPUTE_TYPE=int8  # int8 (CPU) หรือ float16 (GPU)
MODEL_CACHE_DIR=/app/model_cache

# === App ===
APP_ENV=development
LOG_LEVEL=INFO
```

---

## 9. คำสั่งสำคัญ

```bash
# === Start ทุก service ===
docker compose up -d

# === ดู logs ===
docker compose logs -f app
docker compose logs -f worker

# === Ingest dataset (dry-run) ===
docker compose exec app python scripts/ingest_castle.py --max-rows 5 --dry-run

# === Ingest จริง ===
docker compose exec app python scripts/ingest_castle.py \
  --max-rows 1000 \
  --whisper-size base \
  --chunk-sec 30

# === Test search API ===
curl -X POST http://localhost:8000/api/search/episodic \
  -H "Content-Type: application/json" \
  -d '{"query": "lunch break", "top_k": 5, "use_hybrid": true}'

# === Attu (Milvus UI) ===
open http://localhost:3000

# === MinIO Console ===
open http://localhost:9001
# login: minioadmin / minioadmin
```

---

## 10. Known Issues & Gotchas

| ปัญหา | สาเหตุ | วิธีแก้ |
|---|---|---|
| `Cannot find working tool` ตอน extract `.rar` | ต้องการ `unrar` binary | ใช้ `7zip` หรือ `p7zip-full` |
| `model_zoo/` ไฟล์ทั้งหมดว่างเปล่า | archive extract ไม่สมบูรณ์ + ยังไม่เขียน | Copy ไฟล์ที่สร้างในเซสชันนี้เข้าไป |
| CLAP ต้องการ audio 48 kHz | CLAP native SR | `_resample(wav, original_sr, 48000)` ก่อน embed |
| Whisper ต้องการ 16 kHz | Whisper architecture | `_resample(wav, original_sr, 16000)` |
| Milvus `duration_sec` ไม่มีใน schema | computed property | คำนวณจาก `end_sec - start_sec` เสมอ |
| Presigned URL หมดอายุ | MinIO TTL | Search endpoint regenerate ทุก query (1h TTL) |
| `MilvusClient.upsert` return format | version-dependent | ใช้ `.get("upsert_count", len(records))` |

---

## 11. สรุปสิ่งที่ต้องทำทันที (Quick Start Checklist)

```
[ ] 1. Copy ไฟล์ที่เขียนในเซสชันนี้เข้า repo (ดูหัวข้อ 5)
[ ] 2. ตรวจสอบ / เขียน storage/milvus/client.py (หัวข้อ 6.1)
[ ] 3. ตรวจสอบ / เขียน storage/milvus/collections.py (หัวข้อ 6.2)
[ ] 4. ตรวจสอบ / เขียน app/core/config.py (หัวข้อ 6.3)
[ ] 5. ตรวจสอบ / เขียน app/core/exceptions.py (หัวข้อ 6.4)
[ ] 6. ตรวจสอบ / เขียน app/api/v1/router.py (หัวข้อ 6.5)
[ ] 7. Copy .env.example → .env และใส่ค่าจริง (หัวข้อ 6.7)
[ ] 8. docker compose up -d
[ ] 9. python scripts/ingest_castle.py --max-rows 5 --dry-run
[ ] 10. curl POST /api/search/episodic ทดสอบ
[ ] 11. เปิด frontend และทดสอบ SearchPage
```

---

*เอกสารนี้สร้างโดย Claude Sonnet 4.6 | เซสชัน: NTCIR-19 CSAT Implementation*  
*อัปเดตล่าสุด: ครอบคลุม Milestone 1, 2, 3 ทั้งหมด*