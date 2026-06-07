"""
QueryService — Text-to-Episodic-Memory search with WorldMM-style LLM reasoning.

Pipeline
--------
    text query
        → TextQueryEncoder (CLAP text encoder or Wav2Vec2 mean-pool fallback)
            → query_vector  (same embedding space as the indexed audio chunks)
        → MilvusService.search(query_vector, top_k, ...)
            → list[EpisodicHit]  (sorted by cosine similarity)
        → WorldMMPromptBuilder.build(query, hits)
            → structured prompt string
        → LLMBackend.reason(prompt)
            → LLMReasoning  (answer | best_chunk_ids | reasoning | confidence)

The WorldMM reasoning step is optional; set `use_llm=False` to get raw hits.

LLM backends (in priority order)
----------------------------------
1. OpenAI  — async `openai` client (requires OPENAI_API_KEY)
2. Ollama  — local model via HTTP JSON API (requires OLLAMA_URL)
3. None    — returns a no-op LLMReasoning with model_used="none"
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import structlog

from app.core.config import settings
from app.schemas.search import EpisodicHit, EpisodicSearchResponse, LLMReasoning

logger = structlog.get_logger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

class QueryService:
    """
    Orchestrates the full text-to-episodic-memory query pipeline.

    Parameters
    ----------
    embedder_name:
        Must match the model used at ingest time.
        'clap'     → 512-dim, shares audio/text embedding space via CLAP.
        'wav2vec2' → 768-dim, text encoded by a cross-modal sentence-transformer.
    """

    def __init__(self, embedder_name: Literal["clap", "wav2vec2"] = "clap") -> None:
        self.embedder_name = embedder_name
        self._encoder = TextQueryEncoder(embedder_name)

    async def run(
        self,
        query: str,
        top_k: int = 5,
        media_id_filter: str | None = None,
        score_threshold: float | None = None,
        use_llm: bool = True,
    ) -> EpisodicSearchResponse:
        """
        Execute the full pipeline and return a fully populated response.

        Steps
        -----
        1. Encode the text query into a vector.
        2. Search `csat_episodic_memory` in Milvus.
        3. Hydrate results into EpisodicHit objects (with presigned MinIO URLs).
        4. Optionally run the WorldMM LLM reasoning step.
        """
        # ── Step 1: Text → Vector ────────────────────────────────────────────
        query_vector = self._encoder.encode(query)
        logger.info(
            "query_service.encode",
            embedder=self.embedder_name,
            dim=query_vector.shape[0],
        )

        # ── Step 2: Vector → Milvus ANN search ───────────────────────────────
        from storage.milvus.milvus_service import MilvusService
        raw_hits = MilvusService(embedding_dim=query_vector.shape[0]).search(
            query_vector=query_vector,
            top_k=top_k,
            media_id_filter=media_id_filter,
            score_threshold=score_threshold,
        )
        logger.info("query_service.search", n_hits=len(raw_hits))

        # ── Step 3: Hydrate hits (refresh presigned MinIO URLs) ───────────────
        hits = _hydrate_hits(raw_hits)

        # ── Step 4: WorldMM LLM reasoning ────────────────────────────────────
        reasoning: LLMReasoning | None = None
        if use_llm and hits:
            prompt = WorldMMPromptBuilder.build(query, hits)
            reasoning = await _LLMBackend.reason(prompt, hits)

        return EpisodicSearchResponse(
            query=query,
            total_hits=len(hits),
            hits=hits,
            reasoning=reasoning,
            query_vector_dim=int(query_vector.shape[0]),
            embedder_used=self.embedder_name,
        )


# ── Text query encoder ────────────────────────────────────────────────────────

class TextQueryEncoder:
    """
    Encodes a natural-language query into the same vector space used at ingest.

    CLAP mode (default):
        Uses the CLAP text tower — the text and audio towers share a joint
        embedding space, so cosine similarity is meaningful across modalities.

    Wav2Vec2 mode:
        Wav2Vec2 has no text tower. We fall back to a sentence-transformer
        (all-mpnet-base-v2) that is also used for the `text_transcripts`
        collection, giving reasonable cross-modal alignment for transcript-rich
        collections.
    """

    def __init__(self, embedder_name: Literal["clap", "wav2vec2"]) -> None:
        self._name = embedder_name

    def encode(self, query: str) -> np.ndarray:
        if self._name == "clap":
            return self._encode_clap(query)
        return self._encode_wav2vec_fallback(query)

    # ── CLAP text tower ───────────────────────────────────────────────────────
    def _encode_clap(self, query: str) -> np.ndarray:
        """
        Encode using the CLAP text tower (laion/clap-htsat-unfused via HuggingFace
        transformers). Produces a 512-dim L2-normalised vector in the joint
        audio-text embedding space.
        """
        import torch
        from transformers import AutoProcessor, ClapModel

        model_id = "laion/clap-htsat-unfused"
        cache_dir = settings.model_cache_dir

        # Lazy load with double-checked locking via registry
        try:
            from model_zoo.registry import ModelRegistry
            # CLAP registered in registry uses laion_clap lib; for the HF
            # transformers variant (which has a true text tower) we manage it
            # here directly.
            _registry = _ClapTextTowerCache.get(model_id, cache_dir)
        except Exception as exc:
            raise RuntimeError(f"Failed to load CLAP text encoder: {exc}") from exc

        processor, model, device = _registry
        inputs = processor(
            text=[query],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features[0].cpu().numpy().astype(np.float32)

    # ── Sentence-transformer fallback (wav2vec2 ingest path) ─────────────────
    def _encode_wav2vec_fallback(self, query: str) -> np.ndarray:
        """
        sentence-transformers produces 768-dim vectors that align with the
        transcript chunks — a reasonable proxy when the collection was indexed
        with wav2vec2 embeddings.
        """
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            settings.text_encoder_model,
            cache_folder=settings.model_cache_dir,
        )
        vec = model.encode([query], normalize_embeddings=True)[0]
        return vec.astype(np.float32)


class _ClapTextTowerCache:
    """Simple module-level singleton for the HuggingFace CLAP model."""
    _instance: tuple | None = None

    @classmethod
    def get(cls, model_id: str, cache_dir: str) -> tuple:
        if cls._instance is None:
            import torch
            from transformers import AutoProcessor, ClapModel

            logger.info("clap_text_tower.loading", model=model_id)
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            model = ClapModel.from_pretrained(model_id, cache_dir=cache_dir)
            device = (
                settings.device
                if settings.device != "cuda" or torch.cuda.is_available()
                else "cpu"
            )
            model = model.to(device).eval()
            cls._instance = (processor, model, device)
            logger.info("clap_text_tower.ready", device=device)
        return cls._instance


# ── WorldMM prompt builder ────────────────────────────────────────────────────

class WorldMMPromptBuilder:
    """
    Builds a structured "WorldMM Agent" prompt that combines:
      - The user's natural-language query
      - A numbered memory context block (top-K retrieved chunks)
      - A reasoning task specification with JSON output schema

    The prompt asks the LLM to:
      1. Identify which memory segment(s) directly answer the query
      2. Explain its reasoning step-by-step
      3. Return structured JSON: answer, best_chunk_ids, reasoning, confidence
    """

    _SYSTEM = (
        "You are a Multimodal Memory Agent (WorldMM) specialising in episodic "
        "memory retrieval. You receive a user query and a ranked list of audio "
        "memory segments retrieved from a semantic vector store. Each segment "
        "contains a transcript and precise timestamps.\n\n"
        "Your task is to analyse the retrieved segments, identify the one(s) "
        "that best answer the query, and explain your reasoning.\n\n"
        "IMPORTANT: You must respond with a single valid JSON object matching "
        "the schema below — no markdown fences, no extra text.\n\n"
        "Schema:\n"
        "{\n"
        '  "answer": "<direct answer to the user query>",\n'
        '  "best_chunk_ids": ["<chunk_id_1>", ...],\n'
        '  "reasoning": "<step-by-step reasoning>",\n'
        '  "confidence": <float 0.0–1.0>\n'
        "}"
    )

    @classmethod
    def build(cls, query: str, hits: list[EpisodicHit]) -> str:
        """
        Assemble the full prompt string.
        Returns (system_prompt, user_prompt) ready for a chat API.
        The returned string is a JSON-serialised dict with keys
        'system' and 'user' so the LLM backend can dispatch correctly.
        """
        segments_block = cls._format_segments(hits)

        user_prompt = (
            f"## User Query\n{query}\n\n"
            f"## Retrieved Memory Segments ({len(hits)} results, ranked by cosine similarity)\n"
            f"{segments_block}\n\n"
            "## Task\n"
            "Analyse the segments above and produce the JSON response."
        )

        return json.dumps({"system": cls._SYSTEM, "user": user_prompt})

    @staticmethod
    def _format_segments(hits: list[EpisodicHit]) -> str:
        lines: list[str] = []
        for rank, hit in enumerate(hits, start=1):
            ts_start = _seconds_to_timestamp(hit.start_sec)
            ts_end = _seconds_to_timestamp(hit.end_sec)
            transcript = (hit.transcript or "[no transcript]").strip()
            lines.append(
                f"[{rank}] chunk_id={hit.chunk_id}\n"
                f"     Media : {hit.media_id}\n"
                f"     Time  : {ts_start} → {ts_end}  ({hit.duration_sec:.1f}s)\n"
                f"     Score : {hit.score:.4f}\n"
                f"     Audio : {hit.minio_url or 'N/A'}\n"
                f"     Text  : \"{transcript}\""
            )
        return "\n\n".join(lines)


# ── LLM backend ───────────────────────────────────────────────────────────────

class _LLMBackend:
    """
    Dispatches to the configured LLM provider.

    Provider priority:
      openai  → async OpenAI chat completions (GPT-4o-mini default)
      ollama  → local Ollama /api/chat JSON endpoint
      none    → returns a no-op LLMReasoning immediately
    """

    @classmethod
    async def reason(
        cls,
        prompt_json: str,
        hits: list[EpisodicHit],
    ) -> LLMReasoning:
        provider = settings.llm_provider.lower()

        if provider == "openai" and settings.openai_api_key:
            return await cls._openai(prompt_json, hits)
        if provider == "ollama":
            return await cls._ollama(prompt_json, hits)

        logger.info("llm_backend.skip", provider=provider)
        return _noop_reasoning(hits)

    # ── OpenAI ────────────────────────────────────────────────────────────────
    @classmethod
    async def _openai(
        cls, prompt_json: str, hits: list[EpisodicHit]
    ) -> LLMReasoning:
        import openai

        payload = json.loads(prompt_json)
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": payload["system"]},
                    {"role": "user",   "content": payload["user"]},
                ],
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            logger.info(
                "llm_backend.openai.ok",
                model=settings.llm_model,
                tokens=response.usage.total_tokens if response.usage else None,
            )
            return _parse_llm_json(raw, model_used=settings.llm_model, hits=hits)

        except openai.OpenAIError as exc:
            logger.error("llm_backend.openai.error", error=str(exc))
            return _noop_reasoning(hits, error=str(exc))

    # ── Ollama ────────────────────────────────────────────────────────────────
    @classmethod
    async def _ollama(
        cls, prompt_json: str, hits: list[EpisodicHit]
    ) -> LLMReasoning:
        import httpx

        payload = json.loads(prompt_json)
        # Merge system + user into a single message for Ollama's /api/generate
        combined = f"{payload['system']}\n\n{payload['user']}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{settings.ollama_url}/api/generate",
                    json={
                        "model": settings.llm_model,
                        "prompt": combined,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": settings.llm_temperature,
                            "num_predict": settings.llm_max_tokens,
                        },
                    },
                )
                response.raise_for_status()
                raw = response.json().get("response", "{}")
            logger.info("llm_backend.ollama.ok", model=settings.llm_model)
            return _parse_llm_json(raw, model_used=f"ollama/{settings.llm_model}", hits=hits)

        except Exception as exc:
            logger.error("llm_backend.ollama.error", error=str(exc))
            return _noop_reasoning(hits, error=str(exc))


# ── Helper functions ──────────────────────────────────────────────────────────

def _hydrate_hits(raw_hits: list[dict]) -> list[EpisodicHit]:
    """
    Convert raw Milvus result dicts into EpisodicHit objects.
    Refreshes presigned MinIO URLs so they are always valid.
    """
    from storage.minio.operations import get_presigned_url

    hits: list[EpisodicHit] = []
    for r in raw_hits:
        object_key = r.get("object_key") or ""
        minio_url = ""
        if object_key:
            try:
                minio_url = get_presigned_url(object_key, expires_hours=1)
            except Exception as exc:
                logger.warning(
                    "query_service.presign_failed",
                    key=object_key,
                    error=str(exc),
                )

        hits.append(
            EpisodicHit(
                chunk_id=r.get("chunk_id", ""),
                media_id=r.get("media_id", ""),
                score=float(r.get("score", 0.0)),
                start_sec=float(r.get("start_sec", 0.0)),
                end_sec=float(r.get("end_sec", 0.0)),
                duration_sec=float(r.get("duration_sec", 0.0)),
                transcript=r.get("transcript") or "",
                language=r.get("language") or "th",
                minio_url=minio_url,
                object_key=object_key,
                embedding_model=r.get("embedding_model") or "clap",
                created_at=int(r.get("created_at") or 0),
            )
        )
    return hits


def _parse_llm_json(
    raw: str,
    model_used: str,
    hits: list[EpisodicHit],
) -> LLMReasoning:
    """
    Parse and validate the JSON returned by the LLM.
    Falls back to a best-effort LLMReasoning on any parse error.
    """
    # Strip accidental markdown fences the model may still emit
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())

    try:
        data = json.loads(raw)
        return LLMReasoning(
            answer=str(data.get("answer", "")).strip(),
            best_chunk_ids=[str(x) for x in data.get("best_chunk_ids", [])],
            reasoning=str(data.get("reasoning", "")).strip(),
            confidence=_clamp(float(data.get("confidence", 0.5))),
            model_used=model_used,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("query_service.parse_llm_json_failed", error=str(exc), raw=raw[:200])
        # Graceful degradation: return the top hit as best guess
        return LLMReasoning(
            answer=raw[:500] if raw else "Could not parse LLM response.",
            best_chunk_ids=[hits[0].chunk_id] if hits else [],
            reasoning=f"JSON parse error: {exc}",
            confidence=0.0,
            model_used=model_used,
        )


def _noop_reasoning(
    hits: list[EpisodicHit],
    error: str | None = None,
) -> LLMReasoning:
    top_ids = [h.chunk_id for h in hits[:3]]
    return LLMReasoning(
        answer=f"Top result: {hits[0].transcript[:200]}" if hits else "No results found.",
        best_chunk_ids=top_ids,
        reasoning=error or "LLM reasoning disabled or not configured.",
        confidence=0.0,
        model_used="none",
    )


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert float seconds → HH:MM:SS.mmm string."""
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
