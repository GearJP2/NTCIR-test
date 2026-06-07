import structlog

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a concise audio/video content summarizer. "
    "Given a short audio segment, produce a 1-2 sentence factual summary in English."
)


def summarize_segment(segment_id: str, audio_path: str) -> str:
    """
    Generate a short summary for an audio segment.
    Tries OpenAI GPT-4o first; falls back to a local llama.cpp model;
    falls back to empty string if neither is available.
    """
    from app.core.config import settings

    if settings.openai_api_key:
        return _summarize_openai(audio_path)
    return _summarize_local(audio_path)


def _summarize_openai(audio_path: str) -> str:
    try:
        import openai
        from app.core.config import settings

        client = openai.OpenAI(api_key=settings.openai_api_key)
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": transcript.text},
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("summarizer.openai.failed", error=str(exc))
        return ""


def _summarize_local(audio_path: str) -> str:
    """Stub for local llama.cpp summarisation — plug in your model here."""
    return ""


async def expand_query(query: str, n: int = 3) -> list[str]:
    """
    Expand a search query into `n` alternative phrasings using an LLM.
    Falls back to returning the original query repeated if no LLM is available.
    """
    from app.core.config import settings

    if not settings.openai_api_key:
        logger.warning("summarizer.expand.no_llm")
        return [query] * n

    import openai
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    prompt = (
        f"Generate {n} semantically diverse reformulations of the following search query. "
        f"Return only the queries, one per line.\n\nQuery: {query}"
    )
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    lines = response.choices[0].message.content.strip().splitlines()
    return [l.strip() for l in lines if l.strip()][:n]
