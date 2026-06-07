import structlog
from fastapi import APIRouter
from pydantic import BaseModel

logger = structlog.get_logger(__name__)
router = APIRouter()


class ExpandRequest(BaseModel):
    query: str
    num_expansions: int = 3


class ExpandResponse(BaseModel):
    original: str
    expansions: list[str]


@router.post("/", response_model=ExpandResponse)
async def expand_query(request: ExpandRequest):
    """
    Rewrite / expand a search query using a local LLM or OpenAI.
    Returns `num_expansions` alternative phrasings to boost recall.
    """
    from services.memory.summarizer import expand_query as llm_expand
    expansions = await llm_expand(request.query, n=request.num_expansions)
    return ExpandResponse(original=request.query, expansions=expansions)
