from pydantic import BaseModel


class IngestRequest(BaseModel):
    media_id: str
    title: str = ""
    language: str = "th"


class IngestResponse(BaseModel):
    media_id: str
    job_id: str
    status: str  # queued | processing | done | failed
