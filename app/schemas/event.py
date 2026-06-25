from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EventKind(str, Enum):
    SEMANTIC_MICRO = "semantic_micro"
    SEMANTIC_MACRO = "semantic_macro"
    FIXED_30S = "fixed_30s"
    FIXED_120S = "fixed_120s"


class HeartRateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_bpm: float | None = None
    min_bpm: float | None = None
    max_bpm: float | None = None
    std_bpm: float | None = None
    slope_bpm_s: float | None = None
    baseline_delta: float | None = None
    valid_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class GazeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid_fixation_count: int = Field(default=0, ge=0)
    total_fixation_duration_ms: int = Field(default=0, ge=0)
    mean_fixation_duration_ms: float | None = Field(default=None, ge=0)
    attended_objects: list[str] = Field(default_factory=list)
    valid_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class ThermalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_ids: list[str] = Field(default_factory=list)
    image_count: int = Field(default=0, ge=0)
    nearest_time_delta_ms: int | None = Field(default=None, ge=0)
    valid: bool = False
    note: str = ""

    @model_validator(mode="after")
    def validate_image_count(self) -> ThermalSummary:
        if self.image_count != len(self.image_ids):
            raise ValueError("thermal.image_count must match thermal.image_ids")
        if self.valid and self.image_count == 0:
            raise ValueError("thermal.valid requires at least one image")
        return self


class ModalityCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video: bool = True
    transcript: bool = False
    heart_rate: bool = False
    gaze: bool = False
    thermal: bool = False


class RecordingRecord(BaseModel):
    """Minimal audited CASTLE recording needed to construct event intervals."""

    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    video_uri: str = Field(min_length=1)


class EventRecord(BaseModel):
    """Canonical handoff record shared by CASTLE preparation and retrieval."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    processing_version: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    participant_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    event_kind: EventKind
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    parent_event_id: str | None = None
    boundary_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_context_start_ms: int | None = Field(default=None, ge=0)
    retrieval_context_end_ms: int | None = Field(default=None, gt=0)
    video_uri: str = Field(min_length=1)
    transcript: str = ""
    heart_rate: HeartRateSummary | None = None
    gaze: GazeSummary | None = None
    thermal: ThermalSummary | None = None
    coverage: ModalityCoverage
    raw_evidence_uris: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_event_invariants(self) -> EventRecord:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.duration_ms != self.end_ms - self.start_ms:
            raise ValueError("duration_ms must equal end_ms - start_ms")

        if self.event_kind == EventKind.SEMANTIC_MICRO and not self.parent_event_id:
            raise ValueError("semantic_micro events require parent_event_id")
        if self.event_kind != EventKind.SEMANTIC_MICRO and self.parent_event_id:
            raise ValueError("only semantic_micro events may have parent_event_id")

        context_start = self.retrieval_context_start_ms
        context_end = self.retrieval_context_end_ms
        if (context_start is None) != (context_end is None):
            raise ValueError("retrieval context start and end must be provided together")
        if context_start is not None and context_end is not None:
            if context_start > self.start_ms or context_end < self.end_ms:
                raise ValueError("retrieval context must contain the core event interval")
            if context_end <= context_start:
                raise ValueError("retrieval context end must be greater than start")

        expected_coverage = {
            "heart_rate": self.heart_rate is not None,
            "gaze": self.gaze is not None,
            "thermal": self.thermal is not None and self.thermal.valid,
        }
        for modality, expected in expected_coverage.items():
            if getattr(self.coverage, modality) != expected:
                raise ValueError(
                    f"coverage.{modality} must match the attached {modality} evidence"
                )
        if self.coverage.transcript != bool(self.transcript.strip()):
            raise ValueError("coverage.transcript must match transcript availability")
        return self
