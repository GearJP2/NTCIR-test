"""Paper-style selection and CSAT export of already video-reranked candidates."""

import re
from typing import TextIO

from pydantic import BaseModel, Field, model_validator


class CSATResult(BaseModel):
    source_relative_path: str
    start_ms: int = Field(ge=0, strict=True)
    end_ms: int = Field(gt=0, strict=True)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    reranked: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_range(self) -> "CSATResult":
        match = re.fullmatch(r"day([1-9]\d*)/([^/,\s]+)/video/(\d{2})\.mp4",
                             self.source_relative_path)
        if match is None or int(match[3]) > 23:
            raise ValueError("expected dayN/source/video/HH.mp4")
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must exceed start_ms")
        if self.recording_start_ms + self.end_ms >= 86_400_000:
            raise ValueError("result must not cross midnight")
        return self

    @property
    def recording_start_ms(self) -> int:
        # shortcut: HH.mp4 is assumed hour-aligned, use audited anchors for other naming schemes.
        return int(self.source_relative_path.rsplit("/", 1)[1][:2]) * 3_600_000

    @property
    def source_key(self) -> tuple[str, str]:
        parts = self.source_relative_path.split("/")
        return parts[0], parts[1]


def select_csat_results(results: list[CSATResult]) -> list[CSATResult]:
    """PDF §§3.7/4.1: >=0.5, stable score order, no day/source overlaps, at most 100."""
    selected: list[CSATResult] = []
    for result in sorted(results, key=lambda item: item.score, reverse=True):
        if not result.reranked or result.score < 0.5:
            continue
        start = result.recording_start_ms + result.start_ms
        end = result.recording_start_ms + result.end_ms
        if any(
            previous.source_key == result.source_key
            and start < previous.recording_start_ms + previous.end_ms
            and previous.recording_start_ms + previous.start_ms < end
            for previous in selected
        ):
            continue
        selected.append(result)
        if len(selected) == 100:
            break
    return selected


def write_csat_submission(
    handle: TextIO,
    per_topic: dict[str, list[CSATResult]],
    *,
    group_id: str,
    run_id: str,
) -> int:
    for identifier in (group_id, run_id, *per_topic):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError(f"invalid submission identifier: {identifier!r}")
    count = 0
    for topic, results in per_topic.items():
        for result in select_csat_results(results):
            day, source = result.source_key
            boundaries = []
            for offset in (result.start_ms, result.end_ms):
                seconds = (result.recording_start_ms + offset) // 1000
                boundaries.append(f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:"
                                  f"{seconds % 60:02}")
            if boundaries[0] == boundaries[1]:
                raise ValueError("result collapses to zero duration at submission second precision")
            handle.write(", ".join([group_id, run_id, topic, day[3:], *boundaries,
                                    source, f"{result.score:.6f}"]) + "\n")
            count += 1
    return count
