from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median, pstdev

from app.schemas.event import EventRecord, HeartRateSummary
from services.dataset.castle_metadata import parse_elapsed_time_ms


@dataclass(frozen=True)
class HeartRateSample:
    offset_ms: int
    bpm: float
    confidence: float


@dataclass(frozen=True)
class HeartRateSource:
    participant_id: str
    day: str
    path: Path
    source_uri: str


@dataclass(frozen=True)
class HeartRateEnrichmentRow:
    event_id: str
    video_id: str
    event_start_ms: int
    event_end_ms: int
    clock_start_ms: int | None
    clock_end_ms: int | None
    overlapping_samples: int
    valid_samples: int
    mean_bpm: float | None
    min_bpm: float | None
    max_bpm: float | None
    std_bpm: float | None
    slope_bpm_s: float | None
    baseline_delta: float | None
    valid_ratio: float


def load_heart_rate_samples(path: Path) -> list[HeartRateSample]:
    samples: list[HeartRateSample] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            time_value = (row.get("time") or "").strip()
            bpm_value = (row.get("bpm") or "").strip()
            confidence_value = (row.get("confidence") or "").strip()
            if not time_value or not bpm_value:
                continue
            samples.append(
                HeartRateSample(
                    offset_ms=parse_elapsed_time_ms(time_value),
                    bpm=float(bpm_value),
                    confidence=float(confidence_value or 0),
                )
            )
    return sorted(samples, key=lambda sample: sample.offset_ms)


def load_recording_clock_starts(inventory_path: Path) -> dict[str, int]:
    starts: dict[str, int] = {}
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row["source"]
            if source.startswith("auxiliary/"):
                continue
            offset = row.get("first_offset_ms") or ""
            if not offset:
                continue
            video_id = _recording_source_to_video_id(source)
            starts[video_id] = int(offset)
    return starts


def load_heart_rate_sources(inventory_path: Path) -> dict[tuple[str, str], HeartRateSource]:
    sources: dict[tuple[str, str], HeartRateSource] = {}
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row["source"]
            parts = source.split("/")
            if parts[:2] != ["auxiliary", "heartrate"] or len(parts) != 4:
                continue
            participant_id = parts[2]
            day = parts[3]
            sources[(day, participant_id)] = HeartRateSource(
                participant_id=participant_id,
                day=day,
                path=Path(row["path"]),
                source_uri=source,
            )
    return sources


def summarize_heart_rate_enrichment(
    events: list[EventRecord],
    samples: list[HeartRateSample],
    *,
    recording_clock_starts_ms: dict[str, int],
    min_confidence: float = 1.0,
    min_bpm: float = 30.0,
    max_bpm: float = 220.0,
) -> list[HeartRateEnrichmentRow]:
    baseline = _baseline_bpm(
        samples,
        min_confidence=min_confidence,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
    rows: list[HeartRateEnrichmentRow] = []
    for event in events:
        recording_start = recording_clock_starts_ms.get(event.video_id)
        clock_start: int | None = None
        clock_end: int | None = None
        overlapping: list[HeartRateSample] = []
        valid: list[HeartRateSample] = []
        summary: HeartRateSummary | None = None
        if recording_start is not None:
            clock_start = recording_start + event.start_ms
            clock_end = recording_start + event.end_ms
            overlapping = _overlapping_samples(samples, clock_start, clock_end)
            valid = _valid_samples(
                overlapping,
                min_confidence=min_confidence,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            )
            summary = _summarize_overlapping_samples(
                overlapping,
                valid,
                baseline_bpm=baseline,
            )
        rows.append(
            HeartRateEnrichmentRow(
                event_id=event.event_id,
                video_id=event.video_id,
                event_start_ms=event.start_ms,
                event_end_ms=event.end_ms,
                clock_start_ms=clock_start,
                clock_end_ms=clock_end,
                overlapping_samples=len(overlapping),
                valid_samples=len(valid),
                mean_bpm=summary.mean_bpm if summary else None,
                min_bpm=summary.min_bpm if summary else None,
                max_bpm=summary.max_bpm if summary else None,
                std_bpm=summary.std_bpm if summary else None,
                slope_bpm_s=summary.slope_bpm_s if summary else None,
                baseline_delta=summary.baseline_delta if summary else None,
                valid_ratio=summary.valid_ratio if summary else 0.0,
            )
        )
    return rows


def write_heart_rate_enrichment_summary(
    path: Path,
    rows: list[HeartRateEnrichmentRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(HeartRateEnrichmentRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: getattr(row, field) for field in fieldnames}
            )


def attach_heart_rate_to_events(
    events: list[EventRecord],
    samples: list[HeartRateSample],
    *,
    recording_clock_starts_ms: dict[str, int],
    source_uri: str,
    min_confidence: float = 1.0,
    min_bpm: float = 30.0,
    max_bpm: float = 220.0,
) -> list[EventRecord]:
    baseline = _baseline_bpm(
        samples,
        min_confidence=min_confidence,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
    enriched: list[EventRecord] = []
    for event in events:
        recording_start = recording_clock_starts_ms.get(event.video_id)
        summary: HeartRateSummary | None = None
        if recording_start is not None:
            summary = summarize_heart_rate(
                samples,
                start_ms=recording_start + event.start_ms,
                end_ms=recording_start + event.end_ms,
                baseline_bpm=baseline,
                min_confidence=min_confidence,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            )

        coverage = event.coverage.model_copy(update={"heart_rate": summary is not None})
        raw_evidence_uris = dict(event.raw_evidence_uris)
        if summary is not None:
            raw_evidence_uris["heart_rate"] = [source_uri]
        else:
            raw_evidence_uris.pop("heart_rate", None)
        enriched.append(
            event.model_copy(
                update={
                    "heart_rate": summary,
                    "coverage": coverage,
                    "raw_evidence_uris": raw_evidence_uris,
                }
            )
        )
    return enriched


def summarize_heart_rate(
    samples: list[HeartRateSample],
    *,
    start_ms: int,
    end_ms: int,
    baseline_bpm: float | None = None,
    min_confidence: float = 1.0,
    min_bpm: float = 30.0,
    max_bpm: float = 220.0,
) -> HeartRateSummary | None:
    overlapping = _overlapping_samples(samples, start_ms, end_ms)
    if not overlapping:
        return None

    valid = _valid_samples(
        overlapping,
        min_confidence=min_confidence,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )
    return _summarize_overlapping_samples(
        overlapping,
        valid,
        baseline_bpm=baseline_bpm,
    )


def _summarize_overlapping_samples(
    overlapping: list[HeartRateSample],
    valid: list[HeartRateSample],
    *,
    baseline_bpm: float | None,
) -> HeartRateSummary:
    valid_ratio = len(valid) / len(overlapping)
    if not valid:
        return HeartRateSummary(valid_ratio=valid_ratio)
    bpms = [sample.bpm for sample in valid]
    mean_bpm = sum(bpms) / len(bpms)
    return HeartRateSummary(
        mean_bpm=mean_bpm,
        min_bpm=min(bpms),
        max_bpm=max(bpms),
        std_bpm=pstdev(bpms) if len(bpms) > 1 else 0.0,
        slope_bpm_s=_slope_bpm_per_second(valid),
        baseline_delta=mean_bpm - baseline_bpm if baseline_bpm is not None else None,
        valid_ratio=valid_ratio,
    )


def _overlapping_samples(
    samples: list[HeartRateSample],
    start_ms: int,
    end_ms: int,
) -> list[HeartRateSample]:
    return [sample for sample in samples if start_ms <= sample.offset_ms < end_ms]


def _valid_samples(
    samples: list[HeartRateSample],
    *,
    min_confidence: float,
    min_bpm: float,
    max_bpm: float,
) -> list[HeartRateSample]:
    return [
        sample
        for sample in samples
        if min_confidence <= sample.confidence and min_bpm <= sample.bpm <= max_bpm
    ]


def _recording_source_to_video_id(source: str) -> str:
    day, participant_id, filename = source.split("/")
    return f"{day}_{participant_id}_{Path(filename).stem}"


def _baseline_bpm(
    samples: list[HeartRateSample],
    *,
    min_confidence: float,
    min_bpm: float,
    max_bpm: float,
) -> float | None:
    bpms = [
        sample.bpm
        for sample in samples
        if min_confidence <= sample.confidence and min_bpm <= sample.bpm <= max_bpm
    ]
    return float(median(bpms)) if bpms else None


def _slope_bpm_per_second(samples: list[HeartRateSample]) -> float | None:
    if len(samples) < 2:
        return None
    xs = [sample.offset_ms / 1000 for sample in samples]
    ys = [sample.bpm for sample in samples]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return numerator / denominator
