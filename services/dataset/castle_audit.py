from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

CASTLE_REPO_ID = "CASTLE-Dataset/CASTLE2024"


@dataclass(frozen=True)
class RepositoryFile:
    path: str
    size_bytes: int


@dataclass(frozen=True)
class CoverageRow:
    day: str
    participant_id: str
    videos: int
    missing_videos: int
    transcripts: int
    metadata_csv: int
    metadata_gpx: int
    heart_rate_files: int
    gaze_files: int
    thermal_files: int
    auxiliary_photos: int
    auxiliary_videos: int


def build_castle_audit(files: Iterable[RepositoryFile]) -> dict:
    materialized = [
        file
        for file in files
        if not file.path.endswith(".sha256") and file.path not in {"index.html"}
    ]
    category_counts: Counter[str] = Counter()
    category_bytes: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    sensor_counts: Counter[str] = Counter()
    participants: set[str] = set()
    days: set[str] = set()

    for file in materialized:
        category = file_category(file.path)
        category_counts[category] += 1
        category_bytes[category] += file.size_bytes
        extension_counts[PurePosixPath(file.path).suffix.lower() or "[none]"] += 1

        parts = file.path.split("/")
        if len(parts) >= 4 and parts[0] == "main":
            days.add(parts[1])
            participants.add(parts[2])
            if parts[3] == "metadata":
                sensor = metadata_sensor_type(PurePosixPath(file.path).name)
                if sensor:
                    sensor_counts[sensor] += 1
        elif len(parts) >= 3 and parts[0] == "auxiliary":
            if parts[1] in {"gaze", "heartrate", "photo", "video"}:
                participants.add(PurePosixPath(parts[2]).stem)

    coverage = build_coverage_rows(materialized)
    return {
        "repository_id": CASTLE_REPO_ID,
        "total_files": len(materialized),
        "total_bytes": sum(file.size_bytes for file in materialized),
        "participants": sorted(participants),
        "days": sorted(days),
        "category_counts": dict(sorted(category_counts.items())),
        "category_bytes": dict(sorted(category_bytes.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "metadata_sensor_counts": dict(sorted(sensor_counts.items())),
        "coverage": [asdict(row) for row in coverage],
        "timeline_findings": timeline_findings(),
        "risks": [
            "The repository is approximately 8.22 TB; workflows must select files before download.",
            "Main videos are often multi-gigabyte UHD recordings.",
            "Transcript timestamps are recording-relative and may contain malformed intervals.",
            "Heart-rate time is not an absolute timestamp and needs participant/day anchoring.",
            "Gaze uses a session start embedded in the CSV header plus elapsed row time.",
            "Thermal filenames do not expose participant or capture time in their repository path.",
        ],
    }


def build_coverage_rows(files: Iterable[RepositoryFile]) -> list[CoverageRow]:
    counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    auxiliary: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for file in files:
        path = file.path
        parts = path.split("/")
        if len(parts) >= 5 and parts[0] == "main":
            day, participant, modality = parts[1], parts[2], parts[3]
            suffix = PurePosixPath(path).suffix.lower()
            key = (day, participant)
            if modality == "video" and suffix == ".mp4":
                counts[key]["videos"] += 1
            elif modality == "video" and suffix == ".novideo":
                counts[key]["missing_videos"] += 1
            elif modality == "transcript" and suffix == ".json":
                counts[key]["transcripts"] += 1
            elif modality == "metadata" and suffix == ".csv":
                counts[key]["metadata_csv"] += 1
            elif modality == "metadata" and suffix == ".gpx":
                counts[key]["metadata_gpx"] += 1
        elif len(parts) >= 3 and parts[0] == "auxiliary":
            modality = parts[1]
            participant = PurePosixPath(parts[2]).stem
            suffix = PurePosixPath(path).suffix.lower()
            if modality == "heartrate" and suffix == ".csv":
                day = PurePosixPath(path).stem
                counts[(day, participant)]["heart_rate_files"] += 1
            elif modality == "gaze" and suffix == ".csv":
                auxiliary[participant]["gaze_files"] += 1
            elif modality == "photo" and suffix in {".jpg", ".jpeg"}:
                auxiliary[participant]["auxiliary_photos"] += 1
            elif modality == "video" and suffix in {".mp4", ".mov"}:
                auxiliary[participant]["auxiliary_videos"] += 1
            elif modality == "thermal" and suffix == ".bmp":
                auxiliary["[unassigned]"]["thermal_files"] += 1

    rows: list[CoverageRow] = []
    for day, participant in sorted(counts):
        row = counts[(day, participant)]
        aux = auxiliary[participant]
        rows.append(
            CoverageRow(
                day=day,
                participant_id=participant,
                videos=row["videos"],
                missing_videos=row["missing_videos"],
                transcripts=row["transcripts"],
                metadata_csv=row["metadata_csv"],
                metadata_gpx=row["metadata_gpx"],
                heart_rate_files=row["heart_rate_files"],
                gaze_files=aux["gaze_files"],
                thermal_files=0,
                auxiliary_photos=aux["auxiliary_photos"],
                auxiliary_videos=aux["auxiliary_videos"],
            )
        )
    return rows


def file_category(path: str) -> str:
    parts = path.split("/")
    suffix = PurePosixPath(path).suffix.lower()
    if len(parts) >= 4 and parts[0] == "main":
        return f"main/{parts[3]}"
    if len(parts) >= 2 and parts[0] == "auxiliary":
        return f"auxiliary/{parts[1]}"
    if suffix == ".zip":
        return "archive"
    return "repository_metadata"


def metadata_sensor_type(filename: str) -> str | None:
    path = PurePosixPath(filename)
    if path.suffix.lower() == ".gpx":
        return "GPS"
    parts = path.name.split(".")
    if len(parts) >= 3 and parts[-1].lower() == "csv":
        return parts[-2].upper()
    return None


def timeline_findings() -> list[dict[str, str]]:
    return [
        {
            "source": "video",
            "observed_format": "recording file stem under main/dayN/participant/video",
            "canonicalization": "derive absolute start from audited recording metadata",
            "status": "unresolved",
        },
        {
            "source": "transcript",
            "observed_format": "JSON chunks with recording-relative [start_sec, end_sec]",
            "canonicalization": "recording_start_ms + timestamp_sec * 1000",
            "status": "validated sample; malformed intervals exist",
        },
        {
            "source": "main metadata",
            "observed_format": "clock-of-day HH:MM:SS.sss",
            "canonicalization": "combine CASTLE day date, timezone, and clock time",
            "status": "date/timezone unresolved",
        },
        {
            "source": "heart rate",
            "observed_format": "elapsed HH:MM:SS.s plus bpm and confidence",
            "canonicalization": "participant/day session start + elapsed time",
            "status": "session anchor unresolved",
        },
        {
            "source": "gaze",
            "observed_format": "session start in TIME(...) header; elapsed TIME rows",
            "canonicalization": "header session start + elapsed TIME seconds",
            "status": "format validated; timezone unresolved",
        },
        {
            "source": "thermal",
            "observed_format": "sequential BMP filenames",
            "canonicalization": "requires external capture manifest or EXIF-equivalent evidence",
            "status": "unresolved and unassigned",
        },
    ]


def write_audit_outputs(output_dir: Path, audit: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "castle_inventory.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "coverage_matrix.csv", audit["coverage"])
    _write_csv(output_dir / "source_alignment.csv", audit["timeline_findings"])
    (output_dir / "README.md").write_text(render_audit_report(audit), encoding="utf-8")


def render_audit_report(audit: dict) -> str:
    gib = audit["total_bytes"] / 1024**3
    categories = "\n".join(
        f"- `{name}`: {count:,} files"
        for name, count in audit["category_counts"].items()
    )
    sensors = ", ".join(
        f"{name} ({count:,})"
        for name, count in audit["metadata_sensor_counts"].items()
    )
    risks = "\n".join(f"- {risk}" for risk in audit["risks"])
    return f"""# CASTLE Dataset Audit

Repository: `{audit["repository_id"]}`

- Logical files: {audit["total_files"]:,}
- Repository size: {gib:,.2f} GiB
- Recording-source directory names: {len(audit["participants"])}
- Days discovered: {", ".join(audit["days"])}

## File inventory

{categories}

## Main metadata sensor codes

{sensors}

## Outputs

- `castle_inventory.json`: machine-readable inventory and findings
- `coverage_matrix.csv`: participant/day path-level modality coverage
- `source_alignment.csv`: current timestamp formats and unresolved anchors

## Risks and decisions required

{risks}

## Immediate decision

Do not download the full repository. Select one participant/day and a small
number of recording stems after video start-time and participant-view semantics
are confirmed.
"""


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
