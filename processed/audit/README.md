# CASTLE Dataset Audit

Repository: `CASTLE-Dataset/CASTLE2024`

- Logical files: 24,288
- Repository size: 7,669.92 GiB
- Participants discovered: 16
- Days discovered: day1, day2, day3, day4

## File inventory

- `archive`: 1 files
- `auxiliary/gaze`: 7 files
- `auxiliary/heartrate`: 39 files
- `auxiliary/photo`: 10,006 files
- `auxiliary/thermal`: 39 files
- `auxiliary/video`: 16 files
- `main/metadata`: 12,738 files
- `main/transcript`: 659 files
- `main/video`: 780 files
- `repository_metadata`: 3 files

## Main metadata sensor codes

AALP (672), ACCL (668), CORI (667), FACE (583), GPS (453), GPS5 (474), GRAV (433), GYRO (554), HUES (674), IORI (600), ISOE (432), LSKP (668), MSKP (612), MWET (655), SCEN (619), SHUT (673), UNIF (676), WBAL (675), WNDM (674), WRGB (668), YAVG (608)

## Outputs

- `castle_inventory.json`: machine-readable inventory and findings
- `coverage_matrix.csv`: participant/day path-level modality coverage
- `source_alignment.csv`: current timestamp formats and unresolved anchors

## Risks and decisions required

- The repository is approximately 8.22 TB; workflows must select files before download.
- Main videos are often multi-gigabyte UHD recordings.
- Transcript timestamps are recording-relative and may contain malformed intervals.
- Heart-rate time is not an absolute timestamp and needs participant/day anchoring.
- Gaze uses a session start embedded in the CSV header plus elapsed row time.
- Thermal filenames do not expose participant or capture time in their repository path.

## Immediate decision

Do not download the full repository. Select one participant/day and a small
number of recording stems after video start-time and participant-view semantics
are confirmed.
