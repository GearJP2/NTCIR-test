from services.dataset.castle_audit import (
    RepositoryFile,
    build_castle_audit,
    metadata_sensor_type,
)


def test_metadata_sensor_type_extracts_csv_code_and_gps():
    assert metadata_sensor_type("08.ACCL.csv") == "ACCL"
    assert metadata_sensor_type("10.gpx") == "GPS"
    assert metadata_sensor_type("08.mp4") is None


def test_build_castle_audit_counts_modalities_and_coverage():
    audit = build_castle_audit(
        [
            RepositoryFile("main/day1/Allie/video/08.mp4", 100),
            RepositoryFile("main/day1/Allie/video/09.novideo", 0),
            RepositoryFile("main/day1/Allie/transcript/08.json", 20),
            RepositoryFile("main/day1/Allie/metadata/08.ACCL.csv", 30),
            RepositoryFile("main/day1/Allie/metadata/08.gpx", 40),
            RepositoryFile("auxiliary/heartrate/Allie/day1.csv", 10),
            RepositoryFile("auxiliary/gaze/Allie.csv", 10),
            RepositoryFile("auxiliary/photo/Allie/one.jpg", 10),
            RepositoryFile("auxiliary/video/Allie/one.mp4", 10),
            RepositoryFile("auxiliary/thermal/IMG 0003.bmp", 10),
            RepositoryFile("main/day1/Allie/video/08.mp4.sha256", 5),
        ]
    )

    assert audit["total_files"] == 10
    assert audit["metadata_sensor_counts"] == {"ACCL": 1, "GPS": 1}
    assert audit["participants"] == ["Allie"]
    row = audit["coverage"][0]
    assert row["videos"] == 1
    assert row["missing_videos"] == 1
    assert row["transcripts"] == 1
    assert row["heart_rate_files"] == 1
    assert row["gaze_files"] == 1
    assert row["thermal_files"] == 0
