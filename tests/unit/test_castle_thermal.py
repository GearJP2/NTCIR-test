from services.dataset.castle_audit import RepositoryFile
from services.dataset.castle_thermal import inspect_thermal_files, write_thermal_inventory


def test_inspect_thermal_files_marks_flat_bmp_paths_unassigned():
    rows = inspect_thermal_files(
        [
            RepositoryFile("auxiliary/thermal/IMG 0003.bmp", 123),
            RepositoryFile("auxiliary/thermal/notes.txt", 10),
            RepositoryFile("auxiliary/photo/Allie/IMG 0003.jpg", 10),
        ]
    )

    assert len(rows) == 1
    assert rows[0].filename == "IMG 0003.bmp"
    assert rows[0].sequence_index == 3
    assert rows[0].inferred_day == ""
    assert rows[0].inferred_participant_id == ""
    assert rows[0].timestamp_status == "no_timestamp_in_path"
    assert rows[0].assignment_status == "unassigned"


def test_inspect_thermal_files_detects_date_like_filename_but_not_assignment():
    rows = inspect_thermal_files(
        [RepositoryFile("auxiliary/thermal/thermal_20241204_001.bmp", 123)]
    )

    assert rows[0].timestamp_status == "date_like_token_in_filename"
    assert rows[0].assignment_status == "unassigned"


def test_write_thermal_inventory_emits_header(tmp_path):
    rows = inspect_thermal_files([RepositoryFile("auxiliary/thermal/IMG 0003.bmp", 123)])
    output = tmp_path / "thermal.csv"

    write_thermal_inventory(output, rows)

    assert "source,filename,size_bytes" in output.read_text(encoding="utf-8")
