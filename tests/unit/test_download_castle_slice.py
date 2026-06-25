import json

from scripts.download_castle_slice import _load_existing


def test_load_existing_returns_empty_for_missing_manifest(tmp_path):
    assert _load_existing(tmp_path / "missing.jsonl") == []


def test_load_existing_reads_jsonl(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"repo_path": "video.mp4"}) + "\n", encoding="utf-8")

    assert _load_existing(path) == [{"repo_path": "video.mp4"}]
