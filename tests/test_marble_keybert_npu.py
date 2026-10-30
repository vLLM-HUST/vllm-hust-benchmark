"""No model imports, downloads or device allocation in these fixtures."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "marble_keybert_npu", ROOT / "scripts/run_marble_keybert_npu.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_plan_does_not_touch_model_or_read_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(entry, "extract", lambda *a: pytest.fail("model loaded"))
    monkeypatch.setattr(entry, "read_query", lambda *a: pytest.fail("input read"))
    assert entry.main(["--model-root", str(tmp_path / "absent")]) == 0
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [{}, {"query": 1}, {"query": "x", "extra": 2}, []])
def test_query_rejects_changed_interface(value):
    with pytest.raises(ValueError):
        entry.read_query(io.BytesIO(json.dumps(value).encode()))


def test_original_query_preserved_including_empty_and_unicode():
    for value in ["", "原始 query\n不清洗", "  original  "]:
        assert (
            entry.read_query(io.BytesIO(json.dumps({"query": value}).encode())) == value
        )


def test_oversized_query_rejected_without_truncation():
    with pytest.raises(ValueError, match="not truncated"):
        entry.read_query(io.BytesIO(b"x" * (8 * 1024**2 + 1)))


def test_fixture_model_assets_checked_before_model_import(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    raw = b"synthetic-not-model"
    path = model / "fixture.bin"
    path.write_bytes(raw)
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "fixture.bin",
                        "bytes": len(raw),
                        "lfs_sha256": hashlib.sha256(raw).hexdigest(),
                        "git_blob": None,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(entry, "PROFILE", profile)
    assert len(entry.verify_assets(model)["files"]) == 1
    path.write_bytes(b"different-same-size")
    with pytest.raises(ValueError, match="mismatch"):
        entry.verify_assets(model)


def test_pinned_model_matches_actual_qualified_original():
    profile = json.loads(entry.PROFILE.read_text())
    assert profile["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert (
        profile["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )  # pragma: allowlist secret - fixed public model revision
    assert len(profile["files"]) == 11
    assert sum(x["bytes"] for x in profile["files"]) == 91607178
