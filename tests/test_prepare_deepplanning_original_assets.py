"""Asset preparation fixtures, not evidence of native benchmark execution."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "deepplanning_assets", ROOT / "scripts/prepare_deepplanning_original_assets.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_config(root):
    files = []
    for name, value in (
        ("source/LICENSE", b"author fixture"),
        ("databases/task.zip", b"original fixture"),
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        files.append(
            {
                "path": name,
                "bytes": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        )
    return {"code_revision": "fixture", "dataset_revision": "fixture", "files": files}


def test_plan_has_no_output_or_execution(tmp_path):
    output = tmp_path / "plan"
    result = MODULE.prepare(output)
    assert result["expected_files"] == 68
    assert result["model_calls"] == 0
    assert result["source_checkbox_eligible"] is False
    assert not output.exists()


def test_fixed_download_extracts_only_manifest_members(tmp_path, monkeypatch):
    payload = b"publisher fixture"
    database = b"database fixture"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("root/benchmark/file.py", "root/not-selected.txt"):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    original = buffer.getvalue()

    def identity(name, content):
        return {
            "path": name,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    config = {
        "code_revision": "fixture",
        "dataset_revision": "fixture",
        "dataset_repository": "publisher/fixture",
        "code_archive": {
            **identity("archive", original),
            "url": "https://example.invalid/archive",
        },
        "files": [
            identity("source/benchmark/file.py", payload),
            identity("databases/tasks.zip", database),
        ],
    }
    calls = []

    def fetch(request, **kwargs):
        calls.append(request.full_url)
        return io.BytesIO(
            original if request.full_url.endswith("/archive") else database
        )

    monkeypatch.setattr(MODULE, "urlopen", fetch)
    output = tmp_path / "prepared"
    result = MODULE.prepare(output, execute=True, config=config)
    assert len(calls) == 2
    assert (output / "source/benchmark/file.py").read_bytes() == payload
    assert not (output / "source/not-selected.txt").exists()
    assert result["source_checkbox_eligible"] is False


def test_reuse_verifies_all_bytes_and_offline_verification(tmp_path):
    original = tmp_path / "original"
    config = fixture_config(original)
    output = tmp_path / "prepared"
    result = MODULE.prepare(output, execute=True, reuse_root=original, config=config)
    assert result["status"] == "ORIGINAL_ASSETS_VERIFIED_NATIVE_EXECUTION_PENDING"
    assert (
        MODULE.prepare(output, verify_only=True, config=config)["status"]
        == "READ_ONLY_ORIGINAL_ASSETS_VERIFIED"
    )
    assert (
        json.loads((output / "preparation-public.json").read_text())[
            "source_checkbox_eligible"
        ]
        is False
    )
    with pytest.raises(ValueError, match="fresh output"):
        MODULE.prepare(output, execute=True, reuse_root=original, config=config)


def test_tampered_original_rejected_before_output_creation(tmp_path):
    original = tmp_path / "original"
    config = fixture_config(original)
    (original / "source/LICENSE").write_bytes(b"broken")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="mismatch"):
        MODULE.prepare(output, execute=True, reuse_root=original, config=config)
    assert not output.exists()


def test_python310_without_file_digest_supported(tmp_path, monkeypatch):
    config = fixture_config(tmp_path / "original")
    monkeypatch.delattr(MODULE.hashlib, "file_digest", raising=False)
    result = MODULE.prepare(tmp_path / "original", verify_only=True, config=config)
    assert result["status"] == "READ_ONLY_ORIGINAL_ASSETS_VERIFIED"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "bad\\path"])
def test_unsafe_path_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        MODULE.safe_path(tmp_path, name)


def test_rejects_repository_output_and_arbitrary_endpoint(tmp_path):
    with pytest.raises(ValueError, match="outside repository"):
        MODULE.prepare(ROOT / "raw-data", execute=True)
    with pytest.raises(ValueError, match="endpoint"):
        MODULE.prepare(
            tmp_path / "bad", execute=True, hf_endpoint="https://example.invalid"
        )
