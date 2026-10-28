"""Tiny offline transport/CSV fixtures, not native TemporalWiki evaluation."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "temporalwiki_assets", ROOT / "scripts/prepare_temporalwiki_original_assets.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(tmp_path):
    csv = b'subject,relation,object\r\n" a, b ",relation,reference\r\n'
    files = [
        {
            "path": "twiki_probes/fixture.csv",
            **identity(csv),
            "role": "data",
            "columns": ["subject", "relation", "object"],
            "rows": 1,
        }
    ]
    archive = tmp_path / "original.zip"
    with zipfile.ZipFile(archive, "w") as packed:
        packed.writestr("twiki_probes/", b"")
        packed.writestr(files[0]["path"], csv)
    profile = {
        "dataset_repo": "fixture/repo",
        "dataset_revision": "fixture",
        "files": files,
        "archive": {"path": "original.zip", **identity(archive.read_bytes())},
    }
    return archive, profile, csv


def test_original_all_eight_csvs_registered_separately():
    profile = json.loads(runner.PROFILE.read_text("utf-8"))
    data = [item for item in profile["files"] if item["role"] == "data"]
    assert len(data) == 8 and len(profile["files"]) == 9
    assert sum(item["rows"] for item in data) == 35948
    assert all(item["columns"] == ["subject", "relation", "object"] for item in data)
    assert profile["archive"]["bytes"] == 620459
    assert profile["revision"] != profile["dataset_revision"]


def test_extraction_preserves_bytes_and_refuses_overwrite(tmp_path):
    archive, profile, csv = fixture(tmp_path)
    output = tmp_path / "data"
    assert runner.verify_archive(archive, profile) == 1
    runner.extract_archive(archive, output, profile)
    assert (output / profile["files"][0]["path"]).read_bytes() == csv
    with pytest.raises(FileExistsError):
        runner.extract_archive(archive, output, profile)
    (output / "unexpected.csv").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="file set"):
        runner.verify_data(output, profile)


def test_changed_source_or_csv_shape_is_not_accepted(tmp_path):
    archive, profile, _ = fixture(tmp_path)
    profile["files"][0]["rows"] = 2
    with pytest.raises(ValueError, match="row count"):
        runner.verify_archive(archive, profile)
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size"):
        runner.verify_archive(archive, profile)


def test_original_download_url_and_whole_file_budget(tmp_path, monkeypatch):
    archive, profile, _ = fixture(tmp_path)
    raw = archive.read_bytes()
    requests = []

    def response(request, **kwargs):
        requests.append(request.full_url)
        return io.BytesIO(raw)

    monkeypatch.setattr(runner, "urlopen", response)
    target = tmp_path / "download.zip"
    runner.download_archive(target, profile, "https://hf-mirror.com", len(raw))
    assert target.read_bytes() == raw
    assert requests == [
        "https://hf-mirror.com/datasets/fixture/repo/resolve/fixture/original.zip?download=true"
    ]
    runner.download_archive(target, profile, "https://hf-mirror.com", len(raw))
    assert len(requests) == 1  # Reuse verified data; no second download.
    with pytest.raises(ValueError, match="budget"):
        runner.download_archive(
            tmp_path / "bad.zip", profile, "https://hf-mirror.com", len(raw) - 1
        )


def test_default_plan_is_offline_and_does_not_create_archive(tmp_path):
    archive = tmp_path / "not-created.zip"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_temporalwiki_original_assets.py"),
            "--archive",
            str(archive),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert report["original_probe_rows"] == 35948
    assert not report["source_checkbox_eligible"] and not archive.exists()


@pytest.mark.parametrize("fault", ["traversal", "symlink", "duplicate"])
def test_unsafe_archive_is_rejected_before_extraction(tmp_path, fault):
    name = "twiki_probes/../escape.csv" if fault == "traversal" else "twiki_probes/item"
    raw = b"opaque fixture"
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as packed:
        packed.writestr(str(Path(name).parent).replace("\\", "/") + "/", b"")
        item = zipfile.ZipInfo(name)
        if fault == "symlink":
            item.create_system = 3
            item.external_attr = 0o120777 << 16
        packed.writestr(item, raw)
        if fault == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                packed.writestr(item, raw)
    profile = {
        "archive": identity(archive.read_bytes()),
        "files": [{"path": name, **identity(raw), "role": "auxiliary_metadata"}],
    }
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        runner.extract_archive(archive, output, profile)
    assert not output.exists()


def test_failed_download_preserves_partial_without_creating_valid_archive(
    tmp_path, monkeypatch
):
    _, profile, _ = fixture(tmp_path)
    monkeypatch.setattr(
        runner, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad archive")
    )
    target = tmp_path / "bad-download.zip"
    with pytest.raises(ValueError):
        runner.download_archive(target, profile, "https://hf-mirror.com", 100000)
    assert not target.exists()
    assert target.with_name(target.name + ".partial").read_bytes() == b"bad archive"
    with pytest.raises(ValueError, match="existing partial"):
        runner.download_archive(target, profile, "https://hf-mirror.com", 100000)


def test_reference_object_does_not_become_chat_input():
    from vllm_hust_workload_data.adapters import adapt_row
    from vllm_hust_workload_data.registry import adapter_spec

    original = {
        "subject": " original subject ",
        "relation": "relation",
        "object": "reference",
    }
    record = adapt_row(
        "TEMPORALWIKI",
        original,
        source={
            "source_id": "TEMPORALWIKI",
            "file_sha256": "fixture",
            "unit": "original_probe_rows",
        },
        spec=adapter_spec("TEMPORALWIKI"),
        index=0,
    )
    assert record.task["input"] | record.outcome_record() == original
    assert "object" not in record.task["input"]
    assert not record.replay["synthetic_fallback"]
    with pytest.raises(ValueError, match="not a ready"):
        record.request_body()
