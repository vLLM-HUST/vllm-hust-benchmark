import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


@pytest.mark.parametrize("tamper", [False, True])
def test_alternate_run_archive_is_verified_before_packaging(
    tmp_path, monkeypatch, tamper
):
    source = (
        Path(__file__).resolve().parents[1]
        / "reports/vllm-hust-midterm-20261019/package_recovered_quality.py"
    )
    spec = importlib.util.spec_from_file_location("quality_packager", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = "test-native"
    records = {
        f"{run}/receipt.json": {
            "status": "measured",
            "all_devices_released": True,
            "bench_exit_code": 0,
        },
        f"{run}/mmlu-pro.json": {
            "completed": 200,
            "total": 200,
            "failed": 0,
            "correct": 120,
            "accuracy": 0.6,
            "truncated": 3,
            "unparsed": 1,
        },
    }
    raw = tmp_path / "evidence.tar.gz"
    members = []
    with tarfile.open(raw, "w:gz") as archive:
        for name, record in records.items():
            content = json.dumps(record).encode()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
            members.append(
                {
                    "name": name,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    manifest = {
        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "bytes": raw.stat().st_size,
        "members": members,
    }
    if tamper:
        manifest["members"][0]["sha256"] = "0" * 64
    index = tmp_path / "manifest.json"
    index.write_text(json.dumps(manifest))
    destination = tmp_path / "native"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(source),
            "--raw",
            str(raw),
            "--manifest",
            str(index),
            "--run",
            run,
            "--destination",
            str(destination),
        ],
    )
    if tamper:
        with pytest.raises(AssertionError):
            module.main()
        assert not destination.exists()
    else:
        module.main()
        result = json.loads((destination / "archive.json").read_text())
        assert result["run_id"] == run
        assert result["result"] == records[f"{run}/mmlu-pro.json"]
        assert (
            b"".join(
                (destination / part["name"]).read_bytes() for part in result["parts"]
            )
            == raw.read_bytes()
        )
