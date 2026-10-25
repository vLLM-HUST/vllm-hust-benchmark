import json
from pathlib import Path

import pytest
from build_capsule import build, freeze_run, freeze_support, identity, verify_archive


def terminal_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for name in (
        "process.log",
        "engine-args.json",
        "plan.json",
        "npu-before.txt",
        "npu-after.txt",
    ):
        (run / name).write_text("{}\n")
    (run / "sample.npy").write_bytes(b"original-vector-bytes")
    row = {
        "label": "test-failure",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "pid": 2147483647,
        "process_group": 2147483647,
        "exit_code": 1,
        "passed": False,
        "source": "/synthetic",
        "source_commit": "a" * 40,
        "source_status": "",
        "config_sha256": identity(run / "engine-args.json")["sha256"],
        "plan_file_sha256": identity(run / "plan.json")["sha256"],
    }
    (run / "controller.json").write_text(json.dumps(row))
    return run


def test_failure_preserved_and_archive_fully_reverified(tmp_path):
    run = terminal_run(tmp_path)
    manifest = tmp_path / "manifest.json"
    value = freeze_run(run, "failure-v1", manifest)
    assert value["terminal"]["controller_passed"] is False
    result = build(manifest, tmp_path / "capsule")
    assert result["members_verified"] == 7
    checked = verify_archive(
        Path(result["archive"]["path"]), value, result["archive"]["sha256"]
    )
    assert checked["verified"]
    with pytest.raises(FileExistsError):
        build(manifest, tmp_path / "capsule")
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_archive(Path(result["archive"]["path"]), value, "0" * 64)


def test_live_or_unfinished_run_rejected(tmp_path):
    run = terminal_run(tmp_path)
    path = run / "controller.json"
    row = json.loads(path.read_text())
    del row["finished_at"]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="not terminal"):
        freeze_run(run, "unfinished-v1", tmp_path / "manifest.json")


def test_changed_original_rejected_and_failure_receipt_preserved(tmp_path):
    run = terminal_run(tmp_path)
    path = tmp_path / "manifest.json"
    freeze_run(run, "changed-v1", path)
    (run / "sample.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="files changed"):
        build(path, tmp_path / "capsule")
    assert (tmp_path / "capsule/BUILD-FAILED.json").is_file()
    assert not (tmp_path / "capsule/receipt.json").exists()


def test_unlisted_new_file_rejected(tmp_path):
    run = terminal_run(tmp_path)
    path = tmp_path / "manifest.json"
    freeze_run(run, "late-write-v1", path)
    (run / "new.npy").write_bytes(b"late output")
    with pytest.raises(ValueError, match="files changed"):
        build(path, tmp_path / "capsule")


def test_support_requires_explicit_matching_hash(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("support")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": str(source),
                        "member": "support/source.txt",
                        "sha256": "0" * 64,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="explicit selection"):
        freeze_support(selection, "support-v1", tmp_path / "manifest.json")
