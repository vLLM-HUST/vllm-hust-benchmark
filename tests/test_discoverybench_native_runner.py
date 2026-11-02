"""Runner boundary fixtures; no native tasks, Docker or neural inference."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "discovery_native", SCRIPTS / "run_discoverybench_native.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_default_is_plan_without_docker():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_discoverybench_native.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"status": "PLAN_ONLY", "model_calls": 0}


def test_tool_mounts_have_no_reference_queues_or_socket(tmp_path):
    author, runtime, data = (tmp_path / name for name in ("author", "runtime", "data"))
    for path in (author, runtime, data):
        path.mkdir()
    args = SimpleNamespace(
        author_root=author,
        host_bind_root=tmp_path,
        daemon_bind_root=Path("/daemon"),
        image="sha256:" + "0" * 64,
        python="/venv/python",
    )
    command = entry.container_command(
        args, runtime, [(data, "/input", True)], name="owned-tool"
    )
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    assert len(mounts) == 3
    assert all(value.endswith(",readonly") for value in mounts)
    assert not any(
        value in " ".join(mounts)
        for value in ("queue", "reference", "metadata", "docker.sock")
    )
    for flag in ("--read-only", "--cap-drop", "--security-opt", "--pids-limit"):
        assert flag in command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--user") + 1] == "65534:65534"


def test_mount_outside_explicit_daemon_mapping_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="mapped host root"):
        entry.daemon_path(tmp_path, root, Path("/daemon"))


def make_manifest(tmp_path):
    files = []
    for name in ("metadata.json", "reference.json", "data.csv"):
        path = tmp_path / name
        path.write_bytes(b"fixture")
        files.append({"path": name, "sha256": hashlib.sha256(b"fixture").hexdigest()})
    value = {
        "schema_version": "discoverybench-original-cases/v1",
        "source_id": "DISCOVERYBENCH",
        "original_actor_iterations": 5,
        "original_python_call_limit": 1,
        "cases": [
            {
                "original_position": 0,
                "metadata": files[0],
                "reference": files[1],
                "data_files": files[2:],
            }
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(value))
    return path, value


def test_prepared_identity_is_checked(tmp_path):
    path, value = make_manifest(tmp_path)
    assert entry.load_cases(path)[1] == value["cases"]
    (tmp_path / "data.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        entry.load_cases(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("original_actor_iterations", 4),
        ("original_python_call_limit", 2),
        ("source_id", "other"),
    ],
)
def test_original_protocol_not_relaxed(tmp_path, field, value):
    path, manifest = make_manifest(tmp_path)
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol"):
        entry.load_cases(path)


def test_claimed_tool_is_never_resubmitted(tmp_path, monkeypatch):
    (tmp_path / "original-python.request.json").write_text(
        "not parsed when already claimed"
    )
    (tmp_path / "original-python.claim").touch()
    monkeypatch.setattr(
        entry.subprocess, "run", lambda *_a, **_k: pytest.fail("must not run Docker")
    )
    entry.tool_once(None, None, None, tmp_path, tmp_path)


def test_changed_tool_code_does_not_execute(tmp_path, monkeypatch):
    (tmp_path / "original-python.request.json").write_text(
        json.dumps({"code": "fixture", "code_sha256": "wrong"})
    )
    monkeypatch.setattr(
        entry.subprocess, "run", lambda *_a, **_k: pytest.fail("must not run Docker")
    )
    with pytest.raises(ValueError, match="identity differs"):
        entry.tool_once(None, None, None, tmp_path, tmp_path)
    assert not (tmp_path / "original-python.claim").exists()


def test_staging_copies_only_manifest_bound_files(tmp_path):
    prepared = tmp_path / "prepared"
    metadata = prepared / "case-0/metadata/metadata_0.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"original")
    extra = prepared / "case-0/unlisted-private.txt"
    extra.write_bytes(b"must not enter container")
    item = {
        "path": "case-0/metadata/metadata_0.json",
        "sha256": hashlib.sha256(b"original").hexdigest(),
    }
    case = {
        "original_position": 0,
        "metadata": item,
        "reference": {
            "path": "case-0/reference.json",
            "sha256": hashlib.sha256(b"reference").hexdigest(),
        },
        "data_files": [],
    }
    (prepared / "case-0/reference.json").write_bytes(b"reference")
    output = tmp_path / "staged"
    entry.stage_inputs(prepared, case, output)
    assert (output / "metadata/metadata_0.json").read_bytes() == b"original"
    assert not (output / extra.name).exists()


def test_staging_rejects_a_different_original_position(tmp_path):
    case = {
        "original_position": 0,
        "metadata": {"path": "case-1/metadata.json"},
        "reference": {},
        "data_files": [],
    }
    with pytest.raises(ValueError, match="original position"):
        entry.stage_inputs(tmp_path, case, tmp_path / "output")
