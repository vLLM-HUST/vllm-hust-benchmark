"""Official software pins and inert preparation, not native Agent results."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/prepare_jsonschemabench_author_source.py"
PINS = ROOT / "configs/workload-native/bixbench-author-source.json"


def test_official_software_selection_keeps_lock_license_and_original_agent():
    pins = json.loads(PINS.read_text("utf-8"))
    paths = {item["path"] for item in pins["files"]}
    assert len(pins["files"]) == len(paths) == 27
    assert sum(item["bytes"] for item in pins["files"]) == 627314
    assert {
        "uv.lock",
        "LICENSE",
        "pyproject.toml",
        "bixbench/generate_trajectories.py",
        "bixbench/graders.py",
    } <= paths
    assert len([path for path in paths if path.endswith(".py")]) == 13
    assert not any(
        path.startswith(("data/", "models/", "bixbench_results/", "tests/"))
        for path in paths
    )


def test_isolated_plan_has_no_download_or_environment_creation(tmp_path):
    target = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ENTRY),
            "--source",
            "bixbench",
            "--author-root",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["source_id"] == "BIXBENCH"
    assert report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert report["files"] == 27 and report["model_calls"] == 0
    assert not report["source_checkbox_eligible"]
    assert not target.exists()


@pytest.mark.parametrize("change", ["none", "altered", "extra-python"])
def test_verifier_checks_bytes_without_importing_author_code(tmp_path, change):
    spec = importlib.util.spec_from_file_location("source_preparer", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    author = tmp_path / "author"
    author.mkdir()
    raw = b'raise RuntimeError("fixture source must not execute")\n'
    (author / "fixture.py").write_bytes(raw)
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "fixture.py",
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "git_blob": hashlib.sha1(
                            b"blob " + str(len(raw)).encode() + b"\0" + raw
                        ).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    if change == "altered":
        (author / "fixture.py").write_bytes(raw + b"# changed\n")
    elif change == "extra-python":
        (author / "unregistered.py").write_text("# fixture only\n")
    if change == "none":
        assert module.verify_source(author, pins) == author.resolve()
    else:
        with pytest.raises(ValueError):
            module.verify_source(author, pins)
