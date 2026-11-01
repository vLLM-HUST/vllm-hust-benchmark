"""Original ToolBench software preparation contracts; no tools, tasks or model."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/toolbench-author-source.json"


def test_original_plan_excludes_tasks_weights_and_results(tmp_path):
    pins = json.loads(PROFILE.read_text("utf-8"))
    paths = {row["path"] for row in pins["files"]}
    assert len(paths) == 79
    assert sum(row["bytes"] for row in pins["files"]) == 397381
    assert {
        "LICENSE",
        "requirements.txt",
        "toolbench/inference/Algorithms/DFS.py",
        "toolbench/inference/Downstream_tasks/rapidapi.py",
        "toolbench/inference/qa_pipeline.py",
    } <= paths
    assert not any(
        path.startswith(("data/", "data_example/", "assets/")) for path in paths
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "toolbench",
            "--author-root",
            str(tmp_path / "original"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert report["source_id"] == "TOOLBENCH"
    assert report["model_calls"] == 0
    assert report["source_checkbox_eligible"] is False
    assert not (tmp_path / "original").exists()


def test_author_verifier_rejects_changed_or_injected_software(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "toolbench_source_control",
            ROOT / "scripts/prepare_jsonschemabench_author_source.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    raw = b"# invented fixture, not an original benchmark\n"
    source = tmp_path / "source"
    source.mkdir()
    (source / "fixture.py").write_bytes(raw)
    profile = tmp_path / "pins.json"
    profile.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "fixture.py",
                        "bytes": len(raw),
                        "git_blob": hashlib.sha1(
                            b"blob " + str(len(raw)).encode() + b"\0" + raw
                        ).hexdigest(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                ]
            }
        )
    )
    assert module.verify_source(source, profile) == source
    (source / "injected.py").write_bytes(raw)
    with pytest.raises(ValueError, match="unexpected/missing"):
        module.verify_source(source, profile)
    (source / "injected.py").unlink()
    (source / "fixture.py").write_bytes(b"!" * len(raw))
    with pytest.raises(ValueError):
        module.verify_source(source, profile)
