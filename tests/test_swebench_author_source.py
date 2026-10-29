"""Inert software archive fixtures; no original tasks, Docker or models."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "swe_source", SCRIPTS / "prepare_swebench_author_source.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_legacy_release_profile_plan_does_not_create_or_execute(
    tmp_path, monkeypatch, capsys
):
    import json
    import subprocess

    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_swebench_author_source.py"),
            "--profile",
            "v4.1.0",
            "--author-root",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["files"] == 87 and report["bytes"] == 594174
    assert report["status"] == "PLAN_ONLY" and report["model_calls"] == 0
    assert not output.exists()


def test_cpu_recovery_does_not_qualify_repository_tasks_or_new_sources():
    receipt = json.loads(
        (
            SCRIPTS.parent
            / "docs/workload-data/validation/swebench-original-readiness-20261029.json"
        ).read_text(encoding="utf-8")
    )
    recovery = receipt["independent_cpu_recovery"]
    assert recovery["original_daemon_start_ticks_before_and_after_equal"]
    assert not recovery["native_benchmark_runtime_images_qualified"]
    assert recovery["native_benchmark_tasks_executed"] == 0
    assert recovery["source_checkbox_count_increment"] == 0
    components = recovery["restored_render_components"]
    for name in ("matplotlib", "browser"):
        assert components[name]["exact_config_and_all_diff_ids_preserved"]
        assert components[name]["environment_verifier_exit_code"] == 0
    assert len(components["public_controller_controls"]) == 4
    timeout = components["public_controller_controls"][-1]
    assert timeout["status"] == "BLOCKED_COMPONENT_TIME_LIMIT_NOT_MODEL_SCORE"
    assert timeout["render_score"] is None
    assert components["new_daemon_running_containers_after_controls"] == 0
    assert components["model_calls"] == components["native_tasks_executed"] == 0
    assert components["source_checkbox_increment"] == 0
    assert (
        recovery["new_backing_file_capacity_bytes"] + 1024**3
        <= recovery["approved_total_extra_budget_bytes"]
    )


def test_classic_original_asset_plan_is_separate_from_verified():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_reused_research_assets.py"),
            "SWE_BENCH",
            "--mode",
            "plan",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["source_id"] == "SWE_BENCH"
    assert plan["bytes"] == 12097227
    assert len(plan["files"]) == 1
    assert plan["files"][0]["rows"] == 2294
    assert len(plan["files"][0]["fields"]) == 12
    assert "difficulty" not in plan["files"][0]["fields"]
    assert plan["files"][0]["role"] == "data"
    assert plan["model_calls"] == 0
    assert not plan["source_checkbox_eligible"]
    assert not plan["native_execution_complete"]


def test_verified_completion_cannot_complete_classic_or_hide_negative_outcomes():
    """Receipt consistency only; this test does not execute or rescore tasks."""
    folder = SCRIPTS.parent / "docs/workload-data"
    receipt = json.loads(
        (folder / "validation/swebench-original-readiness-20261029.json").read_text(
            encoding="utf-8"
        )
    )
    inventory = json.loads(
        (folder / "dataset-validation-status.json").read_text(encoding="utf-8")
    )["delivery_inventory"]
    assert receipt["source_ids"] == ["SWE_VERIFIED"]
    assert receipt["shared_preparation_source_ids"] == ["SWE_BENCH", "SWE_VERIFIED"]
    assert receipt["source_checkbox_eligible"]
    assert inventory["SWE_VERIFIED"]["checked"]
    if inventory["SWE_BENCH"]["checked"]:
        assert (
            inventory["SWE_BENCH"]["delivery_receipt"]
            != inventory["SWE_VERIFIED"]["delivery_receipt"]
        ), "Classic must close with its own source-scoped evidence"
    assert receipt["public_entry_code_ci"]["both_conclusions"] == "success"
    assert receipt["official_grading_attempts"] == {
        "gold_control": 3,
        "model_predictions": 3,
    }
    assert receipt["original_test_executions_and_score_reports"] == 5
    assert "not all cases passing" in receipt["completion_scope"]
    assert "not SWE_BENCH completion" in receipt["completion_scope"]


@pytest.mark.parametrize("profile,count", [("core", 76), ("agent", 120)])
def test_public_cpu_dependency_locks_are_unique_hash_bound_and_not_inference_packages(
    profile, count
):
    import re

    path = (
        SCRIPTS.parent
        / f"configs/workload-native/swebench-original-{profile}-py312-aarch64.lock"
    )
    rows = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert len(rows) == count
    names = set()
    for line in rows:
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)==([A-Za-z0-9.!+_-]+) --hash=sha256:([0-9a-f]{64})", line
        )
        assert match
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        assert name not in names and name not in {
            "torch",
            "torch-npu",
            "triton",
            "flash-attn",
        }
        names.add(name)
    assert "pyarrow" in names
    if profile == "agent":
        assert {"mcp", "litellm", "jinja2"} <= names


def test_official_agent_profile_is_plan_only_and_pins_all_runtime_configs(tmp_path):
    import subprocess

    output = tmp_path / "not-created-agent"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_swebench_author_source.py"),
            "--profile",
            "mini-swe-agent",
            "--author-root",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["files"] == 78 and report["bytes"] == 302226
    assert report["status"] == "PLAN_ONLY"
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
    assert not output.exists()
    pins = json.loads(
        (
            SCRIPTS.parent / "configs/workload-native/mini-swe-agent-author-source.json"
        ).read_text(encoding="utf-8")
    )
    assert pins["repository"] == "SWE-agent/mini-swe-agent"
    assert {"SWE_BENCH", "SWE_VERIFIED"} == set(pins["shared_source_ids"])
    names = {item["path"] for item in pins["files"]}
    assert {
        "LICENSE.md",
        "pyproject.toml",
        "src/minisweagent/environments/docker.py",
        "src/minisweagent/config/benchmarks/swebench.yaml",
        "src/minisweagent/run/benchmarks/swebench.py",
    } <= names


def fixture(tmp_path, names=None, content=b"inert original software\n", kind=None):
    archive = tmp_path / "source.tar.gz"
    names = names or ["publisher-fixed/swebench/__init__.py"]
    with tarfile.open(archive, "w:gz") as handle:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            if kind is not None:
                info.type = kind
            handle.addfile(info, io.BytesIO(content))
    pins = {
        "files": [
            {
                "path": "swebench/__init__.py",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(content)).encode() + b"\0" + content
                ).hexdigest(),
            }
        ]
    }
    return archive, pins


def test_original_software_preparation_and_identical_resume(tmp_path):
    archive, pins = fixture(tmp_path)
    output = tmp_path / "external"
    assert module.prepare(archive, output, pins) == 1
    assert module.prepare(archive, output, pins) == 1
    assert (
        output / "swebench/__init__.py"
    ).read_bytes() == b"inert original software\n"


@pytest.mark.parametrize(
    "names",
    [
        ["../swebench/__init__.py"],
        ["/publisher/swebench/__init__.py"],
        ["publisher-fixed/missing.py"],
        ["publisher-fixed/swebench/__init__.py"] * 2,
        ["publisher-fixed/swebench/__init__.py", "other-root/README.md"],
    ],
)
def test_unsafe_duplicate_missing_or_multiple_root_archives_rejected(tmp_path, names):
    archive, pins = fixture(tmp_path, names)
    with pytest.raises(ValueError):
        module.prepare(archive, tmp_path / "external", pins)


@pytest.mark.parametrize("field", ["sha256", "git_blob", "bytes"])
def test_identity_mismatch_cannot_publish_file(tmp_path, field):
    archive, pins = fixture(tmp_path)
    pins["files"][0][field] = (
        0 if field == "bytes" else "0" * len(pins["files"][0][field])
    )
    output = tmp_path / "external"
    with pytest.raises(ValueError):
        module.prepare(archive, output, pins)
    assert not (output / "swebench/__init__.py").exists()


def test_nonregular_selected_member_rejected(tmp_path):
    archive, pins = fixture(tmp_path, kind=tarfile.SYMTYPE)
    with pytest.raises(ValueError):
        module.prepare(archive, tmp_path / "external", pins)


def test_existing_different_file_never_overwritten(tmp_path):
    archive, pins = fixture(tmp_path)
    output = tmp_path / "external"
    target = output / "swebench/__init__.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing unrelated bytes")
    with pytest.raises(ValueError):
        module.prepare(archive, output, pins)
    assert target.read_bytes() == b"existing unrelated bytes"


def test_checkout_output_rejected_before_writes(tmp_path):
    archive, pins = fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the checkout"):
        module.prepare(archive, module.ROOT / "never-create-swebench-source", pins)
