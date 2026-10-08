import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_hust_benchmark.evaluation_job_adapter import (
    AdapterError,
    ExecutionPlan,
    build_plan,
    run_plan,
)
from vllm_hust_benchmark.official_targets import REGISTRY_VERSION

ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
SPEC = (
    ROOT
    / "docs/official-baselines/official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json"
)


def _git_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "README").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "frozen",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    core = tmp_path / "core"
    plugin = tmp_path / "plugin"
    model = tmp_path / "model"
    runtime_cwd = tmp_path / "runtime-cwd"
    model.mkdir()
    runtime_cwd.mkdir()
    (model / "refs").mkdir()
    (model / "refs/main").write_text("b" * 40 + "\n", encoding="utf-8")
    core_sha = _git_repo(core)
    plugin_sha = _git_repo(plugin)
    request = {
        "schema_version": 1,
        "repository": "vLLM-HUST/vllm-hust",
        "core_commit": core_sha,
        "plugin_repository": "vLLM-HUST/vllm-ascend-hust",
        "plugin_commit": plugin_sha,
        "target_id": TARGET_ID,
        "target_registry_version": REGISTRY_VERSION,
        "repeat_count": 3,
        "npu_count": 1,
        "priority": "required",
        "requested_by": "unit-test",
        "source_url": "https://example.org/issue/1",
        "metadata": {"CURRENT_EXPORT_ONLY": "1", "load_profile": "untrusted"},
    }
    schedule = {
        "schema_version": "evaluation-112-schedule/v1",
        "schedule_version": "1.0.0",
        "benchmark_root": str(ROOT),
        "core_repo": str(core),
        "plugin_repo": str(plugin),
        "core_repository": request["repository"],
        "plugin_repository": request["plugin_repository"],
        "registry_version": REGISTRY_VERSION,
        "runtime_python": sys.executable,
        "runtime_cwd": str(runtime_cwd),
        "image_id": "a" * 64,
        "cann_version": "test-cann",
        "torch_npu_version": "test-torch-npu",
        "topology": "single-node-910B2",
        "environment": {"HOME": str(tmp_path)},
        "targets": {
            TARGET_ID: {
                "approval": "approved",
                "registry_status": "active",
                "source_spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
                "model_id": "Qwen/Qwen2.5-14B-Instruct",
                "model_precision": "FP16",
                "hardware_chip_model": "910B2",
                "model_path": str(model),
                "model_revision": "b" * 40,
                "load_profile": "fixed-1-rps",
                "campaign_id": "issue-136-current-main/v1",
                "coverage_class": "full-matrix",
                "point_role": "checkpoint",
            }
        },
    }
    request_file = tmp_path / "request.json"
    schedule_file = tmp_path / "schedule.json"
    request_file.write_text(json.dumps(request), encoding="utf-8")
    schedule_file.write_text(json.dumps(schedule), encoding="utf-8")
    return {
        "request": request_file,
        "schedule": schedule_file,
        "output": tmp_path / "result",
        "core": core,
        "plugin": plugin,
    }


def _edit_json(path: Path, edit) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _plan(inputs: dict[str, Path], devices: str = "2") -> ExecutionPlan:
    return build_plan(inputs["request"], devices, inputs["output"], inputs["schedule"])


def test_plan_uses_approved_target_and_ignores_request_metadata(
    inputs: dict[str, Path],
) -> None:
    plan = _plan(inputs)
    assert plan.command[1] == str(ROOT / "scripts/run-campaign-repetitions.sh")
    assert plan.command[-2:] == ("--repetitions", "3")
    assert plan.schedule_version == "1.0.0"
    assert len(plan.schedule_sha256) == 64
    assert hashlib.sha256(plan.schedule_snapshot).hexdigest() == plan.schedule_sha256
    assert plan.environment["CURRENT_EXPORT_ONLY"] == "0"
    assert plan.environment["CAMPAIGN_LOAD_PROFILE"] == "fixed-1-rps"
    assert plan.environment["ASCEND_RT_VISIBLE_DEVICES"] == "2"
    assert plan.environment["CURRENT_RUNTIME_CWD"] == str(
        (inputs["request"].parent / "runtime-cwd").resolve()
    )
    assert "EVALUATION_REQUEST_FILE" not in plan.environment
    assert "SINGLE_REPETITION_RUNNER" not in plan.environment


@pytest.mark.parametrize(
    ("file", "edit", "message"),
    [
        ("request", lambda x: x.update(repeat_count=2), "repeat_count"),
        ("request", lambda x: x.update(priority="urgent"), "priority"),
        ("request", lambda x: x.update(metadata=[]), "metadata"),
        (
            "request",
            lambda x: x.update(repository="not-a-repository"),
            "repository is invalid",
        ),
        ("request", lambda x: x.update(npu_count=2), "assigned NPUs"),
        (
            "request",
            lambda x: x.update(target_registry_version="1.3.7"),
            "registry version",
        ),
        (
            "request",
            lambda x: x.update(target_id="unknown-target"),
            "not explicitly approved",
        ),
        ("request", lambda x: x.update(CURRENT_EXPORT_ONLY="1"), "request fields"),
        (
            "schedule",
            lambda x: x["targets"][TARGET_ID].update(approval="draft"),
            "not explicitly approved",
        ),
        (
            "schedule",
            lambda x: x.update(schedule_version="latest"),
            "schedule_version",
        ),
        (
            "schedule",
            lambda x: x["targets"][TARGET_ID].update(source_spec_sha256="0" * 64),
            "checksum",
        ),
        (
            "schedule",
            lambda x: x["targets"][TARGET_ID].update(registry_status="provisional"),
            "status",
        ),
        (
            "schedule",
            lambda x: x["environment"].update(SINGLE_REPETITION_RUNNER="/tmp/unsafe"),
            "forbidden overrides",
        ),
        (
            "schedule",
            lambda x: x["environment"].update(CURRENT_MODEL_PATH="/tmp/unsafe"),
            "forbidden overrides",
        ),
    ],
)
def test_preflight_rejects_mismatch(
    inputs: dict[str, Path], file: str, edit, message: str
) -> None:
    _edit_json(inputs[file], edit)
    with pytest.raises(AdapterError, match=message):
        _plan(inputs)


def test_preflight_rejects_duplicate_devices(inputs: dict[str, Path]) -> None:
    _edit_json(inputs["request"], lambda x: x.update(npu_count=2))
    with pytest.raises(AdapterError, match="assigned NPUs"):
        _plan(inputs, "2,2")


def test_preflight_rejects_missing_runtime_cwd(inputs: dict[str, Path]) -> None:
    _edit_json(
        inputs["schedule"],
        lambda x: x.update(runtime_cwd=str(inputs["request"].parent / "missing")),
    )
    with pytest.raises(AdapterError, match="runtime_cwd does not exist"):
        _plan(inputs)


def test_preflight_rejects_symlink_runtime_cwd(inputs: dict[str, Path]) -> None:
    link = inputs["request"].parent / "runtime-link"
    link.symlink_to(inputs["request"].parent / "runtime-cwd", target_is_directory=True)
    _edit_json(inputs["schedule"], lambda x: x.update(runtime_cwd=str(link)))
    with pytest.raises(AdapterError, match="runtime_cwd must not be a symlink"):
        _plan(inputs)


def test_plan_accepts_local_model_manifest_digest(inputs: dict[str, Path]) -> None:
    model = Path(
        json.loads(inputs["schedule"].read_text())["targets"][TARGET_ID]["model_path"]
    )
    (model / "weights.safetensors").write_bytes(b"immutable-weights")
    file_sha = hashlib.sha256(b"immutable-weights").hexdigest()
    manifest_sha = hashlib.sha256(
        f"weights.safetensors:{file_sha}".encode()
    ).hexdigest()
    _edit_json(
        inputs["schedule"],
        lambda x: x["targets"][TARGET_ID].update(model_revision=manifest_sha),
    )
    assert _plan(inputs).environment["CURRENT_MODEL_REVISION"] == manifest_sha


def test_plan_rejects_unbound_model_revision(inputs: dict[str, Path]) -> None:
    (
        Path(
            json.loads(inputs["schedule"].read_text())["targets"][TARGET_ID][
                "model_path"
            ]
        )
        / "refs/main"
    ).write_text("c" * 40 + "\n", encoding="utf-8")
    with pytest.raises(AdapterError, match="not bound"):
        _plan(inputs)


def test_preflight_rejects_dirty_or_wrong_commit(inputs: dict[str, Path]) -> None:
    (inputs["core"] / "untracked").write_text("dirty", encoding="utf-8")
    with pytest.raises(AdapterError, match="dirty or at the wrong commit"):
        _plan(inputs)
    (inputs["core"] / "untracked").unlink()
    _edit_json(inputs["request"], lambda x: x.update(core_commit="f" * 40))
    with pytest.raises(AdapterError, match="dirty or at the wrong commit"):
        _plan(inputs)


def test_run_plan_keeps_failure_summary_and_attempt(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    submissions = root / "submissions"
    submissions.mkdir(parents=True)
    output = tmp_path / "output"
    attempt = submissions / "attempt-one"
    script = tmp_path / "fake-runner.py"
    script.write_text(
        "import json,os,pathlib,sys\n"
        "p=pathlib.Path(os.environ['ATTEMPT']); p.mkdir(); (p/'raw.txt').write_text('failed')\n"
        "s={'runs':[{'repeat_index':0,'artifact_dir':str(p)}]}\n"
        "pathlib.Path(os.environ['CAMPAIGN_SUMMARY_FILE']).write_text(json.dumps(s))\n"
        "sys.exit(19)\n",
        encoding="utf-8",
    )
    plan = ExecutionPlan(
        command=(sys.executable, str(script)),
        environment={
            "PATH": os.defpath,
            "ATTEMPT": str(attempt),
            "CAMPAIGN_SUMMARY_FILE": str(output / "campaign-summary.json"),
        },
        benchmark_root=root,
        output_dir=output,
        submissions_dir=submissions,
        summary_file=output / "campaign-summary.json",
        repeat_count=1,
        target_id="test-target",
    )
    assert run_plan(plan) == 19
    assert (output / "attempts/repeat-00/raw.txt").read_text() == "failed"
    assert (output / "STATUS").read_text().startswith("FAILED")


def test_run_plan_never_marks_success_as_publishable(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    root.mkdir()
    output = tmp_path / "output"
    script = tmp_path / "fake-success.py"
    script.write_text(
        "import json,os,pathlib\n"
        "pathlib.Path(os.environ['CAMPAIGN_SUMMARY_FILE']).write_text(json.dumps({'runs':[]}))\n",
        encoding="utf-8",
    )
    plan = ExecutionPlan(
        (sys.executable, str(script)),
        {
            "PATH": os.defpath,
            "CAMPAIGN_SUMMARY_FILE": str(output / "campaign-summary.json"),
        },
        root,
        output,
        root / "submissions",
        output / "campaign-summary.json",
        0,
        "test",
    )
    assert run_plan(plan) == 0
    assert "UNVERIFIED" in (output / "STATUS").read_text()
    assert (output / "schedule-snapshot.json").read_bytes() == b""


def test_run_plan_rejects_missing_attempt_source(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    submissions = root / "submissions"
    submissions.mkdir(parents=True)
    output = tmp_path / "output"
    script = tmp_path / "fake-success.py"
    script.write_text(
        "import json,os,pathlib\n"
        "missing=pathlib.Path(os.environ['SUBMISSIONS'])/'missing'\n"
        "summary={'runs':[{'repeat_index':0,'artifact_dir':str(missing)}]}\n"
        "pathlib.Path(os.environ['CAMPAIGN_SUMMARY_FILE']).write_text(json.dumps(summary))\n",
        encoding="utf-8",
    )
    plan = ExecutionPlan(
        (sys.executable, str(script)),
        {
            "PATH": os.defpath,
            "SUBMISSIONS": str(submissions),
            "CAMPAIGN_SUMMARY_FILE": str(output / "campaign-summary.json"),
        },
        root,
        output,
        submissions,
        output / "campaign-summary.json",
        1,
        "test",
    )
    assert run_plan(plan) == 2
    assert "directory is missing" in (output / "retention-error.txt").read_text()


def test_run_plan_rejects_symlink_in_attempt(tmp_path: Path) -> None:
    root = tmp_path / "benchmark"
    attempts = root / "submissions"
    attempt = attempts / "run"
    attempt.mkdir(parents=True)
    (attempt / "escaped").symlink_to(tmp_path / "outside")
    output = tmp_path / "output"
    script = tmp_path / "fake-success.py"
    script.write_text(
        "import json,os,pathlib\n"
        "pathlib.Path(os.environ['CAMPAIGN_SUMMARY_FILE']).write_text(json.dumps({'runs':[{'repeat_index':0,'artifact_dir':os.environ['ATTEMPT']}]}))\n",
        encoding="utf-8",
    )
    plan = ExecutionPlan(
        (sys.executable, str(script)),
        {
            "PATH": os.defpath,
            "CAMPAIGN_SUMMARY_FILE": str(output / "campaign-summary.json"),
            "ATTEMPT": str(attempt),
        },
        root,
        output,
        attempts,
        output / "campaign-summary.json",
        1,
        "test",
    )
    assert run_plan(plan) == 2
    assert "symlink" in (output / "retention-error.txt").read_text()


def test_campaign_port_probe_does_not_depend_on_ss() -> None:
    script = (ROOT / "scripts/run-campaign-repetitions.sh").read_text(encoding="utf-8")
    probe = script.split("wait_for_port_free()", 1)[1].split(
        "# ─── Main repetition loop", 1
    )[0]
    assert 'python3 - "$port"' in probe
    assert 'probe.bind(("0.0.0.0", port))' in probe
    assert "command -v ss" in probe
