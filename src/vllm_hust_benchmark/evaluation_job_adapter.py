"""Fail-closed bridge from the 112 worker contract to a frozen campaign runner.

This module starts and retains evidence. It does not admit or publish results.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.snapshot_target_binding import load_official_target_registry

COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
DEVICES_RE = re.compile(r"[0-9]+(?:,[0-9]+)*\Z")
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\Z")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
PRIORITIES = {"release", "required", "normal", "diagnostic"}
SCHEDULE_SCHEMA = "evaluation-112-schedule/v1"
REQUIRED_REQUEST = {
    "schema_version",
    "repository",
    "core_commit",
    "plugin_repository",
    "plugin_commit",
    "target_id",
    "target_registry_version",
    "repeat_count",
    "npu_count",
    "priority",
    "requested_by",
    "source_url",
}
REQUIRED_SCHEDULE_ENTRY = {
    "approval",
    "source_spec_sha256",
    "model_path",
    "model_revision",
    "load_profile",
    "campaign_id",
    "coverage_class",
    "point_role",
}
ALLOWED_ADMIN_ENV = {
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TZ",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
    "ASCEND_ATB_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "OMP_NUM_THREADS",
    "HCCL_CONNECT_TIMEOUT",
}


class AdapterError(ValueError):
    """An untrusted request or incomplete machine schedule cannot be executed."""


@dataclass(frozen=True)
class ExecutionPlan:
    command: tuple[str, ...]
    environment: dict[str, str]
    benchmark_root: Path
    output_dir: Path
    submissions_dir: Path
    summary_file: Path
    repeat_count: int
    target_id: str
    schedule_version: str = ""
    schedule_sha256: str = ""
    schedule_snapshot: bytes = b""


def _object_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdapterError(f"expected JSON object: {path}")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{name} must be a non-empty string")
    return value


def _clean_commit(path: Path, expected: str, name: str) -> None:
    if not COMMIT_RE.fullmatch(expected):
        raise AdapterError(f"{name} must be a lowercase full Git commit")
    if not path.is_dir():
        raise AdapterError(f"{name} repository is missing")
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if (
        head.returncode
        or head.stdout.strip() != expected
        or status.returncode
        or status.stdout
    ):
        raise AdapterError(f"{name} repository is dirty or at the wrong commit")


def _assigned_npus(raw: str, count: int) -> str:
    if not DEVICES_RE.fullmatch(raw):
        raise AdapterError("assigned NPUs must be a comma-separated numeric list")
    devices = raw.split(",")
    if len(devices) != count or len(set(devices)) != count:
        raise AdapterError("assigned NPUs do not match request npu_count")
    if any(str(int(device)) != device for device in devices):
        raise AdapterError("assigned NPUs must use canonical numeric IDs")
    return raw


def _model_weight_manifest_digest(model_path: Path) -> str:
    """Fingerprint the actual top-level model weight files.

    This intentionally matches the content-fingerprint convention already used
    by the capacity runner: ``sha256("name:file_sha\n...")``.  Configuration
    files alone are not an immutable model identity.
    """
    entries: list[str] = []
    for path in sorted(model_path.iterdir(), key=lambda item: item.name):
        if not path.name.endswith((".safetensors", ".bin", ".pt")):
            continue
        if path.is_symlink() or not path.is_file():
            raise AdapterError("model weight files must be regular files")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append(f"{path.name}:{digest.hexdigest()}")
    if not entries:
        raise AdapterError("scheduled model path contains no model weight files")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def _bind_model_revision(model_path: Path, revision: str) -> None:
    if SHA256_RE.fullmatch(revision):
        if _model_weight_manifest_digest(model_path) != revision:
            raise AdapterError("model weight manifest digest mismatch")
        return

    candidates = (model_path / "refs/main", model_path / ".cache/refs/main")
    observed = ""
    for candidate in candidates:
        if candidate.is_symlink():
            raise AdapterError("model revision reference must not be a symlink")
        if candidate.is_file():
            observed = candidate.read_text(encoding="utf-8").strip()
            break
    if not observed and (model_path / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(model_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if result.returncode == 0:
            observed = result.stdout.strip()
    if observed != revision:
        raise AdapterError("model revision is not bound to the scheduled model files")


def build_plan(
    request_file: Path, assigned_npus: str, output_dir: Path, schedule_file: Path
) -> ExecutionPlan:
    request = _object_file(request_file)
    schedule_bytes = schedule_file.read_bytes()
    schedule = json.loads(schedule_bytes)
    if not isinstance(schedule, dict):
        raise AdapterError(f"expected JSON object: {schedule_file}")
    if set(request) - (REQUIRED_REQUEST | {"metadata"}) or REQUIRED_REQUEST - set(
        request
    ):
        raise AdapterError("request fields do not match evaluation/v1")
    if request["schema_version"] != 1:
        raise AdapterError("unsupported request schema")
    for name in ("repository", "plugin_repository"):
        if not isinstance(request[name], str) or not REPOSITORY_RE.fullmatch(
            request[name]
        ):
            raise AdapterError(f"{name} is invalid")
    if not isinstance(
        request["target_registry_version"], str
    ) or not VERSION_RE.fullmatch(request["target_registry_version"]):
        raise AdapterError("target_registry_version must be semantic x.y.z")
    if request["priority"] not in PRIORITIES:
        raise AdapterError("request priority is invalid")
    for name in ("requested_by", "source_url", "target_id"):
        _required_string(request[name], name)
    if "metadata" in request and not isinstance(request["metadata"], dict):
        raise AdapterError("request metadata must be an object")
    for name in ("core_commit", "plugin_commit"):
        if not isinstance(request[name], str) or not COMMIT_RE.fullmatch(request[name]):
            raise AdapterError(f"{name} must be a lowercase full Git commit")
    repeats, count = request["repeat_count"], request["npu_count"]
    if type(repeats) is not int or not 3 <= repeats <= 9:
        raise AdapterError("repeat_count must be between 3 and 9")
    if type(count) is not int or not 1 <= count <= 64:
        raise AdapterError("npu_count must be between 1 and 64")
    devices = _assigned_npus(assigned_npus, count)

    if schedule.get("schema_version") != SCHEDULE_SCHEMA:
        raise AdapterError("unsupported administrator schedule schema")
    schedule_version = _required_string(
        schedule.get("schedule_version"), "schedule_version"
    )
    if not VERSION_RE.fullmatch(schedule_version):
        raise AdapterError("schedule_version must be semantic x.y.z")
    schedule_sha256 = hashlib.sha256(schedule_bytes).hexdigest()
    benchmark_root = Path(
        _required_string(schedule.get("benchmark_root"), "benchmark_root")
    ).resolve()
    core_repo = Path(_required_string(schedule.get("core_repo"), "core_repo")).resolve()
    plugin_repo = Path(
        _required_string(schedule.get("plugin_repo"), "plugin_repo")
    ).resolve()
    if request["repository"] != schedule.get("core_repository") or request[
        "plugin_repository"
    ] != schedule.get("plugin_repository"):
        raise AdapterError("request repository is not the scheduled repository")
    if benchmark_root != Path(__file__).resolve().parents[2]:
        raise AdapterError("schedule benchmark_root does not contain this adapter")
    registry = load_official_target_registry(benchmark_root)
    if (
        request["target_registry_version"] != registry.version
        or schedule.get("registry_version") != registry.version
    ):
        raise AdapterError("target registry version mismatch")
    target_id = _required_string(request["target_id"], "target_id")
    target = registry.targets.get(target_id)
    entry = (schedule.get("targets") or {}).get(target_id)
    if (
        target is None
        or not isinstance(entry, dict)
        or entry.get("approval") != "approved"
    ):
        raise AdapterError(
            "target is not explicitly approved in the administrator schedule"
        )
    if REQUIRED_SCHEDULE_ENTRY - set(entry):
        raise AdapterError("approved target schedule is incomplete")
    if target.get("status") not in ("active", "provisional"):
        raise AdapterError("target registry status is not executable")
    if entry.get("registry_status") != target["status"]:
        raise AdapterError("scheduled target status differs from registry")
    spec_ref = target.get("source_spec") or {}
    relative = Path(_required_string(spec_ref.get("path"), "source_spec.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise AdapterError("source spec escapes benchmark repository")
    spec_file = (benchmark_root / relative).resolve()
    if not spec_file.is_relative_to(benchmark_root) or not spec_file.is_file():
        raise AdapterError("source spec is missing or escapes benchmark repository")
    spec_sha = hashlib.sha256(spec_file.read_bytes()).hexdigest()
    if spec_sha != spec_ref.get("sha256") or spec_sha != entry["source_spec_sha256"]:
        raise AdapterError("source spec checksum mismatch")
    spec = _object_file(spec_file)
    if (
        spec.get("id") != target_id
        or spec.get("chip_count") != count
        or spec.get("node_count") != 1
    ):
        raise AdapterError("source spec target or topology does not match request")
    if (target.get("hardware") or {}).get("chip_count") != count:
        raise AdapterError("registry chip count does not match request")
    scheduled_model_path = Path(_required_string(entry["model_path"], "model_path"))
    if scheduled_model_path.is_symlink():
        raise AdapterError("scheduled model path must not be a symlink")
    model_path = scheduled_model_path.resolve()
    if not model_path.is_dir():
        raise AdapterError("scheduled model path is not a local directory")
    model_revision = _required_string(entry["model_revision"], "model_revision")
    if not (COMMIT_RE.fullmatch(model_revision) or SHA256_RE.fullmatch(model_revision)):
        raise AdapterError(
            "model_revision must be an immutable 40-character revision or "
            "64-character manifest digest"
        )
    _bind_model_revision(model_path, model_revision)
    if entry.get("model_id") != spec.get("model") or entry.get(
        "model_precision"
    ) != spec.get("model_precision"):
        raise AdapterError("scheduled model identity or precision differs from spec")
    if entry.get("hardware_chip_model") != spec.get("hardware_chip_model"):
        raise AdapterError("scheduled hardware differs from spec")
    if not entry.get("load_profile") or not entry.get("campaign_id"):
        raise AdapterError("campaign identity and load profile are required")
    coverage = entry["coverage_class"]
    role = entry["point_role"]
    if (coverage, role) not in {
        ("full-matrix", "checkpoint"),
        ("targeted-pair", "baseline"),
        ("targeted-pair", "head"),
    }:
        raise AdapterError("invalid scheduled coverage class or role")
    if coverage == "targeted-pair" and not entry.get("comparison_id"):
        raise AdapterError("targeted-pair schedule requires comparison_id")
    _clean_commit(core_repo, request["core_commit"], "core")
    _clean_commit(plugin_repo, request["plugin_commit"], "plugin")
    for name in (
        "image_id",
        "cann_version",
        "torch_npu_version",
        "topology",
        "runtime_python",
        "runtime_cwd",
    ):
        _required_string(schedule.get(name), name)
    image_id = schedule["image_id"].removeprefix("sha256:")
    if not SHA256_RE.fullmatch(image_id):
        raise AdapterError("image_id must be a full SHA256 digest")
    if not Path(schedule["runtime_python"]).is_file() or not os.access(
        schedule["runtime_python"], os.X_OK
    ):
        raise AdapterError("scheduled runtime_python does not exist")
    scheduled_runtime_cwd = Path(schedule["runtime_cwd"])
    if scheduled_runtime_cwd.is_symlink():
        raise AdapterError("scheduled runtime_cwd must not be a symlink")
    runtime_cwd = scheduled_runtime_cwd.resolve()
    if not runtime_cwd.is_dir():
        raise AdapterError("scheduled runtime_cwd does not exist")
    configured_env = schedule.get("environment", {})
    if not isinstance(configured_env, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or k not in ALLOWED_ADMIN_ENV
        for k, v in configured_env.items()
    ):
        raise AdapterError("administrator environment contains forbidden overrides")
    # Never inherit worker environment: it may contain secrets or execution overrides.
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", **configured_env}
    env.update(
        {
            "CAMPAIGN_REQUIRE_FROZEN_INPUTS": "1",
            "CAMPAIGN_ID": entry["campaign_id"],
            "CAMPAIGN_COVERAGE_CLASS": coverage,
            "CAMPAIGN_POINT_ROLE": role,
            "CAMPAIGN_LOAD_PROFILE": entry["load_profile"],
            "CAMPAIGN_COMPARISON_ID": entry.get("comparison_id", ""),
            "CURRENT_VLLM_HUST_REPO": str(core_repo),
            "CURRENT_VLLM_ASCEND_HUST_REPO": str(plugin_repo),
            "CURRENT_GIT_COMMIT": request["core_commit"],
            "CURRENT_PLUGIN_GIT_COMMIT": request["plugin_commit"],
            "CURRENT_IMAGE_ID": schedule["image_id"],
            "CURRENT_MODEL_REVISION": entry["model_revision"],
            "CURRENT_MODEL_PATH": str(model_path),
            "CURRENT_CANN_VERSION": schedule["cann_version"],
            "CURRENT_TORCH_NPU_VERSION": schedule["torch_npu_version"],
            "CURRENT_TOPOLOGY": schedule["topology"],
            "CURRENT_RUNTIME_PYTHON": schedule["runtime_python"],
            "CURRENT_RUNTIME_CWD": str(runtime_cwd),
            "ASCEND_RT_VISIBLE_DEVICES": devices,
            "ASCEND_VISIBLE_DEVICES": devices,
            "PERFGATE_WARMUP_RUNS": "0",
            "PERFGATE_MEASURED_RUNS": "1",
            "CURRENT_EXPORT_ONLY": "0",
            "CURRENT_USE_MANAGED_SERVER": "0",
            "CURRENT_RUNTIME_SOURCE_MODE": "worktree",
            "MAX_PORT_WAIT_SECONDS": "120",
        }
    )
    if output_dir.is_symlink():
        raise AdapterError("evaluation output directory must not be a symlink")
    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise AdapterError("evaluation output directory is not empty")
    if output_dir.is_relative_to(benchmark_root):
        raise AdapterError(
            "evaluation output directory must be outside benchmark source"
        )
    prefix = "eval112-" + hashlib.sha256(str(output_dir).encode()).hexdigest()[:20]
    summary_file = output_dir / "campaign-summary.json"
    env["CAMPAIGN_SUMMARY_FILE"] = str(summary_file)
    command = (
        "/bin/bash",
        str(benchmark_root / "scripts/run-campaign-repetitions.sh"),
        str(spec_file),
        "--campaign-prefix",
        prefix,
        "--repetitions",
        str(repeats),
    )
    return ExecutionPlan(
        command,
        env,
        benchmark_root,
        output_dir,
        benchmark_root / "submissions",
        summary_file,
        repeats,
        target_id,
        schedule_version,
        schedule_sha256,
        schedule_bytes,
    )


def _retain_attempts(plan: ExecutionPlan) -> None:
    if not plan.summary_file.is_file():
        raise AdapterError("campaign summary is missing")
    summary = _object_file(plan.summary_file)
    rows = summary.get("runs")
    if not isinstance(rows, list):
        raise AdapterError("campaign summary has no runs array")
    if [row.get("repeat_index") for row in rows if isinstance(row, dict)] != list(
        range(plan.repeat_count)
    ):
        raise AdapterError(
            "campaign summary does not contain all requested repeat indices"
        )
    destination = plan.output_dir / "attempts"
    destination.mkdir(exist_ok=True)
    for row in rows:
        index = row.get("repeat_index") if isinstance(row, dict) else None
        raw = row.get("artifact_dir") if isinstance(row, dict) else None
        if type(index) is not int or index < 0 or not isinstance(raw, str):
            raise AdapterError("campaign summary contains an invalid attempt")
        source = Path(raw).resolve()
        if not source.is_relative_to(plan.submissions_dir.resolve()):
            raise AdapterError("campaign artifact path escapes submissions directory")
        if not source.is_dir():
            raise AdapterError(
                f"campaign artifact directory is missing: repeat {index}"
            )
        if any(path.is_symlink() for path in source.rglob("*")):
            raise AdapterError("campaign artifact contains a symlink")
        shutil.copytree(source, destination / f"repeat-{index:02d}", symlinks=False)


def run_plan(plan: ExecutionPlan) -> int:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "schedule-snapshot.json").write_bytes(plan.schedule_snapshot)
    (plan.output_dir / "execution-plan.json").write_text(
        json.dumps(
            {
                "target_id": plan.target_id,
                "command": plan.command,
                "repeat_count": plan.repeat_count,
                "schedule_version": plan.schedule_version,
                "schedule_sha256": plan.schedule_sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    process: subprocess.Popen[bytes] | None = None
    interrupted = False
    terminate_deadline: float | None = None

    def forward(signum: int, _frame: Any) -> None:
        nonlocal interrupted, terminate_deadline
        interrupted = True
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)
            terminate_deadline = time.monotonic() + 20

    previous_term = signal.signal(signal.SIGTERM, forward)
    previous_int = signal.signal(signal.SIGINT, forward)
    exit_code = 2
    try:
        with (plan.output_dir / "runner.log").open("wb") as log:
            process = subprocess.Popen(
                plan.command,
                cwd=plan.benchmark_root,
                env=plan.environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                if (
                    terminate_deadline is not None
                    and time.monotonic() >= terminate_deadline
                ):
                    os.killpg(process.pid, signal.SIGKILL)
                    terminate_deadline = None
                time.sleep(0.2)
            exit_code = process.wait()
    except OSError as exc:
        (plan.output_dir / "runner-error.txt").write_text(
            str(exc) + "\n", encoding="utf-8"
        )
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    try:
        _retain_attempts(plan)
    except (AdapterError, OSError, ValueError) as exc:
        (plan.output_dir / "retention-error.txt").write_text(
            str(exc) + "\n", encoding="utf-8"
        )
        exit_code = exit_code or 2
    (plan.output_dir / "STATUS").write_text(
        "UNVERIFIED: runner completed; publication gate not implemented\n"
        if exit_code == 0 and not interrupted
        else f"FAILED: runner exit {exit_code}\n",
        encoding="utf-8",
    )
    return 130 if interrupted and exit_code == 0 else exit_code


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="112 evaluation worker adapter (no publication)"
    )
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = build_plan(
            Path(os.environ["EVALUATION_REQUEST_FILE"]),
            os.environ["EVALUATION_ASSIGNED_NPUS"],
            Path(os.environ["EVALUATION_OUTPUT_DIR"]),
            args.schedule,
        )
        for utility in ("python3", "jq", "git"):
            if shutil.which(utility, path=plan.environment["PATH"]) is None:
                raise AdapterError(
                    f"required formal runner utility is missing: {utility}"
                )
        return run_plan(plan)
    except (AdapterError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"evaluation adapter rejected job: {exc}", flush=True)
        return 2
