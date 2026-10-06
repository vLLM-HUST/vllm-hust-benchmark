"""Offline admission verifier for immutable 112 evaluation bundles.

The worker seals a bundle before this module runs.  Verification never writes
inside that bundle; the resulting attestation belongs in a separate directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.evaluation_job_adapter import AdapterError

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CHECKSUM_RE = re.compile(r"(?P<sha>[0-9a-f]{64})\s+\*?(?P<path>.+)\Z")
REQUIRED_ARTIFACT_FILES = {
    "STATUS",
    "checksums.sha256",
    "env-manifest.json",
    "leaderboard_manifest.json",
    "pip-packages.json",
    "raw_benchmark_result.json",
    "resolved_same_spec.json",
    "run_leaderboard.json",
}


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"invalid JSON object: {path}")
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_tree_sha256(bundle: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise AdapterError("evaluation bundle contains a symlink")
        if not path.is_file() or path.name == "BUNDLE_SHA256":
            continue
        relative = path.relative_to(bundle).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(artifact: Path) -> None:
    manifest = artifact / "checksums.sha256"
    covered: set[str] = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        match = CHECKSUM_RE.fullmatch(raw)
        if match is None:
            raise AdapterError(f"malformed artifact checksum line: {raw!r}")
        relative = match.group("path").removeprefix("./")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise AdapterError("artifact checksum path escapes artifact")
        target = artifact / candidate
        if target.is_symlink() or not target.is_file():
            raise AdapterError(f"artifact checksum target is missing: {relative}")
        if _sha256_file(target) != match.group("sha"):
            raise AdapterError(f"artifact checksum mismatch: {relative}")
        covered.add(candidate.as_posix())
    required_covered = REQUIRED_ARTIFACT_FILES - {"STATUS", "checksums.sha256"}
    if not required_covered <= covered:
        raise AdapterError("artifact checksum manifest omits required evidence")
    actual = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file() and path.name not in {"STATUS", "checksums.sha256"}
    }
    if covered != actual:
        raise AdapterError(
            "artifact checksum manifest does not cover exactly all evidence"
        )


def _job_assigned_npus(job: dict[str, Any]) -> list[int]:
    assigned = job.get("assigned_npus")
    if isinstance(assigned, str):
        try:
            assigned = json.loads(assigned)
        except json.JSONDecodeError as exc:
            raise AdapterError("API job has malformed assigned_npus") from exc
    if not isinstance(assigned, list) or any(
        type(value) is not int for value in assigned
    ):
        raise AdapterError("API job has invalid assigned_npus")
    return assigned


def _verify_artifact(
    artifact: Path,
    *,
    repeat_index: int,
    request: dict[str, Any],
    schedule_entry: dict[str, Any],
    assigned_devices: str,
) -> None:
    if any(path.is_symlink() for path in artifact.rglob("*")):
        raise AdapterError("retained artifact contains a symlink")
    missing = REQUIRED_ARTIFACT_FILES - {path.name for path in artifact.iterdir()}
    if missing:
        raise AdapterError(f"retained artifact is incomplete: {sorted(missing)}")
    if (artifact / "STATUS").read_text(encoding="utf-8").strip() != "OK":
        raise AdapterError("retained artifact STATUS is not OK")
    _verify_checksums(artifact)

    env = _json_object(artifact / "env-manifest.json")
    if env.get("manifest_version") != "run-env-manifest/v2" or not env.get(
        "frozen_inputs_required"
    ):
        raise AdapterError("artifact does not use frozen environment provenance")
    frozen = env.get("frozen_inputs") or {}
    campaign = env.get("campaign") or {}
    git_info = env.get("git_info") or {}
    if frozen.get("model_revision") != schedule_entry["model_revision"]:
        raise AdapterError("artifact model revision differs from schedule")
    if frozen.get("image_id") != schedule_entry["image_id"]:
        raise AdapterError("artifact image identity differs from schedule")
    if frozen.get("topology") != schedule_entry["topology"]:
        raise AdapterError("artifact topology differs from schedule")
    if (frozen.get("cann") or {}).get("declared") != schedule_entry["cann_version"]:
        raise AdapterError("artifact CANN version differs from schedule")
    if (frozen.get("torch_npu_version") or {}).get("declared") != schedule_entry[
        "torch_npu_version"
    ]:
        raise AdapterError("artifact torch-npu version differs from schedule")
    expected_campaign = {
        "campaign_id": schedule_entry["campaign_id"],
        "coverage_class": schedule_entry["coverage_class"],
        "comparison_id": schedule_entry.get("comparison_id", ""),
        "point_role": schedule_entry["point_role"],
        "load_profile": schedule_entry["load_profile"],
        "repeat_index": repeat_index,
        "repetitions": request["repeat_count"],
    }
    if any(campaign.get(key) != value for key, value in expected_campaign.items()):
        raise AdapterError("artifact campaign lifecycle differs from schedule")
    if (env.get("env_vars") or {}).get("ASCEND_RT_VISIBLE_DEVICES") != assigned_devices:
        raise AdapterError("artifact NPU assignment differs from worker")
    for key, commit in (
        ("vllm_hust", request["core_commit"]),
        ("vllm_ascend_hust", request["plugin_commit"]),
    ):
        source = git_info.get(key) or {}
        if source.get("declared") != commit or source.get("observed") != commit:
            raise AdapterError("artifact source commit differs from request")

    run = _json_object(artifact / "run_leaderboard.json")
    metadata = run.get("metadata") or {}
    if metadata.get("target_contract_id") != request["target_id"]:
        raise AdapterError("artifact target identity differs from request")
    resolved = _json_object(artifact / "resolved_same_spec.json")
    if resolved.get("model") != schedule_entry["model_id"]:
        raise AdapterError("resolved artifact model differs from schedule")
    scenario = resolved.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise AdapterError("resolved artifact scenario is missing")
    if scenario.endswith("-online"):
        if not (artifact / "server.stdout.log").is_file():
            raise AdapterError("online artifact is missing server lifecycle log")
    elif not (artifact / "offline_graph_proof.json").is_file():
        raise AdapterError("offline artifact is missing graph-mode proof")


def verify_bundle(
    bundle: Path, job_record_file: Path, schedule_file: Path
) -> dict[str, Any]:
    bundle = bundle.resolve()
    if not bundle.is_dir() or bundle.is_symlink():
        raise AdapterError("evaluation bundle is missing or is a symlink")
    job = _json_object(job_record_file)
    schedule_bytes = schedule_file.read_bytes()
    schedule = json.loads(schedule_bytes)
    if not isinstance(schedule, dict):
        raise AdapterError("administrator schedule must be an object")

    recorded_bundle_sha = (bundle / "BUNDLE_SHA256").read_text(encoding="utf-8").strip()
    actual_bundle_sha = bundle_tree_sha256(bundle)
    if (
        not SHA256_RE.fullmatch(recorded_bundle_sha)
        or recorded_bundle_sha != actual_bundle_sha
    ):
        raise AdapterError("worker bundle checksum mismatch")
    if job.get("artifact_sha256") != actual_bundle_sha:
        raise AdapterError("API job artifact checksum differs from bundle")
    if job.get("status") != "succeeded" or job.get("exit_code") != 0:
        raise AdapterError("API job did not finish successfully")
    if Path(str(job.get("artifact_path", ""))).resolve() != bundle:
        raise AdapterError("API job artifact path differs from bundle")

    request = job.get("request")
    if not isinstance(request, dict):
        raise AdapterError("API job request is missing")
    request_bytes = (bundle / "request.json").read_bytes()
    if request_bytes != _canonical_json(request) + b"\n":
        raise AdapterError("bundle request is not the canonical API request")
    worker = _json_object(bundle / "worker.json")
    assigned = worker.get("assigned_npus")
    job_assigned = _job_assigned_npus(job)
    if (
        worker.get("job_id") != job.get("id")
        or worker.get("exit_code") != 0
        or not isinstance(assigned, list)
        or assigned != job_assigned
        or worker.get("worker") != job.get("worker_id")
    ):
        raise AdapterError("worker lifecycle differs from API job record")
    if len(assigned) != request.get("npu_count") or any(
        type(device) is not int for device in assigned
    ):
        raise AdapterError("worker NPU assignment differs from request")
    assigned_devices = ",".join(map(str, assigned))

    result = bundle / "result"
    if (result / "STATUS").read_text(encoding="utf-8").strip() != (
        "UNVERIFIED: runner completed; publication gate not implemented"
    ):
        raise AdapterError("adapter result is not in the expected unverified state")
    schedule_snapshot = (result / "schedule-snapshot.json").read_bytes()
    if schedule_snapshot != schedule_bytes:
        raise AdapterError("bundle schedule snapshot differs from verifier schedule")
    plan = _json_object(result / "execution-plan.json")
    schedule_sha = hashlib.sha256(schedule_bytes).hexdigest()
    if (
        plan.get("target_id") != request.get("target_id")
        or plan.get("repeat_count") != request.get("repeat_count")
        or plan.get("schedule_sha256") != schedule_sha
        or plan.get("schedule_version") != schedule.get("schedule_version")
    ):
        raise AdapterError("execution plan differs from request or schedule")
    summary = _json_object(result / "campaign-summary.json")
    repeats = request["repeat_count"]
    rows = summary.get("runs")
    if (
        summary.get("schema_version") != "independent-service-campaign-summary/v1"
        or summary.get("status") != "ok"
        or summary.get("requested_repetitions") != repeats
        or summary.get("attempted_repetitions") != repeats
        or summary.get("successful_repetitions") != repeats
        or not isinstance(rows, list)
        or len(rows) != repeats
    ):
        raise AdapterError("campaign summary is not a complete successful suite")
    target_entry = (schedule.get("targets") or {}).get(request["target_id"])
    if not isinstance(target_entry, dict):
        raise AdapterError("target is missing from verifier schedule")
    schedule_entry = {
        **target_entry,
        "image_id": schedule.get("image_id"),
        "cann_version": schedule.get("cann_version"),
        "torch_npu_version": schedule.get("torch_npu_version"),
        "topology": schedule.get("topology"),
    }
    summary_frozen = summary.get("frozen_inputs") or {}
    expected_frozen = {
        "core_commit": request["core_commit"],
        "backend_commit": request["plugin_commit"],
        "image_id": schedule["image_id"],
        "model_revision": target_entry["model_revision"],
        "cann_version": schedule["cann_version"],
        "torch_npu_version": schedule["torch_npu_version"],
        "topology": schedule["topology"],
    }
    expected_summary = {
        "campaign_id": target_entry["campaign_id"],
        "coverage_class": target_entry["coverage_class"],
        "comparison_id": target_entry.get("comparison_id", ""),
        "point_role": target_entry["point_role"],
        "load_profile": target_entry["load_profile"],
        "visible_devices": assigned_devices,
    }
    if summary_frozen != expected_frozen or any(
        summary.get(key) != value for key, value in expected_summary.items()
    ):
        raise AdapterError(
            "campaign summary provenance differs from request or schedule"
        )
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("repeat_index") != index
            or row.get("exit_code") != 0
            or row.get("status") != "ok"
        ):
            raise AdapterError("campaign summary contains an invalid repetition")
        _verify_artifact(
            result / "attempts" / f"repeat-{index:02d}",
            repeat_index=index,
            request=request,
            schedule_entry=schedule_entry,
            assigned_devices=assigned_devices,
        )

    return {
        "schema_version": "evaluation-112-admission-attestation/v1",
        "status": "VERIFIED",
        "job_id": job["id"],
        "bundle_sha256": actual_bundle_sha,
        "request_sha256": hashlib.sha256(_canonical_json(request)).hexdigest(),
        "schedule_sha256": schedule_sha,
        "target_id": request["target_id"],
        "repeat_count": repeats,
        "assigned_npus": assigned,
    }
