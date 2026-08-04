from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.baseline_recovery import (
    _identity_mismatches,
    _parameter_mismatches,
)
from vllm_hust_benchmark.immutable_input_attestation import (
    validate_attestation_payload,
)
from vllm_hust_benchmark.strict_execution_contract import (
    CANONICAL_WORKER_RULE,
    canonical_worker_key,
)

ATTESTATION_SCHEMA_VERSION = "official-baseline-attestation/v1"
STRICT_EXECUTION_SCHEMA_VERSION = "strict-execution-evidence/v1"
CLEANUP_CHAIN_SCHEMA_VERSION = "cleanup-chain-attestation/v1"
IMMUTABLE_INPUT_SCHEMA_VERSION = "immutable-input-attestation/v1"
TRACE_PROFILE = "production-trace"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_trace_detail_metadata(path: Path) -> tuple[dict[str, Any], int]:
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"trace detail is empty: {path}")
    metadata = json.loads(lines[0])
    if not isinstance(metadata, dict) or metadata.get("type") != "metadata":
        raise ValueError(f"trace detail metadata is missing: {path}")
    return metadata, len(lines) - 1


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"strict execution evidence is missing {field}")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"strict execution evidence has invalid {field}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"strict execution evidence has naive {field}")
    return parsed


def _evidence_path(repeat_dir: Path, value: Any, *, field: str) -> Path:
    relative = Path(str(value or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"strict execution evidence has invalid {field}")
    path = repeat_dir / relative
    if not path.is_file():
        raise ValueError(f"strict execution evidence is missing {field}: {path}")
    return path


def _verify_hashed_evidence_file(
    repeat_dir: Path, record: Mapping[str, Any], *, field: str
) -> Path:
    path = _evidence_path(repeat_dir, record.get("path"), field=f"{field}.path")
    digest = str(record.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or _sha256(path) != digest:
        raise ValueError(f"strict execution evidence hash mismatch: {field}")
    return path


def _validate_strict_execution_evidence(
    repeat_dir: Path,
    target: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_path = repeat_dir / "strict_execution_evidence.json"
    if not evidence_path.is_file():
        raise ValueError(f"strict execution evidence is missing: {repeat_dir}")
    evidence = _load_object(evidence_path)
    if evidence.get("schema_version") != STRICT_EXECUTION_SCHEMA_VERSION:
        raise ValueError(f"strict execution evidence schema mismatch: {repeat_dir}")

    hostname = str(evidence.get("hostname") or "").strip()
    startup_id = str(evidence.get("startup_instance_id") or "").strip()
    container_id = str(evidence.get("container_id") or "").strip()
    service_port = evidence.get("service_port")
    if not hostname or not startup_id:
        raise ValueError(f"strict execution identity is missing: {repeat_dir}")
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise ValueError(f"strict execution container ID is invalid: {repeat_dir}")
    if not isinstance(service_port, int) or not 0 < service_port < 65536:
        raise ValueError(f"strict execution service port is invalid: {repeat_dir}")

    expected_digest = str(
        (target.get("baseline_runtime") or {}).get("runtime_image_digest") or ""
    )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest):
        raise ValueError("official target is missing an immutable runtime image digest")
    if evidence.get("runtime_image_digest") != expected_digest:
        raise ValueError(f"strict execution runtime image mismatch: {repeat_dir}")

    immutable_record = evidence.get("immutable_inputs")
    immutable_record = immutable_record if isinstance(immutable_record, Mapping) else {}
    immutable_path = _verify_hashed_evidence_file(
        repeat_dir, immutable_record, field="immutable_inputs"
    )
    immutable_inputs = _load_object(immutable_path)
    if immutable_inputs.get("schema_version") != IMMUTABLE_INPUT_SCHEMA_VERSION:
        raise ValueError(f"immutable input schema mismatch: {repeat_dir}")
    expected_model = target.get("model") or {}
    expected_model_id = str(expected_model.get("id") or "")
    expected_model_revision = str(expected_model.get("revision") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_model_revision):
        raise ValueError("official target is missing an immutable model revision")
    if immutable_inputs.get("model_id") != expected_model_id:
        raise ValueError(f"immutable input model mismatch: {repeat_dir}")
    if immutable_inputs.get("model_revision") != expected_model_revision:
        raise ValueError(f"immutable input model revision mismatch: {repeat_dir}")
    expected_data_identity = (target.get("workload") or {}).get("data_identity")
    if not isinstance(expected_data_identity, Mapping) or not expected_data_identity:
        raise ValueError("official target is missing an immutable data identity")
    if immutable_inputs.get("data_identity") != expected_data_identity:
        raise ValueError(f"immutable input data identity mismatch: {repeat_dir}")
    workload_name = str((target.get("workload") or {}).get("name") or "")
    data_kind = str(expected_data_identity.get("kind") or "")
    if data_kind == "release-asset":
        expected_input_kind = "production-trace-prompt-token-ids"
    elif "latency" in workload_name:
        expected_input_kind = "latency-prompt-token-ids"
    elif "throughput" in workload_name:
        expected_input_kind = "throughput-sample-requests"
    else:
        expected_input_kind = "serve-sample-requests"
    if immutable_inputs.get("resolved_input_kind") != expected_input_kind:
        raise ValueError(f"resolved input kind mismatch: {repeat_dir}")
    validate_attestation_payload(
        immutable_inputs,
        {
            "model_id": expected_model_id,
            "model_revision": expected_model_revision,
            "data_identity": expected_data_identity,
            "resolved_input_kind": expected_input_kind,
        },
    )
    resolved_input_sha256 = str(immutable_inputs.get("resolved_input_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", resolved_input_sha256):
        raise ValueError(f"resolved input hash is invalid: {repeat_dir}")

    chip_count = int((target.get("hardware") or {}).get("chip_count") or 0)
    lease = evidence.get("lease")
    lease = lease if isinstance(lease, Mapping) else {}
    devices = lease.get("physical_npu_ids")
    if (
        not isinstance(devices, list)
        or len(devices) != chip_count
        or any(not isinstance(device, int) or device < 0 for device in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError(
            f"strict execution lease device scope is invalid: {repeat_dir}"
        )
    acquired_at = _parse_timestamp(lease.get("acquired_at"), field="lease.acquired_at")
    released_at = _parse_timestamp(lease.get("released_at"), field="lease.released_at")
    if released_at <= acquired_at:
        raise ValueError(f"strict execution lease interval is invalid: {repeat_dir}")

    snapshots = evidence.get("pre_start_snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 2:
        raise ValueError(
            f"strict execution requires two pre-start snapshots: {repeat_dir}"
        )
    snapshot_times: list[datetime] = []
    snapshot_summaries: list[dict[str, Any]] = []
    for index, snapshot in enumerate(snapshots, start=1):
        snapshot = snapshot if isinstance(snapshot, Mapping) else {}
        captured_at = _parse_timestamp(
            snapshot.get("captured_at"),
            field=f"pre_start_snapshots[{index}].captured_at",
        )
        if snapshot.get("physical_npu_ids") != devices:
            raise ValueError(f"pre-start snapshot device scope mismatch: {repeat_dir}")
        for field in (
            "external_compute_pids",
            "external_container_ids",
            "lease_conflicts",
        ):
            if snapshot.get(field) != []:
                raise ValueError(f"pre-start snapshot reports {field}: {repeat_dir}")
        if snapshot.get("stable") is not True:
            raise ValueError(f"pre-start snapshot is not stable: {repeat_dir}")
        npu_path = _verify_hashed_evidence_file(
            repeat_dir,
            snapshot.get("npu_smi") or {},
            field=f"pre_start_snapshots[{index}].npu_smi",
        )
        inspect_path = _verify_hashed_evidence_file(
            repeat_dir,
            snapshot.get("container_inspect") or {},
            field=f"pre_start_snapshots[{index}].container_inspect",
        )
        if not npu_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"pre-start npu-smi snapshot is empty: {repeat_dir}")
        if not inspect_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"pre-start container snapshot is empty: {repeat_dir}")
        snapshot_times.append(captured_at)
        snapshot_summaries.append(
            {
                "captured_at": snapshot.get("captured_at"),
                "npu_smi_sha256": _sha256(npu_path),
                "container_inspect_sha256": _sha256(inspect_path),
            }
        )
    if (snapshot_times[1] - snapshot_times[0]).total_seconds() < 15:
        raise ValueError(
            f"pre-start snapshots are less than 15 seconds apart: {repeat_dir}"
        )
    if snapshot_times[0] < acquired_at or snapshot_times[1] >= released_at:
        raise ValueError(f"pre-start snapshots fall outside the lease: {repeat_dir}")

    ownership = evidence.get("ownership")
    if not isinstance(ownership, list) or len(ownership) != chip_count:
        raise ValueError(f"strict execution PID ownership is incomplete: {repeat_dir}")
    host_pids: list[int] = []
    owned_devices: list[int] = []
    for record in ownership:
        record = record if isinstance(record, Mapping) else {}
        host_pid = record.get("host_pid")
        physical_npu_id = record.get("physical_npu_id")
        cgroup = str(record.get("cgroup") or "")
        if not isinstance(host_pid, int) or host_pid <= 0:
            raise ValueError(f"strict execution host PID is invalid: {repeat_dir}")
        if record.get("container_id") != container_id or container_id not in cgroup:
            raise ValueError(
                f"strict execution cgroup ownership mismatch: {repeat_dir}"
            )
        if physical_npu_id not in devices:
            raise ValueError(
                f"strict execution physical NPU mapping mismatch: {repeat_dir}"
            )
        host_pids.append(host_pid)
        owned_devices.append(physical_npu_id)
    if len(set(host_pids)) != chip_count or sorted(owned_devices) != sorted(devices):
        raise ValueError(
            f"strict execution PID/device mapping is not one-to-one: {repeat_dir}"
        )

    owned_processes_record = evidence.get("owned_processes")
    owned_processes_record = (
        owned_processes_record if isinstance(owned_processes_record, Mapping) else {}
    )
    if owned_processes_record.get("selection_rule") != CANONICAL_WORKER_RULE:
        raise ValueError(f"canonical worker rule mismatch: {repeat_dir}")
    owned_processes_path = _verify_hashed_evidence_file(
        repeat_dir,
        owned_processes_record.get("raw") or {},
        field="owned_processes.raw",
    )
    owned_processes_payload = json.loads(
        owned_processes_path.read_text(encoding="utf-8")
    )
    if not isinstance(owned_processes_payload, list) or not owned_processes_payload:
        raise ValueError(f"owned process evidence is empty: {repeat_dir}")
    all_owned_host_pids: list[int] = []
    processes_by_device: dict[int, list[dict[str, Any]]] = {
        device: [] for device in devices
    }
    for value in owned_processes_payload:
        record = value if isinstance(value, dict) else {}
        host_pid = record.get("host_pid")
        physical_npu_id = record.get("physical_npu_id")
        cgroup = str(record.get("cgroup") or "")
        cmdline = record.get("cmdline")
        if (
            not isinstance(host_pid, int)
            or host_pid <= 0
            or physical_npu_id not in devices
            or record.get("container_id") != container_id
            or container_id not in cgroup
            or not isinstance(cmdline, str)
            or not cmdline
        ):
            raise ValueError(f"owned process identity is invalid: {repeat_dir}")
        processes_by_device[physical_npu_id].append(record)
        all_owned_host_pids.append(host_pid)
    canonical_records: list[dict[str, Any]] = []
    for device in devices:
        candidates = processes_by_device[device]
        if not candidates:
            raise ValueError(f"owned process device scope is incomplete: {repeat_dir}")
        canonical_records.append(min(candidates, key=canonical_worker_key))
    if len(all_owned_host_pids) != len(set(all_owned_host_pids)):
        raise ValueError(f"owned process PIDs are not unique: {repeat_dir}")
    canonical_pairs = [
        (record["host_pid"], record["physical_npu_id"]) for record in canonical_records
    ]
    ownership_pairs = [
        (record["host_pid"], record["physical_npu_id"]) for record in ownership
    ]
    if ownership_pairs != canonical_pairs:
        raise ValueError(f"canonical worker selection mismatch: {repeat_dir}")

    hbm_record = evidence.get("hbm_samples")
    hbm_record = hbm_record if isinstance(hbm_record, Mapping) else {}
    hbm_path = _verify_hashed_evidence_file(repeat_dir, hbm_record, field="hbm_samples")
    samples: list[dict[str, Any]] = []
    with hbm_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sample = json.loads(line)
                if not isinstance(sample, dict):
                    raise ValueError(f"HBM sample is not an object: {repeat_dir}")
                samples.append(sample)
    if not samples:
        raise ValueError(f"HBM samples are empty: {repeat_dir}")
    measured_peak = 0.0
    for sample in samples:
        _parse_timestamp(sample.get("captured_at"), field="hbm_samples.captured_at")
        if sample.get("host_pids") != host_pids:
            raise ValueError(f"HBM sample PID scope mismatch: {repeat_dir}")
        per_device = sample.get("physical_npu_hbm_mb")
        per_device = per_device if isinstance(per_device, Mapping) else {}
        if set(per_device) != {str(device) for device in devices}:
            raise ValueError(f"HBM sample device scope mismatch: {repeat_dir}")
        values = list(per_device.values())
        if any(not isinstance(value, (int, float)) or value < 0 for value in values):
            raise ValueError(f"HBM sample value is invalid: {repeat_dir}")
        total = sample.get("total_hbm_mb")
        if not isinstance(total, (int, float)) or total != sum(values):
            raise ValueError(f"HBM sample total mismatch: {repeat_dir}")
        measured_peak = max(measured_peak, float(total))
    declared_peak = evidence.get("peak_hbm_mb")
    metric_peak = metrics.get("peak_mem_mb")
    if (
        measured_peak <= 0
        or declared_peak != measured_peak
        or metric_peak != measured_peak
    ):
        raise ValueError(f"strict execution peak HBM mismatch: {repeat_dir}")

    cleanup_record = evidence.get("cleanup")
    cleanup_record = cleanup_record if isinstance(cleanup_record, Mapping) else {}
    cleanup_path = _verify_hashed_evidence_file(
        repeat_dir, cleanup_record, field="cleanup"
    )
    cleanup = _load_object(cleanup_path)
    if cleanup.get("schema_version") != CLEANUP_CHAIN_SCHEMA_VERSION:
        raise ValueError(f"cleanup-chain schema mismatch: {repeat_dir}")
    expected_cleanup = {
        "hostname": hostname,
        "startup_instance_id": startup_id,
        "container_id": container_id,
        "exit_code": 0,
        "host_pids": host_pids,
        "all_owned_host_pids": all_owned_host_pids,
        "physical_npu_ids": devices,
        "service_port": service_port,
        "container_stopped_or_removed": True,
        "pids_absent": True,
        "port_released": True,
        "npu_processes_absent": True,
        "lease_released": True,
    }
    for field, expected in expected_cleanup.items():
        if cleanup.get(field) != expected:
            raise ValueError(f"cleanup-chain mismatch for {field}: {repeat_dir}")
    finished_at = _parse_timestamp(
        cleanup.get("finished_at"), field="cleanup.finished_at"
    )
    if not acquired_at < finished_at <= released_at:
        raise ValueError(f"cleanup-chain timestamp is outside the lease: {repeat_dir}")

    return {
        "strict_execution_evidence_sha256": _sha256(evidence_path),
        "startup_instance_id": startup_id,
        "hostname": hostname,
        "container_id": container_id,
        "runtime_image_digest": expected_digest,
        "immutable_input_attestation_sha256": _sha256(immutable_path),
        "model_revision": expected_model_revision,
        "data_identity_kind": expected_data_identity.get("kind"),
        "resolved_input_sha256": resolved_input_sha256,
        "service_port": service_port,
        "physical_npu_ids": devices,
        "host_pids": host_pids,
        "pre_start_snapshots": snapshot_summaries,
        "peak_hbm_mb": measured_peak,
        "hbm_samples_sha256": _sha256(hbm_path),
        "owned_processes_sha256": _sha256(owned_processes_path),
        "cleanup_chain_sha256": _sha256(cleanup_path),
    }


def _target_for_entry(
    repo_root: Path, entry: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    registry_path = repo_root / "leaderboard-data" / "official-targets.json"
    checksum_path = repo_root / "leaderboard-data" / "official-targets.sha256"
    registry_sha256 = _sha256(registry_path)
    declared_sha256 = checksum_path.read_text(encoding="utf-8").split()[0]
    if registry_sha256 != declared_sha256:
        raise ValueError("official target registry checksum mismatch")

    target_id = str((entry.get("same_spec") or {}).get("spec_id") or "")
    registry = _load_object(registry_path)
    for target in registry.get("targets", []):
        if isinstance(target, dict) and target.get("target_id") == target_id:
            if target.get("status") != "active":
                raise ValueError(f"target is not active: {target_id}")
            if target.get("intended_use") != "public-leaderboard":
                raise ValueError(f"target is not public-leaderboard: {target_id}")
            return target, registry_sha256
    raise ValueError(f"target not found in registry: {target_id}")


def _validate_exact_target(
    repo_root: Path, entry: Mapping[str, Any], target: Mapping[str, Any]
) -> None:
    same_spec = entry.get("same_spec") or {}
    workload = str((target.get("workload") or {}).get("name") or "")
    mismatches = _identity_mismatches(entry, target)
    mismatches.extend(
        _parameter_mismatches(
            target.get("server_parameters") or {},
            same_spec.get("resolved_server_parameters") or {},
            prefix="server_parameters",
            workload=workload,
        )
    )
    mismatches.extend(
        _parameter_mismatches(
            (target.get("workload") or {}).get("client_parameters") or {},
            same_spec.get("resolved_client_parameters") or {},
            prefix="client_parameters",
            workload=workload,
        )
    )
    source = target.get("source_spec") or {}
    source_path = repo_root / str(source.get("path") or "")
    if not source_path.is_file() or _sha256(source_path) != source.get("sha256"):
        mismatches.append({"field": "source_spec.sha256", "kind": "mismatch"})
    if mismatches:
        raise ValueError(f"exact target mismatch: {json.dumps(mismatches)}")


def attest_completed_baseline(
    repo_root: Path,
    staged_submission_dir: Path,
    result_spec_dir: Path,
    output_dir: Path,
    *,
    verified_by: str,
    verified_at: str | None = None,
    minimum_repeats: int = 3,
) -> dict[str, Any]:
    artifact_path = staged_submission_dir / "run_leaderboard.json"
    manifest_path = staged_submission_dir / "leaderboard_manifest.json"
    entry = _load_object(artifact_path)
    manifest = _load_object(manifest_path)
    target, registry_sha256 = _target_for_entry(repo_root, entry)
    _validate_exact_target(repo_root, entry, target)

    repeat_records: list[dict[str, Any]] = []
    staged_sha256 = _sha256(artifact_path)
    selected_repeat: str | None = None
    expected_core = (
        ((entry.get("metadata") or {}).get("runtime_provenance") or {})
        .get("engine", {})
        .get("commit")
    )
    expected_plugin = (target.get("baseline_runtime") or {}).get("git_commit")
    trace_profile = target.get("profile") == TRACE_PROFILE
    if trace_profile:
        expected_core = (target.get("baseline_runtime") or {}).get("core_commit")
        expected_plugin = (target.get("baseline_runtime") or {}).get("backend_commit")
        if not expected_core or not expected_plugin:
            raise ValueError("production-trace target is missing exact source commits")
    unique_raw_hashes: set[str] = set()
    unique_artifact_hashes: set[str] = set()
    unique_entry_ids: set[str] = set()
    unique_startup_ids: set[str] = set()
    strict_startup_ids: set[str] = set()
    unique_run_ids: set[str] = set()
    trace_signatures: set[str] = set()
    model_artifact_digests: set[str] = set()
    deterministic_input_hashes: set[str] = set()

    for repeat_dir in sorted(result_spec_dir.glob("repeat-*")):
        raw_path = repeat_dir / "raw_benchmark_result.json"
        repeat_artifact_path = repeat_dir / "submission" / "run_leaderboard.json"
        runner_log_path = repeat_dir / "runner.log"
        if not raw_path.is_file() or not repeat_artifact_path.is_file():
            continue
        raw = _load_object(raw_path)
        repeat_entry = _load_object(repeat_artifact_path)
        _validate_exact_target(repo_root, repeat_entry, target)
        metrics = repeat_entry.get("metrics") or {}
        failed = int(raw.get("failed") or 0)
        error_rate = float(metrics.get("error_rate") or 0)
        if failed != 0 or error_rate != 0:
            raise ValueError(f"repeat has failures: {repeat_dir}")
        strict_evidence = _validate_strict_execution_evidence(
            repeat_dir, target, metrics
        )
        strict_startup_id = str(strict_evidence["startup_instance_id"])
        if strict_startup_id in strict_startup_ids:
            raise ValueError(f"duplicate strict startup identity: {repeat_dir}")
        strict_startup_ids.add(strict_startup_id)
        if strict_evidence["data_identity_kind"] != "nondeterministic-vllm-generator":
            deterministic_input_hashes.add(strict_evidence["resolved_input_sha256"])
        provenance = (repeat_entry.get("metadata") or {}).get(
            "runtime_provenance"
        ) or {}
        if (provenance.get("engine") or {}).get("commit") != expected_core:
            raise ValueError(f"core provenance mismatch: {repeat_dir}")
        if (provenance.get("plugin") or {}).get("commit") != expected_plugin:
            raise ValueError(f"plugin provenance mismatch: {repeat_dir}")
        artifact_sha256 = _sha256(repeat_artifact_path)
        raw_sha256 = _sha256(raw_path)
        trace_evidence: dict[str, Any] = {}
        if trace_profile:
            server_log_path = repeat_dir / "server.stdout.log"
            detail_path = repeat_dir / "trace_replay_results.jsonl"
            plan_path = repeat_dir / "trace_replay_plan.json"
            startup_path = repeat_dir / "startup_evidence.json"
            model_provenance_path = repeat_dir / "model_artifact_provenance.json"
            runtime_provenance_path = repeat_dir / "runtime_package_provenance.json"
            required_paths = (
                runner_log_path,
                server_log_path,
                detail_path,
                plan_path,
                startup_path,
                model_provenance_path,
                runtime_provenance_path,
            )
            missing = [str(path) for path in required_paths if not path.is_file()]
            if missing:
                raise ValueError(
                    f"production-trace evidence is missing: {', '.join(missing)}"
                )
            max_requests = int(
                ((target.get("workload") or {}).get("client_parameters") or {}).get(
                    "max_requests"
                )
                or 0
            )
            if int(raw.get("completed") or 0) != max_requests:
                raise ValueError(f"production-trace repeat is incomplete: {repeat_dir}")
            plan = _load_object(plan_path)
            startup = _load_object(startup_path)
            model_provenance = _load_object(model_provenance_path)
            runtime_provenance = _load_object(runtime_provenance_path)
            detail_metadata, detail_count = _load_trace_detail_metadata(detail_path)
            if detail_count != max_requests:
                raise ValueError(
                    f"production-trace detail count mismatch: {repeat_dir}"
                )
            signature = str(plan.get("cohort_setting_signature") or "")
            if not signature or signature != raw.get("cohort_setting_signature"):
                raise ValueError(f"trace cohort signature mismatch: {repeat_dir}")
            if signature != (detail_metadata.get("plan") or {}).get(
                "cohort_setting_signature"
            ):
                raise ValueError(f"trace detail signature mismatch: {repeat_dir}")
            if signature != startup.get("cohort_setting_signature"):
                raise ValueError(f"trace startup signature mismatch: {repeat_dir}")
            model_digest = str(model_provenance.get("model_artifact_digest") or "")
            if not model_digest or model_digest != startup.get("model_artifact_digest"):
                raise ValueError(f"model artifact digest mismatch: {repeat_dir}")
            if startup.get("engine_source_commit") != expected_core:
                raise ValueError(f"startup core commit mismatch: {repeat_dir}")
            if startup.get("plugin_source_commit") != expected_plugin:
                raise ValueError(f"startup plugin commit mismatch: {repeat_dir}")
            expected_runtime = (target.get("baseline_runtime") or {}).get(
                "runtime_packages"
            )
            if runtime_provenance.get("runtime_packages") != expected_runtime:
                raise ValueError(f"runtime package provenance mismatch: {repeat_dir}")
            if startup.get("runtime_packages") != expected_runtime:
                raise ValueError(f"startup runtime packages mismatch: {repeat_dir}")
            expected_environment = (target.get("baseline_runtime") or {}).get(
                "runtime_environment"
            )
            if runtime_provenance.get("runtime_environment") != expected_environment:
                raise ValueError(
                    f"runtime environment provenance mismatch: {repeat_dir}"
                )
            if startup.get("runtime_environment") != expected_environment:
                raise ValueError(f"startup runtime environment mismatch: {repeat_dir}")
            for field in (
                "runtime_image",
                "runtime_image_digest",
            ):
                expected_value = (target.get("baseline_runtime") or {}).get(field)
                if runtime_provenance.get(field) != expected_value:
                    raise ValueError(f"runtime image provenance mismatch: {repeat_dir}")
                if startup.get(field) != expected_value:
                    raise ValueError(f"startup runtime image mismatch: {repeat_dir}")
            if not startup.get("finished_at"):
                raise ValueError(f"startup evidence is not finalized: {repeat_dir}")
            result_hashes = startup.get("result_hashes") or {}
            if result_hashes.get("raw_sha256") != raw_sha256:
                raise ValueError(f"startup raw result hash mismatch: {repeat_dir}")
            if result_hashes.get("detail_sha256") != _sha256(detail_path):
                raise ValueError(f"startup trace detail hash mismatch: {repeat_dir}")
            startup_id = str(startup.get("startup_instance_id") or "")
            run_id = str(startup.get("run_id") or "")
            entry_id = str(repeat_entry.get("entry_id") or "")
            if not startup_id or not run_id or not entry_id:
                raise ValueError(f"repeat identity evidence is missing: {repeat_dir}")
            if startup_id != strict_startup_id:
                raise ValueError(f"startup evidence identity mismatch: {repeat_dir}")
            if startup_id in unique_startup_ids or run_id in unique_run_ids:
                raise ValueError(f"duplicate startup identity: {repeat_dir}")
            if entry_id in unique_entry_ids:
                raise ValueError(f"duplicate leaderboard entry identity: {repeat_dir}")
            if raw_sha256 in unique_raw_hashes:
                raise ValueError(f"duplicate raw result evidence: {repeat_dir}")
            if artifact_sha256 in unique_artifact_hashes:
                raise ValueError(f"duplicate leaderboard artifact: {repeat_dir}")
            unique_startup_ids.add(startup_id)
            unique_run_ids.add(run_id)
            unique_entry_ids.add(entry_id)
            unique_raw_hashes.add(raw_sha256)
            unique_artifact_hashes.add(artifact_sha256)
            trace_signatures.add(signature)
            model_artifact_digests.add(model_digest)
            trace_evidence = {
                "server_log_sha256": _sha256(server_log_path),
                "trace_detail_sha256": _sha256(detail_path),
                "trace_plan_sha256": _sha256(plan_path),
                "startup_evidence_sha256": _sha256(startup_path),
                "model_artifact_provenance_sha256": _sha256(model_provenance_path),
                "runtime_package_provenance_sha256": _sha256(runtime_provenance_path),
                "cohort_setting_signature": signature,
                "model_artifact_digest": model_digest,
                "startup_instance_id": startup_id,
                "run_id": run_id,
            }
        if artifact_sha256 == staged_sha256 and selected_repeat is None:
            selected_repeat = repeat_dir.name
        repeat_records.append(
            {
                "repeat": repeat_dir.name,
                "raw_result_sha256": raw_sha256,
                "leaderboard_artifact_sha256": artifact_sha256,
                "runner_log_sha256": _sha256(runner_log_path)
                if runner_log_path.is_file()
                else None,
                "metrics": metrics,
                "failed_requests": failed,
                **trace_evidence,
                **strict_evidence,
            }
        )

    if len(repeat_records) < minimum_repeats:
        raise ValueError(
            f"insufficient successful repeats: {len(repeat_records)} < {minimum_repeats}"
        )
    if trace_profile and len(trace_signatures) != 1:
        raise ValueError("production-trace repeats use different cohort signatures")
    if trace_profile and len(model_artifact_digests) != 1:
        raise ValueError("production-trace repeats use different model artifacts")
    target_data_kind = str(
        (((target.get("workload") or {}).get("data_identity") or {}).get("kind")) or ""
    )
    if (
        target_data_kind != "nondeterministic-vllm-generator"
        and len(deterministic_input_hashes) != 1
    ):
        raise ValueError("deterministic repeats use different resolved inputs")
    if selected_repeat is None:
        raise ValueError("staged artifact does not match any successful repeat")

    timestamp = verified_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    repeat_suite = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "target_id": target["target_id"],
        "target_version": target["target_version"],
        "target_registry_sha256": registry_sha256,
        "verified_at": timestamp,
        "verified_by": verified_by,
        "minimum_repeats": minimum_repeats,
        "successful_repeats": len(repeat_records),
        "selected_repeat": selected_repeat,
        "exact_target_match": True,
        "zero_failed_requests": True,
        "repeats": repeat_records,
    }

    attested = json.loads(json.dumps(entry))
    metadata = attested.setdefault("metadata", {})
    metadata.update(
        {
            "verified": True,
            "verified_at": timestamp,
            "verified_by": verified_by,
            "target_id": target["target_id"],
            "target_version": target["target_version"],
            "profile_id": target["profile"],
            "target_registry_sha256": registry_sha256,
            "verification_attestation": {
                "schema_version": ATTESTATION_SCHEMA_VERSION,
                "evidence": "repeat_suite.json",
                "successful_repeats": len(repeat_records),
                "selected_repeat": selected_repeat,
            },
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_leaderboard.json").write_text(
        json.dumps(attested, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "leaderboard_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "repeat_suite.json").write_text(
        json.dumps(repeat_suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return attested
