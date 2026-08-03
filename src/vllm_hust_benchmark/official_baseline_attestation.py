from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from vllm_hust_benchmark.baseline_recovery import _identity_mismatches
from vllm_hust_benchmark.baseline_recovery import _parameter_mismatches


ATTESTATION_SCHEMA_VERSION = "official-baseline-attestation/v1"
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
    unique_run_ids: set[str] = set()
    trace_signatures: set[str] = set()
    model_artifact_digests: set[str] = set()

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
