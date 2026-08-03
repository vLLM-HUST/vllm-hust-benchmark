from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from vllm_hust_benchmark.official_targets import PACKAGE_REGISTRY_PATH
from vllm_hust_benchmark.registry import get_scenario

OFFICIAL_BASELINE_SUBMITTER = "official-ascend-baseline"

PRIMARY_METRIC_BY_BENCHMARK_TYPE = {
    "serve": "ttft_ms",
    "latency": "ttft_ms",
    "throughput": "throughput_tps",
}


def load_official_baseline_spec(spec_file: Path) -> dict[str, Any]:
    payload = json.loads(spec_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{spec_file}: official baseline spec must be a JSON object")
    return payload


def get_official_baseline_spec_id(spec: Mapping[str, Any]) -> str:
    spec_id = str(spec.get("id") or "").strip()
    if not spec_id:
        raise ValueError("official baseline spec is missing required field: id")
    return spec_id


def get_canonical_submission_dir(
    spec: Mapping[str, Any], *, submissions_root: Path
) -> Path:
    return submissions_root / get_official_baseline_spec_id(spec)


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def has_canonical_run(spec: Mapping[str, Any], *, submissions_root: Path) -> bool:
    canonical_dir = get_canonical_submission_dir(
        spec, submissions_root=submissions_root
    )
    run_file = canonical_dir / "run_leaderboard.json"
    manifest_file = canonical_dir / "leaderboard_manifest.json"

    run_payload = _load_json_object(run_file)
    manifest_payload = _load_json_object(manifest_file)
    if run_payload is None or manifest_payload is None:
        return False

    same_spec = run_payload.get("same_spec")
    same_spec = same_spec if isinstance(same_spec, Mapping) else {}
    metadata = run_payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    entries = manifest_payload.get("entries")
    entries = entries if isinstance(entries, list) else []

    spec_id = get_official_baseline_spec_id(spec)
    if str(same_spec.get("spec_id") or "").strip() != spec_id:
        return False
    if str(metadata.get("submitter") or "").strip() != OFFICIAL_BASELINE_SUBMITTER:
        return False
    if not any(
        isinstance(entry, Mapping)
        and str(entry.get("leaderboard_artifact") or "").strip()
        == "run_leaderboard.json"
        for entry in entries
    ):
        return False
    return True


def get_official_baseline_benchmark_type(spec: Mapping[str, Any]) -> str:
    scenario_name = str(spec.get("scenario") or "").strip()
    if not scenario_name:
        raise ValueError("official baseline spec is missing required field: scenario")
    return get_scenario(scenario_name).benchmark_type


def get_primary_metric_name_for_benchmark_type(benchmark_type: str) -> str:
    try:
        return PRIMARY_METRIC_BY_BENCHMARK_TYPE[benchmark_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported official baseline benchmark type: {benchmark_type}"
        ) from exc


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_result_artifact_payload(result_dir: Path) -> dict[str, Any] | None:
    return _load_json_object(result_dir / "submission" / "run_leaderboard.json")


def select_canonical_candidate(
    result_dirs: list[Path], *, benchmark_type: str
) -> dict[str, Any]:
    primary_metric_name = get_primary_metric_name_for_benchmark_type(benchmark_type)

    candidates: list[dict[str, Any]] = []
    for index, result_dir in enumerate(result_dirs):
        payload = _load_result_artifact_payload(result_dir)
        if payload is None:
            continue

        metrics = payload.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        primary_metric_value = _safe_float(metrics.get(primary_metric_name))
        if primary_metric_value is None:
            continue

        error_rate = _safe_float(metrics.get("error_rate"))
        candidates.append(
            {
                "result_dir": str(result_dir.resolve()),
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
                "error_rate": float(error_rate or 0.0),
                "index": index,
            }
        )

    if not candidates:
        raise ValueError(
            "no valid repeated runs available for canonical candidate selection"
        )

    metric_median = median(item["primary_metric_value"] for item in candidates)
    for candidate in candidates:
        candidate["distance_to_median"] = abs(
            candidate["primary_metric_value"] - metric_median
        )

    selected = min(
        candidates,
        key=lambda item: (
            item["error_rate"],
            item["distance_to_median"],
            item["index"],
        ),
    )

    return {
        "benchmark_type": benchmark_type,
        "primary_metric_name": primary_metric_name,
        "median_value": float(metric_median),
        "selected_result_dir": selected["result_dir"],
        "candidates": candidates,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_target_parameters(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> None:
    aliases = aliases or {}
    for key, expected_value in expected.items():
        if (
            key == "input_len"
            and isinstance(actual.get("prefix_repetition_prefix_len"), int)
            and isinstance(actual.get("prefix_repetition_suffix_len"), int)
        ):
            actual_value = (
                actual["prefix_repetition_prefix_len"]
                + actual["prefix_repetition_suffix_len"]
            )
            if actual_value != expected_value:
                raise ValueError(
                    f"resolved target parameter mismatch: {key}="
                    f"{actual_value!r}, expected {expected_value!r}"
                )
            continue
        actual_keys = (key, *aliases.get(key, ()))
        actual_key = next((name for name in actual_keys if name in actual), None)
        if actual_key is None:
            raise ValueError(f"resolved target parameter is missing: {key}")
        if actual[actual_key] != expected_value:
            raise ValueError(
                f"resolved target parameter mismatch: {key}="
                f"{actual[actual_key]!r}, expected {expected_value!r}"
            )


def attest_canonical_submission(
    canonical_dir: Path,
    *,
    spec: Mapping[str, Any],
    result_dirs: list[Path],
    selected_result_dir: Path,
    primary_metric_name: str,
    registry_path: Path = PACKAGE_REGISTRY_PATH,
) -> dict[str, Any]:
    """Bind a promoted candidate to its target and repeated-run evidence.

    The matrix starts a fresh runner process for every result directory.  This
    function fails closed unless at least three distinct, zero-error artifacts
    agree on target, resolved configuration, runtime provenance, and required
    environment evidence.
    """

    if len(result_dirs) < 3:
        raise ValueError("canonical verification requires at least three repeats")

    spec_id = get_official_baseline_spec_id(spec)
    registry = _load_json_object(registry_path)
    targets = registry.get("targets") if registry else None
    if not isinstance(targets, list):
        raise ValueError(f"official target registry is invalid: {registry_path}")
    target = next(
        (
            item
            for item in targets
            if isinstance(item, Mapping) and item.get("target_id") == spec_id
        ),
        None,
    )
    if target is None or target.get("status") != "active":
        raise ValueError(f"canonical verification requires an active target: {spec_id}")
    target_version = str(target.get("target_version") or "").strip()
    if not target_version:
        raise ValueError(f"active target is missing target_version: {spec_id}")

    selected_resolved = selected_result_dir.resolve()
    resolved_result_dirs = [path.resolve() for path in result_dirs]
    if selected_resolved not in resolved_result_dirs:
        raise ValueError("selected canonical candidate is not one of the repeats")

    repeat_evidence: list[dict[str, Any]] = []
    identities: set[str] = set()
    resolved_hashes: set[str] = set()
    provenance_pairs: set[tuple[str, str]] = set()
    for repeat_index, result_dir in enumerate(resolved_result_dirs, start=1):
        artifact_path = result_dir / "submission" / "run_leaderboard.json"
        raw_path = result_dir / "raw_benchmark_result.json"
        payload = _load_json_object(artifact_path)
        raw_payload = _load_json_object(raw_path)
        if payload is None or raw_payload is None:
            raise ValueError(f"repeat evidence is incomplete: {result_dir}")

        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        same_spec = payload.get("same_spec")
        same_spec = same_spec if isinstance(same_spec, Mapping) else {}
        metrics = payload.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        environment = payload.get("environment")
        environment = environment if isinstance(environment, Mapping) else {}
        provenance = metadata.get("runtime_provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        engine_provenance = provenance.get("engine")
        engine_provenance = (
            engine_provenance if isinstance(engine_provenance, Mapping) else {}
        )
        plugin_provenance = provenance.get("plugin")
        plugin_provenance = (
            plugin_provenance if isinstance(plugin_provenance, Mapping) else {}
        )

        if same_spec.get("spec_id") != spec_id:
            raise ValueError(f"repeat target mismatch: {result_dir}")
        resolved_server = same_spec.get("resolved_server_parameters")
        resolved_server = (
            resolved_server if isinstance(resolved_server, Mapping) else {}
        )
        resolved_client = same_spec.get("resolved_client_parameters")
        resolved_client = (
            resolved_client if isinstance(resolved_client, Mapping) else {}
        )
        target_server = target.get("server_parameters")
        target_server = target_server if isinstance(target_server, Mapping) else {}
        target_workload = target.get("workload")
        target_workload = target_workload if isinstance(target_workload, Mapping) else {}
        target_client = target_workload.get("client_parameters")
        target_client = target_client if isinstance(target_client, Mapping) else {}
        _assert_target_parameters(target_server, resolved_server)
        _assert_target_parameters(
            target_client,
            resolved_client,
            aliases={
                "input_len": ("random_input_len",),
                "output_len": (
                    "random_output_len",
                    "prefix_repetition_output_len",
                ),
            },
        )
        resolved_hash = str(same_spec.get("resolved_spec_hash") or "").strip()
        if not resolved_hash:
            raise ValueError(f"repeat is missing resolved spec hash: {result_dir}")
        resolved_hashes.add(resolved_hash)
        if metrics.get("error_rate") not in (0, 0.0):
            raise ValueError(f"repeat has nonzero or missing error rate: {result_dir}")
        peak_mem = metrics.get("peak_mem_mb")
        if not isinstance(peak_mem, (int, float)) or peak_mem <= 0:
            raise ValueError(f"repeat is missing measured peak memory: {result_dir}")
        if raw_payload.get("failed", 0) not in (0, 0.0):
            raise ValueError(f"repeat raw result contains failures: {result_dir}")
        if not metadata.get("reproducible_cmd"):
            raise ValueError(f"repeat is missing reproducible command: {result_dir}")
        if metadata.get("workload_config_contract") != "explicit-effective/v1":
            raise ValueError(f"repeat is missing explicit config contract: {result_dir}")
        for field in ("pytorch_version", "cann_version", "driver_version"):
            if not environment.get(field):
                raise ValueError(f"repeat is missing {field}: {result_dir}")

        engine_commit = str(engine_provenance.get("commit") or "").strip()
        plugin_commit = str(plugin_provenance.get("commit") or "").strip()
        if not engine_commit or not plugin_commit:
            raise ValueError(f"repeat is missing runtime provenance: {result_dir}")
        provenance_pairs.add((engine_commit, plugin_commit))

        identity = str(metadata.get("idempotency_key") or "").strip()
        if not identity or identity in identities:
            raise ValueError(f"repeat identity is missing or duplicated: {result_dir}")
        identities.add(identity)
        repeat_evidence.append(
            {
                "repeat_index": repeat_index,
                "idempotency_key": identity,
                "leaderboard_artifact_sha256": _sha256_file(artifact_path),
                "raw_result_sha256": _sha256_file(raw_path),
                "submitted_at": metadata.get("submitted_at"),
            }
        )

    if len(resolved_hashes) != 1:
        raise ValueError("repeats do not share one resolved spec hash")
    if len(provenance_pairs) != 1:
        raise ValueError("repeats do not share one runtime provenance pair")

    canonical_path = canonical_dir / "run_leaderboard.json"
    canonical_payload = _load_json_object(canonical_path)
    if canonical_payload is None:
        raise ValueError(f"canonical artifact is missing: {canonical_path}")
    canonical_metadata = canonical_payload.get("metadata")
    canonical_metadata = (
        dict(canonical_metadata) if isinstance(canonical_metadata, Mapping) else {}
    )
    selected_index = resolved_result_dirs.index(selected_resolved) + 1
    canonical_metadata.update(
        {
            "verified": True,
            "target_id": spec_id,
            "target_version": target_version,
            "verification_attestation": {
                "schema_version": "official-repeat-attestation/v1",
                "successful_repeats": len(repeat_evidence),
                "independent_service_processes": len(repeat_evidence),
                "selection_policy": "median-primary-metric",
                "primary_metric_name": primary_metric_name,
                "selected_repeat_index": selected_index,
                "resolved_spec_hash": next(iter(resolved_hashes)),
                "repeat_evidence": repeat_evidence,
            },
        }
    )
    canonical_payload["metadata"] = canonical_metadata
    canonical_path.write_text(
        json.dumps(canonical_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return canonical_payload
