from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from vllm_hust_benchmark.official_baseline_attestation import _validate_exact_target


SCHEMA_VERSION = "simllm-official-paired-attestation/v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(repo_root: Path, target_id: str) -> tuple[dict[str, Any], str]:
    registry_path = repo_root / "leaderboard-data" / "official-targets.json"
    checksum_path = repo_root / "leaderboard-data" / "official-targets.sha256"
    registry_sha = _sha(registry_path)
    if checksum_path.read_text(encoding="utf-8").split()[0] != registry_sha:
        raise ValueError("official target registry checksum mismatch")
    registry = _load(registry_path)
    for target in registry.get("targets", []):
        if isinstance(target, dict) and target.get("target_id") == target_id:
            if target.get("status") != "active":
                raise ValueError(f"SimLLM target is not active: {target_id}")
            if target.get("profile") != "simllm-warm-cache":
                raise ValueError(f"target is not a SimLLM warm-cache profile: {target_id}")
            return target, registry_sha
    raise ValueError(f"SimLLM target not found: {target_id}")


def _cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else math.inf


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "cv": _cv(values),
        "cv_percent": _cv(values) * 100,
    }


def _normalized_candidate(entry: Mapping[str, Any], baseline_engine: str) -> dict[str, Any]:
    payload = json.loads(json.dumps(entry))
    payload["engine"] = baseline_engine
    return payload


def _validate_runtime(
    evidence: Mapping[str, Any], target: Mapping[str, Any], repeat_dir: Path
) -> None:
    expected = target.get("baseline_runtime") or {}
    runtime = evidence.get("runtime") or {}
    checks = {
        "core_commit": (evidence.get("core_commit"), expected.get("core_commit")),
        "backend_commit": (
            evidence.get("backend_commit"),
            expected.get("backend_commit"),
        ),
        "runtime.image": (runtime.get("image"), expected.get("runtime_image")),
        "runtime.image_digest": (
            runtime.get("image_digest"),
            expected.get("runtime_image_digest"),
        ),
        "runtime.packages": (
            runtime.get("packages"),
            expected.get("runtime_packages"),
        ),
    }
    mismatches = [name for name, (actual, wanted) in checks.items() if actual != wanted]
    if mismatches:
        raise ValueError(f"runtime provenance mismatch {mismatches}: {repeat_dir}")


def attest_simllm_campaign(
    repo_root: Path,
    result_spec_dir: Path,
    output_dir: Path,
    *,
    target_id: str,
    verified_by: str,
    verified_at: str | None = None,
) -> dict[str, Any]:
    target, registry_sha = _target(repo_root, target_id)
    protocol = (target.get("workload") or {}).get("protocol") or {}
    minimum_repeats = int(protocol.get("minimum_independent_repetitions") or 3)
    maximum_cv = float(protocol.get("maximum_primary_metric_cv_percent") or 5) / 100
    baseline_engine = str(protocol.get("baseline_engine") or "")
    candidate_engine = str(protocol.get("candidate_engine") or "")
    expected_requests = int(
        ((target.get("workload") or {}).get("client_parameters") or {}).get(
            "num_prompts"
        )
        or 0
    )
    if not baseline_engine or not candidate_engine or baseline_engine == candidate_engine:
        raise ValueError("SimLLM target has invalid paired engine labels")

    repeats: list[dict[str, Any]] = []
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    pair_signatures: set[tuple[str, str, str]] = set()
    raw_hashes: set[str] = set()

    for repeat_dir in sorted(result_spec_dir.glob("repeat-*")):
        pair_path = repeat_dir / "paired_protocol_evidence.json"
        baseline_dir = repeat_dir / "baseline-disabled"
        candidate_dir = repeat_dir / "enabled-warm-cache"
        baseline_evidence_path = baseline_dir / "arm_evidence.json"
        candidate_evidence_path = candidate_dir / "arm_evidence.json"
        required = (
            pair_path,
            baseline_evidence_path,
            candidate_evidence_path,
            baseline_dir / "device_state_before.txt",
            baseline_dir / "device_state_after.txt",
            candidate_dir / "device_state_before.txt",
            candidate_dir / "device_state_after.txt",
            candidate_dir / "warmup_pass_1.json",
            baseline_dir / "server.stdout.log",
            candidate_dir / "server.stdout.log",
        )
        if not pair_path.exists():
            continue
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"SimLLM repeat evidence is incomplete: {missing}")

        pair = _load(pair_path)
        baseline_evidence = _load(baseline_evidence_path)
        candidate_evidence = _load(candidate_evidence_path)
        if pair.get("schema_version") != "simllm-official-paired-protocol/v1":
            raise ValueError(f"invalid paired protocol evidence: {repeat_dir}")
        if pair.get("spec_id") != target_id:
            raise ValueError(f"paired protocol target mismatch: {repeat_dir}")
        if pair.get("exact_measured_setting_match") is not True:
            raise ValueError(f"paired measured settings do not match: {repeat_dir}")
        if pair.get("zero_failed_requests") is not True:
            raise ValueError(f"paired repeat has failures: {repeat_dir}")
        if baseline_evidence.get("engine") != baseline_engine:
            raise ValueError(f"baseline engine label mismatch: {repeat_dir}")
        if candidate_evidence.get("engine") != candidate_engine:
            raise ValueError(f"candidate engine label mismatch: {repeat_dir}")
        if candidate_evidence.get("rewritten_requests", 0) <= 0:
            raise ValueError(f"candidate has no positive rewrite evidence: {repeat_dir}")
        if candidate_evidence.get("patch_applied") is not True:
            raise ValueError(f"candidate patch activation is missing: {repeat_dir}")
        expected_simllm_config = protocol.get("simllm_config") or {}
        if candidate_evidence.get("simllm_config") != expected_simllm_config:
            raise ValueError(f"candidate SimLLM config mismatch: {repeat_dir}")
        if baseline_evidence.get("simllm_config") != expected_simllm_config:
            raise ValueError(f"baseline SimLLM config provenance mismatch: {repeat_dir}")

        for arm_dir, evidence in (
            (baseline_dir, baseline_evidence),
            (candidate_dir, candidate_evidence),
        ):
            _validate_runtime(evidence, target, repeat_dir)
            raw_path = arm_dir / "raw_benchmark_result.json"
            entry_path = arm_dir / "submission" / "run_leaderboard.json"
            manifest_path = arm_dir / "submission" / "leaderboard_manifest.json"
            cohort_path = arm_dir / "prompt_cohort_evidence.json"
            if not all(path.is_file() for path in (raw_path, entry_path, manifest_path, cohort_path)):
                raise ValueError(f"SimLLM arm output is incomplete: {arm_dir}")
            raw = _load(raw_path)
            if int(raw.get("completed") or 0) != expected_requests:
                raise ValueError(f"SimLLM arm is incomplete: {arm_dir}")
            if int(evidence.get("failed") or 0) != 0:
                raise ValueError(f"SimLLM arm contains failures: {arm_dir}")
            raw_sha = _sha(raw_path)
            if raw_sha in raw_hashes:
                raise ValueError(f"duplicate raw result evidence: {arm_dir}")
            raw_hashes.add(raw_sha)
            if (evidence.get("hashes") or {}).get("raw_result_sha256") != raw_sha:
                raise ValueError(f"raw result hash mismatch: {arm_dir}")
            for state_name in ("device_state_before.txt", "device_state_after.txt"):
                if "No process in device" not in (arm_dir / state_name).read_text(
                    encoding="utf-8"
                ):
                    raise ValueError(f"device state is not clean: {arm_dir / state_name}")

        baseline_entry = _load(baseline_dir / "submission" / "run_leaderboard.json")
        candidate_entry = _load(candidate_dir / "submission" / "run_leaderboard.json")
        _validate_exact_target(repo_root, baseline_entry, target)
        _validate_exact_target(
            repo_root, _normalized_candidate(candidate_entry, baseline_engine), target
        )
        if baseline_entry.get("engine") != baseline_engine:
            raise ValueError(f"baseline artifact engine mismatch: {repeat_dir}")
        if candidate_entry.get("engine") != candidate_engine:
            raise ValueError(f"candidate artifact engine mismatch: {repeat_dir}")

        baseline_tput = float(
            (_load(baseline_dir / "raw_benchmark_result.json")).get(
                "request_throughput"
            )
            or 0
        )
        candidate_tput = float(
            (_load(candidate_dir / "raw_benchmark_result.json")).get(
                "request_throughput"
            )
            or 0
        )
        if baseline_tput <= 0 or candidate_tput <= 0:
            raise ValueError(f"invalid primary metric: {repeat_dir}")
        baseline_values.append(baseline_tput)
        candidate_values.append(candidate_tput)
        signature = (
            str(pair.get("resolved_spec_hash") or ""),
            str(pair.get("prompt_cohort_sha256") or ""),
            json.dumps(candidate_evidence.get("simllm_config"), sort_keys=True),
        )
        if not all(signature):
            raise ValueError(f"paired setting signature is incomplete: {repeat_dir}")
        pair_signatures.add(signature)
        repeats.append(
            {
                "repeat": repeat_dir.name,
                "paired_evidence_sha256": _sha(pair_path),
                "resolved_spec_hash": signature[0],
                "prompt_cohort_sha256": signature[1],
                "baseline_request_throughput": baseline_tput,
                "candidate_request_throughput": candidate_tput,
                "improvement_percent": (
                    (candidate_tput - baseline_tput) / baseline_tput * 100
                ),
                "rewrite_events": int(candidate_evidence.get("rewrite_events") or 0),
                "rewritten_requests": int(
                    candidate_evidence.get("rewritten_requests") or 0
                ),
            }
        )

    if len(repeats) < minimum_repeats:
        raise ValueError(f"insufficient successful repeats: {len(repeats)} < {minimum_repeats}")
    if len(pair_signatures) != 1:
        raise ValueError("SimLLM repeats use different paired setting signatures")
    baseline_stats = _stats(baseline_values)
    candidate_stats = _stats(candidate_values)
    if baseline_stats["cv"] > maximum_cv or candidate_stats["cv"] > maximum_cv:
        raise ValueError(
            "SimLLM primary metric CV exceeds publication gate: "
            f"baseline={baseline_stats['cv_percent']:.3f}% "
            f"candidate={candidate_stats['cv_percent']:.3f}%"
        )

    selected = min(
        range(len(repeats)),
        key=lambda index: abs(
            candidate_values[index] - candidate_stats["median"]
        ),
    )
    selected_repeat = repeats[selected]["repeat"]
    selected_dir = result_spec_dir / selected_repeat
    timestamp = verified_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    attestation = {
        "schema_version": SCHEMA_VERSION,
        "target_id": target_id,
        "target_version": target["target_version"],
        "target_registry_sha256": registry_sha,
        "verified_at": timestamp,
        "verified_by": verified_by,
        "successful_repeats": len(repeats),
        "minimum_repeats": minimum_repeats,
        "selected_repeat": selected_repeat,
        "exact_target_match": True,
        "exact_measured_setting_match": True,
        "zero_failed_requests": True,
        "primary_metric": "request_throughput",
        "maximum_cv_percent": maximum_cv * 100,
        "baseline_statistics": baseline_stats,
        "candidate_statistics": candidate_stats,
        "median_improvement_percent": (
            (candidate_stats["median"] - baseline_stats["median"])
            / baseline_stats["median"]
            * 100
        ),
        "repeats": repeats,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for source_name, output_name, arm in (
        ("baseline-disabled", "baseline-disabled", "baseline-disabled"),
        ("enabled-warm-cache", "simllm-enabled-warm-cache", "simllm-enabled-warm-cache"),
    ):
        source = selected_dir / source_name / "submission"
        destination = output_dir / output_name
        destination.mkdir(parents=True, exist_ok=True)
        entry = _load(source / "run_leaderboard.json")
        entry["workload"]["name"] = str((target.get("workload") or {}).get("name"))
        metadata = entry.setdefault("metadata", {})
        metadata.update(
            {
                "verified": True,
                "verified_at": timestamp,
                "verified_by": verified_by,
                "target_id": target_id,
                "target_version": target["target_version"],
                "profile_id": target["profile"],
                "benchmark_arm": arm,
                "target_registry_sha256": registry_sha,
                "verification_attestation": {
                    "schema_version": SCHEMA_VERSION,
                    "evidence": "../paired_repeat_suite.json",
                    "successful_repeats": len(repeats),
                    "selected_repeat": selected_repeat,
                },
            }
        )
        (destination / "run_leaderboard.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        shutil.copy2(
            source / "leaderboard_manifest.json",
            destination / "leaderboard_manifest.json",
        )
    (output_dir / "paired_repeat_suite.json").write_text(
        json.dumps(attestation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return attestation
