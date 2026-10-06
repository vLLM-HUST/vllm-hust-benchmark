import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.evaluation_bundle_verifier import (
    bundle_tree_sha256,
    verify_bundle,
)
from vllm_hust_benchmark.evaluation_job_adapter import AdapterError


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seal(bundle: Path) -> str:
    sha = bundle_tree_sha256(bundle)
    (bundle / "BUNDLE_SHA256").write_text(sha + "\n", encoding="utf-8")
    return sha


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle = tmp_path / "bundle"
    result = bundle / "result"
    result.mkdir(parents=True)
    target = "target-a"
    request = {
        "schema_version": 1,
        "repository": "org/core",
        "core_commit": "a" * 40,
        "plugin_repository": "org/plugin",
        "plugin_commit": "b" * 40,
        "target_id": target,
        "target_registry_version": "1.0.0",
        "repeat_count": 3,
        "npu_count": 1,
        "priority": "required",
        "requested_by": "test",
        "source_url": "https://example.org/1",
    }
    entry = {
        "model_id": "Qwen/Test",
        "model_revision": "c" * 40,
        "load_profile": "fixed-1-rps",
        "campaign_id": "campaign/v1",
        "coverage_class": "full-matrix",
        "point_role": "checkpoint",
    }
    schedule = {
        "schedule_version": "1.0.0",
        "image_id": "sha256:" + "d" * 64,
        "cann_version": "9.1.0",
        "torch_npu_version": "2.10.0",
        "topology": "single-node-910B2",
        "targets": {target: entry},
    }
    schedule_file = tmp_path / "schedule.json"
    schedule_file.write_text(json.dumps(schedule), encoding="utf-8")
    (bundle / "request.json").write_bytes(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    _write_json(
        bundle / "worker.json",
        {
            "job_id": "eval-test",
            "worker": "host:1",
            "assigned_npus": [2],
            "exit_code": 0,
            "runner_command": ["runner"],
        },
    )
    (result / "STATUS").write_text(
        "UNVERIFIED: runner completed; publication gate not implemented\n",
        encoding="utf-8",
    )
    (result / "schedule-snapshot.json").write_bytes(schedule_file.read_bytes())
    _write_json(
        result / "execution-plan.json",
        {
            "target_id": target,
            "repeat_count": 3,
            "schedule_version": "1.0.0",
            "schedule_sha256": hashlib.sha256(schedule_file.read_bytes()).hexdigest(),
        },
    )
    runs = []
    for index in range(3):
        artifact = result / "attempts" / f"repeat-{index:02d}"
        artifact.mkdir(parents=True)
        files = {
            "pip-packages.json": "[]\n",
            "raw_benchmark_result.json": "{}\n",
            "server.stdout.log": "ready\nstopped\n",
        }
        for name, value in files.items():
            (artifact / name).write_text(value, encoding="utf-8")
        _write_json(
            artifact / "resolved_same_spec.json",
            {"model": "Qwen/Test", "scenario": "random-online"},
        )
        _write_json(
            artifact / "run_leaderboard.json",
            {"metadata": {"target_contract_id": target}},
        )
        _write_json(artifact / "leaderboard_manifest.json", {"artifacts": []})
        _write_json(
            artifact / "env-manifest.json",
            {
                "manifest_version": "run-env-manifest/v2",
                "frozen_inputs_required": True,
                "frozen_inputs": {
                    "image_id": schedule["image_id"],
                    "model_revision": entry["model_revision"],
                    "cann": {"declared": schedule["cann_version"]},
                    "torch_npu_version": {"declared": schedule["torch_npu_version"]},
                    "topology": schedule["topology"],
                },
                "campaign": {
                    "campaign_id": entry["campaign_id"],
                    "coverage_class": entry["coverage_class"],
                    "comparison_id": "",
                    "point_role": entry["point_role"],
                    "load_profile": entry["load_profile"],
                    "repeat_index": index,
                    "repetitions": 3,
                },
                "env_vars": {"ASCEND_RT_VISIBLE_DEVICES": "2"},
                "git_info": {
                    "vllm_hust": {"declared": "a" * 40, "observed": "a" * 40},
                    "vllm_ascend_hust": {
                        "declared": "b" * 40,
                        "observed": "b" * 40,
                    },
                },
            },
        )
        checksum_names = sorted(
            path.name
            for path in artifact.iterdir()
            if path.name not in {"STATUS", "checksums.sha256"}
        )
        (artifact / "checksums.sha256").write_text(
            "".join(
                f"{hashlib.sha256((artifact / name).read_bytes()).hexdigest()}  ./{name}\n"
                for name in checksum_names
            ),
            encoding="utf-8",
        )
        (artifact / "STATUS").write_text("OK\n", encoding="utf-8")
        runs.append(
            {
                "repeat_index": index,
                "exit_code": 0,
                "status": "ok",
                "artifact_dir": f"ignored-{index}",
            }
        )
    _write_json(
        result / "campaign-summary.json",
        {
            "schema_version": "independent-service-campaign-summary/v1",
            "campaign_id": entry["campaign_id"],
            "coverage_class": entry["coverage_class"],
            "comparison_id": "",
            "point_role": entry["point_role"],
            "load_profile": entry["load_profile"],
            "visible_devices": "2",
            "frozen_inputs": {
                "core_commit": "a" * 40,
                "backend_commit": "b" * 40,
                "image_id": schedule["image_id"],
                "model_revision": entry["model_revision"],
                "cann_version": schedule["cann_version"],
                "torch_npu_version": schedule["torch_npu_version"],
                "topology": schedule["topology"],
            },
            "status": "ok",
            "requested_repetitions": 3,
            "attempted_repetitions": 3,
            "successful_repetitions": 3,
            "runs": runs,
        },
    )
    sha = _seal(bundle)
    job = {
        "id": "eval-test",
        "status": "succeeded",
        "exit_code": 0,
        "worker_id": "host:1",
        "assigned_npus": [2],
        "artifact_path": str(bundle),
        "artifact_sha256": sha,
        "request": request,
    }
    job_file = tmp_path / "job.json"
    _write_json(job_file, job)
    return bundle, job_file, schedule_file


def test_verify_bundle_emits_external_verified_attestation(tmp_path: Path) -> None:
    bundle, job, schedule = _fixture(tmp_path)
    attestation = verify_bundle(bundle, job, schedule)
    assert attestation["status"] == "VERIFIED"
    assert attestation["repeat_count"] == 3
    assert (
        attestation["bundle_sha256"] == (bundle / "BUNDLE_SHA256").read_text().strip()
    )


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("request.json", "checksum"),
        ("result/attempts/repeat-01/raw_benchmark_result.json", "checksum"),
    ],
)
def test_verify_bundle_rejects_mutation(
    tmp_path: Path, relative: str, message: str
) -> None:
    bundle, job, schedule = _fixture(tmp_path)
    (bundle / relative).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(AdapterError, match=message):
        verify_bundle(bundle, job, schedule)


def test_verify_bundle_rejects_job_record_mismatch(tmp_path: Path) -> None:
    bundle, job, schedule = _fixture(tmp_path)
    payload = json.loads(job.read_text())
    payload["assigned_npus"] = [3]
    _write_json(job, payload)
    with pytest.raises(AdapterError, match="worker lifecycle"):
        verify_bundle(bundle, job, schedule)


def test_verify_bundle_accepts_sqlite_json_npu_field(tmp_path: Path) -> None:
    bundle, job, schedule = _fixture(tmp_path)
    payload = json.loads(job.read_text())
    payload["assigned_npus"] = "[2]"
    _write_json(job, payload)
    assert verify_bundle(bundle, job, schedule)["assigned_npus"] == [2]
