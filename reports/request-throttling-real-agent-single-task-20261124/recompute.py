"""Project public single-task evidence from locally retained Controller archives."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def project_run(run, protocol, arm, report, grade):
    (task,) = json.loads(report)["tasks"]
    scores = json.loads(grade)
    task_id = task["instance_id"]
    assert set(scores) == {task_id}
    assert task["agent_submitted"] and task["status"] == "Submitted"
    assert task["patch_bytes"] > 0 and task["container_cleanup_exit_code"] == 0
    released = None
    if arm == "B1-system":
        release = task["program_release"]
        assert (
            release["program_id"] == task["program_id"] and release["released"] is True
        )
        released = True
    return task_id, {
        "run": run,
        "protocol": protocol,
        "arm": arm,
        "model_calls": task["model_calls"],
        "elapsed_seconds": task["elapsed_seconds"],
        "patch_bytes": task["patch_bytes"],
        "official_resolved": scores[task_id],
        "program_released": released,
        "batch_report_sha256": digest(report),
        "official_grading_sha256": digest(grade),
    }


def grader_controls(read, task_id):
    result = {}
    for label in ("reference", "empty"):
        grade = json.loads(read(f"{label}-control-grading/eval_results.json"))
        assert set(grade) == {task_id}
        result[label] = grade[task_id]
    assert result == {"reference": True, "empty": False}
    return result


def verify_b0(root):
    directory = root / "standardization-20261124/myc-b1-delay-mmlu8192-1124-r1"
    prefix = "b0-direct-01"
    archive = directory / f"{prefix}.evidence.tar.gz"
    checksum = digest(archive.read_bytes())
    assert checksum == (directory / f"{prefix}.archive-receipt").read_text().split()[-1]
    with tarfile.open(archive) as tar:

        def archived(name):
            return json.load(tar.extractfile(f"{prefix}/{name}"))

        manifest = archived("archive-manifest.json")
        names = {member.name for member in tar.getmembers() if member.isfile()}
        assert len(manifest) + 1 == len(names)
        assert {entry["file"] for entry in manifest} == names - {
            f"{prefix}/archive-manifest.json"
        }
        for entry in manifest:
            assert digest(tar.extractfile(entry["file"]).read()) == entry["sha256"]
        identity = archived("service/runtime-identity.json")
        profile = archived("service/b0-profile-admission.json")
        workload = archived("plan.json")["workload"]
        assert workload["dataset"] == "mmlu-pro" and workload["requests"] == 200
        tp = profile["argv"][profile["argv"].index("--tensor-parallel-size") + 1]
    sidecar = (directory / f"{prefix}.engine.json").read_bytes()
    (engine,) = json.loads(sidecar)
    assert identity["role"] == engine["role"] == "B0"
    assert identity["vllm_version"] == engine["actual_vllm"] == "0.18.0"
    assert identity["ascend_version"] == engine["actual_vllm_ascend"] == "0.18.0"
    assert tp == engine["flags"]["--tensor-parallel-size"] == "2"
    return {
        "dataset": workload["dataset"],
        "variant": profile["variant"],
        "serving_release": engine["release"],
        "vllm": identity["vllm_version"],
        "vllm_ascend": identity["ascend_version"],
        "tensor_parallel_size": int(tp),
        "startup_seconds": engine["startup_seconds"],
        "engine_receipt_sha256": digest(sidecar),
        "archive_sha256": checksum,
        "verified_manifest_members": len(manifest),
    }


def recompute(root):
    base = root / "standardization-20261104"
    prefix = "myc-b1-delay-swe-1104-r"
    r23 = base / f"{prefix}23/raw-completed-private"
    r25 = base / f"{prefix}25/raw-private/r25"
    direct = "B1-engine"
    controller = "B1-system"
    rows = []
    for run, protocol, arm, batch, grade in (
        (
            "R22",
            "original tool format",
            direct,
            r23 / f"{prefix}22/B-batch",
            r23 / f"{prefix}23/B-grading",
        ),
        (
            "R22",
            "original tool format",
            controller,
            r23 / f"{prefix}22/C-batch",
            r23 / f"{prefix}23/C-grading",
        ),
        (
            "R23",
            "tool_response",
            direct,
            r23 / f"{prefix}23/native-tool-full-batch",
            r23 / f"{prefix}23/native-tool-full-grading",
        ),
        ("R25", "tool_response", controller, r25 / "C-batch", r25 / "C-grading"),
    ):
        rows.append(
            project_run(
                run,
                protocol,
                arm,
                (batch / "batch-report.json").read_bytes(),
                (grade / "eval_results.json").read_bytes(),
            )
        )

    task_id = rows[0][0]
    november4 = grader_controls(
        lambda name: (r23 / f"{prefix}23" / name).read_bytes(), task_id
    )
    later = root / "standardization-20261116/myc-b1-delay-swe-1116-r1"
    archive = later / "cpu-raw-evidence.tar.gz"
    checksum = digest(archive.read_bytes())
    assert (
        checksum
        == json.loads((later / "cpu-raw-evidence.tar.verified.json").read_text())[
            "sha256"
        ]
    )
    with tarfile.open(archive) as tar:

        def read(name):
            return tar.extractfile(name).read()

        for key, arm in (("B", direct), ("C", controller)):
            rows.append(
                project_run(
                    "1116-r1",
                    "request-scoped recompute",
                    arm,
                    read(f"{key}-batch/batch-report.json"),
                    read(f"{key}-grading/eval_results.json"),
                )
            )
        november16 = grader_controls(read, task_id)
    assert all(instance == task_id for instance, _ in rows)
    return {
        "distinct_swe_tasks": len({instance for instance, _ in rows}),
        "swe_runs": [row for _, row in rows],
        "swe_grading_controls": {"2026-11-04": november4, "2026-11-16": november16},
        "november_16_cpu_archive_sha256": checksum,
        "separate_b0_startup": verify_b0(root),
    }


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O to enable all evidence checks")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help="Retained request-throttling-controller/results directory",
    )
    args = parser.parse_args()
    print(json.dumps(recompute(args.results), indent=2))
