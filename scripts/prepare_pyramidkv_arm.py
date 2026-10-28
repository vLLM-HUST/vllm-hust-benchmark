"""Prepare separate PyramidKV SWE V1 / Terminal ARM contracts without execution.

Generated settings are candidates, not a backend handoff, scoring approval or
server activation. No Docker, model, task, oracle or grader process is started.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import prepare_pyramidkv_terminal as terminal

from vllm_hust_benchmark import swe_bench_pro as swe


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_ids(ids: list[str]) -> None:
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Explicit nonempty unique task IDs required")
    if any(
        re.fullmatch(r"[A-Za-z0-9_.-]+", x) is None or x in {".", ".."} for x in ids
    ):
        raise ValueError("Unsafe task ID")


def boundary(dataset: str) -> dict:
    return {
        "schema": "pyramidkv-arm-candidate-v1",
        "dataset": dataset,
        "image_platform": "linux/arm64",
        "scope": "CPU-only input/configuration preparation; no task score",
        "preparation_authorized_on": "2026-10-28",
        "preparer_sha256": sha(Path(__file__)),
        "ready_for_scored_execution": False,
        "model_calls": 0,
        "tasks_executed": 0,
        "backend_verified": False,
        "server_activation_verified": False,
        "judge_calibration_requirement": "unchanged",
        "comparison": "candidate same-host compression OFF/ON; not official v0.18 acceptance",
        "execution_gates": [
            "Concrete task/scaffold/tool/budget and rights contract review",
            "Owner-confirmed ARM workspace, enforced resources and model route",
            "Exact installed image/source and fresh-sandbox verifier qualification",
            "Pinned model/tokenizer/template/runtime and input/context preflight",
            "Matched server lifecycle/cache policy and observed OFF/ON activation",
            "Execution window coordinated with benchmark #403 owners",
        ],
    }


def swe_pair(rows: list[dict], ids: list[str], template: dict) -> dict[str, dict]:
    safe_ids(ids)
    by_id = {row["instance_id"]: row for row in rows}
    if len(by_id) != len(rows) or any(x not in by_id for x in ids):
        raise ValueError("Duplicate source rows or selected task absent from frozen V1")
    repos = {by_id[x]["repo"] for x in ids}
    if len(repos) != 1:
        raise ValueError("Shared runner needs a separate plan per repository")
    # Reuse the existing endpoint guard, without creating a Harbor environment.
    terminal.job_config("b0", ["endpoint-check"], template["model"]["api_base"])
    if template.get("integration") or template["agent"].get("task_instructions"):
        raise ValueError(
            "No additional Controller hook or task instruction in this contract"
        )
    if "api_key" in template["model"]:
        raise ValueError("Use api_key_env, not an embedded credential")
    jobs = {}
    for arm in ("b0", "b1"):
        job = copy.deepcopy(template)
        job["run_id"] = "pyramidkv-swe-v1-arm-" + arm
        job["batch"] = {
            "repository": next(iter(repos)),
            "selection": "explicit-task-ids",
            "size": len(ids),
            "arrival_time": 0,
            "client_concurrency_limit": None,
            "independent_agents_per_task": 1,
            "program_instances": len(ids),
            "instance_ids": sorted(ids),
        }
        job["model"]["model_name"] = "openai/qwen35-pyramidkv"
        job["agent"].update(seed=17, temperature=0, thinking=False)
        swe.validate(job)
        jobs[arm] = job
    return jobs


def prepare_swe(
    dataset: Path, ids: list[str], template_path: Path, output: Path
) -> dict:
    if output.exists():
        raise ValueError("Refusing to overwrite existing preparation")
    lock = json.loads(swe.LOCK_PATH.read_text())
    if sha(dataset) != lock["dataset"]["parquet_sha256"]:
        raise ValueError("Frozen V1 parquet hash mismatch; V2 is not interchangeable")
    from pyarrow import parquet

    rows = parquet.read_table(dataset).to_pylist()
    template = json.loads(template_path.read_text())
    jobs = swe_pair(rows, ids, template)
    selected = [row for row in rows if row["instance_id"] in ids]
    source_pins = (
        lock["official_v1"]["sha256"]
        | lock["official_v1"]["nodebb_task_source_sha256"]
        | lock["official_v1"].get("grading_dockerfiles_sha256", {})
    )
    required = []
    for row in selected:
        identifier = row["instance_id"]
        required.extend(
            [
                f"dockerfiles/{kind}/{identifier}/Dockerfile"
                for kind in ("base_dockerfile", "instance_dockerfile")
            ]
            + [
                f"run_scripts/{identifier}/{name}"
                for name in ("run_script.sh", "parser.py")
            ]
        )
    if any(path not in source_pins for path in required):
        raise ValueError("Selected task lacks frozen V1 grading-source coverage")
    result = boundary("SWE-bench-Pro/V1")
    result.update(
        dataset_identity=lock["dataset"],
        source_lock_sha256=sha(swe.LOCK_PATH),
        shared_runner_sha256=sha(Path(swe.__file__)),
        settings_template_sha256=sha(template_path),
        source_row_count=len(rows),
        task_ids=sorted(ids),
        selected_row_sha256={
            row["instance_id"]: hashlib.sha256(encoded(row)).hexdigest()
            for row in selected
        },
        required_grading_source_sha256={p: source_pins[p] for p in sorted(required)},
        agent_revision=lock["agent"]["commit"],
        evaluator_revision=lock["official_v1"]["commit"],
        runtime_paths_status="template only; backend must bind paths and rerun shared check/prepare/verify-inputs",
        images_verified=False,
        settings_sha256={
            arm: hashlib.sha256(encoded(job)).hexdigest() for arm, job in jobs.items()
        },
    )
    output.mkdir(parents=True)
    for arm, job in jobs.items():
        (output / f"settings-{arm}.json").write_bytes(encoded(job))
    (output / "contract.json").write_bytes(encoded(result))
    return result


def validate_arm_reference(reference: dict, task_dir: Path) -> str:
    image = reference.get("image_id", "")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("Immutable ARM Docker image config digest required")
    pins = reference.get("task_files", {})
    # A historical control plan is a reference, not proof of current control success.
    required = {
        "instruction.md",
        "task.toml",
        "environment/Dockerfile",
        "solution/solve.sh",
        "tests/test.sh",
    }
    if not required.issubset(pins) or not any(
        p.startswith("tests/") and p != "tests/test.sh" for p in pins
    ):
        raise ValueError("Incomplete task/control source reference")
    for name, expected in pins.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe source-reference path")
        # These two files describe the ARM derivation, not original task bytes.
        if (
            name not in {"task.toml", "environment/Dockerfile"}
            and sha(task_dir / name) != expected
        ):
            raise ValueError(
                "ARM reference changed task, solution or verifier bytes: " + name
            )
    return image


def prepare_terminal(
    assets: Path, task_id: str, reference_path: Path, api_base: str, output: Path
) -> dict:
    if output.exists():
        raise ValueError("Refusing to overwrite existing preparation")
    safe_ids([task_id])
    jobs = {arm: terminal.job_config(arm, [task_id], api_base) for arm in ("b0", "b1")}
    original = assets / "snapshot/tasks" / task_id
    reference = json.loads(reference_path.read_text())
    image = validate_arm_reference(reference, original)
    terminal.source_audit.audit(assets, output / "source-audit")
    inventory = json.loads((output / "source-audit/inventory.json").read_text())
    found = [row for row in inventory if row["task_id"] == task_id]
    if len(found) != 1:
        raise ValueError("Task absent from frozen Terminal inventory")
    row = found[0]
    staged = output / "tasks" / task_id
    shutil.copytree(original, staged)
    tree = json.loads((assets / "tree.json").read_text())
    modes = {
        entry["path"]: int(entry["mode"], 8) & 0o777
        for entry in tree["tree"]
        if entry["type"] == "blob"
    }
    for name in row["files"]:
        (staged / name).chmod(modes[f"tasks/{task_id}/{name}"])
    path = staged / "task.toml"
    before = path.read_bytes()
    path.write_text(terminal.pin_task_image(before.decode(), row["image"], image))
    result = boundary("Terminal-Bench/2.1/ARM64-derived")
    result.update(
        source_revision=terminal.source_audit.REVISION,
        task_ids=[task_id],
        harbor_version=terminal.HARBOR_VERSION,
        harbor_revision=terminal.HARBOR_REVISION,
        terminal_preparer_sha256=sha(Path(terminal.__file__)),
        source_audit_sha256=sha(Path(terminal.source_audit.__file__)),
        historical_arm_reference_sha256=sha(reference_path),
        arm_image_config_digest=image,
        image_transport_status="not provided or verified; load and inspect exact ARM image before execution",
        historical_control_reused_as_current_qualification=False,
        original_task_toml_sha256=hashlib.sha256(before).hexdigest(),
        staged_task_toml_sha256=sha(path),
        source_inventory_sha256=sha(output / "source-audit/inventory.json"),
        staged_files={
            name: {
                "sha256": sha(staged / name),
                "mode": oct((staged / name).stat().st_mode & 0o777),
            }
            for name in sorted(row["files"])
        },
        build_policy="force_build=false; original Dockerfile retained as source, not a qualified ARM rebuild recipe",
        retry_policy="SDK and trial retries disabled; upstream Harbor/Terminus nested retry decorators retained",
        settings_sha256={
            arm: hashlib.sha256(encoded(job)).hexdigest() for arm, job in jobs.items()
        },
    )
    for arm, job in jobs.items():
        (output / f"job-{arm}.json").write_bytes(encoded(job))
    (output / "contract.json").write_bytes(encoded(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)
    s = sub.add_parser("swe")
    s.add_argument("--parquet", required=True, type=Path)
    s.add_argument("--task-id", action="append", required=True)
    s.add_argument("--settings-template", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    t = sub.add_parser("terminal")
    t.add_argument("--assets", required=True, type=Path)
    t.add_argument("--task-id", required=True)
    t.add_argument("--arm-reference", required=True, type=Path)
    t.add_argument("--api-base", required=True)
    t.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.dataset == "swe":
        result = prepare_swe(
            args.parquet, args.task_id, args.settings_template, args.output
        )
    else:
        result = prepare_terminal(
            args.assets, args.task_id, args.arm_reference, args.api_base, args.output
        )
    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "ready_for_scored_execution": False,
                "task_ids": result["task_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
