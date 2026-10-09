"""Prepare a digest-pinned Terminal-Bench candidate plan without running tasks.

Python 3.11+ prepares the plan. Optional schema validation requires the separately
installed Harbor 0.24.0 on Python 3.12+. External execution gates remain open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit

import audit_pyramidkv_terminal as source_audit

HARBOR_VERSION = "0.24.0"
HARBOR_REVISION = "b53b8134e1241686dca7759af188f987ecc48e8b"  # pragma: allowlist secret


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pin_task_image(text: str, original: str, pinned: str) -> str:
    import tomllib

    before = tomllib.loads(text)
    if before["environment"]["docker_image"] != original:
        raise ValueError("Task image differs from audited mapping")
    pattern = r'(?m)^docker_image\s*=\s*"' + re.escape(original) + r'"\s*$'
    updated, count = re.subn(pattern, "docker_image = " + json.dumps(pinned), text)
    if count != 1:
        raise ValueError("Expected exactly one frozen docker_image assignment")
    after = tomllib.loads(updated)
    after["environment"]["docker_image"] = original
    if after != before:
        raise ValueError("Image pin changed other task semantics")
    return updated


def job_config(arm: str, tasks: list[str], api_base: str) -> dict:
    if arm not in {"b0", "b1"}:
        raise ValueError("Unknown matched arm")
    url = urlsplit(api_base)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/") != "/v1"
    ):
        raise ValueError(
            "Expected credential-free OpenAI-compatible base URL ending /v1"
        )
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(re.fullmatch(r"[a-z0-9][a-z0-9.-]*", task) is None for task in tasks)
    ):
        raise ValueError("Task IDs must be unique safe names")
    return {
        "job_name": "pyramidkv-terminal-" + arm,
        "jobs_dir": "jobs-" + arm,
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "timeout_multiplier": 1.0,
        "retry": {"max_retries": 0},
        "environment": {
            "type": "docker",
            "force_build": False,
            "delete": True,
            "cpu_enforcement_policy": "limit",
            "memory_enforcement_policy": "limit",
        },
        "verifier": {"disable": False},
        "tasks": [{"path": "tasks/" + task} for task in tasks],
        "agents": [
            {
                "name": "terminus-2",
                "model_name": "openai/qwen35-pyramidkv",
                "kwargs": {
                    "api_base": api_base,
                    "temperature": 0,
                    "max_turns": 100,
                    "parser_name": "json",
                    "enable_summarize": False,
                    "proactive_summarization_threshold": 0,
                    "store_all_messages": True,
                    "record_terminal_session": True,
                    "interleaved_thinking": False,
                    "use_responses_api": False,
                    "model_info": {
                        "max_tokens": 32768,
                        "max_input_tokens": 32768,
                        "max_output_tokens": 4096,
                        "input_cost_per_token": 0,
                        "output_cost_per_token": 0,
                    },
                    "llm_kwargs": {"num_retries": 0},
                    "llm_call_kwargs": {
                        "max_tokens": 4096,
                        "seed": 17,
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    },
                },
            }
        ],
    }


def validate_harbor(jobs: dict[str, dict]) -> dict:
    import importlib.metadata

    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.models.job.config import JobConfig

    if importlib.metadata.version("harbor") != HARBOR_VERSION:
        raise ValueError("Harbor version differs from candidate contract")
    for job in jobs.values():
        validated = JobConfig.model_validate(job)
        if len(validated.tasks) != len(job["tasks"]):
            raise ValueError("Harbor changed task inventory")
        Terminus2.options_model.model_validate(job["agents"][0]["kwargs"])
    return {
        "harbor_version": HARBOR_VERSION,
        "job_schemas_validated": len(jobs),
        "agent_option_schemas_validated": len(jobs),
        "task_execution": False,
        "model_calls": 0,
        "scope": "schema validation only, not backend or model validation",
    }


def prepare(
    assets: Path, output: Path, api_base: str, *, validate: bool = False
) -> dict:
    if output.exists():
        raise ValueError("Refusing to replace an existing plan")
    # Verify the full frozen source/image inventory before copying task assets.
    source_audit.audit(assets, output / "source-audit")
    inventory = json.loads((output / "source-audit/inventory.json").read_text())
    jobs = {
        arm: job_config(arm, [r["task_id"] for r in inventory], api_base)
        for arm in ("b0", "b1")
    }
    tree = json.loads((assets / "tree.json").read_text())
    modes = {
        entry["path"]: int(entry["mode"], 8) & 0o777
        for entry in tree["tree"]
        if entry["type"] == "blob"
    }
    modifications = []
    for row in inventory:
        task_id = row["task_id"]
        original = assets / "snapshot/tasks" / task_id
        staged = output / "tasks" / task_id
        shutil.copytree(original, staged)
        for filename in row["files"]:
            (staged / filename).chmod(modes[f"tasks/{task_id}/{filename}"])
        path = staged / "task.toml"
        before = path.read_bytes()
        pinned = (
            row["image"].rsplit(":", 1)[0] + "@" + row["image_audit"]["manifest_digest"]
        )
        after = pin_task_image(before.decode(), row["image"], pinned).encode()
        path.write_bytes(after)
        modifications.append(
            {
                "task_id": task_id,
                "original_image": row["image"],
                "pinned_image": pinned,
                "original_task_toml_sha256": digest(before),
                "staged_task_toml_sha256": digest(after),
            }
        )
    for arm, job in jobs.items():
        (output / f"job-{arm}.json").write_bytes(encoded(job))
    contract = {
        "scope": "candidate Terminal-Bench 2.1 execution plan; not approved or executed",
        "source_revision": source_audit.REVISION,
        "task_count": len(inventory),
        "harbor_version": HARBOR_VERSION,
        "harbor_revision": HARBOR_REVISION,
        "agent": "terminus-2",
        "agent_version": "2.0.0",
        "model_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",  # pragma: allowlist secret
        "source_inventory_sha256": digest(
            (output / "source-audit/inventory.json").read_bytes()
        ),
        "job_sha256": {arm: digest(encoded(job)) for arm, job in jobs.items()},
        "image_pin_changes": modifications,
        "source_git_file_modes_sha256": digest(encoded(modes)),
        "file_mode_policy": "Staged task files restore executable bits from the verified Git tree",
        "attempts_per_task_per_arm": 1,
        "concurrent_trials": 1,
        "agent_turn_limit": 100,
        "output_token_limit_per_call": 4096,
        "model_context_limit": 32768,
        "context_policy": "No summarization, prompt padding or silent truncation; overflow remains a task failure",
        "retry_policy": {
            "whole_trial_retries": 0,
            "litellm_sdk_retries": 0,
            "harbor_llm_layer_attempts": 3,
            "terminus_query_layer_attempts": 3,
            "note": "Nested upstream decorators remain; do not describe this as zero model-request retries. Rejected telemetry parameters can also trigger the upstream compatibility fallback.",
        },
        "task_policy": "Keep upstream CPU/memory/storage/network and task-specific timeouts; no mounts, added instructions, skills or MCP tools",
        "quality_policy": "Report all 89 attempted tasks and failures; validate binary verifier rewards in fresh sandboxes before defining resolved-rate aggregation",
        "execution_gates": [
            "Task and third-party rights review",
            "Supported AMD64 Docker/Harbor backend and effective resource limits",
            "Reachable model endpoint and verified B0/B1 runtime activation",
            "Pinned execution-environment image and dependency artifacts",
            "Fresh-sandbox reference/failure verifier validation and canonical contract review",
        ],
        "ready_for_scored_execution": False,
        "model_calls": 0,
        "tasks_executed": 0,
    }
    (output / "contract.json").write_bytes(encoded(contract))
    if validate:
        (output / "schema-validation.json").write_bytes(encoded(validate_harbor(jobs)))
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--validate-harbor", action="store_true")
    args = parser.parse_args()
    contract = prepare(
        args.assets, args.output, args.api_base, validate=args.validate_harbor
    )
    print(
        json.dumps(
            {k: v for k, v in contract.items() if k != "image_pin_changes"}, indent=2
        )
    )
