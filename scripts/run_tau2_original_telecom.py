"""Official unmodified telecom tasks; local NPU model routing only."""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time


def run_validation():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--task-references", type=Path)
    args = parser.parse_args()
    from urllib.parse import urlparse

    url = urlparse(args.endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost")
        or url.username
        or url.password
    ):
        parser.error("this NPU validation runner only permits local HTTP endpoints")
    if (
        metadata.version("tau2") != "0.2.1.dev0"
        or metadata.version("litellm") != "1.65.0"
    ):
        raise RuntimeError(
            "requires reviewed tau2 v0.2.0 archive package and litellm 1.65.0"
        )
    if (
        hashlib.sha256(args.official_archive.read_bytes()).hexdigest()
        != "65f1d59b3783bfb38664142fb166aa40b369e4b4083170723059d8f1883ccd64"  # pragma: allowlist secret - official source archive checksum
    ):
        raise RuntimeError("official archive identity mismatch")
    run = args.output.resolve()
    run.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.umask(0o077)
    os.environ.update(
        TAU2_DATA_DIR=str(args.data_root.resolve()),
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        LITELLM_TELEMETRY="False",
        LANGFUSE_ENABLED="False",
    )
    import tau2
    import zipfile

    package_root = Path(tau2.__file__).parent
    with zipfile.ZipFile(args.official_archive) as archive:
        for name in archive.namelist():
            if "/data/" in name and not name.endswith("/"):
                relative_data = name.split("/data/", 1)[1]
                if (args.data_root / relative_data).read_bytes() != archive.read(name):
                    raise RuntimeError(
                        "task/environment data differs from pinned official archive"
                    )
            marker = "/src/tau2/"
            if marker in name and name.endswith(".py"):
                relative = name.split(marker, 1)[1]
                if (package_root / relative).read_bytes() != archive.read(name):
                    raise RuntimeError(
                        "installed tau2 code differs from pinned official archive"
                    )
    from tau2.run import get_tasks
    from tau2.utils import llm_utils
    from tau2.cli import main

    tasks = get_tasks("telecom_full", num_tasks=3)
    assert len(tasks) == 3
    if args.task_references:
        references = json.loads(args.task_references.read_text(encoding="utf-8"))
        if isinstance(references, dict):
            references = references["native_task_refs"]
        if not isinstance(references, list) or len(references) != 3:
            raise ValueError("exactly three prepared native references required")
        if [row["native_task_id"] for row in references] != [task.id for task in tasks]:
            raise ValueError(
                "prepared task IDs differ from preselected original loader tasks"
            )
        for row in references:
            if (
                row["source_id"] != "TAUBENCH"
                or row["revision"]
                != "f8de30c298689cbe0117d76a378e7315a17e5bd8"  # pragma: allowlist secret - public upstream revision
                or row["task_set"] != "telecom_full"
                or row["domain"] != "telecom"
                or row["runtime_kind"] != "official_native_task"
            ):
                raise ValueError("invalid native task reference binding")
            reference = row["input_ref"]
            if reference["file"] != "tau2/domains/telecom/tasks_full.json":
                raise ValueError("unexpected native input file")
            raw = (args.data_root / reference["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
                raise ValueError("native task file identity mismatch")
            if json.loads(raw)[reference["row_index"]]["id"] != row["native_task_id"]:
                raise ValueError("native task row identity mismatch")
    assert all(
        [str(v.value) for v in t.evaluation_criteria.reward_basis] == ["ENV_ASSERTION"]
        for t in tasks
    )
    original_completion = llm_utils.completion
    endpoint = args.endpoint.rstrip("/")
    model = "openai/" + args.model

    def local_only_completion(*args, **kwargs):
        if kwargs.get("model") != model or kwargs.get("api_base") != endpoint:
            raise RuntimeError("Refused unapproved model or non-local endpoint")
        return original_completion(*args, **kwargs)

    llm_utils.completion = local_only_completion
    model_args = json.dumps(
        {
            "api_base": endpoint,
            "api_key": "local-unused",  # pragma: allowlist secret - dummy key for loopback-only server
            "temperature": 0,
            "num_retries": 0,
            "timeout": 300,
        }
    )
    receipt = {
        "source_id": "TAUBENCH",
        "revision": "f8de30c298689cbe0117d76a378e7315a17e5bd8",  # pragma: allowlist secret - public upstream revision
        "task_set": "telecom_full",
        "selected_task_ids": [t.id for t in tasks],
        "selection": "first three original official task-loader rows before execution",
        "agent": "llm_agent",
        "user": "user_simulator",
        "model": model,
        "evaluation_type": "ALL (official run_domain default; not ALL_WITH_NL_ASSERTIONS)",
        "max_steps": 200,
        "max_errors": 10,
        "num_trials": 1,
        "status": "STARTED",
        "started_at_unix": time.time(),
        "reference_model_difference": "both roles use deployable local Qwen3.5-27B, not reference GPT-4.1",
        "network_guard": "completion rejects other models/endpoints; no task/prompt/tool changes",
    }
    target = run / "receipt.json"
    if args.task_references:
        receipt["native_task_reference_sha256"] = hashlib.sha256(
            args.task_references.read_bytes()
        ).hexdigest()
        receipt["native_task_reference_handoff_verified"] = True
    target.write_text(json.dumps(receipt, indent=2))
    if args.preflight_only:
        receipt.update(status="PREFLIGHT_VERIFIED_NO_MODEL_CALLS", model_calls=0)
        target.write_text(json.dumps(receipt, indent=2))
        return
    sys.argv = [
        "tau2",
        "run",
        "--domain",
        "telecom",
        "--task-set-name",
        "telecom_full",
        "--task-ids",
        *[t.id for t in tasks],
        "--num-trials",
        "1",
        "--agent-llm",
        model,
        "--user-llm",
        model,
        "--agent-llm-args",
        model_args,
        "--user-llm-args",
        model_args,
        "--max-concurrency",
        "1",
        "--save-to",
        str(run / "official-results"),
        "--log-level",
        "ERROR",
    ]
    try:
        main()
        receipt["status"] = "CLI_RETURNED_RESULTS_REQUIRE_AUDIT"
    except BaseException as error:
        receipt.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        target.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    run_validation()
