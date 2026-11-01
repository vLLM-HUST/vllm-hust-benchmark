"""Run pinned original mini-SWE-agent, not a replacement Agent or scorer.

Plan is inert. Execution requires external original JSON records, prebuilt
original repository images and a caller-owned exclusive model window. Only
deployment bindings and container security/resource options are supplied here.
Official functional grading remains a separate mandatory delivery step.
"""

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from prepare_jsonschemabench_author_source import verify_source

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native"
RUN_ARGS = [
    "--rm",
    "--network=none",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--memory=8g",
    "--memory-swap=8g",
    "--cpus=2",
    "--pids-limit=512",
    "--shm-size=256m",
]


def select_records(path, *, full=False):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        rows = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024 if full else 3):
            rows.extend(batch.to_pylist())
            if not full:
                break
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty original record array required")
    selected = rows if full else rows[:3]
    identities = set()
    required = {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "version",
        "patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    }
    for row in selected:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError("complete original legacy task fields required")
        identity = row["instance_id"]
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity)
            or identity in identities
        ):
            raise ValueError("nonempty unique original instance IDs required")
        if not isinstance(row["problem_statement"], str):
            raise ValueError("original problem statement must be a string")
        identities.add(identity)
    return selected


def execution_config(original, endpoint, model):
    if not model or "\n" in model or "\r" in model:
        raise ValueError("explicit single-line local model name required")
    url = urlsplit(endpoint)
    if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("explicit local model endpoint required; no paid remote API")
    if (
        url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/") != "/v1"
    ):
        raise ValueError("credential-free local /v1 endpoint required")
    config = copy.deepcopy(original)
    config["environment"]["run_args"] = list(RUN_ARGS)
    if config["environment"].get("forward_env"):
        raise ValueError(
            "do not forward client credentials into generated-code containers"
        )
    config["model"]["model_name"] = "openai/" + model
    config["model"].setdefault("model_kwargs", {})["api_base"] = endpoint
    # The official local-model mode tolerates missing price metadata. This does
    # not measure local inference cost or modify the original step/token limits.
    config["model"]["cost_tracking"] = "ignore_errors"
    return config


def execute(args, rows):
    import yaml

    verify_source(args.agent_root, PINS / "mini-swe-agent-author-source.json")
    verify_source(args.harness_root, PINS / "swebench-v410-author-source.json")
    for root in (args.agent_root / "src", args.harness_root):
        sys.path.insert(0, str(root))
    agent = importlib.import_module("minisweagent.run.benchmarks.swebench")
    specs = importlib.import_module("swebench.harness.test_spec.test_spec")
    for module, root in ((agent, args.agent_root), (specs, args.harness_root)):
        if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise ValueError(
                "already-imported software differs from pinned original root"
            )
    original = yaml.safe_load(
        (
            args.agent_root / "src/minisweagent/config/benchmarks/swebench.yaml"
        ).read_text(encoding="utf-8")
    )
    config = execution_config(original, args.endpoint, args.model)
    if not args.docker_url.startswith("unix:///"):
        raise ValueError("explicit private local Docker socket required")
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("private task results must remain outside the checkout")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.environ["DOCKER_HOST"] = args.docker_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    report = {
        "source_id": args.source_id,
        "status": "RUNNING_ORIGINAL_AGENT_NOT_GRADED",
        "original_instance_ids": [row["instance_id"] for row in rows],
        "original_agent_settings_sha256": hashlib.sha256(
            json.dumps(original["agent"], sort_keys=True).encode()
        ).hexdigest(),
        "original_step_limit": original["agent"].get("step_limit"),
        "original_cost_limit": original["agent"].get("cost_limit"),
        "original_agent_settings_preserved": config["agent"] == original["agent"],
        "original_records_sha256": hashlib.sha256(
            args.records_json.read_bytes()
        ).hexdigest(),
        "runtime_image_bindings": {},
        "completed_agent_instances": [],
        "local_price_metadata_unavailable_not_measured_cost": True,
        "official_functional_scoring_executed": False,
        "source_checkbox_eligible": False,
    }
    receipt = args.output / "agent-public.json"

    def save():
        receipt.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    progress = agent.RunBatchProgressManager(
        len(rows), args.output / "exit-statuses-private.yaml"
    )
    save()
    try:
        for row in rows:
            spec = specs.make_test_spec(row, arch=args.arch)
            instance = dict(row, image_name=spec.instance_image_key)
            report["runtime_image_bindings"][row["instance_id"]] = (
                spec.instance_image_key
            )
            save()
            # This is the unmodified publisher's complete process_instance:
            # native templates, model/tool loop, submission and trajectory.
            agent.process_instance(instance, args.output, config, progress)
            trajectory = (
                args.output / row["instance_id"] / (row["instance_id"] + ".traj.json")
            )
            if not trajectory.is_file():
                raise RuntimeError(
                    "original Agent did not save a trajectory; inspect retained error"
                )
            value = json.loads(trajectory.read_text(encoding="utf-8"))
            report["completed_agent_instances"].append(
                {
                    "instance_id": row["instance_id"],
                    "exit_status": value["info"]["exit_status"],
                    "trajectory_sha256": hashlib.sha256(
                        trajectory.read_bytes()
                    ).hexdigest(),
                }
            )
            save()
        report["status"] = "ORIGINAL_AGENT_ATTEMPTS_FINISHED_OFFICIAL_GRADING_REQUIRED"
    except Exception as error:
        report.update(
            status="ORIGINAL_AGENT_FAILURE_RETAINED", error_type=type(error).__name__
        )
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-id", choices=["SWE_BENCH", "SWE_VERIFIED"], required=True
    )
    parser.add_argument(
        "--records-file",
        "--records-json",
        dest="records_json",
        type=Path,
        required=True,
        help="Original Parquet or JSON record array; legacy JSON flag retained",
    )
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--docker-url", required=True)
    parser.add_argument("--arch", choices=["arm64", "x86_64"], default="arm64")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = select_records(args.records_json, full=args.all)
    if not args.execute:
        result = {
            "status": "PLAN_ONLY",
            "original_instance_ids": [r["instance_id"] for r in rows],
            "model_calls": 0,
            "source_checkbox_eligible": False,
        }
    else:
        result = execute(args, rows)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
