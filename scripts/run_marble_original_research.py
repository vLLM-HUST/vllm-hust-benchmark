"""Run the pinned original Research prefix, not a synthetic Agent replacement.

Default plan-only. --prepare-only stages software/configurations without model
imports. --execute invokes unchanged marble.main with the approved evaluator
indentation compatibility copy and disclosed local model/keyword transports.
Neural workers are external NPU-only workers, not part of this CPU runner.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from bind_marble_research_local import bind_config, model_name
from prepare_jsonschemabench_author_source import verify_source
from prepare_marble_evaluator_compat import compatibility_copy
from marble_config_serialization import native_config_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROFILE = ROOT / "configs/workload-native/marble-author-source.json"
WORKER = """
import json,os,pathlib,runpy,sys
stage=pathlib.Path(sys.argv[1]);config=pathlib.Path(sys.argv[2])
sys.path[:0]=[str(stage/'helpers'),str(stage/'author')]
from appworld_model_transport import model_proxy
from marble_live_model_transport import live_model_transport
from marble_keybert_transport import keyword_transport
import marble.environments.research_utils.paper_collector as collector
endpoint=os.environ.get('MARBLE_OWNED_MODEL_API_BASE')
transport=(live_model_transport(config.parent/'live-model-queue',endpoint,port=14005,
           reuse_root=os.environ.get('MARBLE_VERIFIED_RESPONSE_REUSE_ROOT'),
           expected_model=os.environ.get('MARBLE_EXPECTED_REUSE_MODEL'),
           reuse_state=stage/'saved-response-reuse-claims')
           if endpoint else model_proxy(stage/'model-queue',port=14005))
keyword_queue=config.parent/'live-keyword-queue' if endpoint else stage/'keyword-queue'
with keyword_transport(collector,keyword_queue), transport:
    sys.argv=['marble.main','--config_path',str(config)]
    runpy.run_module('marble.main',run_name='__main__')
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_prefix(prepared_root, bound_root):
    prepared, bound = map(
        lambda p: Path(p).resolve(strict=True), (prepared_root, bound_root)
    )
    original = json.loads((prepared / "preparation-public.json").read_text("utf-8"))
    binding = json.loads((bound / "binding-public.json").read_text("utf-8"))
    if (
        original.get("source_id") != "MULTIAGENTBENCH"
        or original.get("status")
        != "OFFICIAL_RESEARCH_RUNTIME_PROFILE_PREPARED_NOT_EXECUTED"
        or original.get("runtime_profile") != "official-research"
        or binding.get("status")
        != "LOCAL_MODELS_BOUND_ORIGINAL_RESEARCH_PROTOCOL_NOT_EXECUTED"
        or binding.get("source_id") != "MULTIAGENTBENCH"
    ):
        raise ValueError(
            "original Research preparation and explicit local binding required"
        )
    actor, judge = binding["actor_model"], binding["judge_model"]
    for name in (actor, judge):
        model_name(name)
        if "together_ai/TA" in name:
            raise ValueError("publisher external-provider special route forbidden")
    originals, bindings = original["configurations"], binding["configurations"]
    # Validation uses first3; public execution supports every original Research
    # prefix produced by the publisher converter, including all100 tasks.
    count = len(originals)
    if (
        not 1 <= count <= 100
        or len(bindings) != count
        or original.get("selected_configurations") != count
        or binding.get("selected_configurations") != count
        or [x["source_row_index"] for x in originals] != list(range(count))
        or [x["source_row_index"] for x in bindings] != list(range(count))
        or any(
            x["source_file"] != "multiagentbench/research/research_main.jsonl"
            for x in originals
        )
    ):
        raise ValueError("matching complete original Research prefix required (1..100)")
    configs = []
    for source, item in zip(originals, bindings):
        paths = [prepared / source["configuration"], bound / item["configuration"]]
        for path, root in zip(paths, (prepared, bound)):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("configuration escapes original preparation")
        if (
            digest(paths[0]) != source["sha256"]
            or digest(paths[1]) != item["bound_configuration_sha256"]
            or item["original_preparation_sha256"] != source["sha256"]
        ):
            raise ValueError("original or bound configuration identity mismatch")
        config = json.loads(paths[1].read_text("utf-8"))
        result = Path(config["output"]["file_path"])
        if not result.is_absolute() or not result.resolve().is_relative_to(bound):
            raise ValueError("private result binding escapes bound directory")
        expected = bind_config(
            json.loads(paths[0].read_text("utf-8")),
            model=actor,
            judge=judge,
            result_path=result,
        )
        if config != expected:
            raise ValueError("original task changed beyond three deployment bindings")
        configs.append(config)
    return configs, actor, judge


def prepare(author_root, prepared_root, bound_root, output):
    author = Path(author_root).resolve(strict=True)
    output = Path(output).resolve()
    roots = [
        author,
        Path(prepared_root).resolve(strict=True),
        Path(bound_root).resolve(strict=True),
        ROOT,
    ]
    if output.exists() or any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in roots
    ):
        raise ValueError("fresh separate private native output required")
    verify_source(author, SOURCE_PROFILE)
    configs, actor, judge = original_prefix(prepared_root, bound_root)
    profile = json.loads(SOURCE_PROFILE.read_text("utf-8"))
    patched, changed = compatibility_copy(
        (author / "marble/evaluator/evaluator.py").read_bytes()
    )
    output.mkdir(mode=0o700)
    for name in (
        "author",
        "helpers",
        "model-queue",
        "keyword-queue",
        "saved-response-reuse-claims",
    ):
        (output / name).mkdir(mode=0o700)
    for item in profile["files"]:
        target = output / "author" / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            patched
            if item["path"] == "marble/evaluator/evaluator.py"
            else (author / item["path"]).read_bytes()
        )
    for name in (
        "appworld_model_transport.py",
        "marble_keybert_transport.py",
        "marble_live_model_transport.py",
    ):
        (output / "helpers" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    (output / "native_worker.py").write_text(WORKER, "utf-8")
    for index, config in enumerate(configs):
        folder = output / f"original-row-{index:05d}"
        folder.mkdir(mode=0o700)
        (folder / "logs").mkdir()
        (folder / "live-keyword-queue").mkdir(mode=0o700)
        (folder / "marble").symlink_to(
            output / "author/marble", target_is_directory=True
        )
        (folder / "evaluator").symlink_to(
            output / "author/marble/evaluator", target_is_directory=True
        )
        runtime = deepcopy(config)
        runtime["output"]["file_path"] = str(folder / "original-result.jsonl")
        (folder / "runtime.private.yaml").write_text(
            native_config_json(runtime), "utf-8"
        )
    report = {
        "source_id": "MULTIAGENTBENCH",
        "status": "ORIGINAL_RESEARCH_NATIVE_RUNTIME_PREPARED_NOT_EXECUTED",
        "fixed_original_rows": list(range(len(configs))),
        "actor_model": actor,
        "judge_model": judge,
        "original_iterations": 5,
        "evaluator_compatibility_changed_lines": changed,
        "original_author_directory_modified": False,
        "official_model_reproduction": False,
        "model_requests": 0,
        "completed": [],
        "source_checkbox_eligible": False,
    }
    (output / "state-private.json").write_text(json.dumps(report, indent=2), "utf-8")
    return report


def saved_model_http_summary(output):
    """Transport evidence only; native exit0 does not make an HTTP400 successful."""
    output = Path(output)
    queues = [output / "model-queue", *output.glob("original-row-*/live-model-queue")]
    counts = {}
    requests = pending = 0
    for queue in queues:
        requested = {
            p.name.removesuffix(".request.json") for p in queue.glob("*.request.json")
        }
        responses = {
            p.name.removesuffix(".response.json"): p
            for p in queue.glob("*.response.json")
        }
        if responses.keys() - requested:
            raise ValueError("model response without matching saved request")
        requests += len(requested)
        pending += len(requested - responses.keys())
        for path in responses.values():
            if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("saved model response escapes private output")
            status = json.loads(path.read_text("utf-8")).get("http_status")
            if type(status) is not int or not 100 <= status <= 599:
                raise ValueError(
                    "saved model response requires an explicit HTTP status"
                )
            key = str(status)
            counts[key] = counts.get(key, 0) + 1
    return {
        "requests": requests,
        "responses": sum(counts.values()),
        "pending": pending,
        "http_status_counts": counts,
        "non200_responses": sum(value for key, value in counts.items() if key != "200"),
        "quality_scored_by_this_summary": False,
    }


def execute(output, report):
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise ValueError("qualified isolated Linux Python3.10 MARBLE client required")
    env = dict(
        os.environ,
        PYTHONPATH="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        OPENAI_API_KEY="local-not-a-secret",  # pragma: allowlist secret - dummy credential for owned loopback proxy only
        OPENAI_API_BASE="http://127.0.0.1:14005/v1",
        OPENAI_BASE_URL="http://127.0.0.1:14005/v1",
    )
    # Do not inherit an accidental endpoint from the caller's environment.
    env.pop("MARBLE_OWNED_MODEL_API_BASE", None)
    env.pop("MARBLE_VERIFIED_RESPONSE_REUSE_ROOT", None)
    env.pop("MARBLE_EXPECTED_REUSE_MODEL", None)
    if report.get("model_api_base") is not None:
        from marble_live_model_transport import loopback_endpoint

        env["MARBLE_OWNED_MODEL_API_BASE"] = loopback_endpoint(report["model_api_base"])
        if report.get("reuse_model_results") is not None:
            env["MARBLE_VERIFIED_RESPONSE_REUSE_ROOT"] = report["reuse_model_results"]
            env["MARBLE_EXPECTED_REUSE_MODEL"] = report["actor_model"].removeprefix(
                "openai/"
            )
    for index in report["fixed_original_rows"]:
        folder = output / f"original-row-{index:05d}"
        report.update(
            status="RUNNING_ORIGINAL_RESEARCH_NATIVE_CLI", active_original_row=index
        )
        (output / "state-private.json").write_text(
            json.dumps(report, indent=2), "utf-8"
        )
        with (
            (folder / "stdout-private.txt").open("w") as out,
            (folder / "stderr-private.txt").open("w") as err,
        ):
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-IB",
                    str(output / "native_worker.py"),
                    str(output),
                    str(folder / "runtime.private.yaml"),
                ],
                cwd=folder,
                env=env,
                stdout=out,
                stderr=err,
            )
            try:
                birth = (
                    Path(f"/proc/{process.pid}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[19]
                )
            except FileNotFoundError:
                if process.poll() is None:
                    raise
                birth = None  # Already terminal; never a live-process identity.
            report["active_child"] = {"pid": process.pid, "birth": birth}
            (output / "state-private.json").write_text(
                json.dumps(report, indent=2), "utf-8"
            )
            exit_code = process.wait()
        report.pop("active_child", None)
        report["completed"].append(
            {
                "original_row": index,
                "cli_exit_code": exit_code,
                "result_file_exists": (folder / "original-result.jsonl").is_file(),
            }
        )
        report["model_requests"] = (
            len(list(output.glob("original-row-*/live-model-queue/*.request.json")))
            if report.get("model_api_base") is not None
            else len(list((output / "model-queue").glob("*.request.json")))
        )
        report["reused_saved_model_responses"] = len(
            list(output.glob("original-row-*/live-model-queue/*.reuse-private.json"))
        )
        (output / "state-private.json").write_text(
            json.dumps(report, indent=2), "utf-8"
        )
    report["status"] = "ORIGINAL_RESEARCH_NATIVE_CLIENT_TERMINAL_AUDIT_PENDING"
    report["model_http"] = saved_model_http_summary(output)
    (output / "state-private.json").write_text(json.dumps(report, indent=2), "utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "prepared-root", "bound-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--original-row",
        type=int,
        choices=range(3),
        help="Execute only this zero-based original row in a fresh recovery output directory",
    )
    parser.add_argument(
        "--model-api-base",
        help="Qualified owned loopback /v1 service: task-scoped direct forwarding; keyword worker remains external",
    )
    parser.add_argument(
        "--reuse-model-results",
        type=Path,
        help="Explicit byte-verified original responses for validation recovery only, never performance measurement",
    )
    args = parser.parse_args(argv)
    if args.original_row is not None and not args.execute:
        parser.error("--original-row requires --execute")
    if args.reuse_model_results is not None and args.model_api_base is None:
        parser.error("--reuse-model-results requires --model-api-base")
    endpoint = None
    if args.model_api_base is not None:
        from marble_live_model_transport import loopback_endpoint

        endpoint = loopback_endpoint(args.model_api_base)
    if not args.execute and not args.prepare_only:
        print(
            json.dumps(
                {
                    "source_id": "MULTIAGENTBENCH",
                    "status": "PLAN_ONLY",
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return 0
    report = prepare(args.author_root, args.prepared_root, args.bound_root, args.output)
    if args.original_row is not None:
        report["full_fixed_original_rows"] = report["fixed_original_rows"]
        report["fixed_original_rows"] = [args.original_row]
        report["partial_recovery_only"] = True
    if endpoint is not None:
        report["model_api_base"] = endpoint
        report["model_transport"] = "TASK_SCOPED_LIVE_LOOPBACK_NONTHINKING"
        if args.reuse_model_results is not None:
            from marble_live_model_transport import saved_responses, uncertain_requests

            cache = args.reuse_model_results.resolve(strict=True)
            saved_responses(cache, report["actor_model"].removeprefix("openai/"))
            report["guarded_unanswered_original_requests"] = len(
                uncertain_requests(cache)
            )
            report["reuse_model_results"] = str(cache)
            report["saved_response_reuse_is_validation_recovery_not_performance"] = True
        (args.output / "state-private.json").write_text(
            json.dumps(report, indent=2), "utf-8"
        )
    if args.execute:
        report = execute(args.output.resolve(), report)
    print(json.dumps(report, indent=2))
    return int(
        args.execute
        and (
            any(
                row["cli_exit_code"] != 0 or not row["result_file_exists"]
                for row in report["completed"]
            )
            or not report.get("model_http", {}).get("requests")
            or report["model_http"]["pending"] > 0
            or report["model_http"]["non200_responses"] > 0
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
