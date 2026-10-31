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

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROFILE = ROOT / "configs/workload-native/marble-author-source.json"
WORKER = """
import json,pathlib,runpy,sys
stage=pathlib.Path(sys.argv[1]);config=pathlib.Path(sys.argv[2])
sys.path[:0]=[str(stage/'helpers'),str(stage/'author')]
from appworld_model_transport import model_proxy
from marble_keybert_transport import keyword_transport
import marble.environments.research_utils.paper_collector as collector
with keyword_transport(collector,stage/'keyword-queue'), model_proxy(stage/'model-queue',port=14005):
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
    # This delivery protocol fixes the first3 original Research tasks.
    if (
        len(originals) != 3
        or len(bindings) != 3
        or [x["source_row_index"] for x in originals] != [0, 1, 2]
        or [x["source_row_index"] for x in bindings] != [0, 1, 2]
        or any(
            x["source_file"] != "multiagentbench/research/research_main.jsonl"
            for x in originals
        )
    ):
        raise ValueError("fixed original first3 Research tasks required")
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
    for name in ("author", "helpers", "model-queue", "keyword-queue"):
        (output / name).mkdir(mode=0o700)
    for item in profile["files"]:
        target = output / "author" / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            patched
            if item["path"] == "marble/evaluator/evaluator.py"
            else (author / item["path"]).read_bytes()
        )
    for name in ("appworld_model_transport.py", "marble_keybert_transport.py"):
        (output / "helpers" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    (output / "native_worker.py").write_text(WORKER, "utf-8")
    for index, config in enumerate(configs):
        folder = output / f"original-row-{index:05d}"
        folder.mkdir(mode=0o700)
        (folder / "logs").mkdir()
        (folder / "marble").symlink_to(
            output / "author/marble", target_is_directory=True
        )
        (folder / "evaluator").symlink_to(
            output / "author/marble/evaluator", target_is_directory=True
        )
        runtime = deepcopy(config)
        runtime["output"]["file_path"] = str(folder / "original-result.jsonl")
        (folder / "runtime.private.yaml").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2), "utf-8"
        )
    report = {
        "source_id": "MULTIAGENTBENCH",
        "status": "ORIGINAL_RESEARCH_NATIVE_RUNTIME_PREPARED_NOT_EXECUTED",
        "fixed_original_rows": [0, 1, 2],
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
    for index in range(3):
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
        report["model_requests"] = len(
            list((output / "model-queue").glob("*.request.json"))
        )
        (output / "state-private.json").write_text(
            json.dumps(report, indent=2), "utf-8"
        )
    report["status"] = "ORIGINAL_RESEARCH_NATIVE_CLIENT_TERMINAL_AUDIT_PENDING"
    (output / "state-private.json").write_text(json.dumps(report, indent=2), "utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "prepared-root", "bound-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
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
    if args.execute:
        report = execute(args.output.resolve(), report)
    print(json.dumps(report, indent=2))
    return int(
        args.execute
        and any(
            row["cli_exit_code"] != 0 or not row["result_file_exists"]
            for row in report["completed"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
