"""Run unchanged author ScienceWorld ReAct episodes against a local NPU API.

The author GPT-4 tokenizer/context/50-token action protocol is retained.
Only API routing/model identity (and an explicitly selected thinking profile)
may differ. This is a fallback integration run, not author-model reproduction.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from run_docvqa_original import idle, local_endpoint

AUTHOR_REVISION = "b8ea624231a6a8d83d78c1c12bea1b44b8ea758b"  # pragma: allowlist secret - public publisher Git identity
PINNED_FILES = [
    {
        "path": "baselines/ReAct_baseline/prompt.jsonl",
        "sha": "90fd70fd0a1960ccf57dce5cc577283bcc4d0a51",  # pragma: allowlist secret - public publisher Git identity
        "size": 147757,
    },
    {
        "path": "baselines/eval_agent_react.py",
        "sha": "709a3aa651661c3c5600c88663cd1d590a4809ba",  # pragma: allowlist secret - public publisher Git identity
        "size": 9536,
    },
    {
        "path": "data_utils/data_utils.py",
        "sha": "09db8efb695e2a0fcef18a74e6464128cbdecfa6",  # pragma: allowlist secret - public publisher Git identity
        "size": 23792,
    },
    {
        "path": "eval_utils.py",
        "sha": "5bdbb1d846c758190a7e93b3e137bfa467779b55",  # pragma: allowlist secret - public publisher Git identity
        "size": 45310,
    },
    {
        "path": "slow_agent/local_llm.py",
        "sha": "6e1a40737588376661d058895f55b12f4ab74bc9",  # pragma: allowlist secret - public publisher Git identity
        "size": 14688,
    },
    {
        "path": "slow_agent/utils.py",
        "sha": "c6137d3bb217e27fc33700e5227603cd054e854a",  # pragma: allowlist secret - public publisher Git identity
        "size": 534,
    },
]


def verify_source(root):
    root = Path(root).resolve()
    for item in PINNED_FILES:
        path = root / item["path"]
        if path.is_symlink():
            raise ValueError("symlink is not original source")
        data = path.read_bytes()
        sha = hashlib.sha1(f"blob {len(data)}".encode() + bytes([0]) + data).hexdigest()
        if len(data) != item["size"] or sha != item["sha"]:
            raise ValueError("author source identity mismatch: " + item["path"])
    return root


def native_args(output, source, dataset_set="test_mini"):
    return {
        "jar_path": "",
        "env_step_limit": 100,
        "simplification_str": "easy",
        "max_episode_per_file": 9999,
        "set": dataset_set,
        "cut_off": False,
        "output_path": str(output),
        "no_stop": True,
        "prompt_file": str(source / "baselines/ReAct_baseline/prompt.jsonl"),
        "model_name": "gpt-4",
    }


def task_indices(value):
    tasks = [int(item) for item in value.split(",")]
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(t not in range(30) for t in tasks)
    ):
        raise ValueError("use unique original task indices0..29")
    return tasks


def execute(args):
    endpoint = local_endpoint(args.endpoint)
    source = verify_source(args.author_source_root)
    output = Path(args.output).resolve()
    if (
        output.exists()
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from author source required")
    tasks = task_indices(args.task_nums)
    if not idle(endpoint, args.api_server_pid, args.npu_engine_pid):
        raise RuntimeError(
            "research/device priority guard: service not exclusively idle"
        )
    output.mkdir(parents=True, mode=0o700)
    os.environ["OPENAI_API_KEY"] = (
        "local-native-client"  # pragma: allowlist secret - loopback placeholder, not a credential
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XDG_CACHE_HOME"] = str(output / "cache")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(Path(args.tokenizer_cache).resolve())
    sys.path.insert(0, str(source))
    import scienceworld
    import openai

    if scienceworld.__version__ != "1.1.3":
        raise ValueError("author scienceworld1.1.3 required")
    spec = importlib.util.spec_from_file_location(
        "author_scienceworld_react", source / "baselines/eval_agent_react.py"
    )
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    original_create = openai.ChatCompletion.create
    openai.api_base = endpoint
    result = {
        "source_id": "SCIENCEWORLD",
        "author_revision": AUTHOR_REVISION,
        "status": "RUNNING_AUTHOR_REACT",
        "source_checkbox_eligible": False,
        "model": args.model,
        "author_context_profile": "gpt-4",
        "disable_thinking": args.disable_thinking,
        "model_calls": 0,
        "task_indices": tasks,
        "author_set": args.dataset_set,
        "completed_task_indices": [],
        "scope": "original first3 test variations per selected author task; not full corpus or author GPT4 reproduction",
    }

    def save():
        (output / "native-public.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    def transport(**kwargs):
        if kwargs["max_tokens"] != 50 or kwargs["temperature"] != 0:
            raise ValueError("original action budget changed")
        if not idle(endpoint, args.api_server_pid, args.npu_engine_pid):
            raise RuntimeError("research priority guard no longer idle")
        kwargs["model"] = args.model
        kwargs["request_timeout"] = 120
        if args.disable_thinking:
            kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        result["model_calls"] += 1
        save()
        return original_create(**kwargs)

    openai.ChatCompletion.create = transport
    save()
    try:
        for task in tasks:
            config = native_args(output / f"task-{task}", source, args.dataset_set)
            logger = native.init_logger(config, str(task))
            native.eval(config, task, logger)
            result["completed_task_indices"].append(task)
            save()
        result["status"] = (
            "AUTHOR_SELECTED_TASK_LOOPS_RETURNED_SAVED_SCORING_AUDIT_PENDING"
        )
    except Exception as exc:
        result.update(status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__)
        raise
    finally:
        openai.ChatCompletion.create = original_create
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-nums", default="3,4,5")
    parser.add_argument(
        "--set",
        dest="dataset_set",
        choices=["train", "dev", "test", "test_mini", "test_mini_2", "test_mini_mini"],
        default="test_mini",
        help="Original author split/profile. Use test for the full original test split, with cut_off=False; test_mini is only the validation sample profile.",
    )
    parser.add_argument("--api-server-pid", type=int, required=True)
    parser.add_argument("--npu-engine-pid", type=int, required=True)
    parser.add_argument("--tokenizer-cache", required=True)
    parser.add_argument("--disable-thinking", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
