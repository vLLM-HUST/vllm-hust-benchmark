"""Execute original GAIA validation tasks with a declared smolagents profile.

The operator supplies a reviewed tool factory. References/annotator plans never
enter that factory or Agent. Default invocation only prints the protocol plan.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
from itertools import islice
import json
import os
from pathlib import Path
import shutil
from urllib.parse import urlsplit

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.assets import asset_path, verify_file


def selected_tasks(data_root, start, count):
    if type(start) is not int or start < 0 or type(count) is not int or count <= 0:
        raise ValueError("explicit nonnegative start and positive count required")
    records = iter_workload(
        "GAIA", data_root=Path(data_root), files=["2023/validation/metadata.parquet"]
    )
    tasks = [deepcopy(r.task) for r in islice(records, start, start + count)]
    if len(tasks) != count:
        raise ValueError("requested original selection exceeds validation split")
    return tasks


def load_factory(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("reviewed operator tool factory digest differs")
    spec = importlib.util.spec_from_loader("gaia_reviewed_operator_tools", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    if not callable(getattr(module, "make_tools", None)):
        raise ValueError(
            "reviewed factory must expose make_tools(task_input, task_dir)"
        )
    return module.make_tools


def run(
    *,
    data_root,
    output,
    factory_file,
    factory_sha256,
    endpoint,
    model,
    start,
    count,
    max_steps,
    max_output_tokens,
    disable_thinking=False,
    model_timeout=600,
):
    target = urlsplit(endpoint)
    if (
        target.scheme != "http"
        or target.hostname not in {"127.0.0.1", "localhost", "::1"}
        or target.username
        or target.password
    ):
        raise ValueError("explicit loopback model transport required")
    if any(type(x) is not int or x <= 0 for x in [max_steps, max_output_tokens]):
        raise ValueError("declare positive Agent step/output budgets before running")
    if type(model_timeout) is not int or not 1 <= model_timeout <= 86400:
        raise ValueError("declare model timeout in seconds within1..86400")
    output = Path(output)
    data_root = Path(data_root).resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh private Agent output required; no automatic rerun")
    resolved = output.resolve()
    if (
        resolved.is_relative_to(Path(__file__).resolve().parents[1])
        or resolved.is_relative_to(data_root)
        or data_root.is_relative_to(resolved)
    ):
        raise ValueError("private output outside checkout and original assets required")
    tasks = selected_tasks(data_root, start, count)
    if importlib.metadata.version("smolagents") != "1.26.0":
        raise ValueError("qualified smolagents1.26.0 required")
    from smolagents import OpenAIModel, ToolCallingAgent

    factory = load_factory(factory_file, factory_sha256)
    output.mkdir(mode=0o700, parents=True)
    receipt = {
        "source_id": "GAIA",
        "status": "RUNNING_ORIGINAL_VALIDATION_AGENT",
        "model": model,
        "endpoint": endpoint,
        "start": start,
        "count": count,
        "framework": "smolagents1.26.0",
        "agent": "ToolCallingAgent",
        "max_steps": max_steps,
        "max_output_tokens": max_output_tokens,
        "disable_thinking": disable_thinking,
        "model_timeout_seconds": model_timeout,
        "tool_factory_sha256": factory_sha256,
        "operator_tool_environment": {
            key: os.environ.get(key)
            for key in [
                "GAIA_SEARCH_ENGINE",
                "GAIA_DOCKER_HOST",
                "GAIA_PYTHON_IMAGE",
                "GAIA_FRAMEWORK_VENV",
            ]
        },
        "selected_original_ids": [t["task_id"] for t in tasks],
        "attempted_original_ids": [],
        "returned_original_ids": [],
        "failed_original_ids": [],
        "source_checkbox_eligible": False,
        "scoring": "NOT_RUN",
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    with (output / "predictions-private.jsonl").open(
        "x", encoding="utf-8"
    ) as predictions:
        for task in tasks:
            identity = task["task_id"]
            directory = output / ("task-" + str(len(receipt["attempted_original_ids"])))
            directory.mkdir(mode=0o700)
            inputs = deepcopy(task["input"])
            if inputs["attachment_asset"]:
                asset = inputs["attachment_asset"]
                verify_file(data_root, asset)
                attachment = directory / "attachments" / inputs["file_name"]
                attachment.parent.mkdir()
                shutil.copyfile(asset_path(data_root, asset["path"]), attachment)
                inputs["local_attachment_path"] = str(attachment)
            # No outcome_record(), reference answer or annotator metadata is supplied.
            (directory / "inputs-private.json").write_text(
                json.dumps(inputs), encoding="utf-8"
            )
            receipt["attempted_original_ids"].append(identity)
            save()
            prediction = None
            agent = None
            with (
                (directory / "stdout-private.txt").open(
                    "x", encoding="utf-8"
                ) as stdout,
                (directory / "stderr-private.txt").open(
                    "x", encoding="utf-8"
                ) as stderr,
            ):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    try:
                        tools = factory(deepcopy(inputs), directory)
                        if not isinstance(tools, list) or not tools:
                            raise ValueError("reviewed nonempty tool profile required")
                        backend = OpenAIModel(
                            model_id=model,
                            api_base=endpoint,
                            api_key="local-only",  # pragma: allowlist secret - public noncredential loopback placeholder
                            client_kwargs={"max_retries": 0, "timeout": model_timeout},
                            max_tokens=max_output_tokens,
                            **(
                                {
                                    "extra_body": {
                                        "chat_template_kwargs": {
                                            "enable_thinking": False
                                        }
                                    }
                                }
                                if disable_thinking
                                else {}
                            ),
                        )
                        agent = ToolCallingAgent(
                            model=backend,
                            tools=tools,
                            max_steps=max_steps,
                            max_tool_threads=1,
                        )
                        extra = (
                            {"attachment_path": inputs["local_attachment_path"]}
                            if inputs.get("local_attachment_path")
                            else None
                        )
                        answer = agent.run(inputs["question"], additional_args=extra)
                        prediction = None if answer is None else str(answer)
                        receipt["returned_original_ids"].append(identity)
                    except Exception as error:
                        (directory / "failure-private.json").write_text(
                            json.dumps({"error_type": type(error).__name__}),
                            encoding="utf-8",
                        )
                        receipt["failed_original_ids"].append(identity)
                    finally:
                        if agent is not None:
                            (directory / "agent-steps-private.json").write_text(
                                json.dumps(agent.memory.get_full_steps(), default=str),
                                encoding="utf-8",
                            )
            predictions.write(
                json.dumps({"task_id": identity, "prediction": prediction}) + "\n"
            )
            predictions.flush()
            save()
    receipt["status"] = "ORIGINAL_AGENT_ATTEMPTS_RETAINED_SCORING_REQUIRED"
    save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    for name in ["data-root", "output", "factory-file"]:
        parser.add_argument("--" + name, type=Path)
    for name in ["factory-sha256", "endpoint", "model"]:
        parser.add_argument("--" + name)
    for name, default in [
        ("start", 0),
        ("count", 3),
        ("max-steps", 20),
        ("max-output-tokens", 4096),
        ("model-timeout", 600),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    args = vars(parser.parse_args())
    if not args.pop("execute"):
        print(
            json.dumps(
                {
                    "source_id": "GAIA",
                    "status": "PLAN_ONLY",
                    "framework": "smolagents1.26.0",
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    if any(value is None for value in args.values()):
        parser.error(
            "execution requires explicit original root, private output, reviewed factory digest and model endpoint"
        )
    result = run(**args)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["source_id", "status", "count", "source_checkbox_eligible"]
            }
        )
    )


if __name__ == "__main__":
    main()
