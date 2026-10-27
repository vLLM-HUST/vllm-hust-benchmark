"""Fixed AgentBench FC ALFWorld tasks through original AgentRL Session/scoring.

Run inside the separately prepared Python-3.9 native environment. Preparation
references come from the Python>=3.10 workload-data component. Private output
contains original task feedback and responses; never publish it or treat a
native environment probe as a model task result.
"""

import argparse
import asyncio
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

REVISION = "d1e4a10db08c87075c78972e48ecc182be03e2d5"  # pragma: allowlist secret - public revision
SOURCE_SHA = "426478d0442a542117d4cf517ccc6c8a04c2bdf44a39983ff2f041ed436051d7"  # pragma: allowlist secret - original source archive checksum
WORKER_SHA = "dd28ad22a079da00a212c13728efce94b40f1d81c98226b48f12cf8ecfc10f38"  # pragma: allowlist secret - official worker wheel checksum
RUNTIME_SHAS = {
    "textworld": "bfbac893fec147297bbb1e07e51adb4b14adcc579db823515e42c801b8e64e19",  # pragma: allowlist secret - official source checksum
    "alfworld": "5e640152e9c2f3fb8534c8918c4f6e0eabe954b13b10635d551620fe89ad9a0a",  # pragma: allowlist secret - official source checksum
}
GAME_SHAS = {
    "json": "25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc",  # pragma: allowlist secret - official release checksum
    "pddl": "913942ebed06659ea0da2f8122512d98bc6add30d84961ca803132d8fbcad585",  # pragma: allowlist secret - official release checksum
    "tw-pddl": "d376c4bbb097c068528a2a9805e358e4cd14bc75c4d68a24e5141aebd234ef67",  # pragma: allowlist secret - official release checksum
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_references(source_archive, game_archives, data_root, references):
    """Original index and release files, not a caller-supplied hash-only trust."""
    if digest(source_archive) != SOURCE_SHA:
        raise ValueError("official AgentBench source archive identity mismatch")
    if not isinstance(references, list) or not references:
        raise ValueError("explicit native task references required")
    with zipfile.ZipFile(source_archive) as bundle:
        names = [
            n for n in bundle.namelist() if n.endswith("/data/alfworld/new_std.json")
        ]
        if len(names) != 1:
            raise ValueError("ambiguous original task index")
        raw = bundle.read(names[0])
    rows = [value for group in json.loads(raw).values() for value in group]
    archives = {}
    for path in game_archives:
        matches = [key for key, value in GAME_SHAS.items() if digest(path) == value]
        if len(matches) != 1 or matches[0] in archives:
            raise ValueError("unknown or repeated game release archive")
        archives[matches[0]] = path
    if set(archives) != set(GAME_SHAS):
        raise ValueError("all original game release archives required")
    indices = []
    base = Path(data_root).resolve()
    for reference in references:
        if any(
            reference.get(key) != value
            for key, value in {
                "schema_version": "native-task-reference/v1",
                "source_id": "AGENTBENCH",
                "revision": REVISION,
                "domain": "alfworld",
                "variant": "FC",
                "task_set": "new_std",
                "runtime_kind": "official_native_task",
                "scoring": "OFFICIAL_RUNNER_REQUIRED",
                "preparation_only_not_execution": True,
            }.items()
        ):
            raise ValueError("native source/version/protocol reference mismatch")
        item = reference["input_ref"]
        index = item["row_index"]
        if (
            type(index) is not int
            or not 0 <= index < len(rows)
            or index in indices
            or reference["native_task_id"] != str(index)
        ):
            raise ValueError("invalid or duplicate original task index")
        relative = rows[index]
        game = (base / relative).resolve()
        if not game.is_relative_to(base) or not relative.endswith("/game.tw-pddl"):
            raise ValueError("unsafe or unsupported original game path")
        if (
            item["file"] != "data/alfworld/new_std.json"
            or item["sha256"] != hashlib.sha256(raw).hexdigest()
            or item["game_file"] != relative
            or item["game_sha256"] != digest(game)
        ):
            raise ValueError("original task index/file identity differs from reference")
        expected = []
        prefix = str(Path(relative).parent).replace("\\", "/") + "/"
        for role, path in archives.items():
            with zipfile.ZipFile(path) as bundle:
                for name in bundle.namelist():
                    if not name.startswith(prefix) or name.endswith("/"):
                        continue
                    target = (base / name).resolve()
                    if not target.is_relative_to(base):
                        raise ValueError("original game asset escapes root")
                    original = bundle.read(name)
                    if target.read_bytes() != original:
                        raise ValueError("original game input differs from release")
                    expected.append(
                        {
                            "file": name,
                            "sha256": hashlib.sha256(original).hexdigest(),
                            "release_role": role,
                        }
                    )
        expected.sort(key=lambda asset: (asset["file"], asset["release_role"]))
        if (
            not expected
            or item["assets"] != expected
            or relative not in {asset["file"] for asset in expected}
        ):
            raise ValueError("original episode asset references differ from release")
        indices.append(index)
    return indices, len(rows)


def verify_runtime(name, archive, expected_sha, package_prefix):
    if digest(archive) != expected_sha:
        raise ValueError("fixed runtime archive identity mismatch")
    with zipfile.ZipFile(archive) as bundle:
        expected = {}
        for name_in_zip in bundle.namelist():
            relative = (
                name_in_zip
                if name == "agentrl-worker"
                else name_in_zip.split("/", 1)[-1]
            )
            if relative.startswith(package_prefix + "/") and relative.endswith(".py"):
                expected[relative] = bundle.read(name_in_zip)
    distribution = metadata.distribution(name)
    count = 0
    for item in distribution.files or []:
        relative = str(item)
        if relative.startswith(package_prefix + "/") and relative.endswith(".py"):
            if (
                relative not in expected
                or distribution.locate_file(item).read_bytes() != expected[relative]
            ):
                raise ValueError(
                    "installed runtime Python source differs from fixed archive: "
                    + relative
                )
            count += 1
    if count == 0 or (name == "agentrl-worker" and distribution.version != "0.4.0"):
        raise ValueError("runtime source coverage/version not established")
    return {"version": distribution.version, "original_python_files_verified": count}


def post(endpoint, body):
    request = Request(
        endpoint + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def native_tasks_scored(tasks):
    """Completion of a selected protocol is not a guarantee of agent success."""
    terminal = {"completed", "task limit reached", "agent invalid action"}
    return len(tasks) >= 3 and all(
        item["status"] in terminal
        and item["reward"] in (0, 1)
        and item["model_calls"] > 0
        for item in tasks
    )


def model_choice(response):
    """Retain an original bounded output, including legal token-budget endings."""
    choices = response.get("choices", [])
    if (
        len(choices) != 1
        or choices[0].get("finish_reason") not in {"stop", "tool_calls", "length"}
        or not isinstance(choices[0].get("message"), dict)
        or choices[0]["message"].get("role") != "assistant"
    ):
        raise ValueError("local model response lacked a supported assistant output")
    return choices[0]


async def execute_original_task(task, index, endpoint, model, directory):
    from agentrl.worker.task import Session
    from agentrl.worker.typings import (
        AgentOutput,
        AgentOutputStatus,
        RewardHistoryItem,
        TaskOutput,
    )

    session = Session(index)
    session.set_tools(task.tools)
    calls = []

    async def episode():
        result = await task.start_sample(index, session)
        output = TaskOutput(index=index, status=result.status, result=result.result)
        await session.controller.env_finish(output)
        return result

    native_job = asyncio.create_task(episode())
    agent_job = asyncio.create_task(session.controller.agent_pull())
    try:
        while True:
            done, _ = await asyncio.wait(
                [native_job, agent_job], return_when=asyncio.FIRST_COMPLETED
            )
            if native_job in done:
                result = native_job.result()
                break
            output = agent_job.result()
            messages = []
            for item in output.history:
                item = getattr(item, "root", item)
                if isinstance(item, RewardHistoryItem):
                    continue  # Framework reward metadata is not a model message.
                if not isinstance(item, dict) or "role" not in item:
                    raise ValueError(
                        "unexpected original Session history representation"
                    )
                messages.append(item)
            body = {
                "model": model,
                "messages": messages,
                "tools": session.tools,
                "temperature": 0,
                "max_tokens": 2048,
            }
            calls.append({"request_attempted": True, "response_received": False})
            (directory / "calls.json").write_text(json.dumps(calls))
            response = await asyncio.to_thread(post, endpoint, body)
            (directory / f"call-{len(calls) - 1}.private.json").write_text(
                json.dumps(response)
            )
            choice = model_choice(response)
            calls[-1].update(
                response_received=True,
                finish_reason=choice["finish_reason"],
                usage=response.get("usage"),
            )
            (directory / "calls.json").write_text(json.dumps(calls))
            message = choice["message"]
            agent_job = asyncio.create_task(
                session.controller.agent_pull(AgentOutput(messages=[message]))
            )
    except BaseException:
        # Deliver the framework's cancellation token on a real execution error,
        # let the official task close its environment; never stop model service.
        if not native_job.done():
            if not agent_job.done():
                agent_job.cancel()
                await asyncio.gather(agent_job, return_exceptions=True)
            cancellation = asyncio.create_task(
                session.controller.agent_pull(
                    AgentOutput(status=AgentOutputStatus.CANCELLED)
                )
            )
            await native_job
            await asyncio.gather(cancellation, return_exceptions=True)
        raise
    finally:
        if not agent_job.done():
            agent_job.cancel()
            await asyncio.gather(agent_job, return_exceptions=True)
    (directory / "result.private.json").write_text(
        json.dumps(
            {
                "result": result.model_dump(mode="json"),
                "history": [
                    getattr(item, "model_dump", lambda **kwargs: item)(mode="json")
                    for item in session.history
                ],
            },
            default=str,
        )
    )
    return result, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--game-archives", type=Path, nargs=3, required=True)
    parser.add_argument(
        "--runtime-source-archives",
        type=Path,
        nargs=2,
        required=True,
        metavar=("TEXTWORLD", "ALFWORLD"),
    )
    parser.add_argument("--worker-wheel", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.task_references = args.task_references.resolve()
    endpoint = args.endpoint.rstrip("/")
    url = urlparse(endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.path != "/v1"
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        parser.error("only local /v1 model endpoints permitted")
    if os.environ.get("PYTHONPATH") or os.environ.get("LD_PRELOAD"):
        parser.error(
            "native task environment must exclude inherited PYTHONPATH/LD_PRELOAD"
        )
    references = json.loads(args.task_references.read_text())
    indices, original_task_count = validate_references(
        args.official_archive, args.game_archives, args.data_root, references
    )
    runtime = {
        name: verify_runtime(name, path, RUNTIME_SHAS[name], name)
        for name, path in zip(("textworld", "alfworld"), args.runtime_source_archives)
    }
    runtime["agentrl-worker"] = verify_runtime(
        "agentrl-worker", args.worker_wheel, WORKER_SHA, "agentrl"
    )
    opencv_version = metadata.version("opencv-python")
    if opencv_version != "4.11.0.86":
        raise ValueError("requires the declared standard OpenCV runtime version")
    try:
        metadata.distribution("opencv-python-headless")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError("standard/headless OpenCV namespace overlap refused")
    runtime["opencv-python"] = {
        "version": opencv_version,
        "headless_namespace_overlap": False,
    }
    os.umask(0o077)
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with zipfile.ZipFile(args.official_archive) as bundle:
        for name in bundle.namelist():
            if not (output / "source" / name).resolve().is_relative_to(output):
                raise ValueError("official source archive path escapes output")
        bundle.extractall(output / "source")
    source = next(path for path in (output / "source").iterdir() if path.is_dir())
    data = output / "native-data"
    shutil.copytree(source / "data/alfworld", data)
    for reference in references:
        for asset in reference["input_ref"]["assets"]:
            destination = data / asset["file"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and digest(destination) != asset["sha256"]:
                raise ValueError("original source/data overlay collision")
            shutil.copyfile(args.data_root / asset["file"], destination)
    sys.path.insert(0, str(source))
    os.chdir(source)
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import yaml
    from agentrl.worker.typings import TaskOutput
    from src.server.tasks.alfworld.task import ALFWorld

    parameters = yaml.safe_load((source / "configs/tasks/alfworld.yaml").read_text())[
        "default"
    ]["parameters"]
    parameters.update(
        data_path=str(data),
        config_path=str(source / "src/server/tasks/alfworld/configs/base_config.yaml"),
        prompts_path=str(
            source
            / "src/server/tasks/alfworld/prompts/alfworld_multiturn_plan_first.json"
        ),
        concurrency=1,
    )
    task = ALFWorld(**parameters)
    if len(task.get_indices()) != original_task_count or any(
        Path(task.data_files[index]) != data / reference["input_ref"]["game_file"]
        for index, reference in zip(indices, references)
    ):
        raise ValueError(
            "official runtime loader order differs from prepared references"
        )
    receipt = {
        "source_id": "AGENTBENCH",
        "revision": REVISION,
        "subsuite": "ALFWorld FC",
        "split": "new_std",
        "selected_indices": indices,
        "original_task_count": original_task_count,
        "status": "STARTED",
        "started_at_unix": time.time(),
        "runtime_original_source_checks": runtime,
        "helper_sha256": digest(__file__),
        "task_references_sha256": digest(args.task_references),
        "model": args.model,
        "official_max_steps": task.max_step,
        "generation_config": {"temperature": 0, "max_tokens": 2048},
        "deployment_changes": [
            "local NPU model route",
            "declared per-call generation configuration",
            "serial scheduling",
            "PDDL-only ARM packaging",
            "standard OpenCV with host/isolated libGL runtime",
            "Python protobuf compatibility codec",
        ],
        "task_prompt_action_and_scoring_code_changed": False,
        "official_container_equivalence_claim": False,
        "full_suite_claim": False,
        "performance_claim": False,
        "tasks": [],
    }
    target = output / "receipt.json"
    results = []
    try:
        target.write_text(json.dumps(receipt, indent=2))
        for index in indices:
            private = output / ("task-" + str(index))
            private.mkdir(mode=0o700)
            result, calls = asyncio.run(
                execute_original_task(task, index, endpoint, args.model, private)
            )
            results.append(
                TaskOutput(index=index, status=result.status, result=result.result)
            )
            receipt["tasks"].append(
                {
                    "index": index,
                    "status": result.status.value,
                    "reward": (result.result or {}).get("result"),
                    "native_environment_steps": len(
                        (result.result or {}).get("log", [])
                    ),
                    "model_calls": len(calls),
                    "private_result_sha256": digest(private / "result.private.json"),
                }
            )
            target.write_text(json.dumps(receipt, indent=2))
        receipt["official_overall"] = task.calculate_overall(results)
        for reference in references:
            for asset in reference["input_ref"]["assets"]:
                if digest(data / asset["file"]) != asset["sha256"]:
                    raise ValueError(
                        "native runtime changed an original episode input file"
                    )
        receipt["original_episode_input_files_unchanged"] = True
        receipt["source_checkbox_eligible"] = native_tasks_scored(receipt["tasks"])
        if any(
            item["status"] in {"task error", "unknown", "cancelled"}
            for item in receipt["tasks"]
        ):
            raise RuntimeError(
                "native task error/cancellation is not successful execution"
            )
        receipt["status"] = "ORIGINAL_NATIVE_TASKS_EXECUTED_AND_SCORED"
    except BaseException as error:
        receipt.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        receipt["model_request_attempts"] = {
            path.parent.name: len(json.loads(path.read_text()))
            for path in output.glob("task-*/calls.json")
        }
        target.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
