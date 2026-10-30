"""Run original scientific Agent tasks, not zero-shot QA; plan by default.

Requires separately prepared original capsules, the pinned author client and
image, private workspace, and an owned idle model window. No downloader,
service restart, accelerator allocation or grading is performed here.
"""

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sha256:b345de18d0c70e3ec454b391d30285684448c2b56523a581ecc17c3ff06b4a9e"  # pragma: allowlist secret


def notebook_policy(config, workspace):
    """Constrain only task-container deployment; preserve original notebook code."""
    bound = config.get("HostConfig", {}).get("Binds", [])
    if (
        config.get("Image") != IMAGE
        or len(bound) != 1
        or set(config.get("HostConfig", {})) != {"Binds"}
    ):
        raise ValueError(
            "pinned official image and one original workspace bind required"
        )
    source, separator, target = bound[0].rpartition(":")
    if (
        separator != ":"
        or target != "/workspace"
        or not Path(source).resolve().is_relative_to(workspace)
    ):
        raise ValueError("task workspace bind escapes private root")
    if config.get("Env"):
        raise ValueError("unexpected environment in original task-container request")
    result = copy.deepcopy(config)
    result.update(User="65534:65534", Env=["HOME=/tmp", "MPLCONFIGDIR=/tmp/matplotlib"])
    result["HostConfig"].update(
        NetworkMode="none",
        ReadonlyRootfs=True,
        CapDrop=["ALL"],
        SecurityOpt=["no-new-privileges"],
        Privileged=False,
        Memory=4 * 1024**3,
        NanoCpus=10**9,
        PidsLimit=256,
        Tmpfs={"/tmp": "rw,nosuid,nodev,size=1g"},
    )
    return result


def validate_options(args):
    timeout = getattr(args, "request_timeout", None)
    if timeout is not None and not 60 <= timeout <= 1800:
        raise ValueError("transport timeout must be between60 and1800 seconds")
    endpoint = urlsplit(args.api_base)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
        or endpoint.username
        or endpoint.password
        or endpoint.path.rstrip("/") != "/v1"
        or endpoint.port is None
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("explicit owned loopback OpenAI endpoint required")
    if not args.model.startswith("openai/") or not args.model.removeprefix("openai/"):
        raise ValueError("explicit OpenAI-compatible local model binding required")
    if not 1 <= args.limit <= 205 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name
    ):
        raise ValueError("positive original prefix and simple run name required")
    return endpoint


async def execute(args, report):
    import yaml
    from aiodocker import Docker
    from aiodocker.containers import DockerContainers

    from prepare_bixbench_original_tasks import CATALOG, verify_asset
    from prepare_jsonschemabench_author_source import verify_source

    author, data, prepared, output, workspace = [
        Path(path).resolve()
        for path in (
            args.author_root,
            args.data_root,
            args.prepared,
            args.output,
            args.workspace,
        )
    ]
    for destination in (output, workspace):
        if (
            destination.exists()
            or not destination.parent.is_dir()
            or destination.is_relative_to(ROOT)
        ):
            raise ValueError("fresh external output and workspace required")
        if any(
            destination.is_relative_to(source) or source.is_relative_to(destination)
            for source in (author, data, prepared)
        ):
            raise ValueError(
                "native output must not overlap original or prepared assets"
            )
    if output.is_relative_to(workspace) or workspace.is_relative_to(output):
        raise ValueError("separate private output and task workspace required")
    pins = ROOT / "configs/workload-native/bixbench-author-source.json"
    verify_source(author, pins)
    entries = {item["path"]: item for item in json.loads(CATALOG.read_text())["files"]}
    metadata = verify_asset(data, entries["BixBench.jsonl"])
    records = [json.loads(line) for line in metadata.read_text().splitlines()]
    if len(records) != 205 or len({row["question_id"] for row in records}) != 205:
        raise ValueError("original205 unique task rows required")
    selected = records[: args.limit]
    preparation = json.loads((prepared / "preparation-public.json").read_text())
    prepared_files = preparation.get("prepared_files")
    if not prepared_files or not preparation.get(
        "original_inputs_and_author_source_unchanged"
    ):
        raise ValueError("public preparation with file identities required")
    actual_files = {
        path.relative_to(prepared / "processed").as_posix(): path
        for path in (prepared / "processed").rglob("*")
        if path.is_file()
    }
    if set(actual_files) != set(prepared_files):
        raise ValueError("prepared file collection changed")
    for name, path in actual_files.items():
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(prepared / "processed")
            or path.stat().st_size != prepared_files[name]["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != prepared_files[name]["sha256"]
        ):
            raise ValueError("prepared capsule identity changed")
    for row in selected:
        verify_asset(data, entries[row["data_folder"]])
        folder = prepared / "processed" / row["data_folder"].removesuffix(".zip")
        if (
            not folder.is_dir()
            or not any(folder.iterdir())
            or list(folder.rglob("*.ipynb"))
        ):
            raise ValueError(
                "original preprocessed capsule missing or reference notebook present"
            )
    # Imports and environment binding happen only after source/asset checks.
    os.environ.update(
        USE_DOCKER="true",
        DOCKER_HOST=args.docker_host,
        NB_ENVIRONMENT_DOCKER_IMAGE=IMAGE,
        OPENAI_API_BASE=args.api_base,
        OPENAI_API_KEY=os.environ[args.api_key_env],
    )
    sys.path.insert(0, str(author))
    from bixbench.generate_trajectories import TrajectoryGenerator

    docker = Docker()
    image = await docker.images.inspect(IMAGE)
    if image["Id"] != IMAGE or image["Architecture"] != "arm64":
        raise ValueError("qualified original ARM64 image not present")
    output.mkdir(mode=0o700)
    workspace.mkdir(mode=0o700)
    os.chown(workspace, 65534, 65534)
    os.chmod(workspace, 0o2770)
    os.umask(0o007)
    config = yaml.safe_load(
        (author / "bixbench/run_configuration/generate_trajectories.yaml").read_text()
    )
    config["run_name"] = args.run_name
    config["agent"]["agent_kwargs"]["llm_model"]["name"] = args.model
    if args.request_timeout is not None:
        config["agent"]["agent_kwargs"]["llm_model"].setdefault("router_kwargs", {})[
            "timeout"
        ] = args.request_timeout
    config["paths"].update(
        workspace_dir=str(workspace),
        trajectories_dir=str(output / "trajectories"),
        data_folder=str(prepared / "processed"),
    )
    config_path = output / "runtime-binding.private.yaml"
    config_path.write_text(yaml.safe_dump(config))
    generator = TrajectoryGenerator(config_path)
    original_create = DockerContainers.create
    created = []
    active_container = None

    async def create(self, config, *, name=None):
        nonlocal active_container
        container = await original_create(
            self, notebook_policy(config, workspace), name=name
        )
        active_container = container.id
        created.append(container.id)
        info = await container.show()
        if (
            info["Config"]["User"] != "65534:65534"
            or info["HostConfig"]["NetworkMode"] != "none"
            or not info["HostConfig"]["ReadonlyRootfs"]
        ):
            raise ValueError("actual notebook container failed deployment checks")
        return container

    DockerContainers.create = create
    report.pop("model_calls")
    report.update(
        status="RUNNING_ORIGINAL_SCIENTIFIC_AGENT",
        completed=[],
        original_max_steps=generator.config.rollout.max_steps,
        actual_actor_model=args.model,
        author_actor_reproduction=False,
        official_scoring_executed=False,
        configured_rollout_type=generator.config.rollout.rollout_type,
        actual_publisher_execution_api="TrajectoryGenerator.vanilla_rollout",
        default_aviary_batch_backend_reproduction=False,
        transport_timeout_override_seconds=args.request_timeout,
    )

    def save():
        (output / "state.private.json").write_text(json.dumps(report, indent=2))

    save()
    try:
        for original in selected:
            report["active_question"] = original["question_id"]
            save()
            question = copy.deepcopy(original)
            question["local_data_folder"] = (
                prepared / "processed" / original["data_folder"].removesuffix(".zip")
            )
            env = generator.environment_factory(question)
            for path in [env.work_dir, *env.work_dir.rglob("*")]:
                if path.is_symlink() or not path.resolve().is_relative_to(workspace):
                    raise ValueError("task runtime path escape; retain failure")
                os.chown(path, 65534, 65534)
                os.chmod(path, 0o2770 if path.is_dir() else 0o660)
            trajectory, env = await generator.vanilla_rollout(
                generator.config.agent_config.construct_agent(), env
            )
            await generator.store_trajectory(trajectory, env)
            report["completed"].append(original["question_id"])
            save()
            await env.state.close()
            await env.state.docker_client.close()
            active_container = None
        verify_source(author, pins)
        report.update(
            status="ORIGINAL_SELECTED_AGENT_TRAJECTORIES_SAVED_SCORING_PENDING",
            task_containers_created=len(created),
        )
    except BaseException as error:
        report.update(
            status="STOP_RETAINED_NATIVE_AGENT_NEEDS_TRIAGE",
            error_type=type(error).__name__,
        )
        # Preserve failed task containers and workspaces, but do not leave an
        # idle notebook process consuming resources after the client exits.
        if active_container:
            try:
                failed = await docker.containers.get(active_container)
                details = await failed.show()
                if details["State"]["Running"]:
                    await failed.stop(t=10)
                report["failed_container_stopped_and_preserved"] = True
            except Exception as cleanup_error:
                report["failed_container_cleanup_error_type"] = type(
                    cleanup_error
                ).__name__
        raise
    finally:
        save()
        DockerContainers.create = original_create
        await docker.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "data-root", "prepared", "output", "workspace"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--docker-host", required=True)
    parser.add_argument("--model-queue", type=Path)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--run-name", default="original-local-actor")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    endpoint = validate_options(args)
    report = {
        "source_id": "BIXBENCH",
        "status": "PLAN_ONLY",
        "limit": args.limit,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if args.execute:
        if (
            sys.platform != "linux"
            or sys.version_info < (3, 12)
            or not args.docker_host.startswith("unix:///")
        ):
            raise ValueError(
                "qualified Linux Python3.12+ and explicit local Docker socket required"
            )
        if args.model_queue and endpoint.hostname != "127.0.0.1":
            raise ValueError("private file proxy requires explicit127.0.0.1 binding")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from appworld_model_transport import model_proxy
        import asyncio

        transport = (
            model_proxy(args.model_queue, port=endpoint.port)
            if args.model_queue
            else nullcontext()
        )
        with transport:
            report = asyncio.run(execute(args, report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
