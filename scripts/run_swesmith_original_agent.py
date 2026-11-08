"""Original fixed-three SWE-agent run, with explicit owned-NPU/runtime bindings.

Original source, templates, task records and call/tool budgets are preserved.
The native RemoteDeployment backend receives private sandbox connection data;
raw tasks are never rewritten to change image_name or invent legacy fields.
"""

import argparse
import asyncio
import hashlib
from importlib.util import find_spec
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
import runpy
import sys
import tarfile
from urllib.parse import urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from appworld_model_transport import model_proxy
from recover_swesmith_fixed3_repository import COMMITS
from run_swesmith_arm_evaluation import fixed_original_rows

AGENT_ARCHIVE_SHA256 = "a02724577cde3c7033343efff74184ad33ca783b9f3b81bea90c13f4c47219a4"  # pragma: allowlist secret - fixed public author source archive
CONFIG_SHA256 = "678478618344a2aef726ba529345e37dd37640b089ea197b46d98b43ec9195e0"  # pragma: allowlist secret - unchanged public author config


def select_task_ids(requested):
    if requested is None:
        return list(COMMITS)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("nonempty unique frozen task IDs required")
    if set(requested) - set(COMMITS):
        raise ValueError("only frozen original task IDs may be retried")
    return [identity for identity in COMMITS if identity in requested]


async def remote_health_with_timeout(original_health, deployment, timeout=None):
    """Bridge SWE-agent's timeout keyword to SWE-ReX 1.4's native health call."""
    if timeout is None:
        return await original_health(deployment)
    if timeout <= 0:
        raise ValueError("positive remote health timeout required")
    return await asyncio.wait_for(original_health(deployment), timeout=timeout)


def validate_bindings(bindings):
    if set(bindings) != set(COMMITS):
        raise ValueError("exactly the three frozen task runtime bindings required")
    for value in bindings.values():
        url = urlsplit(value["host"])
        address = ip_address(url.hostname)
        if (
            url.scheme != "http"
            or not address.is_private
            or address.is_unspecified
            or address.is_multicast
            or url.username
            or url.password
            or url.port
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError("explicit private sandbox HTTP address required")
        if type(value["port"]) is not int or not 1 <= value["port"] <= 65535:
            raise ValueError("valid sandbox port required")
        if not isinstance(value["auth_token"], str) or len(value["auth_token"]) < 20:
            raise ValueError("private authenticated runtime required")


def verify_agent_source(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != AGENT_ARCHIVE_SHA256:
        raise ValueError("fixed original SWE-agent archive required")
    root = Path(find_spec("sweagent").origin).parent
    names = [
        "agent/agents.py",
        "agent/models.py",
        "environment/swe_env.py",
        "environment/repo.py",
        "run/run_batch.py",
        "run/batch_instances.py",
    ]
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for name in names:
            selected = [m for m in members if m.name.endswith("/sweagent/" + name)]
            if len(selected) != 1 or not selected[0].isfile():
                raise ValueError("unique original SWE-agent module required")
            if tar.extractfile(selected[0]).read() != (root / name).read_bytes():
                raise ValueError("original SWE-agent source differs: " + name)
    return root.parent


def run(args):
    os.umask(0o077)
    if args.output.exists():
        raise ValueError("fresh output required; inspect original attempts first")
    if hashlib.sha256(args.original_config.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("unchanged original SWE-smith GPT configuration required")
    source_root = verify_agent_source(args.agent_archive)
    bindings = json.loads(args.runtime_bindings.read_text())
    validate_bindings(bindings)
    rows = fixed_original_rows(args.original_shard)
    selected_ids = select_task_ids(args.task_id)
    registry = json.loads(args.model_registry.read_text())
    model_name = "openai/" + args.model
    entry = registry[model_name]
    if any(
        entry.get(key) != value
        for key, value in {
            "max_input_tokens": 128000,
            "max_output_tokens": 16384,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }.items()
    ):
        raise ValueError(
            "declared local API-fee/reference-token budget binding required"
        )
    if not args.queue.is_dir():
        raise ValueError("dedicated existing private model queue required")
    args.output.mkdir(mode=0o700, parents=True)
    tasks = args.output / "original-tasks.private.json"
    tasks.write_text(json.dumps(rows))
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["LITELLM_TELEMETRY"] = "False"
    import litellm
    from sweagent.run.batch_instances import SWESmithInstances
    from swerex.deployment.config import RemoteDeploymentConfig
    from swerex.deployment.remote import RemoteDeployment

    litellm.telemetry = False
    original_loader = SWESmithInstances.get_instance_configs
    original_health = RemoteDeployment.is_alive

    async def bound_health(deployment, *, timeout=None):
        return await remote_health_with_timeout(original_health, deployment, timeout)

    def bound_instances(instance_source):
        instances = original_loader(instance_source)
        if [item.problem_statement.id for item in instances] != selected_ids:
            raise ValueError("native loader changed the frozen original task set/order")
        for item in instances:
            item.env.deployment = RemoteDeploymentConfig(
                **bindings[item.problem_statement.id]
            )
        return instances

    original_dir, original_argv = Path.cwd(), sys.argv
    SWESmithInstances.get_instance_configs = bound_instances
    RemoteDeployment.is_alive = bound_health
    try:
        os.chdir(source_root)
        with model_proxy(args.queue, port=14001, timeout=600):
            sys.argv = [
                "sweagent",
                "run-batch",
                "--config",
                str(args.original_config),
                "--instances.path",
                str(tasks),
                "--instances.shuffle=False",
                "--instances.filter",
                "^(?:" + "|".join(re.escape(i) for i in selected_ids) + ")$",
                "--num_workers",
                "1",
                "--output_dir",
                str(args.output / "native-output"),
                "--agent.model.name",
                model_name,
                "--agent.model.api_base",
                "http://127.0.0.1:14001/v1",
                "--agent.model.api_key",
                "local-unused-no-external-credential",
                "--agent.model.litellm_model_registry",
                str(args.model_registry),
                "--agent.model.max_input_tokens",
                "128000",
                "--agent.model.max_output_tokens",
                "16384",
            ]
            runpy.run_module("sweagent.run.run", run_name="__main__")
    finally:
        SWESmithInstances.get_instance_configs = original_loader
        RemoteDeployment.is_alive = original_health
        sys.argv = original_argv
        os.chdir(original_dir)
        (args.output / "binding-boundary-public.json").write_text(
            json.dumps(
                {
                    "status": "ORIGINAL_RUNNER_ATTEMPT_REQUIRES_NATIVE_TERMINAL_AND_SCORING_AUDIT",
                    "actual_model": args.model,
                    "task_ids": list(COMMITS),
                    "selected_task_ids": selected_ids,
                    "original_records_modified": False,
                    "original_template_modified": False,
                    "original_config_call_limit": 75,
                    "original_config_cost_limit": 2.0,
                    "local_external_api_token_fee": 0,
                    "cost_boundary": "Zero external API fee is not server compute cost or a comparable hosted-model financial budget.",
                    "reference_input_limit": 128000,
                    "reference_output_limit": 16384,
                    "runtime_binding": "Original RemoteDeployment; no original executor/tool source patch",
                    "health_api_compatibility": "SWE-agent passes timeout to SWE-ReX1.4 RemoteDeployment.is_alive, which lacks that keyword. Controller-only adapter applies asyncio.wait_for to the original health call; original source restored/unmodified, health failures preserved.",
                    "source_checkbox_eligible": False,
                },
                indent=2,
            )
        )
    verify_agent_source(args.agent_archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "original-shard",
        "original-config",
        "runtime-bindings",
        "model-registry",
        "queue",
        "agent-archive",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--task-id",
        action="append",
        choices=list(COMMITS),
        help="Explicit incomplete frozen task to retry; repeat to select several. Default: all three.",
    )
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    run(args)


if __name__ == "__main__":
    main()
