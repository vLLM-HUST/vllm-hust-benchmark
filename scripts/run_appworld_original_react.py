"""Original AppWorld ReAct entry; plan by default, execute only in isolation.

Does not install dependencies, download data, start Docker or run on the host.
The operator supplies an isolated Agent container and separate environment API.
"""

import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import socket
from urllib.parse import urlsplit

PROFILE_SHA256 = "e68a9852ccb20098f4729c87abade366e0ce01d79798bb3b2e5ee627bf077d37"  # pragma: allowlist secret


def validate_profile(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != PROFILE_SHA256:
        raise ValueError("unchanged pinned original test_normal profile required")


def require_isolation():
    if os.name != "posix" or os.geteuid() == 0:
        raise ValueError("nonroot Linux isolated container required")
    if not Path("/.dockerenv").is_file():
        raise ValueError("this entry requires an explicit Docker container")
    # Query the current network namespace; nested Docker sysfs may show the
    # outer namespace even when this process has network=none.
    if {name for _, name in socket.if_nameindex()} != {"lo"}:
        raise ValueError(
            "network-none container required; use private socket transport"
        )
    if not os.statvfs("/").f_flag & os.ST_RDONLY:
        raise ValueError("read-only container root required")
    if any(Path(p).exists() for p in ("/var/run/docker.sock", "/run/docker.sock")):
        raise ValueError("Docker socket must not be exposed")
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    if int(status["CapEff"].strip(), 16) != 0 or status["NoNewPrivs"].strip() != "1":
        raise ValueError("drop all capabilities and set no-new-privileges")


def run(args):
    report = {
        "source_id": "APPWORLD",
        "status": "PLAN_ONLY",
        "selection": "first N original reader examples, no task replacement",
        "limit": args.limit,
        "model": args.model,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if args.limit < 1 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.experiment
    ):
        raise ValueError("positive limit and simple experiment name required")
    if not args.execute:
        return report
    require_isolation()
    validate_profile(args.profile)
    endpoint = urlsplit(args.api_base)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
        or endpoint.username
        or endpoint.password
    ):
        raise ValueError("private loopback model proxy endpoint required")
    if not args.environment_url.startswith("http+unix://"):
        raise ValueError(
            "separate original environment over private Unix socket required"
        )
    # Runtime imports are intentionally absent from plan/CI paths.
    import litellm
    import requests_unixsocket
    from appworld import AppWorld
    from appworld.common.path_store import path_store
    from appworld.common.utils import jsonnet_load, write_json
    from experiments.code.recoma.run import (
        dump_predictions,
        extract_prediction_and_data,
    )
    from recoma.run_inference import build_configurable_systems_from_json

    config = copy.deepcopy(jsonnet_load(args.profile)["config"])
    config["models"]["action"]["generator_params"]["model"] = args.model
    # Endpoint/key are provider configuration, not extra GeneratorParams.
    litellm.api_base = args.api_base
    litellm.api_key = os.environ[args.api_key_env]
    AppWorld.init_defaults.timeout_seconds = 60
    AppWorld.init_defaults.random_seed = 123
    AppWorld.init_defaults.experiment_name = args.experiment
    AppWorld.init_defaults.remote_environment_url = args.environment_url
    output = Path(path_store.experiment_outputs) / args.experiment / "recoma_output"
    output.mkdir(parents=True, exist_ok=False)
    write_json(config, str(output / "bound_config.private.json"), silent=True)
    predictions = []
    with requests_unixsocket.monkeypatch():
        systems = build_configurable_systems_from_json(config, str(output))
        for example in itertools.islice(systems.reader.get_examples(), args.limit):
            example.experiment_name = args.experiment
            prediction = systems.search.predict(example)
            predictions.append(prediction)
            _, data = extract_prediction_and_data(prediction)
            write_json(
                data, str(output / f"{example.unique_id}_data.json"), silent=True
            )
            dump_predictions(predictions, str(output))
    if len(predictions) != args.limit:
        raise ValueError("original reader exhausted before selected sample count")
    report.update(
        status="ORIGINAL_AGENT_RETURNED_AUDIT_PENDING", completed=len(predictions)
    )
    report.pop("model_calls")  # Actual native logs, not an invented count.
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--model", required=True, help="actual LiteLLM model/provider identifier"
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--environment-url", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
