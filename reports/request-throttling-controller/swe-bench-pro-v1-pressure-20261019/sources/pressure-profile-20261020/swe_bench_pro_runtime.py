"""RIDE vllm shim: enter the unchanged frozen HUST launcher, then official serve CLI."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

from run_swe_bench_pro import REPOSITORY, read


def engine_arguments(engine):
    arguments = []
    for key, value in engine.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if type(value) is bool:
            arguments.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, (dict, list)):
            arguments.extend([flag, json.dumps(value)])
        else:
            arguments.extend([flag, str(value)])
    return arguments



def shared_profile_module():
    # The frozen bootstrap may already cache a different shared_test_profile module.
    # Load this repository's registered selector by absolute source identity.
    source = REPOSITORY / "tools/shared_test_profile.py"
    spec = importlib.util.spec_from_file_location("swe_bench_shared_test_profile", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--inside-frozen-runtime", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    c = read(args.settings)
    runtime = c["model_deployment"]
    root = Path(c["paths"]["runtime_home"])
    if not args.inside_frozen_runtime:
        command = ["python3", str(root / "installer/tools/gennova-7/healthy_runtime_entry.py"),
                   "run", "--root", str(root / "source"),
                   "--sdk-python", str(root / "sdk/venv-school-r1/bin/python"),
                   "--cann-env", runtime["cann_environment"],
                   "--atb-env", runtime["atb_environment"],
                   "--manager-config", runtime["manager_config"],
                   "--receipt-dir", runtime["receipt_directory"],
                   "--entry", str(Path(__file__).resolve()), "--",
                   "--settings", args.settings, "--inside-frozen-runtime"]
        # healthy_runtime_entry appends consumer paths after frozen HUST sources.
        command[command.index("--manager-config"):command.index("--manager-config")] = [
            "--consumer-path", str(Path(__file__).resolve().parent)]
        for plugin in runtime["plugins"]:
            command[command.index("--entry"):command.index("--entry")] = ["--plugin", plugin]
        os.execv(sys.executable, [sys.executable, *command[1:]])
    selector = shared_profile_module()
    shared_profile = runtime.get("shared_profile", "baseline")
    profile = selector.load(shared_profile)
    packages = {k: importlib.metadata.version(k) for k in profile["packages"]}
    engine = dict(profile["engine"], model=runtime["model_path"])
    proof = selector.validate_engine(engine, packages, shared_profile)
    proof.update(engine=engine, packages=packages,
                 source_manifest_sha256=hashlib.sha256((root / "source/source-manifest.json").read_bytes()).hexdigest())
    receipt = Path(runtime["receipt_directory"])
    (receipt / "engine-configuration.json").write_text(json.dumps(proof, indent=2))
    # model is a positional serve argument, never a second independently chosen profile.
    cli_engine = {k: v for k, v in engine.items() if k != "model"}
    sys.argv = ["vllm", "serve", engine["model"], "--host", "0.0.0.0",
                "--port", str(runtime["port"]), "--served-model-name", "Qwen3.5-35B-A3B",
                *engine_arguments(cli_engine)]
    (receipt / "serve-command.json").write_text(json.dumps(sys.argv, indent=2))
    print("FROZEN_HUST_SERVE", json.dumps(sys.argv), flush=True)
    from vllm.entrypoints.cli.main import main as official_main
    official_main()


if __name__ == "__main__":
    main()
