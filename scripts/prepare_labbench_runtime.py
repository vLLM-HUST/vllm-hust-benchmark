"""Prepare/verify an isolated frozen-lock official LAB-Bench client environment.

Default is plan-only. No inference, model download or serving-stack mutation.
The verified profile is Linux ARM64 Python3.12; all model calls belong to the
separate native runner and must target an NPU endpoint.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_labbench_original import verify_source

SETUPTOOLS_COMPAT = "setuptools==75.1.0"


def install_compat(uv, environment):
    python = isolated_python(environment)
    subprocess.run(
        [
            str(uv),
            "pip",
            "install",
            "--no-deps",
            "--python",
            str(python),
            SETUPTOOLS_COMPAT,
        ],
        check=True,
    )


def isolated_python(environment):
    environment = Path(environment).resolve()
    config = environment / "pyvenv.cfg"
    if not config.is_file():
        raise ValueError("explicit isolated venv required")
    options = dict(
        line.lower().split("=", 1)
        for line in config.read_text().splitlines()
        if "=" in line
    )
    options = {key.strip(): value.strip() for key, value in options.items()}
    if options.get("include-system-site-packages") != "false":
        raise ValueError("system-site-packages must be disabled")
    python = environment / "bin/python"
    if not python.is_file():
        raise ValueError("Linux client venv Python missing")
    return python


def install_command(uv, python, source):
    return [
        str(uv),
        "sync",
        "--frozen",
        "--no-dev",
        "--python",
        str(python),
        "--project",
        str(source),
    ]


def verify_environment(environment, source, uv):
    python = isolated_python(environment)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PUBLIC_RELEASE"] = "True"
    code = """
import importlib.metadata as m,json,pathlib,platform,sys,tomllib
source=pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(source))
import labbench,torch
assert pathlib.Path(labbench.__file__).resolve()==source/'labbench/__init__.py'
assert sys.version_info[:2]==(3,12) and platform.machine()=='aarch64'
assert torch.version.cuda is None and not torch.cuda.is_available()
lock=tomllib.loads((source/'uv.lock').read_text())
normalize=lambda s:s.lower().replace('_','-').replace('.','-')
allowed={}
for package in lock['package']:
    allowed.setdefault(normalize(package['name']),set()).add(package['version'])
installed={normalize(d.metadata['Name']):d.version for d in m.distributions()}
mismatch={n:v for n,v in installed.items() if n in allowed and v not in allowed[n]}
assert not mismatch,mismatch
print(json.dumps({'status':'OFFICIAL_LOCKED_CLIENT_IMPORT_VERIFIED_NOT_INFERENCE','installed_distributions':len(installed),'locked_version_mismatches':len(mismatch),'torch_cuda_available':False,'model_calls':0}))
"""
    completed = subprocess.run(
        [str(python), "-c", code, str(source)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        completed.check_returncode()
    result = json.loads(completed.stdout)
    subprocess.run(
        [str(uv), "pip", "check", "--python", str(python)], env=env, check=True
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument(
        "--install-compat",
        action="store_true",
        help="Explicitly add the pinned ARM torch setuptools dependency to an existing isolated venv; does not change the author lock",
    )
    args = parser.parse_args()
    command = install_command(args.uv, args.python, args.source_root)
    if not args.execute and not args.verify_only and not args.install_compat:
        print(
            json.dumps(
                {
                    "source_id": "LABBENCH",
                    "status": "PLAN_ONLY",
                    "command": command,
                    "environment": str(args.environment),
                    "model_calls": 0,
                }
            )
        )
        return
    verify_source(args.source_root, args.archive)
    environment = args.environment.resolve()
    if environment.is_relative_to(args.source_root.resolve()):
        raise ValueError("venv must be separate from unchanged publisher source")
    if args.execute:
        if environment.exists():
            raise ValueError("fresh environment required; use verify-only to reuse")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["UV_PROJECT_ENVIRONMENT"] = str(environment)
        env["UV_PYTHON_DOWNLOADS"] = "never"
        subprocess.run(command, env=env, check=True)
    if args.install_compat:
        install_compat(args.uv, environment)
    result = verify_environment(environment, args.source_root, args.uv)
    verify_source(args.source_root, args.archive)
    print(
        json.dumps(
            {"source_id": "LABBENCH", **result, "source_checkbox_eligible": False}
        )
    )


if __name__ == "__main__":
    main()
