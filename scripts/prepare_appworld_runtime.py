"""Prepare an isolated official AppWorld core client, never execute Agent tasks.

Plan-only by default. Protected-code unpacking is a separate explicit option;
all files must remain in the external private environment, never in Git.
No task download, serving-stack mutation, model load or model request.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

VERSION = "0.1.3.post1"
INDEX = "https://pypi.org/simple"


def commands(uv, python, output, install_protected_code=False, constraints=None):
    client = Path(output) / "client"
    executable = client / "bin/python"
    cli = client / "bin/appworld"
    steps = [
        [
            str(uv),
            "venv",
            "--python",
            str(python),
            "--seed",
            "--default-index",
            INDEX,
            str(client),
        ],
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(executable),
            "--index-url",
            INDEX,
            "--link-mode",
            "copy",
            f"appworld=={VERSION}",
        ],
    ]
    if constraints is not None:
        steps[1].extend(["--constraint", str(Path(constraints).resolve())])
    if install_protected_code:
        steps.append([str(cli), "install"])
    steps.extend(
        [
            [str(uv), "pip", "check", "--python", str(executable)],
            [str(cli), "--help"],
        ]
    )
    return steps


def isolated_python(output):
    client = Path(output) / "client"
    config = dict(
        line.lower().split("=", 1)
        for line in (client / "pyvenv.cfg").read_text().splitlines()
        if "=" in line
    )
    config = {key.strip(): value.strip() for key, value in config.items()}
    if config.get("include-system-site-packages") != "false":
        raise ValueError("system-site inheritance is forbidden")
    return client / "bin/python"


def check_python_runtime(python):
    code = "import json,pathlib,platform,sys,sysconfig; print(json.dumps({'version':list(sys.version_info[:3]),'machine':platform.machine(),'headers_present':(pathlib.Path(sysconfig.get_path('include'))/'Python.h').is_file()}))"
    completed = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    if result["version"][:2] != [3, 11]:
        raise ValueError("explicit Python3.11 required for this tested profile")
    if result["machine"] in {"aarch64", "arm64"} and not result["headers_present"]:
        raise ValueError(
            "ARM source dependencies require a complete Python3.11 SDK with Python.h; do not install into the serving stack"
        )
    return result


def prepare(
    output,
    uv,
    python,
    execute=False,
    install_protected_code=False,
    uv_cache=None,
    constraints=None,
):
    output = Path(output).resolve()
    steps = commands(uv, python, output, install_protected_code, constraints)
    result = {
        "source_id": "APPWORLD",
        "status": "PLAN_ONLY",
        "version": VERSION,
        "steps": steps,
        "model_calls": 0,
        "task_data_acquired": False,
        "benchmark_executed": False,
        "source_checkbox_eligible": False,
        "protected_code_unpack_requested": install_protected_code,
        "dependency_constraints": str(Path(constraints).resolve())
        if constraints
        else None,
        "boundary": "Official core package preparation only, not isolation security qualification, Agent execution or author lock reproduction.",
    }
    if not execute:
        return result
    if constraints is not None and not Path(constraints).is_file():
        raise ValueError("explicit dependency constraints file does not exist")
    repository = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.is_relative_to(repository)
        or not output.parent.is_dir()
    ):
        raise ValueError(
            "fresh external private directory with existing parent required"
        )
    if sys.platform != "linux":
        raise ValueError(
            "only Linux core-client preparation is qualified by this profile"
        )
    for executable in (uv, python):
        if not Path(executable).is_file():
            raise ValueError("explicit existing uv/Python executables required")
    result["python_preflight"] = check_python_runtime(python)
    cache = Path(uv_cache).resolve() if uv_cache is not None else output / "cache/uv"
    if uv_cache is not None and (
        not cache.is_dir() or cache.is_relative_to(repository)
    ):
        raise ValueError("explicit existing external software cache required")
    result["software_cache"] = str(cache)
    output.mkdir(mode=0o700)
    for name in ("tmp", "cache", "world"):
        (output / name).mkdir(mode=0o700)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "UV_CACHE_DIR": str(cache),
            "UV_DEFAULT_INDEX": INDEX,
            "PIP_CACHE_DIR": str(output / "cache/pip"),
            "XDG_CACHE_HOME": str(output / "cache/xdg"),
            "TMPDIR": str(output / "tmp"),
            "APPWORLD_ROOT": str(output / "world"),
            "APPWORLD_CACHE": str(output / "cache/appworld"),
            "UV_CONCURRENT_DOWNLOADS": "2",
            "UV_CONCURRENT_BUILDS": "1",
        }
    )
    result.update(status="PREPARING_OFFICIAL_CORE_CLIENT", completed_steps=0)

    def save():
        (output / "preparation-public.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    save()
    try:
        with (output / "installation-private.log").open("x") as log:
            for index, command in enumerate(steps):
                subprocess.run(
                    command,
                    env=env,
                    cwd=output / "world",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1200,
                )
                result["completed_steps"] = index + 1
                if index == 0:
                    isolated_python(output)
                save()
            executable = isolated_python(output)
            check = "import importlib.metadata as m,json,sys; assert sys.version_info[:2]==(3,11); assert m.version('appworld')==sys.argv[1]; print(json.dumps({'python':sys.version.split()[0],'appworld':m.version('appworld'),'prefix':sys.prefix,'installed_distributions':len(list(m.distributions()))}))"
            completed = subprocess.run(
                [str(executable), "-c", check, VERSION],
                env=env,
                cwd=output / "world",
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            result["verified_environment"] = json.loads(completed.stdout)
            freeze = subprocess.run(
                [str(uv), "pip", "freeze", "--python", str(executable)],
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            (output / "environment-freeze-private.txt").write_text(freeze.stdout)
        result.update(
            status="OFFICIAL_CORE_CLIENT_AND_CLI_VERIFIED_NO_TASK_EXECUTION",
            protected_code_installed=install_protected_code,
        )
    except BaseException as error:
        result.update(
            status="FAILED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--install-protected-code", action="store_true")
    parser.add_argument(
        "--constraints",
        type=Path,
        help="Optional observed dependency constraints, not an author lock",
    )
    parser.add_argument(
        "--uv-cache",
        type=Path,
        help="Reuse an explicitly owned external software cache; never use serving-model caches",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.output,
                args.uv,
                args.python,
                args.execute,
                args.install_protected_code,
                args.uv_cache,
                args.constraints,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
