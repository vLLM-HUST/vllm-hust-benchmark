"""Build original CPU video-import dependencies for the native HTTP client.

No model inference or system-package changes. Source archives reproduce fixed
upstream Git trees. Builds use a private prefix and unmodified author commands.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from prepare_chartqa_native_toolkit import acquire_archive

# Public immutable upstream object IDs, not credentials.
PINS = {
    "ffmpeg": (
        "FFmpeg/FFmpeg",
        "76ee3e41df5157317ee209d3e6cc0829af5f963f",  # pragma: allowlist secret
        "778fd46ac63eb7d931125c0ded37d07f73ce68aa",  # pragma: allowlist secret
    ),
    "decord": (
        "dmlc/decord",
        "6a3617cef035535193f390f86399dde139fa2a53",  # pragma: allowlist secret
        "603de81af7126164f2933dd289a3d7544b134228",  # pragma: allowlist secret
    ),
    "dlpack": (
        "dmlc/dlpack",
        "5c792cef3aee54ad8b7000111c9dc1797f327b59",  # pragma: allowlist secret
        "de2b675ac043f49b01de586044691ed8fef5b3d9",  # pragma: allowlist secret
    ),
    "dmlc-core": (
        "dmlc/dmlc-core",
        "d07fb7a443b5db8a89d65a15a024af6a425615a5",  # pragma: allowlist secret
        "8d23ef2785238aaddc12ed23906feb4178ad941d",  # pragma: allowlist secret
    ),
}


def client_environment():
    env = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_USER",
        "PIP_CONFIG_FILE",
    ):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def qualify_client_python(path):
    # Do not resolve interpreter symlinks: CPython finds pyvenv.cfg from the
    # invoked venv entry, not the resolved base executable.
    python = Path(os.path.abspath(path.expanduser()))
    if not python.is_file():
        raise FileNotFoundError("Client interpreter does not exist")
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import json,sys; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'executable':sys.executable}))",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=client_environment(),
    )
    identity = json.loads(result.stdout)
    prefix = Path(identity["prefix"]).resolve(strict=True)
    if prefix == Path(identity["base_prefix"]).resolve(strict=True):
        raise ValueError(
            "A pure client virtualenv is required; system Python installation is forbidden"
        )
    cfg = prefix / "pyvenv.cfg"
    values = {}
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip().lower()] = value.strip().lower()
    if values.get("include-system-site-packages") != "false":
        raise ValueError("Client virtualenv must not inherit system site-packages")
    return python, prefix


def run(root, python, jobs):
    if not 1 <= jobs <= 16:
        raise ValueError("CPU build jobs must be between 1 and 16")
    root, python = root.resolve(), Path(os.path.abspath(python.expanduser()))
    if root.exists() or python.is_relative_to(root):
        raise ValueError(
            "Fresh private output separate from the client interpreter required"
        )
    python, client_prefix = qualify_client_python(python)
    subprocess.run(
        [str(python), "-m", "pip", "--isolated", "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=client_environment(),
    )
    if any(
        shutil.which(command) is None
        for command in ("git", "gcc", "g++", "make", "cmake", "pkg-config")
    ):
        raise ValueError(
            "Install qualified C/C++/Git/CMake/pkg-config tools outside this script first"
        )
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "cpu_jobs": jobs,
        "started": time.time(),
        "model_calls": 0,
        "system_packages_changed": False,
        "client_interpreter_entry": str(python),
        "client_prefix": str(client_prefix),
        "pins": PINS,
    }

    def save():
        (root / "preparation-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        for name in ("ffmpeg", "decord"):
            target = root / name
            target.mkdir()
            repo, revision, tree = PINS[name]
            links = (
                (
                    ("3rdparty/dlpack", PINS["dlpack"][1]),
                    ("3rdparty/dmlc-core", PINS["dmlc-core"][1]),
                )
                if name == "decord"
                else ()
            )
            acquire_archive(
                target, repository=repo, revision=revision, tree=tree, gitlinks=links
            )
        for name in ("dlpack", "dmlc-core"):
            repo, revision, tree = PINS[name]
            acquire_archive(
                root / "decord/3rdparty" / name,
                repository=repo,
                revision=revision,
                tree=tree,
            )
        receipt.update(
            all_four_original_source_trees_verified=True, step="build_ffmpeg"
        )
        save()
        prefix, ff_build, dec_build = (
            root / "install",
            root / "ffmpeg-build",
            root / "decord-build",
        )
        ff_build.mkdir()
        dec_build.mkdir()
        env = client_environment()
        env.update(
            TORCH_DEVICE_BACKEND_AUTOLOAD="0", DECORD_LIBRARY_PATH=str(dec_build)
        )
        env["LD_LIBRARY_PATH"] = (
            str(prefix / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        )

        def command(args, cwd, timeout=1200):
            with (root / "build.private.log").open("ab") as log:
                subprocess.run(
                    args,
                    cwd=cwd,
                    env=env,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )

        command(
            [
                str(root / "ffmpeg/configure"),
                f"--prefix={prefix}",
                "--enable-shared",
                "--disable-static",
                "--disable-doc",
                "--disable-programs",
                "--disable-autodetect",
                "--disable-asm",
                "--enable-avdevice",
            ],
            ff_build,
            180,
        )
        command(["make", f"-j{jobs}"], ff_build)
        command(["make", "install"], ff_build, 180)
        if not (prefix / "lib/libavdevice.so").exists():
            raise ValueError(
                "Original linker requires the real FFmpeg avdevice library"
            )
        receipt["step"] = "build_decord"
        save()
        command(
            [
                "cmake",
                str(root / "decord"),
                "-DUSE_CUDA=0",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                f"-DFFMPEG_DIR={prefix}",
            ],
            dec_build,
            180,
        )
        command(["make", f"-j{jobs}"], dec_build)
        command(
            [
                str(python),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--prefix",
                str(client_prefix),
                "--no-cache-dir",
                "--no-deps",
                "--no-build-isolation",
                str(root / "decord/python"),
            ],
            dec_build,
            180,
        )
        command(
            [
                str(python),
                "-c",
                "import decord,sys; from pathlib import Path; "
                "assert decord.__version__ == '0.6.0'; "
                "assert Path(decord.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
                "print(decord.__version__, decord.__file__)",
            ],
            dec_build,
            60,
        )
        command([str(python), "-m", "pip", "--isolated", "check"], dec_build, 60)
        receipt.update(
            status="ORIGINAL_RUNTIME_BUILT_IMPORTED_AND_PIP_CHECKED_NOT_BENCHMARK",
            exit_code=0,
            ffmpeg_library_dir=str(prefix / "lib"),
        )
    except BaseException as error:
        receipt.update(
            status="FAILED_RETAINED", exit_code=1, error_type=type(error).__name__
        )
        raise
    finally:
        receipt["finished"] = time.time()
        save()
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        required=True,
        help="Qualified pure HTTP-client virtualenv interpreter",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16:
        raise ValueError("CPU build jobs must be between 1 and 16")
    if args.execute:
        run(args.output, args.python, args.jobs)
    else:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY_NO_BUILD_OR_MODEL_CALL",
                    "pins": PINS,
                    "cpu_jobs": args.jobs,
                    "model_calls": 0,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
