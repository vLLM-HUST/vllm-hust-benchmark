"""Rebuild the audited ASR overlay in a fresh venv, never mutate serving Python.

Requires the declared site-local ARM64 base and unchanged core packages.
Default validates the lock and prints a plan; --prepare installs exact hashed
artifacts without dependency resolution, then checks imports without loading
models. Source-distribution builds use the declared base build tools.
"""

import argparse
import ctypes
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import urlsplit


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def load_lock(path):
    lock = json.loads(Path(path).read_text("utf-8"))
    if lock.get("schema_version") != "full-duplex-asr-overlay-lock/v1":
        raise ValueError("audited full-duplex ASR overlay lock required")
    packages = lock.get("private_packages", [])
    protected = {normalize(name) for name in lock["protected_core_versions"]}
    if protected != {
        "torch",
        "torch-npu",
        "torchaudio",
        "torchvision",
        "numpy",
        "triton-ascend",
    }:
        raise ValueError("all six protected base core versions must be declared")
    names = set()
    if not packages or lock.get("missing_wheel_bindings"):
        raise ValueError("complete artifact bindings required")
    for row in packages:
        name, version = row["name"], row["version"]
        url = urlsplit(row["url"])
        if (
            re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
            or name in names
            or name in protected
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+!-]*", version) is None
            or re.fullmatch(r"[a-f0-9]{64}", row["sha256"]) is None
            or url.scheme != "https"
            or url.hostname != "files.pythonhosted.org"
            or url.netloc != "files.pythonhosted.org"
            or any(c.isspace() or ord(c) < 32 for c in row["url"])
            or url.username
            or url.password
            or url.query
            or url.fragment
            or not url.path.endswith((".whl", ".tar.gz", ".zip"))
        ):
            raise ValueError("invalid, duplicate or protected package binding")
        names.add(name)
    return lock


def core_versions(lock):
    return {name: metadata.version(name) for name in lock["protected_core_versions"]}


def verify_overlay(lock, env):
    expected = {row["name"]: row["version"] for row in lock["private_packages"]}
    code = (
        "import importlib.metadata as m,json; names=" + repr(list(expected)) + "; "
        "print(json.dumps({n:{'version':m.version(n),'location':str(m.distribution(n).locate_file(''))} for n in names}))"
    )
    raw = subprocess.check_output([str(env / "bin/python"), "-c", code])
    installed = json.loads(raw)
    for name, version in expected.items():
        if installed[name]["version"] != version or not Path(
            installed[name]["location"]
        ).resolve().is_relative_to(env.resolve()):
            raise ValueError("private overlay version/location differs: " + name)
    return len(expected)


def verify_existing(lock, output):
    env = output / "env"
    if core_versions(lock) != lock["protected_core_versions"]:
        raise ValueError("protected core versions differ")
    count = verify_overlay(lock, env)
    code = (
        "import importlib.metadata as m,json; import nemo.collections.asr; "
        "names=" + repr(list(lock["protected_core_versions"])) + "; "
        "print(json.dumps({n:m.version(n) for n in names}))"
    )
    raw = subprocess.check_output([str(env / "bin/python"), "-c", code])
    if (
        json.loads(raw.decode("utf-8").splitlines()[-1])
        != lock["protected_core_versions"]
    ):
        raise ValueError("imported environment core versions differ")
    return {
        "status": "EXISTING_ASR_OVERLAY_VERIFIED_NO_NEURAL_EXECUTION",
        "private_versions_and_locations_verified": count,
        "protected_core_versions": core_versions(lock),
        "original_install_receipt_preserved": True,
        "model_calls": 0,
        "downloads": 0,
        "source_checkbox_eligible": False,
    }


def prepare(lock, output):
    if platform.machine() != "aarch64" or platform.python_version() != lock["python"]:
        raise ValueError("the declared ARM64 Python version is required")
    if core_versions(lock) != lock["protected_core_versions"]:
        raise ValueError(
            "protected base core versions differ; do not upgrade them in place"
        )
    for name, version in lock.get("build_tools", {}).items():
        if metadata.version(name) != version:
            raise ValueError("declared base build tool differs: " + name)
    # Fail before downloading if the caller has not activated the CANN runtime.
    ctypes.CDLL("libhccl.so")
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("use a fresh external environment directory")
    output.mkdir(parents=True, exist_ok=False)
    requirements = output / "requirements-verified.txt"
    requirements.write_text(
        "".join(
            f"{row['name']} @ {row['url']} --hash=sha256:{row['sha256']}\n"
            for row in lock["private_packages"]
        ),
        encoding="utf-8",
    )
    env = output / "env"
    report = {
        "status": "STARTED",
        "overlay_packages": len(lock["private_packages"]),
        "protected_core_versions": core_versions(lock),
        "model_calls": 0,
        "model_weights_downloaded": False,
        "source_checkbox_eligible": False,
    }
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(env)],
            check=True,
        )
        python = env / "bin/python"
        command = [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--no-deps",
            "--ignore-installed",
            "--require-hashes",
            "--no-build-isolation",
            "--report",
            str(output / "install-report.json"),
            "-r",
            str(requirements),
        ]
        with (output / "install.log").open("xb") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        code = (
            "import importlib.metadata as m,json; import nemo.collections.asr; "
            "names=" + repr(list(lock["protected_core_versions"])) + "; "
            "print(json.dumps({n:m.version(n) for n in names}))"
        )
        with (output / "import.stderr.log").open("xb") as log:
            raw = subprocess.check_output([str(python), "-c", code], stderr=log)
        (output / "import.stdout.log").write_bytes(raw)
        after = json.loads(raw.decode("utf-8").splitlines()[-1])
        report["private_overlay_versions_and_locations_verified"] = verify_overlay(
            lock, env
        )
        if after != lock["protected_core_versions"] or core_versions(lock) != after:
            raise RuntimeError("protected core changed during preparation")
        report.update(
            status="OVERLAY_REBUILT_AND_ASR_IMPORTED_NO_NEURAL_EXECUTION",
            core_versions_unchanged=True,
            neural_execution=False,
        )
    except Exception as error:
        report.update(
            status="FAILED", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        (output / "qualification-public.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock = load_lock(args.lock)
    if args.prepare:
        prepare(lock, args.output.resolve())
    elif args.verify_only:
        print(json.dumps(verify_existing(lock, args.output.resolve())))
    else:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY_NO_DOWNLOAD",
                    "packages": len(lock["private_packages"]),
                    "base_image_digest": lock["base_image_digest"],
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
