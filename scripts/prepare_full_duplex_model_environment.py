"""Build the pinned model dependency overlay without changing serving Python.

The two source trees are intentionally not replaced by PyPI packages or fake
editable metadata. This prepares 182 binary/source-distribution dependencies;
source integration and operator payload qualification remain separate gates.
No model weights, neural inference, SDK installer or NPU context is used.
"""

import argparse
import ctypes
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

from verify_full_duplex_model_runtime import load_lock


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4194304), b""):
            result.update(block)
    return result.hexdigest()


def extract_wheels(lock, archive_root, output):
    """Extract only five named/hash-bound wheel members; never run SDK payloads."""
    required = [
        r for r in lock["packages"] if r["binding"] == "PUBLISHER_ARCHIVE_WHEEL"
    ]
    result = {}
    for archive in lock["publisher_archives"]:
        selected = [r for r in required if r["archive"] == archive["name"]]
        if not selected:
            continue
        path = archive_root / archive["name"]
        if path.is_symlink() or not path.is_file() or digest(path) != archive["sha256"]:
            raise ValueError("pinned publisher archive differs: " + archive["name"])
        with zipfile.ZipFile(path) as source:
            if len(source.namelist()) != len(set(source.namelist())):
                raise ValueError("duplicate publisher archive entries")
            for row in selected:
                info = source.getinfo(row["member"])
                expected = next(
                    m for m in archive["members"] if m["path"] == row["member"]
                )
                if (
                    info.file_size != expected["bytes"]
                    or info.file_size > 128 * 1024**2
                ):
                    raise ValueError("wheel member size differs or exceeds bound")
                target = output / row["filename"]
                with source.open(info) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1048576)
                if digest(target) != row["sha256"]:
                    raise ValueError("wheel content differs from pinned member")
                result[row["name"]] = target
    return result


def verify_overlay(lock, env):
    expected = {
        row["name"]: row["version"]
        for row in lock["packages"]
        if row["binding"] != "EDITABLE_SOURCE"
    }
    code = (
        "import importlib.metadata as m,json; names=" + repr(list(expected)) + "; "
        "print(json.dumps({n:{'version':m.version(n),'location':str(m.distribution(n).locate_file(''))} for n in names}))"
    )
    actual = json.loads(subprocess.check_output([str(env / "bin/python"), "-c", code]))
    for name, version in expected.items():
        if actual[name]["version"] != version or not Path(
            actual[name]["location"]
        ).resolve().is_relative_to(env.resolve()):
            raise ValueError("fresh private package version/location differs: " + name)
    code = (
        "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in "
        + repr(list(lock["protected_core_versions"]))
        + "}))"
    )
    if (
        json.loads(subprocess.check_output([str(env / "bin/python"), "-c", code]))
        != lock["protected_core_versions"]
    ):
        raise ValueError("fresh environment protected core differs")
    return len(expected)


def bind_kernel(lock, source, env):
    """Add only the hash-pinned publisher Python supplement to the new overlay."""
    expected = lock["sources"]["kernel-python-supplement"]["sha256"]
    if digest(source) != expected:
        raise ValueError("publisher Python kernel supplement hash differs")
    package = env / "lib/python3.12/site-packages/sgl_kernel_npu"
    if (
        not package.is_dir()
        or package.is_symlink()
        or not package.resolve().is_relative_to(env.resolve())
    ):
        raise ValueError("private installed kernel package required")
    destination = package / "activation/fused_sigmoid_mul.py"
    if destination.exists():
        raise FileExistsError("do not overwrite an existing Python kernel")
    if not destination.parent.resolve().is_relative_to(env.resolve()):
        raise ValueError("kernel supplement target escapes private overlay")
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    if digest(destination) != expected:
        raise ValueError("publisher supplement changed during copy")
    return {
        "sha256": expected,
        "bytes": destination.stat().st_size,
        "publisher_code_modified": False,
        "binary_extension_modified": False,
    }


def isolate_core_payloads(lock, env):
    """Physically copy protected core files, then stop inheriting base packages."""
    destination_root = env / "lib/python3.12/site-packages"
    if not destination_root.is_dir() or destination_root.is_symlink():
        raise ValueError("fresh private site-packages required")
    configuration = env / "pyvenv.cfg"
    text = configuration.read_text()
    anchor = "include-system-site-packages = true"
    if text.count(anchor) != 1:
        raise ValueError("expected inherited fresh overlay before core isolation")
    files = {}
    skipped = []
    for name, expected in lock["protected_core_versions"].items():
        distribution = metadata.distribution(name)
        if distribution.version != expected or distribution.files is None:
            raise ValueError("base core distribution differs or lacks file inventory")
        root = Path(distribution.locate_file("")).resolve()
        for relative in distribution.files:
            if relative.is_absolute() or ".." in relative.parts:
                skipped.append(str(relative))
                continue  # Base command wrappers outside site-packages are not imported payloads.
            source = Path(distribution.locate_file(relative))
            target = destination_root / relative
            if not source.resolve().is_relative_to(root) or not source.is_file():
                raise ValueError(
                    "core source escapes its base distribution or is missing"
                )
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    "do not overwrite private core payload: " + str(relative)
                )
            files[str(relative)] = (source, target)
    size = sum(source.stat().st_size for source, _ in files.values())
    if shutil.disk_usage(destination_root).free < size + 2**30:
        raise ValueError("not enough space for physical core copies plus1GiB reserve")
    for source, target in files.values():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=True)
    configuration.write_text(
        text.replace(anchor, "include-system-site-packages = false")
    )
    return {
        "physical_core_files_copied": len(files),
        "physical_core_bytes_copied": size,
        "base_command_wrappers_not_copied": skipped,
        "system_site_packages_inherited": False,
        "writable_core_payload_symlinks_created": False,
    }


def prepare(lock, archive_root, output, supplement):
    if (
        supplement is None
        or digest(supplement) != lock["sources"]["kernel-python-supplement"]["sha256"]
    ):
        raise ValueError(
            "explicit hash-pinned publisher Python kernel supplement required"
        )
    if platform.machine() != "aarch64" or platform.python_version() != lock["python"]:
        raise ValueError("declared ARM64 Python base required")
    before = {name: metadata.version(name) for name in lock["protected_core_versions"]}
    if before != lock["protected_core_versions"]:
        raise ValueError("protected base core differs; do not upgrade it in place")
    ctypes.CDLL("libhccl.so")
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("fresh environment must remain outside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 5 * 2**30:
        raise ValueError("at least 5GiB free is required before overlay preparation")
    output.mkdir(exist_ok=False)
    report = {
        "status": "STARTED",
        "model_calls": 0,
        "model_weights_downloaded": False,
        "neural_execution": False,
        "source_checkbox_eligible": False,
        "source_metadata_fabricated": False,
        "global_sdk_modified": False,
        "complete_model_runtime_qualified": False,
    }
    try:
        wheels = output / "wheels"
        wheels.mkdir()
        files = extract_wheels(lock, archive_root, wheels)
        requirements = output / "requirements-verified.txt"
        rows = []
        for row in lock["packages"]:
            if row["binding"] == "EDITABLE_SOURCE":
                continue
            url = (
                row["url"]
                if row["binding"] == "PUBLIC_HASHED_ARTIFACT"
                else files[row["name"]].as_uri()
            )
            rows.append(f"{row['name']} @ {url} --hash=sha256:{row['sha256']}\n")
        requirements.write_text("".join(rows), encoding="utf-8")
        env = output / "env"
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(env)],
            check=True,
        )
        command = [
            str(env / "bin/python"),
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
        count = verify_overlay(lock, env)
        report["publisher_kernel_supplement"] = bind_kernel(lock, supplement, env)
        report["core_isolation"] = isolate_core_payloads(lock, env)
        verify_overlay(lock, env)
        if {name: metadata.version(name) for name in before} != before:
            raise RuntimeError("protected caller environment changed")
        hashes = {
            row["name"]: row["sha256"]
            for row in lock["packages"]
            if row["binding"] == "PUBLISHER_ARCHIVE_WHEEL"
        }
        for name, path in files.items():
            if digest(path) != hashes[name]:
                raise ValueError("temporary wheel changed before cleanup")
            path.unlink()
        report.update(
            status="DEPENDENCY_OVERLAY_REBUILT_NOT_MODEL_QUALIFIED",
            private_package_versions_and_locations_verified=count,
            protected_core_versions_unchanged=True,
            inherited_core_payload_symlinks_created=False,
            temporary_extracted_wheels_removed=len(files),
            source_trees_to_bind=[
                r for r in lock["packages"] if r["binding"] == "EDITABLE_SOURCE"
            ],
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
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--publisher-archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-supplement", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--bind-kernel-only", action="store_true")
    mode.add_argument("--isolate-core-only", action="store_true")
    args = parser.parse_args()
    lock = load_lock(args.lock)
    if args.prepare:
        prepare(
            lock,
            args.publisher_archive_root.resolve(strict=True),
            args.output.resolve(),
            args.kernel_supplement,
        )
    elif args.bind_kernel_only:
        if args.kernel_supplement is None:
            parser.error("--bind-kernel-only requires --kernel-supplement")
        env = args.output.resolve() / "env"
        verify_overlay(lock, env)
        print(json.dumps(bind_kernel(lock, args.kernel_supplement, env)))
    elif args.isolate_core_only:
        env = args.output.resolve() / "env"
        verify_overlay(lock, env)
        report = isolate_core_payloads(lock, env)
        verify_overlay(lock, env)
        print(json.dumps(report))
    elif args.verify_only:
        print(
            json.dumps(
                {
                    "status": "OVERLAY_VERSIONS_AND_LOCATIONS_VERIFIED",
                    "private_packages": verify_overlay(
                        lock, args.output.resolve() / "env"
                    ),
                    "neural_execution": False,
                }
            )
        )
    else:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY_NO_DOWNLOAD",
                    "private_packages": 182,
                    "external_source_bindings": 2,
                    "neural_execution": False,
                    "source_checkbox_eligible": False,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
