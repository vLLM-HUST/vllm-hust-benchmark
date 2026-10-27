"""Build pinned original TraceLab replay in an external persistent target.

No data/model download, model call, cache-probe bypass or source modifications.
Supply an installed Rust toolchain; dependencies are resolved by original lock.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PINS = (
    Path(__file__).resolve().parents[1]
    / "docs/workload-data/validation/tracelab-official-replay-preparation-20261024.json"
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_source(root):
    root = Path(root).resolve(strict=True)
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    for item in pins["files"]:
        path = (root / item["path"]).resolve(strict=True)
        if (
            not path.is_relative_to(root)
            or path.stat().st_size != item["bytes"]
            or digest(path) != item["sha256"]
        ):
            raise ValueError("Pinned original source or lock identity mismatch")
    return pins["source_commit"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--cargo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openssl-prefix", type=Path)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.execute and not args.verify_only:
        print(
            json.dumps(
                {
                    "source_id": "TRACELAB",
                    "status": "PLAN_ONLY",
                    "original_lock": True,
                    "persistent_external_target": True,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    source = args.official_source.resolve(strict=True)
    commit = verify_source(source)
    output = args.output.resolve()
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("Build output must be separate from original source")
    binary = output / "target/release/session_runner"
    receipt_path = output / "build-public.json"
    if args.verify_only:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt["status"] != "ORIGINAL_LOCKED_BUILD_COMPLETE_NOT_REPLAY"
            or receipt["source_commit"] != commit
            or digest(binary) != receipt["binary_sha256"]
        ):
            raise ValueError("Missing or changed original native binary")
        print(
            json.dumps(
                {
                    "source_id": "TRACELAB",
                    "status": "ORIGINAL_BINARY_AND_SOURCE_VERIFIED_READ_ONLY",
                    "model_calls": 0,
                }
            )
        )
        return
    cargo = args.cargo.resolve(strict=True)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    environment = dict(
        os.environ,
        CARGO_HOME=str(output / "cargo-home"),
        CARGO_TARGET_DIR=str(output / "target"),
        CARGO_NET_RETRY="0",
        CARGO_HTTP_TIMEOUT="30",
    )
    environment["PATH"] = str(cargo.parent) + os.pathsep + environment.get("PATH", "")
    if args.openssl_prefix:
        prefix = args.openssl_prefix.resolve(strict=True)
        if not (prefix / "include/openssl/ssl.h").is_file():
            raise ValueError(
                "Existing OpenSSL headers required; no system installation"
            )
        environment["OPENSSL_DIR"] = str(prefix)
        environment["LD_LIBRARY_PATH"] = (
            str(prefix / "lib") + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
    command = [
        str(cargo),
        "build",
        "--locked",
        "--release",
        "--jobs",
        str(args.jobs),
        "--manifest-path",
        str(source / "replay/Cargo.toml"),
        "--bin",
        "session_runner",
    ]
    receipt = {
        "source_id": "TRACELAB",
        "status": "BUILD_RUNNING_NOT_REPLAY",
        "pid": os.getpid(),
        "started_at": time.time(),
        "source_commit": commit,
        "command": command,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    save()
    try:
        receipt["cargo_version"] = subprocess.check_output(
            [str(cargo), "--version"], env=environment, text=True
        ).strip()
        receipt["rustc_version"] = subprocess.check_output(
            [str(cargo.parent / "rustc"), "--version"], env=environment, text=True
        ).strip()
        with (output / "build-private.log").open("xb") as log:
            result = subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        receipt["exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError("Original locked build failed; log retained")
        if verify_source(source) != commit:
            raise ValueError("Original source changed during build")
        receipt.update(
            status="ORIGINAL_LOCKED_BUILD_COMPLETE_NOT_REPLAY",
            binary_sha256=digest(binary),
            original_source_unchanged=True,
        )
    except BaseException as error:
        receipt.update(
            status="BUILD_FAILED_RETAINED_NOT_REPLAY",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        receipt["finished_at"] = time.time()
        save()
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
