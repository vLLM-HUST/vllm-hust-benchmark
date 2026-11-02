"""Replay paired quality prompts through off, approximate, native APC and rollback.

Run with the qualified serving Python after sourcing CANN/ATB. All package
changes are confined to a new virtual environment; each service uses the same
explicitly selected device. This is a small diagnostic, not a formal benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
PACKAGE = "vllm_ascend_simllm"
IDENTITY = """
import hashlib,importlib.metadata,json,pathlib,sys
import vllm_ascend_simllm as package
root=pathlib.Path(package.__file__).parent
print(json.dumps({
    'prefix':sys.prefix, 'package_root':str(root),
    'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*.py'))},
    'versions':{n:importlib.metadata.version(n) for n in
                ('vllm','vllm-ascend','torch','torch-npu','vllm-ascend-simllm')}
}))
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_environment(original: dict[str, str]) -> dict[str, str]:
    """Keep CANN/TBE imports while preventing a source tree shadowing the wheel."""
    result = {
        key: value
        for key, value in original.items()
        if not key.startswith("VLLM_ASCEND_SIMLLM_")
        and key not in {"PYTHONPATH", "VLLM_PLUGINS"}
    }
    paths = []
    for entry in original.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        root = Path(entry)
        if (root / PACKAGE).exists() or (root / (PACKAGE + ".py")).exists():
            continue
        paths.append(entry)
    if paths:
        result["PYTHONPATH"] = os.pathsep.join(paths)
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def wheel_files(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        prefix = PACKAGE + "/"
        return {
            name.removeprefix(prefix): hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith(prefix) and name.endswith(".py")
        }


def assert_port_free(port: int) -> None:
    with socket.socket() as connection:
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        connection.bind(("127.0.0.1", port))


def read_outputs(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [row for row in records if row.get("phase") != "summary"]


def compare_outputs(baseline: list[dict], restored: list[dict]) -> dict:
    if not baseline or len(baseline) != len(restored):
        raise AssertionError("Missing or unequal rollback outputs")
    differences = []
    for before, after in zip(baseline, restored, strict=True):
        if any(
            before.get(key) != after.get(key)
            for key in ("id", "prompt_tokens", "completion_tokens", "output_text")
        ):
            differences.append(before["id"])
    return {"exactly_equal": not differences, "different_ids": differences}


def validate_stage(
    stage: Path,
    rows: list[dict],
    wheel: Path,
    enabled: bool,
    native_apc: bool,
    chunked_prefill: bool,
) -> dict:
    installed = json.loads((stage / "installed.json").read_text())
    if installed["files"] != wheel_files(wheel):
        raise AssertionError("Recorded installed files do not match the wheel")
    launch = json.loads((stage / "launch.json").read_text())
    if launch["settings"]["VLLM_ASCEND_SIMLLM_ENABLED"] != str(int(enabled)):
        raise AssertionError("Recorded activation flag does not match the stage")
    expected_apc_flag = (
        "--enable-prefix-caching" if native_apc else "--no-enable-prefix-caching"
    )
    if expected_apc_flag not in launch["command"]:
        raise AssertionError("Recorded native APC setting does not match the stage")
    expected_chunked_flag = (
        "--enable-chunked-prefill" if chunked_prefill else "--no-enable-chunked-prefill"
    )
    if expected_chunked_flag not in launch["command"]:
        raise AssertionError(
            "Recorded chunked prefill setting does not match the stage"
        )
    results = read_outputs(stage / "outputs.jsonl")
    if [row["id"] for row in results] != [row["id"] for row in rows] or any(
        not row.get("output_text")
        or not row.get("prompt_tokens")
        or row.get("completion_tokens", 0) <= 0
        for row in results
    ):
        raise AssertionError(f"{stage.name} has missing or empty inference results")
    text = (stage / "server.log").read_text(errors="replace")
    reuse = re.findall(r"SimLLM: reusing (\d+) semantic tokens", text)
    banner = "Sim-LLM patch applied" in text
    stored = "SimLLM extract_kv: stored" in text
    # The host may configure the root handler after activation. A runtime reuse
    # event proves this patch executed even when the startup INFO banner is lost.
    if enabled and not reuse:
        raise AssertionError("Final wheel did not exercise semantic KV reuse")
    if not enabled and (banner or stored or reuse):
        raise AssertionError(f"{stage.name} unexpectedly activated SimLLM")
    failures = re.findall(r"SimLLM[^\n]*(?:failed|failure)", text)
    if failures:
        raise AssertionError(f"{stage.name} plugin failures: {failures}")
    return {
        "name": stage.name,
        "requests": len(results),
        "wheel_sha256": digest(wheel),
        "enabled": enabled,
        "native_apc": native_apc,
        "chunked_prefill": chunked_prefill,
        "semantic_reuse_events": len(reuse),
        "semantic_reused_tokens": sum(map(int, reuse)),
        "startup_banner_observed": banner,
        "patch_exercised": bool(stored or reuse),
        "server_stopped": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-wheel", required=True, type=Path)
    parser.add_argument("--rollback-wheel", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--port", type=int, default=8128)
    parser.add_argument("--input", type=Path, default=ROOT / "workload.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready-timeout", type=int, default=1200)
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Revalidate completed stages and run remaining stages",
    )
    args = parser.parse_args()
    for name in ("final_wheel", "rollback_wheel", "input", "output", "model"):
        setattr(args, name, getattr(args, name).resolve())
    if (
        args.output.exists()
        and any(args.output.iterdir())
        and not args.resume_completed
    ):
        parser.error("output directory must be empty")
    if digest(args.final_wheel) == digest(args.rollback_wheel):
        parser.error("rollback must use a different wheel")
    assert_port_free(args.port)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume_completed:
        if digest(args.input) != digest(args.output / "input.jsonl"):
            parser.error("resume input differs from the original replay")
    else:
        shutil.copyfile(args.input, args.output / "input.jsonl")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        parser.error("input must contain unique request IDs")

    env = runtime_environment(dict(os.environ))
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": args.device,
            "COMPILE_CUSTOM_KERNELS": "1",
            "ASCEND_ATB_CXX_ABI": "1",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "PYTHONUNBUFFERED": "1",
            "VLLM_ASCEND_SIMLLM_HYBRID_REUSE_MODE": "tail_anchor",
            "VLLM_ASCEND_SIMLLM_HYBRID_RECOMPUTE_TOKENS": "512",
        }
    )
    overlay = args.output / "venv"
    if not args.resume_completed:
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(overlay)],
            check=True,
            env=env,
        )
    python = str(overlay / "bin/python")

    def identity(executable: str) -> dict:
        result = subprocess.check_output([executable, "-I", "-c", IDENTITY], env=env)
        return json.loads(result)

    base_identity = identity(sys.executable)
    report = {
        "passed": False,
        "diagnostic_only": True,
        "device": args.device,
        "model": str(args.model),
        "model_config_sha256": digest(args.model / "config.json"),
        "input_sha256": digest(args.input),
        "requests_per_arm": len(rows),
        "final_wheel_sha256": digest(args.final_wheel),
        "rollback_wheel_sha256": digest(args.rollback_wheel),
        "stages": [],
    }
    if args.resume_completed:
        previous = json.loads((args.output / "summary.json").read_text())
        for key in (
            "device",
            "model",
            "input_sha256",
            "final_wheel_sha256",
            "rollback_wheel_sha256",
        ):
            if previous[key] != report[key]:
                raise AssertionError(f"Resume configuration changed: {key}")
        if base_identity != json.loads(
            (args.output / "base-installation-before.json").read_text()
        ):
            raise AssertionError("Original serving installation changed before resume")
        shutil.copyfile(
            args.output / "summary.json", args.output / "pre-resume-summary.json"
        )
        report["revalidated_stages"] = []
    else:
        write_json(args.output / "base-installation-before.json", base_identity)
    server = None

    def stop_server() -> None:
        nonlocal server
        if server is None:
            return
        # Only this script's newly created service session may be signaled.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=30)
        for _ in range(30):
            try:
                assert_port_free(args.port)
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("Owned server port did not close after shutdown")
        server = None

    try:
        for index, (name, wheel, enabled, native_apc, chunked_prefill) in enumerate(
            (
                ("baseline", args.rollback_wheel, False, False, False),
                ("approximate", args.final_wheel, True, False, False),
                ("chunked_baseline", args.rollback_wheel, False, False, True),
                ("native_apc", args.rollback_wheel, False, True, True),
                ("rollback", args.rollback_wheel, False, False, False),
            )
        ):
            print(f"STAGE {name}: install {wheel} enabled={enabled}", flush=True)
            stage = args.output / name
            if args.resume_completed and (stage / "outputs.jsonl").exists():
                report["stages"].append(
                    validate_stage(
                        stage, rows, wheel, enabled, native_apc, chunked_prefill
                    )
                )
                report["revalidated_stages"].append(name)
                write_json(args.output / "summary.json", report)
                print(f"STAGE {name}: completed evidence revalidated", flush=True)
                continue
            stage.mkdir()
            if index:
                current = identity(python)
                if not Path(current["package_root"]).is_relative_to(overlay):
                    raise RuntimeError("Refusing to uninstall outside the overlay")
                with (stage / "uninstall.log").open("w") as log:
                    subprocess.run(
                        [python, "-m", "pip", "uninstall", "-y", "vllm-ascend-simllm"],
                        check=True,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            with (stage / "install.log").open("w") as log:
                subprocess.run(
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--ignore-installed",
                        str(wheel),
                    ],
                    check=True,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            installed = identity(python)
            if not Path(installed["package_root"]).is_relative_to(overlay) or installed[
                "files"
            ] != wheel_files(wheel):
                raise RuntimeError("Installed files do not match the requested wheel")
            write_json(stage / "installed.json", installed)
            stage_env = {**env, "VLLM_ASCEND_SIMLLM_ENABLED": str(int(enabled))}
            subprocess.run(
                [
                    python,
                    "-c",
                    (
                        "import importlib.util,pathlib,sys; "
                        "import vllm_ascend_simllm as p; "
                        "assert pathlib.Path(p.__file__).is_relative_to(sys.prefix); "
                        "assert importlib.util.find_spec('acl') is not None, "
                        "'CANN acl is unavailable: source the qualified CANN environment'"
                    ),
                ],
                check=True,
                env=stage_env,
            )
            command = [
                python,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                str(args.model),
                "--served-model-name",
                "qwen35-lifecycle",
                "--trust-remote-code",
                "--quantization",
                "ascend",
                "--safetensors-load-strategy",
                "eager",
                "--tensor-parallel-size",
                "1",
                "--max-model-len",
                "8192",
                "--gpu-memory-utilization",
                "0.85",
                "--block-size",
                "128",
                (
                    "--enable-prefix-caching"
                    if native_apc
                    else "--no-enable-prefix-caching"
                ),
                (
                    "--enable-chunked-prefill"
                    if chunked_prefill
                    else "--no-enable-chunked-prefill"
                ),
                "--max-num-seqs",
                "4",
                "--enforce-eager",
                "--seed",
                "0",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ]
            write_json(
                stage / "launch.json",
                {
                    "command": command,
                    "settings": {
                        k: stage_env[k]
                        for k in sorted(stage_env)
                        if k.startswith("VLLM_ASCEND_SIMLLM_")
                    },
                    "pythonpath": stage_env.get("PYTHONPATH", ""),
                },
            )
            assert_port_free(args.port)
            with (stage / "server.log").open("w") as log:
                server = subprocess.Popen(
                    command,
                    env=stage_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                deadline = time.monotonic() + args.ready_timeout
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"{name} server exited: {server.returncode}")
                    try:
                        with urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2):
                            break
                    except OSError:
                        time.sleep(2)
                else:
                    raise TimeoutError(f"{name} server readiness timed out")
                print(f"STAGE {name}: ready, replay {len(rows)} requests", flush=True)
                client = [
                    python,
                    str(ROOT / "client.py"),
                    "--input",
                    str(args.input),
                    "--url",
                    f"http://127.0.0.1:{args.port}",
                    "--model",
                    "qwen35-lifecycle",
                    "--max-tokens",
                    "64",
                    "--chat",
                    "--capture-text",
                    "--concurrency",
                    "1",
                ]
                write_json(stage / "client-command.json", client)
                with (stage / "outputs.jsonl").open("w") as output:
                    subprocess.run(
                        client,
                        check=True,
                        env=stage_env,
                        stdout=output,
                        stderr=subprocess.PIPE,
                        timeout=600,
                    )
                with contextlib.suppress(OSError):
                    (stage / "metrics.prom").write_bytes(
                        urlopen(
                            f"http://127.0.0.1:{args.port}/metrics", timeout=10
                        ).read()
                    )
                stop_server()
            result = validate_stage(
                stage, rows, wheel, enabled, native_apc, chunked_prefill
            )
            report["stages"].append(result)
            write_json(args.output / "summary.json", report)
            print(
                f"STAGE {name}: PASS, semantic reuse events="
                f"{result['semantic_reuse_events']}",
                flush=True,
            )
        comparison = compare_outputs(
            read_outputs(args.output / "baseline/outputs.jsonl"),
            read_outputs(args.output / "rollback/outputs.jsonl"),
        )
        report["rollback_comparison"] = comparison
        if not comparison["exactly_equal"]:
            raise AssertionError(f"Rollback output differences: {comparison}")
        report["apc_comparison"] = compare_outputs(
            read_outputs(args.output / "chunked_baseline/outputs.jsonl"),
            read_outputs(args.output / "native_apc/outputs.jsonl"),
        )
        after = identity(sys.executable)
        write_json(args.output / "base-installation-after.json", after)
        if after != base_identity:
            raise AssertionError("The original serving installation changed")
        report["base_installation_unchanged"] = True
        report["passed"] = True
    except Exception:
        report["passed"] = False
        report["error"] = traceback.format_exc()
        raise
    finally:
        stop_server()
        write_json(args.output / "summary.json", report)
    print("INFERENCE_AND_WHEEL_ROLLBACK_PASS", flush=True)


if __name__ == "__main__":
    main()
