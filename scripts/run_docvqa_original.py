"""Run fixed original DocVQA tasks via the unchanged pinned Inspect Evals task.

External original files and a frozen Inspect client are required. No model is
loaded by this client. Native PNG thumbnail1024, ANSWER template and ANLS are
retained; this maintained implementation is not the dataset author's harness.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse
from urllib.request import urlopen

TASK_SHA256 = "077acecba59a9b00691134bda5a865c51928841e746eeaeec101526fbd3ba146"  # pragma: allowlist secret - pinned public maintained task identity
LOCK_SHA256 = "2125132b12e0a5d75b6116f51239e15823a08737ad72eab8bdb846bd51c1de2b"  # pragma: allowlist secret - original frozen public lock identity
FIXED_IDS = ["49153", "24580", "57349"]


def local_endpoint(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("explicit loopback HTTP /v1 endpoint required")
    return value.rstrip("/")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_originals(data_root, source_root):
    from vllm_hust_workload_data.assets import verify_file
    from vllm_hust_workload_data.registry import source_spec

    source_root = Path(source_root).resolve()
    if digest(source_root / "src/inspect_evals/docvqa/docvqa.py") != TASK_SHA256:
        raise ValueError("pinned original task identity differs")
    if digest(source_root / "uv.lock") != LOCK_SHA256:
        raise ValueError("pinned original frozen lock differs")
    spec = source_spec("DOCVQA")
    for file in spec["files"]:
        verify_file(Path(data_root), file)
    return spec


def idle(endpoint, api_pid, engine_pid):
    if not Path(f"/proc/{api_pid}/cmdline").is_file():
        return False
    with urlopen(endpoint.removesuffix("/v1") + "/metrics", timeout=10) as response:
        metrics = response.read().decode()
    counts = [
        float(line.split()[-1])
        for line in metrics.splitlines()
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
    ]
    devices = subprocess.check_output(["npu-smi", "info"], text=True)
    owners = set(re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|", devices, re.M))
    return (
        bool(counts)
        and all(value == 0 for value in counts)
        and owners == {str(engine_pid)}
    )


def execute(args):
    endpoint = local_endpoint(args.endpoint)
    output = Path(args.output).resolve()
    data = Path(args.data_root).resolve()
    source = Path(args.framework_source_root).resolve()
    if (
        output.exists()
        or output.is_relative_to(data)
        or data.is_relative_to(output)
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from original data required")
    if args.max_tokens <= 0:
        raise ValueError("positive declared model output budget required")
    manifest = verify_originals(data, source)
    output.mkdir(mode=0o700, parents=True)
    result = {
        "source_id": "DOCVQA",
        "status": "PREPARING_ORIGINAL_NATIVE_TASK",
        "source_checkbox_eligible": False,
        "model": args.model,
        "original_fixed_question_ids": FIXED_IDS,
        "max_tokens": args.max_tokens,
        "protocol": "unchanged maintained Inspect Evals DocVQA template/thumbnail1024/ANLS",
        "model_profile": "available NPU connectivity fallback, not paper quality reproduction",
        "original_validation_rows": sum(f["rows"] for f in manifest["files"]),
    }
    receipt = output / "native-public.json"

    def save():
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        # Cache placement is deployment configuration, not a task modification.
        os.environ["XDG_CACHE_HOME"] = str(Path(args.inspect_cache_root).resolve())
        os.environ["XDG_DATA_HOME"] = str(output / "native-runtime-data")
        os.environ["XDG_CONFIG_HOME"] = str(output / "native-runtime-config")
        os.environ["INSPECT_EVALS_CACHE_DIR"] = str(
            Path(args.image_cache_root).resolve()
        )
        native = importlib.import_module("inspect_evals.docvqa.docvqa")
        if digest(native.__file__) != TASK_SHA256:
            raise ValueError("installed task differs from pinned source")
        from inspect_ai import eval as native_eval
        from inspect_ai.model import GenerateConfig, get_model

        task = native.docvqa(shuffle=False)
        if (
            len(task.dataset) != 5349
            or [str(s.id) for s in task.dataset[:3]] != FIXED_IDS
        ):
            raise ValueError("native original task identity/order differs")
        model = get_model(
            "openai-api/local/" + args.model,
            base_url=endpoint,
            api_key="unused-local",  # pragma: allowlist secret - dummy credential for loopback NPU API only
            max_retries=0,  # Disable SDK retries as well as native evaluator retries.
            config=GenerateConfig(
                max_tokens=args.max_tokens,
                max_retries=0,
                max_connections=1,
                timeout=1200,
            ),
        )
        original_generate = model.api.generate
        attempts = []

        async def guarded_generate(*positional, **keywords):
            # Only guard transport scheduling. Native prompts/solver/scorer stay unchanged.
            while not idle(endpoint, args.api_server_pid, args.npu_engine_pid):
                await asyncio.sleep(10)
            attempts.append({"attempt": len(attempts) + 1, "status": "STARTED"})
            result["attempts"] = attempts
            save()
            try:
                response = await original_generate(*positional, **keywords)
            except Exception as exc:
                attempts[-1].update(status="RAISED", error_type=type(exc).__name__)
                save()
                raise
            attempts[-1]["status"] = "RETURNED_TO_NATIVE_FRAMEWORK"
            save()
            return response

        model.api.generate = guarded_generate
        result["status"] = "RUNNING_FIXED_ORIGINAL_TASKS"
        save()
        logs = native_eval(
            task,
            model=model,
            sample_id=FIXED_IDS,
            max_samples=1,
            epochs=1,
            retry_on_error=0,
            fail_on_error=False,
            continue_on_fail=True,
            log_dir=str(output / "native-logs"),
            log_format="json",
            log_model_api=True,
            display="plain",
        )
        if len(logs) != 1:
            raise ValueError("unexpected native evaluation log count")
        log = logs[0]
        samples = log.samples or []
        result.update(
            native_log_status=log.status,
            native_samples=len(samples),
            native_sample_errors=sum(s.error is not None for s in samples),
            scores=[
                {
                    "question_id": str(s.id),
                    "scores": {
                        name: score.value for name, score in (s.scores or {}).items()
                    },
                }
                for s in samples
            ],
        )
        if (
            log.status != "success"
            or len(samples) != 3
            or {str(s.id) for s in samples} != set(FIXED_IDS)
            or any(s.error is not None or not s.scores for s in samples)
            or len(attempts) != 3
        ):
            raise ValueError("native fixed task flow incomplete; retain all failures")
        verify_originals(data, source)
        result["status"] = (
            "FIXED3_NATIVE_TASKS_SCORED_SAVED_AUDIT_AND_CI_HANDOFF_PENDING"
        )
    except Exception as exc:
        result.update(status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__)
        raise
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-root",
        "framework-source-root",
        "output",
        "inspect-cache-root",
        "image-cache-root",
        "endpoint",
        "model",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--api-server-pid", required=True, type=int)
    parser.add_argument("--npu-engine-pid", required=True, type=int)
    parser.add_argument("--max-tokens", type=int, default=16384)
    print(json.dumps(execute(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()
