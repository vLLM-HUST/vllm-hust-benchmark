"""Run the unchanged pinned BurstGPT example against a loopback completion API.

This is the publisher's external-prompt simulation, not original private prompts.
The compatibility bridge implements only its default non-streaming /generate.
"""

import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from urllib.parse import urlparse
from urllib.request import Request, urlopen


TRACE_SHA = {
    "BurstGPT_1.csv": "4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122",  # pragma: allowlist secret - original release SHA256
    "BurstGPT_2.csv": "44bf5942b03fca42c01a226a545ad3a750ec62688faecd67b6831c56a8928bf7",  # pragma: allowlist secret - original release SHA256
    "BurstGPT_3.csv": "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f",  # pragma: allowlist secret - original release SHA256
}
POOL_BLOB = "4455ef4b856a79e7ec4264cd4e8d19bd0666510d"  # pragma: allowlist secret - publisher Git blob
SOURCE_COMMIT = "7eb2c4f8350f8a6985272386f5c14af1f678b299"  # pragma: allowlist secret - public pinned commit
SOURCE_BLOBS = {
    "LICENSE": "4ea99c213c5c0c005ae4e80df8e52169d06896ec",  # pragma: allowlist secret - publisher Git blob
    "example/README.md": "87518e6b588cf295af12b39c58474fc47e7a911a",  # pragma: allowlist secret - publisher Git blob
    "example/framework_inference_call/__init__.py": "c4cbeb753a8984da15d3790135a04a097e6b8fae",  # pragma: allowlist secret - publisher Git blob
    "example/framework_inference_call/framework_lightllm.py": "65bde0f9f85a439bc62a6b09caa170c62ebaa909",  # pragma: allowlist secret - publisher Git blob
    "example/framework_inference_call/framework_vllm.py": "1c0b52ec11ecb318de0b9f41b358069265333d6e",  # pragma: allowlist secret - publisher Git blob
    "example/preprocess_data/README.md": "da99614beac51b425b884ff8a597f66431864bd7",  # pragma: allowlist secret - publisher Git blob
    "example/preprocess_data/__init__.py": "338fefa8894ff31586f686dbc953727c0cc6a084",  # pragma: allowlist secret - publisher Git blob
    "example/preprocess_data/load_data.py": "7e12c9824c3088cb3cc505bd0d2ddc8f996f2c03",  # pragma: allowlist secret - publisher Git blob
    "example/preprocess_data/pre_shareGPT.py": "b064f89588e1011a7744de4b17136ce665dd9941",  # pragma: allowlist secret - publisher Git blob
    "example/profile_server/__init__.py": "d37aed83d21d6aeeb93b6489fc3d96123a1920ba",  # pragma: allowlist secret - publisher Git blob
    "example/profile_server/config.py": "0549179e7fe8e92aa6f5a45c3f60fd59f5ab3e71",  # pragma: allowlist secret - publisher Git blob
    "example/profile_server/logger.py": "fc3aa5f10a3408042e34acaf61c9700b38ef34bb",  # pragma: allowlist secret - publisher Git blob
    "example/profile_server/prompt_set.py": "1e9eedd28510d5706a9f7fa18bd3b75c4fb07fe6",  # pragma: allowlist secret - publisher Git blob
    "example/profile_server/server.py": "0697cb2241d8ccd9d09024047006a803a07e5ade",  # pragma: allowlist secret - publisher Git blob
    "example/profile_vllm_server.py": "4df5cb1eab19f46369e0aa85f94168e5b35f8cb1",  # pragma: allowlist secret - publisher Git blob
    "example/profile_vllm_trace.sh": "1afaf5202140dbcbec2afb76678826e77a004159",  # pragma: allowlist secret - publisher Git blob
    "example/requirements.txt": "45a9137a1ac4ef24d4c0310b006a872c9ddd4982",  # pragma: allowlist secret - publisher Git blob
    "example/start_vllm.sh": "018be9aed68e0567c5f23c20c174f361619aec88",  # pragma: allowlist secret - publisher Git blob
}
TRACE_ROWS = {
    "BurstGPT_1.csv": 1429737,
    "BurstGPT_2.csv": 3858436,
    "BurstGPT_3.csv": 5344021,
}


def endpoint(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("uncredentialed loopback /v1 endpoint required")
    return value.rstrip("/")


def original_request(body, model):
    expected = {"prompt", "stream", "ignore_eos", "max_tokens", "temperature"}
    if (
        set(body) != expected
        or not isinstance(body["prompt"], str)
        or not body["prompt"]
        or body["stream"] is not False
        or body["ignore_eos"] is not False
        or type(body["max_tokens"]) is not int
        or body["max_tokens"] != 1024
        or body["temperature"] != 0
    ):
        raise ValueError("unexpected publisher default non-streaming request")
    return {**body, "model": model}


def legacy_response(body, result):
    choices = result.get("choices")
    usage = result.get("usage")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0].get("text"), str)
        or choices[0].get("finish_reason") not in {"stop", "length"}
        or not isinstance(usage, dict)
        or type(usage.get("prompt_tokens")) is not int
        or usage["prompt_tokens"] <= 0
        or type(usage.get("completion_tokens")) is not int
        or not 0 <= usage["completion_tokens"] <= body["max_tokens"]
    ):
        raise ValueError("incomplete native completion response")
    # Original vLLM /generate returns prompt + generated text, not just the suffix.
    return {"text": [body["prompt"] + choices[0]["text"]]}


@contextmanager
def bridge(api, model, output):
    records, errors = [], []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/generate" or not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("unsupported bridge request")
                raw = self.rfile.read(length)
                body = json.loads(raw)
                routed = original_request(body, model)
                request = Request(
                    api + "/completions",
                    data=json.dumps(routed).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=4 * 60 * 60) as response:
                    result = json.load(response)
                legacy = legacy_response(body, result)
                with lock:
                    index = len(records)
                    private = output / f"request-{index}.private.json"
                    private.write_text(
                        json.dumps(
                            {
                                "legacy_request": body,
                                "native_request": routed,
                                "native_response": result,
                                "legacy_response": legacy,
                            }
                        )
                    )
                    records.append(
                        {
                            "private_record_sha256": hashlib.sha256(
                                private.read_bytes()
                            ).hexdigest(),
                            "prompt_sha256": hashlib.sha256(
                                body["prompt"].encode()
                            ).hexdigest(),
                            "prompt_tokens": result["usage"]["prompt_tokens"],
                            "completion_tokens": result["usage"]["completion_tokens"],
                            "finish_reason": result["choices"][0]["finish_reason"],
                        }
                    )
                payload = json.dumps(legacy).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as error:
                with lock:
                    errors.append(type(error).__name__)
                self.send_error(502, "native transport or response validation failed")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, records, errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prompt-pool", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    args = parser.parse_args()
    api = endpoint(args.endpoint)
    if (
        not 0 < args.count <= TRACE_ROWS.get(args.trace.name, 0)
        or not math.isfinite(args.scale)
        or args.scale <= 0
    ):
        parser.error(
            "positive count within original trace population and positive finite scale required"
        )
    for name, blob in SOURCE_BLOBS.items():
        path = (args.source_root / name).resolve()
        if not path.is_relative_to(args.source_root.resolve()):
            raise ValueError("unsafe publisher source path")
        raw = path.read_bytes()
        if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != blob:
            raise ValueError("publisher source file changed")
    raw = args.prompt_pool.read_bytes()
    if hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() != POOL_BLOB:
        raise ValueError("publisher external prompt pool changed")
    if hashlib.sha256(args.trace.read_bytes()).hexdigest() != TRACE_SHA.get(
        args.trace.name
    ):
        raise ValueError("original release trace changed")
    os.umask(0o077)
    args.output = args.output.resolve()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "BURSTGPT",
        "status": "STARTED",
        "trace": args.trace.name,
        "requested_original_rows": args.count,
        "scale": args.scale,
        "arrival_time_scaled": args.scale != 1,
        "validation_purpose": "tool_usability_only_not_original_timing_performance",
        "original_timing_performance_claim": False,
        "model": args.model,
        "official_source_commit": SOURCE_COMMIT,
        "original_prompt_replay": False,
        "external_publisher_prompt_pool": True,
        "original_tool_source_modified": False,
        "original_output_budget": 1024,
        "transport": "loopback default non-streaming legacy generate to native completions; prompt and decoding unchanged",
        "source_checkbox_eligible": False,
        "performance_or_quality_claim": False,
        "raw_payloads_published": False,
    }
    try:
        with bridge(api, args.model, args.output) as (port, records, errors):
            command = [
                sys.executable,
                str(args.source_root.resolve() / "example/profile_vllm_server.py"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--temperature",
                "0",
                "--data_path",
                str(args.prompt_pool.resolve()),
                "--use_burstgpt",
                "--burstgpt_path",
                str(args.trace.resolve()),
                "--prompt_num",
                str(args.count),
                "--surplus_prompts_num",
                str(args.count),
                "--scale",
                str(args.scale),
                "--log_path",
                str(args.output / "official-summary.private.json"),
                "--detail_log_path",
                str(args.output / "official-detail.private.json"),
            ]
            with (args.output / "official-process.private.log").open("x") as log:
                environment = dict(os.environ)
                for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "LD_PRELOAD"):
                    environment.pop(key, None)
                environment["PYTHONNOUSERSITE"] = "1"
                process = subprocess.run(
                    command,
                    cwd=args.output,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            receipt.update(
                process_returncode=process.returncode,
                native_responses=records,
                errors=errors,
            )
            if process.returncode or errors or len(records) != args.count:
                raise RuntimeError(
                    "official workflow did not complete all original selected requests"
                )
            if not (args.output / "official-summary.private.json").is_file():
                raise ValueError("official final summary absent")
            receipt["status"] = (
                "OFFICIAL_EXTERNAL_PROMPT_SIMULATION_EXECUTED_REQUIRES_AUDIT"
            )
    except BaseException as error:
        receipt.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
