"""Pinned original TraceLab metadata-driven token-ID replay, not original prompts.

The upstream synthetic text-pool construction and mandatory cache probe remain
unchanged. No research observer, prompt invention or cache-probe bypass added.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from urllib.request import urlopen

from build_tracelab_original import digest, verify_source

TRACE_SHA = "9f077ff969a2d839e6753d4e58e7796fcfdf62fd8bc6f28873b3b6acdc3c7e65"  # pragma: allowlist secret
TEXT_SHA = "159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc"  # pragma: allowlist secret


def selection(path, *, full=False):
    sessions = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            sid = row.get("id") or row.get("session_id")
            if not sid:
                raise ValueError("Original session ID required")
            demand = sum(
                int(row[key]) for key in ("prefix_len", "input_len", "output_len")
            )
            state = sessions.setdefault(sid, {"steps": 0, "max_prompt_plus_output": 0})
            state["steps"] += 1
            state["max_prompt_plus_output"] = max(
                state["max_prompt_plus_output"], demand
            )
    selected = sorted(sessions) if full else sorted(sessions)[:3]
    if len(selected) < 3:
        raise ValueError("At least three original sessions required")
    return (
        selected,
        sum(sessions[sid]["steps"] for sid in selected),
        max(sessions[sid]["max_prompt_plus_output"] for sid in selected),
    )


def validate_summary(summary, expected_steps):
    replay = summary["replay"]
    if (
        summary["workload"]["steps"] != expected_steps
        or replay["attempted_steps"] != expected_steps
    ):
        raise ValueError("Missing native original rounds")
    if (
        replay["success_steps"] != expected_steps
        or replay["failed_steps"]
        or replay["context_overflow_steps"]
    ):
        raise ValueError(
            "Native failures or skipped rounds retained, not source completion"
        )


def verify_rounds(trace, log, selected):
    expected = {}
    with Path(trace).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            sid = row.get("id") or row.get("session_id")
            if sid not in selected:
                continue
            key = (sid, int(row["round_idx"]))
            if key in expected:
                raise ValueError("Duplicate original session/round identity")
            expected[key] = row
    seen = set()
    with Path(log).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            key = (record["session_id"], record["round_idx"])
            if key not in expected or key in seen or record["status"] != "SUCCESS":
                raise ValueError(
                    "Missing, duplicate, substituted or failed native round"
                )
            original = expected[key]
            for source_field, output_field in (
                ("prefix_len", "prefix_len"),
                ("input_len", "input_len"),
                ("output_len", "output_len_target"),
            ):
                if int(original[source_field]) != record[output_field]:
                    raise ValueError("Original native request budget changed")
            if float(original["tool_wait_after_ms"]) != record["tool_wait_after_ms"]:
                raise ValueError("Original tool-wait schedule changed")
            seen.add(key)
    if seen != set(expected):
        raise ValueError("Not all selected original rounds completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-library-dir", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    route = urlparse(args.base_url)
    if (
        route.scheme != "http"
        or route.hostname not in {"127.0.0.1", "localhost", "::1"}
        or route.path != "/v1"
        or route.username
        or route.password
        or route.query
        or route.fragment
    ):
        raise ValueError("Credential-free loopback /v1 endpoint required")
    if not args.execute:
        print(
            json.dumps(
                {
                    "source_id": "TRACELAB",
                    "status": "PLAN_ONLY",
                    "selection": "full original trace"
                    if args.full
                    else "lexicographically first3originalsessions with all rounds",
                    "mandatory_original_cache_probe": True,
                    "original_private_prompts_claim": False,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    source = args.official_source.resolve(strict=True)
    source_commit = verify_source(source)
    build = json.loads((args.build / "build-public.json").read_text(encoding="utf-8"))
    binary = args.build / "target/release/session_runner"
    if (
        build["status"] != "ORIGINAL_LOCKED_BUILD_COMPLETE_NOT_REPLAY"
        or build["source_commit"] != source_commit
        or digest(binary) != build["binary_sha256"]
    ):
        raise ValueError("Verified original native build required")
    if digest(args.trace) != TRACE_SHA or digest(args.text_file) != TEXT_SHA:
        raise ValueError("Pinned original CSV and upstream text pool required")
    selected, steps, demand = selection(args.trace, full=args.full)
    if demand > args.max_model_len:
        raise ValueError(
            "Original prompt plus output exceeds context; no truncation or skipping"
        )
    if args.output.exists():
        raise ValueError(
            "Fresh private output required; no overwrite or automatic retries"
        )
    with urlopen(args.base_url + "/models", timeout=10) as response:
        models = json.load(response)["data"]
    if not any(
        row["id"] == args.model and row.get("max_model_len") == args.max_model_len
        for row in models
    ):
        raise ValueError("Model alias and context must match running native service")
    with urlopen(f"{route.scheme}://{route.netloc}/metrics", timeout=10) as response:
        metrics = response.read().decode()
    active = [
        float(line.split()[-1])
        for line in metrics.splitlines()
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
    ]
    if not active or any(active):
        raise ValueError("Wait for existing users; no automatic competition")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    output = args.output.resolve()
    command = [
        str(binary.resolve()),
        "--trace",
        str(args.trace.resolve()),
        "--text-file",
        str(args.text_file.resolve()),
        "--tokenizer",
        str(args.tokenizer.resolve()),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--max-model-len",
        str(args.max_model_len),
        "--max-active-sessions",
        "1",
        "--log-path",
        str(output / "native-rounds-private.jsonl"),
        "--summary-path",
        str(output / "native-summary-private.json"),
    ]
    if not args.full:
        command += ["--max-sessions", "3"]
    environment = dict(os.environ)
    if args.runtime_library_dir:
        environment["LD_LIBRARY_PATH"] = (
            str(args.runtime_library_dir.resolve(strict=True))
            + os.pathsep
            + environment.get("LD_LIBRARY_PATH", "")
        )
    receipt = {
        "source_id": "TRACELAB",
        "status": "RUNNING_ORIGINAL_CACHE_PROBE_AND_REPLAY",
        "pid": os.getpid(),
        "source_commit": source_commit,
        "binary_sha256": build["binary_sha256"],
        "trace_sha256": TRACE_SHA,
        "text_sha256": TEXT_SHA,
        "tokenizer_sha256": digest(args.tokenizer),
        "original_sessions": len(selected),
        "original_steps": steps,
        "max_prompt_plus_output": demand,
        "original_private_prompts_claim": False,
        "pool_limit_override": False,
        "cache_probe_bypassed": False,
        "research_observer_added": False,
        "source_checkbox_eligible": False,
        "command": command,
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        with (output / "native-private.log").open("xb") as log:
            result = subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        receipt["exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(
                "Original probe or replay failed; retained without bypass"
            )
        summary = json.loads((output / "native-summary-private.json").read_text())
        validate_summary(summary, steps)
        verify_rounds(args.trace, output / "native-rounds-private.jsonl", selected)
        if (
            verify_source(source) != source_commit
            or digest(args.trace) != TRACE_SHA
            or digest(args.text_file) != TEXT_SHA
            or digest(args.tokenizer) != receipt["tokenizer_sha256"]
            or digest(binary) != build["binary_sha256"]
        ):
            raise ValueError("Original source, trace or native binary changed")
        receipt.update(
            status="ORIGINAL_SELECTED_SESSION_REPLAY_COMPLETE_NOT_SOURCE_DELIVERY",
            successful_steps=steps,
            mandatory_native_cache_probe_passed=True,
            output_mismatch_steps=summary["replay"]["output_mismatch_steps"],
        )
    except BaseException as error:
        receipt.update(
            status="FAILED_RETAINED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        save()
    print(
        json.dumps(
            {
                "source_id": "TRACELAB",
                "status": receipt["status"],
                "steps": steps,
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
