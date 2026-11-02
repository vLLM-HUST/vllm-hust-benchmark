"""Offline acceptance-gap evidence; never launch services or publish raw answers."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def unit_cost(
    price: float, days: int, cards: int, seconds: float, tokens: int
) -> float:
    """CNY per million successful output tokens, for an explicit scope."""
    if not all(
        math.isfinite(x) and x > 0 for x in (price, days, cards, seconds, tokens)
    ):
        raise ValueError("Cost inputs must be finite and positive")
    return price * cards * seconds / (days * 86400) * 1_000_000 / tokens


def summarize_requests(records: list[dict], start: float, end: float) -> dict:
    """Only a closed request cohort; budget stops are not silent truncation."""
    if not records or end <= start:
        raise ValueError("Empty cohort or invalid interval")
    ids = [record["case_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate requests")
    lengths = []
    reasons: Counter = Counter()
    input_tokens = output_tokens = success = crossing = 0
    for record in records:
        if not start <= record["started"] <= record["ended"] <= end:
            crossing += 1
            continue
        if not record["success"]:
            continue
        usage = record["usage"]
        incoming, outgoing = usage["prompt_tokens"], usage["completion_tokens"]
        if incoming < 0 or outgoing < 0:
            raise ValueError("Negative token count")
        if usage["total_tokens"] != incoming + outgoing:
            raise ValueError("Inconsistent token count")
        input_tokens += incoming
        output_tokens += outgoing
        lengths.append(incoming + outgoing)
        reasons[record["finish_reason"]] += 1
        success += 1
    return {
        "requests": len(records),
        "successful_in_interval": success,
        "crossing_interval": crossing,
        "failed_in_interval": len(records) - success - crossing,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "actual_context_max_tokens": max(lengths, default=None),
        "actual_context_at_least_16384": sum(n >= 16384 for n in lengths),
        "finish_reasons": dict(sorted(reasons.items())),
        "unknown_finish_reasons": sum(
            n for k, n in reasons.items() if k not in ("stop", "length")
        ),
        "client_envelope_seconds": end - start,
        "five_dataset_stability_pass": None,
        "controlled_4k_16k_comparison_pass": None,
    }


def audit_native_quality(raw: Path) -> dict:
    receipt_path = raw / "receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    # Public sealed receipt SHA256, not a credential.
    expected = "c226cce6036b0e1448c91ae78a72325d43fc5f9bbb728eae76c5d8ba9f482dcd"  # pragma: allowlist secret
    if receipt_sha != expected:
        raise ValueError("Not the sealed official 0.18 MMLU quality run")
    receipt = json.loads(receipt_bytes)
    files = sorted(raw.glob("quality-case-*.json"))
    if len(files) != 200:
        raise ValueError("Expected all 200 cases, not a selected subset")
    records = [json.loads(path.read_bytes()) for path in files]
    result = summarize_requests(
        records, receipt["client_started"], receipt["client_ended"]
    )
    service_bytes = (raw / "service.log").read_bytes()
    log = service_bytes.decode(errors="replace")
    patterns = {
        "oom_markers": r"(?i)out[ -]of[ -]memory|OutOfMemory",
        "deadlock_markers": r"(?i)deadlock",
        "traceback_markers": r"Traceback \(most recent call last\)",
    }
    oom_lines = [
        (number, line)
        for number, line in enumerate(log.split("\n"), 1)
        if re.search(patterns["oom_markers"], line)
    ]
    advisory = (
        "* (Out of Memory) errors or inference hangs. If you encounter such issues,"
    )
    result.update(
        {
            "schema": "fixed-b0-quality-stability-reuse/v1",
            "run_id": "mmlu-quality018-r2",
            "runtime": {"vllm": "0.18.0", "vllm_ascend": "0.18.0"},
            "output_budget": receipt["max_completion_tokens"],
            "participating_devices": receipt["allocated"],
            "reserved_devices": receipt["reserved"],
            "correct": sum(record["correct"] for record in records),
            "service_log_marker_counts": {
                name: len(re.findall(pattern, log))
                for name, pattern in patterns.items()
            },
            "oom_marker_review": {
                "generic_platform_advisory_lines": [
                    n for n, line in oom_lines if advisory in line
                ],
                "non_advisory_oom_marker_lines": [
                    n for n, line in oom_lines if advisory not in line
                ],
                "scope": "The two generic platform warnings mention possible OOM; they are not observed OOM events.",
            },
            "service_shutdown_complete_observed": "Application shutdown complete."
            in log,
            "shutdown_shared_memory_warning_observed": "leaked shared_memory" in log,
            "original_all_devices_released": receipt["all_devices_released"],
            "release_resolution": "Later independent release follow-up passed; original false retained.",
            "release_followup": "reports/mmlu-quality018-20261022/quality018-release-followup-r1.json",
            "source_hashes": {
                "receipt.json": receipt_sha,
                "service.log": hashlib.sha256(service_bytes).hexdigest(),
                **{
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in files
                },
            },
            "scope": [
                "Offline original-case and service-log review; zero new inference or grading.",
                "Known budget finish=length is explicit termination, not evidence of silent truncation.",
                "No matched 4K/16K windows or TTFT/TPOT percentiles are inferred from aggregate records.",
                "No marker found is limited to the retained log, not proof of all five datasets' service stability.",
                "This out8192 quality run is not the out256 fixed-B0 timing run; their resource allocations must not be interchanged.",
                "Hashes identify private original cases; prompts, answers and reasoning are not exported.",
            ],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-quality-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_native_quality(args.native_quality_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "source_hashes"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
