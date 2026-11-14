#!/usr/bin/env python3
"""Recompute the pinned historical TP8 comparison without executing archive code.

Usage: python recompute_tp8.py /path/to/muxpd.tar.gz [--check] [--output FILE]
Requires only Python's standard library. Reads tar members in memory; never extracts.
"""

import argparse
import base64
import gzip
import hashlib
import json
import math
import tarfile
from itertools import pairwise
from pathlib import Path

ARCHIVE_URL = "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/evidence-mod-collections-20261023-v1/muxpd.tar.gz"
MANIFEST_URL = "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json"
ARCHIVE_SHA256 = "937ae9c56f777efe2fe5b7499a29de9b52a0dbd68e37a0cf4e7565bc8612afbe"  # pragma: allowlist secret (public archive SHA-256)
ARCHIVE_BYTES = 120091343
PREFIX = "snapshot/evidence/2026-10-11-tp8-contention/"
PRODUCER_SHA256 = "6f373ff7383ccc61579d4e9406ee0c9b9523ee8a4ef0d5330d76471e421f6712"  # pragma: allowlist secret (public archive SHA-256)
# Byte lengths and compressed-member hashes copied from the fixed manifest above.
MEMBERS = {
    "guard-v2-confirm-p12k-burst.json.gz": {
        "bytes": 2230309,
        "sha256": "4dc65475a4b77d56c48560e13ecb8c78b229b20f4700cef60165f512df68be25",  # pragma: allowlist secret (public archive SHA-256)
    },
    "guard-v2-confirm-p12k.json.gz": {
        "bytes": 919379,
        "sha256": "f4adebd9ba7d978d921627a1e6ff28ea6b35f944edd5da1e7945115df38b9500",  # pragma: allowlist secret (public archive SHA-256)
    },
    "guard-v2-confirm-p8k.json.gz": {
        "bytes": 790133,
        "sha256": "7faeb4c794c16eae9065a3a36f38ba6aeba70ebb67e1f236972f021a14922b47",  # pragma: allowlist secret (public archive SHA-256)
    },
    "guard-v2-initial-p12k-burst.json.gz": {
        "bytes": 2224423,
        "sha256": "e952e56574cee34b20bc3208f65700c5a2516b25247525d34495311c4f36e17f",  # pragma: allowlist secret (public archive SHA-256)
    },
    "guard-v2-initial-p12k.json.gz": {
        "bytes": 910689,
        "sha256": "ef7e9b61a4a889430fda69e1e105ed9d8ee985b6138f671533045c37f78dcf63",  # pragma: allowlist secret (public archive SHA-256)
    },
    "guard-v2-initial-p8k.json.gz": {
        "bytes": 790127,
        "sha256": "8df38d993799fad0c1f557a720978c225aca75e77c6b6677042e4812c541a1eb",  # pragma: allowlist secret (public archive SHA-256)
    },
    "native16k-confirm-p12k-burst.json.gz": {
        "bytes": 2218670,
        "sha256": "5bf7096a4f00128cef0a96ebce2f71430b21696d608d8ed227dc2245d0327f93",  # pragma: allowlist secret (public archive SHA-256)
    },
    "native16k-confirm-p12k.json.gz": {
        "bytes": 910949,
        "sha256": "eb95869f8105ca110c333f644d08ec771cf510ddc219e03f1366f1942f6a1107",  # pragma: allowlist secret (public archive SHA-256)
    },
    "native16k-confirm-p8k.json.gz": {
        "bytes": 778625,
        "sha256": "06c5e5a3b8a9f2ea4f51e1adbe8756cc060c69654d0ccf5919052c3fa963db87",  # pragma: allowlist secret (public archive SHA-256)
    },
    "native16k-initial-p12k-burst.json.gz": {
        "bytes": 2212261,
        "sha256": "b33b7102edf215780b690a2fd6d2e1b1f804fbec6016aa73a11cdf7bbbdc41f6",  # pragma: allowlist secret (public archive SHA-256)
    },
    "native16k-initial-p12k.json.gz": {
        "bytes": 915977,
        "sha256": "ed7b6a1f18c06e88ca7ebaece6dac3a2344ad08d6b24d036dd70d4dbca3c6bd3",  # pragma: allowlist secret (public archive SHA-256)
    },
    "native16k-initial-p8k.json.gz": {
        "bytes": 789734,
        "sha256": "9eb63b1cfae785f85fc746e826b2fbd202ab6d563c8aced08f11448d39eec286",  # pragma: allowlist secret (public archive SHA-256)
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile95(values):
    values = sorted(values)
    position = (len(values) - 1) * 0.95
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def request_events(request):
    require(
        request["transport_passed"]
        and request["http_status"] == 200
        and request["finish_reason"] == "length",
        "incomplete request",
    )
    raw = base64.b64decode(request["raw_sse_base64"], validate=True)
    require(
        sha256(raw) == request["raw_sse_sha256"]
        and len(raw) == request["raw_sse_bytes"],
        "raw SSE hash mismatch",
    )
    times, tokens, payloads, prompts = [], [], [], []
    for event in request["events"]:
        payload = event["raw_data"]
        payloads.append(payload)
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        require(obj == event["json"], "SSE JSON mismatch")
        ids = []
        for choice in obj.get("choices", []):
            ids.extend(choice.get("token_ids") or [])
            if choice.get("prompt_token_ids") is not None:
                prompts.append(choice["prompt_token_ids"])
        if ids:
            require(len(ids) == 1, "multi-token event: individual timing unavailable")
            times.append(event["received_monotonic_s"])
            tokens.extend(ids)
    wire_payloads = [
        line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")
    ]
    require(
        wire_payloads == payloads and payloads.count("[DONE]") == 1,
        "raw SSE/event sequence mismatch",
    )
    require(
        tokens == request["token_ids"]
        and len(tokens) == request["payload"]["max_tokens"],
        "token count mismatch",
    )
    require(
        prompts and all(p == request["payload"]["prompt"] for p in prompts),
        "prompt IDs mismatch",
    )
    usage = request["usage"]
    require(
        usage["prompt_tokens"] == len(request["payload"]["prompt"])
        and usage["completion_tokens"] == len(tokens),
        "usage mismatch",
    )
    require(
        len(times) >= 2 and all(b >= a for a, b in pairwise(times)),
        "missing or unordered token events",
    )
    return times, len(tokens)


def recompute(archive_path):
    require(
        archive_path.stat().st_size == ARCHIVE_BYTES, "whole archive length mismatch"
    )
    require(
        file_sha256(archive_path) == ARCHIVE_SHA256, "whole archive SHA-256 mismatch"
    )
    results, sources = {}, []
    with tarfile.open(archive_path, "r:gz") as archive:
        for role in ("native16k", "guard-v2"):
            for workload in ("p8k", "p12k", "p12k-burst"):
                gaps, ttfts, rounds = [], [], []
                duration, token_count = 0.0, 0
                for phase in ("initial", "confirm"):
                    name = f"{role}-{phase}-{workload}.json.gz"
                    member = archive.extractfile(PREFIX + name)
                    require(member is not None, f"missing member: {name}")
                    compressed = member.read()
                    expected = MEMBERS[name]
                    require(
                        len(compressed) == expected["bytes"]
                        and sha256(compressed) == expected["sha256"],
                        f"member hash mismatch: {name}",
                    )
                    uncompressed = gzip.decompress(compressed)
                    report = json.loads(uncompressed)
                    require(
                        report["complete"] and len(report["repeats"]) == 3,
                        f"incomplete report: {name}",
                    )
                    require(
                        report["producer_sha256"] == PRODUCER_SHA256,
                        f"producer mismatch: {name}",
                    )
                    sources.append(
                        {
                            "member": PREFIX + name,
                            **expected,
                            "uncompressed_sha256": sha256(uncompressed),
                        }
                    )
                    for repeat in report["repeats"]:
                        require(repeat["complete"], "incomplete measured repeat")
                        dt = (
                            repeat["finished_monotonic_s"]
                            - repeat["started_monotonic_s"]
                        )
                        require(dt > 0, "invalid measurement window")
                        round_gaps, round_ttfts, requests = [], [], []
                        round_tokens = 0
                        for request in repeat["requests"]:
                            times, count = request_events(request)
                            round_tokens += count
                            record = {
                                "phase": request["phase"],
                                "index": request["index"],
                                "output_tokens": count,
                                "token_events": len(times),
                            }
                            if request["phase"] == "decode":
                                gap = max((b - a) * 1000 for a, b in pairwise(times))
                                round_gaps.append(gap)
                                record["max_event_gap_ms"] = gap
                            else:
                                require(
                                    request["phase"] == "prefill",
                                    "unknown request role",
                                )
                                ttft = (
                                    times[0] - request["started_monotonic_s"]
                                ) * 1000
                                round_ttfts.append(ttft)
                                record["ttft_ms"] = ttft
                            requests.append(record)
                        expected_p = 3 if workload == "p12k-burst" else 1
                        require(
                            len(round_gaps) == 4 and len(round_ttfts) == expected_p,
                            "unexpected request coverage",
                        )
                        gaps.extend(round_gaps)
                        ttfts.extend(round_ttfts)
                        token_count += round_tokens
                        duration += dt
                        rounds.append(
                            {
                                "phase": phase,
                                "repeat": repeat["repeat"],
                                "duration_s": dt,
                                "output_tokens": round_tokens,
                                "throughput_tokens_s": round_tokens / dt,
                                "d_max_gap_p95_ms": percentile95(round_gaps),
                                "p_ttft_p95_ms": percentile95(round_ttfts),
                                "joint_100ms_1s": max(round_gaps) <= 100
                                and max(round_ttfts) <= 1000,
                                "requests": requests,
                            }
                        )
                results[role + "/" + workload] = {
                    "d_requests": len(gaps),
                    "p_requests": len(ttfts),
                    "d_max_gap_p95_ms": percentile95(gaps),
                    "p_ttft_p95_ms": percentile95(ttfts),
                    "output_tokens": token_count,
                    "duration_s": duration,
                    "throughput_tokens_s": token_count / duration,
                    "d_100ms_passed": sum(g <= 100 for g in gaps),
                    "p_1s_passed": sum(t <= 1000 for t in ttfts),
                    "joint_passed": sum(r["joint_100ms_1s"] for r in rounds),
                    "rounds": rounds,
                }
    changes = {}
    for workload in ("p8k", "p12k", "p12k-burst"):
        native, guard = (
            results["native16k/" + workload],
            results["guard-v2/" + workload],
        )
        changes[workload] = {
            "d_gap_reduction_pct": 100
            * (1 - guard["d_max_gap_p95_ms"] / native["d_max_gap_p95_ms"]),
            "p_ttft_penalty_pct": 100
            * (guard["p_ttft_p95_ms"] / native["p_ttft_p95_ms"] - 1),
            "throughput_change_pct": 100
            * (guard["throughput_tokens_s"] / native["throughput_tokens_s"] - 1),
        }
    return {
        "schema_version": 1,
        "archive_url": ARCHIVE_URL,
        "manifest_url": MANIFEST_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_bytes": ARCHIVE_BYTES,
        "producer_sha256": PRODUCER_SHA256,
        "method": {
            "d_gap": "maximum adjacent token-bearing SSE event arrival interval over each complete D request, in ms",
            "p_ttft": "actual request dispatch to first token-bearing SSE arrival, in ms",
            "p95": "linear interpolation at (n-1)*0.95 after pooling all measured requests in six rounds",
            "throughput": "sum of D+P output tokens / sum of complete measured window durations",
            "selection": "native16k and guard-v2 initial/confirm; three measured repeats per phase; warmups excluded",
            "validation": "whole archive and 12 member hashes; raw SSE hash and event payload sequence; one token per event; prompt IDs, output IDs/counts, usage and transport",
        },
        "sources": sources,
        "results": results,
        "relative_changes": changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="local copy of pinned muxpd.tar.gz")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("tp8-recomputed.json")
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare exact stable JSON bytes with --output; write nothing",
    )
    args = parser.parse_args()
    try:
        result = recompute(args.archive)
        encoded = (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        if args.check:
            require(
                args.output.read_bytes() == encoded,
                "recomputed JSON differs from --output",
            )
            print(
                "PASS: archive, 12 members, 36 measured rounds, 204 requests; exact JSON matches"
            )
        else:
            args.output.write_bytes(encoded)
            print(f"Wrote {args.output}: 12 members, 36 measured rounds, 204 requests")
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
