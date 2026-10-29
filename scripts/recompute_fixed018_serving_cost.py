"""Recompute fixed018 serving/cost from allowlisted original scalar records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

from audit_acceptance_gaps import unit_cost

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/acceptance-gaps-20261029"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_npu(text):
    """Only the npu-smi device table, never process names or commands."""
    result = {}
    current = None
    for line in text.splitlines():
        device = re.match(r"\|\s*(\d+)\s+910B2\s*\|", line)
        if device:
            current = device.group(1)
        elif current is not None and re.search(
            r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d", line
        ):
            fields = line.split("|")
            result[current] = float(fields[3].split()[0])
            current = None
    return result


def project(read, receipt_sha):
    names = [
        "receipt.json",
        "execution-plan.json",
        "mmlu-pro.json",
        "monitor.jsonl",
        "boundary.jsonl",
    ]
    sources = {name: read(name) for name in names}
    if digest(sources["receipt.json"]) != receipt_sha:
        raise ValueError("Receipt is not the independently pinned measurement")
    manifest = {}
    for line in read("SHA256SUMS").decode().splitlines():
        checksum, name = line.split(maxsplit=1)
        manifest[name.lstrip("*./")] = checksum
    for name, data in sources.items():
        if manifest.get(name) != digest(data):
            raise ValueError("Sealed member SHA mismatch: " + name)
    plan = json.loads(sources["execution-plan.json"])
    measured = json.loads(sources["mmlu-pro.json"])
    boundary = json.loads(sources["boundary.jsonl"].splitlines()[-1])
    receipt = json.loads(sources["receipt.json"])
    rows = []
    arrays = [
        measured[k]
        for k in ("start_times", "ttfts", "itls", "input_lens", "output_lens")
    ]
    if not all(len(a) == 200 for a in arrays):
        raise ValueError("Expected complete200 request cohort")
    for index, (start, ttft, itls, incoming, outgoing) in enumerate(zip(*arrays)):
        decode = math.fsum(itls)
        rows.append(
            {
                "index": index,
                "start_perf": start,
                "ttft_seconds": ttft,
                "decode_seconds": decode,
                "input_tokens": incoming,
                "output_tokens": outgoing,
                "last_output_perf": start + ttft + decode,
            }
        )
    samples = []
    for line in sources["monitor.jsonl"].splitlines():
        record = json.loads(line)
        if "npu" in record:
            util = parse_npu(record["npu"])
            if set(util) != {str(i) for i in plan["reserved"]}:
                raise ValueError("NPU sample has unexpected/missing reserved cards")
            samples.append(
                {
                    "wall_time": record["time"],
                    "status": record["status"],
                    "ai_core_percent": util,
                }
            )
    clocks = []
    for call in boundary["calls"]:
        if call["path"] == "/collective_rpc":
            for worker in call["response"]["results"]:
                for key in ("clock_before", "clock_after"):
                    if key in worker:
                        c = worker[key]
                        clocks.append(
                            {
                                "phase": worker["phase"],
                                "rank": worker["rank"],
                                "wall_ns": c["wall_ns"],
                                "perf_before_ns": c["perf_before_ns"],
                                "perf_after_ns": c["perf_after_ns"],
                            }
                        )
    return {
        "source_sha256": {name: digest(data) for name, data in sources.items()},
        "source_manifest_sha256": digest(read("SHA256SUMS")),
        "verified_selected_manifest_members": len(sources),
        "run_id": plan["run_id"],
        "reserved_physical_devices": plan["reserved"],
        "participating_logical_devices": plan["participating"],
        "completed": measured["completed"],
        "failed": measured["failed"],
        "client_exit_code": receipt.get("client_exit_code"),
        "receipt_end_wall": receipt["ended"],
        "controller_sha256": plan.get("source_sha256", {}).get(
            "/root/m01-b0-accounting-20261024/run_b0_measurement.py"
        ),
        "materialized_input_sha256": plan["source_sha256"][
            "/root/gennova-evidence/20261019-midterm-aligned/materialized.200.jsonl"
        ],
        "synchronized_start_perf": boundary["start_perf_counter"],
        "synchronized_end_perf": boundary["end_perf_counter"],
        "synchronized_seconds": boundary["synchronized_duration_seconds"],
        "client_seconds": boundary["client_duration_seconds"],
        "original_client_aggregate_ms": {
            key: measured[key]
            for key in ("mean_e2el_ms", "mean_ttft_ms", "mean_tpot_ms")
        },
        "calls": [
            {
                "path": c["path"],
                "request": c["request"],
                "start_ns": c["start_ns"],
                "end_ns": c["end_ns"],
            }
            for c in boundary["calls"]
        ],
        "requests": rows,
        "npu_samples": samples,
        "clock_brackets": clocks,
    }


def lifecycle_utilization_bounds(arm):
    """Bound missing alignment under the explicit held-sample/no-clock-jump model."""
    # Public source checksum, not a credential. Read back from storage and matched
    # against the original execution plan before interpreting controller order.
    controller = "860ea97a8ef9d8445897768fbf12c750fec9d0a8f7a2a5ad17132dbd223bafcf"  # pragma: allowlist secret
    if arm.get("controller_sha256") != controller:
        return None
    samples = arm["npu_samples"]
    lower = max(s["wall_time"] for s in samples if s["status"] == "starting")
    upper = arm["receipt_end_wall"]
    duration = arm["synchronized_seconds"]
    if upper - lower < duration or samples[-1]["wall_time"] > upper:
        raise ValueError("Invalid controller lifecycle bounds")
    candidates = {lower, upper - duration}
    candidates.update(
        s["wall_time"] + shift
        for s in samples
        for shift in (0, -duration)
        if lower <= s["wall_time"] + shift <= upper - duration
    )

    def average(start, unknown):
        total = 0.0
        for left, right in zip(samples, samples[1:]):
            width = max(
                0,
                min(start + duration, right["wall_time"])
                - max(start, left["wall_time"]),
            )
            total += width * statistics.mean(left["ai_core_percent"].values())
        # No held extrapolation beyond the final observed sample: use0..100%.
        total += (
            max(0, start + duration - max(start, samples[-1]["wall_time"])) * unknown
        )
        return total / duration

    return {
        "mean_percent_lower": min(average(s, 0) for s in candidates),
        "mean_percent_upper": max(average(s, 100) for s in candidates),
        "possible_start_wall": [lower, upper - duration],
        "latest_end_wall": upper,
        "assumptions": "No wall-clock jump; left-held command-completion NPU samples. Controller emits final starting sample before client launch and receipt ended after drain. Extremize all possible fixed-duration windows; unsampled tail ranges0..100%. Not physical-utilization confidence bounds or exact same-window measurement.",
    }


def time_weighted_utilization(arm):
    samples = arm["npu_samples"]
    if len(samples) < 2 or any(
        a["wall_time"] >= b["wall_time"] for a, b in zip(samples, samples[1:])
    ):
        raise ValueError("NPU sample timestamps must strictly increase")
    cards = {str(i) for i in arm["reserved_physical_devices"]}
    for sample in samples:
        if set(sample["ai_core_percent"]) != cards or any(
            not 0 <= v <= 100 for v in sample["ai_core_percent"].values()
        ):
            raise ValueError("Invalid reserved-card utilization sample")
    clocks = arm["clock_brackets"]
    if not clocks:
        return {
            "status": "missing_same_run_wall_perf_bridge",
            "mean_percent": None,
            "per_card_percent": None,
            "conditional_lifecycle_bounds": lifecycle_utilization_bounds(arm),
            "reason": "Original monitor uses wall time, request/sync window uses perf_counter; no same-run clock bracket was retained. Do not borrow the B1 clock offset or label all measuring-stage samples as the exact window.",
        }
    offsets = [
        (c["wall_ns"] - (c["perf_before_ns"] + c["perf_after_ns"]) / 2) / 1e9
        for c in clocks
    ]
    offset = statistics.median(offsets)
    start = arm["synchronized_start_perf"] + offset
    end = arm["synchronized_end_perf"] + offset
    if samples[0]["wall_time"] > start or samples[-1]["wall_time"] < end:
        raise ValueError("NPU samples do not bracket the synchronized window")
    weighted = {str(i): 0.0 for i in arm["reserved_physical_devices"]}
    covered = 0.0
    gaps = []
    for left, right in zip(samples, samples[1:]):
        a, b = max(start, left["wall_time"]), min(end, right["wall_time"])
        if b <= a:
            continue
        gaps.append(right["wall_time"] - left["wall_time"])
        covered += b - a
        for card in weighted:
            weighted[card] += left["ai_core_percent"][card] * (b - a)
    if not math.isclose(covered, end - start, abs_tol=1e-6):
        raise ValueError("Incomplete NPU time coverage")
    means = {k: v / covered for k, v in weighted.items()}
    return {
        "status": "retrospective_sample_estimate",
        "mean_percent": statistics.mean(means.values()),
        "per_card_percent": means,
        "wall_interval": [start, end],
        "covered_seconds": covered,
        "max_sampling_gap_seconds": max(gaps),
        "clock_offset_spread_seconds": max(offsets) - min(offsets),
        "method": "Same-run clock-bracket median offset; command-completion timestamps, left-held samples clipped to the same synchronized cost interval. Sampling error is not a confidence interval.",
    }


def calculate(inputs):
    price_bytes = (OUTPUT / "pricing-and-conditional-cost.json").read_bytes()
    price = json.loads(price_bytes)["price_basis"]
    result = {
        "schema": "fixed018-serving-cost-reuse/v1",
        "new_model_runs": 0,
        "price_basis": "pricing-and-conditional-cost.json",
        "price_basis_sha256": digest(price_bytes),
        "arms": {},
    }
    for name in ("B0", "B1"):
        arm = inputs[name]
        rows = arm["requests"]
        if (
            arm["completed"] != 200
            or arm["failed"] != 0
            or arm["client_exit_code"] != 0
            or len(rows) != 200
            or [r["index"] for r in rows] != list(range(200))
        ):
            raise ValueError("Incomplete successful cohort")
        if any(
            not math.isfinite(r[k]) or r[k] < 0
            for r in rows
            for k in (
                "start_perf",
                "ttft_seconds",
                "decode_seconds",
                "input_tokens",
                "output_tokens",
                "last_output_perf",
            )
        ):
            raise ValueError("Invalid request scalar")
        start, end = arm["synchronized_start_perf"], arm["synchronized_end_perf"]
        if not math.isclose(end - start, arm["synchronized_seconds"], abs_tol=1e-8):
            raise ValueError("Inconsistent synchronized boundary")
        if any(
            not start <= r["start_perf"] <= r["last_output_perf"] <= end for r in rows
        ):
            raise ValueError("Request crosses the synchronized interval")
        tokens = sum(r["output_tokens"] for r in rows)
        if {r["output_tokens"] for r in rows} != {256}:
            raise ValueError(
                "Aggregate TPOT reconstruction requires the observed uniform256 outputs"
            )
        aggregate = arm["original_client_aggregate_ms"]
        mean_ttft = statistics.mean(r["ttft_seconds"] for r in rows) * 1000
        mean_tpot = (aggregate["mean_e2el_ms"] - mean_ttft) / 255
        if not math.isclose(mean_ttft, aggregate["mean_ttft_ms"], abs_tol=1e-7):
            raise ValueError("TTFT aggregate does not match per-request data")
        if not math.isclose(mean_tpot, aggregate["mean_tpot_ms"], abs_tol=1e-7):
            raise ValueError("Equal-request TPOT aggregate mismatch")
        cards = len(arm["reserved_physical_devices"])
        seconds = end - start
        result["arms"][name] = {
            "run_id": arm["run_id"],
            "requests": len(rows),
            "successful_output_tokens": tokens,
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "all_requests_in_interval": True,
            "reserved_cards": cards,
            "reserved_cards_verified": True,
            "synchronized_seconds": seconds,
            "output_tokens_per_second": tokens / seconds,
            "equal_request_mean_TTFT_ms": mean_ttft,
            "equal_request_mean_TPOT_ms": mean_tpot,
            "compute_rental_CNY_per_million_output_tokens": unit_cost(
                price["monthly_per_card_equivalent_CNY"],
                price["analytical_month_days"],
                cards,
                seconds,
                tokens,
            ),
            "same_interval_ai_core_utilization": time_weighted_utilization(arm),
        }
    if (
        inputs["B0"]["reserved_physical_devices"]
        != inputs["B1"]["reserved_physical_devices"]
    ):
        raise ValueError("Different physical allocation")
    if (
        not inputs["B0"]["materialized_input_sha256"]
        or inputs["B0"]["materialized_input_sha256"]
        != inputs["B1"]["materialized_input_sha256"]
    ):
        raise ValueError("Different materialized input identity")
    if [r["input_tokens"] for r in inputs["B0"]["requests"]] != [
        r["input_tokens"] for r in inputs["B1"]["requests"]
    ]:
        raise ValueError("Actual ordered input token lengths differ")
    b0, b1 = result["arms"]["B0"], result["arms"]["B1"]
    result["improvements_percent"] = {
        "synchronized_window_output_throughput": (
            b1["output_tokens_per_second"] / b0["output_tokens_per_second"] - 1
        )
        * 100,
        "equal_request_mean_TTFT": (
            1 - b1["equal_request_mean_TTFT_ms"] / b0["equal_request_mean_TTFT_ms"]
        )
        * 100,
        "equal_request_mean_TPOT": (
            1 - b1["equal_request_mean_TPOT_ms"] / b0["equal_request_mean_TPOT_ms"]
        )
        * 100,
        "same_rate_compute_rental_cost": (
            1
            - b1["compute_rental_CNY_per_million_output_tokens"]
            / b0["compute_rental_CNY_per_million_output_tokens"]
        )
        * 100,
    }
    result["numeric_cost_reduction_at_least_10_percent"] = (
        result["improvements_percent"]["same_rate_compute_rental_cost"] >= 10
    )
    result["M06_pass"] = None
    result["utilization_improvement_pass"] = None
    result["limits"] = [
        "Fixed B0 is0.18/0.18; current corrected B1 is0.28.1/Ascend0.25.1rc2 development, not a same-version ablation.",
        "Both use the original same200 out256 workload and same two reserved physical cards; startup excluded, in-window host accounting and synchronization retained without subtracting overhead.",
        "Cost interval is the original synchronized envelope, containing every successful request and end drain; request-only envelope is not silently substituted.",
        "TTFT is independently averaged from all200 request values. All outputs are exactly256 tokens, so equal-request TPOT=(original meanE2E-recomputed meanTTFT)/255 exactly matches the original client aggregate. Chunk intervals only locate the last output, not final protocol completion.",
        "Public cloud compute-rental estimate excludes separately billed disk/network and is not actual comprehensive cost.",
        "B0 has no exact same-window wall/perf alignment. Conditional lifecycle bounds can test sensitivity under a stated sampling/no-clock-jump model; they do not invent a measured point or formal utilization verdict.",
        "Out8192 quality scores remain a separate measurement. The combined quality/SLO and utilization acceptance is not inferred from positive scalar improvements.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-raw", type=Path)
    parser.add_argument("--b1-archive", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = OUTPUT / "recovered-serving-inputs.json"
    if args.b0_raw or args.b1_archive:
        if not (args.b0_raw and args.b1_archive) or args.check:
            parser.error(
                "Both private sources required for projection; --check uses committed public projection"
            )
        with tarfile.open(args.b1_archive) as archive:

            def read_b1(name):
                return archive.extractfile(name).read()

            inputs = {
                "schema": "allowlisted-serving-scalars/v1",
                "privacy": "No prompts, generated text, reasoning, environment or full source snapshots; scalar request timings/token counts, device utilization and hashes only.",
                "B0": project(
                    lambda n: (args.b0_raw / n).read_bytes(),
                    "7804976d9b53cf441eb8e8d0a3e438f83fe363681f9f2661758ba50877da774d",  # pragma: allowlist secret
                ),  # pragma: allowlist secret
                "B1": project(
                    read_b1,
                    "ada9a8d88c42ad2f507fb0e1943e1c6bb0642e40a0718ba29d6e97637764294c",  # pragma: allowlist secret
                ),  # pragma: allowlist secret
            }
        target.write_text(json.dumps(inputs, indent=2) + "\n")
    inputs = json.loads(target.read_bytes())
    result = calculate(inputs)
    result["public_input_sha256"] = digest(target.read_bytes())
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    summary = OUTPUT / "recovered-serving-summary.json"
    if args.check:
        if summary.read_text() != rendered:
            raise SystemExit("Derived serving/cost summary is stale")
        print(
            "Serving/cost arithmetic matches the committed allowlisted original scalar inputs"
        )
    else:
        summary.write_text(rendered)
        print(rendered)


if __name__ == "__main__":
    main()
