"""Offline, versioned evidence review; no hardware execution or original edits."""

import csv
import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/vllm-hust-midterm-20261019"
OUT = BASE / "milestone-v15/remaining-audit-r1.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def members(path):
    with tarfile.open(path) as bundle:
        return {m.name: bundle.extractfile(m).read() for m in bundle if m.isfile()}


def quantile(values, p):
    """Nearest rank, declared for this descriptive reanalysis only."""
    return sorted(values)[math.ceil(p * len(values)) - 1] if values else None


def utilization(rows, start, end):
    points = []
    for row in rows:
        cards = {}
        previous = None
        for line in row["raw"].splitlines():
            match = re.match(r"\|\s*(\d+)\s+910B2\s*\|", line)
            if match:
                previous = match[1]
            elif previous and "0000:" in line:
                fields = line.split("|")
                cards[previous] = float(fields[3].split()[0])
                previous = None
        if cards:
            points.append((row["end_monotonic"], cards))
    points.sort()
    sums, covered, gaps = {}, {}, []
    # Left-held samples, clipped to the request envelope; never extend last sample.
    for (t, cards), (nxt, _) in zip(points, points[1:]):
        width = max(0, min(end, nxt) - max(start, t))
        if not width:
            continue
        gaps.append(nxt - t)
        for card, value in cards.items():
            sums[card] = sums.get(card, 0) + value * width
            covered[card] = covered.get(card, 0) + width
    return {
        "method": "Left hold at sampling command end; clipped to envelope; no extension beyond final sample",
        "per_card_percent": {c: sums[c] / covered[c] for c in sums},
        "covered_seconds": covered,
        "maximum_sampling_gap_seconds": max(gaps) if gaps else None,
        "node_identity": "Same archived execution host; card IDs from npu-smi, not a cross-node aggregate",
        "scope": "Retrospective sampled estimate, not continuous instrumentation or current fixed-B0 acceptance",
    }


def costs():
    directory = BASE / "m06-paired-20261021"
    manifest = read_json(directory / "archive.json")
    indexed = {r["name"]: r for r in manifest["files"]}
    arms = {}
    for arm in ("b0", "b1"):
        name = f"m06-prechecked-{arm}-r1-evidence.tar.gz"
        raw = (directory / name).read_bytes()
        assert sha(raw) == indexed[name]["sha256"]
        files = members(directory / name)
        prefix = f"m01-recovery-model-m06-prechecked-{arm}-r1/"
        bench = json.loads(files[prefix + "mmlu-pro.json"])
        receipt = json.loads(files[prefix + "receipt.json"])
        n = bench["completed"]
        assert n == 200 and bench["failed"] == 0 and not any(bench["errors"])
        for key in ("start_times", "latencies", "input_lens", "output_lens", "errors"):
            assert len(bench[key]) == n
        start = min(bench["start_times"])
        end = max(
            t + d for t, d in zip(bench["start_times"], bench["latencies"], strict=True)
        )
        assert (
            receipt["bench_started_monotonic"]
            < start
            < end
            < receipt["bench_ended_monotonic"]
        )
        output = sum(bench["output_lens"])
        assert output == bench["total_output_tokens"]
        samples = [
            json.loads(line)
            for line in files[prefix + "device-samples.jsonl"].splitlines()
        ]
        k = receipt["reserved_cards"]
        arms[arm] = {
            "identity": "0.28.1/0.25.1rc2 development control, not fixed B0"
            if arm == "b0"
            else "E24/R36 + KV recomputation + Prefill Microflow B1 on development stack",
            "archive_path": str((directory / name).relative_to(ROOT)),
            "archive_sha256": sha(raw),
            "request_json_sha256": sha(files[prefix + "mmlu-pro.json"]),
            "interval_monotonic": [start, end],
            "interval_seconds": end - start,
            "interval_method": "Retrospective first request start to last successful response; differs from original benchmark duration, which remains unchanged",
            "successful_requests": n,
            "cross_interval_requests_in_this_burst": 0,
            "input_tokens": sum(bench["input_lens"]),
            "new_input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": output,
            "reserved_cards": k,
            "card_seconds_per_output_token": k * (end - start) / output,
            "yuan_per_million_coefficient_times_P_over_D": 1e6
            * k
            * (end - start)
            / output
            / 86400,
            "monthly_fee_P": None,
            "month_days_D": None,
            "yuan_per_million_output_tokens": None,
            "utilization": utilization(samples, start, end),
        }
    return {
        "arms": arms,
        "equal_fee_basis_resource_reduction_percent": 100
        * (
            1
            - arms["b1"]["card_seconds_per_output_token"]
            / arms["b0"]["card_seconds_per_output_token"]
        ),
        "current_passed": None,
        "remaining": [
            "Fixed-B0 representative comparison with applicable quality/SLO",
            "Documented all-in fee P and month-day D, not chart display assumptions",
            "Request-level new/cache-hit input split; retrospective utilization method is not a frozen acceptance protocol",
        ],
    }


def swe_windows():
    path = (
        ROOT
        / "reports/swe-bench-pro-real-agent-threeway128-1h-20261022-r2/data/normal-client-returns.csv.gz"
    )
    raw = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode())))
    result = []
    for arm in sorted({r["arm"] for r in rows}):
        for start, end in ((0, 600), (600, 1800), (1800, 3600), (0, 3600)):
            selected = [
                r for r in rows if r["arm"] == arm and start <= float(r["t"]) < end
            ]
            long = [r for r in selected if int(r["input"]) + int(r["output"]) >= 16384]
            result.append(
                {
                    "arm": arm,
                    "window_seconds": [start, end],
                    "normal_returns": len(selected),
                    "actual_context_ge16k_returns": len(long),
                    "output_tokens": sum(int(r["output"]) for r in selected),
                    "delivered_output_tokens_per_second": sum(
                        int(r["output"]) for r in selected
                    )
                    / (end - start),
                    "e2e_p95_seconds": quantile(
                        [float(r["latency"]) for r in selected], 0.95
                    ),
                    "e2e_p99_seconds": quantile(
                        [float(r["latency"]) for r in selected], 0.99
                    ),
                    "ttft_p95_seconds": None,
                    "tpot_p99_seconds": None,
                }
            )
    return {
        "source_path": str(path.relative_to(ROOT)),
        "sha256": sha(raw),
        "windows": result,
        "scope": "Existing one-hour real-agent records; completion-time windows and nearest-rank E2E quantiles. Actual input+output context, not configured maximum. E2E is not TTFT/TPOT; mixed context is not a controlled 4K/16K pair; censored requests absent from this normal-return extract remain in original report. No five-task pass.",
        "passed": None,
    }


def build():
    accounting = BASE / "m01-recovery/boundary-accounting-20261023"
    manifest = read_json(accounting / "manifest.json")
    records = {}
    for entry in manifest["archives"]:
        path = accounting / entry["archive"]
        assert sha(path.read_bytes()) == entry["sha256"]
        files = members(path)
        assert {k: sha(v) for k, v in files.items()} == entry["members"]
        records.update(
            {k: json.loads(v) for k, v in files.items() if k.endswith(".json")}
        )
    matrix = read_json(ROOT / "website-publication/data/dataset-matrix-v1.json")
    inventory = [
        {
            k: r.get(k)
            for k in ("id", "dataset", "solution", "quality", "report", "limitations")
        }
        for r in matrix["records"]
    ]
    mmlu = read_json(
        ROOT / "reports/mmlu-quality018-20261022/quality018-completed-audit-r1.json"
    )
    return {
        "schema": "v15-remaining-audit/v1",
        "new_measurements_started": False,
        "source_baseline_commit": "3f317690161e996803ef3c297280604faa7d6b97",  # pragma: allowlist secret
        "m01": {
            "source_path": str((accounting / "manifest.json").relative_to(ROOT)),
            "derivations": records,
            "effective_flops": None,
            "reference_seconds": None,
            "passed": None,
        },
        "cost": costs(),
        "swe_continuous_records": swe_windows(),
        "quality_inventory_all_solutions": inventory,
        "mmlu_fixed_b0_completion": {
            "source_path": "reports/mmlu-quality018-20261022/quality018-completed-audit-r1.json",
            "counts": mmlu["counts"],
            "candidates": mmlu["comparisons"],
        },
        "stability_passed": None,
        "boundary": "Completion, budget-length termination, scoring unresolved and service anomaly are different states. This inventory does not assert absence of OOM/deadlock/silent truncation in logs not inspected.",
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
