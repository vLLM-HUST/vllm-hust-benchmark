"""Recompute biological native-task measurements from their archived records."""

from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def latency_statistics(values, quantiles):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Latency samples must be positive finite durations")
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "standard_deviation_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "quantiles_ms": {
            str(q): float(np.quantile(values, q, method="linear")) for q in quantiles
        },
    }


def cluster_interval(medians):
    generator = random.Random(1729)
    means = [
        statistics.mean(generator.choices(medians, k=len(medians))) for _ in range(2000)
    ]
    return {
        "method": "percentile bootstrap of independent-start medians",
        "confidence": 0.95,
        "resamples": 2000,
        "seed": 1729,
        "independent_starts": len(medians),
        "lower_ms": float(np.quantile(means, 0.025, method="linear")),
        "upper_ms": float(np.quantile(means, 0.975, method="linear")),
    }


def validate_request(row, ordinal):
    duration = (row["finished_ns"] - row["started_ns"]) / 1_000_000
    if (
        row["status"] != "passed"
        or row.get("error") is not None
        or not math.isfinite(row["elapsed_ms"])
        or duration <= 0
    ):
        raise ValueError("Native request failed or has an invalid duration")
    if abs(duration - row["elapsed_ms"]) > 1e-6 or row["prior_task_calls"] != ordinal:
        raise ValueError("Request clock or execution order is inconsistent")
    if row["timing_scope"] != "prepared_native_call_to_cpu_output":
        raise ValueError("Incompatible measurement scope")
    for memory in [row["memory_before"], row["memory_after"]]:
        if any(
            type(memory[name]) is not int or memory[name] < 0
            for name in [
                "allocated_bytes",
                "reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            ]
        ):
            raise ValueError("Invalid worker memory observation")
        if (
            memory["peak_allocated_bytes"] < memory["allocated_bytes"]
            or memory["reserved_bytes"] < memory["allocated_bytes"]
        ):
            raise ValueError("Worker memory observations contradict each other")


def worker_exited(identity):
    path = Path("/proc") / str(identity["pid"]) / "stat"
    try:
        fields = path.read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return True
    return int(fields[19]) != identity["pid_start_ticks"] or fields[0] == "Z"


def check_outputs(directory, suite, cases, rows, native, registry=None):
    from safetensors.numpy import load_file
    from experiments.benchmark.contracts import file_digest
    from experiments.benchmark.representative.native import work_counts
    from experiments.benchmark.representative.reference import (
        compare_output,
        materialize,
    )

    comparison = read_json(directory / "comparison.json")
    if registry is not None and comparison.get("registry") != registry.identity():
        raise ValueError("Reference registry differs from the archived registry")
    if not comparison["passed"]:
        raise ValueError("Independent reference comparison failed")
    for filename, expected in [
        ("outputs.safetensors", native["tensors_sha256"]),
        ("reference.safetensors", comparison["tensors_sha256"]),
    ]:
        if file_digest(directory / filename) != expected:
            raise ValueError("Numerical evidence file changed")
    actual_arrays = load_file(str(directory / "outputs.safetensors"))
    expected_arrays = load_file(str(directory / "reference.safetensors"))
    comparisons = {item["request_id"]: item for item in comparison["comparisons"]}
    if len(comparisons) != len(comparison["comparisons"]) or set(comparisons) != {
        row["request_id"] for row in rows
    }:
        raise ValueError("Numerical comparisons omit or duplicate requests")
    count = 0
    for row in rows:
        case = cases[row["case_id"]]
        identity = native["inputs"][row["case_id"]]
        if (
            identity["sequences"] != case["sequences"]
            or identity["source_samples"] != case["samples"]
        ):
            raise ValueError("Recorded input differs from the frozen dataset windows")
        actual = materialize(row["output"], actual_arrays)
        if work_counts(case, identity, actual, registry) != row["work"]:
            raise ValueError("Work counters differ from actual inputs and outputs")
        recorded = comparisons[row["request_id"]]
        expected = materialize(
            comparison["references"][recorded["reference_id"]], expected_arrays
        )
        diagnostic = materialize(native["diagnostics"].get(case["id"]), actual_arrays)
        compare = (
            registry.bind(case["profile_id"], case["task_id"]).compare
            if registry is not None
            else compare_output
        )
        metrics = compare(case, identity, actual, expected, diagnostic, suite["gates"])
        if (
            not recorded["passed"]
            or recorded["case_id"] != row["case_id"]
            or metrics != recorded["metrics"]
            or not all(item["passed"] for item in metrics)
        ):
            raise ValueError(
                "Recomputed numerical results disagree with their recorded acceptance"
            )
        count += len(metrics)
    return count


def summarize(root, *, numeric=True):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    suite = read_json(root / "suite.json")
    registry = None
    if "registry" in manifest:
        from vllm_hust_benchmark.bio_registry import Registry

        registry = Registry.load(root / "registry.json")
        if registry.identity() != manifest["registry"]:
            raise ValueError("Archived registry identity differs from the run manifest")
    cases = {case["id"]: case for case in manifest["resolved_cases"]}
    protocol = manifest["effective_protocol"]
    definitions = [item for item in manifest["runs"] if not item["fault"]]
    case_runs, engine_runs, faults = {}, [], []
    numerical_checks = 0
    seen_requests = set()
    for item in manifest["runs"]:
        directory = root / item["directory"]
        native = read_json(directory / "native.json")
        if registry is not None and native.get("registry") != registry.identity():
            raise ValueError("Native registry differs from the archived registry")
        if not native["closed"] or not worker_exited(native["worker_initial"]):
            raise ValueError("An owned native worker remains active")
        if item["fault"]:
            expected = "benchmark_injected_host_failure_after_native_output"
            if (
                native["passed"]
                or native["error"] != {"type": "RuntimeError", "message": expected}
                or native["requests"] != 1
            ):
                raise ValueError(
                    "The expected host exception and cleanup were not observed"
                )
            faults.append(
                {
                    "profile_id": item["profile_id"],
                    "closed": True,
                    "kind": "host driver exception after a real native result",
                }
            )
            continue
        if (
            not native["passed"]
            or native["error"] is not None
            or native["protocol"] != protocol
        ):
            raise ValueError(
                "Native run failed or used a different repetition protocol"
            )
        if (
            native["profile_id"] != item["profile_id"]
            or native["start_index"] != item["start_index"]
        ):
            raise ValueError("Engine identity disagrees with the run plan")
        rows = [
            json.loads(line)
            for line in (directory / "requests.jsonl").read_text().splitlines()
        ]
        if len(rows) != native["requests"]:
            raise ValueError("Request count differs from the engine manifest")
        for ordinal, row in enumerate(rows):
            validate_request(row, ordinal)
            if ordinal and row["started_ns"] < rows[ordinal - 1]["finished_ns"]:
                raise ValueError("Serial measurement requests overlap")
            if row["request_id"] in seen_requests or row["case_id"] not in cases:
                raise ValueError("Duplicate request or unknown case")
            seen_requests.add(row["request_id"])
            if (
                row["profile_id"] != item["profile_id"]
                or row["start_index"] != item["start_index"]
            ):
                raise ValueError("Request identity differs from its engine")
        expected_cases = {
            key
            for key, case in cases.items()
            if case["profile_id"] == item["profile_id"]
        }
        observed = {row["case_id"] for row in rows if row["phase"] != "recovery"}
        if observed != expected_cases:
            raise ValueError("Native execution omitted planned cases")
        expected_probes = (
            registry.profiles[item["profile_id"]]["rejection_probes"]
            if registry is not None
            else None
        )
        expected_count = len(expected_probes) if expected_probes is not None else 8
        names = [probe["name"] for probe in native["rejections"]]
        if (
            len(names) != expected_count
            or len(set(names)) != expected_count
            or (expected_probes is not None and names != expected_probes)
            or (
                registry is not None
                and native.get("expected_rejections") != expected_probes
            )
        ):
            raise ValueError("Native input rejection coverage is incomplete")
        recovery = [row for row in rows if row["phase"] == "recovery"]
        if len(recovery) != expected_count or sorted(
            row["repetition"] for row in recovery
        ) != list(range(expected_count)):
            raise ValueError("A rejected input lacks a successful recovery request")
        stop_required = (
            any(cases[key].get("require_early_stop", False) for key in expected_cases)
            if registry is not None
            else item["profile_id"] == "progen2-small"
        )
        if stop_required and not any(
            row["work"].get("early_stop_sequences", 0) for row in rows
        ):
            raise ValueError(
                "No native stop token terminated a generated sequence early"
            )
        for rejection in native["rejections"]:
            if rejection["error"]["type"] != "ValueError":
                raise ValueError("Unexpected exception type in an input rejection")
            growth = (
                rejection["memory_after"]["allocated_bytes"]
                - rejection["memory_before"]["allocated_bytes"]
            )
            if growth > protocol["memory_growth_limit_bytes"]:
                raise ValueError(
                    "Worker memory grew after an input rejection and recovery"
                )
        for case_id in sorted(expected_cases):
            subset = [
                row
                for row in rows
                if row["case_id"] == case_id and row["phase"] != "recovery"
            ]
            counts = Counter(row["phase"] for row in subset)
            expected_counts = {
                "first": 1,
                "warmup": protocol["warmup_repetitions"],
                "steady": protocol["steady_repetitions"],
            }
            if dict(counts) != expected_counts:
                raise ValueError(
                    "First-call, warmup or steady-state samples are missing"
                )
            expected_order = (
                ["first"]
                + ["warmup"] * protocol["warmup_repetitions"]
                + ["steady"] * protocol["steady_repetitions"]
            )
            if [row["phase"] for row in subset] != expected_order:
                raise ValueError(
                    "First-call, warmup and steady-state order differs from the protocol"
                )
            for phase, expected_count in expected_counts.items():
                if sorted(
                    row["repetition"] for row in subset if row["phase"] == phase
                ) != list(range(expected_count)):
                    raise ValueError(
                        "Repeated sample indices are missing or duplicated"
                    )
            warmup = [row for row in subset if row["phase"] == "warmup"]
            steady = [row for row in subset if row["phase"] == "steady"]
            growth = (
                steady[-1]["memory_after"]["allocated_bytes"]
                - warmup[-1]["memory_after"]["allocated_bytes"]
            )
            if growth > protocol["memory_growth_limit_bytes"]:
                raise ValueError(
                    "Allocated worker memory grows during steady-state calls"
                )
            window = (steady[-1]["finished_ns"] - steady[0]["started_ns"]) / 1e9
            latencies = [row["elapsed_ms"] for row in steady]
            work = {
                name: sum(row["work"][name] for row in steady)
                for name in steady[0]["work"]
            }
            result = {
                "start_index": item["start_index"],
                "first_call_ms": subset[0]["elapsed_ms"],
                "first_call_is_engine_first_task": subset[0]["prior_task_calls"] == 0,
                "steady": latency_statistics(latencies, protocol["quantiles"]),
                "steady_window_seconds": window,
                "active_call_seconds": sum(latencies) / 1000,
                "work": work,
                "allocated_growth_bytes": growth,
                "peak_allocated_bytes": max(
                    row["memory_after"]["peak_allocated_bytes"] for row in subset
                ),
                "peak_reserved_bytes": max(
                    row["memory_after"]["peak_reserved_bytes"] for row in subset
                ),
            }
            rss = [row["memory_after"].get("rss_bytes") for row in subset]
            result["max_observed_worker_rss_bytes"] = (
                max(rss) if all(value is not None for value in rss) else None
            )
            case_runs.setdefault(case_id, []).append((result, latencies))
        if numeric:
            numerical_checks += check_outputs(
                directory, suite, cases, rows, native, registry
            )
        engine_runs.append(
            {
                "profile_id": item["profile_id"],
                "start_index": item["start_index"],
                "engine_construct_ms": native["engine_construct_ms"],
                "requests": len(rows),
                "rejections": len(native["rejections"]),
                "closed": native["closed"],
            }
        )
    for profile in {case["profile_id"] for case in cases.values()}:
        starts = [
            item["start_index"] for item in definitions if item["profile_id"] == profile
        ]
        if sorted(starts) != list(range(protocol["independent_starts"])):
            raise ValueError("Independent engine starts are missing or duplicated")
        if sum(item["profile_id"] == profile for item in faults) != 1:
            raise ValueError("An owned engine lacks an exception-cleanup test")
    summaries = []
    for case_id in sorted(cases):
        runs = case_runs[case_id]
        window = sum(result["steady_window_seconds"] for result, _ in runs)
        work = {
            name: sum(result["work"][name] for result, _ in runs)
            for name in runs[0][0]["work"]
        }
        durations = [value for _, values in runs for value in values]
        summaries.append(
            {
                "case_id": case_id,
                "profile_id": cases[case_id]["profile_id"],
                "task_id": cases[case_id]["task_id"],
                "scale": cases[case_id]["scale"],
                "independent_starts": len(runs),
                "per_start": [result for result, _ in runs],
                "pooled_steady": latency_statistics(durations, protocol["quantiles"]),
                "start_median_interval": cluster_interval(
                    [statistics.median(values) for _, values in runs]
                ),
                "sum_steady_windows_seconds": window,
                "rates_per_second": {
                    name: count / window for name, count in work.items()
                },
                "work": work,
            }
        )
    return {
        "schema_version": 1,
        "suite_id": suite["id"],
        "pilot": manifest["pilot"],
        "protocol": protocol,
        "cases": summaries,
        "engines": engine_runs,
        "faults": faults,
        "numerical_checks": numerical_checks,
        "request_count": len(seen_requests),
        "rejected_inputs": sum(item["rejections"] for item in engine_runs),
        "throughput_scope": "sum of steady measurement windows including inter-call observation gaps",
    }


def render_markdown(summary):
    lines = [
        "# Biological representative-model Benchmark",
        "",
        f"Suite `{summary['suite_id']}`",
        "",
        f"Measured requests {summary['request_count']}. Numerical checks {summary['numerical_checks']}.",
        "",
        "| Case | Starts | Steady samples | Mean ms | p50 ms | p90 ms | Jobs/s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in summary["cases"]:
        values = case["pooled_steady"]
        lines.append(
            f"| {case['case_id']} | {case['independent_starts']} | {values['count']} | {values['mean_ms']:.3f} | {values['quantiles_ms']['0.5']:.3f} | {values['quantiles_ms']['0.9']:.3f} | {case['rates_per_second']['jobs']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Durations cover prepared public native calls through CPU output delivery.",
            "Throughput uses the recorded steady windows and includes observation gaps.",
            "Per-start samples, memory, first-call state and cluster intervals are in summary.json.",
            "",
        ]
    )
    return "\n".join(lines)
