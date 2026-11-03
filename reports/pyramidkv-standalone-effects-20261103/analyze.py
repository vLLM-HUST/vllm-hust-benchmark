"""Recompute descriptive statistics from recorded requests, without model calls."""

import math
import re
import statistics
import string
from collections import Counter

from evidence import digest, read_json, require

COUNTERS = (
    "num_preemptions",
    "spec_decode_num_draft_tokens",
    "spec_decode_num_accepted_tokens",
)


def qa_f1(prediction: str, answers: list[str]) -> float:
    def normalized(text):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            text.lower().translate(str.maketrans("", "", string.punctuation)),
        ).split()

    pred = normalized(prediction)
    best = 0.0
    for answer in answers:
        gold = normalized(answer)
        common = sum((Counter(pred) & Counter(gold)).values())
        if common:
            best = max(best, 2 * common / (len(pred) + len(gold)))
    return best


def distribution(values: list[float]) -> dict:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Empty or nonfinite measurement distribution")
    ordered = sorted(values)

    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "mean": statistics.mean(values),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
    }


def summarize_groups(batches: list[tuple[list[dict], list[dict]]]) -> dict:
    selected, rates, elapsed = [], [], []
    for groups, rows in batches:
        by_id = {row["case_id"]: row for row in rows}
        for group in groups:
            requests = [by_id[key] for key in group["case_ids"]]
            wall = group["wall_seconds"]
            if not math.isfinite(wall) or wall <= 0:
                raise ValueError("Invalid cohort wall time")
            good = [row for row in requests if row["success"]]
            rates.append(sum(row["usage"]["completion_tokens"] for row in good) / wall)
            elapsed.append(wall)
            selected.extend(requests)
    good = [row for row in selected if row["success"]]
    tokens = sum(row["usage"]["completion_tokens"] for row in good)
    if not elapsed:
        raise ValueError("Empty cohort selection")
    result = {
        "cohorts": len(rates),
        "successful_requests": len(good),
        "failed_requests": len(selected) - len(good),
        "output_tokens": tokens,
        "wall_seconds": math.fsum(elapsed),
        "output_tokens_per_second": tokens / math.fsum(elapsed),
        "cohort_output_tokens_per_second": rates,
        "cached_prompt_tokens": sum(
            row["usage"]["prompt_tokens_details"]["cached_tokens"] for row in good
        ),
    }
    for metric in ("ttft_seconds", "e2e_seconds", "tpot_seconds"):
        result[metric] = distribution([row[metric] for row in good])
    return result


def service_comparison(runs: dict) -> dict:
    names = {
        f"service{budget}-{arm}-{repeat}"
        for budget in (8, 2)
        for arm in ("off", "on")
        for repeat in (1, 2)
    }
    require(set(runs) == names, "Service process inventory differs")
    cells = {
        f"{tokens}-c{concurrency}": concurrency
        for tokens in (1024, 8192, 24576)
        for concurrency in (1, 4)
    }
    formal = {}
    for name, run in runs.items():
        groups = [group for group in run["groups"] if group["kind"] == "performance"]
        expected = [f"perf-{cell}-r{repeat}" for cell in cells for repeat in range(3)]
        require(
            [group["name"] for group in groups] == expected,
            "Service formal cohorts differ",
        )
        lookup = {row["case_id"]: row for row in run["rows"]}
        for group in groups:
            match = re.fullmatch(r"perf-(\d+)-c([14])-r[012]", group["name"])
            tokens, concurrency = map(int, match.groups())
            require(
                group["concurrency"] == concurrency
                and len(group["case_ids"]) == (3 if concurrency == 1 else 8),
                "Service cohort concurrency or request count differs",
            )
            require(
                all(
                    key in lookup and lookup[key]["prompt_tokens"] == tokens
                    for key in group["case_ids"]
                ),
                "Service formal request missing or length changed",
            )
        formal[name] = groups

    def summarize(names, cell=None):
        batches = []
        for name in names:
            groups = formal[name]
            if cell is not None:
                groups = [
                    group
                    for group in groups
                    if group["name"].startswith(f"perf-{cell}-r")
                ]
            batches.append((groups, runs[name]["rows"]))
        return summarize_groups(batches)

    result = {}
    for budget in (8, 2):
        arms = {
            arm: [f"service{budget}-{arm}-{repeat}" for repeat in (1, 2)]
            for arm in ("off", "on")
        }
        report = {
            "all_formal": {arm: summarize(names) for arm, names in arms.items()},
            "cells": {},
            "processes": {},
        }
        for names in arms.values():
            for name in names:
                report["processes"][name] = summarize([name])
                report["processes"][name]["cohort_details"] = [
                    {
                        "name": group["name"],
                        **summarize_groups([([group], runs[name]["rows"])]),
                    }
                    for group in formal[name]
                ]
        for cell in cells:
            value = {arm: summarize(names, cell) for arm, names in arms.items()}
            value["throughput_ratio_on_over_off"] = (
                value["on"]["output_tokens_per_second"]
                / value["off"]["output_tokens_per_second"]
            )
            value["pairs"] = []
            for repeat in (1, 2):
                names = {
                    arm: f"service{budget}-{arm}-{repeat}" for arm in ("off", "on")
                }
                pair = {arm: summarize([name], cell) for arm, name in names.items()}
                pair["processes"] = names
                pair["throughput_ratio_on_over_off"] = (
                    pair["on"]["output_tokens_per_second"]
                    / pair["off"]["output_tokens_per_second"]
                )
                value["pairs"].append(pair)
            report["cells"][cell] = value
        result[f"{budget}gib"] = report
    return result


def capacity_comparison(runs: dict) -> dict:
    expected = {
        f"{stem}-{arm}" for stem in ("quality8", "capacity2") for arm in ("off", "on")
    }
    require(set(runs) == expected, "Capacity process inventory differs")
    result = {}
    for budget, stem in ((8, "quality8"), (2, "capacity2")):
        result[f"{budget}gib"] = {f"c{c}": {} for c in (1, 2, 4)}
        for arm in ("off", "on"):
            run = runs[f"{stem}-{arm}"]
            groups = [group for group in run["groups"] if group["kind"] == "capacity"]
            require(
                [group["concurrency"] for group in groups] == [1, 2, 4],
                "Capacity operating points differ",
            )
            lookup = {row["case_id"]: row for row in run["rows"]}
            for group in groups:
                require(
                    len(group["case_ids"]) == group["concurrency"],
                    "Capacity request count differs",
                )
                require(
                    all(
                        key in lookup
                        and lookup[key]["prompt_tokens"] == 32704
                        and lookup[key]["max_tokens"] == 64
                        for key in group["case_ids"]
                    ),
                    "Capacity prompt or output budget differs",
                )
                result[f"{budget}gib"][f"c{group['concurrency']}"][arm] = (
                    summarize_groups([([group], run["rows"])])
                )
    return result


def quality_comparison(off: list[dict], on: list[dict]) -> dict:
    tasks = ("narrativeqa", "qasper", "2wikimqa")
    expected = {
        f"{task}-{index:03d}": (task, "development" if index < 50 else "holdout")
        for task in tasks
        for index in range(100)
    }
    by_arm = {}
    for arm, rows in (("off", off), ("on", on)):
        lookup = {row["case_id"]: row for row in rows if row["kind"] == "quality"}
        if len(lookup) != sum(row["kind"] == "quality" for row in rows) or set(
            lookup
        ) != set(expected):
            raise ValueError("Quality membership differs from the 300 fixed rows")
        for key, (task, subset) in expected.items():
            row = lookup[key]
            score = row["qa_f1"]
            if (
                (row["task"], row["subset"]) != (task, subset)
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError("Quality membership or score is invalid")
        by_arm[arm] = lookup
    subsets = {}
    for subset in ("development", "holdout"):
        scores = {
            arm: {
                task: 100
                * statistics.mean(
                    row["qa_f1"]
                    for row in rows.values()
                    if row["task"] == task and row["subset"] == subset
                )
                for task in tasks
            }
            for arm, rows in by_arm.items()
        }
        losses = {task: scores["off"][task] - scores["on"][task] for task in tasks}
        mean = statistics.mean(losses.values())
        subsets[subset] = {
            "scores": scores,
            "baseline_minus_pyramidkv_f1_points": losses,
            "mean_drop_points": mean,
            "samples_per_task": dict.fromkeys(tasks, 50),
            "gate_passed": mean <= 3 and max(losses.values()) <= 5,
        }
    return {
        "subsets": subsets,
        "gate_passed": all(row["gate_passed"] for row in subsets.values()),
    }


def counter_deltas(before: str, after: str) -> dict:
    boundaries = []
    for text in (before, after):
        snapshot = {}
        for name in COUNTERS:
            entries = re.findall(
                rf"^(vllm:{name}_total(?:\{{[^\n]*\}})?)\s+(\S+)$", text, re.MULTILINE
            )
            if not entries or len(entries) != len({key for key, _ in entries}):
                raise ValueError(f"Counter missing or duplicated: {name}")
            values = [float(value) for _, value in entries]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Counter invalid: {name}")
            snapshot[name] = sum(values)
        if snapshot[COUNTERS[2]] > snapshot[COUNTERS[1]]:
            raise ValueError("Counter accepted tokens exceed drafts")
        boundaries.append(snapshot)
    delta = {name: boundaries[1][name] - boundaries[0][name] for name in COUNTERS}
    if (
        any(value < 0 for value in delta.values())
        or delta[COUNTERS[2]] > delta[COUNTERS[1]]
    ):
        raise ValueError("Counter reset or impossible acceptance delta")
    return delta


def hbm_snapshot(text: str, devices: list[int]) -> dict:
    observed, pending = {}, None
    for line in text.splitlines():
        match = re.search(r"\|\s*(\d+)\s+910B2", line)
        if match:
            pending = int(match[1])
        elif pending is not None:
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
            if pairs:
                used, total = map(int, pairs[-1])
                if pending in observed or total != 65536 or not 0 <= used <= total:
                    raise ValueError("Telemetry invalid HBM sample")
                observed[pending] = used
                pending = None
    if set(observed) != set(devices):
        raise ValueError("Telemetry physical device identity differs")
    return observed


def resource_summary(samples: list[dict], devices: list[int]) -> dict:
    counter_keys = ["vllm:" + name + "_total" for name in COUNTERS]
    peaks = {
        "vllm:kv_cache_usage_perc": "peak_kv_usage_fraction",
        "vllm:num_requests_running": "peak_running_requests",
        "vllm:num_requests_waiting": "peak_waiting_requests",
    }
    hbm = {str(device): [] for device in devices}
    phases = {}
    previous_time, previous_counters = -math.inf, dict.fromkeys(counter_keys, 0)
    for sample in samples:
        if "error" in sample:
            raise ValueError("Telemetry contains a sampling error")
        at = sample["time_unix"]
        if not math.isfinite(at) or at < previous_time:
            raise ValueError("Telemetry timestamps are invalid or reversed")
        previous_time = at
        metrics = sample.get("metrics", {})
        for key in [*counter_keys, *peaks]:
            value = metrics.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Telemetry missing or invalid metric: {key}")
        if metrics["vllm:kv_cache_usage_perc"] > 1:
            raise ValueError("Telemetry invalid KV utilization fraction")
        if any(metrics[key] < previous_counters[key] for key in counter_keys):
            raise ValueError("Telemetry counter reset")
        previous_counters = {key: metrics[key] for key in counter_keys}
        if metrics[counter_keys[2]] > metrics[counter_keys[1]]:
            raise ValueError("Telemetry accepted tokens exceed drafts")
        phase = phases.setdefault(
            sample["phase"],
            {
                "samples": 0,
                "first_time_unix": at,
                "last_time_unix": at,
                "first_counters": previous_counters.copy(),
                "last_counters": {},
                **dict.fromkeys(peaks.values(), 0),
                "hbm_peak_mib": {},
            },
        )
        phase["samples"] += 1
        phase["last_time_unix"] = at
        phase["last_counters"] = previous_counters.copy()
        for metric, field in peaks.items():
            phase[field] = max(phase[field], metrics[metric])
        if "npu_smi" in sample:
            for device, used in hbm_snapshot(sample["npu_smi"], devices).items():
                key = str(device)
                hbm[key].append(used)
                phase["hbm_peak_mib"][key] = max(
                    phase["hbm_peak_mib"].get(key, 0), used
                )
    if not samples or any(not values for values in hbm.values()):
        raise ValueError("Telemetry lacks samples or physical HBM observations")
    return {
        "samples": len(samples),
        "sampling_errors": 0,
        "phases": phases,
        "peak_kv_usage_fraction": max(
            phase["peak_kv_usage_fraction"] for phase in phases.values()
        ),
        "device_hbm_mib": {
            key: {"minimum": min(values), "peak": max(values)}
            for key, values in hbm.items()
        },
    }


def analyze(raw: dict[str, bytes], verification: dict) -> dict:
    """Deterministically recompute all reported effects from verified originals."""
    contract = read_json(raw["contract.json"])
    runs = {}
    for lifecycle in contract["lifecycles"]:
        name = lifecycle["name"]
        prefix = f"runs/{name}/evaluation/"
        runs[name] = {
            "groups": read_json(raw[prefix + "groups.json"]),
            "rows": read_json(raw[prefix + "results.json"]),
        }
    quality = quality_comparison(
        runs["quality8-off"]["rows"], runs["quality8-on"]["rows"]
    )
    service = service_comparison(
        {name: run for name, run in runs.items() if name.startswith("service")}
    )
    capacity = capacity_comparison(
        {
            name: run
            for name, run in runs.items()
            if name.startswith(("quality", "capacity"))
        }
    )
    probes = {}
    smoke = {}
    for name, run in runs.items():
        if not name.startswith("service"):
            continue
        hashes = [
            digest(row["output"].encode())
            for row in run["rows"]
            if row["kind"] == "repeatability"
        ]
        probes[name] = {
            "requests": len(hashes),
            "unique_output_hashes": len(set(hashes)),
            "output_sha256_in_order": hashes,
        }
        tasks = {}
        for row in run["rows"]:
            if row["kind"] == "quality":
                tasks.setdefault(row["task"], []).append(row["qa_f1"])
        smoke[name] = {
            task: {
                "requests": len(scores),
                "mean_f1_points": 100 * statistics.mean(scores),
            }
            for task, scores in tasks.items()
        }
    processes = verification["lifecycles"]
    return {
        "schema_version": 1,
        "identity": {
            key: contract[key]
            for key in (
                "release_source",
                "runner_source",
                "wheel_sha256",
                "physical_devices",
            )
        },
        "requests": sum(row["requests"] for row in processes.values()),
        "compression_transactions": sum(
            row["compression_transactions"] for row in processes.values()
        ),
        "quality": quality,
        "service": service,
        "capacity": capacity,
        "repeatability_probes_excluded_from_performance": probes,
        "service_quality_smoke_not_the_full_quality_gate": smoke,
        "process_resources_and_runtime": processes,
        "limits": {
            "capacity": "C1/C2/C4 tested operating points; not maximum resident capacity",
            "timings": "Original client observations; SSE bodies do not independently reproduce receive timestamps",
            "private_inputs": "Saved-score aggregates are public; official F1 and QA payload reconstruction require pinned private inputs",
            "runtime_snapshot_gaps": verification["runtime_snapshot_gaps"],
            "model_weights_rehashed_after_series": verification[
                "model_weights_rehashed_after_series"
            ],
            "service_replicates_per_arm_per_budget": 2,
            "pressure_cache_attribution": "Unique per-request salts, phase preemptions and pinned host source; no individual request preemption event trace",
            "hbm": "Sampled physical HBM; KV block release is not HBM savings",
        },
    }
