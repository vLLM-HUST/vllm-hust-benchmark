"""Recompute descriptive statistics from recorded requests, without model calls."""

import math
import re
import statistics
import string
from collections import Counter

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
        "wall_seconds": sum(elapsed),
        "output_tokens_per_second": tokens / sum(elapsed),
        "cohort_output_tokens_per_second": rates,
        "cached_prompt_tokens": sum(
            row["usage"]["prompt_tokens_details"]["cached_tokens"] for row in good
        ),
    }
    for metric in ("ttft_seconds", "e2e_seconds", "tpot_seconds"):
        result[metric] = distribution([row[metric] for row in good])
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
