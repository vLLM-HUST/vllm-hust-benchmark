"""Recompute descriptive statistics from recorded requests, without model calls."""

import math
import re
import statistics
import string
from collections import Counter


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
