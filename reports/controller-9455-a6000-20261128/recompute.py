"""Verify the archived A6000 pairing and recompute request-level diagnostics."""

import gzip
import json
from collections import Counter, defaultdict
from itertools import groupby
from math import isclose
from pathlib import Path
from statistics import median, quantiles

ARCHIVE = Path(__file__).resolve().parents[2] / "archive/controller-9455-a6000-20261128"
KV = "vllm:kv_cache_usage_perc"
COUNTERS = (
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
)


def read(name):
    return json.loads((ARCHIVE / f"{name}.json").read_text())


def rows(name):
    with gzip.open(ARCHIVE / f"{name}.jsonl.gz", "rt") as stream:
        return list(map(json.loads, stream))


def percentile95(values):
    return quantiles(values, n=100, method="inclusive")[94]


def distribution(values):
    return {"p50": round(median(values), 4), "p95": round(percentile95(values), 4)}


def ttft(row):
    return row["first_token_seconds"] - row["sent_seconds"]


def verify_arm(arm, summary):
    terminal = read(f"{arm}-terminal")
    requests = rows(f"{arm}-requests")
    by_key = {(row["program"], row["round_idx"]): row for row in requests}
    assert set(by_key) == {
        (program, turn) for program in range(1, 193) for turn in range(3)
    }
    assert len(requests) == terminal["rounds_succeeded"] == 576
    assert terminal["rounds_failed"] == 0 and terminal["sessions_completed"] == 192
    assert all(row["status"] == "ok" for row in requests)
    assert (
        0
        <= terminal["makespan_seconds"]
        - max(row["finished_seconds"] for row in requests)
        < 1
    )
    assert terminal["makespan_seconds"] == summary["makespan_s"]
    assert (
        sum(row["engine_prompt_tokens"] for row in requests) == summary["prompt_tokens"]
    )
    assert (
        sum(row["engine_completion_tokens"] for row in requests)
        == terminal["output_tokens_total"]
        == summary["output_tokens"]
    )
    assert isclose(
        percentile95([ttft(row) for row in requests if row["round_idx"] == 0]),
        summary["first_turn_ttft_s_p95"],
    )
    assert isclose(
        percentile95(
            [row["finished_seconds"] - row["sent_seconds"] for row in requests]
        ),
        summary["request_latency_s_p95"],
    )

    engines = defaultdict(list)
    for row in rows(f"{arm}-engine"):
        engines[row["worker"]].append(row)
    assert engines.keys() == summary["by_host"].keys()
    pressure = {}
    for worker, samples in engines.items():
        expected = summary["by_host"][worker]
        assert max(row[KV] for row in samples) == expected["kv_peak"]
        assert len(samples) == expected["kv_samples"]
        assert (
            max(row["vllm:num_requests_waiting"] for row in samples)
            == expected["waiting_max"]
        )
        pressure[worker] = max(
            (
                sum(1 for _ in run)
                for above, run in groupby(samples, lambda row: row[KV] >= 0.95)
                if above
            ),
            default=0,
        )
        for counter, key in zip(
            COUNTERS, ("preemptions", "prefix_hit_tokens", "prefix_query_tokens")
        ):
            values = [row[counter] for row in samples]
            assert all(left <= right for left, right in zip(values, values[1:]))
            assert values[-1] - values[0] == expected[key]
    return by_key, pressure


def compare(prefix):
    pair = read("reverse-summary" if prefix else "discovery-summary")
    native, pressure = verify_arm(f"{prefix}native", pair["native"])
    limited, _ = verify_arm(f"{prefix}limited", pair["limited"])
    gate = read(f"{prefix}native-pressure-gate")
    assert pressure == gate["consecutive_samples_at_least_95pct"]
    assert gate["qualifies"] == all(count >= 3 for count in pressure.values())
    assert gate["kv_peaks"] == {
        worker: metrics["kv_peak"]
        for worker, metrics in pair["native"]["by_host"].items()
    }
    assert isclose(
        pair["speedup"], pair["native"]["makespan_s"] / pair["limited"]["makespan_s"]
    )

    turns = []
    for turn in range(3):
        keys = [key for key in native if key[1] == turn]
        matches = sum(
            native[key]["prompt_token_ids_sha256"]
            == limited[key]["prompt_token_ids_sha256"]
            for key in keys
        )
        assert matches == pair["matching_prompt_token_hashes_by_round"][str(turn)]
        turns.append(
            {
                "round": turn + 1,
                "native_ttft_seconds": distribution(
                    [ttft(native[key]) for key in keys]
                ),
                "limited_ttft_seconds": distribution(
                    [ttft(limited[key]) for key in keys]
                ),
                "native_first_to_finish_seconds": distribution(
                    [
                        native[key]["finished_seconds"]
                        - native[key]["first_token_seconds"]
                        for key in keys
                    ]
                ),
                "limited_first_to_finish_seconds": distribution(
                    [
                        limited[key]["finished_seconds"]
                        - limited[key]["first_token_seconds"]
                        for key in keys
                    ]
                ),
                "programs_with_slower_ttft": sum(
                    ttft(limited[key]) > ttft(native[key]) for key in keys
                ),
                "matching_prompt_hashes": matches,
            }
        )

    batches = read(f"{prefix}limited-scheduler-batches")
    classes = ("reasoning", "new", "acting")
    pure_new = set()
    for batch in batches:
        assert sum(batch[key] for key in classes) == len(batch["events"])
        if batch["new"] and not (batch["reasoning"] or batch["acting"]):
            pure_new.update(event["program"] for event in batch["events"])
    action_counts = read("scheduler-actions")[f"{prefix}limited"]
    assert (
        sum(batch["new"] for batch in batches)
        == action_counts["first_admissions_via_scheduler"]
    )
    assert (
        sum(batch["reasoning"] for batch in batches)
        == action_counts["resumed_existing_reasoning"]
    )
    assert (
        sum(batch["acting"] for batch in batches)
        == action_counts["resumed_existing_acting"]
    )

    first_turn = [key for key in limited if key[1] == 0]
    tail = sorted(first_turn, key=lambda key: ttft(limited[key]))[-10:]
    classes_per_batch = Counter(
        sum(batch[key] > 0 for key in classes) for batch in batches
    )
    return {
        "speedup": round(pair["speedup"], 4),
        "native_pressure_consecutive_samples": pressure,
        "turns": turns,
        "programs_finishing_earlier": sum(
            limited[key]["finished_seconds"] < native[key]["finished_seconds"]
            for key in native
            if key[1] == 2
        ),
        "native_control_admission_timestamps_present": sum(
            row["control_admission_seconds"] is not None for row in native.values()
        ),
        "limited_control_admission_timestamps_present": sum(
            row["control_admission_seconds"] is not None for row in limited.values()
        ),
        "scheduler_batches": len(batches),
        "mixed_class_scheduler_batches": sum(
            count for classes, count in classes_per_batch.items() if classes > 1
        ),
        "new_class_resume_events": action_counts["first_admissions_via_scheduler"],
        "verified_pure_new_batch_programs": len(pure_new),
        "verified_pure_new_batch_ttft_seconds": distribution(
            [ttft(limited[program, 0]) for program in pure_new]
        ),
        "verified_pure_new_batch_programs_in_slowest_ten": sum(
            program in pure_new for program, _ in tail
        ),
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {"native_first": compare(""), "limited_first": compare("reverse-")},
            indent=2,
        )
    )
