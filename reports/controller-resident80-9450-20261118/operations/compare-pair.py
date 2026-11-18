"""Recompute the paired result from the complete replay records."""

import json
import sys
from pathlib import Path


def load_arm(root):
    summary = json.loads((root / "summary.json").read_text())
    result = summary["result"]
    rounds = [
        json.loads(line)
        for line in (root / "rounds-live.jsonl").read_text().splitlines()
    ]
    keys = {(row["session_id"], row["round_idx"]): row for row in rounds}
    assert len(keys) == len(rounds), "duplicate request identities"
    succeeded = [row for row in rounds if row["status"] == "ok"]
    stats = {
        "tasks": result["sessions"],
        "completed_tasks": result["sessions_completed"],
        "unfinished_tasks": result["sessions_unfinished"],
        "successful_requests": len(succeeded),
        "failed_requests": result["rounds_failed"],
        "elapsed_seconds": result["makespan_seconds"],
        "input_tokens": sum(row["engine_prompt_tokens"] for row in succeeded),
        "cached_input_percent": 100
        * sum(row["engine_cached_tokens"] for row in succeeded)
        / sum(row["engine_prompt_tokens"] for row in succeeded),
        "zero_hit_continuations": sum(
            row["round_idx"] > 0 and row["engine_cached_tokens"] == 0
            for row in succeeded
        ),
        "output_tokens": sum(row["engine_completion_tokens"] for row in succeeded),
        "cached_input_tokens": sum(row["engine_cached_tokens"] for row in succeeded),
        "first_round_cached_input_tokens": sum(
            row["engine_cached_tokens"] for row in succeeded if row["round_idx"] == 0
        ),
        "task_latency_seconds": result["program_end_to_end_latency_seconds"],
        "request_latency_seconds": result["request_ready_to_finish_seconds"],
        "controller_admission_seconds": result["controller_admission_seconds"],
    }
    return keys, stats


direct_keys, direct = load_arm(Path(sys.argv[1]))
throttled_keys, throttled = load_arm(Path(sys.argv[2]))
same_requests = direct_keys.keys() == throttled_keys.keys()
complete = same_requests and all(
    row["tasks"] == row["completed_tasks"] == 112
    and row["successful_requests"] == 336
    and row["failed_requests"] == 0
    for row in (direct, throttled)
)
data = {
    "completed_pair": complete,
    "direct": direct,
    "request_throttling_controller": throttled,
    "matched_prompt_token_hashes": sum(
        direct_keys[key]["prompt_token_ids_sha256"]
        == throttled_keys[key]["prompt_token_ids_sha256"]
        for key in direct_keys.keys() & throttled_keys.keys()
    ),
}
if complete and direct["elapsed_seconds"] > 0:
    seconds = direct["elapsed_seconds"] - throttled["elapsed_seconds"]
    data["seconds_saved"] = seconds
    data["time_reduction_percent"] = seconds / direct["elapsed_seconds"] * 100
print(json.dumps(data, indent=2))
