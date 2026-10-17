#!/usr/bin/env python3
"""Convert a run of the earlier agent-program benchmark (E1-E4,
experiments/control_plane/agent_program_benchmark.py) into the layout
plot_run.py reads, so its timeline can be drawn the same way.

    python experiments/workload/import_agent_program_run.py \\
        data/from_ascend91/E3-five-engines-c320/<run> <out directory> \\
        --kv-capacity-tokens 755328

What the old records have and how they map:
    steps.jsonl    one line per LLM call: program_id, slot, step, t_start, t_end
                   (wall clock), prompt/completion tokens, ok
      -> rounds.jsonl: session_id = program_id, round_idx = step,
         sent_seconds / finished_seconds relative to the run's start.
         The old benchmark did not stream, so there is no first-token time:
         first_token_seconds is set to sent_seconds, and the whole call shows
         as "reasoning, decoding" (queueing and a router holding it back are
         inside it, not separable).
    metrics.jsonl  one line per engine per 10 s: kv_cache_usage_perc, running,
                   waiting, and counter *increments* since the previous sample
      -> metrics.jsonl with running totals, as engine_metrics.py writes them.
    manifest.json  -> summary.json with `engines` (capacity from
                   --kv-capacity-tokens: the engine log's "GPU KV cache size")
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

COUNTERS = {
    "prompt_tokens_delta": "vllm:prompt_tokens_total",
    "generation_tokens_delta": "vllm:generation_tokens_total",
    "prefix_cache_queries_delta": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits_delta": "vllm:prefix_cache_hits_total",
    "num_preemptions_delta": "vllm:num_preemptions_total",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--kv-capacity-tokens", type=int, required=True)
    arguments = parser.parse_args()

    manifest = json.loads((arguments.run / "manifest.json").read_text())
    start = manifest["started_at"]
    arguments.out.mkdir(parents=True, exist_ok=True)

    rounds = []
    for line in (arguments.run / "steps.jsonl").read_text().splitlines():
        step = json.loads(line)
        if step.get("event") != "step" or "t_end" not in step:
            continue
        sent = round(step["t_start"] - start, 4)
        rounds.append({
            "session_id": step["program_id"], "round_idx": step["step"],
            "sent_seconds": sent, "first_token_seconds": sent,
            "finished_seconds": round(step["t_end"] - start, 4),
            "status": "ok" if step.get("ok") else "error",
            "slot": step.get("slot"),
        })  # fmt: skip
    (arguments.out / "rounds.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rounds)
    )

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    lines = []
    for line in (arguments.run / "metrics.jsonl").read_text().splitlines():
        sample = json.loads(line)
        engine = sample["backend"].rstrip("/") + "/v1"
        for delta, total in COUNTERS.items():
            totals[engine][total] += sample.get(delta, 0.0)
        lines.append(
            {
                "t": round(sample["t"] - start, 3),
                "engine": engine,
                "vllm:kv_cache_usage_perc": sample["kv_cache_usage_perc"],
                "vllm:num_requests_running": sample["num_requests_running"],
                "vllm:num_requests_waiting": sample["num_requests_waiting"],
            }
            | dict(totals[engine])
        )
    (arguments.out / "metrics.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines)
    )
    engines = sorted({x["engine"] for x in lines})
    (arguments.out / "summary.json").write_text(json.dumps({
        "imported_from": str(arguments.run),
        "parameters": manifest.get("params"),
        "old_summary": manifest.get("summary"),
        "engines": [{"endpoint": e, "kv_capacity_tokens_device":
                     arguments.kv_capacity_tokens, "kv_capacity_tokens_host": 0}
                    for e in engines],
    }, indent=1) + "\n")  # fmt: skip
    print(f"{len(rounds)} rounds, {len(lines)} metric samples, engines {engines}")


if __name__ == "__main__":
    main()
