"""Compare frozen deployments, retaining differences and task-quality limits."""

import json
from pathlib import Path
import sys
from verify_long_run import verify


def compare(baseline, candidate):
    a, b = verify(baseline), verify(candidate)
    pa = json.loads((baseline / "plan.json").read_text())
    pb = json.loads((candidate / "plan.json").read_text())
    for key in [
        "input_sha256",
        "model_config_sha256",
        "prompt_tokens",
        "concurrency",
        "qualification_seconds",
        "measurement_seconds",
        "max_tokens",
        "temperature",
        "seed",
        "thinking",
        "cache_policy",
        "timing",
    ]:
        assert pa[key] == pb[key], key
    for key in [
        "dtype",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "kv_cache_memory_bytes",
    ]:
        assert pa["engine"][key] == pb["engine"][key], key
    for root in [baseline, candidate]:
        receipt = json.loads((root / "native-controller-r2/receipt.json").read_text())
        assert receipt["devices"] == [4, 5]
    x, y = a["phases"]["formal"], b["phases"]["formal"]
    changes = {
        "throughput_gain_percent": 100 * (y["output_tps"] / x["output_tps"] - 1),
        "ttft_mean_reduction_percent": 100
        * (1 - y["ttft"]["mean_s"] / x["ttft"]["mean_s"]),
        "tpot_mean_reduction_percent": 100
        * (1 - y["tpot"]["mean_s"] / x["tpot"]["mean_s"]),
        "equal_rate_window_cost_reduction_percent": 100
        * (
            1
            - y["active_window_card_seconds_per_token"]
            / x["active_window_card_seconds_per_token"]
        ),
    }
    differences = {
        k: {"B0": pa["engine"].get(k), "B1": pb["engine"].get(k)}
        for k in sorted(pa["engine"].keys() | pb["engine"].keys())
        if pa["engine"].get(k) != pb["engine"].get(k)
    }

    def tokens(root):
        values = {i: set() for i in range(8)}
        for line in (root / "native/requests.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["phase"] == "formal":
                values[int(row["id"].split("-")[1])].add(tuple(row["output_token_ids"]))
        return values

    ta, tb = tokens(baseline), tokens(candidate)
    return {
        "project": "vLLM-HUST",
        "B0_path": str(baseline),
        "B1_path": str(candidate),
        "scope": "whole deployment comparison on fixed synthetic C8; not single-MOD attribution",
        "changes": changes,
        "engine_differences": differences,
        "package_versions": {"B0": pa["packages"], "B1": pb["packages"]},
        "synthetic_exact_token_sets_by_lane": {
            str(i): {
                "B0_sequences": len(ta[i]),
                "B1_sequences": len(tb[i]),
                "equal": ta[i] == tb[i],
            }
            for i in range(8)
        },
        "initial_serving_threshold_observations": {
            k: changes[k] >= limit
            for k, limit in [
                ("throughput_gain_percent", 10),
                ("ttft_mean_reduction_percent", 5),
                ("tpot_mean_reduction_percent", 5),
                ("equal_rate_window_cost_reduction_percent", 10),
            ]
        },
        "task_quality_qualified": False,
        "effective_compute_qualified": False,
        "overall_midterm_acceptance": False,
        "limits": [
            "One sequential B0/B1 pair, not independent repeats",
            "No formal primary-dataset correctness score from this workload",
            "Resource cost covers measured window only, not full startup lifecycle",
        ],
        "B0": a,
        "B1": b,
    }


if __name__ == "__main__":
    data = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps(data["changes"], indent=2))
