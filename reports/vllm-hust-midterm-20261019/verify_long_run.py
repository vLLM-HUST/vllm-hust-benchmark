"""Independently verify completed synthetic B0 evidence without loading models."""

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


def verify(root):
    receipt = json.loads((root / "native-controller-r2/receipt.json").read_text())
    assert receipt["status"] == "passed" and receipt["exit_code"] == 0
    assert receipt["owned_processes_gone"]
    for name, digest in receipt["input_hashes"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    rows = [
        json.loads(line)
        for line in (root / "native/requests.jsonl").read_text().splitlines()
    ]
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["success"] and r["actual_input_tokens"] == 16384
        assert sum(e["tokens"] for e in r["events"]) == r["output_tokens"]
        assert len(r["output_token_ids"]) == r["output_tokens"]
        times = [r["start"]] + [e["time"] for e in r["events"]] + [r["end"]]
        assert times == sorted(times)
    plan = json.loads((root / "plan.json").read_text())
    result = {
        "project": "vLLM-HUST",
        "arm": plan["arm"],
        "scope": "synthetic engine-stream timing; not HTTP or dataset quality",
        "candidate_measured": plan.get("candidate") is not None,
        "phases": {},
    }
    for phase, seconds in [("qualification", 60), ("formal", 300)]:
        summary = json.loads((root / f"native/{phase}-summary.json").read_text())
        a, b = summary["start"], summary["measurement_end"]
        assert math.isclose(b - a, seconds)
        selected = [r for r in rows if r["phase"] == phase]
        assert selected and all(a <= r["start"] < b for r in selected)
        assert all(r["end"] <= summary["drain_end"] for r in selected)
        total = sum(
            e["tokens"] for r in selected for e in r["events"] if a <= e["time"] < b
        )
        for w in summary["windows"]:
            count = sum(
                e["tokens"]
                for r in selected
                for e in r["events"]
                if w["start"] <= e["time"] < w["end"]
            )
            assert count == w["output_tokens"]
            assert math.isclose(count / (w["end"] - w["start"]), w["output_tps"])
        assert total == sum(w["output_tokens"] for w in summary["windows"])
        ttft = [r["events"][0]["time"] - r["start"] for r in selected if r["events"]]
        tpot = [
            (r["events"][-1]["time"] - r["events"][0]["time"])
            / (r["output_tokens"] - 1)
            for r in selected
            if r["output_tokens"] > 1
        ]

        def stats(values):
            return (
                {
                    "n": len(values),
                    "mean_s": statistics.mean(values),
                    "p95_s": sorted(values)[math.ceil(0.95 * len(values)) - 1],
                    "p99_s": sorted(values)[math.ceil(0.99 * len(values)) - 1],
                }
                if values
                else None
            )

        result["phases"][phase] = {
            "seconds": seconds,
            "requests": len(selected),
            "failed": 0,
            "completed_in_window": sum(r["end"] < b for r in selected),
            "completed_in_drain": sum(r["end"] >= b for r in selected),
            "window_tokens": total,
            "output_tps": total / seconds,
            "drain_tokens": sum(r["output_tokens"] for r in selected) - total,
            "drain_seconds": summary["drain_end"] - b,
            "ttft": stats(ttft),
            "tpot": stats(tpot),
            "active_window_card_seconds_per_token": 2 * seconds / total,
            "windows": summary["windows"],
        }
    formal = json.loads((root / "native/formal-summary.json").read_text())
    samples = [
        json.loads(line)
        for line in (root / "native-controller-r2/device-samples.jsonl")
        .read_text()
        .splitlines()
    ]
    device = {
        card: {"weighted_sum": 0.0, "covered_seconds": 0.0, "samples": 0}
        for card in receipt["devices"]
    }
    for sample, following in zip(samples, samples[1:]):
        if sample["returncode"] != 0:
            continue
        a = max(formal["start"], sample["end_monotonic"])
        b = min(
            formal["measurement_end"],
            following["end_monotonic"],
            sample["end_monotonic"] + 5,
        )
        if b <= a:
            continue
        lines = sample["raw"].splitlines()
        for i, line in enumerate(lines[:-1]):
            match = re.match(r"\|\s*(\d+)\s+910B2\s*\|", line)
            if match and int(match[1]) in device:
                value = float(lines[i + 1].split("|")[3].split()[0])
                assert 0 <= value <= 100
                d = device[int(match[1])]
                d["weighted_sum"] += value * (b - a)
                d["covered_seconds"] += b - a
                d["samples"] += 1
    result["device_utilization"] = {
        "method": "Last observed npu-smi AI Core value, time weighted; hold capped at 5s; not FLOP utilization",
        "cards": {
            str(card): {
                "mean_ai_core_percent": d["weighted_sum"] / d["covered_seconds"]
                if d["covered_seconds"]
                else None,
                "covered_seconds": d["covered_seconds"],
                "uncovered_seconds": 300 - d["covered_seconds"],
                "samples": d["samples"],
            }
            for card, d in device.items()
        },
    }
    return result


if __name__ == "__main__":
    data = json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False)
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(data + "\n")
    print(data)
