"""Audit fresh matched-effective-depth comparison without importing old run analyses."""

import ast
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoring import extract_answer  # noqa: E402 - frozen runner root import


def metric(r, name):
    def value(phase):
        matches = [m["value"] for m in r[phase] if m["name"] == "vllm:" + name]
        assert len(matches) == 1, (phase, name)
        return matches[0]

    return value("metrics_after") - value("metrics_before")


def run(arm, i):
    label = f"gamma4-performance-{arm}-r{i}"
    e = ROOT / "evidence"
    assert json.loads((e / f"{label}-exit.json").read_text())["returncode"] == 0
    r = json.loads((e / f"{label}-result.json").read_text())
    rows = r["rows"]
    expected = json.loads((e / "screen-manifest.json").read_text())["performance_ids"]
    assert [x["question_id"] for x in rows] == expected
    for row in rows:
        assert extract_answer(row["text"]) == row["prediction"]
        assert (row["prediction"] == row["gold"]) == row["correct"]
    preemptions = metric(r, "num_preemptions")
    assert preemptions == 0
    drafted = metric(r, "spec_decode_num_draft_tokens")
    accepted = metric(r, "spec_decode_num_accepted_tokens")
    tokens = sum(len(x["output_tokens"]) for x in rows)
    record = {
        "label": label,
        "seconds": r["elapsed"],
        "output_tokens": tokens,
        "output_tokens_per_second": tokens / r["elapsed"],
        "correct": sum(x["correct"] for x in rows),
        "truncated": sum(x["finish_reason"] == "length" for x in rows),
        "invalid": sum(x["invalid"] for x in rows),
        "preemptions": preemptions,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance_rate": accepted / drafted,
    }
    witness = json.loads((e / f"{label}-witness.json").read_text())
    assert len(witness["before"]) == len(witness["after"]) == 2
    sets = []
    for phase in ["before", "after"]:
        for w in witness[phase]:
            keys = {
                (k["num_tokens"], k["num_reqs"])
                for k in w["graph_keys"]["FULL"]
                if k["num_tokens"] == 5 * k["num_reqs"]
            }
            assert keys == {(5 * n, n) for n in range(1, 17)}, (label, phase, keys)
            sets.append(w["resolved_capture_sizes"])
    record["resolved_capture_sizes"] = sets[0]
    if arm == "q35-adaptive":
        summaries = [
            ast.literal_eval(line.split("vSpec Online Adaptive summary: ", 1)[1])
            for line in (e / f"{label}.log").read_text().split("\n")
            if "vSpec Online Adaptive summary: " in line
        ]
        assert summaries and all(
            s["gamma_switches"] == 0 and set(s["selection_counts"]) <= {"4"}
            for s in summaries
        )
        record["gamma_summary"] = {
            k: summaries[-1][k]
            for k in ["initial_gamma", "gamma_switches", "selection_counts"]
        }
    return record, r


if __name__ == "__main__":
    allruns = {}
    raw = {}
    for arm in ["q35-mtp4", "q35-adaptive"]:
        allruns[arm] = []
        raw[arm] = []
        for i in range(1, 4):
            record, r = run(arm, i)
            allruns[arm].append(record)
            raw[arm].append(r)
    paired = []
    for i, (a, b) in enumerate(zip(allruns["q35-mtp4"], allruns["q35-adaptive"])):
        left = raw["q35-mtp4"][i]["rows"]
        right = raw["q35-adaptive"][i]["rows"]
        assert [(x["question_id"], x["gold"], x["input_tokens"]) for x in left] == [
            (x["question_id"], x["gold"], x["input_tokens"]) for x in right
        ]
        paired.append(
            {
                "repeat": i + 1,
                "native_over_candidate_wall_ratio": a["seconds"] / b["seconds"],
                "candidate_over_native_output_tps": b["output_tokens_per_second"]
                / a["output_tokens_per_second"],
                "identical_sequences": sum(
                    x["output_tokens"] == y["output_tokens"]
                    for x, y in zip(left, right)
                ),
                "candidate_wins": sum(
                    not x["correct"] and y["correct"] for x, y in zip(left, right)
                ),
                "candidate_losses": sum(
                    x["correct"] and not y["correct"] for x, y in zip(left, right)
                ),
            }
        )
    medians = {
        arm: {
            key: statistics.median(x[key] for x in runs)
            for key in ["seconds", "output_tokens_per_second"]
        }
        for arm, runs in allruns.items()
    }
    out = {
        "runs": allruns,
        "paired": paired,
        "medians": medians,
        "native_over_candidate_median_wall_ratio": medians["q35-mtp4"]["seconds"]
        / medians["q35-adaptive"]["seconds"],
        "candidate_over_native_median_output_tps": medians["q35-adaptive"][
            "output_tokens_per_second"
        ]
        / medians["q35-mtp4"]["output_tokens_per_second"],
        "scope": "fresh56x3 per arm, shared host, no per-step probes; not full MMLU or equal-output speedup",
    }
    (ROOT / "gamma4/results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
