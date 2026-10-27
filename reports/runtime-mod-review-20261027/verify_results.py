import ast
import json
from pathlib import Path
import statistics
import sys

root = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else Path(__file__).resolve().parent
)
s = root / "research/speculative-mmlu-pro"
e = s / "evidence"
sys.path.insert(0, str(s))
from scoring import extract_answer  # noqa: E402 - load the recovered frozen parser

manifest = json.loads((e / "screen-manifest.json").read_text())
labels = (
    [f"screen-quality-q35-{a}-r1" for a in ("native", "mtp2", "adaptive")]
    + [
        f"screen-performance-q35-{a}-r{i}"
        for a in ("native", "mtp2", "adaptive")
        for i in (1, 2, 3)
    ]
    + [
        f"screen-performance-c04b16-{a}-r1"
        for a in ("diff-native", "native-eagle", "diffspec")
    ]
)


def delta(d, name):
    return sum(
        m.get("value", 0) for m in d["metrics_after"] if m["name"] == "vllm:" + name
    ) - sum(
        m.get("value", 0) for m in d["metrics_before"] if m["name"] == "vllm:" + name
    )


report = {"runs": {}, "parity": {}, "opset": []}
raw = {}
for label in labels:
    p = e / f"{label}-result.json"
    d = json.loads(p.read_text())
    raw[label] = d
    assert json.loads((e / f"{label}-exit.json").read_text())["returncode"] == 0
    ids = manifest["screen_ids" if "quality" in label else "performance_ids"]
    assert [r["question_id"] for r in d["rows"]] == ids
    for r in d["rows"]:
        pred = extract_answer(r["text"])
        assert pred == r["prediction"]
        assert r["correct"] == (pred == r["gold"])
        assert r["invalid"] == (pred is None)
        assert len(r["output_tokens"]) <= 4000
        assert not (
            len(r["output_tokens"]) >= 16
            and r["output_tokens"].count(0) / len(r["output_tokens"]) > 0.9
        )
    assert delta(d, "num_preemptions") == 0
    tok = sum(len(r["output_tokens"]) for r in d["rows"])
    report["runs"][label] = {
        "n": len(ids),
        "correct": sum(r["correct"] for r in d["rows"]),
        "truncated": sum(r["finish_reason"] == "length" for r in d["rows"]),
        "invalid": sum(r["invalid"] for r in d["rows"]),
        "output_tokens": tok,
        "seconds": d["elapsed"],
        "output_tokens_per_second": tok / d["elapsed"],
        "preemptions": delta(d, "num_preemptions"),
        "accepted": delta(d, "spec_decode_num_accepted_tokens"),
        "drafted": delta(d, "spec_decode_num_draft_tokens"),
    }
for family in [
    ["screen-quality-q35-" + a + "-r1" for a in ("native", "mtp2", "adaptive")],
    [
        "screen-performance-c04b16-" + a + "-r1"
        for a in ("diff-native", "native-eagle", "diffspec")
    ],
]:
    for a in family[1:]:
        assert [
            (r["question_id"], r["gold"], r["input_tokens"]) for r in raw[a]["rows"]
        ] == [
            (r["question_id"], r["gold"], r["input_tokens"])
            for r in raw[family[0]]["rows"]
        ]
configs = json.loads((s / "q35-engine-configs.json").read_text())
environments = json.loads((s / "q35-environments.json").read_text())
for label, d in raw.items():
    run_config = json.loads((e / f"{label}-config.json").read_text())
    assert run_config["sampling"] == {"temperature": 0, "top_p": 1, "max_tokens": 4000}
    if "q35-" in label:
        arm = d["arm"]
        assert run_config["engine"] == configs[arm]
        command = json.loads((e / f"{label}-command.json").read_text())["environment"]
        for key, value in environments[arm].items():
            if key.startswith(
                (
                    "HUST_VSPEC_",
                    "VSPEC_",
                    "VLLM_PLUGINS",
                    "VLLM_WORKER_MULTIPROC_METHOD",
                )
            ):
                assert command[key] == value
    else:
        assert run_config["engine"]["max_num_seqs"] == 16
        assert run_config["engine"]["gpu_memory_utilization"] == 0.4
for name in ("screen280", "perf56"):
    inputs = [
        json.loads(line) for line in (s / f"{name}.jsonl").read_text().splitlines()
    ]
    ids = manifest["screen_ids" if name == "screen280" else "performance_ids"]
    assert [row["question_id"] for row in inputs] == ids
    gold = {row["question_id"]: row["answer"] for row in inputs}
    for d in raw.values():
        if d["dataset"] == name:
            assert all(row["gold"] == gold[row["question_id"]] for row in d["rows"])
for phase in ("warm-witness", "post-witness"):
    witnesses = json.loads(
        (e / f"screen-performance-c04b16-diffspec-r1-{phase}.json").read_text()
    )
    assert len(witnesses) == 4
    for w in witnesses:
        assert w["proposer"] == "AscendDiffSpecEagleProposer" and w["cache_initialized"]
        assert w["runtime_metrics"]["retrieval_count"] == 0
report["parity"]["ids_gold_input_lengths"] = (
    "equal within each family; token lengths alone do not prove prompt byte identity"
)
med = {
    a: statistics.median(
        report["runs"][f"screen-performance-q35-{a}-r{i}"]["seconds"] for i in (1, 2, 3)
    )
    for a in ("native", "mtp2", "adaptive")
}
report["q35_wall_ratio"] = {
    "vs_target_only": med["native"] / med["adaptive"],
    "vs_mtp2": med["mtp2"] / med["adaptive"],
}
for a in ("native", "mtp2", "adaptive"):
    for i in (1, 2, 3):
        label = f"screen-performance-q35-{a}-r{i}"
        cmd = json.loads((e / f"{label}-command.json").read_text())["environment"]
        if a == "adaptive":
            assert (
                cmd["HUST_VSPEC_CONFIDENCE_ACCEPT_MARGIN"] == ""
                and cmd["HUST_VSPEC_ADAPTIVE_FULL_GRAPH"] == "1"
            )
        else:
            assert cmd["VLLM_PLUGINS"] == "ascend" and not any(
                k.startswith("HUST_VSPEC_") for k in cmd
            )
for c in (2, 7):
    d = json.loads(
        (root / f"research/opset/reproduction/evidence/card-{c}.json").read_text()
    )
    for case in d["cases"]:
        base = statistics.mean(x["timings"]["aclnn"]["mean_ms"] for x in case["rounds"])
        fused = statistics.mean(
            x["timings"]["fused"]["mean_ms"] for x in case["rounds"]
        )
        assert all(
            len(x["timings"]["aclnn"]["samples_ms"]) == 50 for x in case["rounds"]
        )
        report["opset"].append(
            {
                "card": c,
                "M": case["M"],
                "aclnn_ms": base,
                "fused_ms": fused,
                "latency_increase_percent": (fused / base - 1) * 100,
            }
        )
# Recompute paired intervals using the frozen analysis implementation, without its I/O.
tree = ast.parse((s / "analyze_screen.py").read_text())
selected = [
    n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
]
ns = {}
exec(compile(ast.Module(body=selected, type_ignores=[]), "frozen_analysis", "exec"), ns)
report["paired_quality"] = {}
for base, other in [("native", "mtp2"), ("native", "adaptive"), ("mtp2", "adaptive")]:
    report["paired_quality"][other + "_vs_" + base] = ns["compare"](
        raw["screen-quality-q35-" + base + "-r1"],
        raw["screen-quality-q35-" + other + "-r1"],
    )
# Emit derived data only; never rewrite raw measurements.
print(json.dumps(report, indent=2))
