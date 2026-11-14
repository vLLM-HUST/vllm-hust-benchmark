"""Project audited one-click results into public metrics; no task text or private paths."""

from pathlib import Path
import json
import sys
import hashlib
import math
import argparse
from project_goal_assessment import (
    assess,
    merge_requirements,
    REFERENCE_SHA256,
    unmeasured_current_compute_target,
)
from stability_assessment import assess as assess_stability, milestone_target
from publication_campaign import validate, roots, provenance, report_date, device_cards

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--engine-tools", type=Path, required=True)
parser.add_argument("--aggregation-commit", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert len(args.aggregation_commit) == 40 and all(
    c in "0123456789abcdef" for c in args.aggregation_commit
)
R = args.output.parent
R.mkdir(parents=True, exist_ok=True)
run = args.run.resolve()
report = run.with_name(run.name + "-report")
data = json.loads((report / "summary.json").read_text())
reused_count, new_count = validate(data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, digest in json.loads((report / "checksums.json").read_text()).items():
    assert sha(report / name) == digest
for name, expected in data["plan"]["sources"].items():
    if sha(args.engine_tools / name) != expected:
        raise ValueError("Export tools differ from captured execution source: " + name)
source_identity = args.engine_tools / "SOURCE-COMMIT.json"
if (
    source_identity.exists()
    and json.loads(source_identity.read_bytes())["commit"] != args.aggregation_commit
):
    raise ValueError("Export Engine commit differs from its verified source identity")
sys.path.insert(0, str(args.engine_tools / "scripts"))
from telemetry_audit import utilization  # noqa: E402 - supplied fixed Engine tools

from third_party_report import verify_campaign_bindings  # noqa: E402 - fixed Engine tools

# Recheck the actual files consumed below, not merely the earlier report JSON.
verify_campaign_bindings(run)
reuse_binding = (
    json.loads((run / "reused-stages.json").read_bytes())
    if (run / "reused-stages.json").exists()
    else {"origins": []}
)
for origin in reuse_binding["origins"]:
    source_root = Path(origin["source_root"])
    for filename, field in (
        ("campaign.json", "campaign_sha256"),
        ("campaign-checksums.json", "campaign_seal_sha256"),
        ("frozen-execution-plan.json", "plan_sha256"),
    ):
        if sha(source_root / filename) != origin[field]:
            raise ValueError("Reused source changed before public export: " + filename)
    verify_campaign_bindings(source_root)

stage_roots = roots(run, data)


def quantile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo = math.floor(i)
    hi = math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def rows(root, native=False):
    result = []
    for p in (
        (root / "service").glob("*.json")
        if native
        else (root / "service/evaluation/requests").glob("*.json")
    ):
        row = json.loads(p.read_bytes())
        if not isinstance(row, dict) or not all(
            k in row for k in ["start", "end", "usage"]
        ):
            continue
        if native and not row.get("success"):
            continue
        if not native and row.get("status") not in ["ok", "invalid_judge"]:
            continue
        result.append(row)
    return result


price = data["existing_project_evidence"]["price_basis"]
rate = price["instance_monthly_price"] / (price["month_days_for_analysis"] * 86400)


def extras(root, native=False):
    rs = rows(root, native)
    assert rs
    ts = [1000 * (r["first_token"]["monotonic"] - r["start"]["monotonic"]) for r in rs]
    ps = [
        1000
        * (r["end"]["monotonic"] - r["first_token"]["monotonic"])
        / (r["usage"]["completion_tokens"] - 1)
        for r in rs
        if r["usage"]["completion_tokens"] > 1
    ]
    ins = [r["usage"]["prompt_tokens"] for r in rs]
    outs = [r["usage"]["completion_tokens"] for r in rs]
    start = min(r["start"]["monotonic"] for r in rs)
    end = max(r["end"]["monotonic"] for r in rs)
    if native:
        phase = json.loads((root / "service/joint.json").read_text())
        start = phase["start"]["monotonic"]
        end = phase["end"]["monotonic"]
    samples = [
        json.loads(s)
        for s in (root / "device-samples.jsonl").read_text().splitlines()
        if s
    ]
    return {
        "ttft_p50_ms": quantile(ts, 0.5),
        "ttft_p95_ms": quantile(ts, 0.95),
        "ttft_p99_ms": quantile(ts, 0.99),
        "tpot_p50_ms": quantile(ps, 0.5),
        "tpot_p95_ms": quantile(ps, 0.95),
        "tpot_p99_ms": quantile(ps, 0.99),
        "input_tokens_sum": sum(ins),
        "input_tokens_min": min(ins),
        "input_tokens_max": max(ins),
        "input_at_least16384_cases": sum(n >= 16384 for n in ins),
        "input_at_least32768_cases": sum(n >= 32768 for n in ins),
        "input_plus_output_at_least32768_cases": sum(
            a + b >= 32768 for a, b in zip(ins, outs)
        ),
        "input_plus_output_at_least16384_cases": sum(
            a + b >= 16384 for a, b in zip(ins, outs)
        ),
        "sampled_utilization": utilization(
            samples, start, end, expected=device_cards(root)
        ),
        "estimated_cny_per_million_output_tokens": 1e6
        * rate
        * (end - start)
        / sum(outs),
    }


fields = [
    "planned",
    "recorded",
    "successful",
    "transport_completed",
    "failed",
    "unreturned",
    "unresolved_grades",
    "length_stops",
    "successful_output_tokens",
    "duration_seconds",
    "output_tokens_per_second",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "actual_context_16k_cases",
]
datasets = []
for name, title, count, kind in [
    ("mmlu", "MMLU-Pro fixed200", 200, "quality_and_inference"),
    ("hle", "HLE stratified200", 200, "quality_and_inference"),
    ("frontier", "FrontierScience160", 160, "quality_and_inference"),
    ("swe", "SWE-bench Pro trace32", 32, "inference_only"),
    ("terminal", "Terminal-Bench2.1 trace32", 32, "inference_only"),
]:
    arms = {}
    judges = {}
    quality = {}
    for role in ["B0", "B1"]:
        if name == "mmlu":
            src = data["current_mmlu_pair"]["arms"][role]
            group = src["groups"]["joint200"]
            a = {
                "planned": 200,
                "recorded": group["recorded"],
                "successful": group["successful"],
                "failed": group["failed"],
                "unreturned": 0,
                "length_stops": src["quality"]["truncated_recorded"],
                "successful_output_tokens": group["successful_output_tokens"],
                "duration_seconds": group["cohort_wall_seconds"],
                "output_tokens_per_second": group["output_tokens_per_second"],
                "mean_ttft_ms": group["mean_ttft_ms"],
                "mean_tpot_ms": group["mean_tpot_ms"],
            }
            quality[role] = {
                k: v for k, v in src["quality"].items() if k not in ["scorer_path"]
            }
            root = stage_roots["mmlu-pair"] / role
        elif name in ["hle", "frontier"]:
            src = data["current_dataset_performance"][name + "-solver-" + role]
            a = {k: src.get(k) for k in fields}
            root = stage_roots[name + "-solver-" + role]
            quality[role] = data["current_dataset_quality"][name + "-" + role]
            source = stage_roots[name + "-judge-" + role]
            raw = json.loads(
                (source.with_name(source.name + "-report") / "summary.json").read_text()
            )
            judges[role] = {k: raw.get(k) for k in fields}
            judges[role].update(extras(source))
        else:
            src = data["current_trace_performance"][name]["arms"][role]
            a = {k: src.get(k) for k in fields}
            root = stage_roots[name + "-trace-" + role]
        expected_requests = {
            "mmlu": 200,
            "hle": 200,
            "frontier": 160,
            "swe": 64,
            "terminal": 48,
        }[name]
        if (
            a["planned"] != expected_requests
            or a["recorded"] != expected_requests
            or a["failed"] != 0
            or a["unreturned"] != 0
        ):
            raise ValueError(
                "Standard solver/trace request denominator or execution differs: "
                + name
                + "/"
                + role
            )
        a.update(extras(root, name == "mmlu"))
        a["estimated_cny_per_million_output_tokens"] = (
            1e6 * rate / a["output_tokens_per_second"]
        )
        a["source_stage_seal_sha256"] = sha(root / ("checksums.json"))
        a["runtime_identity_sha256"] = sha(root / "service/runtime-identity.json")
        arms[role] = a
    gain = {
        k: 100 * (arms["B1"][k] / arms["B0"][k] - 1)
        for k in ["output_tokens_per_second", "mean_ttft_ms", "mean_tpot_ms"]
    }
    gain["unit_cost_change_percent"] = 100 * (
        arms["B1"]["estimated_cny_per_million_output_tokens"]
        / arms["B0"]["estimated_cny_per_million_output_tokens"]
        - 1
    )
    gain["ai_core_utilization_gain_pp"] = (
        arms["B1"]["sampled_utilization"]["mean_percent"]
        - arms["B0"]["sampled_utilization"]["mean_percent"]
    )
    datasets.append(
        {
            "id": name,
            "name": title,
            "tasks_per_arm": count,
            "mode": kind,
            "arms": arms,
            "changes_percent": gain,
            "quality": quality or None,
            "judge_phases": judges or None,
        }
    )
continuous = {}
for role in ["B0", "B1"]:
    root = stage_roots["continuous-" + role]
    old = json.loads(
        (root.with_name(root.name + "-report") / "summary.json").read_text()
    )
    continuous[role] = {
        "source_seal_sha256": sha(root / "checksums.json"),
        "groups": old["groups"],
        "all_requests_recorded": old["recorded_requests"],
        "failed_requests": old["failed_requests"],
        "formal_seconds_per_context": 300,
        "warmup_seconds_per_context": 60,
        "concurrency": 8,
        "input_context_tokens": [4096, 16384],
    }
reuse = reuse_binding
origins = [{k: v for k, v in o.items() if k != "source_root"} for o in reuse["origins"]]
stages = provenance(data)
targets = data["formal_targets"]
targets["3.4.1"] = unmeasured_current_compute_target(targets["3.4.1"])
stability = assess_stability(stage_roots, datasets)
targets["3.4.3"] = milestone_target(stability)
result = {
    "schema": "sealed-five-dataset-comparison/v1",
    "run_id": run.name,
    "standard_id": "five-datasets-20261113-v1",
    "scope": "all_five_frozen_profiles_not_all_upstream_items",
    "report_date": report_date(data),
    "aggregation_engine_commit": args.aggregation_commit,
    "evidence_mode": data["receipt"].get("evidence_mode", "new_native_execution"),
    "single_new_native_run": reused_count == 0,
    "execution_complete": True,
    "formal_acceptance_complete": False,
    "reused_stages": reused_count,
    "new_inference_stages": new_count,
    "hardware": {
        "model": "Qwen3.5-35B-A3B",
        "dtype": "BF16",
        "accelerator": "Ascend910B2",
        "cards": 2,
        "parallelism": "TP2",
    },
    "runtime": {
        "B0": {
            **json.loads(
                (stage_roots["mmlu-pair"] / "frozen-pair-plan.json").read_text()
            )["baseline"]["versions"],
            "source_map_sha256": data["plan"]["baseline_source_map_sha256"],
        },
        "B1": data["plan"]["optimized_dependencies"]["metadata"],
        "actual_worker": "corrected_serving_runtime.CorrectedWorker",
        "engine_release": data["plan"].get(
            "engine_release", "full-answer-b1-20261030-r4"
        ),
        "serving_warmup": data["plan"].get("serving_warmup"),
        "task_serving_warmup": data["plan"].get("task_serving_warmup"),
        "hle_configuration_note": "HLE uses the separately recorded multimodal configuration; other profiles remain text/trace. Original per-stage source/version identities are retained, not rewritten to the aggregation commit.",
    },
    "datasets": datasets,
    "continuous": continuous,
    "stability_assessment": stability,
    "formal_targets": targets,
    "price_basis": price,
    "source_provenance": {
        "campaign_seal_sha256": sha(run / "campaign-checksums.json"),
        "report_summary_sha256": sha(report / "summary.json"),
        "report_seal_sha256": sha(report / "checksums.json"),
        "reused_origins": origins,
        "stages": stages,
    },
    "metric_notes": {
        "throughput": "Actual successful/completed output tokens divided by solver batch wall time. Judge traffic is separate.",
        "tpot": "Per-request (completion-first content token)/(output_tokens-1), equal-weight mean; output_tokens>1 only.",
        "percentiles": "Linear interpolation at (n-1)*q; finite samples, not continuous stability.",
        "quality": "Original graders and planned denominators. Unresolved are nonsuccess, not confirmed incorrect; no statistical equivalence claim.",
        "cost": "Same fixed historical public analogous-machine rate; sample estimate, not actual invoice.",
        "reuse": "Every reused stage re-audited by the one-click runner; original source dates/controllers/failed parent statuses retained. No duplicated inference.",
    },
}
reference_path = (
    Path(__file__).resolve().parents[1]
    / "reports/m01-qualified-pair-20261103/summary.json"
)
if sha(reference_path) != REFERENCE_SHA256:
    raise ValueError("Project final-target source changed")
result["milestone_assessment"] = {
    "profile": "milestone-v1.5",
    "targets": targets,
    "scope": "V1.5里程碑四项要求；本表不构成项目最终目标验收。",
}
result["project_final_assessment"] = assess(
    json.loads(reference_path.read_bytes()), datasets
)
quality_denominators = []
for dataset in datasets:
    if dataset["mode"] != "quality_and_inference":
        continue
    for role in ("B0", "B1"):
        quality = dataset["quality"][role]
        n = quality.get("planned_denominator")
        if n is None:
            n = sum(t["total"] for t in quality["tracks"].values())
        quality_denominators.append(n == dataset["tasks_per_arm"])
result["merge_requirements"] = merge_requirements(
    dict(
        execution_complete=True,
        raw_sources_verified=True,
        quality_denominators_accounted=bool(quality_denominators)
        and all(quality_denominators),
    )
)
result["milestone_numeric_requirements_met"] = all(
    t["accepted"] for t in targets.values()
)
previous_projection = R / "summary-public-v1.json"
if previous_projection.exists() and args.output != previous_projection:
    result["revision"] = {
        "version": 2,
        "supersedes": previous_projection.name,
        "supersedes_sha256": sha(previous_projection),
        "reason": "Separate answer quality from V1.5 stability; count total context and expose continuous C8 tails. Original measurements and v1 remain unchanged.",
    }
# Only structured result metrics and public provenance are published, never prompts/answers or private endpoints.
raw = (
    json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
).encode()
assert not any(
    x in raw for x in [b"/root/", b"/tmp/", b"http://127.", b"/data/shared_upload"]
)
args.output.write_bytes(raw)
(R / "stability-audit-v1.json").write_text(
    json.dumps(stability, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
)
print("Exported", len(raw), "bytes", sha(args.output))
print([(d["id"], d["changes_percent"]) for d in datasets])
