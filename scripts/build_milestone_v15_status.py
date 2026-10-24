"""CPU-only reuse audit. Never writes experiments or promotes controls to B0."""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"
DEST = REPORT / "milestone-v15"


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def build():
    manifest = json.loads((DEST / "manifest.json").read_text())
    for item in manifest["files"]:
        raw = (DEST / item["name"]).read_bytes()
        assert len(raw) == item["bytes"]
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    method = json.loads((REPORT / "m01-method-v527.json").read_text())
    method["method_version"] = "V1.5"
    method.pop("metric_method_inherited_from")
    method["outline"] = {
        "path": (DEST / "project4-v1.5.txt").relative_to(ROOT).as_posix(),
        **{k: v for k, v in manifest["files"][1].items() if k != "name"},
        "source_docx_sha256": manifest["source"]["sha256"],
        "source_docx_available_in_this_repository": False,
        "scope": "Exact Project 4 extract; source DOCX digest supplied by document owner",
    }
    method["historical_instrumented"]["classification"] = (
        "historical empirical-resource reference, not current V1.5 efficiency or acceptance"
    )
    method["additional_existing_evidence"] = [
        "reports/vllm-hust-midterm-20261019/m01-recovery/collective-graph-20261023/manifest.json",
        "reports/vllm-hust-midterm-20261019/m01-recovery/boundary-measured-20261023/manifest.json",
    ]
    boundary = json.loads(
        (ROOT / method["additional_existing_evidence"][1]).read_text()
    )
    method["measured_synchronized_b1"] = boundary["measurement"]
    method["pending_evidence"][1] = (
        "Associate the measured 120.65347687713802-second synchronized B1 envelope with valid effective F and same-workload reference; do not substitute the older ~62-second run"
    )
    timing_dir = REPORT / "m01-recovery/storage-timing-20261024"
    timing_manifest = json.loads((timing_dir / "archive.json").read_text())
    timing_raw = (timing_dir / "arithmetic.json").read_bytes()
    assert (
        hashlib.sha256(timing_raw).hexdigest()
        == timing_manifest["derived_sha256"]["arithmetic.json"]
    )
    timing = json.loads(timing_raw)
    method["status"] = "compute-measured-reference-pending"
    method["current"].update(
        effective_flops=timing["basic_operation_flops"],
        participating_chips=timing["participating_chips"],
        matched_synchronized_seconds=timing["synchronized_seconds"],
        effective_tflops_per_chip=timing["basic_operation_tflops_per_chip"],
    )
    method["current_measurement"] = {
        "run_id": timing["run_id"],
        "evidence_path": (timing_dir / "arithmetic.json").relative_to(ROOT).as_posix(),
        "arithmetic_convention": timing["scope"],
        "implementation_variant": timing["implementation_variant"],
        "strict_uninstrumented": False,
        "host_prefill_counting": True,
        "native_detailed_acceptance": True,
        "all_measurement_cost_retained": True,
        "device_journals": False,
        "model_profiler": False,
        "replaces_prior_best_b1_performance": False,
    }
    method["requirements"]["timing_refinement"] = (
        "This run uses predeclared host-only prefill aggregates and native detailed acceptance "
        "statistics, with all costs retained inside synchronized T. It is not strictly "
        "uninstrumented. Same-run F and C are reported under the disclosed basic-operation "
        "convention; no efficiency or attainment until the independent reference is complete."
    )
    method["pending_evidence"] = [
        "Complete independent whole-model reference, actual dependencies and applicability",
        "Quantify reference-model uncertainty without fitting B0/B1 time or selecting faster samples",
    ]
    c16 = module(ROOT / "reports/c16-baseline018-reuse-20261022/audit.py").audit()
    pair = module(REPORT / "render_m06_pair.py").load(REPORT / "m06-paired-20261021")
    arms = pair["arms"]
    costs = {
        arm: 4 * row["burst_s"] / row["output_tokens"] for arm, row in arms.items()
    }
    reduction = (1 - costs["b1"] / costs["b0"]) * 100
    assert abs(reduction - pair["request_cost_reduction_percent"]) < 1e-10
    return {
        "schema": "milestone-v15-reuse/v1",
        "method": method,
        "cost_method": {
            "section": "3.4.4",
            "allocated_expense": "P * k * T / (24 * D * 3600)",
            "yuan_per_million_output_tokens": "1e6 * allocated_expense / N",
            "request_population": "Same requests initiated and successfully completed inside the measurement interval; cross-interval usage separate",
            "cards": "All occupied cards; no utilization discount on fixed fees",
            "utilization": "Same interval, per-node/card AI Core samples, time-weighted means",
            "lifecycle_is_current_gate": False,
            "passed": None,
        },
        "stability_method": {
            "section": "3.4.3",
            "task_scope": [
                "MMLU-Pro",
                "HLE-Verified",
                "SWE-bench-Pro",
                "FrontierScience",
                "TerminalBench 2.1",
            ],
            "actual_context_minimum": 16384,
            "compatible_32k_reusable": True,
            "requires_continuous_intervals": True,
            "six_probe_pass_is_task_stability_pass": False,
            "passed": None,
        },
        "recomputed": {
            "c16_fixed_b0": {
                **c16,
                "evidence_path": "reports/c16-baseline018-reuse-20261022/manifest.json",
                "passed": None,
            },
            "historical_development_pair_resource_cost": {
                "source_path": "reports/vllm-hust-midterm-20261019/m06-paired-20261021/m06-prechecked-pair-result-r1.json",
                "historical_b0_field_means": "0.28.1/0.25.1rc2 development control, NOT fixed B0",
                "reserved_cards": 4,
                "card_seconds_per_output_token": costs,
                "reduction_percent": reduction,
                "current_currency_cost_passed": None,
                "scope": "Recomputed historical resource coefficient; not new V1.5 monetary acceptance",
            },
        },
        "decision_relevant_gaps": {
            "m01": method["pending_evidence"],
            "performance": [
                "C16 client identity and B1 resource occupancy",
                "Applicable fixed-B0 latency/quality/SLO comparison, not universal extra repetitions",
            ],
            "stability": [
                "Five-task completion and anomaly coverage",
                "Actual >=16K continuous intervals with comparable 4K load and window statistics",
            ],
            "cost": [
                "Representative fixed-B0 comparison and applicable quality/SLO",
                "Documented monthly all-in fee and month-day basis",
                "Same successful-request population and time-weighted same-interval utilization",
            ],
        },
        "new_measurements_started": False,
    }


if __name__ == "__main__":
    result = build()
    (DEST / "reuse-audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (DEST / "method-status.json").write_text(
        json.dumps(result["method"], ensure_ascii=False, indent=2) + "\n"
    )
