"""Project audited own-run compute without manufacturing reference efficiency."""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = (
    ROOT
    / "reports/vllm-hust-midterm-20261019/m01-recovery/official018-paired-compute-20261024"
)
FILES = (
    "paired-own-metrics-closed-r2.json",
    "profiling-final-export-audit-r2.json",
    "profiling-group-sum-audit-r2.json",
    "formal-necessary-full-export-audit-r2.json",
    "native-align-independent-source-review-r2.json",
    "B1-native-align-applicability-source-review-r1.json",
)


def build(directory=DEST):
    pair = json.loads((directory / FILES[0]).read_text())
    audit = json.loads((directory / FILES[1]).read_text())
    assert audit["status"] == "final_export_and_own_metrics_verified"
    assert pair["rows"] == audit["rows"]
    for row in pair["rows"].values():
        assert all(
            type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] > 0
            for k in ("F", "chips", "T_seconds", "C_TFLOPS_per_chip")
        )
        assert row["chips"] == 2
        assert math.isclose(
            row["C_TFLOPS_per_chip"],
            row["F"] / (row["chips"] * row["T_seconds"] * 1e12),
            rel_tol=1e-14,
        )
    rows = pair["rows"]
    gain = (rows["B1"]["C_TFLOPS_per_chip"] / rows["B0"]["C_TFLOPS_per_chip"] - 1) * 100
    assert math.isclose(gain, pair["C_gain_percent"], rel_tol=1e-14)
    assert (
        pair["eta_percent"]
        is pair["reference_seconds"]
        is pair["reference_tflops_per_chip"]
        is None
    )
    return {
        "schema": "m01-own-compute-pair/v1",
        "status": "own_F_T_C_closed_reference_method_pending",
        "rows": rows,
        "C_gain_percent": gain,
        "reference_seconds": None,
        "reference_tflops_per_chip": None,
        "eta_percent": None,
        "attainment": None,
        "reference_method_status": "whole_chain_execution_protocol_not_frozen",
        "baseline_is_reference_ceiling": False,
        "identities": {
            "B0": "vLLM 0.18.0 / vLLM-Ascend 0.18.0 with disclosed text-only compatibility patches",
            "B1": "vLLM 0.28.1.post1.dev143+gf18cf803c / vLLM-Ascend 0.25.1rc2 development stack; vLLM-HUST combined optimization",
        },
        "workload": {
            "model": "Qwen3.5-35B-A3B",
            "dtype": "BF16",
            "hardware": "Ascend 910B2",
            "TP": 2,
            "requests_per_arm": 200,
            "dataset": "MMLU-Pro",
            "actual_input_tokens_per_arm": 282265,
            "actual_output_tokens_per_arm": 51200,
            "output_budget": 256,
            "ignore_eos": False,
        },
        "measurement": {
            "same_run_F_T": True,
            "all_accounting_cost_retained": True,
            "strict_uninstrumented": False,
            "fresh_per_run_cache": True,
        },
        "scope": "Effective-compute gain, not independent reference attainment or a task-quality score. Actual implementation differences and necessary arithmetic differ.",
        "supersedes": {
            "B0_r1_F": 1829506984194363,
            "reason": "Native Mamba aligned-block publication dependencies corrected; original timing and raw requests unchanged.",
        },
        "evidence": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in FILES
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.source_root:
        for name in FILES:
            shutil.copyfile(args.source_root / name, DEST / name)
    (DEST / "summary.json").write_text(
        json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
