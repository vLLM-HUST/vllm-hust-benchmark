"""Verify the separate eight-case prospective diagnostic, never full200 credit."""

import hashlib
import json
from pathlib import Path


def verify():
    root = Path(__file__).parent / "mmlu8-pressure03-v1"
    raw = (root / "source-manifest.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "aaa8b284e6e924c8519832a722fd037d63f66d694db7eb5f05f72cc058b331bd"
    )
    members = json.loads(raw)["files"]
    assert len(members) == 25
    for name, checksum in members.items():
        stored = (
            "original-README.md.source"
            if name == "README.md"
            else name + ".source"
            if name.endswith(".py")
            else name
        )
        assert hashlib.sha256((root / stored).read_bytes()).hexdigest() == checksum
    plan = json.loads((root / "execution-plan.json").read_bytes())
    assert plan["study_scope"] == "supplemental_original_mmlu_first8_eos"
    assert plan["selection_indices"] == list(range(8))
    assert plan["planned_total"] == 32 and plan["planned_each"] == 8
    assert plan["output_budget_each"] == 8192 and plan["overall_seconds"] == 14400
    assert plan["reverse_BA_gate"]["throughput_gain_min_pct"] == 5
    cpu_raw = (root / "linux-cpu-admission.json").read_bytes()
    assert hashlib.sha256(cpu_raw).hexdigest() == (
        "882bf73adb90d1cc4557d7d8057b632d853976bfdf72208705fd6e010a9da8d4"
    )
    cpu = json.loads(cpu_raw)
    assert cpu["passed"] and cpu["inference_generations"] == 0
    assert cpu["selected_original_indices"] == list(range(8))
    assert cpu["original_output_budgets"] == [8192] * 8
    tests = json.loads((root / "linux-cpu-tests-owned-receipt.json").read_bytes())
    assert tests["actual_returncode"] == tests["inference_generations"] == 0
    assert tests["tests_passed"] == 39 and tests["passed"]
    return {"frozen_members": 25, "actual_linux_cpu_tests": 39, "formal200": False}


if __name__ == "__main__":
    print(json.dumps(verify()))
