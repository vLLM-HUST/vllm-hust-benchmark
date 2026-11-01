"""Recompute the explicitly versioned B0-reference effective-compute comparison."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
PAIR = "reports/vllm-hust-midterm-20261019/m01-recovery/official018-paired-compute-20261024/summary.json"
PARTS = "reports/b0-reference-compute-20261101/accounting-parts.json"
REPORT = "reports/b0-reference-compute-20261101/summary.json"
PUBLIC = "website-publication/data/b0-reference-compute-20261101.json"
METHOD = "docs/B0_REFERENCE_COMPUTE.md"
QUALITY = "website-publication/data/midterm-v103.json"


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A numeric measurement is required")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Measurements must be finite and positive")
    return value


def calculate(f0, t0, k0, f1, t1, k1, threshold_percent):
    for value in (f0, t0, k0, f1, t1, k1, threshold_percent):
        positive(value)
    if any(not isinstance(v, int) for v in (f0, k0, f1, k1)):
        raise ValueError("Operation and participating-chip counts must be integers")
    c0, c1 = f0 / (k0 * t0 * 1e12), f1 / (k1 * t1 * 1e12)
    rows = []
    for role, f, seconds, chips, c in (
        ("B0", f0, t0, k0, c0),
        ("B1", f1, t1, k1, c1),
    ):
        ratio = (
            Fraction(f * k0, f0 * chips) * Fraction(str(t0)) / Fraction(str(seconds))
        )
        exact_eta = 100 * ratio
        eta = float(exact_eta)
        rows.append(
            {
                "role": role,
                "F": f,
                "T_seconds": seconds,
                "chips": chips,
                "C_TFLOPS_per_chip": c,
                "eta_compute_b0_percent": eta,
                "equivalent_reference_seconds": float(
                    Fraction(f * k0, f0 * chips) * Fraction(str(t0))
                ),
                "reference_target_met": exact_eta >= Fraction(str(threshold_percent)),
                "margin_percentage_points": eta - threshold_percent,
            }
        )
    return {
        "reference_C_TFLOPS_per_chip": c0,
        "target_percent": threshold_percent,
        "target_C_TFLOPS_per_chip": c0 * threshold_percent / 100,
        "C_gain_percent": 100 * (c1 / c0 - 1),
        "arms": rows,
    }


def recompute(root=ROOT):
    sources = {}
    loaded = {}
    for name in (PAIR, PARTS, QUALITY):
        raw = (root / name).read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        loaded[name] = json.loads(raw)
    pair, parts = loaded[PAIR], loaded[PARTS]
    if pair["schema"] != "m01-own-compute-pair/v1":
        raise ValueError("The audited paired source is required")
    if not pair["measurement"]["same_run_F_T"]:
        raise ValueError("Each numerator must belong to its timed run")
    for role in ("B0", "B1"):
        rows = parts[role]["parts"]
        if not rows or any(type(v) is not int or v < 0 for v in rows.values()):
            raise ValueError("Nonnegative integer component counts are required")
        f = sum(rows.values())
        if f != parts[role]["basic_F"] or f != pair["rows"][role]["F"]:
            raise ValueError("Component sum disagrees with the sealed necessary work")
        arm = pair["rows"][role]
        if arm["C_TFLOPS_per_chip"] != f / (arm["chips"] * arm["T_seconds"] * 1e12):
            raise ValueError("Source arithmetic mismatch")
    quality = loaded[QUALITY]["acceptance_completion"]["out256_quality"]
    a, b = pair["rows"]["B0"], pair["rows"]["B1"]
    metrics = calculate(
        a["F"], a["T_seconds"], a["chips"], b["F"], b["T_seconds"], b["chips"], 90
    )
    return {
        "schema": "project-b0-reference-compute/v1",
        "method": METHOD,
        "report_path": "reports/b0-reference-compute-20261101/README.md",
        "method_date": "2026-11-01",
        "analysis_kind": "retrospective_recomputation_of_audited_pair",
        "reference_selection": "User-designated official B0, common to both arms",
        "reference_kind": "measured_official_baseline_not_physical_ceiling",
        "target_scope": "project_B0_reference_compute_target",
        "independent_outline_eta_ref_pass": None,
        "original_independent_reference_reclassified": False,
        "baseline_percent_is_normalization_identity": True,
        "new_model_calls_for_this_calculation": 0,
        "independent_pair_count": 1,
        "workload": pair["workload"],
        "measurement": pair["measurement"],
        "identities": pair["identities"],
        "source_sha256": sources,
        "component_totals": {role: parts[role]["parts"] for role in ("B0", "B1")},
        "B0_family_totals": parts["B0"]["families"],
        "quality": quality,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = recompute()
    for name in (REPORT, PUBLIC):
        path = ROOT / name
        if args.check:
            if result != json.loads(path.read_text()):
                raise SystemExit(f"Stale derived result: {name}")
        elif args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["arms"], indent=2))


if __name__ == "__main__":
    main()
