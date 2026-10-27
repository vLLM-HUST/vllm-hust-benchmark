"""Generate a compact, source-linked receipt for the completed qualification pair."""

import argparse
import json
from pathlib import Path
from tp2_pilot import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--comparison", type=Path)
    a = p.parse_args()
    root = a.campaign.resolve()
    receipts = {}
    for arm, name in [("HOST", "pilot-host-r1"), ("AIV", "pilot-aiv")]:
        rows = [
            json.loads(line)
            for line in (root / name / "receipt.jsonl").read_text().splitlines()
        ]
        if rows[-1]["kind"] != "complete":
            raise RuntimeError("Incomplete capture")
        receipts[arm] = {r["kind"]: r for r in rows}
        if receipts[arm]["effective_configuration"]["graph_mode"] != "PIECEWISE":
            raise RuntimeError("Wrong effective graph mode")
        if not receipts[arm]["profile_outputs"]["warmup_parity"]:
            raise RuntimeError("Warmup mismatch")
    left, right = receipts["HOST"], receipts["AIV"]
    if left["profile_outputs"]["outputs"] != right["profile_outputs"]["outputs"]:
        raise RuntimeError("Backend token outputs differ")
    for key in (
        "script_sha256",
        "runtime_manifest_sha256",
        "model_config_sha256",
        "prompt_token_ids",
        "output_tokens",
    ):
        if left["configuration"][key] != right["configuration"][key]:
            raise RuntimeError(("Capture control differs", key))
    comparison_path = (
        a.comparison.resolve() if a.comparison else root / "comparison.json"
    )
    comparison = json.loads(comparison_path.read_text())
    if not all(all(r.values()) for r in comparison["answer_agreement"]):
        raise RuntimeError("Native and raw decomposition differ")
    workers = []
    for arm in ("host", "aiv"):
        materialization = json.loads(
            (root / f"analysis-{arm}/materialization.json").read_text()
        )
        for rank in range(2):
            path = root / f"raw-{arm}-worker{rank}.json"
            raw = json.loads(path.read_text())
            if (
                raw["unresolved"]
                or raw["missing_from_joint"]
                or raw["absent_from_raw"]
                or raw["coordinate_audit"]["differences"]
            ):
                raise RuntimeError("Incomplete raw correspondence")
            workers.append(
                dict(
                    arm=arm.upper(),
                    rank=rank,
                    native_launches=raw["native_launches"],
                    accepted_launches=raw["accepted"],
                    raw_members=raw["raw_members"],
                    joint_members=raw["joint_members"],
                    repeated_units=raw["repeated_units"],
                    model_sequence_period=raw["exact_model_sequence_period"],
                    matching_coordinates=raw["coordinate_audit"]["equal"],
                    raw_reconstruction_seconds=raw["raw_reconstruction_seconds"],
                    materialization_seconds=materialization[rank]["elapsed_seconds"],
                    receipt=str(path),
                    receipt_sha256=digest(path),
                    input_sha256=raw["input_sha256"],
                )
            )
    result = dict(
        status="QUALIFIED_PAIR_WITH_EQUAL_RAW_AND_NATIVE_ANSWERS",
        question="Cross-backend repeated-region membership and cost decomposition",
        campaign=str(root),
        model=left["configuration"]["model"],
        runtime_manifest_sha256=left["configuration"]["runtime_manifest_sha256"],
        independent_processes=2,
        paired_worker_profiles=4,
        profile_requests_per_process=2,
        output_tokens_equal=True,
        workers=workers,
        totals={
            k: sum(w[k] for w in workers)
            for k in (
                "native_launches",
                "raw_members",
                "joint_members",
                "repeated_units",
                "matching_coordinates",
            )
        },
        per_rank=[
            dict(
                members_per_unit=r["members_per_unit"],
                left_medians_ns=r["left_costs"]["medians_ns"],
                right_medians_ns=r["right_costs"]["medians_ns"],
            )
            for r in comparison["ranks"]
        ],
        answer_agreement=comparison["answer_agreement"],
        comparison_receipt=str(comparison_path),
        comparison_sha256=digest(comparison_path),
        qualification="Current HOST/AIV data is usable; the originally proposed FFTS+ setting is rejected by current CANN.",
        interpretation="Both paths recover the same complete lanes, coordinates and costs on this regular workload. "
        "This establishes correctness and feasibility, not structural superiority or a StriaTrace tool result.",
        next_comparison="Test irregular/multi-regime execution and measured construction/reuse against an equally capable raw workflow. "
        "Do not expand identical pilot repetitions merely to claim a capability gap.",
        formal_ABBA_executed=False,
        unprofiled_backend_speedup_measured=False,
        script_sha256=digest(__file__),
        code_sha256={
            p.name: digest(p)
            for p in Path(__file__).parent.glob("*tp2*")
            if p.is_file() and (p.suffix == ".py" or p.name.startswith("tp2_protocol"))
        },
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            dict(status=result["status"], totals=result["totals"], outputs_equal=True),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
