"""Audit immutable campaign seals and export only scalar r8 paired observations."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def export(summary_path, output):
    summary = json.loads(summary_path.read_text())
    run = Path(summary["run"])
    receipt = summary["receipt"]
    assert receipt["status"] == "completed" and receipt["failure"] is None
    assert len(receipt["stages"]) == 15
    assert all(
        s["status"] in ["completed", "reused_verified"] for s in receipt["stages"]
    )
    audit = []
    roots = {}
    for stage in receipt["stages"]:
        name = stage["name"]
        source = stage.get("source")
        root = Path(source["source_root"]) / name if source else run / name
        children = (
            [("mmlu-B0", root / "B0"), ("mmlu-B1", root / "B1")]
            if name == "mmlu-pair"
            else [(name, root)]
        )
        for label, arm_root in children:
            manifest = arm_root / "checksums.json"
            files = json.loads(manifest.read_text())
            for relative, expected in files.items():
                path = arm_root / relative
                assert path.is_relative_to(arm_root), relative
                assert digest(path) == expected, str(path)
            roots[label] = arm_root
            audit.append(
                dict(
                    stage=label,
                    source_seal_sha256=digest(manifest),
                    files=len(files),
                    reused=bool(source),
                )
            )
            print("Verified", label, len(files), flush=True)
    seals = {row["stage"]: row["source_seal_sha256"] for row in audit}
    pair = summary["current_mmlu_pair"]
    measurements = []
    for profile, planned in [
        ("mmlu", 200),
        ("hle", 200),
        ("frontier", 160),
        ("swe", 64),
        ("terminal", 48),
    ]:
        arms = []
        for role in ["B0", "B1"]:
            if profile == "mmlu":
                report = pair["arms"][role]
                raw = report["groups"]["joint200"]
                stage = "mmlu-" + role
                metrics = dict(
                    output_tokens_per_second=raw["output_tokens_per_second"],
                    mean_ttft_ms=raw["mean_ttft_ms"],
                    mean_tpot_ms=raw["mean_tpot_ms"],
                    output_tokens=raw["successful_output_tokens"],
                    duration_seconds=raw["cohort_wall_seconds"],
                    requests=raw["recorded"],
                    failed=raw["failed"],
                )
                quality = report["quality"]
                extra = dict(
                    correct=quality["correct"],
                    truncated=quality["truncated_recorded"],
                    unparsed=quality["unparsed_recorded"],
                )
            else:
                stage = (
                    profile
                    + ("-trace-" if profile in ["swe", "terminal"] else "-solver-")
                    + role
                )
                report = (
                    summary["current_trace_performance"][profile]["arms"][role]
                    if profile in ["swe", "terminal"]
                    else summary["current_dataset_performance"][stage]
                )
                metrics = dict(
                    output_tokens_per_second=report["output_tokens_per_second"],
                    mean_ttft_ms=report["mean_ttft_ms"],
                    mean_tpot_ms=report["mean_tpot_ms"],
                    output_tokens=report["successful_output_tokens"],
                    duration_seconds=report["duration_seconds"],
                    requests=report["recorded"],
                    failed=report["failed"],
                )
                extra = dict(truncated=report["length_stops"])
                if profile in ["swe", "terminal"]:
                    extra.update(
                        {
                            k: report[k]
                            for k in [
                                "ttft_p95_ms",
                                "ttft_p99_ms",
                                "tpot_p95_ms",
                                "tpot_p99_ms",
                            ]
                        }
                    )
            assert report["source_seal_sha256"] == seals[stage]
            assert metrics["requests"] == planned and metrics["failed"] == 0
            assert (
                abs(
                    metrics["output_tokens"] / metrics["duration_seconds"]
                    - metrics["output_tokens_per_second"]
                )
                < 1e-8
            )
            arms.append(
                dict(
                    role=role if role == "B0" else "r8",
                    **metrics,
                    **extra,
                    source_seal_sha256=seals[stage],
                )
            )
        measurements.append(
            dict(
                id=profile,
                label={
                    "mmlu": "MMLU200",
                    "hle": "HLE200",
                    "frontier": "Frontier160",
                    "swe": "SWE trace64",
                    "terminal": "Terminal trace48",
                }[profile],
                control="B0",
                candidate="r8",
                baseline_id="original018-cann8.5.1",
                arms=arms,
            )
        )
    quality = {}
    for profile in ["hle", "frontier"]:
        quality[profile] = {}
        for role in ["B0", "B1"]:
            report = summary["current_dataset_quality"][profile + "-" + role]
            tracks = report["tracks"]
            quality[profile][role] = dict(
                planned=sum(t["total"] for t in tracks.values()),
                graded=sum(t["graded"] for t in tracks.values()),
                correct=sum(t["successes"] for t in tracks.values()),
                unresolved=sum(
                    t.get("unresolved", t.get("judge_unresolved", 0))
                    for t in tracks.values()
                ),
                truncated=sum(t["solver_truncated"] for t in tracks.values()),
                judge_invalid=sum(t["judge_invalid"] for t in tracks.values()),
                scoring_policy=report["scoring_policy"],
            )
    for row in measurements:
        for arm in row["arms"]:
            role = "B0" if arm["role"] == "B0" else "B1"
            if row["id"] in quality:
                arm.update(
                    {
                        k: quality[row["id"]][role][k]
                        for k in ["correct", "unresolved", "judge_invalid"]
                    }
                )
            elif row["id"] == "mmlu":
                arm["unresolved"] = arm["unparsed"]
    public = dict(
        schema="r8-fixed-five-paired-publication/v1",
        engine_release="full-answer-b1-20261117-r8",
        campaign_completed_utc="2026-11-18T03:16:29Z",
        source_summary_sha256=digest(summary_path),
        selected_execution_complete=True,
        all_profile_performance_pairs_complete=True,
        project_acceptance_complete=False,
        measurements=measurements,
        quality=quality,
        judge_model="Qwen3.5-35B-A3B",
        trace_tasks_per_profile=32,
        planned_inference_requests_per_arm=672,
        seal_audit=audit,
        scope="Frozen five-profile inference measurements; same-model grading is separate. SWE/Terminal are real public request snapshots, not agent task success. Retain truncated/unparsed cases in planned denominators. Reference accounting and context/tenant acceptance are separate evidence.",
        reference_scope="Existing out256/native denominator cannot be applied directly to this out8192/HTTP cohort; no historical efficiency percentage copied.",
    )
    dump(output, public)
    print("Exported", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.summary, args.output)
